# backend/app/services/evaluation/calibrate.py

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

EPS = 1e-9


def clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def safe_log(x: float) -> float:
    return math.log(max(EPS, float(x)))


def logit(p: float) -> float:
    p = min(1.0 - EPS, max(EPS, float(p)))
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    z = float(z)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def brier_score(probs: List[float], y: List[int]) -> float:
    if not probs or not y:
        return 0.0
    n = min(len(probs), len(y))
    if n <= 0:
        return 0.0
    return sum((clamp01(probs[i]) - int(y[i])) ** 2 for i in range(n)) / n


def log_loss(probs: List[float], y: List[int]) -> float:
    if not probs or not y:
        return 0.0
    n = min(len(probs), len(y))
    if n <= 0:
        return 0.0
    total = 0.0
    for i in range(n):
        p = clamp01(probs[i])
        yi = int(y[i])
        total += -(yi * safe_log(p) + (1 - yi) * safe_log(1.0 - p))
    return total / n


def expected_calibration_error(
    probs: List[float],
    y: List[int],
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    if not probs or not y:
        return 0.0, []

    n_bins = max(1, int(n_bins))
    n = min(len(probs), len(y))
    if n <= 0:
        return 0.0, []

    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for i in range(n):
        p = clamp01(probs[i])
        b = min(n_bins - 1, int(p * n_bins))  # p==1.0 -> last bin
        bins[b].append(i)

    ece = 0.0
    out_bins: List[Dict[str, Any]] = []
    for b, idxs in enumerate(bins):
        lower, upper = b / n_bins, (b + 1) / n_bins
        if not idxs:
            out_bins.append(
                {"bin": b, "lower": lower, "upper": upper, "count": 0, "avg_conf": 0.0, "avg_acc": 0.0}
            )
            continue

        avg_conf = sum(clamp01(probs[i]) for i in idxs) / len(idxs)
        avg_acc = sum(int(y[i]) for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(avg_acc - avg_conf)

        out_bins.append(
            {"bin": b, "lower": lower, "upper": upper, "count": len(idxs), "avg_conf": avg_conf, "avg_acc": avg_acc}
        )

    return ece, out_bins


def summarize_metrics(probs: List[float], y: List[int], n_bins: int) -> Dict[str, Any]:
    ece, bins = expected_calibration_error(probs, y, n_bins=n_bins)
    return {
        "n": min(len(probs), len(y)),
        "ece": float(ece),
        "brier": float(brier_score(probs, y)),
        "log_loss": float(log_loss(probs, y)),
        "bins": bins,
    }


@dataclass
class PredRow:
    id: str
    label_family: Optional[str]
    pred_family: Optional[str]
    confidence: float
    counted: bool = True

    @property
    def correct(self) -> Optional[int]:
        if self.label_family is None or self.pred_family is None:
            return None
        return 1 if str(self.label_family).strip() == str(self.pred_family).strip() else 0


def load_predictions_jsonl(path: Union[str, Path]) -> List[PredRow]:
    path = Path(path)
    rows: List[PredRow] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            conf = obj.get("confidence_score")
            if conf is None:
                conf = obj.get("confidence")
            if conf is None:
                conf = obj.get("prob", 0.0)

            rows.append(
                PredRow(
                    id=str(obj.get("id", f"row-{ln}")),
                    label_family=obj.get("label_family"),
                    pred_family=obj.get("pred_family"),
                    confidence=clamp01(float(conf or 0.0)),
                    counted=bool(obj.get("counted", True)),
                )
            )
    return rows


def build_calib_dataset(
    rows: List[PredRow],
    counted_only: bool = False,
) -> Tuple[List[float], List[int], List[str]]:
    confs: List[float] = []
    y: List[int] = []
    groups: List[str] = []

    for r in rows:
        if counted_only and not r.counted:
            continue
        c = r.correct
        if c is None:
            continue
        confs.append(clamp01(r.confidence))
        y.append(int(c))
        groups.append(str(r.pred_family or ""))

    return confs, y, groups


class Calibrator:
    def predict(self, p_raw: float, group: Optional[str] = None) -> float:
        raise NotImplementedError

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError


class PlattCalibrator(Calibrator):
    def __init__(self, a: float, b: float, feature: str = "logit"):
        self.a = float(a)
        self.b = float(b)
        self.feature = str(feature or "logit")

    def _feat(self, p: float) -> float:
        p = clamp01(p)
        return p if self.feature == "raw" else logit(p)

    def predict(self, p_raw: float, group: Optional[str] = None) -> float:
        return clamp01(sigmoid(self.a * self._feat(p_raw) + self.b))

    def to_json(self) -> Dict[str, Any]:
        return {"type": "platt", "a": self.a, "b": self.b, "feature": self.feature}


def fit_platt(
    confs: List[float],
    y: List[int],
    *,
    feature: str = "logit",
    l2: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> PlattCalibrator:
    n = min(len(confs), len(y))
    if n <= 0:
        return PlattCalibrator(1.0, 0.0, feature)

    xs = [clamp01(confs[i]) if feature == "raw" else logit(clamp01(confs[i])) for i in range(n)]
    ys = [int(y[i]) for i in range(n)]

    a, b = 1.0, 0.0
    l2 = float(l2)

    for _ in range(int(max_iter)):
        ps = [sigmoid(a * x + b) for x in xs]
        ws = [max(EPS, p * (1.0 - p)) for p in ps]

        ga = sum((ps[i] - ys[i]) * xs[i] for i in range(n)) + l2 * a
        gb = sum((ps[i] - ys[i]) for i in range(n)) + l2 * b

        haa = sum(ws[i] * xs[i] * xs[i] for i in range(n)) + l2
        hab = sum(ws[i] * xs[i] for i in range(n))
        hbb = sum(ws) + l2

        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break

        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det

        a_new, b_new = a - da, b - db
        if abs(a_new - a) + abs(b_new - b) < tol:
            a, b = a_new, b_new
            break

        a, b = a_new, b_new

    return PlattCalibrator(a, b, feature)


class IsotonicCalibrator(Calibrator):
    def __init__(self, breakpoints: List[float], values: List[float]):
        m = min(len(breakpoints or []), len(values or []))
        bps = breakpoints[:m]
        vals = values[:m]

        self.breakpoints = [clamp01(float(x)) for x in bps]
        self.values = [clamp01(float(v)) for v in vals]

        if self.breakpoints:
            self.breakpoints[-1] = 1.0  # ensure coverage to 1.0

    def predict(self, p_raw: float, group: Optional[str] = None) -> float:
        p_raw = clamp01(p_raw)
        if not self.breakpoints or not self.values:
            return p_raw
        for bp, v in zip(self.breakpoints, self.values):
            if p_raw <= bp:
                return clamp01(v)
        return clamp01(self.values[-1])

    def to_json(self) -> Dict[str, Any]:
        return {"type": "isotonic", "breakpoints": self.breakpoints, "values": self.values}


def fit_isotonic(confs: List[float], y: List[int]) -> IsotonicCalibrator:
    n = min(len(confs), len(y))
    if n <= 0:
        return IsotonicCalibrator([1.0], [0.5])

    pairs = sorted(((clamp01(confs[i]), int(y[i])) for i in range(n)), key=lambda t: t[0])
    xs = [p for p, _ in pairs]
    ys = [yi for _, yi in pairs]

    # PAV blocks
    sum_y: List[float] = []
    cnt: List[int] = []
    max_x: List[float] = []

    for x, yi in zip(xs, ys):
        sum_y.append(float(yi))
        cnt.append(1)
        max_x.append(float(x))

        while len(sum_y) >= 2:
            if (sum_y[-2] / cnt[-2]) <= (sum_y[-1] / cnt[-1]):
                break
            sum_y[-2] += sum_y[-1]
            cnt[-2] += cnt[-1]
            max_x[-2] = max_x[-1]
            sum_y.pop()
            cnt.pop()
            max_x.pop()

    breakpoints = max_x[:]
    values = [sy / c for sy, c in zip(sum_y, cnt)]
    if breakpoints:
        breakpoints[-1] = 1.0

    return IsotonicCalibrator(breakpoints, values)


class HistogramCalibrator(Calibrator):
    def __init__(self, edges: List[float], values: List[float]):
        m = min(len(edges or []), len(values or []))
        ed = edges[:m]
        vals = values[:m]

        self.edges = [clamp01(float(e)) for e in ed]
        self.values = [clamp01(float(v)) for v in vals]

        if self.edges:
            self.edges[-1] = 1.0  # ensure coverage to 1.0

    def predict(self, p_raw: float, group: Optional[str] = None) -> float:
        p_raw = clamp01(p_raw)
        if not self.edges or not self.values:
            return p_raw
        for e, v in zip(self.edges, self.values):
            if p_raw <= e:
                return clamp01(v)
        return clamp01(self.values[-1])

    def to_json(self) -> Dict[str, Any]:
        return {"type": "histogram", "edges": self.edges, "values": self.values}


def fit_histogram(confs: List[float], y: List[int], n_bins: int = 10) -> HistogramCalibrator:
    n = min(len(confs), len(y))
    if n <= 0:
        return HistogramCalibrator([1.0], [0.5])

    n_bins = max(1, int(n_bins))
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for i in range(n):
        p = clamp01(confs[i])
        b = min(n_bins - 1, int(p * n_bins))
        bins[b].append(i)

    edges: List[float] = []
    values: List[float] = []
    for b in range(n_bins):
        edges.append((b + 1) / n_bins)
        if bins[b]:
            values.append(sum(int(y[i]) for i in bins[b]) / len(bins[b]))
        else:
            values.append(0.5)

    edges[-1] = 1.0
    return HistogramCalibrator(edges, values)


class PerLabelCalibrator(Calibrator):
    def __init__(self, by_label: Dict[str, Calibrator], fallback: Calibrator):
        self.by_label = dict(by_label)
        self.fallback = fallback

    def predict(self, p_raw: float, group: Optional[str] = None) -> float:
        if group is not None:
            g = str(group)
            cal = self.by_label.get(g)
            if cal is not None:
                return cal.predict(p_raw)
        return self.fallback.predict(p_raw)

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "per_label",
            "fallback": self.fallback.to_json(),
            "by_label": {k: v.to_json() for k, v in self.by_label.items()},
        }


def apply_calibrator(confs: List[float], groups: List[str], cal: Calibrator) -> List[float]:
    n = min(len(confs), len(groups))
    return [cal.predict(confs[i], group=groups[i]) for i in range(n)]


def fit_calibrator(
    confs: List[float],
    y: List[int],
    groups: List[str],
    *,
    method: str = "platt",
    per_label: bool = False,
    n_bins: int = 10,
    platt_feature: str = "logit",
    min_group_n: int = 50,
) -> Calibrator:
    method = str(method or "platt").lower().strip()

    def _fit_one(c: List[float], yy: List[int]) -> Calibrator:
        if method == "platt":
            return fit_platt(c, yy, feature=platt_feature)
        if method == "isotonic":
            return fit_isotonic(c, yy)
        if method == "histogram":
            return fit_histogram(c, yy, n_bins=n_bins)
        raise ValueError(f"Unknown method: {method}")

    n = min(len(confs), len(y), len(groups))
    confs = confs[:n]
    y = y[:n]
    groups = groups[:n]

    if not per_label:
        return _fit_one(confs, y)

    by: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(str(g), []).append(i)

    fitted: Dict[str, Calibrator] = {}
    for g, idxs in by.items():
        if len(idxs) >= int(min_group_n):
            cg = [confs[i] for i in idxs]
            yg = [y[i] for i in idxs]
            fitted[g] = _fit_one(cg, yg)

    fallback = _fit_one(confs, y)
    return PerLabelCalibrator(fitted, fallback)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_bins_csv(path: Path, bins: List[Dict[str, Any]]) -> None:
    if not bins:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin", "lower", "upper", "count", "avg_conf", "avg_acc"])
        w.writeheader()
        for b in bins:
            w.writerow({k: b.get(k) for k in ["bin", "lower", "upper", "count", "avg_conf", "avg_acc"]})


def _calibrator_from_dict(obj: Dict[str, Any]) -> Calibrator:
    t = str(obj.get("type", "")).lower().strip()
    if t == "platt":
        return PlattCalibrator(float(obj["a"]), float(obj["b"]), str(obj.get("feature", "logit")))
    if t == "isotonic":
        return IsotonicCalibrator(list(obj.get("breakpoints", [])), list(obj.get("values", [])))
    if t == "histogram":
        return HistogramCalibrator(list(obj.get("edges", [])), list(obj.get("values", [])))
    if t == "per_label":
        by_label = {k: _calibrator_from_dict(v) for k, v in (obj.get("by_label", {}) or {}).items()}
        fallback = _calibrator_from_dict(obj["fallback"])
        return PerLabelCalibrator(by_label, fallback)
    raise ValueError(f"Unknown calibrator type: {t}")


class ConfidenceCalibratorAdapter:
    def __init__(self, base: Calibrator):
        self.base = base

    def calibrate(self, p_raw: float, group: Optional[str] = None) -> float:
        return float(self.base.predict(clamp01(p_raw), group=group))


class IdeologyScoringCalibrator:
    """
    Adapter to satisfy app.services.ideology_scoring.configure_calibrator()
    which expects:
      - calibrate_overall(p)
      - calibrate_axis(p, axis)
    """

    def __init__(self, adapter: ConfidenceCalibratorAdapter):
        self.adapter = adapter

    def calibrate_overall(self, raw_confidence: float) -> float:
        return float(self.adapter.calibrate(raw_confidence, group=None))

    def calibrate_axis(self, raw_confidence: float, axis: str) -> float:
        return float(self.adapter.calibrate(raw_confidence, group=f"axis:{axis}"))


_CONFIGURED: Optional[ConfidenceCalibratorAdapter] = None


def configure_calibrator(calib: Optional[ConfidenceCalibratorAdapter]) -> None:
    global _CONFIGURED
    _CONFIGURED = calib


def calibrate_confidence(p_raw: float, group: Optional[str] = None) -> float:
    if _CONFIGURED is None:
        return clamp01(p_raw)
    return _CONFIGURED.calibrate(p_raw, group=group)


def load_calibrator(path: Union[str, Path]) -> ConfidenceCalibratorAdapter:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        base = _calibrator_from_dict(json.load(f))
    return ConfidenceCalibratorAdapter(base)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train predictions.jsonl")
    p.add_argument("--val", default="", help="Optional validation predictions.jsonl")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--method", default="platt", help="platt | isotonic | histogram")
    p.add_argument("--per-label", action="store_true", help="Fit per pred_family group")
    p.add_argument("--counted-only", action="store_true", help="Use only counted==true")
    p.add_argument("--bins", type=int, default=10, help="Bins for ECE")
    p.add_argument("--platt-feature", default="logit", help="logit | raw")
    p.add_argument("--min-group-n", type=int, default=50, help="Minimum samples to fit a per-label calibrator")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train_path = Path(args.train)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    train_rows = load_predictions_jsonl(train_path)
    train_confs, train_y, train_groups = build_calib_dataset(train_rows, counted_only=args.counted_only)
    if not train_confs:
        raise SystemExit("No valid calibration rows in train set (need label_family + pred_family).")

    calibrator = fit_calibrator(
        train_confs,
        train_y,
        train_groups,
        method=args.method,
        per_label=args.per_label,
        n_bins=args.bins,
        platt_feature=args.platt_feature,
        min_group_n=args.min_group_n,
    )

    train_before = summarize_metrics(train_confs, train_y, args.bins)
    train_after = summarize_metrics(apply_calibrator(train_confs, train_groups, calibrator), train_y, args.bins)

    report: Dict[str, Any] = {
        "train": {
            "before": {k: v for k, v in train_before.items() if k != "bins"},
            "after": {k: v for k, v in train_after.items() if k != "bins"},
        },
        "settings": {
            "method": args.method,
            "per_label": args.per_label,
            "counted_only": args.counted_only,
            "bins": args.bins,
            "platt_feature": args.platt_feature,
            "min_group_n": args.min_group_n,
        },
        "calibrator": calibrator.to_json(),
    }

    save_bins_csv(out_dir / "train_bins_before.csv", train_before["bins"])
    save_bins_csv(out_dir / "train_bins_after.csv", train_after["bins"])

    if args.val:
        val_path = Path(args.val)
        if val_path.exists():
            val_rows = load_predictions_jsonl(val_path)
            val_confs, val_y, val_groups = build_calib_dataset(val_rows, counted_only=args.counted_only)
            if val_confs:
                val_before = summarize_metrics(val_confs, val_y, args.bins)
                val_after = summarize_metrics(apply_calibrator(val_confs, val_groups, calibrator), val_y, args.bins)
                report["val"] = {
                    "before": {k: v for k, v in val_before.items() if k != "bins"},
                    "after": {k: v for k, v in val_after.items() if k != "bins"},
                }
                save_bins_csv(out_dir / "val_bins_before.csv", val_before["bins"])
                save_bins_csv(out_dir / "val_bins_after.csv", val_after["bins"])

    save_json(out_dir / "calibrator.json", calibrator.to_json())
    save_json(out_dir / "report.json", report)

    print(f"Saved calibrator + report to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


__all__ = [
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "summarize_metrics",
    "PredRow",
    "load_predictions_jsonl",
    "build_calib_dataset",
    "Calibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "HistogramCalibrator",
    "PerLabelCalibrator",
    "fit_platt",
    "fit_isotonic",
    "fit_histogram",
    "fit_calibrator",
    "ConfidenceCalibratorAdapter",
    "IdeologyScoringCalibrator",
    "load_calibrator",
    "configure_calibrator",
    "calibrate_confidence",
]