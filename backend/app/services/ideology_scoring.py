# backend/app/services/ideology_scoring.py

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol, Union

from app.services.marpor_definitions import (
    hybrid_marpor_analyzer,
    LIB_FAMILY,
    AUTH_FAMILY,
    ECON_LEFT,
    ECON_RIGHT,
    CENTRIST_FAMILY,
)

logger = logging.getLogger(__name__)


class ConfidenceCalibrator(Protocol):
    def calibrate_overall(self, raw_confidence: float) -> float: ...
    def calibrate_axis(self, raw_confidence: float, axis: str) -> float: ...


_CALIBRATOR: Optional[ConfidenceCalibrator] = None


def configure_calibrator(calibrator: Optional[ConfidenceCalibrator]) -> None:
    global _CALIBRATOR
    _CALIBRATOR = calibrator


def _calibrate_overall(raw: float) -> float:
    c = _CALIBRATOR
    if c is None:
        return float(raw)
    try:
        return float(c.calibrate_overall(float(raw)))
    except Exception:
        logger.warning("Confidence calibration failed (overall). Using raw.", exc_info=True)
        return float(raw)


def _calibrate_axis(raw: float, axis: str) -> float:
    c = _CALIBRATOR
    if c is None:
        return float(raw)
    try:
        return float(c.calibrate_axis(float(raw), str(axis)))
    except Exception:
        logger.warning("Confidence calibration failed (axis=%s). Using raw.", axis, exc_info=True)
        return float(raw)


DEFAULT_CODE_THRESHOLD = 0.60
DEFAULT_USE_SEMANTIC = True

RESEARCH_GRADE_MIN_CONF = 0.65
RESEARCH_GRADE_MIN_EVIDENCE = 2
RESEARCH_GRADE_MIN_PATTERN_SOCIAL = 0.50

ATTRIBUTION_HINT_RATIO_MAX = 0.50
QUOTE_EVIDENCE_RATIO_MAX = 0.60

IDEOLOGY_FAMILIES = {LIB_FAMILY, AUTH_FAMILY, ECON_LEFT, ECON_RIGHT}


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp01(x: Any) -> float:
    v = _as_float(x, 0.0)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


_FAMILY_ALIASES: Dict[str, str] = {
    "libertarian": LIB_FAMILY,
    "lib": LIB_FAMILY,
    "social_lib": LIB_FAMILY,
    "social_libertarian": LIB_FAMILY,
    "authoritarian": AUTH_FAMILY,
    "auth": AUTH_FAMILY,
    "social_auth": AUTH_FAMILY,
    "social_authoritarian": AUTH_FAMILY,
    "economic_left": ECON_LEFT,
    "econ_left": ECON_LEFT,
    "left": ECON_LEFT,
    "economic_right": ECON_RIGHT,
    "econ_right": ECON_RIGHT,
    "right": ECON_RIGHT,
    "neutral": CENTRIST_FAMILY,
    "centrist": CENTRIST_FAMILY,
    "moderate": CENTRIST_FAMILY,
    "none": CENTRIST_FAMILY,
    "no_signal": CENTRIST_FAMILY,
}


def _norm_family_label(label: Any) -> str:
    raw = str(label or "").strip()
    if not raw:
        return CENTRIST_FAMILY
    key = re.sub(r"[\s\-]+", "_", raw.lower()).strip("_")
    mapped = _FAMILY_ALIASES.get(key, raw)

    if mapped in IDEOLOGY_FAMILIES or mapped == CENTRIST_FAMILY:
        return mapped
    return CENTRIST_FAMILY


def _empty_attribution_risk_summary() -> Dict[str, Any]:
    return {
        "is_risky": False,
        "quote_ratio": 0.0,
        "attribution_hint_ratio": 0.0,
        "evidence_count": 0,
    }


def _summarize_attribution_risk(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    ev = [e for e in (evidence or []) if isinstance(e, dict)]
    n = len(ev)
    if n == 0:
        return _empty_attribution_risk_summary()

    quote_n = 0
    attrib_n = 0
    for e in ev:
        if bool(e.get("inside_quotes") or False):
            quote_n += 1
        if bool(e.get("attribution_hint") or False):
            attrib_n += 1

    quote_ratio = float(quote_n) / float(n)
    attrib_ratio = float(attrib_n) / float(n)
    is_risky = (quote_ratio > QUOTE_EVIDENCE_RATIO_MAX) or (attrib_ratio > ATTRIBUTION_HINT_RATIO_MAX)

    return {
        "is_risky": bool(is_risky),
        "quote_ratio": round(quote_ratio, 3),
        "attribution_hint_ratio": round(attrib_ratio, 3),
        "evidence_count": n,
    }


def _axis_labels_block() -> Dict[str, Any]:
    return {
        "x_axis": {"name": "Economic", "negative": "Left", "positive": "Right"},
        "y_axis": {"name": "Social", "negative": "Authoritarian", "positive": "Libertarian"},
    }


def _axis_directions_from_strengths(axis_strengths: Dict[str, Any]) -> Dict[str, str]:
    axis = _as_dict(axis_strengths)
    soc = _as_dict(axis.get("social"))
    eco = _as_dict(axis.get("economic"))

    s_lib = _as_float(soc.get("libertarian"), 0.0)
    s_auth = _as_float(soc.get("authoritarian"), 0.0)
    s_total = _as_float(soc.get("total"), s_lib + s_auth)

    e_left = _as_float(eco.get("left"), 0.0)
    e_right = _as_float(eco.get("right"), 0.0)
    e_total = _as_float(eco.get("total"), e_left + e_right)

    social_dir = ""
    if s_total > 0.0:
        social_dir = "Libertarian" if s_lib >= s_auth else "Authoritarian"

    economic_dir = ""
    if e_total > 0.0:
        economic_dir = "Right" if e_right >= e_left else "Left"

    return {"social": social_dir, "economic": economic_dir}


def _empty_2d() -> Dict[str, Any]:
    return {
        "axis_labels": _axis_labels_block(),
        "axis_strengths": {
            "social": {"libertarian": 0.0, "authoritarian": 0.0, "total": 0.0},
            "economic": {"left": 0.0, "right": 0.0, "total": 0.0},
        },
        "coordinates": {"social": 0.0, "economic": 0.0},
        "coordinates_xy": {"x": 0.0, "y": 0.0},
        "confidence_2d": {"social": 0.0, "economic": 0.0, "overall": 0.0},
        "confidence": {"social": 0.0, "economic": 0.0, "overall": 0.0},
        "quadrant_2d": {"magnitude": 0.0, "axis_directions": {"social": "", "economic": ""}},
    }


def _coords_from_strengths(axis_strengths: Dict[str, Any]) -> Dict[str, float]:
    axis = _as_dict(axis_strengths)
    soc = _as_dict(axis.get("social"))
    eco = _as_dict(axis.get("economic"))

    lib = _as_float(soc.get("libertarian"), 0.0)
    auth = _as_float(soc.get("authoritarian"), 0.0)
    left = _as_float(eco.get("left"), 0.0)
    right = _as_float(eco.get("right"), 0.0)

    soc_total = lib + auth
    eco_total = left + right

    social = (lib - auth) / soc_total if soc_total > 0.0 else 0.0
    economic = (right - left) / eco_total if eco_total > 0.0 else 0.0

    return {"social": float(social), "economic": float(economic)}


def _normalize_2d_payload(block: Any) -> Dict[str, Any]:
    if not isinstance(block, dict) or not block:
        return _empty_2d()

    out = dict(block)
    out["axis_labels"] = _axis_labels_block()

    axis = _as_dict(out.get("axis_strengths"))
    soc = _as_dict(axis.get("social"))
    eco = _as_dict(axis.get("economic"))

    s_lib = _as_float(soc.get("libertarian"), 0.0)
    s_auth = _as_float(soc.get("authoritarian"), 0.0)
    e_left = _as_float(eco.get("left"), 0.0)
    e_right = _as_float(eco.get("right"), 0.0)

    axis["social"] = {
        "libertarian": round(float(s_lib), 6),
        "authoritarian": round(float(s_auth), 6),
        "total": round(float(s_lib + s_auth), 6),
    }
    axis["economic"] = {
        "left": round(float(e_left), 6),
        "right": round(float(e_right), 6),
        "total": round(float(e_left + e_right), 6),
    }
    out["axis_strengths"] = axis

    coords = _coords_from_strengths(axis)
    social = round(float(coords["social"]), 3)
    econ = round(float(coords["economic"]), 3)
    out["coordinates"] = {"social": social, "economic": econ}
    out["coordinates_xy"] = {"x": econ, "y": social}

    c2d = out.get("confidence_2d")
    if not isinstance(c2d, dict):
        c2d = out.get("confidence") if isinstance(out.get("confidence"), dict) else {}
    c2d = _as_dict(c2d)

    raw_social = _clamp01(c2d.get("social", 0.0))
    raw_econ = _clamp01(c2d.get("economic", 0.0))
    raw_overall = _clamp01(c2d.get("overall", 0.0))

    cal_social = _clamp01(_calibrate_axis(raw_social, axis="social"))
    cal_econ = _clamp01(_calibrate_axis(raw_econ, axis="economic"))
    cal_overall = _clamp01(_calibrate_overall(raw_overall))

    conf = {
        "social": round(float(cal_social), 3),
        "economic": round(float(cal_econ), 3),
        "overall": round(float(cal_overall), 3),
    }
    out["confidence_2d"] = conf
    out["confidence"] = dict(conf)

    out.pop("families_2d", None)

    mag = float(math.sqrt((social * social) + (econ * econ)))
    out["quadrant_2d"] = {
        "magnitude": round(float(mag), 3),
        "axis_directions": _axis_directions_from_strengths(axis),
    }
    return out


def calculate_marpor_breakdown(
    evidence: List[Dict[str, Any]],
    categories: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    code_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in (evidence or []):
        if not isinstance(ev, dict):
            continue
        code = str(ev.get("code", "") or "").strip()
        if not code:
            continue
        code_items[code].append(ev)

    total_strength = 0.0
    strength_by_code: Dict[str, float] = {}
    for code, items in code_items.items():
        if code not in categories:
            continue
        s = sum(_as_float(e.get("strength"), 0.0) for e in items)
        strength_by_code[code] = float(s)
        total_strength += float(s)

    out: Dict[str, Dict[str, Any]] = {}
    for code, items in code_items.items():
        cat = categories.get(code)
        if not cat:
            continue

        strength = float(strength_by_code.get(code, 0.0))
        pct = (strength / total_strength * 100.0) if total_strength > 0 else 0.0
        match_count = len(items)
        avg_strength = sum(_as_float(e.get("strength"), 0.0) for e in items) / max(1, match_count)

        ev_confs: List[float] = []
        for e in items:
            ec = e.get("evidence_confidence")
            if isinstance(ec, dict):
                ev_confs.append(_as_float(ec.get("evidence_confidence"), 0.0))
        avg_ev_conf = (sum(ev_confs) / max(1, len(ev_confs))) if ev_confs else 0.0

        polarity = {"support": 0, "oppose": 0, "neutral": 0}
        for e in items:
            pol = int(e.get("polarity", 0) or 0)
            if pol > 0:
                polarity["support"] += 1
            elif pol < 0:
                polarity["oppose"] += 1
            else:
                polarity["neutral"] += 1

        out[code] = {
            "code": code,
            "label": str(getattr(cat, "label", "") or ""),
            "description": str(getattr(cat, "description", "") or ""),
            "tendency": str(getattr(cat, "tendency", "") or ""),
            "weight": float(getattr(cat, "weight", 1.0) or 1.0),
            "percentage": round(float(pct), 2),
            "match_count": int(match_count),
            "avg_strength": round(float(avg_strength), 3),
            "avg_evidence_confidence": round(float(avg_ev_conf), 3),
            "polarity": polarity,
            "evidence_strength": round(float(strength), 6),
            "social_tendency": float(getattr(cat, "social_tendency", 0.0) or 0.0),
            "economic_tendency": float(getattr(cat, "economic_tendency", 0.0) or 0.0),
            "social_weight": float(getattr(cat, "social_weight", 0.0) or 0.0),
            "economic_weight": float(getattr(cat, "economic_weight", 0.0) or 0.0),
        }

    return dict(sorted(out.items(), key=lambda kv: kv[1]["percentage"], reverse=True))


def score_text(
    text: str,
    *,
    use_semantic: bool = DEFAULT_USE_SEMANTIC,
    code_threshold: float = DEFAULT_CODE_THRESHOLD,
) -> Dict[str, Any]:
    text = (text or "").strip()
    if len(text) < 10:
        return {
            "ideology_family": CENTRIST_FAMILY,
            "ideology_subtype": None,
            "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, ECON_LEFT: 0.0, ECON_RIGHT: 0.0},
            "evidence": [],
            "evidence_count": 0,
            "marpor_codes": [],
            "marpor_breakdown": {},
            "marpor_code_analysis": {},
            "raw_confidence_score": 0.0,
            "confidence_score": 0.0,
            "pattern_confidence": 0.0,
            "axis_dominance": 0.0,
            "total_strength": 0.0,
            "signal_strength": 0.0,
            "filtered_topic_count": 0,
            "is_ideology_evidence": False,
            "is_ideology_evidence_2d": False,
            "research_grade": False,
            "ideology_2d": _empty_2d(),
            "attribution_risk": False,
            "attribution_risk_summary": _empty_attribution_risk_summary(),
            "quote_ratio": 0.0,
            "attribution_hint_ratio": 0.0,
            "attribution_evidence_count": 0,
            "analysis_level": "segment",
            "analysis_mode": "too_short",
            "calibration_applied": bool(_CALIBRATOR is not None),
            "method": "marpor_evidence_v10_strict_2d_calibratable",
        }

    try:
        res = hybrid_marpor_analyzer.classify_text_detailed(
            text=text,
            use_semantic=bool(use_semantic),
            threshold=float(code_threshold),
        )
    except Exception as e:
        logger.error("score_text: analyzer failed: %s", e, exc_info=True)
        return {
            "ideology_family": CENTRIST_FAMILY,
            "ideology_subtype": None,
            "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, ECON_LEFT: 0.0, ECON_RIGHT: 0.0},
            "evidence": [],
            "evidence_count": 0,
            "marpor_codes": [],
            "marpor_breakdown": {},
            "marpor_code_analysis": {},
            "raw_confidence_score": 0.0,
            "confidence_score": 0.0,
            "pattern_confidence": 0.0,
            "axis_dominance": 0.0,
            "total_strength": 0.0,
            "signal_strength": 0.0,
            "filtered_topic_count": 0,
            "is_ideology_evidence": False,
            "is_ideology_evidence_2d": False,
            "research_grade": False,
            "ideology_2d": _empty_2d(),
            "attribution_risk": False,
            "attribution_risk_summary": _empty_attribution_risk_summary(),
            "quote_ratio": 0.0,
            "attribution_hint_ratio": 0.0,
            "attribution_evidence_count": 0,
            "analysis_level": "segment",
            "analysis_mode": "analyzer_error",
            "calibration_applied": bool(_CALIBRATOR is not None),
            "method": "marpor_evidence_v10_strict_2d_calibratable",
        }

    ideology_family = _norm_family_label(res.get("ideology_family", CENTRIST_FAMILY))
    ideology_subtype = res.get("ideology_subtype", None)
    if ideology_family not in (LIB_FAMILY, AUTH_FAMILY):
        ideology_subtype = None

    scores = _as_dict(res.get("scores"))
    scores_out = {
        LIB_FAMILY: round(_as_float(scores.get(LIB_FAMILY), 0.0), 2),
        AUTH_FAMILY: round(_as_float(scores.get(AUTH_FAMILY), 0.0), 2),
        ECON_LEFT: round(_as_float(scores.get(ECON_LEFT), 0.0), 2),
        ECON_RIGHT: round(_as_float(scores.get(ECON_RIGHT), 0.0), 2),
    }

    evidence = [e for e in _as_list(res.get("evidence", [])) if isinstance(e, dict)]
    evidence_count = int(res.get("evidence_count", 0) or 0)
    if evidence_count <= 0 and evidence:
        evidence_count = len(evidence)

    marpor_codes = [str(x).strip() for x in _as_list(res.get("marpor_codes", [])) if str(x).strip()]
    if not marpor_codes:
        seen = set()
        for ev in evidence:
            code = str(ev.get("code", "") or "").strip()
            if code and code not in seen:
                seen.add(code)
                marpor_codes.append(code)

    raw_confidence = _clamp01(res.get("confidence_score", 0.0))
    calibrated_confidence = _clamp01(_calibrate_overall(raw_confidence))

    signal_strength = float(_as_float(res.get("signal_strength", 0.0), 0.0))
    axis_dominance = float(_as_float(res.get("axis_dominance", 0.0), 0.0))
    filtered_topic_count = int(res.get("filtered_topic_count", 0) or 0)

    is_ideology_evidence = bool(res.get("is_ideology_evidence", False))
    is_ideology_evidence_2d = bool(res.get("is_ideology_evidence_2d", False))

    if ideology_family == CENTRIST_FAMILY:
        is_ideology_evidence = False
        is_ideology_evidence_2d = False

    total_strength = float(_as_float(res.get("total_evidence_strength", 0.0), 0.0))
    ideology_2d = _normalize_2d_payload(res.get("ideology_2d"))

    if ideology_family in IDEOLOGY_FAMILIES and not is_ideology_evidence_2d:
        axis = _as_dict(ideology_2d.get("axis_strengths"))
        soc_total = _as_float(_as_dict(axis.get("social")).get("total"), 0.0)
        eco_total = _as_float(_as_dict(axis.get("economic")).get("total"), 0.0)
        if soc_total > 0.0 or eco_total > 0.0:
            is_ideology_evidence_2d = True

    marpor_breakdown = res.get("marpor_breakdown")
    if not isinstance(marpor_breakdown, dict):
        marpor_breakdown = calculate_marpor_breakdown(evidence, hybrid_marpor_analyzer.categories)

    marpor_code_analysis = res.get("marpor_code_analysis")
    if not isinstance(marpor_code_analysis, dict):
        marpor_code_analysis = dict(marpor_breakdown)

    ev_conf_block = _as_dict(res.get("evidence_level_confidence"))
    pat = _as_dict(ev_conf_block.get("pattern_confidence"))
    pattern_confidence = float(_as_float(pat.get("pattern_confidence", 0.0), 0.0))

    attribution_risk_summary = _summarize_attribution_risk(evidence)
    attribution_risk = bool(attribution_risk_summary.get("is_risky", False))

    if ideology_family in (LIB_FAMILY, AUTH_FAMILY):
        research_grade = (
            is_ideology_evidence
            and calibrated_confidence >= RESEARCH_GRADE_MIN_CONF
            and evidence_count >= RESEARCH_GRADE_MIN_EVIDENCE
            and pattern_confidence >= RESEARCH_GRADE_MIN_PATTERN_SOCIAL
            and not attribution_risk
        )
    elif ideology_family in (ECON_LEFT, ECON_RIGHT):
        research_grade = (
            is_ideology_evidence
            and calibrated_confidence >= RESEARCH_GRADE_MIN_CONF
            and evidence_count >= RESEARCH_GRADE_MIN_EVIDENCE
            and not attribution_risk
        )
    else:
        research_grade = False

    analysis_level = str(res.get("analysis_level", "segment") or "segment")
    analysis_mode = str(res.get("analysis_mode", "classifier_default") or "classifier_default")

    quote_ratio = float(attribution_risk_summary.get("quote_ratio", 0.0) or 0.0)
    attribution_hint_ratio = float(attribution_risk_summary.get("attribution_hint_ratio", 0.0) or 0.0)
    attribution_evidence_count = int(attribution_risk_summary.get("evidence_count", 0) or 0)

    return {
        "ideology_family": ideology_family,
        "ideology_subtype": ideology_subtype,
        "scores": scores_out,
        "evidence": evidence,
        "evidence_count": int(evidence_count),
        "marpor_codes": marpor_codes,
        "marpor_breakdown": marpor_breakdown,
        "marpor_code_analysis": marpor_code_analysis,
        "raw_confidence_score": round(float(raw_confidence), 3),
        "confidence_score": round(float(calibrated_confidence), 3),
        "pattern_confidence": round(float(pattern_confidence), 4),
        "axis_dominance": round(float(axis_dominance), 6),
        "total_strength": round(float(total_strength), 6),
        "signal_strength": round(float(signal_strength), 2),
        "filtered_topic_count": int(filtered_topic_count),
        "is_ideology_evidence": bool(is_ideology_evidence),
        "is_ideology_evidence_2d": bool(is_ideology_evidence_2d),
        "research_grade": bool(research_grade),
        "ideology_2d": ideology_2d,
        "attribution_risk": attribution_risk,
        "attribution_risk_summary": attribution_risk_summary,
        "quote_ratio": round(float(quote_ratio), 3),
        "attribution_hint_ratio": round(float(attribution_hint_ratio), 3),
        "attribution_evidence_count": int(attribution_evidence_count),
        "analysis_level": analysis_level,
        "analysis_mode": analysis_mode,
        "calibration_applied": bool(_CALIBRATOR is not None),
        "method": "marpor_evidence_v10_strict_2d_calibratable",
    }


def score_segments(
    segments: List[Union[str, Dict[str, Any]]],
    *,
    use_semantic: bool = DEFAULT_USE_SEMANTIC,
    threshold: float = DEFAULT_CODE_THRESHOLD,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments or []):
        if isinstance(seg, str):
            seg_dict: Dict[str, Any] = {"section_index": idx, "text": seg}
        elif isinstance(seg, dict):
            seg_dict = dict(seg)
            seg_dict.setdefault("section_index", idx)
        else:
            continue

        seg_text = (seg_dict.get("text") or "").strip()
        scored = score_text(seg_text, use_semantic=use_semantic, code_threshold=threshold)
        seg_dict.update(scored)
        out.append(seg_dict)

    return out


def configure_embedder(embedder: Optional[Any]) -> None:
    if embedder is None:
        return
    hybrid_marpor_analyzer.set_embedder(embedder)


__all__ = [
    "configure_embedder",
    "configure_calibrator",
    "score_text",
    "score_segments",
    "calculate_marpor_breakdown",
    "LIB_FAMILY",
    "AUTH_FAMILY",
    "ECON_LEFT",
    "ECON_RIGHT",
    "CENTRIST_FAMILY",
]