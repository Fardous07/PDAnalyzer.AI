# backend/app/api/routes/speeches.py
from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import or_, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal, get_db
from app.database.models import Analysis, Question, Speech, User

# Auth
try:
    from app.services.auth_service import get_current_user

    _AUTH_AVAILABLE = True
except Exception:
    get_current_user = None
    _AUTH_AVAILABLE = False

from app.services.speech_ingestion import ingest_speech

_INGEST_AVAILABLE = True

# Transcription
try:
    from app.services.transcription import transcribe_media_file

    _TRANSCRIPTION_AVAILABLE = True
except Exception:
    transcribe_media_file = None
    _TRANSCRIPTION_AVAILABLE = False

# Semantic embedder
try:
    from sentence_transformers import SentenceTransformer

    _EMBEDDER_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _EMBEDDER_AVAILABLE = False

# Question generator
try:
    from app.services.question_generator import question_generator

    _QUESTION_GENERATOR_AVAILABLE = True
except Exception:
    question_generator = None
    _QUESTION_GENERATOR_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/speeches", tags=["speeches"])

# =============================================================================
# CONFIG
# =============================================================================
MIN_TEXT_CHARS = 50
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

MEDIA_ROOT = Path("media")
UPLOAD_DIR = MEDIA_ROOT / "uploads"
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEXT_FORMATS = {".txt", ".md"}
DOCUMENT_FORMATS = {".pdf", ".doc", ".docx"}
AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}

_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
_global_embedder: Optional[Any] = None

# ✅ IMPORTANT: keep background runs consistent with ingestion intent (recall-friendly)
DEFAULT_CODE_THRESHOLD = 0.35

# =============================================================================
# FAMILY CONSTANTS (strict)
# =============================================================================
LIB_FAMILY = "Libertarian"
AUTH_FAMILY = "Authoritarian"
ECON_LEFT = "Economic-Left"
ECON_RIGHT = "Economic-Right"
CENTRIST_FAMILY = "Centrist"
LEGACY_NEUTRAL = "Neutral"

_ALLOWED_FAMILIES = {LIB_FAMILY, AUTH_FAMILY, ECON_LEFT, ECON_RIGHT, CENTRIST_FAMILY}
_IDEOLOGICAL_FAMILIES = {LIB_FAMILY, AUTH_FAMILY, ECON_LEFT, ECON_RIGHT}


def _normalize_family(x: Any) -> str:
    s = str(x or "").strip()
    if s == LEGACY_NEUTRAL:
        return CENTRIST_FAMILY
    return s if s in _ALLOWED_FAMILIES else CENTRIST_FAMILY


def _normalize_subtype(family: str, subtype: Any) -> Optional[str]:
    fam = _normalize_family(family)
    if fam not in (LIB_FAMILY, AUTH_FAMILY):
        return None
    sub = str(subtype or "").strip()
    return sub or None


# =============================================================================
# HELPERS
# =============================================================================
def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _response(
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    message: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = {"success": success, "data": data, "error": error, "message": message, "timestamp": _now_iso()}
    if meta:
        out["metadata"] = meta
    return out


def _parse_iso_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        if len(v) == 10 and v.count("-") == 2:
            return datetime.strptime(v, "%Y-%m-%d")
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO (YYYY-MM-DD or full ISO datetime).")


def _word_count(text: str) -> int:
    return len((text or "").split())


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_filename(original: str, *, prefix: str) -> str:
    original = original or "upload"
    p = Path(original)
    ext = p.suffix.lower()
    base = p.stem
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    base = base.strip("._-") or "file"
    base = base[:120]
    return f"{prefix}_{base}{ext}"


def _safe_full_results(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        if s.startswith("{") and s.endswith("}"):
            try:
                obj2 = ast.literal_eval(s)
                return obj2 if isinstance(obj2, dict) else {}
            except Exception:
                return {}
        return {}
    try:
        return dict(x)
    except Exception:
        return {}


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x) if x is not None else default
    except Exception:
        return default


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =============================================================================
# 2D IDEOLOGY SANITIZATION
# =============================================================================
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

    social_dir = ("Libertarian" if s_lib >= s_auth else "Authoritarian") if s_total > 0 else ""
    economic_dir = ("Right" if e_right >= e_left else "Left") if e_total > 0 else ""
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


def _has_2d_mass(ide2d: Any) -> bool:
    if not isinstance(ide2d, dict):
        return False
    axis = _as_dict(ide2d.get("axis_strengths"))
    soc = _as_dict(axis.get("social"))
    eco = _as_dict(axis.get("economic"))
    soc_total = _as_float(soc.get("total"), 0.0)
    eco_total = _as_float(eco.get("total"), 0.0)
    return (soc_total > 0.0) or (eco_total > 0.0)


def _sanitize_ideology_2d(block: Any) -> Dict[str, Any]:
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

    axis_strengths = {
        "social": {"libertarian": s_lib, "authoritarian": s_auth, "total": s_lib + s_auth},
        "economic": {"left": e_left, "right": e_right, "total": e_left + e_right},
    }
    out["axis_strengths"] = axis_strengths

    coords = _as_dict(out.get("coordinates"))
    social = round(_as_float(coords.get("social"), 0.0), 3)
    econ = round(_as_float(coords.get("economic"), 0.0), 3)
    out["coordinates"] = {"social": social, "economic": econ}
    out["coordinates_xy"] = {"x": econ, "y": social}

    c2d = _as_dict(out.get("confidence_2d") or out.get("confidence") or {})
    conf = {
        "social": round(_as_float(c2d.get("social"), 0.0), 3),
        "economic": round(_as_float(c2d.get("economic"), 0.0), 3),
        "overall": round(_as_float(c2d.get("overall"), 0.0), 3),
    }
    out["confidence_2d"] = conf
    out["confidence"] = dict(conf)

    out.pop("families_2d", None)

    mag = (social * social + econ * econ) ** 0.5
    out["quadrant_2d"] = {"magnitude": round(mag, 3), "axis_directions": _axis_directions_from_strengths(axis_strengths)}
    return out


def _extract_ideology_2d_anywhere(full_results_any: Any) -> Dict[str, Any]:
    fr = _safe_full_results(full_results_any)
    candidates = [
        fr.get("ideology_2d"),
        _as_dict(fr.get("speech_level")).get("ideology_2d"),
        _as_dict(fr.get("metadata")).get("ideology_2d"),
    ]
    for c in candidates:
        if isinstance(c, dict) and isinstance(c.get("axis_strengths"), dict):
            return _sanitize_ideology_2d(c)
    return _empty_2d()


def _dominant_family_from_2d(ideology_2d: Dict[str, Any]) -> str:
    axis = _as_dict(ideology_2d.get("axis_strengths"))
    soc = _as_dict(axis.get("social"))
    eco = _as_dict(axis.get("economic"))
    s_total = _as_float(soc.get("total"), 0.0)
    e_total = _as_float(eco.get("total"), 0.0)

    if s_total > 0:
        return LIB_FAMILY if _as_float(soc.get("libertarian"), 0.0) >= _as_float(soc.get("authoritarian"), 0.0) else AUTH_FAMILY
    if e_total > 0:
        return ECON_RIGHT if _as_float(eco.get("right"), 0.0) >= _as_float(eco.get("left"), 0.0) else ECON_LEFT
    return CENTRIST_FAMILY


def _sum_axis_strengths_from_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fallback: aggregate axis_strengths from per-item ideology_2d blocks.
    Only counts ideologically-labeled items (not Centrist).
    """
    s_lib = s_auth = e_left = e_right = 0.0

    for it in items or []:
        if not isinstance(it, dict):
            continue
        fam = _normalize_family(it.get("ideology_family"))
        if fam not in _IDEOLOGICAL_FAMILIES:
            continue
        b = it.get("ideology_2d")
        if not isinstance(b, dict):
            continue
        axis = _as_dict(b.get("axis_strengths"))
        soc = _as_dict(axis.get("social"))
        eco = _as_dict(axis.get("economic"))
        s_lib += _as_float(soc.get("libertarian"), 0.0)
        s_auth += _as_float(soc.get("authoritarian"), 0.0)
        e_left += _as_float(eco.get("left"), 0.0)
        e_right += _as_float(eco.get("right"), 0.0)

    social_total = s_lib + s_auth
    econ_total = e_left + e_right

    social_coord = (s_lib - s_auth) / social_total if social_total > 0 else 0.0
    econ_coord = (e_right - e_left) / econ_total if econ_total > 0 else 0.0

    out = _empty_2d()
    out["axis_strengths"] = {
        "social": {"libertarian": round(s_lib, 6), "authoritarian": round(s_auth, 6), "total": round(social_total, 6)},
        "economic": {"left": round(e_left, 6), "right": round(e_right, 6), "total": round(econ_total, 6)},
    }
    out["coordinates"] = {"social": round(float(social_coord), 3), "economic": round(float(econ_coord), 3)}
    out["coordinates_xy"] = {"x": out["coordinates"]["economic"], "y": out["coordinates"]["social"]}
    out["quadrant_2d"] = {
        "magnitude": round(float((social_coord * social_coord + econ_coord * econ_coord) ** 0.5), 3),
        "axis_directions": _axis_directions_from_strengths(out["axis_strengths"]),
    }
    return _sanitize_ideology_2d(out)


# =============================================================================
# EVIDENCE NORMALIZATION
# =============================================================================
def _coerce_evidence_item(it: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(it, dict):
        return {}
    fam = _normalize_family(it.get("ideology_family") or it.get("family") or it.get("dominant_family") or it.get("label_family"))
    sub = _normalize_subtype(fam, it.get("ideology_subtype") or it.get("subtype") or it.get("dominant_subtype") or it.get("label_subtype"))
    txt = str(it.get("text") or it.get("full_text") or it.get("content") or "")
    out = dict(it)
    out["ideology_family"] = fam
    out["ideology_subtype"] = sub
    out["text"] = txt
    out.setdefault("full_text", txt)

    if "start_char" in out:
        out["start_char"] = _as_int(out.get("start_char"), 0)
    if "end_char" in out:
        out["end_char"] = _as_int(out.get("end_char"), out.get("start_char", 0))

    if out.get("confidence_score") is None and out.get("confidence") is not None:
        out["confidence_score"] = out.get("confidence")
    out["confidence_score"] = _as_float(out.get("confidence_score"), 0.0)

    out["evidence_count"] = _as_int(out.get("evidence_count"), 0)
    out["signal_strength"] = _as_float(out.get("signal_strength"), 0.0)

    mc = out.get("marpor_codes") or []
    out["marpor_codes"] = [str(x).strip() for x in (mc if isinstance(mc, list) else []) if str(x).strip()]

    if "ideology_2d" in out:
        out["ideology_2d"] = _sanitize_ideology_2d(out.get("ideology_2d"))

    return out


def _coerce_argument_span(sp: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sp, dict):
        return {}
    out = dict(sp)
    out["role"] = str(out.get("role") or "transition").strip() or "transition"
    sr = out.get("sentence_range")
    out["sentence_range"] = [_as_int(sr[0], 0), _as_int(sr[1], 0)] if isinstance(sr, (list, tuple)) and len(sr) == 2 else [0, 0]
    out["start_char"] = _as_int(out.get("start_char"), 0)
    out["end_char"] = _as_int(out.get("end_char"), out["start_char"])
    out["text"] = str(out.get("text") or "")
    out["segment_count"] = _as_int(out.get("segment_count"), 0)
    out["evidence_sentence_count_2d"] = _as_int(out.get("evidence_sentence_count_2d"), 0)
    out["evidence_sentence_count"] = _as_int(out.get("evidence_sentence_count"), out["evidence_sentence_count_2d"])
    out["ideology_2d"] = _sanitize_ideology_2d(out.get("ideology_2d"))
    out["labels"] = _as_dict(out.get("labels"))
    return out


def _coerce_argument_unit(u: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(u, dict):
        return {}
    out = dict(u)
    out["argument_unit_index"] = _as_int(out.get("argument_unit_index"), 0)
    sr = out.get("sentence_range")
    out["sentence_range"] = [_as_int(sr[0], 0), _as_int(sr[1], 0)] if isinstance(sr, (list, tuple)) and len(sr) == 2 else [0, 0]
    out["start_char"] = _as_int(out.get("start_char"), 0)
    out["end_char"] = _as_int(out.get("end_char"), out["start_char"])
    out["text"] = str(out.get("text") or "")
    out["ideology_2d"] = _sanitize_ideology_2d(out.get("ideology_2d"))

    spans = out.get("spans")
    out["spans"] = [_coerce_argument_span(x) for x in (spans if isinstance(spans, list) else []) if isinstance(x, dict)]

    out["pivot_detected"] = bool(out.get("pivot_detected", False))
    pa = out.get("pivot_axes")
    out["pivot_axes"] = [str(x).strip() for x in (pa if isinstance(pa, list) else []) if str(x).strip()]
    return out


# =============================================================================
# ANALYSIS SUMMARY BUILDER (UNIT COUNTS)
# =============================================================================
def _ensure_summary_aliases(summary: Any) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {"evidence_counts": {}, "total_evidence": 0}
    out = dict(summary)

    if "evidence_counts" not in out:
        counts = out.get("statement_counts_by_family_ideology")
        out["evidence_counts"] = counts if isinstance(counts, dict) else {}

    if "total_evidence" not in out:
        if "ideological_statements" in out:
            out["total_evidence"] = int(_as_int(out.get("ideological_statements"), 0))
        else:
            ec = out.get("evidence_counts") if isinstance(out.get("evidence_counts"), dict) else {}
            out["total_evidence"] = int(sum(int(_as_int(v, 0)) for v in ec.values()))

    return out


def _build_analysis_summary_from_full_results(full_results_any: Any) -> Dict[str, Any]:
    fr = _safe_full_results(full_results_any)

    statements = fr.get("statements") if isinstance(fr.get("statements"), list) else None
    if statements is None:
        base = fr.get("sections") or fr.get("segments") or fr.get("statement_list") or []
        statements = base if isinstance(base, list) else []

    key_statements = fr.get("key_statements") if isinstance(fr.get("key_statements"), list) else []

    ideology_counts = {LIB_FAMILY: 0, AUTH_FAMILY: 0, ECON_LEFT: 0, ECON_RIGHT: 0}
    subtype_counts_by_family: Dict[str, Dict[str, int]] = {LIB_FAMILY: {}, AUTH_FAMILY: {}}

    total_statements = 0
    ideological_statements = 0
    non_ideology_statement_count = 0

    total_evidence_count_weighted = 0
    w_conf_num = w_sig_num = w_den = 0.0
    marpor_union = set()

    for raw in statements:
        if not isinstance(raw, dict):
            continue
        st = _coerce_evidence_item(raw)
        fam = _normalize_family(st.get("ideology_family"))
        sub = st.get("ideology_subtype")

        total_statements += 1
        if fam in _IDEOLOGICAL_FAMILIES:
            ideological_statements += 1
            ideology_counts[fam] = ideology_counts.get(fam, 0) + 1
            if fam in (LIB_FAMILY, AUTH_FAMILY) and sub:
                m = subtype_counts_by_family.setdefault(fam, {})
                m[sub] = int(m.get(sub, 0)) + 1
        else:
            non_ideology_statement_count += 1

        w = max(1.0, _as_float(st.get("sentence_count") or st.get("segment_count") or 1.0, 1.0))
        w_conf_num += _as_float(st.get("confidence_score"), 0.0) * w
        w_sig_num += _as_float(st.get("signal_strength"), 0.0) * w
        w_den += w

        total_evidence_count_weighted += _as_int(st.get("evidence_count"), 0)

        for c in (st.get("marpor_codes") or []):
            if (cs := str(c).strip()):
                marpor_union.add(cs)

    for raw in key_statements:
        if isinstance(raw, dict):
            for c in (_coerce_evidence_item(raw).get("marpor_codes") or []):
                if (cs := str(c).strip()):
                    marpor_union.add(cs)

    avg_conf = (w_conf_num / w_den) if w_den > 0 else 0.0
    avg_sig = (w_sig_num / w_den) if w_den > 0 else 0.0

    summary = {
        "statement_counts_by_family_ideology": ideology_counts,
        "evidence_counts": ideology_counts,
        "non_ideological_statement_count": int(non_ideology_statement_count),
        "subtype_counts_by_family": subtype_counts_by_family,
        "total_statements": int(total_statements),
        "ideological_statements": int(ideological_statements),
        "total_evidence_count": int(total_evidence_count_weighted),  # diagnostic
        "total_evidence": int(ideological_statements),  # UI units
        "avg_confidence_score": round(float(avg_conf), 4),
        "avg_signal_strength": round(float(avg_sig), 2),
        "key_statement_count": int(len([x for x in key_statements if isinstance(x, dict)])),
        "marpor_codes": sorted(marpor_union),
        "notes": ["Metadata-only summary derived from evidence-labeled statements."],
    }
    return _ensure_summary_aliases(summary)


# =============================================================================
# MEDIA TIME MAPPING
# =============================================================================
def _estimate_time_from_char(start_char: Optional[int], total_chars: int, duration: float) -> Optional[float]:
    if start_char is None or total_chars <= 0 or duration <= 0:
        return None
    return round(_clamp(start_char / total_chars, 0.0, 1.0) * duration, 3)


def _ensure_time_fields_for_items(
    items: List[Dict[str, Any]],
    transcript_len: int,
    duration: float,
    start_key: str = "start_char",
    end_key: str = "end_char",
    out_begin_key: str = "time_begin",
    out_end_key: str = "time_end",
) -> None:
    if not items or transcript_len <= 0 or duration <= 0:
        return
    n = len(items)
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue

        tb = it.get(out_begin_key)
        if not isinstance(tb, (int, float)) or _as_float(tb, -1.0) < 0:
            sc = it.get(start_key)
            try:
                sc_int = int(sc) if sc is not None else None
            except Exception:
                sc_int = None
            t = _estimate_time_from_char(sc_int, transcript_len, duration)
            it[out_begin_key] = t if t is not None else round(((i + 1) / (n + 1)) * duration, 3)

        te = it.get(out_end_key)
        if not isinstance(te, (int, float)) or _as_float(te, -1.0) < 0:
            ec = it.get(end_key)
            try:
                ec_int = int(ec) if ec is not None else None
            except Exception:
                ec_int = None
            t_end = _estimate_time_from_char(ec_int, transcript_len, duration)
            if t_end is None:
                base = _as_float(it.get(out_begin_key), 0.0)
                t_end = round(_clamp(base + min(1.5, max(0.25, duration * 0.01)), 0.0, duration), 3)
            it[out_end_key] = t_end


def _apply_media_jump_time_support(payload: Dict[str, Any], duration: Optional[float]) -> Dict[str, Any]:
    payload = payload or {}
    dur = _as_float(duration, 0.0)
    if dur <= 0:
        return payload

    transcript = payload.get("text") or payload.get("transcript_text") or ""
    transcript_len = len(transcript) if isinstance(transcript, str) else 0
    if transcript_len <= 0:
        return payload

    for key in ("key_statements", "statements", "sections", "segments", "statement_list", "argument_units"):
        arr = payload.get(key)
        if isinstance(arr, list):
            _ensure_time_fields_for_items([x for x in arr if isinstance(x, dict)], transcript_len, dur)

    au = payload.get("argument_units")
    if isinstance(au, list):
        for u in au:
            if isinstance(u, dict) and isinstance(u.get("spans"), list):
                _ensure_time_fields_for_items([x for x in u["spans"] if isinstance(x, dict)], transcript_len, dur)

    return payload


# =============================================================================
# AUTH / PERMISSIONS
# =============================================================================
def optional_current_user_dep() -> Any:
    if _AUTH_AVAILABLE and get_current_user is not None:
        return Depends(get_current_user)

    async def _none() -> None:
        return None

    return Depends(_none)


def _require_auth_if_enabled(user: Optional[User]) -> None:
    if _AUTH_AVAILABLE and user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


def _is_admin(user: Optional[User]) -> bool:
    if not user:
        return False
    return bool(getattr(user, "is_admin", False)) or bool(getattr(user, "role", None) == "admin")


def _can_read(speech: Speech, user: Optional[User]) -> bool:
    if getattr(speech, "is_public", False):
        return True
    if user is None:
        return False
    return getattr(speech, "user_id", None) == user.id or _is_admin(user)


def _require_write_access(speech: Speech, user: Optional[User]) -> None:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if getattr(speech, "user_id", None) == user.id or _is_admin(user):
        return
    raise HTTPException(status_code=403, detail="Not permitted")


# =============================================================================
# EMBEDDER WRAPPER
# =============================================================================
class _STEmbedder:
    def __init__(self, model_name: str):
        if not _EMBEDDER_AVAILABLE or SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available")
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        kwargs.pop("show_progress_bar", None)
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


def _get_embedder(use_semantic: bool = True) -> Optional[Any]:
    global _global_embedder
    if not use_semantic or not _EMBEDDER_AVAILABLE or SentenceTransformer is None:
        return None
    if _global_embedder is None:
        try:
            _global_embedder = _STEmbedder(_EMBEDDER_MODEL)
            logger.info("Loaded embedder model: %s", _EMBEDDER_MODEL)
        except Exception as e:
            logger.warning("Failed to load embedder: %s", e, exc_info=True)
            _global_embedder = None
    return _global_embedder


@contextmanager
def _fresh_db_session() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# FILE EXTRACTION
# =============================================================================
def _extract_text_from_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in TEXT_FORMATS:
        return _clean_text(file_path.read_text(encoding="utf-8", errors="replace"))

    if ext == ".pdf":
        try:
            import PyPDF2
        except Exception:
            raise HTTPException(status_code=500, detail="PDF processing requires PyPDF2")
        try:
            out = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    out += (page.extract_text() or "") + "\n"
            return _clean_text(out)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {e}")

    if ext in {".doc", ".docx"}:
        try:
            import docx
        except Exception:
            raise HTTPException(status_code=500, detail="Word processing requires python-docx")
        try:
            doc = docx.Document(str(file_path))
            out = "\n".join([p.text for p in doc.paragraphs])
            return _clean_text(out)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract Word text: {e}")

    raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")


# =============================================================================
# REQUEST MODELS
# =============================================================================
class SpeechCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=1, max_length=500)
    speaker: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=MIN_TEXT_CHARS)
    date: Optional[str] = Field(None)
    location: Optional[str] = Field(None, max_length=200)
    event: Optional[str] = Field(None, max_length=200)
    source_url: Optional[str] = Field(None, max_length=500)
    source_type: Optional[str] = Field(None, max_length=50)
    language: str = Field("en", max_length=10)
    is_public: bool = Field(False)
    llm_provider: Optional[str] = Field("openai")
    llm_model: Optional[str] = Field("gpt-4o-mini")
    use_semantic_segmentation: bool = Field(True)
    use_semantic_scoring: bool = Field(True)
    analyze_immediately: bool = Field(True)

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v or len(v) < MIN_TEXT_CHARS:
            raise ValueError(f"Text too short (minimum {MIN_TEXT_CHARS} characters)")
        return v


class SpeechUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    speaker: Optional[str] = Field(None, min_length=1, max_length=200)
    text: Optional[str] = Field(None, min_length=MIN_TEXT_CHARS)
    date: Optional[str] = Field(None)
    location: Optional[str] = Field(None, max_length=200)
    event: Optional[str] = Field(None, max_length=200)
    source_url: Optional[str] = Field(None, max_length=500)
    source_type: Optional[str] = Field(None, max_length=50)
    language: Optional[str] = Field(None, max_length=10)
    is_public: Optional[bool] = None


class GenerateQuestionsBody(BaseModel):
    """
    Legacy compatibility endpoint body:
    POST /speeches/{speech_id}/questions/generate

    Frontend sometimes sends:
      { speech_id, question_type, max_questions, llm_provider, llm_model }
    """

    speech_id: Optional[int] = None
    question_type: str = Field(default="journalistic")
    max_questions: int = Field(default=5, ge=1, le=8)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


# =============================================================================
# CANONICALIZE FULL RESULTS (IMPORTANT)
# =============================================================================
def _canonicalize_full_results(*, speech: Speech, fr_in: Dict[str, Any]) -> Dict[str, Any]:
    fr = dict(fr_in or {})

    fr.setdefault("text", speech.text)
    fr.setdefault("transcript_text", speech.text)
    fr.setdefault("title", speech.title)
    fr.setdefault("speaker", speech.speaker)

    # canonical statements
    statements: List[Dict[str, Any]] = []
    if isinstance(fr.get("statements"), list):
        statements = [_coerce_evidence_item(x) for x in fr["statements"] if isinstance(x, dict)]
    else:
        base = None
        for k in ("sections", "segments", "statement_list"):
            if isinstance(fr.get(k), list) and fr.get(k):
                base = fr.get(k)
                break
        if isinstance(base, list):
            statements = [_coerce_evidence_item(x) for x in base if isinstance(x, dict)]

    fr["statements"] = statements
    fr["sections"] = statements
    fr["segments"] = statements
    fr["statement_list"] = statements

    if isinstance(fr.get("key_statements"), list):
        fr["key_statements"] = [_coerce_evidence_item(x) for x in fr["key_statements"] if isinstance(x, dict)]
    else:
        fr["key_statements"] = []

    if isinstance(fr.get("argument_units"), list):
        fr["argument_units"] = [_coerce_argument_unit(x) for x in fr["argument_units"] if isinstance(x, dict)]
    else:
        fr["argument_units"] = []

    # ideology_2d: prefer provided, else aggregate from items
    ide2d = _extract_ideology_2d_anywhere(fr)
    if not _has_2d_mass(ide2d) and statements:
        ide2d = _sum_axis_strengths_from_items(statements)

    fr["ideology_2d"] = _sanitize_ideology_2d(ide2d)

    if isinstance(fr.get("speech_level"), dict):
        fr["speech_level"]["ideology_2d"] = _sanitize_ideology_2d(fr["ideology_2d"])
    else:
        fr["speech_level"] = {"ideology_2d": _sanitize_ideology_2d(fr["ideology_2d"])}

    # summary
    if not isinstance(fr.get("analysis_summary"), dict):
        fr["analysis_summary"] = _build_analysis_summary_from_full_results(fr)
    fr["analysis_summary"] = _ensure_summary_aliases(fr["analysis_summary"])

    return fr


# =============================================================================
# SPEECH -> DICT  (FIXED: canonicalize analysis.full_results)
# =============================================================================
def _speech_to_dict(
    s: Speech,
    include_text: bool = False,
    include_analysis_summary: bool = True,
    analysis: Optional[Analysis] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "id": s.id,
        "user_id": getattr(s, "user_id", None),
        "title": s.title,
        "speaker": s.speaker,
        "date": s.date.isoformat() if s.date else None,
        "location": s.location,
        "event": s.event,
        "language": s.language,
        "word_count": s.word_count,
        "source_url": s.source_url,
        "source_type": s.source_type,
        "media_url": getattr(s, "media_url", None),
        "status": s.status,
        "is_public": s.is_public,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        "analyzed_at": s.analyzed_at.isoformat() if s.analyzed_at else None,
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "use_semantic_segmentation": bool(getattr(s, "use_semantic_segmentation", True)),
        "use_semantic_scoring": bool(getattr(s, "use_semantic_scoring", True)),
        "has_analysis": bool(analysis is not None),
    }
    if include_text:
        d["text"] = s.text

    if include_analysis_summary and analysis is not None:
        fr = _safe_full_results(getattr(analysis, "full_results", None))
        fr = _canonicalize_full_results(speech=s, fr_in=fr)
        d["analysis_summary"] = fr.get("analysis_summary") or _ensure_summary_aliases({})
        d["ideology_2d"] = fr.get("ideology_2d") or _empty_2d()

    return d


# =============================================================================
# ANALYSIS UPSERT
# =============================================================================
def _upsert_analysis_row(
    db: Session,
    *,
    speech: Speech,
    full_results: Dict[str, Any],
    processing_time_seconds: Optional[float],
) -> Analysis:
    fr = _canonicalize_full_results(speech=speech, fr_in=full_results)

    summary = fr.get("analysis_summary") if isinstance(fr.get("analysis_summary"), dict) else {}
    conf = _as_float(summary.get("avg_confidence_score"), 0.0)
    marpor_codes = [str(x).strip() for x in (summary.get("marpor_codes") or []) if str(x).strip()]

    dom_family = _normalize_family(_dominant_family_from_2d(fr["ideology_2d"]))
    dom_subtype = None

    axis = _as_dict(_as_dict(fr["ideology_2d"]).get("axis_strengths"))
    soc = _as_dict(axis.get("social"))
    s_lib = _as_float(soc.get("libertarian"), 0.0)
    s_auth = _as_float(soc.get("authoritarian"), 0.0)
    s_tot = max(0.0, s_lib + s_auth)

    lib_score = round((s_lib / s_tot) * 100.0, 2) if s_tot > 0 else 0.0
    auth_score = round((s_auth / s_tot) * 100.0, 2) if s_tot > 0 else 0.0

    sl = _as_dict(fr.get("speech_level"))
    total_sent = _as_float(sl.get("total_sentences"), 0.0)
    cent_sent = _as_float(sl.get("centrist_sentences"), 0.0)
    centrist_score = round((_clamp(cent_sent / total_sent, 0.0, 1.0) * 100.0), 2) if total_sent > 0 else 0.0

    existing = db.query(Analysis).filter(Analysis.speech_id == speech.id).first()
    if existing:
        existing.ideology_family = dom_family
        existing.ideology_subtype = dom_subtype
        existing.libertarian_score = float(lib_score)
        existing.authoritarian_score = float(auth_score)
        if hasattr(existing, "centrist_score"):
            existing.centrist_score = float(centrist_score)
        existing.confidence_score = float(conf)
        existing.marpor_codes = list(marpor_codes)
        existing.full_results = fr

        if hasattr(existing, "processing_time_seconds"):
            existing.processing_time_seconds = float(processing_time_seconds) if processing_time_seconds is not None else None
        if hasattr(existing, "key_statement_count"):
            existing.key_statement_count = int(_as_int(summary.get("key_statement_count"), 0))
        if hasattr(existing, "segment_count"):
            existing.segment_count = int(len(fr.get("statements") or []))
        if hasattr(existing, "updated_at"):
            existing.updated_at = datetime.utcnow()

        db.add(existing)
        return existing

    created_kwargs: Dict[str, Any] = dict(
        speech_id=speech.id,
        ideology_family=dom_family,
        ideology_subtype=dom_subtype,
        libertarian_score=float(lib_score),
        authoritarian_score=float(auth_score),
        confidence_score=float(conf),
        marpor_codes=list(marpor_codes),
        full_results=fr,
        created_at=datetime.utcnow(),
    )
    if hasattr(Analysis, "centrist_score"):
        created_kwargs["centrist_score"] = float(centrist_score)
    if hasattr(Analysis, "processing_time_seconds"):
        created_kwargs["processing_time_seconds"] = float(processing_time_seconds) if processing_time_seconds is not None else None
    if hasattr(Analysis, "key_statement_count"):
        created_kwargs["key_statement_count"] = int(_as_int(summary.get("key_statement_count"), 0))
    if hasattr(Analysis, "segment_count"):
        created_kwargs["segment_count"] = int(len(fr.get("statements") or []))

    created = Analysis(**created_kwargs)
    db.add(created)
    return created


# =============================================================================
# BACKGROUND ANALYSIS
# =============================================================================
async def _run_analysis_and_persist(speech_id: int) -> None:
    if not _INGEST_AVAILABLE or ingest_speech is None:
        logger.error("ingest_speech not available; cannot analyze speech_id=%s", speech_id)
        return

    with _fresh_db_session() as db:
        speech = db.query(Speech).filter(Speech.id == speech_id).first()
        if not speech:
            logger.error("Speech not found for analysis: %s", speech_id)
            return

        try:
            speech.status = "processing"
            db.commit()

            use_sem_scoring = bool(getattr(speech, "use_semantic_scoring", True))
            embedder = _get_embedder(use_semantic=use_sem_scoring)

            # ✅ do NOT hard-code 0.6; keep consistent with ingestion default (recall-friendly)
            code_threshold = float(DEFAULT_CODE_THRESHOLD)
            code_threshold = _clamp(code_threshold, 0.10, 0.95)

            t0 = time.time()
            try:
                result = await ingest_speech(
                    text=speech.text,
                    speech_title=speech.title or "",
                    speaker=speech.speaker or "",
                    use_semantic_scoring=use_sem_scoring,
                    embedder=embedder,
                    code_threshold=code_threshold,
                )
            except TypeError:
                # Older signatures
                result = await ingest_speech(
                    text=speech.text,
                    speech_title=speech.title or "",
                    speaker=speech.speaker or "",
                    use_semantic_scoring=use_sem_scoring,
                    embedder=embedder,
                )
            dt = time.time() - t0

            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result.get("error")))

            fr = result if isinstance(result, dict) else _safe_full_results(result)
            if not isinstance(fr, dict):
                raise RuntimeError("Ingestion returned invalid result type")

            # persist actual settings used (debuggable)
            fr.setdefault("metadata", {})
            if isinstance(fr.get("metadata"), dict):
                fr["metadata"]["code_threshold"] = code_threshold
                fr["metadata"]["use_semantic_scoring"] = bool(use_sem_scoring)

            fr = _canonicalize_full_results(speech=speech, fr_in=fr)
            analysis = _upsert_analysis_row(db, speech=speech, full_results=fr, processing_time_seconds=float(dt))

            # optional question generation (store in DB)
            if _QUESTION_GENERATOR_AVAILABLE and question_generator is not None:
                try:
                    summary = fr.get("analysis_summary") if isinstance(fr.get("analysis_summary"), dict) else {}
                    ideology_result = {
                        "ideology_family": "Evidence-labeled",
                        "ideology_subtype": None,
                        "confidence_score": _as_float(summary.get("avg_confidence_score"), 0.0),
                        "marpor_codes": summary.get("marpor_codes") if isinstance(summary.get("marpor_codes"), list) else [],
                        "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, ECON_LEFT: 0.0, ECON_RIGHT: 0.0, CENTRIST_FAMILY: 0.0},
                        "ideology_2d": fr.get("ideology_2d") or _extract_ideology_2d_anywhere(fr),
                    }
                    key_segments = [_coerce_evidence_item(x) for x in (fr.get("key_statements") or []) if isinstance(x, dict)]
                    qs = await question_generator.generate_questions_with_llm(
                        question_type="journalistic",
                        speech_title=speech.title or "",
                        speaker=speech.speaker or "",
                        ideology_result=ideology_result,
                        key_segments=key_segments[:6],
                        llm_provider=speech.llm_provider or "openai",
                        llm_model=speech.llm_model or "gpt-4o-mini",
                        max_questions=3,
                    )
                    if isinstance(qs, list) and qs:
                        db.query(Question).filter(Question.speech_id == speech.id).delete()
                        for i, q_text in enumerate(qs):
                            db.add(
                                Question(
                                    speech_id=speech.id,
                                    question_text=str(q_text),
                                    question_type="journalistic",
                                    question_order=i,
                                )
                            )
                except Exception as e:
                    logger.warning("Speech %s: question generation failed: %s", speech_id, e, exc_info=True)

            speech.status = "completed"
            speech.analyzed_at = datetime.utcnow()
            db.commit()
            db.refresh(analysis)

            logger.info("Analysis completed for speech_id=%s in %.2fs", speech_id, dt)

        except Exception as e:
            logger.error("Analysis failed for speech_id=%s: %s", speech_id, e, exc_info=True)
            try:
                speech.status = "failed"
                speech.analyzed_at = None
                db.commit()
            except Exception:
                db.rollback()


# =============================================================================
# ENDPOINTS
# =============================================================================
@router.post("/upload")
async def upload_speech(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    speaker: str = Form(...),
    date_str: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    event: Optional[str] = Form(None),
    language: str = Form("en"),
    is_public: bool = Form(False),
    llm_provider: str = Form("openai"),
    llm_model: str = Form("gpt-4o-mini"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    _require_auth_if_enabled(current_user)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = Path(file.filename).suffix.lower()
    all_formats = TEXT_FORMATS | DOCUMENT_FORMATS | AUDIO_FORMATS | VIDEO_FORMATS
    if file_ext not in all_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")

    needs_transcription = file_ext in (AUDIO_FORMATS | VIDEO_FORMATS)
    if needs_transcription and not _TRANSCRIPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Audio/video transcription not available.")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_filename = _safe_filename(file.filename, prefix=ts)
    file_path = UPLOAD_DIR / safe_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    transcript = None
    media_url = None
    try:
        if needs_transcription:
            if transcribe_media_file is None:
                raise HTTPException(status_code=503, detail="Transcription service not available")
            tr = transcribe_media_file(
                str(file_path),
                language=None if language == "en" else language,
                with_timestamps=False,
            )
            transcript = _clean_text((tr or {}).get("text", "") or "")
            media_url = f"/media/uploads/{safe_filename}"
        else:
            transcript = _extract_text_from_file(file_path)
            # non-media: delete extracted file after reading
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
    except HTTPException:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

    if not transcript or len(transcript.strip()) < MIN_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"Extracted text too short (min {MIN_TEXT_CHARS} chars)")

    speech_date = None
    if date_str:
        try:
            speech_date = _parse_iso_date(date_str)
        except HTTPException:
            speech_date = None

    wc = _word_count(transcript)

    try:
        db_speech = Speech(
            user_id=(current_user.id if current_user else None),
            title=title.strip(),
            speaker=speaker.strip(),
            text=transcript,
            date=speech_date,
            location=location,
            event=event,
            source_url=None,
            source_type="upload",
            media_url=media_url,
            language=language or "en",
            word_count=wc,
            llm_provider=llm_provider,
            llm_model=llm_model,
            use_semantic_segmentation=True,
            use_semantic_scoring=True,
            is_public=bool(is_public),
            status="uploaded",
        )
        db.add(db_speech)
        db.commit()
        db.refresh(db_speech)

        if _INGEST_AVAILABLE and ingest_speech is not None:
            background_tasks.add_task(_run_analysis_and_persist, db_speech.id)

        return _response(
            True,
            data=_speech_to_dict(db_speech, include_text=False, include_analysis_summary=False, analysis=None),
            message="File uploaded successfully. Analysis queued.",
            meta={"speech_id": db_speech.id},
        )
    except SQLAlchemyError as e:
        db.rollback()
        # cleanup media file if DB write failed (only for audio/video uploads)
        if media_url:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.post("/")
async def create_speech(
    payload: SpeechCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    _require_auth_if_enabled(current_user)

    speech_date = _parse_iso_date(payload.date)
    cleaned = _clean_text(payload.text)
    wc = _word_count(cleaned)

    try:
        speech = Speech(
            user_id=(current_user.id if current_user else None),
            title=payload.title.strip(),
            speaker=payload.speaker.strip(),
            text=cleaned,
            date=speech_date,
            location=payload.location,
            event=payload.event,
            source_url=payload.source_url,
            source_type=payload.source_type,
            language=payload.language or "en",
            word_count=wc,
            llm_provider=payload.llm_provider,
            llm_model=payload.llm_model,
            use_semantic_segmentation=bool(payload.use_semantic_segmentation),
            use_semantic_scoring=bool(payload.use_semantic_scoring),
            is_public=bool(payload.is_public),
            status="pending",
        )
        db.add(speech)
        db.commit()
        db.refresh(speech)

        if payload.analyze_immediately and _INGEST_AVAILABLE and ingest_speech is not None:
            background_tasks.add_task(_run_analysis_and_persist, speech.id)

        return _response(True, data=_speech_to_dict(speech, include_text=False, include_analysis_summary=False, analysis=None))
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.get("/stats")
async def stats(db: Session = Depends(get_db)):
    total = int(db.execute(text("SELECT COUNT(*) FROM speeches")).scalar() or 0)
    analyzed = int(db.execute(text("SELECT COUNT(*) FROM speeches s JOIN analyses a ON a.speech_id = s.id")).scalar() or 0)
    avg_conf = db.execute(text("SELECT AVG(confidence_score) FROM analyses")).scalar()
    return _response(
        True,
        data={
            "total_speeches": int(total),
            "analyzed_speeches": int(analyzed),
            "pending_speeches": int(total - analyzed),
            "average_confidence_score": round(float(avg_conf or 0.0), 4),
        },
    )


@router.get("/search")
async def search_speeches(
    q: str = Query(..., min_length=2),
    search_in: str = Query("all"),
    include_public: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    term = f"%{q.strip()}%"
    query = db.query(Speech)

    if current_user is not None:
        query = query.filter(or_(Speech.user_id == current_user.id, Speech.is_public == True)) if include_public else query.filter(Speech.user_id == current_user.id)
    else:
        query = query.filter(Speech.is_public == True)

    if search_in == "title":
        query = query.filter(Speech.title.ilike(term))
    elif search_in == "speaker":
        query = query.filter(Speech.speaker.ilike(term))
    elif search_in == "text":
        query = query.filter(Speech.text.ilike(term))
    else:
        query = query.filter(or_(Speech.title.ilike(term), Speech.speaker.ilike(term), Speech.text.ilike(term)))

    speeches = query.order_by(Speech.created_at.desc()).limit(limit).all()
    analysis_map = {a.speech_id: a for a in db.query(Analysis).filter(Analysis.speech_id.in_([s.id for s in speeches])).all()} if speeches else {}

    return _response(
        True,
        data={
            "speeches": [_speech_to_dict(s, include_text=False, include_analysis_summary=True, analysis=analysis_map.get(s.id)) for s in speeches],
            "count": len(speeches),
        },
    )


@router.get("/")
async def list_speeches(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    include_public: bool = Query(False),
    speaker: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    has_analysis: Optional[bool] = Query(None),
    is_public: Optional[bool] = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    q = db.query(Speech)

    if current_user is not None:
        q = q.filter(or_(Speech.user_id == current_user.id, Speech.is_public == True)) if include_public else q.filter(Speech.user_id == current_user.id)
    else:
        q = q.filter(Speech.is_public == True)

    if speaker:
        q = q.filter(Speech.speaker.ilike(f"%{speaker.strip()}%"))
    if status_filter:
        q = q.filter(Speech.status == status_filter)
    if is_public is not None:
        q = q.filter(Speech.is_public == bool(is_public))

    if has_analysis is True:
        q = q.filter(text("EXISTS (SELECT 1 FROM analyses a WHERE a.speech_id = speeches.id)"))
    elif has_analysis is False:
        q = q.filter(text("NOT EXISTS (SELECT 1 FROM analyses a WHERE a.speech_id = speeches.id)"))

    sort_col = getattr(Speech, sort_by, None) or Speech.created_at
    q = q.order_by(sort_col.asc() if str(sort_order).lower() == "asc" else sort_col.desc())

    total = q.count()
    offset = (page - 1) * page_size
    rows = q.offset(offset).limit(page_size).all()

    analysis_map = {a.speech_id: a for a in db.query(Analysis).filter(Analysis.speech_id.in_([s.id for s in rows])).all()} if rows else {}
    speeches = [_speech_to_dict(s, include_text=False, include_analysis_summary=True, analysis=analysis_map.get(s.id)) for s in rows]
    total_pages = (total + page_size - 1) // page_size

    return _response(True, data={"speeches": speeches, "total": total, "page": page, "page_size": page_size, "total_pages": total_pages})


@router.get("/{speech_id}/status")
async def get_speech_status(
    speech_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    if not _can_read(speech, current_user):
        raise HTTPException(status_code=403, detail="Not permitted")
    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    return _response(
        True,
        data={
            "speech_id": speech_id,
            "status": speech.status,
            "analyzed_at": speech.analyzed_at.isoformat() if speech.analyzed_at else None,
            "has_analysis": bool(analysis is not None),
        },
    )


@router.get("/{speech_id}/media")
async def get_speech_media(
    speech_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    if not _can_read(speech, current_user):
        raise HTTPException(status_code=403, detail="Not permitted")
    if not getattr(speech, "media_url", None):
        raise HTTPException(status_code=404, detail="No media file available")
    return _response(True, data={"media_url": speech.media_url})


@router.get("/{speech_id}/analysis")
async def get_analysis(
    speech_id: int,
    media_duration_seconds: Optional[float] = Query(None, ge=0.0),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    if not _can_read(speech, current_user):
        raise HTTPException(status_code=403, detail="Not permitted")

    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found. Speech status: {speech.status}.")

    fr = _safe_full_results(getattr(analysis, "full_results", None))
    fr = _canonicalize_full_results(speech=speech, fr_in=fr)

    merged = dict(fr)
    merged.update({"speech_id": speech_id, "title": speech.title, "speaker": speech.speaker, "text": speech.text, "transcript_text": speech.text})
    merged = _apply_media_jump_time_support(merged, duration=media_duration_seconds)
    return _response(True, data=merged)


@router.get("/{speech_id}/key-statements")
async def get_key_statements(
    speech_id: int,
    media_duration_seconds: Optional[float] = Query(None, ge=0.0),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    if not _can_read(speech, current_user):
        raise HTTPException(status_code=403, detail="Not permitted")

    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    fr = _safe_full_results(getattr(analysis, "full_results", None))
    fr = _canonicalize_full_results(speech=speech, fr_in=fr)

    payload = {"text": speech.text, "transcript_text": speech.text, "key_statements": fr.get("key_statements") or []}
    payload = _apply_media_jump_time_support(payload, duration=media_duration_seconds)
    out_ks = payload.get("key_statements") if isinstance(payload.get("key_statements"), list) else []
    return _response(True, data={"key_statements": out_ks, "count": len(out_ks)})


@router.get("/{speech_id}/sections")
async def get_sections(
    speech_id: int,
    media_duration_seconds: Optional[float] = Query(None, ge=0.0),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    if not _can_read(speech, current_user):
        raise HTTPException(status_code=403, detail="Not permitted")

    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    fr = _safe_full_results(getattr(analysis, "full_results", None))
    fr = _canonicalize_full_results(speech=speech, fr_in=fr)

    sections = fr.get("statements") if isinstance(fr.get("statements"), list) else []
    payload = {"text": speech.text, "transcript_text": speech.text, "sections": sections, "segments": sections, "statements": sections}
    payload = _apply_media_jump_time_support(payload, duration=media_duration_seconds)
    out_sections = payload.get("sections") if isinstance(payload.get("sections"), list) else sections
    return _response(True, data={"sections": out_sections, "count": len(out_sections)})


@router.get("/{speech_id}")
async def get_speech(
    speech_id: int,
    include_text: bool = Query(True),
    include_analysis: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    if not _can_read(speech, current_user):
        raise HTTPException(status_code=403, detail="Not permitted")

    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first() if include_analysis else None
    return _response(True, data=_speech_to_dict(speech, include_text=include_text, include_analysis_summary=include_analysis, analysis=analysis))


@router.get("/{speech_id}/full")
async def get_speech_full(
    speech_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    if not _can_read(speech, current_user):
        raise HTTPException(status_code=403, detail="Not permitted")

    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    data = _speech_to_dict(speech, include_text=True, include_analysis_summary=True, analysis=analysis)

    if analysis:
        fr = _safe_full_results(getattr(analysis, "full_results", None))
        fr = _canonicalize_full_results(speech=speech, fr_in=fr)
        data["analysis"] = fr

    qs = db.query(Question).filter(Question.speech_id == speech_id).order_by(Question.question_order.asc()).all()
    data["questions"] = [
        {
            "id": q.id,
            "question_text": q.question_text,
            "question_type": q.question_type,
            "order": q.question_order,
            "created_at": q.created_at.isoformat() if q.created_at else None,
        }
        for q in qs
    ]
    return _response(True, data=data)


@router.post("/{speech_id}/questions/generate")
async def generate_questions_for_speech(
    speech_id: int,
    body: GenerateQuestionsBody,
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    """
    Legacy compatibility endpoint used by frontend fallback:
      POST /api/speeches/{speech_id}/questions/generate
    """
    if not _QUESTION_GENERATOR_AVAILABLE or question_generator is None:
        raise HTTPException(status_code=503, detail="Question generation service is not available.")

    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    if not _can_read(speech, current_user):
        raise HTTPException(status_code=403, detail="Not permitted")

    provider = (body.llm_provider or getattr(speech, "llm_provider", None) or "openai").strip() or "openai"
    model = (body.llm_model or getattr(speech, "llm_model", None) or "gpt-4o-mini").strip() or "gpt-4o-mini"

    qt = (body.question_type or "journalistic").strip().lower()
    if qt not in ("journalistic", "technical"):
        qt = "journalistic"

    latest = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    if not latest:
        raise HTTPException(status_code=404, detail="No analysis found for this speech.")

    fr = _safe_full_results(getattr(latest, "full_results", None))
    fr = _canonicalize_full_results(speech=speech, fr_in=fr)

    summary = fr.get("analysis_summary") if isinstance(fr.get("analysis_summary"), dict) else _ensure_summary_aliases({})
    ideology_result = {
        "ideology_family": "Evidence-labeled",
        "ideology_subtype": None,
        "confidence_score": _as_float(summary.get("avg_confidence_score"), 0.0),
        "marpor_codes": summary.get("marpor_codes") if isinstance(summary.get("marpor_codes"), list) else [],
        "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, ECON_LEFT: 0.0, ECON_RIGHT: 0.0, CENTRIST_FAMILY: 0.0},
        "ideology_2d": fr.get("ideology_2d") or _extract_ideology_2d_anywhere(fr),
    }

    ks = fr.get("key_statements") if isinstance(fr.get("key_statements"), list) else []
    key_segments = [
        {
            "text": str(x.get("text") or x.get("full_text") or ""),
            "ideology_family": x.get("ideology_family"),
            "ideology_subtype": x.get("ideology_subtype"),
            "marpor_codes": x.get("marpor_codes") if isinstance(x.get("marpor_codes"), list) else [],
        }
        for x in ks[:6]
        if isinstance(x, dict)
    ]

    try:
        questions = await question_generator.generate_questions_with_llm(
            question_type=qt,
            speech_title=speech.title or "",
            speaker=speech.speaker or "",
            ideology_result=ideology_result,
            key_segments=key_segments,
            llm_provider=provider,
            llm_model=model,
            max_questions=int(body.max_questions),
        )
        questions = [str(q).strip() for q in (questions if isinstance(questions, list) else []) if str(q).strip()]
    except Exception as e:
        logger.error("Questions generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Question generation failed.")

    return _response(True, data={"speech_id": speech_id, "question_type": qt, "questions": questions}, message="Questions generated.")


@router.put("/{speech_id}")
async def update_speech(
    speech_id: int,
    payload: SpeechUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    _require_auth_if_enabled(current_user)

    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    _require_write_access(speech, current_user)

    update = payload.model_dump(exclude_unset=True)
    text_changed = False

    if "date" in update:
        update["date"] = _parse_iso_date(update.get("date"))

    if "text" in update and update["text"] is not None:
        new_text = _clean_text(update["text"])
        if new_text != (speech.text or ""):
            text_changed = True
            speech.word_count = _word_count(new_text)
        update["text"] = new_text

    for k, v in update.items():
        setattr(speech, k, v)

    if text_changed:
        speech.status = "pending"
        speech.analyzed_at = None

    speech.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(speech)

    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    return _response(
        True,
        data=_speech_to_dict(speech, include_text=False, include_analysis_summary=True, analysis=analysis),
        message="Speech updated." + (" Analysis marked pending due to text change." if text_changed else ""),
    )


@router.delete("/{speech_id}")
async def delete_speech(
    speech_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    _require_auth_if_enabled(current_user)

    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    _require_write_access(speech, current_user)

    media_url = getattr(speech, "media_url", None)
    if isinstance(media_url, str) and media_url.startswith("/media/uploads/"):
        filename = media_url.split("/media/uploads/")[-1]
        p = UPLOAD_DIR / filename
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    db.query(Analysis).filter(Analysis.speech_id == speech_id).delete()
    db.query(Question).filter(Question.speech_id == speech_id).delete()
    db.delete(speech)
    db.commit()

    return _response(True, message="Speech deleted.")


@router.post("/{speech_id}/analyze")
async def analyze_existing_speech(
    speech_id: int,
    background_tasks: BackgroundTasks,
    force: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    _require_auth_if_enabled(current_user)
    if not _INGEST_AVAILABLE or ingest_speech is None:
        raise HTTPException(status_code=503, detail="Analysis service not available")

    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=404, detail="Speech not found")
    _require_write_access(speech, current_user)

    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    if speech.status == "processing":
        return _response(True, data={"speech_id": speech_id, "status": "processing"}, message="Analysis already running.")
    if speech.status == "completed" and analysis is not None and not force:
        return _response(
            True,
            data=_speech_to_dict(speech, include_text=False, include_analysis_summary=True, analysis=analysis),
            message="Analysis already exists. Use force=true to re-run.",
        )

    speech.status = "pending"
    db.commit()

    background_tasks.add_task(_run_analysis_and_persist, speech_id)
    return _response(True, data={"speech_id": speech_id, "status": "queued"}, message="Analysis queued.")


__all__ = ["router"]