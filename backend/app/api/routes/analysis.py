
# backend/app/api/routes/analysis.py

"""
DiscourseAI Backend — Analysis Routes (Aligned with AnalysisPage)

Exposes:
- GET  /api/analysis/speech/{speech_id}
    - Optional query param: media_duration_seconds
- POST /api/analysis/speech
    - Analyze by speech_id or raw text (creates Speech if needed)
- POST /api/analysis/speech/{speech_id}/reanalyze
    - Re-run ingestion and UPSERT Analysis (one analysis row per speech)
- POST /api/analysis/questions/generate
- GET  /api/analysis/health

POLICY UPDATE:
- Neutral is removed from the pipeline outputs.
- Centrist is the only non-ideological family label.
- Centrist has no subtype (always None).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.utils.responses import create_response
from app.database.connection import get_db
from app.database.models import Analysis, Speech

# Project optional
try:
    from app.database.models import Project  # type: ignore
    _PROJECT_AVAILABLE = True
except Exception:
    Project = None  # type: ignore
    _PROJECT_AVAILABLE = False

# Ingestion optional
try:
    from app.services.speech_ingestion import ingest_speech  # type: ignore
    _INGEST_AVAILABLE = True
except Exception:
    ingest_speech = None  # type: ignore
    _INGEST_AVAILABLE = False

# Question generator optional
try:
    from app.services.question_generator import question_generator  # type: ignore
    _QUESTION_GENERATOR_AVAILABLE = True
except Exception:
    question_generator = None  # type: ignore
    _QUESTION_GENERATOR_AVAILABLE = False


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIB_FAMILY = "Libertarian"
AUTH_FAMILY = "Authoritarian"
CENTRIST_FAMILY = "Centrist"

_ALLOWED_FAMILIES = {LIB_FAMILY, AUTH_FAMILY, CENTRIST_FAMILY}

# Legacy inputs you might still have in persisted JSON from older runs:
LEGACY_NEUTRAL = "Neutral"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalyzeSpeechRequest(BaseModel):
    speech_id: Optional[int] = Field(default=None, description="Existing speech ID to analyze")
    text: Optional[str] = Field(default=None, description="Speech text to analyze (creates new speech if no speech_id)")

    title: Optional[str] = None
    speaker: Optional[str] = None

    project_id: Optional[int] = None

    use_semantic: bool = True
    threshold: float = 0.6

    include_questions: bool = True
    question_types: List[str] = Field(default_factory=lambda: ["journalistic", "technical"])
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    max_questions: int = 6

    media_duration_seconds: Optional[float] = Field(
        default=None,
        description=(
            "If provided, backend computes time_begin/time_end for key_statements (and segments) "
            "using start_char/end_char ratios against transcript length."
        ),
    )


class ReanalyzeRequest(BaseModel):
    use_semantic: bool = True
    threshold: float = 0.6
    include_questions: bool = True
    question_types: List[str] = Field(default_factory=lambda: ["journalistic", "technical"])
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    max_questions: int = 6
    media_duration_seconds: Optional[float] = None


class GenerateQuestionsRequest(BaseModel):
    speech_id: int = Field(..., ge=1)
    question_type: str = Field(default="journalistic")
    max_questions: int = Field(default=5, ge=1, le=8)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_z() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize_question_types(qtypes: List[str]) -> List[str]:
    qtypes = [str(t or "").strip().lower() for t in (qtypes or [])]
    qtypes = [t for t in qtypes if t in ("journalistic", "technical")]
    return qtypes or ["journalistic", "technical"]


def _normalize_family(fam: Any) -> str:
    """
    Canonical family normalization for API:
    - Always emits one of: Libertarian | Authoritarian | Centrist
    - Unknown or legacy values -> Centrist
    """
    f = str(fam or "").strip()
    if not f:
        return CENTRIST_FAMILY
    if f == LEGACY_NEUTRAL:
        return CENTRIST_FAMILY
    if f not in _ALLOWED_FAMILIES:
        return CENTRIST_FAMILY
    return f


def _normalize_subtype(family: str, subtype: Any) -> Optional[str]:
    """
    Policy:
    - Centrist has no subtype.
    - Lib/Auth subtype: keep if non-empty.
    """
    fam = _normalize_family(family)
    if fam == CENTRIST_FAMILY:
        return None
    sub = str(subtype).strip() if subtype is not None else ""
    return sub or None


def _sanitize_scores_dict(scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure scores dict uses Centrist key, never Neutral.
    If both present, Centrist wins.
    """
    if not isinstance(scores, dict):
        return {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, CENTRIST_FAMILY: 0.0}

    out = dict(scores)

    # If Centrist missing but legacy Neutral exists, move it.
    if CENTRIST_FAMILY not in out and LEGACY_NEUTRAL in out:
        out[CENTRIST_FAMILY] = out.get(LEGACY_NEUTRAL)

    # Remove legacy Neutral key always
    out.pop(LEGACY_NEUTRAL, None)

    # Make sure canonical keys exist
    out.setdefault(LIB_FAMILY, 0.0)
    out.setdefault(AUTH_FAMILY, 0.0)
    out.setdefault(CENTRIST_FAMILY, 0.0)

    return out


def _has_evidence_for_family(
    speech_level: Dict[str, Any],
    family: str,
) -> bool:
    """
    Robust evidence presence check:
    Preferred: speech_level.diagnostics.{lib_evidence_count,auth_evidence_count}
    Fallback: speech_level.subtype_breakdown evidence_count by subtype naming
    """
    if not isinstance(speech_level, dict):
        return False

    fam = _normalize_family(family)
    diagnostics = speech_level.get("diagnostics") or {}
    if isinstance(diagnostics, dict):
        if fam == LIB_FAMILY:
            c = diagnostics.get("lib_evidence_count", None)
            if isinstance(c, (int, float)) and int(c) > 0:
                return True
        if fam == AUTH_FAMILY:
            c = diagnostics.get("auth_evidence_count", None)
            if isinstance(c, (int, float)) and int(c) > 0:
                return True

    subtype_breakdown = speech_level.get("subtype_breakdown", {})
    if not isinstance(subtype_breakdown, dict) or not subtype_breakdown:
        return False

    fam_lower = fam.lower()
    for subtype_name, subtype_data in subtype_breakdown.items():
        if not isinstance(subtype_data, dict):
            continue
        subtype_lower = str(subtype_name).lower()
        if fam_lower not in subtype_lower:
            continue
        evidence_count = subtype_data.get("evidence_count", 0)
        if isinstance(evidence_count, (int, float)) and int(evidence_count) > 0:
            return True

    return False


def _extract_scores_top(payload: Dict[str, Any]) -> Tuple[float, float, float, float, str, Optional[str], List[str]]:
    """
    Returns:
      lib, auth, centrist, conf, dominant_family, dominant_subtype, marpor_codes

    Accepts legacy 'Neutral' if present but never returns it.
    Applies evidence-validation to avoid ghost scores.
    """
    payload = payload or {}
    speech_level = payload.get("speech_level") or {}

    # Preferred: speech_level.scores
    if isinstance(speech_level, dict) and isinstance(speech_level.get("scores"), dict):
        scores = _sanitize_scores_dict(speech_level.get("scores") or {})

        lib = _as_float(scores.get(LIB_FAMILY), 0.0)
        auth = _as_float(scores.get(AUTH_FAMILY), 0.0)
        cen = _as_float(scores.get(CENTRIST_FAMILY), 0.0)

        conf = _as_float(speech_level.get("confidence_score"), 0.0)

        dom_fam_raw = speech_level.get("dominant_family") or payload.get("ideology_family") or CENTRIST_FAMILY
        dom_sub_raw = speech_level.get("dominant_subtype") or payload.get("ideology_subtype")

        dom_fam = _normalize_family(dom_fam_raw)
        dom_sub = _normalize_subtype(dom_fam, dom_sub_raw)

        marpor_codes = speech_level.get("marpor_codes") or payload.get("marpor_codes") or []
        if isinstance(marpor_codes, list):
            marpor_codes = [str(x) for x in marpor_codes if str(x).strip()]
        else:
            marpor_codes = []

        # Evidence validation: if no evidence for a family, zero it, then renormalize
        lib_has = _has_evidence_for_family(speech_level, LIB_FAMILY)
        auth_has = _has_evidence_for_family(speech_level, AUTH_FAMILY)

        if not lib_has:
            lib = 0.0
        if not auth_has:
            auth = 0.0

        total = lib + auth + cen
        if total > 0:
            lib = (lib / total) * 100.0
            auth = (auth / total) * 100.0
            cen = (cen / total) * 100.0
        else:
            lib, auth, cen = 0.0, 0.0, 100.0

        if lib == 0.0 and auth == 0.0:
            dom_fam = CENTRIST_FAMILY
            dom_sub = None
        elif lib > 0 and auth == 0.0:
            dom_fam = LIB_FAMILY
        elif auth > 0 and lib == 0.0:
            dom_fam = AUTH_FAMILY

        return lib, auth, cen, conf, dom_fam, dom_sub, marpor_codes

    # Fallback: top-level scores
    scores = payload.get("scores") or payload.get("overview", {}).get("scores") or {}
    scores = _sanitize_scores_dict(scores) if isinstance(scores, dict) else _sanitize_scores_dict({})

    lib = _as_float(payload.get("libertarian_score", scores.get(LIB_FAMILY)), 0.0)
    auth = _as_float(payload.get("authoritarian_score", scores.get(AUTH_FAMILY)), 0.0)

    cen = payload.get("centrist_score", None)
    if cen is None:
        cen = scores.get(CENTRIST_FAMILY)
    if cen is None and LEGACY_NEUTRAL in (payload.get("scores") or {}):
        cen = (payload.get("scores") or {}).get(LEGACY_NEUTRAL)
    cen = _as_float(cen, 0.0)

    conf = _as_float(payload.get("confidence_score") or payload.get("scientific_summary", {}).get("avg_confidence"), 0.0)

    dom_fam_raw = payload.get("ideology_family") or payload.get("dominant_family") or CENTRIST_FAMILY
    dom_sub_raw = payload.get("ideology_subtype") or payload.get("dominant_subtype")

    dom_fam = _normalize_family(dom_fam_raw)
    dom_sub = _normalize_subtype(dom_fam, dom_sub_raw)

    marpor_codes = payload.get("marpor_codes") or []
    if isinstance(marpor_codes, list):
        marpor_codes = [str(x) for x in marpor_codes if str(x).strip()]
    else:
        marpor_codes = []

    # Evidence validation (fallback path)
    sl2 = payload.get("speech_level") or payload or {}
    if isinstance(sl2, dict):
        lib_has = _has_evidence_for_family(sl2, LIB_FAMILY)
        auth_has = _has_evidence_for_family(sl2, AUTH_FAMILY)
        if not lib_has:
            lib = 0.0
        if not auth_has:
            auth = 0.0
        total = lib + auth + cen
        if total > 0:
            lib = (lib / total) * 100.0
            auth = (auth / total) * 100.0
            cen = (cen / total) * 100.0
        else:
            lib, auth, cen = 0.0, 0.0, 100.0
        if lib == 0.0 and auth == 0.0:
            dom_fam = CENTRIST_FAMILY
            dom_sub = None

    return lib, auth, cen, conf, dom_fam, dom_sub, marpor_codes


def _estimate_time_from_char(start_char: Optional[int], total_chars: int, duration: float) -> Optional[float]:
    if start_char is None:
        return None
    if total_chars <= 0 or duration <= 0:
        return None
    frac = _clamp(start_char / total_chars, 0.0, 1.0)
    return round(frac * duration, 3)


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
            if t is None:
                t = round(((i + 1) / (n + 1)) * duration, 3)
            it[out_begin_key] = t

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
                nudge = min(1.5, max(0.25, duration * 0.01))
                t_end = round(_clamp(base + nudge, 0.0, duration), 3)

            it[out_end_key] = t_end


def _apply_media_jump_time_support(payload: Dict[str, Any], duration: Optional[float]) -> Dict[str, Any]:
    payload = payload or {}
    dur = _as_float(duration, 0.0)
    if dur <= 0:
        return payload

    transcript_text = payload.get("transcript_text") or payload.get("text") or ""
    transcript_len = len(transcript_text) if isinstance(transcript_text, str) else 0
    if transcript_len <= 0:
        return payload

    for key in ("key_statements", "segments", "sections"):
        arr = payload.get(key)
        if isinstance(arr, list):
            _ensure_time_fields_for_items(
                items=[x for x in arr if isinstance(x, dict)],
                transcript_len=transcript_len,
                duration=dur,
            )

    return payload


def _normalize_analysis_shape(data: Dict[str, Any], *, speech: Speech) -> Dict[str, Any]:
    """
    Ensure the analysis JSON has consistent keys expected by frontend:
    - text + transcript_text
    - key_statements array
    - segments/sections arrays (both)
    """
    out: Dict[str, Any] = dict(data or {})

    text = (speech.text or "") if speech else (out.get("text") or out.get("transcript_text") or "")
    out["text"] = text
    out["transcript_text"] = text

    out.setdefault("title", getattr(speech, "title", None))
    out.setdefault("speaker", getattr(speech, "speaker", None))

    if not isinstance(out.get("key_statements"), list):
        for alt in ("key_segments", "highlights"):
            if isinstance(out.get(alt), list):
                out["key_statements"] = out.get(alt)
                break

    if not isinstance(out.get("key_statements"), list):
        sl = out.get("speech_level")
        if isinstance(sl, dict) and isinstance(sl.get("key_statements"), list):
            out["key_statements"] = sl.get("key_statements")

    if isinstance(out.get("sections"), list) and not isinstance(out.get("segments"), list):
        out["segments"] = out["sections"]
    if isinstance(out.get("segments"), list) and not isinstance(out.get("sections"), list):
        out["sections"] = out["segments"]

    return out


def _purge_legacy_neutral_everywhere(obj: Any) -> Any:
    """
    Recursively:
      - Sanitize any 'scores' dict to ensure Centrist/never Neutral.
      - Normalize family/subtype keys:
          * family keys: ideology_family, dominant_family, family
          * subtype keys: ideology_subtype, dominant_subtype, subtype
        Ensure: Centrist => subtype=None.
      - Leaves unrelated keys untouched.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _purge_legacy_neutral_everywhere(v)

        # Sanitize any nested scores
        if isinstance(out.get("scores"), dict):
            out["scores"] = _sanitize_scores_dict(out["scores"])

        # Normalize family/subtype pairs commonly used in payloads
        family_keys = ("ideology_family", "dominant_family", "family")
        subtype_keys = ("ideology_subtype", "dominant_subtype", "subtype")

        detected_fam_key = next((fk for fk in family_keys if fk in out), None)
        if detected_fam_key:
            fam = _normalize_family(out.get(detected_fam_key))
            out[detected_fam_key] = fam
            for sk in subtype_keys:
                if sk in out:
                    out[sk] = _normalize_subtype(fam, out.get(sk))

        return out

    if isinstance(obj, list):
        return [_purge_legacy_neutral_everywhere(x) for x in obj]

    return obj


async def _generate_questions_safe(
    *,
    include_questions: bool,
    question_types: List[str],
    llm_provider: str,
    llm_model: str,
    max_questions: int,
    speech_title: str,
    speaker: str,
    ideology_result: Dict[str, Any],
    key_segments: List[Dict[str, Any]],
) -> List[str]:
    if not include_questions:
        return []
    if not _QUESTION_GENERATOR_AVAILABLE or question_generator is None:
        return []

    qtypes = _normalize_question_types(question_types)
    max_q = max(1, min(int(max_questions or 6), 8))
    per_type = max(1, min(8, max_q // max(1, len(qtypes))))

    out: List[str] = []
    for qt in qtypes:
        try:
            qs = await question_generator.generate_questions_with_llm(
                question_type=qt,
                speech_title=speech_title or "",
                speaker=speaker or "",
                ideology_result=ideology_result or {},
                key_segments=key_segments or [],
                llm_provider=llm_provider,
                llm_model=llm_model,
                max_questions=per_type,
            )
            if isinstance(qs, list):
                out.extend([str(x).strip() for x in qs if str(x).strip()])
        except Exception as e:
            logger.warning("Question generation failed for type=%s: %s", qt, e, exc_info=True)

    seen = set()
    final: List[str] = []
    for q in out:
        if q in seen:
            continue
        seen.add(q)
        final.append(q)
    return final[:max_q]


async def _run_full_analysis(
    *,
    text: str,
    title: str,
    speaker: str,
    use_semantic: bool,
    threshold: float,
    include_questions: bool,
    question_types: List[str],
    llm_provider: str,
    llm_model: str,
    max_questions: int,
    media_duration_seconds: Optional[float],
) -> Dict[str, Any]:
    if not _INGEST_AVAILABLE or ingest_speech is None:
        raise HTTPException(status_code=503, detail="Analysis ingestion service is not available.")

    ingest_result = await ingest_speech(
        text=text,
        speech_title=title or "",
        speaker=speaker or "",
        use_semantic_scoring=bool(use_semantic),
        code_threshold=float(threshold),
    )
    if not isinstance(ingest_result, dict):
        raise HTTPException(status_code=500, detail="Ingestion pipeline returned invalid result.")

    if ingest_result.get("error"):
        raise HTTPException(status_code=400, detail=str(ingest_result.get("error")))

    merged: Dict[str, Any] = dict(ingest_result)
    merged["text"] = text
    merged["transcript_text"] = text

    # ensure sections/segments symmetry
    if isinstance(merged.get("sections"), list) and not isinstance(merged.get("segments"), list):
        merged["segments"] = merged["sections"]
    if isinstance(merged.get("segments"), list) and not isinstance(merged.get("sections"), list):
        merged["sections"] = merged["segments"]

    # sanitize speech_level.scores (if present)
    sl = merged.get("speech_level")
    if isinstance(sl, dict) and isinstance(sl.get("scores"), dict):
        sl["scores"] = _sanitize_scores_dict(sl.get("scores") or {})

    # NEW: deep sanitize the entire payload (no Neutral anywhere; Centrist has no subtype)
    merged = _purge_legacy_neutral_everywhere(merged)

    # Extract normalized top fields (with evidence validation)
    lib, auth, cen, conf, dom_fam, dom_sub, marpor_codes = _extract_scores_top(merged)

    # Always provide top-level canonical scores dict
    merged["scores"] = {LIB_FAMILY: lib, AUTH_FAMILY: auth, CENTRIST_FAMILY: cen}
    merged["ideology_family"] = dom_fam
    merged["ideology_subtype"] = dom_sub
    merged["libertarian_score"] = lib
    merged["authoritarian_score"] = auth
    merged["centrist_score"] = cen
    merged["confidence_score"] = conf
    merged["marpor_codes"] = marpor_codes

    # Questions
    merged["questions"] = []
    if include_questions:
        ideology_result_for_questions: Dict[str, Any] = {
            "scores": merged.get("scores") or {},
            "ideology_family": merged.get("ideology_family"),
            "ideology_subtype": merged.get("ideology_subtype"),
            "confidence_score": merged.get("confidence_score"),
            "marpor_codes": merged.get("marpor_codes") or [],
        }

        key_segments_for_questions: List[Dict[str, Any]] = []
        ks = merged.get("key_statements")
        if isinstance(ks, list):
            for item in ks[:6]:
                if isinstance(item, dict):
                    key_segments_for_questions.append({"text": item.get("text") or item.get("full_text") or ""})

        merged["questions"] = await _generate_questions_safe(
            include_questions=True,
            question_types=question_types,
            llm_provider=llm_provider,
            llm_model=llm_model,
            max_questions=max_questions,
            speech_title=title or "",
            speaker=speaker or "",
            ideology_result=ideology_result_for_questions,
            key_segments=key_segments_for_questions,
        )

    merged = _apply_media_jump_time_support(merged, duration=media_duration_seconds)
    return merged


def _get_speech_llm_provider_model(speech: Speech, req_provider: Optional[str], req_model: Optional[str]) -> Tuple[str, str]:
    provider = (req_provider or getattr(speech, "llm_provider", None) or "openai").strip()
    model = (req_model or getattr(speech, "llm_model", None) or "gpt-4o-mini").strip()
    return provider, model


def _persist_analysis_upsert(db: Session, *, speech_id: int, merged: Dict[str, Any]) -> Analysis:
    """
    Persist analysis. DB schema may still be legacy (neutral_score). We write Centrist into:
    - Analysis.centrist_score if present, else
    - Analysis.neutral_score if present (compatibility until DB migration).
    """
    lib, auth, cen, conf, dom_fam, dom_sub, marpor_codes = _extract_scores_top(merged or {})

    dom_fam = _normalize_family(dom_fam)
    dom_sub = _normalize_subtype(dom_fam, dom_sub)

    existing = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    if existing:
        existing.ideology_family = dom_fam
        existing.ideology_subtype = dom_sub
        existing.libertarian_score = float(lib)
        existing.authoritarian_score = float(auth)

        if hasattr(existing, "centrist_score"):
            setattr(existing, "centrist_score", float(cen))
        elif hasattr(existing, "neutral_score"):
            # legacy compatibility only
            setattr(existing, "neutral_score", float(cen))

        existing.confidence_score = float(conf)
        existing.marpor_codes = list(marpor_codes or [])
        existing.full_results = merged or {}
        if hasattr(existing, "updated_at"):
            existing.updated_at = datetime.utcnow()
        db.add(existing)
        return existing

    created_kwargs: Dict[str, Any] = dict(
        speech_id=speech_id,
        ideology_family=dom_fam,
        ideology_subtype=dom_sub,
        libertarian_score=float(lib),
        authoritarian_score=float(auth),
        confidence_score=float(conf),
        marpor_codes=list(marpor_codes or []),
        full_results=merged or {},
        created_at=datetime.utcnow(),
    )

    # legacy compatibility: choose available column
    if hasattr(Analysis, "centrist_score"):
        created_kwargs["centrist_score"] = float(cen)
    elif hasattr(Analysis, "neutral_score"):
        created_kwargs["neutral_score"] = float(cen)

    created = Analysis(**created_kwargs)
    db.add(created)
    return created


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/speech")
async def analyze_speech(req: AnalyzeSpeechRequest, db: Session = Depends(get_db)):
    start = time.time()

    speech: Optional[Speech] = None

    if req.speech_id is not None:
        speech = db.query(Speech).filter(Speech.id == req.speech_id).first()
        if not speech:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speech not found.")

        text = (speech.text or "").strip()
        title = (speech.title or req.title or "Untitled Speech").strip()
        spk = (speech.speaker or req.speaker or "Unknown").strip() or "Unknown"
        provider, model = _get_speech_llm_provider_model(speech, req.llm_provider, req.llm_model)
    else:
        text = (req.text or "").strip()
        title = (req.title or "Untitled Speech").strip()
        spk = (req.speaker or "Unknown").strip() or "Unknown"
        provider = (req.llm_provider or "openai").strip()
        model = (req.llm_model or "gpt-4o-mini").strip()

    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No speech text provided.")

    # Create speech if needed
    if speech is None:
        try:
            speech = Speech(
                title=title or "Untitled Speech",
                speaker=spk,
                text=text,
                created_at=datetime.utcnow(),
            )

            if hasattr(speech, "language"):
                speech.language = getattr(speech, "language", None) or "en"
            if hasattr(speech, "word_count"):
                speech.word_count = len(text.split())
            if hasattr(speech, "status"):
                speech.status = "pending"
            if hasattr(speech, "is_public"):
                speech.is_public = False
            if hasattr(speech, "llm_provider"):
                speech.llm_provider = provider
            if hasattr(speech, "llm_model"):
                speech.llm_model = model

            if req.project_id and _PROJECT_AVAILABLE and Project is not None and hasattr(speech, "projects"):
                proj = db.query(Project).filter(Project.id == req.project_id).first()
                if proj:
                    speech.projects.append(proj)

            db.add(speech)
            db.commit()
            db.refresh(speech)
        except Exception as e:
            db.rollback()
            logger.error("Failed to create Speech: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to create speech record.")

    # Mark processing
    try:
        if hasattr(speech, "status"):
            speech.status = "processing"
        db.commit()
    except Exception:
        db.rollback()

    # Run analysis
    try:
        merged = await _run_full_analysis(
            text=text,
            title=title,
            speaker=spk,
            use_semantic=req.use_semantic,
            threshold=req.threshold,
            include_questions=req.include_questions,
            question_types=req.question_types,
            llm_provider=provider,
            llm_model=model,
            max_questions=req.max_questions,
            media_duration_seconds=req.media_duration_seconds,
        )
    except HTTPException:
        try:
            if hasattr(speech, "status"):
                speech.status = "failed"
            db.commit()
        except Exception:
            db.rollback()
        raise
    except Exception as e:
        logger.error("Analysis pipeline failed: %s", e, exc_info=True)
        try:
            if hasattr(speech, "status"):
                speech.status = "failed"
            db.commit()
        except Exception:
            db.rollback()
        raise HTTPException(status_code=500, detail="Analysis pipeline failed.")

    # Persist analysis
    try:
        analysis = _persist_analysis_upsert(db, speech_id=speech.id, merged=merged)

        if hasattr(speech, "status"):
            speech.status = "completed"
        if hasattr(speech, "analyzed_at"):
            speech.analyzed_at = datetime.utcnow()

        db.commit()
        db.refresh(analysis)

        merged["analysis_id"] = analysis.id
        merged["speech_id"] = speech.id
        merged["analysis_persisted"] = True
    except Exception as e:
        db.rollback()
        logger.error("Failed to persist Analysis: %s", e, exc_info=True)
        merged["analysis_persisted"] = False
        merged["analysis_persist_error"] = "Failed to persist analysis to database."
        merged["speech_id"] = speech.id

    return create_response(
        success=True,
        data=merged,
        message="Analysis completed.",
        processing_time=round(time.time() - start, 4),
        timestamp=_now_z(),
    )


@router.get("/speech/{speech_id}")
async def get_latest_analysis_for_speech(
    speech_id: int,
    media_duration_seconds: Optional[float] = Query(default=None, ge=0.0),
    db: Session = Depends(get_db),
):
    start = time.time()

    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speech not found.")

    analysis = db.query(Analysis).filter(Analysis.speech_id == speech_id).first()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No analysis found for this speech.")

    data = analysis.full_results or {}
    if not isinstance(data, dict):
        data = {}

    data = _normalize_analysis_shape(data, speech=speech)

    # NEW: Deep sanitize persisted payload prior to canonical extraction
    data = _purge_legacy_neutral_everywhere(data)

    # Extract normalized fields (with evidence validation)
    lib, auth, cen, conf, dom_fam, dom_sub, marpor_codes = _extract_scores_top(data)

    # Ensure canonical top-level fields (Centrist-only)
    data.update(
        {
            "analysis_id": analysis.id,
            "speech_id": speech_id,
            "ideology_family": dom_fam,
            "ideology_subtype": dom_sub,
            "libertarian_score": lib,
            "authoritarian_score": auth,
            "centrist_score": cen,
            "confidence_score": conf,
            "marpor_codes": marpor_codes,
            "scores": {LIB_FAMILY: lib, AUTH_FAMILY: auth, CENTRIST_FAMILY: cen},
        }
    )

    # Sanitize nested speech_level.scores if present (redundant but safe)
    sl = data.get("speech_level")
    if isinstance(sl, dict) and isinstance(sl.get("scores"), dict):
        sl["scores"] = _sanitize_scores_dict(sl.get("scores") or {})

    if media_duration_seconds is not None and float(media_duration_seconds) > 0:
        data = _apply_media_jump_time_support(data, duration=media_duration_seconds)

    return create_response(
        success=True,
        data=data,
        message="Latest analysis fetched.",
        processing_time=round(time.time() - start, 4),
        timestamp=_now_z(),
    )


@router.post("/speech/{speech_id}/reanalyze")
async def reanalyze_speech(speech_id: int, req: ReanalyzeRequest, db: Session = Depends(get_db)):
    start = time.time()

    speech = db.query(Speech).filter(Speech.id == speech_id).first()
    if not speech:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speech not found.")

    text = (speech.text or "").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Speech has no text to analyze.")

    title = (speech.title or "Untitled Speech").strip()
    spk = (speech.speaker or "Unknown").strip() or "Unknown"
    provider, model = _get_speech_llm_provider_model(speech, req.llm_provider, req.llm_model)

    try:
        if hasattr(speech, "status"):
            speech.status = "processing"
        db.commit()
    except Exception:
        db.rollback()

    try:
        merged = await _run_full_analysis(
            text=text,
            title=title,
            speaker=spk,
            use_semantic=req.use_semantic,
            threshold=req.threshold,
            include_questions=req.include_questions,
            question_types=req.question_types,
            llm_provider=provider,
            llm_model=model,
            max_questions=req.max_questions,
            media_duration_seconds=req.media_duration_seconds,
        )
    except HTTPException:
        try:
            if hasattr(speech, "status"):
                speech.status = "failed"
            db.commit()
        except Exception:
            db.rollback()
        raise
    except Exception as e:
        logger.error("Reanalysis pipeline failed: %s", e, exc_info=True)
        try:
            if hasattr(speech, "status"):
                speech.status = "failed"
            db.commit()
        except Exception:
            db.rollback()
        raise HTTPException(status_code=500, detail="Reanalysis pipeline failed.")

    try:
        analysis = _persist_analysis_upsert(db, speech_id=speech.id, merged=merged)

        if hasattr(speech, "status"):
            speech.status = "completed"
        if hasattr(speech, "analyzed_at"):
            speech.analyzed_at = datetime.utcnow()

        db.commit()
        db.refresh(analysis)

        merged["analysis_id"] = analysis.id
        merged["speech_id"] = speech.id
        merged["analysis_persisted"] = True
    except Exception as e:
        db.rollback()
        logger.error("Failed to persist reanalysis: %s", e, exc_info=True)
        merged["analysis_persisted"] = False
        merged["analysis_persist_error"] = "Failed to persist analysis to database."
        merged["speech_id"] = speech.id

    return create_response(
        success=True,
        data=merged,
        message="Reanalysis completed.",
        processing_time=round(time.time() - start, 4),
        timestamp=_now_z(),
    )


@router.post("/questions/generate")
async def generate_questions(req: GenerateQuestionsRequest, db: Session = Depends(get_db)):
    start = time.time()

    if not _QUESTION_GENERATOR_AVAILABLE or question_generator is None:
        raise HTTPException(status_code=503, detail="Question generation service is not available.")

    speech = db.query(Speech).filter(Speech.id == req.speech_id).first()
    if not speech:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speech not found.")

    provider = (req.llm_provider or getattr(speech, "llm_provider", None) or "openai").strip()
    model = (req.llm_model or getattr(speech, "llm_model", None) or "gpt-4o-mini").strip()

    qt = (req.question_type or "journalistic").strip().lower()
    if qt not in ("journalistic", "technical"):
        qt = "journalistic"

    ideology_result: Dict[str, Any] = {}
    key_segments: List[Dict[str, Any]] = []

    try:
        latest = db.query(Analysis).filter(Analysis.speech_id == speech.id).first()
        if latest and isinstance(latest.full_results, dict):
            fr = latest.full_results or {}
            ideology_result = fr.get("speech_level") or {}
            ks = fr.get("key_statements") or fr.get("key_segments") or fr.get("highlights") or []
            if isinstance(ks, list):
                key_segments = [x for x in ks if isinstance(x, dict)]
    except Exception:
        ideology_result = {}
        key_segments = []

    try:
        questions = await question_generator.generate_questions_with_llm(
            question_type=qt,
            speech_title=speech.title or "",
            speaker=speech.speaker or "",
            ideology_result=ideology_result or {},
            key_segments=key_segments or [],
            llm_provider=provider,
            llm_model=model,
            max_questions=int(req.max_questions),
        )
        if not isinstance(questions, list):
            questions = []
        questions = [str(q).strip() for q in questions if str(q).strip()]
    except Exception as e:
        logger.error("Questions generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Question generation failed.")

    return create_response(
        success=True,
        data={"speech_id": req.speech_id, "question_type": qt, "questions": questions},
        message="Questions generated.",
        processing_time=round(time.time() - start, 4),
        timestamp=_now_z(),
    )


@router.get("/health")
def analysis_healthcheck():
    return create_response(
        success=True,
        data={"service": "analysis", "status": "ok"},
        message="OK",
        processing_time=0.0,
        timestamp=_now_z(),
    )


__all__ = ["router"]
