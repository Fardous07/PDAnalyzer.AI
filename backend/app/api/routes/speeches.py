
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
from typing import Any, Dict, List, Optional, Tuple

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import bindparam, inspect as sa_inspect, or_, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# Pydantic compatibility (v1/v2)
try:
    from pydantic.v1 import BaseModel, Field, validator
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field, validator  # type: ignore

# DB session + models
from app.database.connection import SessionLocal, get_db
from app.database.models import Question, Speech, User

# Auth dependency (optional import)
try:
    from app.services.auth_service import get_current_user  # returns User
    _AUTH_AVAILABLE = True
except Exception:
    get_current_user = None  # type: ignore
    _AUTH_AVAILABLE = False

# Ingestion service
try:
    from app.services.speech_ingestion import ingest_speech
    _INGEST_AVAILABLE = True
except Exception:
    ingest_speech = None  # type: ignore
    _INGEST_AVAILABLE = False

# Transcription service
try:
    from app.services.transcription import transcribe_media_file
    _TRANSCRIPTION_AVAILABLE = True
except Exception:
    transcribe_media_file = None  # type: ignore
    _TRANSCRIPTION_AVAILABLE = False

# Embeddings (optional)
try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDER_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _EMBEDDER_AVAILABLE = False

# Question generator (optional)
try:
    from app.services.question_generator import question_generator
    _QUESTION_GENERATOR_AVAILABLE = True
except Exception:
    question_generator = None  # type: ignore
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
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEXT_FORMATS = {".txt", ".md"}
DOCUMENT_FORMATS = {".pdf", ".doc", ".docx"}
AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}

_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
_global_embedder: Optional[Any] = None


# =============================================================================
# FAMILY NORMALIZATION (Centrist-only non-ideological policy)
# =============================================================================

LIB_FAMILY = "Libertarian"
AUTH_FAMILY = "Authoritarian"
CENTRIST_FAMILY = "Centrist"
_ALLOWED_FAMILIES = {LIB_FAMILY, AUTH_FAMILY, CENTRIST_FAMILY}


def _normalize_family(x: Any) -> str:
    s = str(x or "").strip()
    if s in _ALLOWED_FAMILIES:
        return s
    return CENTRIST_FAMILY


def _normalize_subtype(family: str, subtype: Any) -> Optional[str]:
    """
    Centrist has no subtype.
    Lib/Auth subtype: keep if non-empty; otherwise None.
    """
    fam = _normalize_family(family)
    if fam == CENTRIST_FAMILY:
        return None
    sub = str(subtype or "").strip()
    return sub or None


def _normalize_scores(scores: Dict[str, Any]) -> Dict[str, float]:
    """
    Canonical scores:
      {Libertarian, Authoritarian, Centrist}

    Backward compatibility:
    - If upstream stored a third key with a different name, we do not rely on it.
      We compute Centrist as the remainder (100 - lib - auth), clamped to [0,100].
    """
    scores = scores or {}
    lib = float(scores.get(LIB_FAMILY, 0.0) or 0.0)
    auth = float(scores.get(AUTH_FAMILY, 0.0) or 0.0)

    # Prefer explicit Centrist if present; else compute complement.
    if CENTRIST_FAMILY in scores:
        cen = float(scores.get(CENTRIST_FAMILY, 0.0) or 0.0)
    else:
        cen = max(0.0, 100.0 - lib - auth)

    return {LIB_FAMILY: lib, AUTH_FAMILY: auth, CENTRIST_FAMILY: float(cen)}


# =============================================================================
# EMBEDDER WRAPPER (accept **kwargs safely)
# =============================================================================

class _STEmbedder:
    """Stable wrapper so embedder.encode(...) is consistent."""
    def __init__(self, model_name: str):
        if not _EMBEDDER_AVAILABLE or SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available")
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        kwargs = dict(kwargs or {})
        kwargs.pop("show_progress_bar", None)
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


# =============================================================================
# REQUEST MODELS
# =============================================================================

class SpeechCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    speaker: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=MIN_TEXT_CHARS)

    date: Optional[str] = Field(None, description="ISO date/datetime (YYYY-MM-DD or full ISO datetime)")
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

    @validator("text")
    def _text_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Text cannot be empty")
        if len(v) < MIN_TEXT_CHARS:
            raise ValueError(f"Text too short (minimum {MIN_TEXT_CHARS} characters)")
        return v


class SpeechUpdate(BaseModel):
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

    use_semantic_segmentation: Optional[bool] = None
    use_semantic_scoring: Optional[bool] = None

    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


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
    out = {
        "success": success,
        "data": data,
        "error": error,
        "message": message,
        "timestamp": _now_iso(),
    }
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
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use ISO (YYYY-MM-DD or full ISO datetime).",
        )


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
    """
    full_results may be:
    - dict / SQLAlchemy MutableDict
    - JSON string
    - legacy stringified dict
    Always return a dict.
    """
    if x is None:
        return {}

    if isinstance(x, dict):
        try:
            return dict(x)
        except Exception:
            return {}

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


# =============================================================================
# AUTH / PERMISSIONS
# =============================================================================

def _require_auth_if_enabled(user: Optional[User]) -> None:
    if _AUTH_AVAILABLE and user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


def _is_admin(user: Optional[User]) -> bool:
    return bool(user and getattr(user, "role", None) == "admin")


def _can_read(speech: Speech, user: Optional[User]) -> bool:
    if speech.is_public:
        return True
    if user is None:
        return False
    return speech.user_id == user.id or _is_admin(user)


def _require_write_access(speech: Speech, user: Optional[User]) -> None:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if speech.user_id == user.id:
        return
    if _is_admin(user):
        return
    raise HTTPException(status_code=403, detail="Not permitted")


def optional_current_user_dep() -> Any:
    if _AUTH_AVAILABLE and get_current_user is not None:
        return Depends(get_current_user)

    async def _none() -> None:
        return None

    return Depends(_none)


# =============================================================================
# DB-SCHEMA SAFE ACCESS TO "analyses" (NO ORM LOADS)
# =============================================================================

_ANALYSIS_THIRD_COL_CACHE: Optional[str] = None


def _analysis_third_score_column(db: Session) -> str:
    """
    DB may be legacy or migrated. We need the "third score" column.
    Preferred (future): centrist_score
    Fallback: any other *_score column besides libertarian_score/authoritarian_score/confidence_score.
    """
    global _ANALYSIS_THIRD_COL_CACHE
    if _ANALYSIS_THIRD_COL_CACHE:
        return _ANALYSIS_THIRD_COL_CACHE

    try:
        eng = db.get_bind()
        insp = sa_inspect(eng)
        cols = {c["name"] for c in insp.get_columns("analyses")}

        if "centrist_score" in cols:
            _ANALYSIS_THIRD_COL_CACHE = "centrist_score"
            return _ANALYSIS_THIRD_COL_CACHE

        # Heuristic fallback: find other score column (legacy schemas)
        score_cols = sorted([c for c in cols if c.endswith("_score")])
        ignore = {"libertarian_score", "authoritarian_score", "confidence_score"}
        candidates = [c for c in score_cols if c not in ignore]

        _ANALYSIS_THIRD_COL_CACHE = candidates[0] if candidates else "centrist_score"
    except Exception:
        _ANALYSIS_THIRD_COL_CACHE = "centrist_score"

    return _ANALYSIS_THIRD_COL_CACHE


def _analysis_rows_by_speech_ids(db: Session, speech_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not speech_ids:
        return {}

    third = _analysis_third_score_column(db)
    stmt = (
        text(
            f"""
            SELECT
              speech_id,
              ideology_family,
              ideology_subtype,
              libertarian_score,
              authoritarian_score,
              {third} AS third_score,
              confidence_score,
              marpor_codes,
              full_results
            FROM analyses
            WHERE speech_id IN :ids
            """
        ).bindparams(bindparam("ids", expanding=True))
    )

    rows = db.execute(stmt, {"ids": speech_ids}).fetchall()
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        sid = int(r[0])
        out[sid] = {
            "speech_id": sid,
            "ideology_family": r[1],
            "ideology_subtype": r[2],
            "libertarian_score": float(r[3] or 0.0),
            "authoritarian_score": float(r[4] or 0.0),
            "centrist_score": float(r[5] or 0.0),
            "confidence_score": float(r[6] or 0.0),
            "marpor_codes": r[7],
            "full_results": r[8],
        }
    return out


def _analysis_row_for_speech(db: Session, speech_id: int) -> Optional[Dict[str, Any]]:
    rows = _analysis_rows_by_speech_ids(db, [speech_id])
    return rows.get(speech_id)


# =============================================================================
# full_results extractors (robust) + Centrist policy
# =============================================================================

def _sanitize_scores_dict(scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure 'scores' dict never contains 'Neutral'.
    If 'Centrist' missing but 'Neutral' is present, map it to 'Centrist'.
    Ensure all canonical keys exist.
    """
    if not isinstance(scores, dict):
        return {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, CENTRIST_FAMILY: 0.0}
    out = dict(scores)
    if CENTRIST_FAMILY not in out and "Neutral" in out:
        out[CENTRIST_FAMILY] = out.get("Neutral")
    out.pop("Neutral", None)
    out.setdefault(LIB_FAMILY, 0.0)
    out.setdefault(AUTH_FAMILY, 0.0)
    out.setdefault(CENTRIST_FAMILY, 0.0)
    return out


def _purge_legacy_neutral_everywhere(obj: Any) -> Any:
    """
    Recursively:
      - Sanitize any 'scores' dict to ensure Centrist/never Neutral.
      - Normalize family/subtype keys:
          * family keys: ideology_family, dominant_family, family
          * subtype keys: ideology_subtype, dominant_subtype, subtype
        Ensure: Centrist => subtype=None.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _purge_legacy_neutral_everywhere(v)

        # Sanitize nested scores
        if isinstance(out.get("scores"), dict):
            out["scores"] = _sanitize_scores_dict(out["scores"])

        # Normalize common family/subtype fields
        family_keys = ("ideology_family", "dominant_family", "family")
        subtype_keys = ("ideology_subtype", "dominant_subtype", "subtype")
        fam_key = next((fk for fk in family_keys if fk in out), None)
        if fam_key:
            fam = _normalize_family(out.get(fam_key))
            out[fam_key] = fam
            for sk in subtype_keys:
                if sk in out:
                    out[sk] = _normalize_subtype(fam, out.get(sk))
        return out

    if isinstance(obj, list):
        return [_purge_legacy_neutral_everywhere(x) for x in obj]

    return obj


def _extract_speech_level(full_results_any: Any) -> Dict[str, Any]:
    fr = _safe_full_results(full_results_any)
    sl = fr.get("speech_level")
    return sl if isinstance(sl, dict) else {}


def _extract_scores_top(full_results_any: Any) -> Tuple[float, float, float, float, str, Optional[str], List[Any]]:
    """
    Returns: (lib, auth, centrist, conf, dominant_family, dominant_subtype, marpor_codes)

    Policy:
    - Any unknown family becomes Centrist.
    - Centrist subtype => None.
    """
    fr = _safe_full_results(full_results_any)
    speech_level = _extract_speech_level(fr)

    scores = speech_level.get("scores")
    scores = scores if isinstance(scores, dict) else {}
    scores = _normalize_scores(scores)

    lib = float(scores.get(LIB_FAMILY, 0.0) or 0.0)
    auth = float(scores.get(AUTH_FAMILY, 0.0) or 0.0)
    cen = float(scores.get(CENTRIST_FAMILY, 0.0) or 0.0)

    dominant_family = _normalize_family(
        speech_level.get("dominant_family") or fr.get("ideology_family") or CENTRIST_FAMILY
    )
    dominant_subtype = _normalize_subtype(
        dominant_family,
        speech_level.get("dominant_subtype") or fr.get("ideology_subtype"),
    )

    conf = float(speech_level.get("confidence_score", fr.get("confidence_score", 0.0)) or 0.0)

    marpor_codes = speech_level.get("marpor_codes", fr.get("marpor_codes", []))
    if isinstance(marpor_codes, list):
        marpor_codes = [x for x in marpor_codes if str(x).strip()]
    else:
        marpor_codes = []

    return lib, auth, cen, conf, dominant_family, dominant_subtype, marpor_codes


def _extract_key_statements_best(full_results_any: Any) -> List[Dict[str, Any]]:
    fr = _safe_full_results(full_results_any)
    for k in ("key_statements", "key_segments", "highlights"):
        v = fr.get(k)
        if isinstance(v, list) and v:
            return [x for x in v if isinstance(x, dict)]
    speech_level = _extract_speech_level(fr)
    v2 = speech_level.get("key_statements")
    if isinstance(v2, list) and v2:
        return [x for x in v2 if isinstance(x, dict)]
    return []


def _extract_sections_best(full_results_any: Any) -> List[Dict[str, Any]]:
    fr = _safe_full_results(full_results_any)
    for k in ("sections", "segments", "scored_segments"):
        v = fr.get(k)
        if isinstance(v, list) and v:
            return [x for x in v if isinstance(x, dict)]
    meta = fr.get("metadata")
    if isinstance(meta, dict):
        v2 = meta.get("sections")
        if isinstance(v2, list) and v2:
            return [x for x in v2 if isinstance(x, dict)]
    return []


def _analysis_summary_from_full_results(full_results_any: Any) -> Dict[str, Any]:
    lib, auth, cen, conf, dom_fam, dom_sub, marpor = _extract_scores_top(full_results_any)
    return {
        "ideology_family": _normalize_family(dom_fam),
        "ideology_subtype": dom_sub,  # None for Centrist
        "scores": {LIB_FAMILY: float(lib), AUTH_FAMILY: float(auth), CENTRIST_FAMILY: float(cen)},
        "confidence_score": float(conf),
        "marpor_codes": marpor or [],
    }


def _speech_to_dict(
    s: Speech,
    include_text: bool = False,
    include_analysis_summary: bool = True,
    analysis_row: Optional[Dict[str, Any]] = None,
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
        "has_analysis": bool(analysis_row is not None),
    }

    if include_text:
        d["text"] = s.text

    if include_analysis_summary and analysis_row is not None:
        d["analysis_summary"] = _analysis_summary_from_full_results(analysis_row.get("full_results"))

    return d


# =============================================================================
# MEDIA TIME MAPPING
# =============================================================================

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _estimate_time_from_char(start_char: Optional[int], total_chars: int, duration: float) -> Optional[float]:
    if start_char is None or total_chars <= 0 or duration <= 0:
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

    transcript = payload.get("text") or payload.get("transcript_text") or ""
    transcript_len = len(transcript) if isinstance(transcript, str) else 0
    if transcript_len <= 0:
        return payload

    for key in ("key_statements", "segments", "sections"):
        arr = payload.get(key)
        if isinstance(arr, list):
            _ensure_time_fields_for_items([x for x in arr if isinstance(x, dict)], transcript_len, dur)

    return payload


# =============================================================================
# FILE EXTRACTION
# =============================================================================

def _extract_text_from_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext in TEXT_FORMATS:
        return _clean_text(file_path.read_text(encoding="utf-8", errors="replace"))

    if ext == ".pdf":
        try:
            import PyPDF2  # type: ignore
        except Exception:
            raise HTTPException(status_code=500, detail="PDF processing requires PyPDF2 (pip install PyPDF2)")
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
            import docx  # type: ignore
        except Exception:
            raise HTTPException(status_code=500, detail="Word processing requires python-docx (pip install python-docx)")
        try:
            doc = docx.Document(str(file_path))
            out = "\n".join([p.text for p in doc.paragraphs])
            return _clean_text(out)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract Word text: {e}")

    raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")


# =============================================================================
# EMBEDDER + SESSION
# =============================================================================

def _get_embedder(use_semantic: bool = True) -> Optional[Any]:
    global _global_embedder
    if not use_semantic:
        return None
    if not _EMBEDDER_AVAILABLE or SentenceTransformer is None:
        return None
    if _global_embedder is None:
        try:
            _global_embedder = _STEmbedder(_EMBEDDER_MODEL)
            logger.info("Loaded embedder model: %s", _EMBEDDER_MODEL)
        except Exception as e:
            logger.warning("Failed to load embedder: %s", e)
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
# ANALYSIS PERSISTENCE (schema-safe, NO ORM on analyses table)
# =============================================================================

def _upsert_analysis_row(
    db: Session,
    *,
    speech_id: int,
    ideology_family: str,
    ideology_subtype: Optional[str],
    libertarian_score: float,
    authoritarian_score: float,
    centrist_score: float,
    confidence_score: float,
    marpor_codes: List[Any],
    full_results: Dict[str, Any],
    processing_time_seconds: Optional[float],
    segment_count: Optional[int],
    siu_count: Optional[int],
    key_statement_count: Optional[int],
) -> None:
    third = _analysis_third_score_column(db)
    now = datetime.utcnow()

    existing_id = db.execute(
        text("SELECT id FROM analyses WHERE speech_id = :sid"),
        {"sid": speech_id},
    ).scalar()

    fam = _normalize_family(ideology_family)
    sub = _normalize_subtype(fam, ideology_subtype)

    params = {
        "sid": speech_id,
        "fam": fam,
        "sub": (sub if sub is not None else None),
        "lib": float(libertarian_score or 0.0),
        "auth": float(authoritarian_score or 0.0),
        "cen": float(centrist_score or 0.0),
        "conf": float(confidence_score or 0.0),
        "marpor": json.dumps(list(marpor_codes or [])),
        "fr": json.dumps(full_results or {}),
        "pt": float(processing_time_seconds or 0.0) if processing_time_seconds is not None else None,
        "seg": int(segment_count or 0) if segment_count is not None else None,
        "siu": int(siu_count or 0) if siu_count is not None else None,
        "ks": int(key_statement_count or 0) if key_statement_count is not None else None,
        "c": now,
        "u": now,
    }

    if existing_id:
        db.execute(
            text(
                f"""
                UPDATE analyses
                SET ideology_family = :fam,
                    ideology_subtype = :sub,
                    libertarian_score = :lib,
                    authoritarian_score = :auth,
                    {third} = :cen,
                    confidence_score = :conf,
                    marpor_codes = :marpor,
                    full_results = :fr,
                    processing_time_seconds = :pt,
                    segment_count = :seg,
                    siu_count = :siu,
                    key_statement_count = :ks,
                    updated_at = :u
                WHERE speech_id = :sid
                """
            ),
            params,
        )
    else:
        db.execute(
            text(
                f"""
                INSERT INTO analyses (
                    speech_id, ideology_family, ideology_subtype,
                    libertarian_score, authoritarian_score, {third},
                    confidence_score, marpor_codes, full_results,
                    processing_time_seconds, segment_count, siu_count, key_statement_count,
                    created_at, updated_at
                )
                VALUES (
                    :sid, :fam, :sub,
                    :lib, :auth, :cen,
                    :conf, :marpor, :fr,
                    :pt, :seg, :siu, :ks,
                    :c, :u
                )
                """
            ),
            params,
        )


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

            t0 = time.time()
            result = await ingest_speech(
                text=speech.text,
                speech_title=speech.title or "",
                speaker=speech.speaker or "",
                use_semantic_scoring=use_sem_scoring,
                embedder=embedder,
            )
            dt = time.time() - t0

            # Sanitize full_results deeply (no Neutral anywhere; Centrist subtype=None)
            fr = result if isinstance(result, dict) else _safe_full_results(result)
            fr = _purge_legacy_neutral_everywhere(fr)

            lib, auth, cen, conf, dom_fam, dom_sub, marpor_codes = _extract_scores_top(fr)

            meta = (fr.get("metadata", {}) if isinstance(fr, dict) else {}) or {}
            if not isinstance(meta, dict):
                meta = {}

            _upsert_analysis_row(
                db,
                speech_id=speech.id,
                ideology_family=_normalize_family(dom_fam),
                ideology_subtype=_normalize_subtype(dom_fam, dom_sub),
                libertarian_score=float(lib),
                authoritarian_score=float(auth),
                centrist_score=float(cen),
                confidence_score=float(conf),
                marpor_codes=list(marpor_codes or []),
                full_results=fr if isinstance(fr, dict) else {},
                processing_time_seconds=float(dt),
                segment_count=int(meta.get("segment_count", meta.get("sentence_count", 0)) or 0),
                siu_count=int(meta.get("siu_count", 0) or 0),
                key_statement_count=int(meta.get("key_statement_count", 0) or 0),
            )

            # Optional question persistence
            if _QUESTION_GENERATOR_AVAILABLE and question_generator is not None:
                try:
                    speech_level = _extract_speech_level(fr)
                    key_statements = _extract_key_statements_best(fr)

                    qs = await question_generator.generate_questions_with_llm(
                        question_type="journalistic",
                        speech_title=speech.title or "",
                        speaker=speech.speaker or "",
                        ideology_result=speech_level,
                        key_segments=key_statements,
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
    topic: Optional[str] = Form(None),
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
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(sorted(all_formats))}",
        )

    needs_transcription = file_ext in (AUDIO_FORMATS | VIDEO_FORMATS)
    if needs_transcription and not _TRANSCRIPTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Audio/video transcription not available (configure OpenAI key).")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_filename = _safe_filename(file.filename, prefix=ts)
    file_path = UPLOAD_DIR / safe_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("File saved: %s", file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    transcript: Optional[str] = None
    media_url: Optional[str] = None

    try:
        if needs_transcription:
            if transcribe_media_file is None:
                raise HTTPException(status_code=503, detail="Transcription service not available")
            logger.info("Transcribing media: %s", file_path)
            tr = transcribe_media_file(
                str(file_path),
                language=None if language == "en" else language,
                with_timestamps=False,
            )
            transcript = _clean_text((tr or {}).get("text", "") or "")
            media_url = f"/media/uploads/{safe_filename}"
        else:
            transcript = _extract_text_from_file(file_path)
            # Keep original doc/text file on disk is optional; delete to reduce storage:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
            media_url = None

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

        background_tasks.add_task(_run_analysis_and_persist, db_speech.id)

        return {
            "success": True,
            "speech_id": db_speech.id,
            "message": "File uploaded successfully. Analysis queued.",
            "data": _speech_to_dict(db_speech, include_text=False, include_analysis_summary=False, analysis_row=None),
        }

    except SQLAlchemyError as e:
        db.rollback()
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

        if payload.analyze_immediately:
            background_tasks.add_task(_run_analysis_and_persist, speech.id)

        return _response(True, data=_speech_to_dict(speech, include_text=False, include_analysis_summary=False))

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.get("/stats")
async def stats(db: Session = Depends(get_db)):
    third = _analysis_third_score_column(db)

    total = int(db.execute(text("SELECT COUNT(*) FROM speeches")).scalar() or 0)
    analyzed = int(db.execute(text("SELECT COUNT(*) FROM speeches s JOIN analyses a ON a.speech_id = s.id")).scalar() or 0)

    # Distribution
    dist_rows = db.execute(text("SELECT ideology_family, COUNT(*) FROM analyses GROUP BY ideology_family")).fetchall()
    ideology_dist_dict: Dict[str, int] = {}
    for fam, cnt in dist_rows:
        fam_n = _normalize_family(fam)
        ideology_dist_dict[fam_n] = ideology_dist_dict.get(fam_n, 0) + int(cnt)

    # Averages
    avg_row = db.execute(
        text(
            f"""
            SELECT
              AVG(libertarian_score),
              AVG(authoritarian_score),
              AVG({third}),
              AVG(confidence_score)
            FROM analyses
            """
        )
    ).fetchone()

    avg_scores = avg_row or (0.0, 0.0, 0.0, 0.0)

    return _response(
        True,
        data={
            "total_speeches": int(total),
            "analyzed_speeches": int(analyzed),
            "pending_speeches": int(total - analyzed),
            "ideology_distribution": ideology_dist_dict,
            "average_scores": {
                "libertarian": float(avg_scores[0] or 0.0),
                "authoritarian": float(avg_scores[1] or 0.0),
                "centrist": float(avg_scores[2] or 0.0),
                "confidence": float(avg_scores[3] or 0.0),
            },
        },
    )


@router.get("/search")
async def search_speeches(
    q: str = Query(..., min_length=2),
    search_in: str = Query("all", description="title|speaker|text|all"),
    include_public: bool = Query(False, description="If true, include public speeches in results for authenticated users."),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = optional_current_user_dep(),
):
    term = f"%{q.strip()}%"
    query = db.query(Speech)

    if current_user is not None:
        if include_public:
            query = query.filter(or_(Speech.user_id == current_user.id, Speech.is_public == True))
        else:
            query = query.filter(Speech.user_id == current_user.id)
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
    analysis_map = _analysis_rows_by_speech_ids(db, [s.id for s in speeches])

    return _response(
        True,
        data={
            "speeches": [
                _speech_to_dict(s, include_text=False, include_analysis_summary=True, analysis_row=analysis_map.get(s.id))
                for s in speeches
            ],
            "count": len(speeches),
        },
    )


@router.get("/")
async def list_speeches(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    include_public: bool = Query(False, description="If true, include public speeches in results for authenticated users."),
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
        if include_public:
            q = q.filter(or_(Speech.user_id == current_user.id, Speech.is_public == True))
        else:
            q = q.filter(Speech.user_id == current_user.id)
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

    analysis_map = _analysis_rows_by_speech_ids(db, [s.id for s in rows])
    speeches = [
        _speech_to_dict(s, include_text=False, include_analysis_summary=True, analysis_row=analysis_map.get(s.id))
        for s in rows
    ]

    total_pages = (total + page_size - 1) // page_size

    return _response(
        True,
        data={
            "speeches": speeches,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        },
    )


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

    analysis_row = _analysis_row_for_speech(db, speech_id)

    return {
        "success": True,
        "data": {
            "speech_id": speech_id,
            "status": speech.status,
            "analyzed_at": speech.analyzed_at.isoformat() if speech.analyzed_at else None,
            "has_analysis": analysis_row is not None,
        },
    }


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

    return {"success": True, "data": {"media_url": speech.media_url}}


# Legacy analysis endpoint (frontend uses /api/analysis/speech/{id})
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

    analysis_row = _analysis_row_for_speech(db, speech_id)
    if not analysis_row:
        raise HTTPException(status_code=404, detail=f"Analysis not found. Speech status: {speech.status}.")

    full_results = _safe_full_results(analysis_row.get("full_results"))
    # Deep sanitize persisted payload prior to canonical extraction
    full_results = _purge_legacy_neutral_everywhere(full_results)

    lib, auth, cen, conf, dom_fam, dom_sub, marpor_codes = _extract_scores_top(full_results)

    key_statements = _extract_key_statements_best(full_results)
    sections = _extract_sections_best(full_results)

    merged = dict(full_results)
    merged.update(
        {
            "ideology_family": dom_fam,
            "ideology_subtype": dom_sub,
            "libertarian_score": lib,
            "authoritarian_score": auth,
            "centrist_score": cen,
            "confidence_score": conf,
            "marpor_codes": marpor_codes,
            "speech_id": speech_id,
            "title": speech.title,
            "speaker": speech.speaker,
            "text": speech.text,
            "transcript_text": speech.text,
            "key_statements": key_statements,
            "sections": sections,
            "segments": sections,
            "scores": {LIB_FAMILY: lib, AUTH_FAMILY: auth, CENTRIST_FAMILY: cen},
        }
    )

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

    analysis_row = _analysis_row_for_speech(db, speech_id)
    if not analysis_row:
        raise HTTPException(status_code=404, detail="Analysis not found")

    fr = _safe_full_results(analysis_row.get("full_results"))
    ks = _extract_key_statements_best(fr)

    payload = {"text": speech.text, "key_statements": ks}
    payload = _apply_media_jump_time_support(payload, duration=media_duration_seconds)

    out_ks = payload.get("key_statements") if isinstance(payload.get("key_statements"), list) else ks
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

    analysis_row = _analysis_row_for_speech(db, speech_id)
    if not analysis_row:
        raise HTTPException(status_code=404, detail="Analysis not found")

    fr = _safe_full_results(analysis_row.get("full_results"))
    sections = _extract_sections_best(fr)

    payload = {"text": speech.text, "sections": sections, "segments": sections}
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

    analysis_row = _analysis_row_for_speech(db, speech_id) if include_analysis else None
    return _response(
        True,
        data=_speech_to_dict(speech, include_text=include_text, include_analysis_summary=include_analysis, analysis_row=analysis_row),
    )


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

    analysis_row = _analysis_row_for_speech(db, speech_id)
    data = _speech_to_dict(speech, include_text=True, include_analysis_summary=True, analysis_row=analysis_row)

    if analysis_row:
        fr = _safe_full_results(analysis_row.get("full_results"))
        fr = _purge_legacy_neutral_everywhere(fr)  # sanitize before returning
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

    update = payload.dict(exclude_unset=True)
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

    msg = "Speech updated." + (" Analysis marked pending due to text change." if text_changed else "")
    analysis_row = _analysis_row_for_speech(db, speech_id)
    return _response(True, data=_speech_to_dict(speech, include_text=False, include_analysis_summary=True, analysis_row=analysis_row), message=msg)


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

    # Remove analysis row (schema-safe)
    try:
        db.execute(text("DELETE FROM analyses WHERE speech_id = :sid"), {"sid": speech_id})
    except Exception:
        db.rollback()
        # continue; speech delete below will still work

    db.query(Question).filter(Question.speech_id == speech_id).delete()
    db.delete(speech)
    db.commit()
    return _response(True, message="Speech deleted.")


@router.post("/{speech_id}/analyze")
async def analyze_existing_speech(
    speech_id: int,
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force re-analysis even if completed"),
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

    analysis_row = _analysis_row_for_speech(db, speech_id)

    if speech.status == "processing":
        return _response(True, data={"speech_id": speech_id, "status": "processing"}, message="Analysis already running.")

    if speech.status == "completed" and analysis_row is not None and not force:
        return _response(
            True,
            data=_speech_to_dict(speech, include_text=False, include_analysis_summary=True, analysis_row=analysis_row),
            message="Analysis already exists. Use force=true to re-run.",
        )

    speech.status = "pending"
    db.commit()

    background_tasks.add_task(_run_analysis_and_persist, speech_id)
    return _response(True, data={"speech_id": speech_id, "status": "queued"}, message="Analysis queued.")


__all__ = ["router"]
