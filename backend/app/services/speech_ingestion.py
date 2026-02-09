# backend/app/services/speech_ingestion.py

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services.attribution_parser import parse_attribution
from app.services.ideology_scoring import configure_embedder, score_text
from app.services.marpor_definitions import (
    AUTH_FAMILY,
    CENTRIST_FAMILY,
    ECON_LEFT,
    ECON_RIGHT,
    LIB_FAMILY,
)

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 50
DEFAULT_MAX_CONCURRENT = 12
DEFAULT_CODE_THRESHOLD = 0.35

IDEOLOGICAL_FAMILIES = {LIB_FAMILY, AUTH_FAMILY, ECON_LEFT, ECON_RIGHT}
ALLOWED_FAMILIES = IDEOLOGICAL_FAMILIES | {CENTRIST_FAMILY}

STATEMENT_MIN_SENTENCES = 2
STATEMENT_MAX_SENTENCES = 28

ANCHOR_MIN_CONF = 0.40
ANCHOR_MIN_EVIDENCE = 1
ANCHOR_ATTRIBUTION_MIN_CONF = 0.50
ANCHOR_MIN_COMMITMENT = 0.45

TOPIC_SIM_CONSEC_MIN = 0.12
TOPIC_SIM_GROUP_MIN = 0.15
TOPIC_MIN_TOKENS = 3

ENABLE_CENTRIST_BRIDGE = True
BRIDGE_GAP_MAX_SENTENCES = 3

TOP_KEY_STATEMENTS = 5

CONFLICT_DETECTION_ENABLED = True
CROSS_STATEMENT_CONFLICT = True
CROSS_TOPIC_MIN_JACCARD = 0.14

AXIS_MIN_TOTAL = 0.06

ATTR_HINT_RATIO_THRESHOLD = 0.40

BELIEF_PATTERNS = [
    re.compile(r"\bi\s+believe\b", re.IGNORECASE),
    re.compile(r"\bi\s+support\b", re.IGNORECASE),
    re.compile(r"\bi\s+stand\s+for\b", re.IGNORECASE),
    re.compile(r"\bi\s+am\s+committed\s+to\b", re.IGNORECASE),
    re.compile(r"\bi\s+value\b", re.IGNORECASE),
    re.compile(r"\bwe\s+believe\b", re.IGNORECASE),
    re.compile(r"\bwe\s+support\b", re.IGNORECASE),
    re.compile(r"\bwe\s+stand\s+for\b", re.IGNORECASE),
]

ACTION_PATTERNS = [
    re.compile(r"\bi\s+will\b", re.IGNORECASE),
    re.compile(r"\bwe\s+will\b", re.IGNORECASE),
    re.compile(r"\bwe\s+must\b", re.IGNORECASE),
    re.compile(r"\bshould\b", re.IGNORECASE),
    re.compile(r"\bneed(?:s)?\s+to\b", re.IGNORECASE),
    re.compile(r"\bban\b", re.IGNORECASE),
    re.compile(r"\brestrict\b", re.IGNORECASE),
    re.compile(r"\bpunish\b", re.IGNORECASE),
    re.compile(r"\bprosecute\b", re.IGNORECASE),
    re.compile(r"\bcontrol\b", re.IGNORECASE),
    re.compile(r"\bregulate\b", re.IGNORECASE),
]

ISSUE_TAGS_BY_CODES: Dict[str, Set[str]] = {
    "speech_press": {"201", "203"},
    "immigration_border": {"608", "607"},
    "law_order": {"605"},
    "environment": {"501", "ENV_AUTH"},
    "economic_system": {"401", "404", "412", "413", "407", "PROT"},
    "taxation_fiscal": {"414", "503"},
    "welfare_social": {"504", "505"},
    "labor_unions": {"701", "702"},
    "education": {"507"},
    "government_structure": {"301", "302", "305"},
    "social_values": {"603", "604"},
    "national_identity": {"601"},
    "equality_social_justice": {"503", "SJ"},
    "populism_anti_elite": {"POP"},
    "healthcare_services": {"504"},
    "decentralization": {"301", "302"},
    "trade_globalization": {"407", "PROT"},
    "property_rights": {"413", "401"},
    "religion_politics": {"603"},
    "criminal_justice": {"605"},
    "privacy_autonomy": {"201", "604"},
}


def _is_speaker_belief(text: str) -> bool:
    t = (text or "").lower()
    return any(p.search(t) for p in BELIEF_PATTERNS)


def _is_action_statement(text: str) -> bool:
    t = (text or "").lower()
    return any(p.search(t) for p in ACTION_PATTERNS)


def _opposing_families(fam1: str, fam2: str) -> bool:
    return (fam1, fam2) in {
        (LIB_FAMILY, AUTH_FAMILY),
        (AUTH_FAMILY, LIB_FAMILY),
        (ECON_LEFT, ECON_RIGHT),
        (ECON_RIGHT, ECON_LEFT),
    }


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


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


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _estimate_signal_strength(confidence_01: float, evidence_count: int) -> float:
    if evidence_count <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * confidence_01 * (1.0 + 0.25 * math.log(evidence_count + 1.0))))


def _derive_codes_from_evidence(evidence: List[Dict[str, Any]]) -> List[str]:
    out, seen = [], set()
    for e in evidence or []:
        if isinstance(e, dict) and (c := e.get("code")):
            s = str(c).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _quality_tier_from(conf: float, signal: float, ev_n: int, anchors: int) -> str:
    c, s = _clamp01(conf), max(0.0, min(100.0, float(signal)))
    if anchors >= 2 and ev_n >= 3 and c >= 0.65 and s >= 60.0:
        return "high"
    if anchors >= 1 and ev_n >= 1 and c >= 0.50 and s >= 40.0:
        return "medium"
    return "low"


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "to", "up", "with", "without", "over", "under", "again", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "we", "our", "us", "you", "your", "they", "their", "he", "she", "it",
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "me", "my", "mine", "him", "his", "her", "hers", "its", "them", "who", "whom", "which", "what", "why", "how",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")


def _topic_tokens(text: str) -> List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in _STOPWORDS and len(t) >= 3]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _union_tokens(segs: List["ScoredSegment"]) -> List[str]:
    u: Set[str] = set()
    for s in segs or []:
        for t in (s.topic_tokens or []):
            u.add(t)
    return list(u)


def _has_attribution_hints(evidence: List[Dict[str, Any]]) -> bool:
    ev = [e for e in (evidence or []) if isinstance(e, dict)]
    if not ev:
        return False
    hint_n = sum(1 for e in ev if bool(e.get("attribution_hint")))
    return (hint_n / max(1, len(ev))) >= ATTR_HINT_RATIO_THRESHOLD


def _issue_tags_from_codes(codes: List[str]) -> List[str]:
    cset = {str(c).strip() for c in (codes or []) if str(c).strip()}
    out: List[str] = []
    for tag, tag_codes in ISSUE_TAGS_BY_CODES.items():
        if cset & tag_codes:
            out.append(tag)
    return out


def _segment_attribution(text: str, evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        span = None
        for ev in evidence or []:
            if isinstance(ev, dict) and isinstance(ev.get("span"), (list, tuple)) and len(ev["span"]) == 2:
                try:
                    span = (int(ev["span"][0]), int(ev["span"][1]))
                    break
                except Exception:
                    span = None
        attr = parse_attribution(text, span)
        return attr if isinstance(attr, dict) else None
    except Exception:
        return None


def _attribution_subject(attr: Optional[Dict[str, Any]]) -> str:
    if not isinstance(attr, dict):
        return "ambiguous"
    return str(attr.get("subject") or "ambiguous")


def _attribution_conf(attr: Optional[Dict[str, Any]]) -> float:
    if not isinstance(attr, dict):
        return 0.0
    return _clamp01(attr.get("confidence", 0.0))


def _attribution_commitment(attr: Optional[Dict[str, Any]]) -> float:
    if not isinstance(attr, dict):
        return 0.0
    return _clamp01(attr.get("commitment", 0.0))


def _attribution_inside_quotes(attr: Optional[Dict[str, Any]]) -> bool:
    return bool(isinstance(attr, dict) and attr.get("inside_quotes", False))


def _normalize_family_subtype(family: str, subtype: Optional[str]) -> Tuple[str, Optional[str]]:
    fam = (family or "").strip()
    if fam == "Neutral":
        fam = CENTRIST_FAMILY
    if fam not in ALLOWED_FAMILIES:
        fam = CENTRIST_FAMILY

    if fam not in (LIB_FAMILY, AUTH_FAMILY):
        return fam, None

    sub = (subtype or "").strip() if subtype else ""
    return fam, (sub if sub else None)


def _axis_labels_block() -> Dict[str, Any]:
    return {
        "x_axis": {"name": "Economic", "negative": "Left", "positive": "Right"},
        "y_axis": {"name": "Social", "negative": "Authoritarian", "positive": "Libertarian"},
    }


def _axis_directions_from_masses(
    *,
    s_lib: float,
    s_auth: float,
    e_left: float,
    e_right: float,
    axis_min_total: float = AXIS_MIN_TOTAL,
) -> Dict[str, str]:
    soc_total = max(0.0, s_lib) + max(0.0, s_auth)
    eco_total = max(0.0, e_left) + max(0.0, e_right)

    social_dir = ""
    if soc_total >= axis_min_total:
        social_dir = "Libertarian" if s_lib >= s_auth else "Authoritarian"

    economic_dir = ""
    if eco_total >= axis_min_total:
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


def _has_2d_mass(block: Dict[str, Any]) -> bool:
    axis = _as_dict(block.get("axis_strengths"))
    soc = _as_dict(axis.get("social"))
    eco = _as_dict(axis.get("economic"))

    soc_total = _as_float(
        soc.get("total"),
        _as_float(soc.get("libertarian", 0.0)) + _as_float(soc.get("authoritarian", 0.0)),
    )
    eco_total = _as_float(
        eco.get("total"),
        _as_float(eco.get("left", 0.0)) + _as_float(eco.get("right", 0.0)),
    )

    return soc_total >= AXIS_MIN_TOTAL or eco_total >= AXIS_MIN_TOTAL


def _axis_confidence(pos_mass: float, neg_mass: float) -> float:
    total = max(0.0, pos_mass) + max(0.0, neg_mass)
    if total <= 0:
        return 0.0
    dominance = max(pos_mass, neg_mass) / total
    mass_term = min(1.0, total / 2.0)
    return _clamp01(mass_term * dominance)


def _build_full_2d_from_masses(*, s_lib: float, s_auth: float, e_left: float, e_right: float) -> Dict[str, Any]:
    s_lib = max(0.0, float(s_lib))
    s_auth = max(0.0, float(s_auth))
    e_left = max(0.0, float(e_left))
    e_right = max(0.0, float(e_right))

    soc_total = s_lib + s_auth
    eco_total = e_left + e_right

    social_coord = (s_lib - s_auth) / soc_total if soc_total > 0 else 0.0
    econ_coord = (e_right - e_left) / eco_total if eco_total > 0 else 0.0

    soc_conf = _axis_confidence(s_lib, s_auth)
    eco_conf = _axis_confidence(e_left, e_right)
    overall_conf = _clamp01((soc_conf + eco_conf) / 2.0)

    magnitude = math.sqrt((social_coord ** 2) + (econ_coord ** 2))

    return {
        "axis_labels": _axis_labels_block(),
        "axis_strengths": {
            "social": {
                "libertarian": round(float(s_lib), 6),
                "authoritarian": round(float(s_auth), 6),
                "total": round(float(soc_total), 6),
            },
            "economic": {
                "left": round(float(e_left), 6),
                "right": round(float(e_right), 6),
                "total": round(float(eco_total), 6),
            },
        },
        "coordinates": {
            "social": round(float(social_coord), 3),
            "economic": round(float(econ_coord), 3),
        },
        "coordinates_xy": {
            "x": round(float(econ_coord), 3),
            "y": round(float(social_coord), 3),
        },
        "confidence_2d": {
            "social": round(float(soc_conf), 3),
            "economic": round(float(eco_conf), 3),
            "overall": round(float(overall_conf), 3),
        },
        "confidence": {
            "social": round(float(soc_conf), 3),
            "economic": round(float(eco_conf), 3),
            "overall": round(float(overall_conf), 3),
        },
        "quadrant_2d": {
            "magnitude": round(float(magnitude), 3),
            "axis_directions": _axis_directions_from_masses(
                s_lib=s_lib, s_auth=s_auth, e_left=e_left, e_right=e_right, axis_min_total=AXIS_MIN_TOTAL
            ),
        },
    }


def _looks_first_person(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\b(i|we|my|our|us)\b", t))


def _speaker_like(subj: str, is_attrib_other: bool) -> bool:
    return (subj == "speaker") or (subj == "ambiguous" and not is_attrib_other)


@dataclass
class Sentence:
    text: str
    index: int
    start_char: int
    end_char: int


@dataclass
class Segment:
    text: str
    sentence_indices: List[int]
    start_char: int
    end_char: int
    sentences: List[Sentence]


@dataclass
class ScoredSegment:
    segment: Segment
    scores: Dict[str, float]
    ideology_family: str
    ideology_subtype: Optional[str]
    confidence: float
    marpor_codes: List[str]
    evidence: List[Dict[str, Any]]
    marpor_code_analysis: Dict[str, Any]
    total_strength: float
    pattern_confidence: float
    research_grade: bool
    is_ideology_evidence: bool
    is_ideology_evidence_2d: bool
    signal_strength: float
    evidence_count: int
    topic_tokens: List[str]
    ideology_2d: Dict[str, Any]
    attribution: Optional[Dict[str, Any]] = None
    is_speaker_belief: bool = False
    is_action_statement: bool = False
    is_attributed_to_others: bool = False
    issue_tags: List[str] = field(default_factory=list)


@dataclass
class ConflictInfo:
    has_conflict: bool = False
    conflict_type: Optional[str] = None
    belief_family: Optional[str] = None
    action_family: Optional[str] = None
    description: Optional[str] = None
    warning_level: str = "none"
    related_statement_index: Optional[int] = None
    related_sentence_range: Optional[Tuple[int, int]] = None
    related_text: Optional[str] = None


@dataclass
class IdeologicalStatement:
    sentence_start: int
    sentence_end: int
    start_char: int
    end_char: int
    ideology_family: str
    ideology_subtype: Optional[str]
    sentences: List[ScoredSegment]
    full_text: str
    anchor_count: int
    total_evidence: int
    marpor_codes: List[str]
    avg_confidence_evidence: float
    max_confidence: float
    avg_signal_strength: float
    sentence_count: int
    ideology_2d: Dict[str, Any]
    conflict_info: ConflictInfo = field(default_factory=ConflictInfo)
    families_present: List[str] = field(default_factory=list)


@dataclass
class KeyStatement:
    text: str
    context_before: str
    context_after: str
    ideology_family: str
    ideology_subtype: Optional[str]
    confidence: float
    keyness_score: float
    marpor_codes: List[str]
    start_char: int
    end_char: int
    statement_index: int
    sentence_range: Tuple[int, int]
    quality_tier: str = "low"
    ideology_2d: Optional[Dict[str, Any]] = None
    signal_strength: float = 0.0
    evidence_count: int = 0
    is_key_statement: bool = True
    conflict_info: Optional[Dict[str, Any]] = None


def _is_anchor_candidate(s: ScoredSegment) -> bool:
    if s.ideology_family not in IDEOLOGICAL_FAMILIES:
        return False

    conf = _clamp01(s.confidence)
    if conf < ANCHOR_MIN_CONF:
        return False

    if not _has_2d_mass(s.ideology_2d or {}):
        return False

    if int(s.evidence_count or 0) < ANCHOR_MIN_EVIDENCE and not (s.marpor_codes or []):
        return False

    attr = s.attribution
    if _attribution_inside_quotes(attr):
        return False

    subj = _attribution_subject(attr)
    if subj in {"opponent", "third_party"}:
        return False

    speaker_ok = (subj == "speaker") or (subj == "ambiguous" and not s.is_attributed_to_others)
    if not speaker_ok:
        return False

    if subj == "speaker" and _attribution_conf(attr) >= ANCHOR_ATTRIBUTION_MIN_CONF:
        commitment_ok = _attribution_commitment(attr) >= ANCHOR_MIN_COMMITMENT
    else:
        commitment_ok = bool(s.is_action_statement or s.is_speaker_belief or _looks_first_person(s.segment.text))

    return bool(commitment_ok)


def _aggregate_2d(segs: List[ScoredSegment]) -> Dict[str, Any]:
    if not segs:
        return _empty_2d()

    soc_lib = soc_auth = eco_left = eco_right = 0.0

    for s in segs or []:
        block = _as_dict(s.ideology_2d or {})
        axis = _as_dict(block.get("axis_strengths"))
        soc = _as_dict(axis.get("social"))
        eco = _as_dict(axis.get("economic"))

        soc_lib += _as_float(soc.get("libertarian", 0.0))
        soc_auth += _as_float(soc.get("authoritarian", 0.0))
        eco_left += _as_float(eco.get("left", 0.0))
        eco_right += _as_float(eco.get("right", 0.0))

    return _build_full_2d_from_masses(s_lib=soc_lib, s_auth=soc_auth, e_left=eco_left, e_right=eco_right)


def _aggregate_speech_level_2d(statements: List[IdeologicalStatement]) -> Dict[str, Any]:
    if not statements:
        return _empty_2d()

    soc_lib = soc_auth = eco_left = eco_right = 0.0

    for st in statements or []:
        if st.ideology_family not in IDEOLOGICAL_FAMILIES:
            continue

        block = _as_dict(st.ideology_2d or {})
        axis = _as_dict(block.get("axis_strengths"))
        soc = _as_dict(axis.get("social"))
        eco = _as_dict(axis.get("economic"))

        w_conf = max(0.1, _clamp01(st.avg_confidence_evidence))
        w_ev = math.log(max(0.0, float(st.total_evidence)) + 2.0)
        weight = w_conf * w_ev

        soc_lib += _as_float(soc.get("libertarian", 0.0)) * weight
        soc_auth += _as_float(soc.get("authoritarian", 0.0)) * weight
        eco_left += _as_float(eco.get("left", 0.0)) * weight
        eco_right += _as_float(eco.get("right", 0.0)) * weight

    return _build_full_2d_from_masses(s_lib=soc_lib, s_auth=soc_auth, e_left=eco_left, e_right=eco_right)


async def score_sentence_segments(
    segments: List[Segment],
    *,
    use_semantic_scoring: bool = True,
    code_threshold: float = DEFAULT_CODE_THRESHOLD,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> List[ScoredSegment]:
    if not segments:
        return []

    semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

    async def _score_one(seg: Segment, index: int) -> Optional[ScoredSegment]:
        async with semaphore:
            try:
                res = await asyncio.to_thread(
                    score_text,
                    seg.text,
                    use_semantic=use_semantic_scoring,
                    code_threshold=code_threshold,
                )
                if not isinstance(res, dict):
                    return None

                evidence = [e for e in _as_list(res.get("evidence", [])) if isinstance(e, dict)]
                marpor_code_analysis = _as_dict(res.get("marpor_code_analysis")) or _as_dict(res.get("marpor_breakdown"))

                marpor_codes = [str(x).strip() for x in _as_list(res.get("marpor_codes", [])) if str(x).strip()]
                if not marpor_codes and evidence:
                    marpor_codes = _derive_codes_from_evidence(evidence)
                if not marpor_codes and marpor_code_analysis:
                    marpor_codes = [str(k).strip() for k in marpor_code_analysis.keys() if str(k).strip()]

                ideology_family, ideology_subtype = _normalize_family_subtype(
                    str(res.get("ideology_family", "")),
                    str(res.get("ideology_subtype", "")).strip() if res.get("ideology_subtype") else None,
                )

                confidence = _clamp01(res.get("confidence_score", 0.0))
                total_strength = _as_float(res.get("total_strength", res.get("total_evidence_strength", 0.0)))
                is_gate = bool(res.get("is_ideology_evidence", False))
                evidence_count = int(res.get("evidence_count", 0) or 0)
                is_gate_2d = bool(res.get("is_ideology_evidence_2d", False))

                if ideology_family == CENTRIST_FAMILY:
                    ideology_subtype = None
                    is_gate = False
                    is_gate_2d = False

                signal_strength = res.get("signal_strength")
                if signal_strength is None:
                    signal_strength = _estimate_signal_strength(confidence, evidence_count)
                signal_strength = max(0.0, min(100.0, _as_float(signal_strength)))

                attr = _segment_attribution(seg.text, evidence)
                subj = _attribution_subject(attr)

                has_attr_hints = _has_attribution_hints(evidence)
                fp = _looks_first_person(seg.text)

                is_attrib_other = (
                    subj in {"opponent", "third_party"}
                    or _attribution_inside_quotes(attr)
                    or (has_attr_hints and not fp)
                )

                speaker_like = _speaker_like(subj, is_attrib_other)

                is_belief = _is_speaker_belief(seg.text) and speaker_like
                is_action = _is_action_statement(seg.text) and speaker_like

                if (subj == "speaker") and _attribution_commitment(attr) >= 0.80 and not is_action:
                    is_action = True

                ide2d = res.get("ideology_2d") or {}
                if not isinstance(ide2d, dict):
                    ide2d = {}

                issue_tags = _issue_tags_from_codes(marpor_codes)

                return ScoredSegment(
                    segment=seg,
                    scores=res.get("scores", {}) or {},
                    ideology_family=ideology_family,
                    ideology_subtype=ideology_subtype,
                    confidence=confidence,
                    marpor_codes=marpor_codes,
                    evidence=evidence,
                    marpor_code_analysis=marpor_code_analysis,
                    total_strength=total_strength,
                    pattern_confidence=_as_float(res.get("pattern_confidence", 0.0)),
                    research_grade=bool(res.get("research_grade", False)),
                    is_ideology_evidence=is_gate,
                    is_ideology_evidence_2d=is_gate_2d,
                    signal_strength=signal_strength,
                    evidence_count=evidence_count,
                    topic_tokens=_topic_tokens(seg.text),
                    ideology_2d=ide2d,
                    attribution=attr,
                    is_speaker_belief=is_belief,
                    is_action_statement=is_action,
                    is_attributed_to_others=is_attrib_other,
                    issue_tags=issue_tags,
                )
            except Exception as e:
                logger.error("Segment %s scoring failed: %s", index, e, exc_info=True)
                return None

    results = await asyncio.gather(*[_score_one(seg, i) for i, seg in enumerate(segments)], return_exceptions=True)
    out = [r for r in results if isinstance(r, ScoredSegment)]
    out.sort(key=lambda s: s.segment.sentence_indices[0] if s.segment.sentence_indices else 10**9)
    return out


def _collect_cross_time_contradictions(statements: List[IdeologicalStatement]) -> List[Dict[str, Any]]:
    if not CROSS_STATEMENT_CONFLICT or not CONFLICT_DETECTION_ENABLED:
        return []
    if not statements or len(statements) < 2:
        return []

    belief_events: List[Dict[str, Any]] = []
    action_events: List[Dict[str, Any]] = []

    for si, st in enumerate(statements):
        st_union_tokens = set(_union_tokens(st.sentences))
        for seg in st.sentences:
            if seg.ideology_family not in IDEOLOGICAL_FAMILIES:
                continue
            if seg.is_attributed_to_others:
                continue
            if not _has_2d_mass(seg.ideology_2d or {}):
                continue

            ev = {
                "statement_index": si,
                "sentence_index": seg.segment.sentence_indices[0] if seg.segment.sentence_indices else -1,
                "family": seg.ideology_family,
                "topic_tokens": list(set(seg.topic_tokens or []) | st_union_tokens),
                "issue_tags": list(set(seg.issue_tags or [])),
                "text": seg.segment.text,
                "start_char": seg.segment.start_char,
                "end_char": seg.segment.end_char,
                "confidence": float(_clamp01(seg.confidence)),
            }

            if seg.is_speaker_belief and (seg.is_ideology_evidence or seg.is_ideology_evidence_2d):
                belief_events.append(ev)
            if seg.is_action_statement:
                action_events.append(ev)

    if not belief_events or not action_events:
        return []

    flags: List[Dict[str, Any]] = []
    used_pairs: Set[Tuple[int, int]] = set()

    def _topic_match_score(b: Dict[str, Any], a: Dict[str, Any]) -> float:
        jac = _jaccard(b.get("topic_tokens", []), a.get("topic_tokens", []))
        tag_overlap = bool(set(b.get("issue_tags", [])) & set(a.get("issue_tags", [])))
        return max(float(jac), 0.25 if tag_overlap else 0.0)

    for b in belief_events:
        best = None
        best_score = 0.0

        for a in action_events:
            if a["sentence_index"] <= b["sentence_index"]:
                continue
            if not _opposing_families(b["family"], a["family"]):
                continue

            score = _topic_match_score(b, a)
            if score <= 0:
                continue

            if score < CROSS_TOPIC_MIN_JACCARD and not (set(b.get("issue_tags", [])) & set(a.get("issue_tags", []))):
                continue

            if score > best_score:
                best_score = score
                best = a

        if best is None:
            continue

        pair = (b["statement_index"], best["statement_index"])
        if pair in used_pairs:
            continue
        used_pairs.add(pair)

        belief_stmt = statements[b["statement_index"]]
        action_stmt = statements[best["statement_index"]]

        conf = _clamp01(0.50 * best_score + 0.25 * b["confidence"] + 0.25 * best["confidence"])

        flags.append(
            {
                "type": "cross_time_contradiction",
                "issue_tags": sorted(set(b.get("issue_tags", [])) | set(best.get("issue_tags", []))),
                "belief": {
                    "statement_index": b["statement_index"],
                    "sentence_range": [belief_stmt.sentence_start, belief_stmt.sentence_end],
                    "start_char": belief_stmt.start_char,
                    "end_char": belief_stmt.end_char,
                    "text": belief_stmt.full_text,
                    "ideology_family": belief_stmt.ideology_family,
                },
                "action": {
                    "statement_index": best["statement_index"],
                    "sentence_range": [action_stmt.sentence_start, action_stmt.sentence_end],
                    "start_char": action_stmt.start_char,
                    "end_char": action_stmt.end_char,
                    "text": action_stmt.full_text,
                    "ideology_family": action_stmt.ideology_family,
                },
                "severity": "warning",
                "confidence": round(float(conf), 3),
                "description": f"Earlier {b['family']} belief appears contradicted by later {best['family']} action on a related topic.",
            }
        )

    return flags


def _attach_cross_statement_conflicts(statements: List[IdeologicalStatement]) -> None:
    flags = _collect_cross_time_contradictions(statements)
    if not flags:
        return

    by_belief: Dict[int, Dict[str, Any]] = {}
    for f in flags:
        bi = _as_int(_as_dict(f.get("belief")).get("statement_index"), -1)
        if bi >= 0 and bi not in by_belief:
            by_belief[bi] = f

    for bi, f in by_belief.items():
        if bi < 0 or bi >= len(statements):
            continue
        st = statements[bi]
        if st.conflict_info.has_conflict:
            continue

        belief = _as_dict(f.get("belief"))
        action = _as_dict(f.get("action"))

        a_si_int = _as_int(action.get("statement_index"), -1)
        related_stmt_idx: Optional[int] = a_si_int if a_si_int >= 0 else None

        sr = action.get("sentence_range")
        related_sr: Optional[Tuple[int, int]] = None
        if isinstance(sr, list) and len(sr) == 2:
            related_sr = (_as_int(sr[0], 0), _as_int(sr[1], 0))

        a_txt = str(action.get("text") or "")
        st.conflict_info = ConflictInfo(
            has_conflict=True,
            conflict_type="cross_statement_rhetoric_action",
            belief_family=str(belief.get("ideology_family") or "") or None,
            action_family=str(action.get("ideology_family") or "") or None,
            description=str(f.get("description") or ""),
            warning_level=str(f.get("severity") or "warning"),
            related_statement_index=related_stmt_idx,
            related_sentence_range=related_sr,
            related_text=(a_txt[:240] + ("..." if len(a_txt) > 240 else "")) if a_txt else None,
        )


def build_ideological_statements(scored: List[ScoredSegment]) -> List[IdeologicalStatement]:
    if not scored:
        return []

    scored = sorted(scored, key=lambda s: s.segment.sentence_indices[0] if s.segment.sentence_indices else 10**9)

    def _dominant_family(segs: List[ScoredSegment]) -> str:
        weights = defaultdict(float)
        for s in segs:
            if s.ideology_family not in IDEOLOGICAL_FAMILIES:
                continue
            if not (s.is_ideology_evidence or s.is_ideology_evidence_2d):
                continue
            if not _has_2d_mass(s.ideology_2d or {}):
                continue
            if s.is_attributed_to_others:
                continue

            base = max(s.total_strength, 1.0) * (0.5 + 0.5 * _clamp01(s.confidence))
            weights[s.ideology_family] += base

        return max(weights.items(), key=lambda x: x[1])[0] if weights else CENTRIST_FAMILY

    def _dominant_subtype(segs: List[ScoredSegment], family: str) -> Optional[str]:
        if family not in (LIB_FAMILY, AUTH_FAMILY):
            return None
        vals = [
            s.ideology_subtype
            for s in segs
            if s.ideology_family == family and s.ideology_subtype and not s.is_attributed_to_others
        ]
        return Counter(vals).most_common(1)[0][0] if vals else None

    def _topic_coherent(cur: ScoredSegment, prev: ScoredSegment, group_sig: List[str]) -> bool:
        a, b = cur.topic_tokens or [], prev.topic_tokens or []
        if len(a) < TOPIC_MIN_TOKENS or len(b) < TOPIC_MIN_TOKENS:
            return True

        if cur.ideology_family == prev.ideology_family and cur.ideology_family in IDEOLOGICAL_FAMILIES:
            return True

        if set(cur.marpor_codes or []) & set(prev.marpor_codes or []):
            return True

        if _jaccard(a, b) >= TOPIC_SIM_CONSEC_MIN:
            return True
        if group_sig and _jaccard(a, group_sig) >= TOPIC_SIM_GROUP_MIN:
            return True
        return False

    def _make_statement(buf: List[ScoredSegment]) -> Optional[IdeologicalStatement]:
        if not buf or len(buf) < STATEMENT_MIN_SENTENCES:
            return None

        anchors = [s for s in buf if _is_anchor_candidate(s)]
        if not anchors:
            return None

        dom_family = _dominant_family(buf)
        if dom_family not in IDEOLOGICAL_FAMILIES:
            return None

        ev_segs = [
            s for s in buf
            if s.ideology_family in IDEOLOGICAL_FAMILIES
            and (s.is_ideology_evidence or s.is_ideology_evidence_2d)
            and _has_2d_mass(s.ideology_2d or {})
            and not s.is_attributed_to_others
        ]
        total_ev = sum(max(0, int(s.evidence_count or 0)) for s in ev_segs)

        codes: Set[str] = set()
        for s in ev_segs:
            for c in (s.marpor_codes or []):
                if c:
                    codes.add(str(c))

        avg_conf_ev = (sum(_clamp01(s.confidence) for s in ev_segs) / len(ev_segs)) if ev_segs else 0.0
        avg_sig = (sum(s.signal_strength for s in ev_segs) / len(ev_segs)) if ev_segs else 0.0

        ideology_2d = _aggregate_2d(ev_segs if ev_segs else anchors)

        fams_present = sorted({s.ideology_family for s in ev_segs if s.ideology_family in IDEOLOGICAL_FAMILIES})

        return IdeologicalStatement(
            sentence_start=buf[0].segment.sentence_indices[0],
            sentence_end=buf[-1].segment.sentence_indices[0],
            start_char=buf[0].segment.start_char,
            end_char=buf[-1].segment.end_char,
            ideology_family=dom_family,
            ideology_subtype=_dominant_subtype(ev_segs, dom_family),
            sentences=buf[:],
            full_text=" ".join(s.segment.text for s in buf).strip(),
            anchor_count=len(anchors),
            total_evidence=int(total_ev),
            marpor_codes=sorted(codes),
            avg_confidence_evidence=float(avg_conf_ev),
            max_confidence=max(_clamp01(s.confidence) for s in buf),
            avg_signal_strength=float(avg_sig),
            sentence_count=len(buf),
            ideology_2d=ideology_2d,
            conflict_info=ConflictInfo(),
            families_present=fams_present,
        )

    statements: List[IdeologicalStatement] = []
    buf: List[ScoredSegment] = []
    group_sig: List[str] = []
    gap = 0
    last_idx: Optional[int] = None

    def _flush() -> None:
        nonlocal buf, group_sig, gap
        st = _make_statement(buf)
        if st:
            statements.append(st)
        buf = []
        group_sig = []
        gap = 0

    for cur in scored:
        if not cur.segment.sentence_indices:
            continue
        cur_idx = cur.segment.sentence_indices[0]

        if last_idx is not None and cur_idx != last_idx + 1:
            _flush()
        last_idx = cur_idx

        if not buf:
            if cur.ideology_family in IDEOLOGICAL_FAMILIES or cur.is_speaker_belief or cur.is_action_statement:
                buf = [cur]
                group_sig = list(set(cur.topic_tokens or []))
                gap = 0
            continue

        prev = buf[-1]
        if not _topic_coherent(cur, prev, group_sig):
            _flush()
            if cur.ideology_family in IDEOLOGICAL_FAMILIES or cur.is_speaker_belief or cur.is_action_statement:
                buf = [cur]
                group_sig = list(set(cur.topic_tokens or []))
                gap = 0
            continue

        if cur.ideology_family == CENTRIST_FAMILY:
            if ENABLE_CENTRIST_BRIDGE and gap < BRIDGE_GAP_MAX_SENTENCES:
                buf.append(cur)
                gap += 1
            else:
                _flush()
            continue

        buf.append(cur)
        group_sig = list(set(group_sig) | set(cur.topic_tokens or []))
        gap = 0

        if len(buf) >= STATEMENT_MAX_SENTENCES:
            _flush()

    if buf:
        _flush()

    _attach_cross_statement_conflicts(statements)

    if not statements:
        for seg in scored:
            if _is_anchor_candidate(seg):
                ev_n = max(1, int(seg.evidence_count or 0))
                codes = seg.marpor_codes or _derive_codes_from_evidence(seg.evidence)
                ideology_2d = _aggregate_2d([seg])
                statements.append(
                    IdeologicalStatement(
                        sentence_start=seg.segment.sentence_indices[0],
                        sentence_end=seg.segment.sentence_indices[0],
                        start_char=seg.segment.start_char,
                        end_char=seg.segment.end_char,
                        ideology_family=seg.ideology_family,
                        ideology_subtype=seg.ideology_subtype if seg.ideology_family in (LIB_FAMILY, AUTH_FAMILY) else None,
                        sentences=[seg],
                        full_text=seg.segment.text.strip(),
                        anchor_count=1,
                        total_evidence=ev_n,
                        marpor_codes=codes,
                        avg_confidence_evidence=float(_clamp01(seg.confidence)),
                        max_confidence=float(_clamp01(seg.confidence)),
                        avg_signal_strength=float(seg.signal_strength),
                        sentence_count=1,
                        ideology_2d=ideology_2d,
                        conflict_info=ConflictInfo(),
                        families_present=[seg.ideology_family],
                    )
                )

    return statements


def _keyness_score(st: IdeologicalStatement) -> float:
    conf = _clamp01(st.avg_confidence_evidence)
    sig = max(0.0, min(100.0, st.avg_signal_strength)) / 100.0
    ev = max(0, st.total_evidence)
    codes = len(set(st.marpor_codes or []))
    anchors = max(1, st.anchor_count)
    length = max(1, st.sentence_count)
    return conf * sig * math.log(ev + 1.0) * math.log(codes + 2.0) * math.log(anchors + 1.0) * math.log(length + 1.0)


def select_key_statements(statements: List[IdeologicalStatement], top_n: int = TOP_KEY_STATEMENTS) -> List[KeyStatement]:
    if not statements:
        return []

    candidates: List[KeyStatement] = []
    for idx, st in enumerate(statements):
        if st.ideology_family not in IDEOLOGICAL_FAMILIES:
            continue
        if st.anchor_count <= 0 or st.total_evidence <= 0:
            continue

        k = _keyness_score(st)
        tier = _quality_tier_from(
            _clamp01(st.avg_confidence_evidence),
            st.avg_signal_strength,
            st.total_evidence,
            st.anchor_count,
        )

        candidates.append(
            KeyStatement(
                text=st.full_text,
                context_before="",
                context_after="",
                ideology_family=st.ideology_family,
                ideology_subtype=st.ideology_subtype,
                confidence=_clamp01(st.avg_confidence_evidence),
                keyness_score=k,
                marpor_codes=st.marpor_codes,
                start_char=st.start_char,
                end_char=st.end_char,
                statement_index=idx,
                sentence_range=(st.sentence_start, st.sentence_end),
                quality_tier=tier,
                ideology_2d=st.ideology_2d,
                signal_strength=st.avg_signal_strength,
                evidence_count=st.total_evidence,
                is_key_statement=True,
                conflict_info=None,
            )
        )

    candidates.sort(key=lambda x: x.keyness_score, reverse=True)
    return candidates[: max(1, int(top_n))]


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


class SentenceSplitter:
    def split(self, text: str) -> List[Sentence]:
        if not text:
            return []

        sentences: List[Sentence] = []
        current = ""
        char_pos = 0

        for i, ch in enumerate(text):
            current += ch
            if ch in ".!?" and i < len(text) - 1 and text[i + 1] in " \n":
                s = current.strip()
                if s:
                    sentences.append(
                        Sentence(
                            text=s,
                            index=len(sentences),
                            start_char=char_pos,
                            end_char=char_pos + len(current),
                        )
                    )
                char_pos += len(current)
                current = ""

        if current.strip():
            sentences.append(
                Sentence(
                    text=current.strip(),
                    index=len(sentences),
                    start_char=char_pos,
                    end_char=char_pos + len(current),
                )
            )

        return sentences


_sentence_splitter: Optional[SentenceSplitter] = None


def get_sentence_splitter() -> SentenceSplitter:
    global _sentence_splitter
    if _sentence_splitter is None:
        _sentence_splitter = SentenceSplitter()
    return _sentence_splitter


def sentences_to_segments(sentences: List[Sentence]) -> List[Segment]:
    return [
        Segment(
            text=s.text,
            sentence_indices=[s.index],
            start_char=s.start_char,
            end_char=s.end_char,
            sentences=[s],
        )
        for s in sentences
    ]


def _context_window(sentences: List[Sentence], start: int, end: int, n: int = 2) -> Tuple[str, str]:
    lo = max(0, int(start) - int(n))
    hi = min(len(sentences), int(end) + 1 + int(n))
    before = " ".join(s.text for s in sentences[lo:int(start)]).strip()
    after = " ".join(s.text for s in sentences[int(end) + 1:hi]).strip()
    return before, after


async def ingest_speech(
    text: str,
    speech_title: str = "",
    speaker: str = "",
    *,
    use_semantic_scoring: bool = True,
    code_threshold: float = DEFAULT_CODE_THRESHOLD,
    embedder: Optional[Any] = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    top_key_statements: int = TOP_KEY_STATEMENTS,
) -> Dict[str, Any]:
    clean_text = preprocess_text(text)
    if len(clean_text) < MIN_TEXT_CHARS:
        return {"error": f"Text too short (minimum {MIN_TEXT_CHARS} characters)."}

    if embedder is not None:
        configure_embedder(embedder)

    sentences = get_sentence_splitter().split(clean_text)
    if not sentences:
        return {"error": "No sentences detected in text."}

    scored_segments = await score_sentence_segments(
        sentences_to_segments(sentences),
        use_semantic_scoring=use_semantic_scoring,
        code_threshold=code_threshold,
        max_concurrent=max_concurrent,
    )
    if not scored_segments:
        return {"error": "Scoring produced no results."}

    statements = build_ideological_statements(scored_segments)
    key_statements = select_key_statements(statements, top_n=top_key_statements)

    speech_level_ideology_2d = _aggregate_speech_level_2d(statements)
    contradiction_flags = _collect_cross_time_contradictions(statements)

    statement_list_out: List[Dict[str, Any]] = []
    for st in statements:
        statement_list_out.append(
            {
                "sentence_range": [st.sentence_start, st.sentence_end],
                "ideology_family": st.ideology_family,
                "ideology_subtype": st.ideology_subtype,
                "start_char": st.start_char,
                "end_char": st.end_char,
                "confidence_score": round(_clamp01(st.avg_confidence_evidence), 3),
                "signal_strength": round(st.avg_signal_strength, 2),
                "anchor_count": st.anchor_count,
                "sentence_count": st.sentence_count,
                "marpor_codes": st.marpor_codes,
                "evidence_count": st.total_evidence,
                "ideology_2d": st.ideology_2d,
                "is_key_statement": False,
                "text": st.full_text,
                "full_text": st.full_text,
                "families_present": st.families_present,
                "conflict_info": (
                    {
                        "has_conflict": st.conflict_info.has_conflict,
                        "conflict_type": st.conflict_info.conflict_type,
                        "description": st.conflict_info.description,
                        "warning_level": st.conflict_info.warning_level,
                        "related_statement_index": st.conflict_info.related_statement_index,
                        "related_sentence_range": list(st.conflict_info.related_sentence_range)
                        if st.conflict_info.related_sentence_range
                        else None,
                        "related_text": st.conflict_info.related_text,
                    }
                    if st.conflict_info.has_conflict
                    else None
                ),
            }
        )

    key_statements_out: List[Dict[str, Any]] = []
    for ks in key_statements:
        before, after = _context_window(sentences, ks.sentence_range[0], ks.sentence_range[1], n=2)
        key_statements_out.append(
            {
                "text": ks.text,
                "context_before": before,
                "context_after": after,
                "ideology_family": ks.ideology_family,
                "ideology_subtype": ks.ideology_subtype,
                "confidence": round(ks.confidence, 3),
                "keyness_score": round(ks.keyness_score, 3),
                "marpor_codes": ks.marpor_codes,
                "start_char": ks.start_char,
                "end_char": ks.end_char,
                "statement_index": ks.statement_index,
                "sentence_range": [ks.sentence_range[0], ks.sentence_range[1]],
                "quality_tier": ks.quality_tier,
                "signal_strength": round(ks.signal_strength, 2),
                "evidence_count": ks.evidence_count,
                "ideology_2d": ks.ideology_2d,
                "conflict_info": None,
            }
        )

    centrist_sentences = sum(1 for s in scored_segments if s.ideology_family == CENTRIST_FAMILY)
    ideological_sentences = sum(1 for s in scored_segments if s.ideology_family in IDEOLOGICAL_FAMILIES)
    anchor_sentences = sum(1 for s in scored_segments if _is_anchor_candidate(s))
    conflict_count = sum(1 for st in statements if st.conflict_info.has_conflict)

    return {
        "text": clean_text,
        "ideology_2d": speech_level_ideology_2d,
        "speech_level": {
            "total_sentences": len(scored_segments),
            "ideological_sentences": ideological_sentences,
            "centrist_sentences": centrist_sentences,
            "anchor_sentences": anchor_sentences,
            "statement_count": len(statements),
            "key_statement_count": len(key_statements),
            "conflict_count": conflict_count,
            "contradiction_flag_count": int(len(contradiction_flags)),
            "ideology_2d": speech_level_ideology_2d,
        },
        "statements": statement_list_out,
        "key_statements": key_statements_out,
        "contradiction_flags": contradiction_flags,
        "metadata": {
            "title": speech_title,
            "speaker": speaker,
            "method": "discourse_sentence_scoring_anchor_based",
            "constraints": {
                "statements_min_sentences": STATEMENT_MIN_SENTENCES,
                "anchor_required": True,
                "anchor_min_confidence": ANCHOR_MIN_CONF,
                "anchor_speaker_owned": True,
                "centrist_as_bridge": ENABLE_CENTRIST_BRIDGE,
                "cross_statement_conflict_detection": CROSS_STATEMENT_CONFLICT,
                "centrist_is_non_ideology": True,
                "axis_min_total": AXIS_MIN_TOTAL,
                "attr_hint_ratio_threshold": ATTR_HINT_RATIO_THRESHOLD,
            },
        },
    }


__all__ = [
    "ingest_speech",
    "preprocess_text",
    "get_sentence_splitter",
    "sentences_to_segments",
    "score_sentence_segments",
    "build_ideological_statements",
    "select_key_statements",
]