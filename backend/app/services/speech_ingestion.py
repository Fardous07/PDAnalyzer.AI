"""
backend/app/services/speech_ingestion.py

SPEECH INGESTION MODULE - Discourse-Grade Ideological Statement Extraction
========================================================================

GOAL
- Do NOT treat every sentence as ideology.
- Identify sentences that express the speaker's ideological view (vs non-ideological).
- Build multi-sentence ideological STATEMENTS by grouping sequential sentences that:
  (a) share the same ideology family (Libertarian or Authoritarian), AND
  (b) share the same topic (topic continuity).
- Allow Centrist / unclassified sentences to be included as SUPPORTING sentences if:
  - they are sequential and topic-coherent with the statement, AND
  - the statement contains >= 1 strong-evidence ANCHOR sentence.

KEY POLICY
- Only 3 families exist in this module: Libertarian, Authoritarian, Centrist.
- Centrist is non-ideological: never an anchor, never the ideology-family of a statement.
- Centrist has NO subtype (always None).
- Any unknown / unrecognized family labels from the scorer are coerced to Centrist.
- Key statements are selected ONLY from ideological statements (exclude Centrist).
- Speech-level aggregation is statement-level (weighted by sentence_count and anchor_count).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

from app.services.ideology_scoring import (
    score_text,  # synchronous
    aggregate_segment_scores,
    configure_embedder,
)

# Optional spaCy sentence splitting
_SPACY_AVAILABLE = False
try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_TEXT_CHARS = 50
DEFAULT_MAX_CONCURRENT = 12
DEFAULT_CODE_THRESHOLD = 0.60

# Families (Centrist replaces any prior "non-ideological" label)
LIB_FAMILY = "Libertarian"
AUTH_FAMILY = "Authoritarian"
CENTRIST_FAMILY = "Centrist"
ALLOWED_FAMILIES = {LIB_FAMILY, AUTH_FAMILY, CENTRIST_FAMILY}

# Statement building limits
STATEMENT_MIN_SENTENCES = 1
STATEMENT_MAX_SENTENCES = 10  # allow a bit longer for discourse runs

# Anchor gating: statement must contain >=1 anchor
ANCHOR_MIN_CONF = 0.65
ANCHOR_MIN_EVIDENCE = 1

# Topic continuity thresholds (Jaccard over content-word sets)
TOPIC_SIM_CONSEC_MIN = 0.15     # lowered to catch diverse topic transitions
TOPIC_SIM_GROUP_MIN = 0.18      # lowered for mixed speeches
TOPIC_MIN_TOKENS = 3            # below this, topic similarity is unreliable

# Include centrist/weak sentences inside a statement if anchored + topic-coherent
ALLOW_WEAK_EVIDENCE_IN_GROUP = True

# Allow centrist bridging between same-family ideological sentences
ENABLE_CENTRIST_BRIDGE = True
BRIDGE_GAP_MAX_SENTENCES = 3
BRIDGE_SIM_GROUP_MIN = 0.15  # tolerant reconnect threshold for bridging

# Optional semantic topic continuity fallback (embeddings)
USE_SEMANTIC_TOPIC_SIMILARITY = True
SEMANTIC_TOPIC_SIM_MIN = 0.25  # cosine similarity threshold

# After initial extraction, merge two statements if:
# - they are sequential (optionally separated by <= BRIDGE_GAP_MAX_SENTENCES centrist sentences),
# - same ideology family,
# - same topic (statement-level topic similarity).
ENABLE_STATEMENT_MERGE_PASS = True
STATEMENT_MERGE_SIM_MIN = 0.35

# Optional attribution-risk filtering (affects anchor eligibility only)
DEFAULT_FILTER_ATTRIBUTED = True
ATTRIBUTION_HINT_RATIO_MAX = 0.50
QUOTE_EVIDENCE_RATIO_MAX = 0.60

# Key statement selection
TOP_KEY_STATEMENTS = 10
KEY_CONTEXT_SENTENCES = 2

# Statement-level weighting for aggregation
STATEMENT_WEIGHT_ANCHOR_BETA = 0.60  # weight = sentence_count * (1 + beta * anchor_count)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Sentence:
    text: str
    index: int
    start_char: int
    end_char: int


@dataclass
class Segment:
    """Segment is exactly one sentence."""
    text: str
    sentence_indices: List[int]
    start_char: int
    end_char: int
    sentences: List[Sentence]


@dataclass
class ScoredSegment:
    segment: Segment

    scores: Dict[str, float]
    ideology_family: str                 # "Libertarian" | "Authoritarian" | "Centrist"
    ideology_subtype: Optional[str]      # None for Centrist
    confidence: float

    marpor_codes: List[str]
    evidence: List[Dict[str, Any]]
    marpor_code_analysis: Dict[str, Any]

    total_strength: float
    pattern_confidence: float
    research_grade: bool

    is_ideology_evidence: bool
    signal_strength: float
    evidence_count: int

    topic_tokens: List[str]


@dataclass
class IdeologicalStatement:
    """
    Contiguous, topic-coherent ideological statement (discourse unit).
    - Contains 1..N sentences (sequential)
    - Statement has one ideology family (Libertarian or Authoritarian)
    - Must contain >=1 anchor sentence
    - May include centrist supporting sentences if topic-coherent
    """
    sentence_start: int
    sentence_end: int
    start_char: int
    end_char: int

    ideology_family: str                 # never Centrist
    ideology_subtype: Optional[str]      # dominant subtype among evidence sentences (optional)
    sentences: List[ScoredSegment]
    full_text: str

    anchor_count: int
    total_evidence: int
    marpor_codes: List[str]

    avg_confidence_evidence: float
    max_confidence: float
    avg_signal_strength: float

    sentence_count: int


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
    time_begin: Optional[float] = None
    time_end: Optional[float] = None
    quality_tier: str = "low"


# =============================================================================
# HELPERS
# =============================================================================

def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _derive_codes_from_evidence(evidence: List[Dict[str, Any]]) -> List[str]:
    codes: List[str] = []
    seen = set()
    for e in evidence or []:
        if not isinstance(e, dict):
            continue
        for k in ("code", "marpor_code", "category", "marpor"):
            v = e.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s and s not in seen:
                seen.add(s)
                codes.append(s)
    return codes


def _estimate_signal_strength(confidence_01: float, evidence_count: int) -> float:
    if evidence_count <= 0:
        return 0.0
    s = 100.0 * confidence_01 * (1.0 + 0.25 * math.log(evidence_count + 1.0))
    return float(max(0.0, min(100.0, s)))


def _quality_tier_from(conf: float, signal: float, codes_n: int, ev_n: int, anchors: int) -> str:
    c = _clamp01(conf)
    s = max(0.0, min(100.0, float(signal)))
    if anchors >= 2 and ev_n >= 3 and codes_n >= 2 and c >= 0.70 and s >= 65.0:
        return "high"
    if anchors >= 1 and ev_n >= 1 and codes_n >= 1 and c >= 0.55 and s >= 45.0:
        return "medium"
    return "low"


# --- Topic tokenization -------------------------------------------------------

_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","as","at","by","for","from","in","into",
    "of","on","to","up","with","without","over","under","again","once","here","there","all","any","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very",
    "can","will","just","don","don't","should","now","we","our","us","you","your","they","their","he","she","it",
    "this","that","these","those","is","are","was","were","be","been","being","have","has","had","do","does","did",
    "i","me","my","mine","him","his","her","hers","its","them","who","whom","which","what","why","how"
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")

def _topic_tokens(text: str) -> List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    toks = [t for t in toks if t not in _STOPWORDS and len(t) >= 3]
    return toks


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return float(inter / union) if union else 0.0


def _union_tokens(segs: List[ScoredSegment]) -> List[str]:
    u = set()
    for s in segs:
        u.update(s.topic_tokens or [])
    return list(u)


# --- Semantic topic similarity (optional) -------------------------------------

_EMBEDDER_AVAILABLE = False
_GLOBAL_TOPIC_EMBEDDER = None

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    _EMBEDDER_AVAILABLE = True
except Exception:
    _EMBEDDER_AVAILABLE = False


def _get_global_topic_embedder():
    global _GLOBAL_TOPIC_EMBEDDER
    if not _EMBEDDER_AVAILABLE or not USE_SEMANTIC_TOPIC_SIMILARITY:
        return None
    if _GLOBAL_TOPIC_EMBEDDER is None:
        try:
            _GLOBAL_TOPIC_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded semantic topic embedder (all-MiniLM-L6-v2)")
        except Exception as e:
            logger.warning("Failed to load semantic topic embedder: %s", e)
            _GLOBAL_TOPIC_EMBEDDER = None
    return _GLOBAL_TOPIC_EMBEDDER


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity using sentence embeddings (cosine)."""
    embedder = _get_global_topic_embedder()
    if not embedder or not text_a or not text_b:
        return 0.0
    try:
        emb_a = embedder.encode(text_a, convert_to_numpy=True)
        emb_b = embedder.encode(text_b, convert_to_numpy=True)
        sim = util.cos_sim(emb_a, emb_b)[0][0].item()
        sim_f = float(sim)
        return float(max(0.0, min(1.0, sim_f)))
    except Exception:
        return 0.0


# --- Attribution risk ---------------------------------------------------------

def _is_attribution_risky(evidence: List[Dict[str, Any]]) -> bool:
    ev = evidence or []
    if not ev:
        return False
    n = len(ev)
    quote_n = sum(1 for e in ev if isinstance(e, dict) and bool(e.get("inside_quotes", False)))
    attrib_n = sum(1 for e in ev if isinstance(e, dict) and bool(e.get("attribution_hint", False)))
    if n > 0 and (quote_n / n) > QUOTE_EVIDENCE_RATIO_MAX:
        return True
    if n > 0 and (attrib_n / n) > ATTRIBUTION_HINT_RATIO_MAX:
        return True
    return False


# =============================================================================
# STAGE 1: TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# =============================================================================
# STAGE 2: SENTENCE SPLITTING
# =============================================================================

class SentenceSplitter:
    def __init__(self):
        self.use_spacy = False
        self.nlp = None

        if _SPACY_AVAILABLE:
            try:
                try:
                    self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
                except Exception:
                    self.nlp = spacy.blank("en")
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
                self.use_spacy = True
            except Exception as e:
                logger.warning("SentenceSplitter: spaCy init failed, using regex. Error: %s", e)
                self.use_spacy = False
                self.nlp = None

    def split(self, text: str) -> List[Sentence]:
        if not text:
            return []
        if self.use_spacy and self.nlp:
            return self._split_spacy(text)
        return self._split_regex(text)

    def _split_spacy(self, text: str) -> List[Sentence]:
        doc = self.nlp(text)
        out: List[Sentence] = []
        idx = 0
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            out.append(Sentence(text=s, index=idx, start_char=sent.start_char, end_char=sent.end_char))
            idx += 1
        return out

    def _split_regex(self, text: str) -> List[Sentence]:
        boundary = re.compile(r"([.!?])\s+(?=(?:[\"'\(\[])?[A-Z])")
        spans: List[Tuple[int, int]] = []
        last = 0
        for m in boundary.finditer(text):
            end = m.end(1)
            spans.append((last, end))
            last = m.end()
        if last < len(text):
            spans.append((last, len(text)))

        raw: List[Sentence] = []
        for (a, b) in spans:
            s = text[a:b].strip()
            if not s:
                continue
            raw.append(Sentence(text=s, index=len(raw), start_char=a, end_char=b))

        abbreviations = {
            "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "inc.", "ltd.", "co."
        }

        merged: List[Sentence] = []
        i = 0
        out_idx = 0
        while i < len(raw):
            cur = raw[i]
            tail = cur.text.strip().lower()
            tail_last = tail.split()[-1] if tail.split() else ""
            if tail_last in abbreviations and i + 1 < len(raw):
                nxt = raw[i + 1]
                merged.append(
                    Sentence(
                        text=(cur.text + " " + nxt.text).strip(),
                        index=out_idx,
                        start_char=cur.start_char,
                        end_char=nxt.end_char,
                    )
                )
                out_idx += 1
                i += 2
            else:
                merged.append(
                    Sentence(
                        text=cur.text.strip(),
                        index=out_idx,
                        start_char=cur.start_char,
                        end_char=cur.end_char,
                    )
                )
                out_idx += 1
                i += 1
        return merged


_sentence_splitter: Optional[SentenceSplitter] = None

def get_sentence_splitter() -> SentenceSplitter:
    global _sentence_splitter
    if _sentence_splitter is None:
        _sentence_splitter = SentenceSplitter()
    return _sentence_splitter


# =============================================================================
# STAGE 3: SENTENCES -> SEGMENTS
# =============================================================================

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


# =============================================================================
# IDEOLOGY LABEL NORMALIZATION
# =============================================================================

def _normalize_family_subtype(family: str, subtype: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Normalize ideology family/subtype:
    - Only allowed families here: Libertarian, Authoritarian, Centrist.
    - Any unknown/unrecognized family is coerced to Centrist.
    - Centrist never carries a subtype (always None).
    """
    fam = (family or "").strip()
    sub = (subtype or "").strip() if subtype else ""

    if fam not in ALLOWED_FAMILIES:
        fam = CENTRIST_FAMILY

    if fam == CENTRIST_FAMILY:
        return (CENTRIST_FAMILY, None)

    return (fam, sub if sub else None)


def _is_anchor_candidate(s: ScoredSegment, filter_attributed: bool) -> bool:
    if s.ideology_family == CENTRIST_FAMILY:
        return False
    if not s.is_ideology_evidence:
        return False
    if int(s.evidence_count) < ANCHOR_MIN_EVIDENCE:
        return False
    if _clamp01(s.confidence) < ANCHOR_MIN_CONF:
        return False
    if filter_attributed and _is_attribution_risky(s.evidence):
        return False
    return True


# =============================================================================
# STAGE 4: SCORE SENTENCES
# =============================================================================

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

                raw_family = str(res.get("ideology_family", "") or "")
                raw_subtype = res.get("ideology_subtype", None)
                raw_subtype_s = str(raw_subtype).strip() if raw_subtype is not None else None

                ideology_family, ideology_subtype = _normalize_family_subtype(raw_family, raw_subtype_s)

                confidence = _clamp01(res.get("confidence_score", 0.0))
                total_strength = float(_as_float(res.get("total_strength", 0.0), 0.0))
                pattern_confidence = float(_as_float(res.get("pattern_confidence", 0.0), 0.0))
                research_grade = bool(res.get("research_grade", False))

                is_gate = bool(res.get("is_ideology_evidence", False))
                evidence_count = int(res.get("evidence_count", 0) or 0)

                # If Centrist, force non-evidence and no subtype
                if ideology_family == CENTRIST_FAMILY:
                    ideology_subtype = None
                    is_gate = False
                    evidence_count = 0

                signal_strength = res.get("signal_strength", None)
                if signal_strength is None:
                    signal_strength_f = _estimate_signal_strength(confidence, evidence_count)
                else:
                    signal_strength_f = float(_as_float(signal_strength, 0.0))
                    signal_strength_f = max(0.0, min(100.0, signal_strength_f))

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
                    pattern_confidence=pattern_confidence,
                    research_grade=research_grade,
                    is_ideology_evidence=is_gate,
                    signal_strength=signal_strength_f,
                    evidence_count=evidence_count,
                    topic_tokens=_topic_tokens(seg.text),
                )
            except Exception as e:
                logger.error("Sentence segment %s scoring failed: %s", index, e, exc_info=True)
                return None

    results = await asyncio.gather(*[_score_one(seg, i) for i, seg in enumerate(segments)], return_exceptions=True)

    out: List[ScoredSegment] = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            continue
        out.append(r)

    out.sort(key=lambda s: s.segment.sentence_indices[0] if s.segment.sentence_indices else 10**9)
    return out


# =============================================================================
# STAGE 5: BUILD IDEOLOGICAL STATEMENTS (SEQUENTIAL + SAME FAMILY + SAME TOPIC)
# =============================================================================

def build_ideological_statements(
    scored: List[ScoredSegment],
    *,
    filter_attributed: bool = DEFAULT_FILTER_ATTRIBUTED,
) -> List[IdeologicalStatement]:
    """
    Build statements as groups of sequential sentences with:
    - same ideology family (Libertarian or Authoritarian),
    - same topic (topic continuity),
    - >= 1 anchor in the group,
    - allow centrist supporting sentences inside the group if topic-coherent.

    Also merges two extracted statements if they are sequential (optionally separated by <=
    BRIDGE_GAP_MAX_SENTENCES centrist sentences), same family, and same topic.
    """
    if not scored:
        return []

    by_idx: Dict[int, ScoredSegment] = {s.segment.sentence_indices[0]: s for s in scored if s.segment.sentence_indices}

    def _dominant_subtype(evidence_segs: List[ScoredSegment]) -> Optional[str]:
        vals = [s.ideology_subtype for s in evidence_segs if s.ideology_subtype]
        if not vals:
            return None
        return Counter(vals).most_common(1)[0][0]

    def _topic_ok(prev: ScoredSegment, cur: ScoredSegment, group_sig: List[str]) -> bool:
        """
        FIXED: Use both lexical and semantic similarity.
        Lexical accepts if either:
        - consecutive Jaccard >= TOPIC_SIM_CONSEC_MIN, OR
        - current vs group signature Jaccard >= TOPIC_SIM_GROUP_MIN.
        Semantic similarity is a fallback if lexical fails.
        """
        a = prev.topic_tokens
        b = cur.topic_tokens

        # Lexical similarity
        if len(a) >= TOPIC_MIN_TOKENS and len(b) >= TOPIC_MIN_TOKENS:
            j = _jaccard(a, b)
            if j >= TOPIC_SIM_CONSEC_MIN:
                return True
            if _jaccard(b, group_sig) >= TOPIC_SIM_GROUP_MIN:
                return True

        # Semantic fallback
        if USE_SEMANTIC_TOPIC_SIMILARITY:
            semantic_sim = _semantic_similarity(prev.segment.text, cur.segment.text)
            if semantic_sim >= SEMANTIC_TOPIC_SIM_MIN:
                logger.debug("Semantic match: %.3f", semantic_sim)
                return True

        return False

    def _reconnect_ok(cur: ScoredSegment, group_sig: List[str], group_text: str) -> bool:
        """
        FIXED: Use semantic similarity for reconnection.
        """
        if len(cur.topic_tokens) < TOPIC_MIN_TOKENS:
            return False

        # Lexical check
        if len(group_sig) >= TOPIC_MIN_TOKENS:
            if _jaccard(cur.topic_tokens, group_sig) >= BRIDGE_SIM_GROUP_MIN:
                return True

        # Semantic check
        if USE_SEMANTIC_TOPIC_SIMILARITY and group_text:
            semantic_sim = _semantic_similarity(cur.segment.text, group_text)
            if semantic_sim >= SEMANTIC_TOPIC_SIM_MIN:
                logger.debug("Semantic bridge: %.3f", semantic_sim)
                return True

        return False

    def _make_statement(buf: List[ScoredSegment], family: str) -> Optional[IdeologicalStatement]:
        if not buf or family == CENTRIST_FAMILY:
            return None

        anchors = [s for s in buf if _is_anchor_candidate(s, filter_attributed) and s.ideology_family == family]
        if not anchors:
            return None

        start_char = buf[0].segment.start_char
        end_char = buf[-1].segment.end_char
        sentence_start = buf[0].segment.sentence_indices[0]
        sentence_end = buf[-1].segment.sentence_indices[0]
        full_text = " ".join(x.segment.text for x in buf).strip()

        ev = [s for s in buf if s.ideology_family == family and s.is_ideology_evidence and int(s.evidence_count) > 0]
        total_ev = sum(int(s.evidence_count) for s in ev)

        codes = set()
        for s in ev:
            for c in (s.marpor_codes or []):
                if c:
                    codes.add(c)

        if ev:
            avg_conf_ev = sum(_clamp01(s.confidence) for s in ev) / len(ev)
            avg_sig = sum(float(s.signal_strength) for s in ev) / len(ev)
        else:
            avg_conf_ev = sum(_clamp01(s.confidence) for s in buf) / len(buf)
            avg_sig = sum(float(s.signal_strength) for s in buf) / len(buf)

        max_conf = max(_clamp01(s.confidence) for s in buf)
        subtype = _dominant_subtype(ev)

        return IdeologicalStatement(
            sentence_start=sentence_start,
            sentence_end=sentence_end,
            start_char=start_char,
            end_char=end_char,
            ideology_family=family,
            ideology_subtype=subtype,
            sentences=buf[:],
            full_text=full_text,
            anchor_count=int(len(anchors)),
            total_evidence=int(total_ev),
            marpor_codes=sorted(codes),
            avg_confidence_evidence=float(avg_conf_ev),
            max_confidence=float(max_conf),
            avg_signal_strength=float(avg_sig),
            sentence_count=int(sentence_end - sentence_start + 1),
        )

    # -------------------------
    # Pass 1: extract statements
    # -------------------------
    statements: List[IdeologicalStatement] = []

    active_family: str = CENTRIST_FAMILY
    buf: List[ScoredSegment] = []
    pending_gap: List[ScoredSegment] = []

    i = 0
    while i < len(scored):
        cur = scored[i]
        cur_idx = cur.segment.sentence_indices[0]

        # If not inside a statement, only start at an anchor
        if not buf:
            pending_gap = []
            if _is_anchor_candidate(cur, filter_attributed):
                active_family = cur.ideology_family
                buf = [cur]
            i += 1
            continue

        # Enforce sequentiality across buf and (if present) pending_gap
        last_idx = (pending_gap[-1].segment.sentence_indices[0] if pending_gap else buf[-1].segment.sentence_indices[0])
        if cur_idx != last_idx + 1:
            st = _make_statement(buf, active_family)
            if st:
                statements.append(st)
            buf = []
            pending_gap = []
            active_family = CENTRIST_FAMILY
            continue  # reprocess cur

        # Length guard (buf + gap)
        if len(buf) + len(pending_gap) >= STATEMENT_MAX_SENTENCES:
            st = _make_statement(buf, active_family)
            if st:
                statements.append(st)
            buf = []
            pending_gap = []
            active_family = CENTRIST_FAMILY
            continue  # reprocess cur

        # If we see a conflicting anchor, cut boundary
        if _is_anchor_candidate(cur, filter_attributed) and cur.ideology_family != active_family:
            st = _make_statement(buf, active_family)
            if st:
                statements.append(st)
            buf = []
            pending_gap = []
            active_family = CENTRIST_FAMILY
            continue  # reprocess cur

        # If we have a pending centrist gap, attempt reconnection
        if pending_gap:
            group_sig = _union_tokens(buf)
            group_text = " ".join(s.segment.text for s in buf)

            # Reconnect only if (cur is same family or centrist-support) and matches group topic
            if cur.ideology_family in (active_family, CENTRIST_FAMILY) and _reconnect_ok(cur, group_sig, group_text):
                # merge gap into buf
                buf.extend(pending_gap)
                pending_gap = []

                # now decide if cur is includable
                if cur.ideology_family == active_family:
                    buf.append(cur)
                    i += 1
                    continue
                if cur.ideology_family == CENTRIST_FAMILY and ALLOW_WEAK_EVIDENCE_IN_GROUP:
                    buf.append(cur)
                    i += 1
                    continue

            # reconnection failed -> finalize current statement (do NOT include gap)
            st = _make_statement(buf, active_family)
            if st:
                statements.append(st)
            buf = []
            pending_gap = []
            active_family = CENTRIST_FAMILY
            continue  # reprocess cur

        # No pending gap: normal topic join logic
        last = buf[-1]
        group_sig = _union_tokens(buf)

        # If sentence is same ideological family: require topic continuity
        if cur.ideology_family == active_family:
            if _topic_ok(last, cur, group_sig):
                buf.append(cur)
                i += 1
                continue

            # topic continuity failed -> boundary
            st = _make_statement(buf, active_family)
            if st:
                statements.append(st)
            buf = []
            pending_gap = []
            active_family = CENTRIST_FAMILY
            continue  # reprocess cur

        # If sentence is centrist: allow as supporting if topic continuity holds;
        # otherwise, optionally hold as bridge if enabled.
        if cur.ideology_family == CENTRIST_FAMILY and ALLOW_WEAK_EVIDENCE_IN_GROUP:
            if _topic_ok(last, cur, group_sig):
                buf.append(cur)
                i += 1
                continue

            if ENABLE_CENTRIST_BRIDGE and len(pending_gap) < BRIDGE_GAP_MAX_SENTENCES:
                pending_gap.append(cur)
                i += 1
                continue

            st = _make_statement(buf, active_family)
            if st:
                statements.append(st)
            buf = []
            pending_gap = []
            active_family = CENTRIST_FAMILY
            continue  # reprocess cur

        # Any other family (non-centrist, non-active) that is not an anchor: conservative boundary
        st = _make_statement(buf, active_family)
        if st:
            statements.append(st)
        buf = []
        pending_gap = []
        active_family = CENTRIST_FAMILY
        continue  # reprocess cur

    # finalize remaining
    if buf:
        st = _make_statement(buf, active_family)
        if st:
            statements.append(st)

    # Apply minimum sentence constraint
    statements = [st for st in statements if (st.sentence_end - st.sentence_start + 1) >= STATEMENT_MIN_SENTENCES]

    # ---------------------------------------------
    # Pass 2: merge sequential statements if same
    # family + same topic, with <= gap centrist
    # ---------------------------------------------
    if not ENABLE_STATEMENT_MERGE_PASS or len(statements) <= 1:
        return statements

    def _statement_sig(st: IdeologicalStatement) -> List[str]:
        return _union_tokens(st.sentences)

    merged: List[IdeologicalStatement] = []
    idx = 0
    while idx < len(statements):
        cur_st = statements[idx]

        # attempt repeated merges forward
        while idx + 1 < len(statements):
            nxt = statements[idx + 1]

            if cur_st.ideology_family != nxt.ideology_family:
                break

            gap = nxt.sentence_start - cur_st.sentence_end - 1
            if gap < 0 or gap > BRIDGE_GAP_MAX_SENTENCES:
                break

            # ensure the gap sentences (if any) are Centrist
            gap_segs: List[ScoredSegment] = []
            ok_gap = True
            for j in range(cur_st.sentence_end + 1, nxt.sentence_start):
                s = by_idx.get(j)
                if s is None:
                    ok_gap = False
                    break
                if s.ideology_family != CENTRIST_FAMILY:
                    ok_gap = False
                    break
                gap_segs.append(s)
            if not ok_gap:
                break

            sig_cur = _statement_sig(cur_st)
            sig_nxt = _statement_sig(nxt)
            if len(sig_cur) < TOPIC_MIN_TOKENS or len(sig_nxt) < TOPIC_MIN_TOKENS:
                break

            sim = _jaccard(sig_cur, sig_nxt)
            if sim < STATEMENT_MERGE_SIM_MIN:
                break

            # merge them
            new_buf = cur_st.sentences + gap_segs + nxt.sentences
            new_stmt = _make_statement(new_buf, cur_st.ideology_family)
            if not new_stmt:
                break

            cur_st = new_stmt
            idx += 1  # consume nxt and continue attempting merge

        merged.append(cur_st)
        idx += 1

    return merged


# =============================================================================
# STAGE 6: KEY STATEMENTS (IDEOLOGY ONLY, EXCLUDE CENTRIST)
# =============================================================================

def _keyness_score(st: IdeologicalStatement) -> float:
    conf = _clamp01(st.avg_confidence_evidence)
    signal = max(0.0, min(100.0, st.avg_signal_strength)) / 100.0
    ev = max(0, int(st.total_evidence))
    codes = len(set(st.marpor_codes or []))
    anchors = max(1, int(st.anchor_count))
    length = max(1, int(st.sentence_count))
    return float(
        conf
        * signal
        * math.log(ev + 1.0)
        * math.log(codes + 1.0)
        * math.log(anchors + 1.0)
        * math.log(length + 1.0)
    )


def _context_text(sentences: List[Sentence], start_idx: int, end_idx: int) -> Tuple[str, str]:
    before_start = max(0, start_idx - KEY_CONTEXT_SENTENCES)
    after_end = min(len(sentences) - 1, end_idx + KEY_CONTEXT_SENTENCES)
    before = " ".join(s.text for s in sentences[before_start:start_idx]).strip()
    after = " ".join(s.text for s in sentences[end_idx + 1: after_end + 1]).strip()
    return before, after


def select_key_statements(
    statements: List[IdeologicalStatement],
    sentences: List[Sentence],
    *,
    top_n: int = TOP_KEY_STATEMENTS,
) -> List[KeyStatement]:
    if not statements:
        return []

    candidates: List[KeyStatement] = []
    for idx, st in enumerate(statements):
        if st.ideology_family == CENTRIST_FAMILY:
            continue
        if int(st.anchor_count) <= 0:
            continue
        if not st.marpor_codes:
            continue

        k = _keyness_score(st)
        before, after = _context_text(sentences, st.sentence_start, st.sentence_end)

        tier = _quality_tier_from(
            conf=_clamp01(st.avg_confidence_evidence),
            signal=float(st.avg_signal_strength),
            codes_n=len(set(st.marpor_codes or [])),
            ev_n=int(st.total_evidence),
            anchors=int(st.anchor_count),
        )

        candidates.append(
            KeyStatement(
                text=st.full_text,
                context_before=before,
                context_after=after,
                ideology_family=st.ideology_family,
                ideology_subtype=st.ideology_subtype,
                confidence=_clamp01(st.avg_confidence_evidence),
                keyness_score=k,
                marpor_codes=st.marpor_codes,
                start_char=st.start_char,
                end_char=st.end_char,
                statement_index=idx,
                sentence_range=(st.sentence_start, st.sentence_end),
                time_begin=None,
                time_end=None,
                quality_tier=tier,
            )
        )

    # NOTE: you explicitly requested to KEEP keyness sorting (no position-order change)
    candidates.sort(key=lambda x: x.keyness_score, reverse=True)
    return candidates[: max(1, int(top_n or TOP_KEY_STATEMENTS))]


# =============================================================================
# SERIALIZERS
# =============================================================================

def _serialize_key_statement(ks: KeyStatement) -> Dict[str, Any]:
    d = asdict(ks)
    d["confidence_score"] = round(float(ks.confidence), 3)
    d["marpor_code"] = (ks.marpor_codes[0] if ks.marpor_codes else None)
    return d


def _serialize_statement(st: IdeologicalStatement) -> Dict[str, Any]:
    return {
        "sentence_range": [st.sentence_start, st.sentence_end],
        "ideology_family": st.ideology_family,
        "ideology_subtype": st.ideology_subtype,
        "start_char": st.start_char,
        "end_char": st.end_char,
        "confidence_score": round(_clamp01(st.avg_confidence_evidence), 3),
        "signal_strength": round(float(st.avg_signal_strength), 2),
        "anchor_count": int(st.anchor_count),
        "sentence_count": int(st.sentence_count),
        "marpor_codes": st.marpor_codes,
        "evidence_count": int(st.total_evidence),
        "text": st.full_text,
        "full_text": st.full_text,
    }


def _serialize_scored_segment(s: ScoredSegment) -> Dict[str, Any]:
    return {
        "text": s.segment.text,
        "start_char": int(s.segment.start_char),
        "end_char": int(s.segment.end_char),
        "sentence_index": int(s.segment.sentence_indices[0]) if s.segment.sentence_indices else None,
        "ideology_family": s.ideology_family,
        "ideology_subtype": s.ideology_subtype,
        "confidence_score": round(float(_clamp01(s.confidence)), 3),
        "signal_strength": round(float(s.signal_strength), 2),
        "evidence_count": int(s.evidence_count),
        "is_ideology_evidence": bool(s.is_ideology_evidence),
        "marpor_codes": s.marpor_codes,
        "marpor_code_analysis": s.marpor_code_analysis,
        "pattern_confidence": round(float(s.pattern_confidence), 6),
        "total_strength": round(float(s.total_strength), 6),
        "research_grade": bool(s.research_grade),
        "evidence": s.evidence,
        "scores": s.scores or {},
        "topic_tokens": s.topic_tokens,
    }


def _build_sections(statements: List[IdeologicalStatement], key_statements: List[KeyStatement]) -> List[Dict[str, Any]]:
    ks_by_statement = defaultdict(list)
    for ks in key_statements:
        ks_by_statement[ks.statement_index].append(_serialize_key_statement(ks))

    sections: List[Dict[str, Any]] = []
    for i, st in enumerate(statements):
        sections.append(
            {
                "section_index": i,
                "section_name": f"Statement {i+1}",
                "ideology_family": st.ideology_family,
                "ideology_subtype": st.ideology_subtype,
                "start_char": st.start_char,
                "end_char": st.end_char,
                "sentence_range": [st.sentence_start, st.sentence_end],
                "segment_count": int(st.sentence_count),
                "confidence_score": round(_clamp01(st.avg_confidence_evidence), 3),
                "signal_strength": round(float(st.avg_signal_strength), 2),
                "anchor_count": int(st.anchor_count),
                "sentence_count": int(st.sentence_count),
                "marpor_codes": st.marpor_codes,
                "evidence_count": int(st.total_evidence),
                "text": st.full_text,
                "full_text": st.full_text,
                "text_preview": (st.full_text[:220] + "...") if len(st.full_text) > 220 else st.full_text,
                "key_statements": ks_by_statement.get(i, []),
            }
        )
    return sections


def _generate_scientific_summary(
    scored: List[ScoredSegment],
    statements: List[IdeologicalStatement],
    key_statements: List[KeyStatement],
    total_sentences: int,
) -> Dict[str, Any]:
    evidence_sentences = [
        s for s in scored
        if s.ideology_family != CENTRIST_FAMILY and s.is_ideology_evidence and int(s.evidence_count) > 0
    ]
    evidence_ratio = (len(evidence_sentences) / max(1, int(total_sentences)))

    avg_conf = 0.0
    if evidence_sentences:
        avg_conf = sum(_clamp01(s.confidence) for s in evidence_sentences) / len(evidence_sentences)

    anchor_total = sum(int(st.anchor_count) for st in statements)

    if len(statements) >= 3 and anchor_total >= 3 and evidence_ratio >= 0.04 and avg_conf >= 0.65 and len(key_statements) >= 3:
        tier = "high"
    elif len(statements) >= 1 and anchor_total >= 1 and evidence_ratio >= 0.01 and avg_conf >= 0.50 and len(key_statements) >= 1:
        tier = "medium"
    else:
        tier = "low"

    return {
        "overall_confidence": tier,
        "evidence_sentence_count": int(len(evidence_sentences)),
        "evidence_sentence_ratio": round(float(evidence_ratio), 4),
        "avg_evidence_confidence": round(float(avg_conf), 4),
        "ideological_statement_count": int(len(statements)),
        "key_statement_count": int(len(key_statements)),
        "anchor_sentence_count": int(sum(1 for s in evidence_sentences if _clamp01(s.confidence) >= ANCHOR_MIN_CONF)),
        "methodological_notes": [
            "Centrist is treated as non-ideological (no subtype).",
            "Statements are groups of sequential sentences with same ideology family and topic continuity.",
            "Statements require at least one strong-evidence anchor; centrist sentences may join only as topic-coherent support.",
            "Sequential same-family same-topic statements are merged (including up to a limited centrist gap).",
            "Key statements are selected only from ideological statements (centrist excluded).",
            "Topic continuity uses lexical similarity with optional semantic fallback.",
        ],
    }


# =============================================================================
# STATEMENT-LEVEL AGGREGATION HELPERS
# =============================================================================

def _statement_weight(st: IdeologicalStatement) -> float:
    sc = max(1, int(st.sentence_count))
    ac = max(1, int(st.anchor_count))
    return float(sc * (1.0 + STATEMENT_WEIGHT_ANCHOR_BETA * ac))


def _statement_to_segment_score(st: IdeologicalStatement) -> Dict[str, Any]:
    # Statement-level "segment" for aggregate_segment_scores()
    ev = [s for s in st.sentences if s.ideology_family == st.ideology_family and s.is_ideology_evidence and s.evidence_count > 0]
    base = ev if ev else [s for s in st.sentences if s.ideology_family == st.ideology_family]

    def _score_val(s: ScoredSegment, key: str) -> float:
        sc = s.scores or {}
        if key in sc:
            try:
                return float(sc.get(key, 0.0) or 0.0)
            except Exception:
                return 0.0
        if key == CENTRIST_FAMILY:
            # Derive from complement if scorer hasn't been updated yet
            try:
                lib = float(sc.get(LIB_FAMILY, 0.0) or 0.0)
                auth = float(sc.get(AUTH_FAMILY, 0.0) or 0.0)
                cen = 100.0 - lib - auth
                return float(max(0.0, min(100.0, cen)))
            except Exception:
                return 0.0
        return 0.0

    def _avg_score(key: str) -> float:
        if not base:
            return 0.0
        return sum(_score_val(s, key) for s in base) / len(base)

    scores = {
        LIB_FAMILY: round(_avg_score(LIB_FAMILY), 2),
        AUTH_FAMILY: round(_avg_score(AUTH_FAMILY), 2),
        CENTRIST_FAMILY: round(_avg_score(CENTRIST_FAMILY), 2),
    }

    merged_strength = defaultdict(float)
    merged_matches = defaultdict(int)
    merged_pol = defaultdict(lambda: {"support": 0, "oppose": 0, "neutral": 0})
    meta = {}

    for s in ev:
        ca = s.marpor_code_analysis or {}
        for code, b in ca.items():
            try:
                merged_strength[code] += float(b.get("evidence_strength", 0.0))
                merged_matches[code] += int(b.get("match_count", 0))
                pol = b.get("polarity", {}) or {}
                merged_pol[code]["support"] += int(pol.get("support", 0))
                merged_pol[code]["oppose"] += int(pol.get("oppose", 0))
                merged_pol[code]["neutral"] += int(pol.get("neutral", 0))
                if code not in meta:
                    meta[code] = {
                        "label": b.get("label", ""),
                        "description": b.get("description", ""),
                        "tendency": b.get("tendency", ""),
                        "weight": b.get("weight", 1.0),
                    }
            except Exception:
                continue

    total_strength = sum(merged_strength.values()) or 0.0
    final_code_analysis: Dict[str, Dict[str, Any]] = {}
    for code, strength in merged_strength.items():
        pct = (strength / total_strength * 100.0) if total_strength > 0 else 0.0
        md = meta.get(code, {})
        final_code_analysis[code] = {
            "code": code,
            "label": md.get("label", ""),
            "description": md.get("description", ""),
            "percentage": round(float(pct), 2),
            "match_count": int(merged_matches[code]),
            "tendency": md.get("tendency", ""),
            "weight": md.get("weight", 1.0),
            "polarity": merged_pol[code],
            "evidence_strength": round(float(strength), 6),
        }
    final_code_analysis = dict(sorted(final_code_analysis.items(), key=lambda kv: kv[1]["percentage"], reverse=True))

    return {
        "scores": scores,
        "ideology_family": st.ideology_family,
        "ideology_subtype": st.ideology_subtype,
        "confidence_score": float(_clamp01(st.avg_confidence_evidence)),
        "marpor_code_analysis": final_code_analysis,
        "marpor_breakdown": final_code_analysis,
        "marpor_codes": list(st.marpor_codes or []),
        "total_strength": float(total_strength),
        "pattern_confidence": 0.0,
        "research_grade": bool(st.anchor_count >= 2 and st.total_evidence >= 3 and st.avg_confidence_evidence >= 0.65),
        "is_ideology_evidence": True,
        "evidence_count": int(st.total_evidence),
        "signal_strength": float(st.avg_signal_strength),
        "sentence_count": int(st.sentence_count),
        "anchor_count": int(st.anchor_count),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def ingest_speech(
    text: str,
    speech_title: str = "",
    speaker: str = "",
    *,
    use_semantic_scoring: bool = True,
    code_threshold: float = DEFAULT_CODE_THRESHOLD,
    embedder: Optional[Any] = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    filter_attributed: bool = DEFAULT_FILTER_ATTRIBUTED,
    top_key_statements: int = TOP_KEY_STATEMENTS,
) -> Dict[str, Any]:
    clean_text = preprocess_text(text)
    if len(clean_text) < MIN_TEXT_CHARS:
        return {"error": f"Text too short (minimum {MIN_TEXT_CHARS} characters)."}

    if embedder is not None:
        configure_embedder(embedder)

    splitter = get_sentence_splitter()
    sentences = splitter.split(clean_text)
    if not sentences:
        return {"error": "No sentences detected in text."}

    sentence_segments = sentences_to_segments(sentences)

    scored_segments = await score_sentence_segments(
        sentence_segments,
        use_semantic_scoring=use_semantic_scoring,
        code_threshold=code_threshold,
        max_concurrent=max_concurrent,
    )
    if not scored_segments:
        return {"error": "Scoring produced no results. Check logs for score_text failures."}

    # Build statements: sequential + same family + same topic (with centrist support + merging)
    statements = build_ideological_statements(scored_segments, filter_attributed=filter_attributed)

    # Key statements: ideology statements only
    key_statements = select_key_statements(statements, sentences, top_n=int(top_key_statements))

    # Speech-level aggregation: statement-level weighted
    statement_results = [_statement_to_segment_score(st) for st in statements]
    statement_weights = [_statement_weight(st) for st in statements]
    speech_level = aggregate_segment_scores(statement_results, weights=statement_weights, evidence_only=True)

    sections = _build_sections(statements, key_statements)
    scientific_summary = _generate_scientific_summary(
        scored=scored_segments,
        statements=statements,
        key_statements=key_statements,
        total_sentences=len(sentences),
    )

    sentence_segments_out = [_serialize_scored_segment(s) for s in scored_segments]

    evidence_sentence_count = sum(
        1 for s in scored_segments
        if s.ideology_family != CENTRIST_FAMILY and s.is_ideology_evidence and int(s.evidence_count) > 0
    )

    return {
        "text": clean_text,
        "transcript_text": clean_text,

        "speech_level": speech_level,

        "sections": sections,
        "segments": sections,  # alias for frontend compatibility

        "key_statements": [_serialize_key_statement(k) for k in key_statements],

        "statement_list": [_serialize_statement(st) for st in statements],
        "sentence_segments": sentence_segments_out,

        "scientific_summary": scientific_summary,

        "diagnostics": {
            "filter_attributed": bool(filter_attributed),
            "scored_sentence_count": int(len(scored_segments)),
            "evidence_sentence_count": int(evidence_sentence_count),
            "ideological_statement_count": int(len(statements)),
            "top_key_statements_requested": int(top_key_statements),
            "statement_aggregation": {
                "beta": float(STATEMENT_WEIGHT_ANCHOR_BETA),
                "weights_example_first3": statement_weights[:3],
            },
            "statement_merge": {
                "enabled": bool(ENABLE_STATEMENT_MERGE_PASS),
                "merge_sim_min": float(STATEMENT_MERGE_SIM_MIN),
                "centrist_gap_max": int(BRIDGE_GAP_MAX_SENTENCES),
            },
            "topic_similarity": {
                "consec_min": float(TOPIC_SIM_CONSEC_MIN),
                "group_min": float(TOPIC_SIM_GROUP_MIN),
                "semantic_enabled": bool(USE_SEMANTIC_TOPIC_SIMILARITY),
                "semantic_min": float(SEMANTIC_TOPIC_SIM_MIN),
            },
        },
        "metadata": {
            "title": speech_title,
            "speaker": speaker,
            "sentence_count": int(len(sentences)),
            "method": "DISCOURSE_statement_sequential_same_family_same_topic_anchor_required_merge_if_same_topic",
            "code_threshold": float(code_threshold),
            "use_semantic_scoring": bool(use_semantic_scoring),
            "centrist_policy": "Centrist is non-ideological (no subtype); unknown/unrecognized labels coerced to Centrist; Centrist excluded from ideology statements/key statements",
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
