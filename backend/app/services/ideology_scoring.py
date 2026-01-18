# backend/app/services/ideology_scoring.py
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from app.services.marpor_definitions import hybrid_marpor_analyzer, IDEOLOGY_SUBTYPES

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CODE_THRESHOLD = 0.60

# Lexical-first, semantic fallback only if lexical gate fails
DEFAULT_USE_SEMANTIC = True
SEMANTIC_FALLBACK_ENABLED = True

# Research-grade heuristic (segment-level)
RESEARCH_GRADE_MIN_CONF = 0.65
RESEARCH_GRADE_MIN_EVIDENCE = 2
RESEARCH_GRADE_MIN_PATTERN = 0.50

# Families (Neutral removed; Centrist is non-ideological family label)
LIB_FAMILY = "Libertarian"
AUTH_FAMILY = "Authoritarian"
CENTRIST_FAMILY = "Centrist"


# =============================================================================
# AGGREGATION-LEVEL CONFIDENCE
# =============================================================================

class AggregationConfidence:
    """Scientific confidence calculation for aggregating multiple evidence pieces."""

    @staticmethod
    def calculate_aggregated_confidence(
        evidence_counts: List[int],
        confidence_scores: List[float],
        strengths: List[float],
        family_consistency: float,
    ) -> Dict[str, Any]:
        if not evidence_counts or not confidence_scores:
            return {
                "aggregation_confidence": 0.0,
                "evidence_sufficiency": 0.0,
                "consistency_score": 0.0,
                "statistical_significance": "insufficient",
            }

        total_evidence = sum(int(x) for x in evidence_counts)

        # Evidence sufficiency (scaled; not a gate)
        evidence_sufficiency = min(1.0, total_evidence / 6.0)

        avg_confidence = sum(float(x) for x in confidence_scores) / max(1, len(confidence_scores))
        avg_strength = sum(float(x) for x in strengths) / max(1, len(strengths))

        evidence_density = total_evidence / max(1, len(evidence_counts))
        statistical_power = 1.0 - math.exp(-evidence_density / 2.0)

        aggregation_confidence = (
            avg_confidence * 0.45
            + evidence_sufficiency * 0.25
            + float(family_consistency) * 0.20
            + statistical_power * 0.10
        )

        if total_evidence >= 5 and avg_confidence >= 0.7:
            significance = "high"
        elif total_evidence >= 3 and avg_confidence >= 0.6:
            significance = "medium"
        elif total_evidence >= 2 and avg_confidence >= 0.55:
            significance = "low"
        else:
            significance = "insufficient"

        return {
            "aggregation_confidence": round(float(aggregation_confidence), 4),
            "evidence_sufficiency": round(float(evidence_sufficiency), 4),
            "avg_confidence": round(float(avg_confidence), 4),
            "avg_strength": round(float(avg_strength), 6),
            "consistency_score": round(float(family_consistency), 4),
            "statistical_power": round(float(statistical_power), 4),
            "total_evidence_count": int(total_evidence),
            "evidence_density": round(float(evidence_density), 2),
            "statistical_significance": significance,
            "confidence_interval": (
                round(max(0.0, aggregation_confidence - 0.1), 4),
                round(min(1.0, aggregation_confidence + 0.1), 4),
            ),
        }

    @staticmethod
    def calculate_subtype_aggregation_confidence(
        subtype_codes: List[str],
        aggregated_code_strengths: Dict[str, float],
        total_segments: int,
        segments_with_subtype: int,
    ) -> Dict[str, Any]:
        if not subtype_codes or total_segments <= 0:
            return {
                "subtype_confidence": 0.0,
                "coverage_score": 0.0,
                "consistency_score": 0.0,
                "specificity_score": 0.0,
                "segments_with_subtype": int(segments_with_subtype),
                "total_segments": int(total_segments),
            }

        present_codes = [c for c in subtype_codes if float(aggregated_code_strengths.get(c, 0.0)) > 0.1]
        coverage = len(present_codes) / max(1, len(subtype_codes))
        consistency = segments_with_subtype / max(1, total_segments)

        total_codes_present = len([v for v in aggregated_code_strengths.values() if float(v) > 0.1])
        specificity = len(present_codes) / max(1, total_codes_present)

        subtype_confidence = coverage * consistency * specificity

        return {
            "subtype_confidence": round(float(subtype_confidence), 4),
            "coverage_score": round(float(coverage), 4),
            "consistency_score": round(float(consistency), 4),
            "specificity_score": round(float(specificity), 4),
            "codes_present": int(len(present_codes)),
            "codes_expected": int(len(subtype_codes)),
            "segments_with_subtype": int(segments_with_subtype),
            "total_segments": int(total_segments),
        }

    @staticmethod
    def calculate_family_aggregation_confidence(
        family_strength: float,
        opposing_strength: float,
        family_segments: int,
        total_segments: int,
    ) -> Dict[str, Any]:
        if total_segments <= 0 or family_strength <= 0:
            return {
                "family_confidence": 0.0,
                "dominance_ratio": 0.0,
                "segment_support": 0.0,
                "effect_size": 0.0,
                "family_segments": int(family_segments),
                "total_segments": int(total_segments),
            }

        total_strength = family_strength + opposing_strength
        dominance_ratio = family_strength / total_strength if total_strength > 0 else 0.0
        segment_support = family_segments / max(1, total_segments)

        mean_difference = family_strength - opposing_strength
        pooled_variance = (family_strength + opposing_strength) / 2.0
        effect_size = mean_difference / math.sqrt(pooled_variance) if pooled_variance > 0 else 0.0

        family_confidence = (
            dominance_ratio * 0.55
            + segment_support * 0.30
            + min(1.0, max(0.0, effect_size) * 0.5) * 0.15
        )

        return {
            "family_confidence": round(float(family_confidence), 4),
            "dominance_ratio": round(float(dominance_ratio), 4),
            "segment_support": round(float(segment_support), 4),
            "effect_size": round(float(effect_size), 4),
            "family_segments": int(family_segments),
            "total_segments": int(total_segments),
        }


# =============================================================================
# UTILITIES
# =============================================================================

def _weighted_mean(pairs: List[Tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    num = 0.0
    den = 0.0
    for v, w in pairs:
        w = float(w)
        if w <= 0:
            continue
        num += float(v) * w
        den += w
    return (num / den) if den > 0 else 0.0


def _auto_weight(seg: Dict[str, Any]) -> float:
    """
    Optional statement-aware weighting (backward compatible):
    - If you're aggregating statement objects, include sentence_count + anchor_count.
    - If those fields are missing, weight=1.0.
    """
    try:
        sc = int(seg.get("sentence_count", 0) or 0)
        ac = int(seg.get("anchor_count", 0) or 0)
    except Exception:
        sc, ac = 0, 0

    w = 1.0
    if sc > 1:
        w *= 1.0 + 0.08 * float(sc - 1)
    if ac > 0:
        w *= 1.0 + 0.35 * float(ac)
    return float(max(0.1, min(10.0, w)))


def _empty_segment_result() -> Dict[str, Any]:
    return {
        "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, CENTRIST_FAMILY: 100.0},
        "evidence": [],
        "marpor_code_analysis": {},
        "marpor_breakdown": {},
        "ideology_family": CENTRIST_FAMILY,
        "ideology_subtype": None,
        "confidence_score": 0.0,
        "pattern_confidence": 0.0,
        "is_ideology_evidence": False,
        "evidence_count": 0,
        "filtered_topic_count": 0,
        "signal_strength": 0.0,
        "research_grade": False,
        "total_strength": 0.0,
        "analysis_level": "segment",
        "method": "marpor_evidence_v6_centrist_family",
        "marpor_codes": [],
        "centrist_evidence": [],
        "centrist_evidence_count": 0,
        "centrist_marpor_breakdown": {},
    }


# =============================================================================
# MARPOR BREAKDOWN (NO DOUBLE-WEIGHTING)
# =============================================================================

def calculate_marpor_breakdown(
    evidence: List[Dict[str, Any]],
    categories: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregates evidence at code level.

    IMPORTANT:
    - marpor_definitions.py already applies category.weight when computing Evidence.strength.
    - Do NOT multiply by category.weight again here.
    """
    code_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for ev in (evidence or []):
        code = str(ev.get("code", "")).strip()
        if not code:
            continue
        code_items[code].append(ev)

    code_strength: Dict[str, float] = {}
    total_strength = 0.0

    for code, items in code_items.items():
        if code not in categories:
            continue
        s = 0.0
        for e in items:
            s += float(e.get("strength", 0.0))
        code_strength[code] = s
        total_strength += s

    breakdown: Dict[str, Dict[str, Any]] = {}

    for code, items in code_items.items():
        cat = categories.get(code)
        if not cat:
            continue

        strength = float(code_strength.get(code, 0.0))
        pct = (strength / total_strength * 100.0) if total_strength > 0 else 0.0

        match_count = len(items)
        avg_strength = sum(float(e.get("strength", 0.0)) for e in items) / max(1, match_count)

        evidence_confidences = [
            float(e.get("evidence_confidence", {}).get("evidence_confidence", 0.0))
            for e in items
            if isinstance(e.get("evidence_confidence"), dict)
        ]
        avg_evidence_conf = (
            sum(evidence_confidences) / max(1, len(evidence_confidences))
            if evidence_confidences
            else 0.0
        )

        polarity = {"support": 0, "oppose": 0, "centrist": 0}
        for e in items:
            pol = int(e.get("polarity", 0))
            if pol > 0:
                polarity["support"] += 1
            elif pol < 0:
                polarity["oppose"] += 1
            else:
                polarity["centrist"] += 1

        breakdown[code] = {
            "code": code,
            "label": getattr(cat, "label", ""),
            "description": getattr(cat, "description", ""),
            "percentage": round(pct, 2),
            "match_count": match_count,
            "avg_confidence": round(avg_strength, 3),
            "avg_evidence_confidence": round(avg_evidence_conf, 3),
            "tendency": getattr(cat, "tendency", "centrist"),
            "weight": float(getattr(cat, "weight", 1.0)),
            "polarity": polarity,
            "evidence_strength": round(strength, 6),
        }

    return dict(sorted(breakdown.items(), key=lambda kv: kv[1]["percentage"], reverse=True))


# =============================================================================
# SEGMENT SCORING (lexical-first, semantic-if-needed)
# =============================================================================

def _classify_text(
    text: str,
    *,
    use_semantic: bool,
    code_threshold: float,
) -> Dict[str, Any]:
    return hybrid_marpor_analyzer.classify_text_detailed(
        text=text,
        use_semantic=bool(use_semantic),
        threshold=float(code_threshold),
    )


def score_text(
    text: str,
    *,
    use_semantic: bool = DEFAULT_USE_SEMANTIC,
    code_threshold: float = DEFAULT_CODE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Single-gate approach:
    - Do NOT recompute evidence gate here.
    - Use marpor_definitions.py analyzer outputs as truth.
    """
    text = (text or "").strip()
    if len(text) < 10:
        return _empty_segment_result()

    # 1) Lexical-only pass
    lexical_res = _classify_text(text, use_semantic=False, code_threshold=code_threshold)
    chosen_res = lexical_res

    # 2) Semantic fallback only if lexical gate fails
    if use_semantic and SEMANTIC_FALLBACK_ENABLED and not bool(lexical_res.get("is_ideology_evidence", False)):
        try:
            semantic_res = _classify_text(text, use_semantic=True, code_threshold=code_threshold)

            lex_ev = int(lexical_res.get("evidence_count", 0) or 0)
            sem_ev = int(semantic_res.get("evidence_count", 0) or 0)

            lex_conf = float(lexical_res.get("confidence_score", 0.0) or 0.0)
            sem_conf = float(semantic_res.get("confidence_score", 0.0) or 0.0)

            # Switch only if semantic produces IDEOLOGICAL evidence
            if bool(semantic_res.get("is_ideology_evidence", False)):
                if (sem_ev > lex_ev) or (sem_conf >= lex_conf + 0.05) or (sem_ev >= max(1, lex_ev)):
                    chosen_res = semantic_res
        except Exception as e:
            logger.warning("Semantic fallback failed (ignored): %s", e)

    evidence = chosen_res.get("evidence", []) or []          # ideological evidence only
    marpor_codes = chosen_res.get("marpor_codes", []) or []

    ideology_family = str(chosen_res.get("ideology_family", CENTRIST_FAMILY) or CENTRIST_FAMILY)
    ideology_subtype = chosen_res.get("ideology_subtype", None)

    # Centrist is non-ideological and has no subtype
    if ideology_family == CENTRIST_FAMILY:
        ideology_subtype = None
    else:
        ideology_subtype = str(ideology_subtype or ideology_family)

    lib_strength = float(chosen_res.get("libertarian_strength", 0.0) or 0.0)
    auth_strength = float(chosen_res.get("authoritarian_strength", 0.0) or 0.0)
    cen_strength = float(chosen_res.get("centrist_strength", 0.0) or 0.0)

    total_strength = float(chosen_res.get("total_evidence_strength", 0.0) or (lib_strength + auth_strength))

    denom = lib_strength + auth_strength + cen_strength
    if denom > 0:
        lib_pct = (lib_strength / denom) * 100.0
        auth_pct = (auth_strength / denom) * 100.0
        cen_pct = (cen_strength / denom) * 100.0
    else:
        lib_pct, auth_pct, cen_pct = 0.0, 0.0, 100.0

    is_evidence = bool(chosen_res.get("is_ideology_evidence", False))
    evidence_count = int(chosen_res.get("evidence_count", 0) or 0)
    topic_count = int(chosen_res.get("filtered_topic_count", 0) or len(set(marpor_codes)))
    confidence_score = float(chosen_res.get("confidence_score", 0.0) or 0.0)
    signal_strength = float(chosen_res.get("signal_strength", 0.0) or 0.0)

    categories = hybrid_marpor_analyzer.categories
    marpor_code_analysis = calculate_marpor_breakdown(evidence, categories)

    ev_conf_block = chosen_res.get("evidence_level_confidence", {}) or {}
    pattern_conf = ev_conf_block.get("pattern_confidence", {}) or {}
    pattern_confidence = float(pattern_conf.get("pattern_confidence", 0.0) or 0.0)

    research_grade = (
        bool(is_evidence)
        and float(confidence_score) >= RESEARCH_GRADE_MIN_CONF
        and int(evidence_count) >= RESEARCH_GRADE_MIN_EVIDENCE
        and float(pattern_confidence) >= RESEARCH_GRADE_MIN_PATTERN
    )

    # Centrist diagnostics (if marpor_definitions provides them)
    centrist_evidence = chosen_res.get("centrist_evidence", []) or []
    centrist_evidence_count = int(chosen_res.get("centrist_evidence_count", 0) or 0)
    centrist_marpor_breakdown = chosen_res.get("centrist_marpor_breakdown", {}) or {}

    return {
        "scores": {
            LIB_FAMILY: round(float(lib_pct), 2),
            AUTH_FAMILY: round(float(auth_pct), 2),
            CENTRIST_FAMILY: round(float(cen_pct), 2),
        },
        "evidence": evidence,
        "marpor_code_analysis": marpor_code_analysis,
        "marpor_breakdown": marpor_code_analysis,

        "ideology_family": ideology_family,
        "ideology_subtype": ideology_subtype,

        "confidence_score": round(float(confidence_score), 3),
        "pattern_confidence": round(float(pattern_confidence), 4),
        "is_ideology_evidence": bool(is_evidence),
        "evidence_count": int(evidence_count),
        "filtered_topic_count": int(topic_count),

        "signal_strength": round(float(signal_strength), 2),
        "total_strength": round(float(total_strength), 6),
        "research_grade": bool(research_grade),

        "analysis_level": "segment",
        "method": "marpor_evidence_v6_centrist_family",
        "marpor_codes": list(marpor_codes),

        # Centrist diagnostics
        "centrist_evidence": centrist_evidence,
        "centrist_evidence_count": centrist_evidence_count,
        "centrist_marpor_breakdown": centrist_marpor_breakdown,
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


# =============================================================================
# SUBTYPE SCORING (IDEOLOGICAL ONLY; Centrist excluded)
# =============================================================================

def calculate_subtype_scores(
    *,
    marpor_code_analysis: Dict[str, Dict[str, Any]],
    family: str,
    family_score_pct: float,
    total_segments: int,
    segments_by_subtype: Dict[str, int],
    sentences_by_subtype: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    family = (family or "").strip()
    if family not in (LIB_FAMILY, AUTH_FAMILY):
        return {}
    if family_score_pct <= 0:
        return {}

    def belongs(subtype_name: str) -> bool:
        s = (subtype_name or "").lower()
        if family == LIB_FAMILY:
            return "libertarian" in s
        if family == AUTH_FAMILY:
            return "authoritarian" in s
        return False

    family_subtypes = {st: codes for st, codes in IDEOLOGY_SUBTYPES.items() if belongs(st)}
    if not family_subtypes:
        return {}

    raw_strength: Dict[str, float] = {}
    evidence_counts: Dict[str, int] = {}
    contributing: Dict[str, Dict[str, Any]] = {}
    conf_pairs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for st, codes in family_subtypes.items():
        st_strength = 0.0
        st_evc = 0
        st_codes: Dict[str, Any] = {}

        for c in codes:
            b = marpor_code_analysis.get(c)
            if not b:
                continue

            code_strength = float(b.get("evidence_strength", 0.0))
            st_strength += code_strength
            st_evc += int(b.get("match_count", 0))

            code_avg_conf = float(b.get("avg_evidence_confidence", b.get("avg_confidence", 0.0)))
            conf_pairs[st].append((code_avg_conf, code_strength))

            st_codes[c] = {
                "percentage": b.get("percentage", 0.0),
                "matches": b.get("match_count", 0),
                "avg_confidence": b.get("avg_confidence", 0.0),
                "avg_evidence_confidence": b.get("avg_evidence_confidence", 0.0),
                "evidence_strength": b.get("evidence_strength", 0.0),
                "label": b.get("label", ""),
            }

        raw_strength[st] = st_strength
        evidence_counts[st] = st_evc
        contributing[st] = st_codes

    denom = sum(raw_strength.values())
    if denom <= 0:
        return {}

    agg = AggregationConfidence()
    out: Dict[str, Dict[str, Any]] = {}

    for st, codes in family_subtypes.items():
        share = raw_strength.get(st, 0.0) / denom if denom > 0 else 0.0
        score = share * float(family_score_pct)
        if score <= 0.01:
            continue

        supporting_segments = int(segments_by_subtype.get(st, 0))
        supporting_sentences_est = None
        if isinstance(sentences_by_subtype, dict):
            supporting_sentences_est = int(sentences_by_subtype.get(st, 0))

        subtype_confidence = agg.calculate_subtype_aggregation_confidence(
            subtype_codes=list(codes),
            aggregated_code_strengths={
                k: float(marpor_code_analysis[k]["evidence_strength"])
                for k in codes
                if k in marpor_code_analysis
            },
            total_segments=int(total_segments),
            segments_with_subtype=supporting_segments,
        )

        codes_sorted = sorted(
            contributing.get(st, {}).keys(),
            key=lambda c: float(contributing[st][c].get("evidence_strength", 0.0)),
            reverse=True,
        )

        out[st] = {
            "score": round(float(score), 2),
            "share_within_family": round(float(share), 4),
            "confidence": round(float(_weighted_mean(conf_pairs.get(st, []))), 3),
            "evidence_count": int(evidence_counts.get(st, 0)),
            "raw_strength": round(float(raw_strength.get(st, 0.0)), 6),
            "primary_codes": codes_sorted[:3],
            "contributing_codes": contributing.get(st, {}),
            "supporting_segments": supporting_segments,
            "supporting_sentences_est": supporting_sentences_est,
            "aggregation_confidence": subtype_confidence,
        }

    return dict(sorted(out.items(), key=lambda kv: kv[1]["score"], reverse=True))


# =============================================================================
# AGGREGATION
# =============================================================================

def _empty_aggregation_result() -> Dict[str, Any]:
    return {
        "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, CENTRIST_FAMILY: 100.0},
        "subtype_breakdown": {},
        "subscores": {LIB_FAMILY: {}, AUTH_FAMILY: {}},
        "marpor_code_analysis": {},
        "marpor_breakdown": {},
        "dominant_family": CENTRIST_FAMILY,
        "dominant_subtype": None,
        "confidence_score": 0.0,
        "scientific_summary": {
            "overall_confidence": "insufficient",
            "research_grade": False,
            "research_grade_percentage": 0.0,
            "avg_confidence": 0.0,
            "evidence_segments_analyzed": 0,
            "total_segments": 0,
            "statistical_significance": "insufficient",
        },
        "segment_count": 0,
        "evidence_segment_count": 0,
        "research_grade_segments": 0,
        "method": "weighted_scientific_aggregation_v6_centrist_family",
        "diagnostics": {"evidence_only": False},
        "marpor_codes": [],
    }


def aggregate_segment_scores(
    segment_scores: List[Dict[str, Any]],
    weights: Optional[List[float]] = None,
    *,
    evidence_only: bool = False,
) -> Dict[str, Any]:
    """
    Aggregates either:
    - sentence segments (weight ~1.0), or
    - statement objects (if sentence_count/anchor_count exist, auto-weights).

    If evidence_only=True, only segments with is_ideology_evidence=True and evidence_count>0 are included.

    KEY FIX:
    - Zero out scores for ideological families that have no actual evidence in aggregated segments.
    """
    if not segment_scores:
        return _empty_aggregation_result()

    total_segments = len(segment_scores)

    # If caller didn't pass weights, optionally auto-derive from statement metadata
    if weights is None or len(weights) != total_segments:
        weights = [_auto_weight(seg) for seg in segment_scores]

    if evidence_only:
        active_scores = []
        active_weights = []
        for s, w in zip(segment_scores, weights):
            if bool(s.get("is_ideology_evidence", False)) and int(s.get("evidence_count", 0) or 0) > 0:
                active_scores.append(s)
                active_weights.append(float(w))
    else:
        active_scores = list(segment_scores)
        active_weights = [float(w) for w in weights]

    evidence_segments = len(active_scores)
    if evidence_only and evidence_segments == 0:
        r = _empty_aggregation_result()
        r["segment_count"] = total_segments
        r["evidence_segment_count"] = 0
        r["diagnostics"] = {"evidence_only": True, "reason": "no_segments_passed_evidence_gate"}
        return r

    total_w = sum(active_weights) or 1.0

    lib_sum = auth_sum = cen_sum = 0.0
    conf_sum = 0.0

    family_counts = {LIB_FAMILY: 0, AUTH_FAMILY: 0, CENTRIST_FAMILY: 0}
    subtype_counts: Dict[str, int] = defaultdict(int)

    evidence_counts: List[int] = []
    confidence_scores: List[float] = []
    strengths: List[float] = []

    merged_strength = defaultdict(float)
    merged_matches = defaultdict(int)
    merged_pol = defaultdict(lambda: {"support": 0, "oppose": 0, "centrist": 0})
    merged_avg_conf_num = defaultdict(float)       # avg_confidence * strength
    merged_avg_evconf_num = defaultdict(float)     # avg_evidence_confidence * strength

    meta: Dict[str, Dict[str, Any]] = {}

    research_grade_count = 0
    all_codes = set()

    segments_by_subtype: Dict[str, int] = defaultdict(int)
    sentences_by_subtype: Dict[str, int] = defaultdict(int)
    has_sentence_counts = False

    # Track actual evidence per ideological family (fix for ghost scores)
    lib_evidence_count = 0
    auth_evidence_count = 0

    for seg, w in zip(active_scores, active_weights):
        sc = seg.get("scores", {}) or {}
        fam = str(seg.get("ideology_family", CENTRIST_FAMILY) or CENTRIST_FAMILY)
        sub = seg.get("ideology_subtype", None)
        c = float(seg.get("confidence_score", 0.0) or 0.0)
        ev_count = int(seg.get("evidence_count", 0) or 0)

        family_counts[fam] = family_counts.get(fam, 0) + 1

        if fam == LIB_FAMILY and ev_count > 0:
            lib_evidence_count += ev_count
        elif fam == AUTH_FAMILY and ev_count > 0:
            auth_evidence_count += ev_count

        # Only ideological families contribute to subtype stats
        if fam != CENTRIST_FAMILY:
            sub_s = str(sub or fam)
            subtype_counts[sub_s] += 1
            segments_by_subtype[sub_s] += 1

            if "sentence_count" in seg:
                has_sentence_counts = True
                try:
                    sentences_by_subtype[sub_s] += int(seg.get("sentence_count") or 0)
                except Exception:
                    pass

        conf_sum += c * w

        evidence_counts.append(ev_count)
        confidence_scores.append(c)
        strengths.append(float(seg.get("total_strength", 0.0) or 0.0))

        if bool(seg.get("research_grade", False)):
            research_grade_count += 1

        lib_sum += float(sc.get(LIB_FAMILY, 0.0)) * w
        auth_sum += float(sc.get(AUTH_FAMILY, 0.0)) * w

        # Prefer explicit Centrist score; otherwise derive complement
        if CENTRIST_FAMILY in sc:
            cen_sum += float(sc.get(CENTRIST_FAMILY, 0.0)) * w
        else:
            try:
                libv = float(sc.get(LIB_FAMILY, 0.0) or 0.0)
                authv = float(sc.get(AUTH_FAMILY, 0.0) or 0.0)
                cenv = max(0.0, 100.0 - libv - authv)
            except Exception:
                cenv = 0.0
            cen_sum += float(cenv) * w

        for code in (seg.get("marpor_codes", []) or []):
            all_codes.add(str(code))

        code_analysis = seg.get("marpor_code_analysis") or seg.get("marpor_breakdown") or {}
        for code, b in (code_analysis or {}).items():
            strength = float(b.get("evidence_strength", 0.0))
            merged_strength[code] += strength * w
            merged_matches[code] += int(b.get("match_count", 0))

            pol = b.get("polarity", {}) or {}
            merged_pol[code]["support"] += int(pol.get("support", 0))
            merged_pol[code]["oppose"] += int(pol.get("oppose", 0))
            merged_pol[code]["centrist"] += int(pol.get("centrist", 0))

            merged_avg_conf_num[code] += float(b.get("avg_confidence", 0.0)) * (strength * w)
            merged_avg_evconf_num[code] += float(b.get("avg_evidence_confidence", 0.0)) * (strength * w)

            if code not in meta:
                meta[code] = {
                    "label": b.get("label", ""),
                    "description": b.get("description", ""),
                    "tendency": b.get("tendency", ""),
                    "weight": b.get("weight", 1.0),
                }

    # =========================================================================
    # KEY FIX: Zero out scores for ideological families with NO actual evidence
    # =========================================================================
    if lib_evidence_count == 0:
        lib_sum = 0.0
    if auth_evidence_count == 0:
        auth_sum = 0.0

    # Normalize averaged percent scores
    lib = lib_sum / total_w
    auth = auth_sum / total_w
    cen = cen_sum / total_w
    tot = lib + auth + cen
    if tot > 0:
        lib = (lib / tot) * 100.0
        auth = (auth / tot) * 100.0
        cen = (cen / tot) * 100.0
    else:
        lib, auth, cen = 0.0, 0.0, 100.0

    overall_conf = conf_sum / total_w

    # Dominant ideology: only Lib vs Auth; Centrist is returned when ideology is essentially absent
    dominant_family = LIB_FAMILY if lib >= auth else AUTH_FAMILY
    if (lib + auth) < 1.0:
        dominant_family = CENTRIST_FAMILY

    family_consistency = 0.0
    if dominant_family != CENTRIST_FAMILY:
        family_consistency = family_counts.get(dominant_family, 0) / max(1, evidence_segments)

    # Final merged marpor breakdown
    total_strength_agg = sum(merged_strength.values())
    final_code_analysis: Dict[str, Dict[str, Any]] = {}

    for code, strength_w in merged_strength.items():
        pct = (strength_w / total_strength_agg * 100.0) if total_strength_agg > 0 else 0.0
        md = meta.get(code, {})

        avg_conf = 0.0
        avg_evconf = 0.0
        if strength_w > 0:
            avg_conf = merged_avg_conf_num[code] / strength_w
            avg_evconf = merged_avg_evconf_num[code] / strength_w

        final_code_analysis[code] = {
            "code": code,
            "label": md.get("label", ""),
            "description": md.get("description", ""),
            "percentage": round(float(pct), 2),
            "match_count": int(merged_matches[code]),
            "avg_confidence": round(float(avg_conf), 3),
            "avg_evidence_confidence": round(float(avg_evconf), 3),
            "tendency": md.get("tendency", ""),
            "weight": md.get("weight", 1.0),
            "polarity": merged_pol[code],
            "evidence_strength": round(float(strength_w / total_w), 6),
        }

    final_code_analysis = dict(sorted(final_code_analysis.items(), key=lambda kv: kv[1]["percentage"], reverse=True))

    agg = AggregationConfidence()

    if dominant_family == CENTRIST_FAMILY:
        family_strength = 0.0
        opposing_strength = 0.0
        family_agg_conf = {
            "family_confidence": 0.0,
            "dominance_ratio": 0.0,
            "segment_support": 0.0,
            "effect_size": 0.0,
            "family_segments": int(family_counts.get(CENTRIST_FAMILY, 0)),
            "total_segments": int(evidence_segments),
        }
    else:
        family_strength = float(lib if dominant_family == LIB_FAMILY else auth)
        opposing_strength = float(auth if dominant_family == LIB_FAMILY else lib)
        family_agg_conf = agg.calculate_family_aggregation_confidence(
            family_strength=family_strength,
            opposing_strength=opposing_strength,
            family_segments=int(family_counts.get(dominant_family, 0)),
            total_segments=int(evidence_segments),
        )

    agg_confidence = agg.calculate_aggregated_confidence(
        evidence_counts=evidence_counts,
        confidence_scores=confidence_scores,
        strengths=strengths,
        family_consistency=family_consistency,
    )

    # Subtypes: only for ideological families
    subtype_breakdown: Dict[str, Dict[str, Any]] = {}
    dominant_subtype: Optional[str] = None
    if dominant_family in (LIB_FAMILY, AUTH_FAMILY):
        subtype_breakdown = calculate_subtype_scores(
            marpor_code_analysis=final_code_analysis,
            family=dominant_family,
            family_score_pct=float(family_strength),
            total_segments=int(evidence_segments),
            segments_by_subtype=dict(segments_by_subtype),
            sentences_by_subtype=(dict(sentences_by_subtype) if has_sentence_counts else None),
        )
        if subtype_breakdown:
            dominant_subtype = max(subtype_breakdown.items(), key=lambda kv: float(kv[1].get("score", 0.0)))[0]

    subscores: Dict[str, Dict[str, Any]] = {
        LIB_FAMILY: calculate_subtype_scores(
            marpor_code_analysis=final_code_analysis,
            family=LIB_FAMILY,
            family_score_pct=float(lib),
            total_segments=int(evidence_segments),
            segments_by_subtype=dict(segments_by_subtype),
            sentences_by_subtype=(dict(sentences_by_subtype) if has_sentence_counts else None),
        ),
        AUTH_FAMILY: calculate_subtype_scores(
            marpor_code_analysis=final_code_analysis,
            family=AUTH_FAMILY,
            family_score_pct=float(auth),
            total_segments=int(evidence_segments),
            segments_by_subtype=dict(segments_by_subtype),
            sentences_by_subtype=(dict(sentences_by_subtype) if has_sentence_counts else None),
        ),
    }

    research_grade_percentage = (research_grade_count / max(1, evidence_segments)) * 100.0

    agg_val = float(agg_confidence.get("aggregation_confidence", 0.0) or 0.0)
    if agg_val >= 0.70:
        overall_confidence_tier = "high"
    elif agg_val >= 0.55:
        overall_confidence_tier = "medium"
    elif agg_val >= 0.40:
        overall_confidence_tier = "low"
    else:
        overall_confidence_tier = "insufficient"

    scientific_summary = {
        "overall_confidence": overall_confidence_tier,
        "research_grade": bool(research_grade_percentage >= 60.0),
        "research_grade_percentage": round(float(research_grade_percentage), 1),
        "avg_confidence": round(float(overall_conf), 3),
        "evidence_segments_analyzed": int(evidence_segments),
        "total_segments": int(total_segments),
        "family_consistency": round(float(family_consistency), 3),
        "aggregation_confidence": agg_confidence,
        "family_confidence": family_agg_conf,
        "statistical_significance": agg_confidence.get("statistical_significance", "insufficient"),
        "notes": (
            ["supporting_sentences_est is available"]
            if has_sentence_counts
            else ["supporting_sentences_est unavailable (missing sentence_count in segments)"]
        ),
    }

    return {
        "scores": {
            LIB_FAMILY: round(float(lib), 2),
            AUTH_FAMILY: round(float(auth), 2),
            CENTRIST_FAMILY: round(float(cen), 2),
        },
        "dominant_family": dominant_family,
        "dominant_subtype": dominant_subtype,
        "confidence_score": round(float(overall_conf), 3),

        "marpor_code_analysis": final_code_analysis,
        "marpor_breakdown": final_code_analysis,

        "subtype_breakdown": subtype_breakdown,
        "subscores": subscores,

        "scientific_summary": scientific_summary,

        "segment_count": int(total_segments),
        "evidence_segment_count": int(evidence_segments),
        "research_grade_segments": int(research_grade_count),
        "family_counts": family_counts,
        "subtype_counts": dict(subtype_counts),
        "marpor_codes": sorted(all_codes),
        "method": "weighted_scientific_aggregation_v6_centrist_family_fixed",
        "diagnostics": {
            "evidence_only": bool(evidence_only),
            "lib_evidence_count": int(lib_evidence_count),
            "auth_evidence_count": int(auth_evidence_count),
        },
    }


def configure_embedder(embedder: Optional[Any]) -> None:
    hybrid_marpor_analyzer.set_embedder(embedder)


__all__ = [
    "configure_embedder",
    "score_text",
    "score_segments",
    "aggregate_segment_scores",
    "calculate_marpor_breakdown",
    "calculate_subtype_scores",
    "AggregationConfidence",
    "LIB_FAMILY",
    "AUTH_FAMILY",
    "CENTRIST_FAMILY",
]
