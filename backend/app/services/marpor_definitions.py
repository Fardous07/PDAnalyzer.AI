"""
backend/app/services/marpor_definitions.py

RESEARCH-GRADE MARPOR DEFINITIONS + EVIDENCE GATE (STRICT, NO FALLBACK)
======================================================================

Design goals (aligned with your discourse ingestion rules):
1) Only sentences that MATCH MARPOR (lexical/semantic) can be evidence.
2) No fallback that marks all sentences as evidence.
3) Evidence gate is consistent and stable:
   - evidence_count is derived from VALIDATED MARPOR evidence hits
   - is_ideology_evidence is True only when the IDEOLOGICAL gate passes
4) Polarity affects ideological direction (support/oppose), but does not erase evidence existence.

Key policy for DISCOURSE (updated):
- "Neutral" is REMOVED.
- "Centrist" is the only non-ideological family label.
- Centrist has NO subtype.
- Centrist discourse categories (CENT/BI/REF) can exist as non-ideological evidence but do NOT
  count toward ideological evidence_count / ideological gate.

How do we decide Centrist vs Ideological?
- A sentence is CENTRIST if it has:
  (a) no validated ideological evidence, OR
  (b) ideological evidence exists but fails the ideological evidence gate, OR
  (c) only centrist-discourse evidence exists (CENT/BI/REF).

Outputs expected by downstream (ideology_scoring + speech_ingestion):
- ideology_family: "Libertarian" | "Authoritarian" | "Centrist"
- ideology_subtype: subtype str for Lib/Auth; None for Centrist
- evidence: list[...] (IDEOLOGICAL validated hits only; includes negative/neutral polarity)
- evidence_count (ideological validated hits)
- support_evidence_count (ideological validated hits with polarity > 0)
- marpor_codes (ideological codes only)
- is_ideology_evidence (True only if ideological gate passes)
- confidence_score (0..1)
- signal_strength (0..100) based on ideological total strength

Additional helpful outputs:
- centrist_evidence / centrist_evidence_count / centrist_marpor_codes (diagnostic only)
- scores: Libertarian/Authoritarian/Centrist percent share by strength
- marpor_breakdown / marpor_code_analysis (derived from validated evidence; no new evidence created)
  * marpor_breakdown: ideological only
  * centrist_marpor_breakdown: centrist only
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# SCIENTIFIC CONFIGURATION (UNIFIED GATE + SEMANTIC CONTROLS)
# =============================================================================

LIB_FAMILY = "Libertarian"
AUTH_FAMILY = "Authoritarian"
CENTRIST_FAMILY = "Centrist"

# ---- IDEOLOGICAL evidence gate (Centrist never passes) ----
EVIDENCE_MIN_EVIDENCE_COUNT = 1
EVIDENCE_MIN_TOPIC_COUNT = 1
EVIDENCE_MIN_TOTAL_STRENGTH = 0.20
EVIDENCE_MIN_DOMINANCE = 0.35  # dominance computed over Lib/Auth only

# Evidence pruning
MIN_EVIDENCE_STRENGTH_KEEP = 0.20

# ---- Semantic thresholds ----
SEMANTIC_THRESHOLD = 0.70
SEMANTIC_NEG_THRESHOLD = 0.80
SEMANTIC_MARGIN = 0.06
SEMANTIC_TOPK = 5
SEMANTIC_LEXICAL_OVERRIDE = 0.75  # skip semantic if lexical already strong

# Evidence-level confidence tiers
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.55


# =============================================================================
# IDEOLOGY SUBTYPE MAPPING (IDEOLOGICAL ONLY; Centrist has no subtype)
# =============================================================================

IDEOLOGY_SUBTYPES: Dict[str, List[str]] = {
    # Libertarian subtypes
    "Right-Libertarianism": ["401", "407", "414", "505", "702", "301", "507"],
    "Left-Libertarianism": ["201", "203", "SJ", "604", "607", "503", "501"],
    "Cultural Libertarianism": ["201", "604", "607", "503"],
    "Geo-Libertarianism": ["501", "401", "407", "301"],
    "Paleo-Libertarianism": ["401", "414", "505", "603", "601", "702"],

    # Authoritarian subtypes
    "Right-Authoritarian": ["305", "605", "603", "601", "608", "302", "PROT", "POP"],
    "Left-Authoritarian": ["404", "412", "413", "504", "701", "ENV_AUTH", "SJ"],
}


# =============================================================================
# EVIDENCE WEIGHTING SYSTEM
# =============================================================================

class EvidenceWeight:
    """Weighting for different evidence types / conditions."""

    # Base weights
    LEXICAL_PRIMARY: float = 1.0
    LEXICAL_SECONDARY: float = 0.7
    SEMANTIC_DIRECT: float = 0.9
    SEMANTIC_PARAPHRASE: float = 0.75
    SEMANTIC_WEAK: float = 0.6

    # Context modifiers
    CONTEXT_REQUIRED_MET: float = 1.1
    CONTEXT_REQUIRED_MISSED: float = 0.7
    CONTEXT_EXCLUDED_PRESENT: float = 0.3
    INSIDE_QUOTES: float = 0.6
    OUTSIDE_QUOTES: float = 1.0

    # Polarity modifiers (affects evidence strength only; not existence)
    SUPPORT_STRENGTH: float = 1.0
    OPPOSE_STRENGTH: float = 0.5
    CENTRIST_STRENGTH: float = 0.7

    # Position modifiers
    OPENING_STATEMENT: float = 1.2
    CLOSING_STATEMENT: float = 1.15
    MIDDLE_STATEMENT: float = 1.0

    @classmethod
    def get_evidence_quality(cls, match_type: str, similarity: Optional[float] = None) -> float:
        if match_type == "lexical_primary":
            return cls.LEXICAL_PRIMARY
        if match_type == "lexical_secondary":
            return cls.LEXICAL_SECONDARY
        if match_type == "semantic":
            if similarity is None:
                return cls.SEMANTIC_WEAK
            if similarity >= 0.85:
                return cls.SEMANTIC_DIRECT
            if similarity >= 0.75:
                return cls.SEMANTIC_PARAPHRASE
            return cls.SEMANTIC_WEAK
        return 0.5

    @classmethod
    def get_position_modifier(cls, char_position: int, total_chars: int) -> float:
        if total_chars <= 0:
            return cls.MIDDLE_STATEMENT
        ratio = char_position / total_chars
        if ratio <= 0.10:
            return cls.OPENING_STATEMENT
        if ratio >= 0.90:
            return cls.CLOSING_STATEMENT
        return cls.MIDDLE_STATEMENT


# =============================================================================
# EVIDENCE-LEVEL CONFIDENCE
# =============================================================================

class EvidenceConfidence:
    @staticmethod
    def calculate_evidence_confidence(
        *,
        match_type: str,
        similarity: Optional[float],
        context_hits: int,
        context_misses: int,
        inside_quotes: bool,
        polarity: int,
        char_position: int,
        total_chars: int,
    ) -> Dict[str, Any]:
        base_quality = EvidenceWeight.get_evidence_quality(match_type, similarity)

        context_factor = 1.0
        if context_misses > 0:
            context_factor *= 0.8
        if context_hits > 0:
            context_factor *= 1.1

        quote_factor = 0.6 if inside_quotes else 1.0

        if polarity > 0:
            polarity_factor = EvidenceWeight.SUPPORT_STRENGTH
        elif polarity < 0:
            polarity_factor = EvidenceWeight.OPPOSE_STRENGTH
        else:
            polarity_factor = EvidenceWeight.CENTRIST_STRENGTH

        semantic_boost = 1.0
        if match_type == "semantic" and similarity is not None:
            semantic_boost = 0.9 + (float(similarity) * 0.2)

        position_modifier = EvidenceWeight.get_position_modifier(char_position, total_chars)

        evidence_conf = base_quality * context_factor * quote_factor * polarity_factor * semantic_boost * position_modifier
        evidence_conf = float(min(1.0, max(0.0, evidence_conf)))

        reliability = min(1.0, evidence_conf * 1.2)

        if match_type == "lexical_primary":
            error_margin = 0.05
        elif match_type == "lexical_secondary":
            error_margin = 0.10
        elif match_type == "semantic":
            error_margin = max(0.15, 0.25 - (float(similarity or 0.0) * 0.2))
        else:
            error_margin = 0.15

        if evidence_conf >= HIGH_CONFIDENCE_THRESHOLD:
            tier = "high"
        elif evidence_conf >= MEDIUM_CONFIDENCE_THRESHOLD:
            tier = "medium"
        elif evidence_conf >= LOW_CONFIDENCE_THRESHOLD:
            tier = "low"
        else:
            tier = "insufficient"

        return {
            "evidence_confidence": round(evidence_conf, 4),
            "base_quality": round(base_quality, 4),
            "context_factor": round(context_factor, 4),
            "quote_factor": quote_factor,
            "polarity_factor": polarity_factor,
            "position_modifier": round(position_modifier, 4),
            "reliability_score": round(reliability, 4),
            "error_margin": round(error_margin, 4),
            "quality_tier": tier,
            "confidence_interval": (
                round(max(0.0, evidence_conf - error_margin), 4),
                round(min(1.0, evidence_conf + error_margin), 4),
            ),
        }

    @staticmethod
    def calculate_pattern_confidence_single(evidence_codes: List[str], subtype_codes: List[str]) -> Dict[str, Any]:
        if not evidence_codes or not subtype_codes:
            return {
                "pattern_coverage": 0.0,
                "pattern_specificity": 0.0,
                "pattern_confidence": 0.0,
                "codes_present": 0,
                "codes_expected": len(subtype_codes) if subtype_codes else 0,
            }

        present = set(evidence_codes) & set(subtype_codes)
        expected = len(subtype_codes)

        coverage = len(present) / expected if expected > 0 else 0.0
        specificity = len(present) / len(set(evidence_codes)) if evidence_codes else 0.0
        pc = coverage * specificity

        return {
            "pattern_coverage": round(coverage, 4),
            "pattern_specificity": round(specificity, 4),
            "pattern_confidence": round(pc, 4),
            "codes_present": len(present),
            "codes_expected": expected,
        }


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarporCategory:
    code: str
    label: str
    description: str
    tendency: str  # "libertarian" | "authoritarian" | "centrist"
    weight: float = 1.0

    primary_keywords: List[str] = field(default_factory=list)
    secondary_keywords: List[str] = field(default_factory=list)

    semantic_positive: List[str] = field(default_factory=list)
    semantic_negative: List[str] = field(default_factory=list)

    requires_context: List[str] = field(default_factory=list)
    excludes_context: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)

    # Hard-context control: when True, lexical hits require >=1 context hit to be kept.
    requires_context_hard: bool = False

    subtype_group: str = "general"

    @property
    def ideology_family(self) -> str:
        if self.tendency == "libertarian":
            return LIB_FAMILY
        if self.tendency == "authoritarian":
            return AUTH_FAMILY
        return CENTRIST_FAMILY

    def get_all_keywords(self) -> List[str]:
        return list(self.primary_keywords) + list(self.secondary_keywords)


@dataclass
class Evidence:
    code: str
    matched_text: str
    span: Tuple[int, int]
    match_type: str  # "lexical_primary" | "lexical_secondary" | "semantic"
    base_strength: float
    strength: float
    polarity: int  # +1 support, -1 oppose, 0 centrist/ambiguous
    polarity_reason: str
    inside_quotes: bool = False
    semantic_similarity: Optional[float] = None
    context_hits: List[str] = field(default_factory=list)
    context_misses: List[str] = field(default_factory=list)
    evidence_confidence: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SEMANTIC SERVICE
# =============================================================================

class SemanticService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.model = SentenceTransformer(model_name)
            logger.info("Loaded semantic model: %s", model_name)
        except Exception as e:
            logger.warning("Could not load sentence transformer (%s). Semantic matching disabled.", e)
            self.model = None

    def encode_many(self, texts: List[str]) -> Optional[np.ndarray]:
        if self.model is None or not texts:
            return None
        try:
            try:
                emb = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return np.asarray(emb, dtype=np.float32)
            except TypeError:
                emb = self.model.encode(texts, show_progress_bar=False)
                emb = np.asarray(emb, dtype=np.float32)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.where(norms == 0.0, 1.0, norms)
                return emb / norms
        except Exception as e:
            logger.warning("Semantic encode_many failed: %s", e)
            return None

    def encode_one(self, text: str) -> Optional[np.ndarray]:
        out = self.encode_many([text])
        if out is None or len(out) == 0:
            return None
        return out[0]

    @staticmethod
    def cosine_similarity_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
        if vec is None or mat is None or mat.size == 0:
            return np.zeros((0,), dtype=np.float32)
        v = vec.astype(np.float32)
        M = mat.astype(np.float32)

        v_norm = np.linalg.norm(v) or 1.0
        M_norm = np.linalg.norm(M, axis=1)
        M_norm = np.where(M_norm == 0.0, 1.0, M_norm)

        sims = (M @ v) / (M_norm * v_norm)
        return sims.astype(np.float32)


# =============================================================================
# MAIN ANALYZER
# =============================================================================

class HybridMarporAnalyzer:
    """
    Evidence-level analyzer:
    - Lexical matching + selective semantic matching
    - Context validation + polarity
    - IDEOLOGICAL evidence gate (Centrist excluded from gate)
    """

    def __init__(self, semantic_model: Optional[str] = "all-MiniLM-L6-v2"):
        self.categories: Dict[str, MarporCategory] = {}
        self.patterns: Dict[str, List[Tuple[str, re.Pattern]]] = {}

        self.semantic = SemanticService(semantic_model or "all-MiniLM-L6-v2")
        self.conf_calc = EvidenceConfidence()
        self.weights = EvidenceWeight()

        self._sem_pos_emb: Dict[str, np.ndarray] = {}
        self._sem_neg_emb: Dict[str, np.ndarray] = {}
        self._sem_pos_texts: Dict[str, List[str]] = {}
        self._sem_neg_texts: Dict[str, List[str]] = {}

        self._init_all_categories()
        self._compile_patterns()
        self._precompute_semantic_embeddings()

        self.negation_patterns = [
            re.compile(r"\b(not|never|no|none|neither|nor|without)\b", re.IGNORECASE),
            re.compile(r"\b(cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't)\b", re.IGNORECASE),
            re.compile(r"\b(against|opposed to|anti|counter to)\b", re.IGNORECASE),
            re.compile(r"\b(refuse|reject|deny|oppose)\b", re.IGNORECASE),
        ]
        self.opposition_patterns = [
            re.compile(r"\b(oppose|reject|condemn|denounce|criticize|criticise)\b", re.IGNORECASE),
            re.compile(r"\b(fight|resist|ban|prohibit|abolish|stop|end)\b", re.IGNORECASE),
            re.compile(r"\b(disagree|dispute|challenge|contest)\b", re.IGNORECASE),
        ]
        self.endorsement_patterns = [
            re.compile(r"\b(support|endorse|favor|favour|back|approve)\b", re.IGNORECASE),
            re.compile(r"\b(we must|we should|we need|we have to|we will)\b", re.IGNORECASE),
            re.compile(r"\b(implement|create|establish|expand|strengthen|protect|defend)\b", re.IGNORECASE),
            re.compile(r"\b(advocate|promote|champion|embrace)\b", re.IGNORECASE),
        ]

        logger.info("Initialized HybridMarporAnalyzer with %d categories", len(self.categories))

    def set_embedder(self, embedder: Any) -> None:
        if hasattr(embedder, "encode"):
            self.semantic.model = embedder
            self._precompute_semantic_embeddings()
            logger.info("External embedder set and semantic embeddings rebuilt.")

    # -------------------------------------------------------------------------
    # CATEGORY INITIALIZATION
    # -------------------------------------------------------------------------

    def _init_all_categories(self) -> None:
        # ==================== LIBERTARIAN ====================

        self.categories["201"] = MarporCategory(
            code="201",
            label="Freedom and Human Rights",
            description="Individual freedoms, civil liberties, human rights",
            tendency="libertarian",
            weight=1.0,
            primary_keywords=[
                "freedom", "liberty", "civil liberties", "human rights", "free speech",
                "individual rights", "personal freedom", "civil rights"
            ],
            secondary_keywords=[
                "choice", "consent", "agency", "sovereignty", "empowerment",
                "autonomy", "self-determination", "free will"
            ],
            semantic_positive=[
                "We must protect individual freedoms and civil liberties.",
                "Free speech is the foundation of a free society.",
                "Privacy and civil rights must be safeguarded.",
                "People should have the right to make their own choices.",
                "Individual autonomy is essential for human dignity.",
                "Personal freedom is a fundamental human right.",
                "Civil liberties are the cornerstone of democracy."
            ],
            semantic_negative=[
                "We need to limit some freedoms for security.",
                "Some rights must be restricted for the greater good.",
                "Free speech can be dangerous and needs controls."
            ],
            requires_context=[
                "rights", "civil", "liberties", "speech", "amendment", "due process",
                "privacy", "protect", "restrict", "petition", "assemble"
            ],
            excludes_context=["may god bless", "god bless", "applause", "thank you"],
            requires_context_hard=True,
        )

        self.categories["203"] = MarporCategory(
            code="203",
            label="Constitutionalism and Rule of Law",
            description="Rule of law, constitutional limits, due process, judicial independence",
            tendency="libertarian",
            weight=0.9,
            primary_keywords=[
                "rule of law", "constitution", "constitutional", "due process",
                "checks and balances", "separation of powers", "judicial independence",
                "constitutional rights", "legal protections"
            ],
            secondary_keywords=[
                "accountability", "oversight", "transparency", "judicial review",
                "constitutional limits", "legal framework", "procedural fairness"
            ],
            semantic_positive=[
                "We must uphold constitutional limits on government power.",
                "The rule of law must prevail over political expediency.",
                "An independent judiciary protects our rights.",
                "Due process ensures fairness for everyone.",
                "Police must be held accountable through proper oversight.",
                "Constitutional protections safeguard individual liberty.",
                "Separation of powers prevents tyranny."
            ],
            semantic_negative=[
                "We need strong police powers without interference.",
                "Constitutional limits hinder effective governance.",
                "Judicial independence is less important than security."
            ],
            requires_context=["oversight", "accountability", "reform", "transparency", "protect", "amendment", "rights", "due process"],
            excludes_context=["crackdown", "zero tolerance", "strict enforcement", "war on"],
            requires_context_hard=False,
        )

        self.categories["301"] = MarporCategory(
            code="301",
            label="Decentralization",
            description="Power to local/regional levels, federalism, local control",
            tendency="libertarian",
            weight=0.8,
            primary_keywords=["decentralize", "local control", "states' rights", "federalism", "devolution", "local autonomy", "regional power"],
            secondary_keywords=["subsidiarity", "local governance", "community control", "bottom-up", "grassroots", "local decision-making"],
            semantic_positive=[
                "Power should be devolved to local communities.",
                "Decentralization makes government more responsive.",
                "Local control ensures decisions reflect community needs.",
                "Federalism balances national and local interests.",
                "Communities know their needs better than distant bureaucrats."
            ],
        )

        self.categories["401"] = MarporCategory(
            code="401",
            label="Free Enterprise",
            description="Free-market capitalism, private sector, entrepreneurship",
            tendency="libertarian",
            weight=1.0,
            primary_keywords=["free market", "free enterprise", "economic freedom", "private sector", "deregulation", "capitalism", "market economy", "entrepreneurship"],
            secondary_keywords=["business freedom", "competition", "innovation", "private ownership", "market forces", "economic liberty", "free trade"],
            semantic_positive=[
                "Free markets create prosperity and opportunity.",
                "Private enterprise drives growth better than bureaucracy.",
                "Economic freedom unleashes innovation and entrepreneurship.",
                "Competition benefits consumers through lower prices and better quality.",
                "Markets allocate resources more efficiently than government.",
                "Capitalism has lifted billions out of poverty."
            ],
        )

        self.categories["407"] = MarporCategory(
            code="407",
            label="Free Trade",
            description="Free trade, open markets, against protectionism",
            tendency="libertarian",
            weight=0.8,
            primary_keywords=["free trade", "open markets", "trade liberalization", "remove tariffs", "globalization", "international trade", "trade agreements"],
            secondary_keywords=["trade openness", "market access", "trade barriers removal", "economic integration", "global commerce"],
            semantic_positive=[
                "Free trade benefits consumers and drives innovation.",
                "Open markets create opportunities for businesses and workers.",
                "Trade liberalization increases economic efficiency.",
                "Global trade reduces poverty worldwide.",
                "Tariffs hurt consumers and reduce economic growth."
            ],
        )

        self.categories["414"] = MarporCategory(
            code="414",
            label="Economic Orthodoxy",
            description="Tax cuts, balanced budgets, fiscal responsibility",
            tendency="libertarian",
            weight=0.9,
            primary_keywords=["tax cuts", "cut taxes", "balanced budget", "fiscal responsibility", "deficit reduction", "fiscal discipline", "lower taxes"],
            secondary_keywords=["fiscal prudence", "sound money", "budget balance", "tax relief", "spending restraint", "fiscal conservatism"],
            semantic_positive=[
                "Tax cuts stimulate economic growth and job creation.",
                "Fiscal responsibility ensures long-term economic stability.",
                "Balanced budgets prevent burdening future generations.",
                "Lower taxes leave more money in people's pockets.",
                "Government should live within its means like families do."
            ],
        )

        self.categories["505"] = MarporCategory(
            code="505",
            label="Welfare State Limitation",
            description="Limit welfare/social programs, promote self-reliance",
            tendency="libertarian",
            weight=0.8,
            primary_keywords=["welfare reform", "limit welfare", "personal responsibility", "self-reliance", "work requirements", "welfare reduction"],
            secondary_keywords=["self-sufficiency", "dependency reduction", "welfare-to-work", "individual responsibility", "earned benefits"],
            semantic_positive=[
                "Welfare reform should promote work and self-sufficiency.",
                "Personal responsibility is key to breaking the cycle of poverty.",
                "Work requirements restore dignity and encourage employment.",
                "We need to limit welfare to those truly in need.",
                "Dependency undermines individual initiative and dignity."
            ],
        )

        self.categories["507"] = MarporCategory(
            code="507",
            label="Education Limitation / School Choice",
            description="School choice, vouchers, charter schools",
            tendency="libertarian",
            weight=0.7,
            primary_keywords=["school choice", "vouchers", "charter schools", "parental choice", "education vouchers", "school competition"],
            secondary_keywords=["education freedom", "parental rights", "private schools", "education alternatives", "school options"],
            semantic_positive=[
                "Parents should have the right to choose their children's education.",
                "School choice creates competition that improves all schools.",
                "Educational freedom empowers families and students.",
                "Vouchers give low-income families access to better schools.",
                "Competition drives quality improvement in education."
            ],
        )

        self.categories["702"] = MarporCategory(
            code="702",
            label="Anti-Union / Workplace Freedom",
            description="Against compulsory unionism, right-to-work",
            tendency="libertarian",
            weight=0.7,
            primary_keywords=["right to work", "reduce union power", "workplace freedom", "union reform", "voluntary unionism"],
            secondary_keywords=["individual bargaining", "union accountability", "worker choice", "labor flexibility", "employment freedom"],
            semantic_positive=[
                "Right-to-work laws protect individual worker choice.",
                "Workers should decide whether to join a union.",
                "Union reform increases workplace flexibility and competitiveness.",
                "Compulsory union membership violates freedom of association."
            ],
        )

        self.categories["503"] = MarporCategory(
            code="503",
            label="Social Justice (Equality)",
            description="Equality, equity, social justice, redistribution",
            tendency="libertarian",
            weight=0.7,
            primary_keywords=["social justice", "equality", "equity", "redistribution", "fairness", "equal opportunity", "income equality"],
            secondary_keywords=["economic justice", "wealth redistribution", "equal rights", "social equity", "progressive taxation"],
            semantic_positive=[
                "We need greater economic equality for a just society.",
                "Social justice requires addressing income inequality.",
                "Fair redistribution ensures everyone benefits from economic growth.",
                "Equal opportunity means removing barriers to success.",
                "Economic inequality threatens social cohesion and democracy."
            ],
        )

        self.categories["604"] = MarporCategory(
            code="604",
            label="Social Freedom",
            description="Social freedom, against moral policing, lifestyle autonomy",
            tendency="libertarian",
            weight=0.8,
            primary_keywords=["personal autonomy", "individual choice", "secular", "social freedom", "lifestyle freedom", "moral liberty"],
            secondary_keywords=["personal choice", "private life", "individual lifestyle", "freedom of conscience", "personal morality"],
            semantic_positive=[
                "Personal lifestyle choices should remain private.",
                "Individuals should be free to live as they choose.",
                "Social freedom allows diverse lifestyles and beliefs.",
                "The government shouldn't dictate personal moral choices.",
                "Private decisions are not the state's business."
            ],
        )

        self.categories["607"] = MarporCategory(
            code="607",
            label="Multiculturalism",
            description="Pro-diversity, inclusion, plural society",
            tendency="libertarian",
            weight=0.7,
            primary_keywords=["multiculturalism", "diversity", "inclusion", "plural society", "cultural diversity", "pluralism"],
            secondary_keywords=["multicultural", "inclusive society", "cultural pluralism", "diversity strength", "cultural richness"],
            semantic_positive=[
                "Diversity and inclusion strengthen our society.",
                "Multiculturalism enriches our culture and economy.",
                "An inclusive society benefits everyone.",
                "Cultural diversity is a source of strength.",
                "Different perspectives and backgrounds drive innovation."
            ],
        )

        self.categories["SJ"] = MarporCategory(
            code="SJ",
            label="Social Justice",
            description="Equity, systemic reform, anti-discrimination, modern social justice",
            tendency="libertarian",
            weight=0.8,
            primary_keywords=["social justice", "equity", "systemic racism", "racial justice", "inclusion", "dei", "anti-discrimination"],
            secondary_keywords=["structural inequality", "institutional bias", "systemic change", "inclusive justice", "equitable society"],
            semantic_positive=[
                "We must address systemic racism and inequality.",
                "Social justice requires dismantling structural barriers.",
                "Equity means ensuring everyone has what they need to succeed.",
                "We need to build a more inclusive and just society.",
                "Diversity, equity, and inclusion make organizations stronger.",
                "Systemic change is needed to address historical injustices."
            ],
        )

        self.categories["501"] = MarporCategory(
            code="501",
            label="Environmental Protection",
            description="Environmental protection, climate action, sustainability",
            tendency="libertarian",
            weight=0.7,
            primary_keywords=["climate change", "environment", "renewable energy", "clean energy", "sustainability", "environmental protection", "green energy"],
            secondary_keywords=["climate action", "ecological", "conservation", "green economy", "carbon reduction", "sustainable development"],
            semantic_positive=[
                "We must protect our environment for future generations.",
                "Clean energy creates jobs and protects our planet.",
                "Market-based solutions can effectively reduce emissions.",
                "Environmental stewardship is our responsibility.",
                "Climate change threatens our prosperity and security."
            ],
        )

        # ==================== AUTHORITARIAN ====================

        self.categories["605"] = MarporCategory(
            code="605",
            label="Law and Order",
            description="Tough on crime, strict law enforcement, public safety emphasis",
            tendency="authoritarian",
            weight=0.85,
            primary_keywords=["law and order", "tough on crime", "crackdown", "zero tolerance", "public safety", "crime control", "strict enforcement"],
            secondary_keywords=["law enforcement", "crime prevention", "public security", "order maintenance", "criminal justice", "police power"],
            semantic_positive=[
                "We need tough law enforcement to keep our streets safe.",
                "Zero tolerance policies deter crime effectively.",
                "Strong policing is essential for public safety.",
                "We must crack down on crime and disorder.",
                "Law and order is the foundation of a civilized society."
            ],
            requires_context=["crime", "enforcement", "crackdown", "zero tolerance", "public safety", "strict"],
            excludes_context=["oversight", "accountability", "reform", "community policing"],
        )

        self.categories["302"] = MarporCategory(
            code="302",
            label="Centralization",
            description="Centralized power, top-down control, strong central government",
            tendency="authoritarian",
            weight=0.9,
            primary_keywords=["centralize", "centralization", "strong central government", "top-down control", "federal authority", "central power"],
            secondary_keywords=["national control", "unified command", "central coordination", "federal supremacy", "centralized decision-making"],
            semantic_positive=[
                "We need strong central authority to ensure uniform action.",
                "Centralized planning is more efficient than local control.",
                "Top-down direction ensures coordinated national response.",
                "Consolidating power at the federal level improves governance.",
                "Only central government has resources for major challenges."
            ],
            requires_context=["control", "authority", "command", "central", "consolidate"],
        )

        self.categories["305"] = MarporCategory(
            code="305",
            label="Political Authority",
            description="Strong leadership, decisive action, authority emphasis",
            tendency="authoritarian",
            weight=0.8,
            primary_keywords=["strong leadership", "decisive action", "firm hand", "authority", "leadership", "strong government", "executive power"],
            secondary_keywords=["decisive leadership", "authoritative", "command", "firm governance", "strong executive"],
            semantic_positive=[
                "We need strong leadership to restore order and stability.",
                "Decisive action is necessary in times of crisis.",
                "A firm hand guides the nation through challenges.",
                "Strong authority ensures effective governance.",
                "Leadership requires the will to act decisively."
            ],
            requires_context=["authority", "executive", "command", "enforce", "order", "security"],
            requires_context_hard=True,
        )

        self.categories["603"] = MarporCategory(
            code="603",
            label="Traditional Morality",
            description="Traditional values, family values, religious morality",
            tendency="authoritarian",
            weight=0.8,
            primary_keywords=["traditional values", "family values", "religious", "moral standards", "pro-life", "traditional family", "moral order"],
            secondary_keywords=["religious values", "moral tradition", "family structure", "traditional marriage", "moral foundation"],
            semantic_positive=[
                "We must defend traditional family values.",
                "Religious faith provides moral guidance for society.",
                "Traditional values strengthen communities and families.",
                "The sanctity of life must be protected.",
                "Moral standards are essential for social cohesion."
            ],
        )

        self.categories["601"] = MarporCategory(
            code="601",
            label="National Way of Life",
            description="National identity, patriotism, sovereignty, cultural heritage",
            tendency="authoritarian",
            weight=0.8,
            primary_keywords=["national identity", "patriotism", "sovereignty", "our nation", "national heritage", "national pride", "cultural identity"],
            secondary_keywords=["national culture", "national traditions", "national unity", "cultural heritage", "national values"],
            semantic_positive=[
                "We must defend our national identity and sovereignty.",
                "Patriotism means loving and supporting our country.",
                "Our national heritage and traditions must be preserved.",
                "A strong national identity unites our people.",
                "National pride is essential for social cohesion."
            ],
            requires_context=["sovereignty", "identity", "heritage", "tradition", "patriot", "national pride"],
            excludes_context=["may god bless", "god bless", "thank you"],
            requires_context_hard=True,
        )

        self.categories["608"] = MarporCategory(
            code="608",
            label="Anti-Multiculturalism / Immigration Restriction",
            description="Assimilation emphasis, immigration restriction, border control",
            tendency="authoritarian",
            weight=0.8,
            primary_keywords=["assimilation", "limit immigration", "border security", "secure the border", "deport", "immigration control", "cultural unity"],
            secondary_keywords=["cultural integration", "immigration restriction", "border control", "cultural cohesion", "controlled immigration"],
            semantic_positive=[
                "Immigrants must assimilate to our national culture.",
                "We need strong borders to protect our national identity.",
                "Cultural unity is essential for social cohesion.",
                "Illegal immigration threatens our sovereignty and security.",
                "Controlled immigration protects jobs and wages."
            ],
        )

        # --- KEYWORD FIXES REQUESTED: 404 / 504 / 701 --------------------------

        self.categories["404"] = MarporCategory(
            code="404",
            label="Economic Planning",
            description="State economic planning, industrial policy, government-led development",
            tendency="authoritarian",
            weight=0.9,
            primary_keywords=[
                "economic planning", "planned economy", "central planning",
                "economic justice", "economic intervention", "industrial policy",
                "government investment", "public investment",
                "economic development", "rebuild economy",
                "government planning", "state-led", "state-led development", "directed economy",
            ],
            secondary_keywords=[
                "economic coordination", "strategic planning", "state guidance", "government intervention",
                "public spending", "national development plan", "planning commission",
            ],
            semantic_positive=[
                "Government-led economic planning ensures balanced development.",
                "Industrial policy guides strategic sectors for national benefit.",
                "Central planning coordinates economic activity efficiently.",
                "State direction of the economy prevents market failures.",
                "Strategic planning is essential for long-term development.",
                "We need economic justice and government investment to rebuild the economy.",
            ],
        )

        self.categories["412"] = MarporCategory(
            code="412",
            label="Controlled Economy",
            description="State control of economy, price controls, heavy regulation",
            tendency="authoritarian",
            weight=0.9,
            primary_keywords=["state control", "controlled economy", "price controls", "wage controls", "government regulation", "economic control"],
            secondary_keywords=["regulatory control", "state intervention", "managed economy", "price regulation", "economic regulation"],
            semantic_positive=[
                "The state must control key prices to protect consumers.",
                "Government control of the economy ensures fair outcomes.",
                "Price controls prevent exploitation by corporations.",
                "Strong economic regulation protects public interest.",
                "Market failures require government intervention."
            ],
        )

        self.categories["413"] = MarporCategory(
            code="413",
            label="Nationalization",
            description="State ownership of industry, public ownership",
            tendency="authoritarian",
            weight=0.9,
            primary_keywords=["nationalization", "state ownership", "public ownership", "nationalize industry", "government ownership"],
            secondary_keywords=["public control", "state enterprise", "national ownership", "socialization", "collective ownership"],
            semantic_positive=[
                "Key industries should be nationalized for public benefit.",
                "State ownership ensures essential services are affordable.",
                "Public control of utilities protects consumers from exploitation.",
                "Nationalized industries serve the national interest.",
                "Strategic sectors must be under public ownership."
            ],
        )

        self.categories["504"] = MarporCategory(
            code="504",
            label="Welfare State Expansion",
            description="Expand welfare/social programs, universal services",
            tendency="authoritarian",
            weight=0.9,
            primary_keywords=[
                "welfare", "welfare state", "welfare expansion", "expand welfare",
                "social programs", "safety net", "entitlement", "public assistance", "social services",
                "healthcare is a right", "universal healthcare", "healthcare is a right, not a privilege",
                "expand obamacare", "obamacare expansion", "affordable care act", "aca",
                "medicare for all", "medicaid expansion", "social security",
                "unemployment benefits", "food stamps", "snap", "wic",
                "public services", "universal services",
            ],
            secondary_keywords=[
                "social safety net", "universal provision", "public provision", "welfare benefits",
                "child benefit", "family allowance", "income support",
            ],
            semantic_positive=[
                "We must expand social programs to protect the vulnerable.",
                "Universal healthcare is a right, not a privilege.",
                "A strong safety net ensures no one falls through the cracks.",
                "Government should provide essential services to all.",
                "Social programs are investments in human capital.",
                "We should expand Obamacare and strengthen the safety net.",
            ],
        )

        self.categories["701"] = MarporCategory(
            code="701",
            label="Labour Groups",
            description="Pro-union, collective bargaining, workers' rights",
            tendency="authoritarian",
            weight=0.8,
            primary_keywords=[
                "unions", "labor unions", "trade unions", "collective bargaining",
                "workers' rights", "worker rights", "union rights", "organized labor",
                "unions built", "union workers", "union membership",
                "right to organize", "strike", "picket line",
                "labor movement", "workers solidarity", "worker solidarity",
                "essential workers", "working families", "middle class workers",
            ],
            secondary_keywords=[
                "labor organization", "union protection", "collective rights", "labor strength",
                "shop steward", "union contract", "collective agreement",
            ],
            semantic_positive=[
                "Strong unions protect workers from exploitation.",
                "Collective bargaining ensures fair wages and conditions.",
                "Worker solidarity is essential for economic justice.",
                "The right to organize must be protected and expanded.",
                "Unions built the middle class and remain essential."
            ],
        )

        self.categories["PROT"] = MarporCategory(
            code="PROT",
            label="Protectionism",
            description="Tariffs, trade barriers, economic nationalism",
            tendency="authoritarian",
            weight=0.85,
            primary_keywords=["tariffs", "protectionism", "trade barriers", "buy american", "america first", "economic nationalism", "protect jobs"],
            secondary_keywords=["trade protection", "domestic industry", "import restrictions", "trade defense", "local production"],
            semantic_positive=[
                "Tariffs protect workers and industries.",
                "We need trade barriers to bring manufacturing back home.",
                "Buy local policies support our domestic economy.",
                "Protectionism is necessary for national economic security.",
                "Free trade has devastated our manufacturing base."
            ],
        )

        self.categories["POP"] = MarporCategory(
            code="POP",
            label="Populism",
            description="People vs elite, anti-establishment",
            tendency="authoritarian",
            weight=0.75,
            primary_keywords=["the people", "ordinary citizens", "elites", "establishment", "drain the swamp", "corrupt elite", "people's will"],
            secondary_keywords=["common people", "working people", "elite corruption", "anti-establishment", "popular will"],
            semantic_positive=[
                "This is a movement of ordinary people against the political establishment.",
                "We're taking power back from the elites and giving it to the people.",
                "The silent majority has been ignored for too long.",
                "We need to drain the swamp of corruption.",
                "Elites have rigged the system against working people."
            ],
            requires_context=["elite", "elites", "establishment", "corrupt", "rigged", "swamp"],
            requires_context_hard=True,
        )

        self.categories["ENV_AUTH"] = MarporCategory(
            code="ENV_AUTH",
            label="Environmental Authoritarianism",
            description="Strong state environmental regulation, climate emergency measures",
            tendency="authoritarian",
            weight=0.7,
            primary_keywords=["climate emergency", "environmental regulation", "green new deal", "climate action", "mandatory reduction", "environmental mandate"],
            secondary_keywords=["climate mandate", "environmental enforcement", "green policy", "climate compliance", "environmental control"],
            semantic_positive=[
                "We need strong government action to address the climate emergency.",
                "Environmental regulations must be strictly enforced.",
                "The state must lead the transition to a green economy.",
                "Climate action requires top-down coordination and control.",
                "Only government has the power to enforce climate compliance."
            ],
        )

        # ==================== CENTRIST DISCOURSE (NOT IDEOLOGY) ====================

        self.categories["CENT"] = MarporCategory(
            code="CENT",
            label="Centrist/Moderate",
            description="Balanced approach, pragmatism, moderation, compromise",
            tendency="centrist",
            weight=0.6,
            primary_keywords=["balance", "moderation", "pragmatism", "compromise", "middle ground", "moderate approach", "balanced solution"],
            secondary_keywords=["practical solution", "centrist", "moderate position", "balanced policy", "pragmatic approach"],
            semantic_positive=[
                "We need a balanced approach that considers all perspectives.",
                "Practical solutions are more important than ideological purity.",
                "Finding common ground benefits everyone.",
                "Moderation and compromise are essential in politics.",
                "Extreme positions on either side are counterproductive."
            ],
        )

        self.categories["BI"] = MarporCategory(
            code="BI",
            label="Bipartisan/Cross-party",
            description="Cross-party cooperation, bipartisan solutions",
            tendency="centrist",
            weight=0.7,
            primary_keywords=["bipartisan", "cross-party", "working together", "cooperation", "unity", "both sides", "across the aisle"],
            secondary_keywords=["collaboration", "joint effort", "unified approach", "coalition", "partnership"],
            semantic_positive=[
                "We need bipartisan solutions to solve our problems.",
                "Working across the aisle produces better results.",
                "Cooperation between parties is essential for progress.",
                "Unity and collaboration benefit the entire nation.",
                "Both sides must come together for the common good."
            ],
        )

        self.categories["REF"] = MarporCategory(
            code="REF",
            label="Reform/Improvement",
            description="Incremental reform, improvement, fixing systems",
            tendency="centrist",
            weight=0.7,
            primary_keywords=["reform", "improve", "fix", "better", "progress", "modernize", "update", "enhance"],
            secondary_keywords=["improvement", "refinement", "upgrading", "enhancement", "optimization", "strengthening"],
            semantic_positive=[
                "We need to reform and improve existing systems.",
                "Incremental progress is better than radical change.",
                "Fixing what's broken benefits everyone.",
                "Steady improvement leads to lasting results.",
                "Reform is essential for maintaining relevance."
            ],
        )

    # -------------------------------------------------------------------------
    # PATTERN COMPILATION
    # -------------------------------------------------------------------------

    def _compile_patterns(self) -> None:
        self.patterns = {}
        for code, category in self.categories.items():
            pats: List[Tuple[str, re.Pattern]] = []
            for keyword in category.get_all_keywords():
                escaped = re.escape(keyword.lower())
                if " " in keyword:
                    escaped = r"\s+".join(map(re.escape, keyword.lower().split()))
                pattern_str = r"\b" + escaped + r"\b"
                try:
                    pats.append((keyword, re.compile(pattern_str, re.IGNORECASE)))
                except re.error as e:
                    logger.warning("Failed to compile pattern for '%s': %s", keyword, e)
            self.patterns[code] = pats

    # -------------------------------------------------------------------------
    # SEMANTIC PRECOMPUTATION
    # -------------------------------------------------------------------------

    def _precompute_semantic_embeddings(self) -> None:
        self._sem_pos_emb.clear()
        self._sem_neg_emb.clear()
        self._sem_pos_texts.clear()
        self._sem_neg_texts.clear()

        if self.semantic.model is None:
            return

        for code, cat in self.categories.items():
            pos = [t.strip() for t in (cat.semantic_positive or []) if t and t.strip()]
            neg = [t.strip() for t in (cat.semantic_negative or []) if t and t.strip()]

            self._sem_pos_texts[code] = pos
            self._sem_neg_texts[code] = neg

            if pos:
                emb = self.semantic.encode_many(pos)
                if emb is not None and len(emb) == len(pos):
                    self._sem_pos_emb[code] = emb
            if neg:
                emb = self.semantic.encode_many(neg)
                if emb is not None and len(emb) == len(neg):
                    self._sem_neg_emb[code] = emb

    # -------------------------------------------------------------------------
    # BREAKDOWN (VALIDATED evidence only; no new evidence created)
    # -------------------------------------------------------------------------

    def _build_marpor_breakdown_from_validated(self, validated: List[Evidence]) -> Dict[str, Dict[str, Any]]:
        if not validated:
            return {}

        by_code: DefaultDict[str, List[Evidence]] = defaultdict(list)
        for ev in validated:
            by_code[str(ev.code)].append(ev)

        total_strength = 0.0
        strength_by_code: Dict[str, float] = {}
        for code, evs in by_code.items():
            s = sum(float(e.strength) for e in evs)
            strength_by_code[code] = s
            total_strength += s

        out: Dict[str, Dict[str, Any]] = {}
        for code, evs in by_code.items():
            cat = self.categories.get(code)
            if not cat:
                continue

            strength = float(strength_by_code.get(code, 0.0))
            pct = (strength / total_strength * 100.0) if total_strength > 0 else 0.0

            polarity = {"support": 0, "oppose": 0, "centrist": 0}
            ev_conf: List[float] = []
            for e in evs:
                if e.polarity > 0:
                    polarity["support"] += 1
                elif e.polarity < 0:
                    polarity["oppose"] += 1
                else:
                    polarity["centrist"] += 1
                try:
                    ev_conf.append(float((e.evidence_confidence or {}).get("evidence_confidence", 0.0)))
                except Exception:
                    pass

            avg_ev_conf = (sum(ev_conf) / len(ev_conf)) if ev_conf else 0.0
            avg_strength = (sum(float(e.strength) for e in evs) / max(1, len(evs)))

            out[code] = {
                "code": code,
                "label": cat.label,
                "description": cat.description,
                "tendency": cat.tendency,
                "weight": float(cat.weight),
                "percentage": round(float(pct), 2),
                "match_count": int(len(evs)),
                "avg_strength": round(float(avg_strength), 3),
                "avg_evidence_confidence": round(float(avg_ev_conf), 3),
                "polarity": polarity,
                "evidence_strength": round(float(strength), 6),
            }

        return dict(sorted(out.items(), key=lambda kv: kv[1]["percentage"], reverse=True))

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def classify_text_detailed(self, text: str, use_semantic: bool = True, threshold: float = 0.6) -> Dict[str, Any]:
        # threshold is accepted for compatibility; gating is internal and stable.
        text = (text or "").strip()
        if not text:
            return self._empty_result()

        total_chars = len(text)

        try:
            evidence, lexical_max_strength = self._find_all_evidence(text, use_semantic=use_semantic, total_chars=total_chars)
        except Exception as e:
            logger.error("Evidence finding failed: %s", e, exc_info=True)
            return self._empty_result()

        try:
            validated_all = self._validate_evidence_with_context(text, evidence, total_chars=total_chars)
        except Exception as e:
            logger.error("Evidence validation failed: %s", e, exc_info=True)
            validated_all = []

        if not validated_all:
            return self._empty_result()

        # Split validated evidence into ideological vs centrist discourse
        validated_ideol: List[Evidence] = []
        validated_centrist: List[Evidence] = []
        for ev in validated_all:
            cat = self.categories.get(ev.code)
            if not cat:
                continue
            if cat.tendency == "centrist":
                validated_centrist.append(ev)
            else:
                validated_ideol.append(ev)

        # If no ideological evidence exists, classify as Centrist
        if not validated_ideol:
            centrist_breakdown = self._build_marpor_breakdown_from_validated(validated_centrist)
            centrist_strength = float(sum(float(e.strength) for e in validated_centrist))
            return self._centrist_result(validated_centrist, centrist_breakdown, centrist_strength)

        # Compute ideological strengths (Lib/Auth only)
        code_strengths: DefaultDict[str, float] = defaultdict(float)
        lib_strength = 0.0
        auth_strength = 0.0
        centrist_strength = float(sum(float(e.strength) for e in validated_centrist))

        for ev in validated_ideol:
            cat = self.categories.get(ev.code)
            if not cat:
                continue

            # polarity weighting for code_strengths bookkeeping
            if ev.polarity > 0:
                pol_w = 1.0
            elif ev.polarity == 0:
                pol_w = 0.7
            else:
                pol_w = 0.4
            code_strengths[ev.code] += float(ev.strength) * pol_w

            # family strength attribution (ideological only)
            if cat.tendency == "libertarian":
                if ev.polarity > 0:
                    lib_strength += float(ev.strength)
                elif ev.polarity < 0:
                    auth_strength += float(ev.strength) * 0.8
                else:
                    lib_strength += float(ev.strength) * 0.35
                    auth_strength += float(ev.strength) * 0.10

            elif cat.tendency == "authoritarian":
                if ev.polarity > 0:
                    auth_strength += float(ev.strength)
                elif ev.polarity < 0:
                    lib_strength += float(ev.strength) * 0.8
                else:
                    auth_strength += float(ev.strength) * 0.35
                    lib_strength += float(ev.strength) * 0.10

        ideol_total_strength = lib_strength + auth_strength

        dominance = 0.0
        if ideol_total_strength > 0:
            dominance = max(lib_strength, auth_strength) / ideol_total_strength

        # Family decision (can still be Centrist if ideologically ambiguous / weak)
        family = CENTRIST_FAMILY
        if ideol_total_strength > 0:
            margin = 0.12
            lib_ratio = lib_strength / ideol_total_strength
            auth_ratio = auth_strength / ideol_total_strength
            if lib_ratio > auth_ratio + margin:
                family = LIB_FAMILY
            elif auth_ratio > lib_ratio + margin:
                family = AUTH_FAMILY
            else:
                family = CENTRIST_FAMILY

        evidence_count = len(validated_ideol)
        support_evidence_count = sum(1 for ev in validated_ideol if ev.polarity > 0)
        topic_count = len(set(ev.code for ev in validated_ideol))
        marpor_codes = sorted(set(ev.code for ev in validated_ideol))

        is_ev = self._is_ideology_evidence(
            evidence_count=evidence_count,
            topic_count=topic_count,
            total_strength=ideol_total_strength,
            dominance=dominance,
            family=family,
        )

        # If gate fails, classify as Centrist (critical to block slogans / weak rhetoric)
        if not is_ev:
            centrist_breakdown = self._build_marpor_breakdown_from_validated(validated_centrist)
            return self._centrist_result(validated_centrist, centrist_breakdown, centrist_strength)

        subtype: Optional[str] = None
        if family in (LIB_FAMILY, AUTH_FAMILY):
            subtype = self._determine_subtype_evidence_level(dict(code_strengths), family)

        pattern_conf: Dict[str, Any] = {}
        if subtype:
            subtype_codes = IDEOLOGY_SUBTYPES.get(subtype, [])
            if subtype_codes:
                pattern_conf = self.conf_calc.calculate_pattern_confidence_single(marpor_codes, subtype_codes)

        avg_ev_conf = (
            sum(float(ev.evidence_confidence.get("evidence_confidence", 0.0)) for ev in validated_ideol)
            / max(1, len(validated_ideol))
        )
        confidence_score = float(min(1.0, max(0.0, avg_ev_conf * (0.6 + 0.4 * dominance))))

        signal_strength = float((1.0 - math.exp(-max(0.0, ideol_total_strength))) * 100.0)
        signal_strength = float(min(100.0, max(0.0, signal_strength)))

        total_for_scores = lib_strength + auth_strength + centrist_strength
        if total_for_scores > 0:
            scores = {
                LIB_FAMILY: round((lib_strength / total_for_scores) * 100.0, 2),
                AUTH_FAMILY: round((auth_strength / total_for_scores) * 100.0, 2),
                CENTRIST_FAMILY: round((centrist_strength / total_for_scores) * 100.0, 2),
            }
        else:
            scores = {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, CENTRIST_FAMILY: 100.0}

        marpor_breakdown = self._build_marpor_breakdown_from_validated(validated_ideol)
        centrist_breakdown = self._build_marpor_breakdown_from_validated(validated_centrist)

        return {
            "ideology_family": family,
            "ideology_subtype": subtype,  # None for Centrist

            "scores": scores,

            # ideological evidence only
            "evidence": self._serialize_evidence(validated_ideol),
            "evidence_count": int(evidence_count),
            "support_evidence_count": int(support_evidence_count),
            "marpor_codes": marpor_codes,

            "marpor_breakdown": marpor_breakdown,
            "marpor_code_analysis": marpor_breakdown,

            "total_evidence_strength": float(ideol_total_strength),
            "libertarian_strength": float(lib_strength),
            "authoritarian_strength": float(auth_strength),
            "centrist_strength": float(centrist_strength),
            "dominance": float(dominance),

            "is_ideology_evidence": True,
            "filtered_topic_count": int(topic_count),

            "confidence_score": float(confidence_score),
            "signal_strength": float(signal_strength),

            # diagnostics: centrist discourse evidence (not ideology)
            "centrist_evidence": self._serialize_evidence(validated_centrist),
            "centrist_evidence_count": int(len(validated_centrist)),
            "centrist_marpor_codes": sorted(set(ev.code for ev in validated_centrist)),
            "centrist_marpor_breakdown": centrist_breakdown,

            "evidence_level_confidence": {
                "avg_evidence_confidence": round(float(avg_ev_conf), 4),
                "pattern_confidence": pattern_conf,
                "evidence_quality": self._analyze_evidence_quality(validated_ideol),
            },

            "code_strengths": dict(code_strengths),
            "analysis_level": "evidence_only",
            "timestamp": datetime.now().isoformat(),
        }

    # -------------------------------------------------------------------------
    # IDEOLOGICAL EVIDENCE GATE (Centrist never passes)
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_ideology_evidence(*, evidence_count: int, topic_count: int, total_strength: float, dominance: float, family: str) -> bool:
        if family == CENTRIST_FAMILY:
            return False
        if evidence_count < EVIDENCE_MIN_EVIDENCE_COUNT:
            return False
        if topic_count < EVIDENCE_MIN_TOPIC_COUNT:
            return False
        if total_strength < EVIDENCE_MIN_TOTAL_STRENGTH:
            return False
        if dominance < EVIDENCE_MIN_DOMINANCE:
            return False
        return True

    def _centrist_result(
        self,
        validated_centrist: List[Evidence],
        centrist_breakdown: Dict[str, Any],
        centrist_strength: float,
    ) -> Dict[str, Any]:
        return {
            "ideology_family": CENTRIST_FAMILY,
            "ideology_subtype": None,
            "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, CENTRIST_FAMILY: 100.0},

            "evidence": [],
            "evidence_count": 0,
            "support_evidence_count": 0,
            "marpor_codes": [],

            "marpor_breakdown": {},
            "marpor_code_analysis": {},

            "total_evidence_strength": 0.0,
            "libertarian_strength": 0.0,
            "authoritarian_strength": 0.0,
            "centrist_strength": float(centrist_strength),
            "dominance": 0.0,

            "is_ideology_evidence": False,
            "filtered_topic_count": 0,
            "confidence_score": 0.0,
            "signal_strength": 0.0,

            "centrist_evidence": self._serialize_evidence(validated_centrist),
            "centrist_evidence_count": int(len(validated_centrist)),
            "centrist_marpor_codes": sorted(set(ev.code for ev in validated_centrist)),
            "centrist_marpor_breakdown": centrist_breakdown,

            "evidence_level_confidence": {
                "avg_evidence_confidence": 0.0,
                "pattern_confidence": {},
                "evidence_quality": self._analyze_evidence_quality(validated_centrist),
            },

            "code_strengths": {},
            "analysis_level": "evidence_only",
            "timestamp": datetime.now().isoformat(),
        }

    # -------------------------------------------------------------------------
    # EVIDENCE FINDING (lexical first, semantic selectively)
    # -------------------------------------------------------------------------

    def _find_all_evidence(
        self,
        text: str,
        *,
        use_semantic: bool,
        total_chars: int,
    ) -> Tuple[List[Evidence], Dict[str, float]]:
        evidence: List[Evidence] = []
        text_lower = text.lower()

        lexical_max_strength: Dict[str, float] = defaultdict(float)

        # Lexical hits
        for code, patterns in self.patterns.items():
            cat = self.categories[code]
            for keyword, pattern in patterns:
                try:
                    for m in pattern.finditer(text_lower):
                        # anti-pattern filter
                        if cat.anti_patterns:
                            window = text_lower[m.start(): min(len(text_lower), m.end() + 50)]
                            if any((anti and anti.lower() in window) for anti in cat.anti_patterns):
                                continue

                        match_type = "lexical_primary" if keyword in cat.primary_keywords else "lexical_secondary"
                        base = self.weights.get_evidence_quality(match_type)
                        lexical_max_strength[code] = max(float(lexical_max_strength[code]), float(base))

                        evidence.append(
                            Evidence(
                                code=code,
                                matched_text=m.group(),
                                span=(m.start(), m.end()),
                                match_type=match_type,
                                base_strength=float(base),
                                strength=float(base),
                                polarity=0,
                                polarity_reason="pending",
                                inside_quotes=self._is_inside_quotes(text, m.start()),
                            )
                        )
                except Exception as e:
                    logger.warning("Pattern matching failed for %s/%s: %s", code, keyword, e)

        # Semantic hits (selective)
        if not use_semantic or self.semantic.model is None:
            return evidence, dict(lexical_max_strength)

        text_emb = self.semantic.encode_one(text)
        if text_emb is None:
            return evidence, dict(lexical_max_strength)

        for code, cat in self.categories.items():
            if float(lexical_max_strength.get(code, 0.0)) >= SEMANTIC_LEXICAL_OVERRIDE:
                continue

            pos_emb = self._sem_pos_emb.get(code)
            neg_emb = self._sem_neg_emb.get(code)
            pos_texts = self._sem_pos_texts.get(code, [])
            neg_texts = self._sem_neg_texts.get(code, [])

            if (pos_emb is None or len(pos_texts) == 0) and (neg_emb is None or len(neg_texts) == 0):
                continue

            best_pos = None
            second_pos = None
            best_pos_text = None
            if pos_emb is not None and pos_emb.shape[0] > 0:
                sims = self.semantic.cosine_similarity_matrix(text_emb, pos_emb)
                if sims.size > 0:
                    topk_idx = np.argsort(sims)[::-1][: max(1, min(SEMANTIC_TOPK, sims.size))]
                    topk_vals = sims[topk_idx]
                    best_pos = float(topk_vals[0])
                    second_pos = float(topk_vals[1]) if len(topk_vals) > 1 else float(-1.0)
                    best_pos_text = pos_texts[int(topk_idx[0])] if int(topk_idx[0]) < len(pos_texts) else None

            best_neg = None
            second_neg = None
            best_neg_text = None
            if neg_emb is not None and neg_emb.shape[0] > 0:
                sims = self.semantic.cosine_similarity_matrix(text_emb, neg_emb)
                if sims.size > 0:
                    topk_idx = np.argsort(sims)[::-1][: max(1, min(SEMANTIC_TOPK, sims.size))]
                    topk_vals = sims[topk_idx]
                    best_neg = float(topk_vals[0])
                    second_neg = float(topk_vals[1]) if len(topk_vals) > 1 else float(-1.0)
                    best_neg_text = neg_texts[int(topk_idx[0])] if int(topk_idx[0]) < len(neg_texts) else None

            chosen_polarity = 0
            chosen_sim = None
            chosen_example = None
            chosen_reason = None

            if best_neg is not None and best_neg >= SEMANTIC_NEG_THRESHOLD and (best_neg - float(second_neg or -1.0)) >= SEMANTIC_MARGIN:
                if best_pos is None or best_neg >= (best_pos + SEMANTIC_MARGIN):
                    chosen_polarity = -1
                    chosen_sim = best_neg
                    chosen_example = best_neg_text
                    chosen_reason = "semantic_oppose"

            if chosen_polarity == 0 and best_pos is not None and best_pos >= SEMANTIC_THRESHOLD and (best_pos - float(second_pos or -1.0)) >= SEMANTIC_MARGIN:
                chosen_polarity = 1
                chosen_sim = best_pos
                chosen_example = best_pos_text
                chosen_reason = "semantic_support"

            if chosen_polarity == 0 or chosen_sim is None:
                continue

            base_q = self.weights.get_evidence_quality("semantic", chosen_sim)
            strength = float(base_q) * float(chosen_sim)

            evidence.append(
                Evidence(
                    code=code,
                    matched_text=f"[Semantic: {(chosen_example or '')[:70]}...]",
                    span=(0, len(text)),
                    match_type="semantic",
                    base_strength=float(base_q),
                    strength=float(strength),
                    polarity=int(chosen_polarity),
                    polarity_reason=str(chosen_reason),
                    inside_quotes=False,
                    semantic_similarity=float(chosen_sim),
                )
            )

        return evidence, dict(lexical_max_strength)

    # -------------------------------------------------------------------------
    # EVIDENCE VALIDATION (context + polarity + confidence)
    # -------------------------------------------------------------------------

    def _validate_evidence_with_context(self, text: str, evidence: List[Evidence], *, total_chars: int) -> List[Evidence]:
        text_lower = text.lower()
        validated: List[Evidence] = []

        for ev in evidence:
            cat = self.categories.get(ev.code)
            if not cat:
                continue

            out = Evidence(**ev.__dict__)

            # Context validation
            if cat.requires_context:
                hits = [c for c in cat.requires_context if c and c.lower() in text_lower]
                misses = [c for c in cat.requires_context if c and c.lower() not in text_lower]
                out.context_hits = hits
                out.context_misses = misses

                # HARD context: discard lexical hits with zero context hits
                if cat.requires_context_hard and out.match_type.startswith("lexical") and not hits:
                    continue

                if hits:
                    out.strength *= self.weights.CONTEXT_REQUIRED_MET
                if misses:
                    out.strength *= self.weights.CONTEXT_REQUIRED_MISSED

            if cat.excludes_context:
                for excluded in cat.excludes_context:
                    if excluded and excluded.lower() in text_lower:
                        out.strength *= self.weights.CONTEXT_EXCLUDED_PRESENT
                        out.context_misses.append(f"excluded:{excluded}")
                        break

            # Polarity detection (if not already assigned by semantic)
            if out.polarity == 0:
                pol, reason = self._detect_polarity(text, out)
                out.polarity = int(pol)
                out.polarity_reason = str(reason)

            # Polarity strength modifier
            if out.polarity < 0:
                out.strength *= self.weights.OPPOSE_STRENGTH
            elif out.polarity == 0:
                out.strength *= self.weights.CENTRIST_STRENGTH
            else:
                out.strength *= self.weights.SUPPORT_STRENGTH

            # Quote modifier + category weight
            out.strength *= (self.weights.INSIDE_QUOTES if out.inside_quotes else self.weights.OUTSIDE_QUOTES)
            out.strength *= float(cat.weight)

            out.strength = float(min(1.0, max(0.0, out.strength)))
            if out.strength <= MIN_EVIDENCE_STRENGTH_KEEP:
                continue

            out.evidence_confidence = self.conf_calc.calculate_evidence_confidence(
                match_type=out.match_type,
                similarity=out.semantic_similarity,
                context_hits=len(out.context_hits),
                context_misses=len(out.context_misses),
                inside_quotes=out.inside_quotes,
                polarity=out.polarity,
                char_position=int(out.span[0]),
                total_chars=int(total_chars),
            )

            validated.append(out)

        return validated

    def _detect_polarity(self, text: str, evidence: Evidence) -> Tuple[int, str]:
        text_lower = text.lower()
        window = 100
        start = max(0, evidence.span[0] - window)
        end = min(len(text), evidence.span[1] + window)
        ctx = text_lower[start:end]

        for pat in self.opposition_patterns:
            if pat.search(ctx):
                return -1, f"opposition:{pat.pattern[:24]}"

        for pat in self.negation_patterns:
            for m in pat.finditer(ctx):
                if abs(m.start() - (evidence.span[0] - start)) < 80:
                    return -1, f"negation:{pat.pattern[:24]}"

        for pat in self.endorsement_patterns:
            if pat.search(ctx):
                return 1, f"support:{pat.pattern[:24]}"

        if evidence.match_type == "semantic" and evidence.polarity != 0:
            return evidence.polarity, evidence.polarity_reason or "semantic_assigned"

        return 0, "centrist_or_ambiguous"

    @staticmethod
    def _is_inside_quotes(text: str, position: int) -> bool:
        try:
            before = text[:position]
            q = before.count('"')
            return (q % 2) == 1
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # SUBTYPE DETERMINATION (IDEOLOGICAL ONLY)
    # -------------------------------------------------------------------------

    def _determine_subtype_evidence_level(self, code_strengths: Dict[str, float], family: str) -> str:
        fam_l = (family or "").strip().lower()
        if fam_l not in ("libertarian", "authoritarian"):
            return family or CENTRIST_FAMILY

        def belongs(subtype_name: str) -> bool:
            s = (subtype_name or "").lower()
            return ("libertarian" in s) if fam_l == "libertarian" else ("authoritarian" in s)

        scores: Dict[str, float] = {}
        for subtype, codes in IDEOLOGY_SUBTYPES.items():
            if not belongs(subtype):
                continue
            scores[subtype] = sum(float(code_strengths.get(c, 0.0)) for c in codes)

        return max(scores.items(), key=lambda x: x[1])[0] if scores else (family or CENTRIST_FAMILY)

    # -------------------------------------------------------------------------
    # QUALITY / SERIALIZATION / EMPTY
    # -------------------------------------------------------------------------

    def _analyze_evidence_quality(self, evidence: List[Evidence]) -> Dict[str, Any]:
        if not evidence:
            return {
                "total_evidence": 0,
                "lexical_primary_count": 0,
                "lexical_secondary_count": 0,
                "semantic_count": 0,
                "avg_semantic_similarity": 0.0,
                "evidence_diversity": 0.0,
                "quality_score": 0.0,
            }

        lexical_primary = sum(1 for e in evidence if e.match_type == "lexical_primary")
        lexical_secondary = sum(1 for e in evidence if e.match_type == "lexical_secondary")
        semantic = [e for e in evidence if e.match_type == "semantic" and e.semantic_similarity is not None]

        avg_sem = float(sum(float(e.semantic_similarity or 0.0) for e in semantic) / len(semantic)) if semantic else 0.0
        unique_codes = len(set(e.code for e in evidence))
        diversity = unique_codes / len(evidence) if evidence else 0.0

        quality = (lexical_primary * 1.0 + lexical_secondary * 0.7 + len(semantic) * 0.8) / max(1, len(evidence))

        return {
            "total_evidence": len(evidence),
            "lexical_primary_count": lexical_primary,
            "lexical_secondary_count": lexical_secondary,
            "semantic_count": len(semantic),
            "avg_semantic_similarity": round(avg_sem, 3),
            "evidence_diversity": round(diversity, 3),
            "quality_score": round(float(quality), 3),
        }

    @staticmethod
    def _serialize_evidence(evidence: List[Evidence]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ev in evidence:
            out.append({
                "code": ev.code,
                "matched_text": ev.matched_text,
                "match_type": ev.match_type,
                "strength": round(float(ev.strength), 3),
                "polarity": int(ev.polarity),
                "polarity_reason": ev.polarity_reason,
                "inside_quotes": bool(ev.inside_quotes),
                "semantic_similarity": round(float(ev.semantic_similarity), 3) if ev.semantic_similarity is not None else None,
                "context_hits": list(ev.context_hits),
                "context_misses": list(ev.context_misses),
                "evidence_confidence": dict(ev.evidence_confidence or {}),
            })
        return out

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "ideology_family": CENTRIST_FAMILY,
            "ideology_subtype": None,
            "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, CENTRIST_FAMILY: 100.0},

            "evidence": [],
            "evidence_count": 0,
            "support_evidence_count": 0,
            "marpor_codes": [],

            "marpor_breakdown": {},
            "marpor_code_analysis": {},

            "centrist_evidence": [],
            "centrist_evidence_count": 0,
            "centrist_marpor_codes": [],
            "centrist_marpor_breakdown": {},

            "total_evidence_strength": 0.0,
            "libertarian_strength": 0.0,
            "authoritarian_strength": 0.0,
            "centrist_strength": 0.0,
            "dominance": 0.0,

            "is_ideology_evidence": False,
            "filtered_topic_count": 0,
            "confidence_score": 0.0,
            "signal_strength": 0.0,

            "evidence_level_confidence": {
                "avg_evidence_confidence": 0.0,
                "pattern_confidence": {},
                "evidence_quality": {"total_evidence": 0},
            },

            "code_strengths": {},
            "analysis_level": "evidence_only",
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

hybrid_marpor_analyzer = HybridMarporAnalyzer()

__all__ = [
    "HybridMarporAnalyzer",
    "hybrid_marpor_analyzer",
    "EvidenceConfidence",
    "EvidenceWeight",
    "MarporCategory",
    "Evidence",
    "IDEOLOGY_SUBTYPES",
    "LIB_FAMILY",
    "AUTH_FAMILY",
    "CENTRIST_FAMILY",
]
