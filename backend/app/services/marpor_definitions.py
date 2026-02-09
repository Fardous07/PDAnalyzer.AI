# backend/app/services/marpor_definitions.py

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

LIB_FAMILY = "Libertarian"
AUTH_FAMILY = "Authoritarian"
ECON_LEFT = "Economic-Left"
ECON_RIGHT = "Economic-Right"
CENTRIST_FAMILY = "Centrist"

EVIDENCE_MIN_EVIDENCE_COUNT = 1
EVIDENCE_MIN_TOPIC_COUNT = 1
EVIDENCE_MIN_TOTAL_STRENGTH = 0.20
EVIDENCE_MIN_AXIS_DOMINANCE = 0.35

MIN_EVIDENCE_STRENGTH_KEEP = 0.20

AXIS_ATTR_SUPPORT_MULT = 1.0
AXIS_ATTR_NEUTRAL_MULT = 0.6
AXIS_ATTR_OPPOSE_MULT = 0.3

AXIS_MIN_TOTAL = 0.15

SEMANTIC_THRESHOLD = 0.70
SEMANTIC_NEG_THRESHOLD = 0.80
SEMANTIC_MARGIN = 0.06
SEMANTIC_TOPK = 5
SEMANTIC_LEXICAL_OVERRIDE = 0.75

HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.55

IDEOLOGY_SUBTYPES: Dict[str, List[str]] = {
    "Right-Libertarianism": ["401", "407", "414", "505", "702", "301", "507"],
    "Left-Libertarianism": ["201", "203", "SJ", "604", "607", "503", "501"],
    "Cultural Libertarianism": ["201", "604", "607", "503"],
    "Geo-Libertarianism": ["501", "401", "407", "301"],
    "Paleo-Libertarianism": ["401", "414", "505", "603", "601", "702"],
    "Right-Authoritarian": ["305", "605", "603", "601", "608", "302", "PROT", "POP"],
    "Left-Authoritarian": ["404", "412", "413", "504", "701", "ENV_AUTH", "SJ"],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _dominance(a: float, b: float) -> float:
    tot = a + b
    return (max(a, b) / tot) if tot > 0 else 0.0


def _axis_pol_mult(polarity: int) -> float:
    if polarity > 0:
        return AXIS_ATTR_SUPPORT_MULT
    if polarity < 0:
        return AXIS_ATTR_OPPOSE_MULT
    return AXIS_ATTR_NEUTRAL_MULT


def _axis_confidence(pos_mass: float, neg_mass: float) -> float:
    tot = pos_mass + neg_mass
    if tot <= 0:
        return 0.0
    dom = _dominance(pos_mass, neg_mass)
    mass_term = _clamp(tot / 2.0, 0.0, 1.0)
    return float(_clamp(mass_term * dom, 0.0, 1.0))


def _coord_from_masses(pos_mass: float, neg_mass: float) -> float:
    tot = pos_mass + neg_mass
    if tot <= 0:
        return 0.0
    return float((pos_mass - neg_mass) / tot)


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
    axis_min_total: float,
) -> Dict[str, str]:
    social_total = s_lib + s_auth
    econ_total = e_left + e_right

    social_dir = ""
    if social_total >= axis_min_total:
        social_dir = "Libertarian" if s_lib >= s_auth else "Authoritarian"

    econ_dir = ""
    if econ_total >= axis_min_total:
        econ_dir = "Right" if e_right >= e_left else "Left"

    return {"social": social_dir, "economic": econ_dir}


def _threshold_scale(threshold: float) -> float:
    t = float(threshold)
    t = _clamp(t, 0.10, 0.95)
    return float(_clamp(t / 0.60, 0.40, 1.60))


class EvidenceWeight:
    LEXICAL_PRIMARY: float = 1.0
    LEXICAL_SECONDARY: float = 0.7
    SEMANTIC_DIRECT: float = 0.9
    SEMANTIC_PARAPHRASE: float = 0.75
    SEMANTIC_WEAK: float = 0.6

    CONTEXT_REQUIRED_MET: float = 1.1
    CONTEXT_REQUIRED_MISSED: float = 0.7
    CONTEXT_EXCLUDED_PRESENT: float = 0.3
    INSIDE_QUOTES: float = 0.6
    OUTSIDE_QUOTES: float = 1.0

    SUPPORT_STRENGTH: float = 1.0
    OPPOSE_STRENGTH: float = 0.5
    NEUTRAL_STRENGTH: float = 0.7

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
            polarity_factor = EvidenceWeight.NEUTRAL_STRENGTH

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
            "evidence_confidence": round(float(evidence_conf), 4),
            "base_quality": round(float(base_quality), 4),
            "context_factor": round(float(context_factor), 4),
            "quote_factor": float(quote_factor),
            "polarity_factor": float(polarity_factor),
            "position_modifier": round(float(position_modifier), 4),
            "reliability_score": round(float(reliability), 4),
            "error_margin": round(float(error_margin), 4),
            "quality_tier": tier,
            "confidence_interval": (
                round(float(max(0.0, evidence_conf - error_margin)), 4),
                round(float(min(1.0, evidence_conf + error_margin)), 4),
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
            "pattern_coverage": round(float(coverage), 4),
            "pattern_specificity": round(float(specificity), 4),
            "pattern_confidence": round(float(pc), 4),
            "codes_present": int(len(present)),
            "codes_expected": int(expected),
        }


@dataclass
class MarporCategory:
    code: str
    label: str
    description: str
    tendency: str
    weight: float = 1.0
    social_tendency: float = 0.0
    economic_tendency: float = 0.0
    social_weight: float = 0.0
    economic_weight: float = 0.0
    subtype_weights: Dict[str, float] = field(default_factory=dict)
    primary_keywords: List[str] = field(default_factory=list)
    secondary_keywords: List[str] = field(default_factory=list)
    semantic_positive: List[str] = field(default_factory=list)
    semantic_negative: List[str] = field(default_factory=list)
    requires_context: List[str] = field(default_factory=list)
    excludes_context: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)
    requires_context_hard: bool = False
    subtype_group: str = "general"

    def get_all_keywords(self) -> List[str]:
        return list(self.primary_keywords) + list(self.secondary_keywords)


@dataclass
class Evidence:
    code: str
    matched_text: str
    span: Tuple[int, int]
    match_type: str
    base_strength: float
    strength: float
    polarity: int
    polarity_reason: str
    inside_quotes: bool = False
    semantic_similarity: Optional[float] = None
    context_hits: List[str] = field(default_factory=list)
    context_misses: List[str] = field(default_factory=list)
    evidence_confidence: Dict[str, Any] = field(default_factory=dict)
    attribution_hint: bool = False
    attribution_reason: Optional[str] = None


class SemanticService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Any = None

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        self.model = SentenceTransformer(self.model_name)

    def encode_many(self, texts: List[str]) -> Optional[np.ndarray]:
        self.ensure_loaded()
        if self.model is None or not texts:
            return None
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_one(self, text: str) -> Optional[np.ndarray]:
        out = self.encode_many([text])
        if out is None or len(out) == 0:
            return None
        return out[0]

    @staticmethod
    def cosine_similarity_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
        if vec is None or mat is None or getattr(mat, "size", 0) == 0:
            return np.zeros((0,), dtype=np.float32)

        v = vec.astype(np.float32)
        M = mat.astype(np.float32)

        v_norm = np.linalg.norm(v) or 1.0
        M_norm = np.linalg.norm(M, axis=1)
        M_norm = np.where(M_norm == 0.0, 1.0, M_norm)

        sims = (M @ v) / (M_norm * v_norm)
        return sims.astype(np.float32)


class HybridMarporAnalyzer:
    _CEREMONIAL_RE = re.compile(
        r"^\s*(thank you|thanks|god bless|applause|welcome|good (morning|afternoon|evening)|"
        r"ladies and gentlemen|my fellow|let me be clear|it is an honor)\b",
        re.IGNORECASE,
    )

    _ACTION_MARKER_RE = re.compile(
        r"\b("
        r"will|must|should|need(?:s)?\s+to|have\s+to|going\s+to|plan\s+to|intend\s+to|"
        r"ban|restrict|regulate|control|enforce|prosecute|punish|"
        r"protect|defend|guarantee|secure|"
        r"expand|strengthen|restore|repeal|pass|fund|invest|cut|raise"
        r")\b",
        re.IGNORECASE,
    )

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
        self._semantic_ready: bool = False

        self._init_all_categories()
        self._compile_patterns()

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

        self.attribution_patterns = [
            re.compile(r"\baccording to\b", re.IGNORECASE),
            re.compile(r"\bas (?:he|she|they)\s+(?:say|said|claim(?:ed)?|argue(?:d)?)\b", re.IGNORECASE),
            re.compile(r"\bhe (?:says|said|claims|claimed)\b", re.IGNORECASE),
            re.compile(r"\bshe (?:says|said|claims|claimed)\b", re.IGNORECASE),
            re.compile(r"\bthey (?:say|said|claim|claimed)\b", re.IGNORECASE),
            re.compile(r"\b(my opponent|our opponents|the opposition|the other side)\b", re.IGNORECASE),
            re.compile(r"\b(experts? say|studies? show|research(?:ers)? find)\b", re.IGNORECASE),
        ]

    def set_embedder(self, embedder: Any) -> None:
        if hasattr(embedder, "encode"):
            self.semantic.model = embedder
            self._semantic_ready = False

    def _init_all_categories(self) -> None:
        C = self.categories

        C["201"] = MarporCategory(
            code="201",
            label="Freedom and Human Rights",
            description="Civil liberties, human rights, free speech, individual freedom",
            tendency="libertarian",
            weight=1.0,
            social_tendency=+0.90,
            social_weight=1.0,
            primary_keywords=[
                "civil liberties", "human rights", "free speech", "freedom of speech",
                "civil rights", "individual rights", "freedom of expression",
                "privacy rights", "freedom of assembly", "freedom of religion",
                "freedom of the press", "press freedom", "free press",
                "voting rights", "right to vote", "election access", "voter access",
                "voter suppression", "protect the vote",
                "reproductive freedom", "reproductive rights", "abortion rights",
                "right to choose", "contraception", "birth control",
                "ivf", "in vitro fertilization", "pre-existing conditions", "preexisting conditions",
            ],
            secondary_keywords=[
                "liberty", "freedom", "rights", "autonomy", "self-determination",
                "due process rights", "bodily autonomy", "personal freedom",
                "civil liberties protections", "first amendment", "constitutional rights",
                "freedom to vote", "protect voting access",
                "privacy", "data privacy", "right to privacy",
            ],
            semantic_positive=[
                "We must protect civil liberties and fundamental rights.",
                "Free speech is essential in a democratic society.",
                "Freedom of the press must be protected from intimidation or retaliation.",
                "The state must not violate personal privacy without due process.",
                "Voting rights must be protected so every eligible citizen can vote.",
                "Reproductive freedom and bodily autonomy must be respected by the state.",
            ],
            semantic_negative=[
                "Some freedoms must be restricted for security and order.",
                "Free speech should be limited to prevent harmful ideas.",
                "Rights can be suspended when the state deems it necessary.",
                "News outlets that criticize leaders should face consequences.",
            ],
            requires_context=["rights", "libert", "speech", "privacy", "due process", "civil", "freedom", "vote", "press"],
            excludes_context=["applause", "thank you", "god bless", "welcome"],
            requires_context_hard=False,
        )

        C["203"] = MarporCategory(
            code="203",
            label="Constitutionalism and Rule of Law",
            description="Rule of law, constitutional limits, checks and balances",
            tendency="libertarian",
            weight=0.9,
            social_tendency=+0.75,
            social_weight=1.0,
            primary_keywords=[
                "rule of law", "constitutional", "constitution",
                "checks and balances", "separation of powers",
                "judicial independence", "due process",
                "constitutional limits", "legal safeguards",
                "free and fair elections", "peaceful transfer of power",
                "election integrity", "protect democracy", "democratic norms",
                "john lewis voting rights act", "voting rights act", "freedom to vote act",
            ],
            secondary_keywords=[
                "oversight", "accountability", "transparency",
                "independent courts", "procedural fairness",
                "anti-corruption", "government accountability", "constitutional order",
            ],
            semantic_positive=[
                "The constitution limits state power and protects liberty.",
                "No one is above the law; institutions must be accountable.",
                "Courts must be independent from political interference.",
                "Due process is required before depriving anyone of rights.",
                "Checks and balances prevent abuse of executive power.",
                "Free and fair elections and peaceful transfers of power must be protected.",
            ],
            semantic_negative=[
                "Courts should not block decisive government action.",
                "Legal constraints must not hinder public security measures.",
                "Elections should be controlled to ensure the right outcomes.",
            ],
            requires_context=["law", "constitution", "court", "judicial", "due process", "oversight", "election"],
            excludes_context=["applause", "thank you", "god bless"],
        )

        C["301"] = MarporCategory(
            code="301",
            label="Decentralization",
            description="Devolution, federalism, local control, subsidiarity",
            tendency="libertarian",
            weight=0.8,
            social_tendency=+0.45,
            social_weight=1.0,
            primary_keywords=[
                "decentralization", "decentralize", "devolution", "federalism",
                "local control", "regional autonomy", "states' rights", "subsidiarity",
            ],
            secondary_keywords=["local governance", "community control", "municipal authority", "local autonomy"],
            semantic_positive=[
                "Decisions should be made closer to local communities.",
                "Power should be devolved to regions and municipalities.",
                "Local control improves responsiveness and accountability.",
            ],
        )

        C["604"] = MarporCategory(
            code="604",
            label="Social Freedom",
            description="Lifestyle autonomy; opposition to moral policing",
            tendency="libertarian",
            weight=0.8,
            social_tendency=+0.75,
            social_weight=1.0,
            primary_keywords=[
                "social freedom", "lifestyle freedom", "personal autonomy",
                "private life", "freedom of conscience",
                "reproductive freedom", "reproductive rights",
                "abortion", "abortion rights", "birth control", "contraception",
                "ivf", "in vitro fertilization",
                "bodily autonomy",
                "marriage equality", "same-sex marriage",
            ],
            secondary_keywords=[
                "personal choice", "live as they choose", "moral policing", "individual lifestyle",
                "privacy", "private decisions",
            ],
            semantic_positive=[
                "The state should not police private moral choices.",
                "People should be free to live as they choose.",
                "Lifestyle decisions are not the government's business.",
                "Bodily autonomy and private medical choices should be respected.",
            ],
        )

        C["607"] = MarporCategory(
            code="607",
            label="Multiculturalism",
            description="Diversity, inclusion, pluralism, pro-minority rights",
            tendency="libertarian",
            weight=0.7,
            social_tendency=+0.55,
            social_weight=1.0,
            primary_keywords=["multiculturalism", "pluralism", "diversity", "inclusion"],
            secondary_keywords=["inclusive society", "cultural diversity", "plural society", "equal respect"],
            semantic_positive=[
                "Diversity and inclusion strengthen society.",
                "A pluralistic society respects different cultures and identities.",
                "Minorities deserve equal protection and equal rights.",
            ],
            requires_context=["divers", "inclus", "plural", "minority", "culture"],
        )

        C["401"] = MarporCategory(
            code="401",
            label="Free Enterprise",
            description="Markets, private sector, deregulation, entrepreneurship",
            tendency="libertarian",
            weight=1.0,
            economic_tendency=+0.95,
            economic_weight=1.0,
            primary_keywords=[
                "free enterprise", "free market", "market economy", "deregulation",
                "private sector", "entrepreneurship", "economic freedom",
                "small business", "job creators", "reduce red tape",
            ],
            secondary_keywords=["competition", "private investment", "business friendly", "reduce regulation"],
            semantic_positive=[
                "Free markets drive growth and innovation.",
                "The private sector creates jobs better than bureaucracy.",
                "Deregulation reduces barriers for entrepreneurs.",
                "Competition improves quality and lowers prices.",
            ],
            semantic_negative=[
                "Markets must be replaced by state control and planning.",
                "Private enterprise should be subordinated to government direction.",
            ],
            requires_context=["market", "private", "business", "dereg", "competition"],
        )

        C["407"] = MarporCategory(
            code="407",
            label="Free Trade",
            description="Open trade, reduce tariffs, trade liberalization",
            tendency="libertarian",
            weight=0.8,
            economic_tendency=+0.75,
            economic_weight=1.0,
            primary_keywords=["free trade", "trade liberalization", "open markets", "remove tariffs", "trade agreement"],
            secondary_keywords=["lower tariffs", "reduce trade barriers", "market access", "global trade"],
            semantic_positive=[
                "Open trade benefits consumers and increases efficiency.",
                "Reducing tariffs expands markets and lowers prices.",
                "Trade agreements can create new opportunities for exporters.",
            ],
            semantic_negative=[
                "We should raise tariffs and restrict imports to protect domestic industry.",
                "Trade barriers are necessary to stop foreign competition.",
            ],
            requires_context=["trade", "tariff", "import", "export", "agreement", "market"],
        )

        C["414"] = MarporCategory(
            code="414",
            label="Economic Orthodoxy",
            description="Tax cuts, balanced budgets, fiscal restraint",
            tendency="libertarian",
            weight=0.9,
            economic_tendency=+0.80,
            economic_weight=1.0,
            primary_keywords=[
                "tax cuts", "cut taxes", "balanced budget", "fiscal responsibility", "deficit reduction",
                "tax breaks", "tax relief", "middle class tax cut", "middle-class tax cut",
                "reduce spending", "spending restraint",
            ],
            secondary_keywords=["lower taxes", "spending restraint", "fiscal discipline", "budget balance"],
            semantic_positive=[
                "Lower taxes encourage investment and growth.",
                "Balanced budgets protect future generations from debt.",
                "Fiscal discipline is necessary to stabilize the economy.",
            ],
            semantic_negative=[
                "Deficits do not matter; we should spend without restraint.",
                "Taxes must rise substantially to expand government permanently.",
            ],
            requires_context=["tax", "budget", "deficit", "debt", "fiscal"],
        )

        C["505"] = MarporCategory(
            code="505",
            label="Welfare State Limitation",
            description="Limit welfare; emphasize work requirements and self-reliance",
            tendency="libertarian",
            weight=0.8,
            economic_tendency=+0.60,
            economic_weight=1.0,
            primary_keywords=["welfare reform", "work requirements", "limit welfare", "self-reliance", "personal responsibility"],
            secondary_keywords=["reduce benefits", "welfare dependency", "targeted assistance", "welfare-to-work"],
            semantic_positive=[
                "Welfare programs should encourage work and self-sufficiency.",
                "Work requirements reduce dependency and restore dignity.",
                "Benefits should be targeted to those truly in need.",
            ],
            semantic_negative=[
                "We should expand welfare benefits universally with no conditions.",
                "Government must provide permanent income support for everyone.",
            ],
            requires_context=["welfare", "benefit", "work", "assistance", "dependency"],
        )

        C["507"] = MarporCategory(
            code="507",
            label="School Choice / Education Limitation",
            description="Vouchers, charter schools, parental choice, competition",
            tendency="libertarian",
            weight=0.7,
            social_tendency=+0.20,
            social_weight=0.3,
            economic_tendency=+0.45,
            economic_weight=0.7,
            primary_keywords=["school choice", "vouchers", "charter schools", "parental choice"],
            secondary_keywords=["education freedom", "private schools", "education market", "competition in education"],
            semantic_positive=[
                "Parents should be able to choose the best school for their children.",
                "Vouchers can expand options for low-income families.",
                "Competition can improve school performance.",
            ],
            semantic_negative=[
                "School choice undermines public education and should be abolished.",
                "We should ban charter schools and vouchers.",
            ],
            requires_context=["school", "education", "voucher", "charter", "parent"],
        )

        C["702"] = MarporCategory(
            code="702",
            label="Anti-Union / Workplace Freedom",
            description="Right-to-work, limit compulsory unionism",
            tendency="libertarian",
            weight=0.7,
            social_tendency=+0.15,
            social_weight=0.3,
            economic_tendency=+0.40,
            economic_weight=0.7,
            primary_keywords=["right to work", "voluntary unionism", "union reform", "reduce union power"],
            secondary_keywords=["workplace freedom", "union accountability", "worker choice"],
            semantic_positive=[
                "Workers should choose whether to join a union.",
                "Compulsory union membership violates freedom of association.",
                "Right-to-work protects individual choice in the workplace.",
            ],
            semantic_negative=[
                "Unions must be strengthened through mandatory membership.",
                "We should require workers to join unions to protect labor.",
            ],
            requires_context=["union", "worker", "workplace", "bargaining", "right-to-work"],
        )

        C["503"] = MarporCategory(
            code="503",
            label="Social Justice (Equality)",
            description="Equality, redistribution, progressive taxation, equity framing",
            tendency="libertarian",
            weight=0.7,
            social_tendency=+0.20,
            social_weight=0.3,
            economic_tendency=-0.60,
            economic_weight=1.0,
            primary_keywords=[
                "redistribution", "income inequality", "wealth tax", "progressive taxation", "economic equality",
                "billionaire tax", "millionaire tax", "tax the rich", "close tax loopholes",
                "raise corporate taxes", "corporate tax rate", "fair share", "tax fairness",
                "tax breaks for the rich", "tax breaks for billionaires",
            ],
            secondary_keywords=[
                "reduce inequality", "equity", "fair share", "social justice", "equal opportunity",
                "working families", "middle class", "middle-class",
            ],
            semantic_positive=[
                "We must reduce inequality through progressive taxation.",
                "Redistribution can make growth fairer for working families.",
                "The wealthy should pay their fair share to fund public goods.",
            ],
            semantic_negative=[
                "Redistribution is harmful; inequality is not a problem to address.",
                "Progressive taxes should be eliminated entirely.",
            ],
            requires_context=["inequal", "redistrib", "tax", "wealth", "equity", "equality", "billion", "loophole"],
        )

        C["SJ"] = MarporCategory(
            code="SJ",
            label="Modern Social Justice / Anti-Discrimination",
            description="Anti-discrimination, systemic inequality, inclusion, DEI",
            tendency="libertarian",
            weight=0.8,
            social_tendency=+0.35,
            social_weight=0.6,
            economic_tendency=-0.35,
            economic_weight=0.6,
            primary_keywords=["systemic racism", "anti-discrimination", "dei", "equity", "inclusion", "racial justice"],
            secondary_keywords=["structural inequality", "institutional bias", "marginalized communities", "inclusive justice"],
            semantic_positive=[
                "We must address systemic discrimination and structural barriers.",
                "Equal protection requires enforcing anti-discrimination laws.",
                "Inclusion and equity expand opportunity for marginalized groups.",
            ],
            semantic_negative=[
                "Anti-discrimination policy is unnecessary and should be rolled back.",
                "We should abolish DEI programs entirely.",
            ],
            requires_context=["discrimin", "equity", "inclus", "rac", "bias", "systemic"],
        )

        C["501"] = MarporCategory(
            code="501",
            label="Environmental Protection",
            description="Climate action, sustainability, conservation, clean energy",
            tendency="libertarian",
            weight=0.7,
            social_tendency=+0.10,
            social_weight=0.2,
            economic_tendency=-0.25,
            economic_weight=0.5,
            primary_keywords=["climate change", "renewable energy", "clean energy", "sustainability", "environmental protection"],
            secondary_keywords=["carbon emissions", "conservation", "green transition", "net zero"],
            semantic_positive=[
                "We must protect the environment for future generations.",
                "Clean energy investment reduces emissions and creates jobs.",
                "Climate action is necessary to reduce risk and harm.",
            ],
            semantic_negative=[
                "Climate policy is a hoax and should be abandoned.",
                "Environmental protections should be dismantled to boost industry.",
            ],
            requires_context=["climate", "emission", "renew", "environment", "carbon", "pollution"],
        )

        C["605"] = MarporCategory(
            code="605",
            label="Law and Order",
            description="Tough policing, strict enforcement, crackdown, zero tolerance",
            tendency="authoritarian",
            weight=0.85,
            social_tendency=-0.85,
            social_weight=1.0,
            primary_keywords=["law and order", "tough on crime", "zero tolerance", "crackdown", "strict enforcement"],
            secondary_keywords=["public safety", "strong policing", "harsher penalties", "more police"],
            semantic_positive=[
                "We must crack down on crime with strict enforcement.",
                "Zero tolerance policies deter disorder and violence.",
                "Strong policing is necessary for public safety.",
            ],
            semantic_negative=[
                "We should reduce police powers and avoid punitive enforcement.",
                "Strict policing creates harm and must be rolled back.",
            ],
            requires_context=["crime", "police", "enforce", "order", "safety", "penalt"],
            excludes_context=["accountability", "oversight", "reform", "community policing"],
        )

        C["GUN"] = MarporCategory(
            code="GUN",
            label="Gun Regulation / Weapons Control",
            description="Firearms regulation, background checks, assault weapons bans, restrictions on weapons",
            tendency="authoritarian",
            weight=0.75,
            social_tendency=-0.55,
            social_weight=0.9,
            economic_tendency=0.0,
            economic_weight=0.0,
            primary_keywords=[
                "gun control", "firearms regulation", "weapons regulation",
                "universal background checks", "background checks",
                "assault weapons ban", "assault weapon ban", "ban assault weapons",
                "red flag laws", "red-flag laws",
                "gun violence",
            ],
            secondary_keywords=[
                "restrict guns", "restrict firearms", "weapon restrictions",
                "gun safety law", "gun safety laws", "firearm restrictions",
                "magazine limits", "high-capacity magazine ban", "safe storage laws",
            ],
            semantic_positive=[
                "We need universal background checks to reduce gun violence.",
                "Assault weapons should be banned to protect public safety.",
                "Red flag laws can prevent tragedies by temporarily restricting access to firearms.",
                "Gun safety laws must be strengthened and enforced.",
            ],
            semantic_negative=[
                "Gun control is an attack on individual rights and should be rejected.",
                "Restrictions on firearms should be rolled back.",
            ],
            requires_context=["gun", "firearm", "weapon", "background check", "assault", "violence"],
            excludes_context=["video game", "metaphor", "shoot for", "shooting for"],
            requires_context_hard=False,
        )

        C["302"] = MarporCategory(
            code="302",
            label="Centralization",
            description="Strong centralized power, top-down control, national command",
            tendency="authoritarian",
            weight=0.9,
            social_tendency=-0.80,
            social_weight=1.0,
            primary_keywords=["centralization", "centralize", "strong central government", "top-down control"],
            secondary_keywords=["national control", "unified command", "central authority", "federal supremacy"],
            semantic_positive=[
                "Central authority is needed to coordinate national policy.",
                "Top-down direction ensures uniform enforcement and compliance.",
                "Decisions should be centralized for efficiency and control.",
            ],
            semantic_negative=[
                "Power should be devolved to local communities instead of centralized.",
            ],
            requires_context=["central", "authority", "control", "command", "national"],
        )

        C["305"] = MarporCategory(
            code="305",
            label="Political Authority",
            description="Strong leadership, executive power, decisive authority",
            tendency="authoritarian",
            weight=0.8,
            social_tendency=-0.75,
            social_weight=1.0,
            primary_keywords=["strong leadership", "executive power", "decisive action", "firm hand", "state authority"],
            secondary_keywords=["authority", "strong government", "order and stability", "rule with strength"],
            semantic_positive=[
                "We need strong leadership to restore stability.",
                "Decisive authority is necessary in times of crisis.",
                "A strong executive must act without delay to ensure order.",
            ],
            semantic_negative=[
                "Strong executive authority is dangerous and should be constrained.",
                "We must limit leaders through checks and balances.",
            ],
            requires_context=["authority", "executive", "order", "stability", "enforce"],
            requires_context_hard=False,
        )

        C["603"] = MarporCategory(
            code="603",
            label="Traditional Morality",
            description="Traditional values, religious morality, social conservatism",
            tendency="authoritarian",
            weight=0.8,
            social_tendency=-0.70,
            social_weight=1.0,
            primary_keywords=["traditional values", "family values", "religious values", "moral standards", "sanctity of life"],
            secondary_keywords=["traditional marriage", "moral order", "faith-based", "religious tradition"],
            semantic_positive=[
                "We must defend traditional family values.",
                "Religious faith provides moral guidance for society.",
                "Traditional norms strengthen social cohesion and stability.",
            ],
            semantic_negative=[
                "Traditional morality should not be enforced by the state.",
                "Religious values must not dictate public policy.",
            ],
            requires_context=["traditional", "moral", "relig", "family", "values"],
            excludes_context=["thank you", "god bless", "applause"],
        )

        C["601"] = MarporCategory(
            code="601",
            label="National Way of Life",
            description="National identity, sovereignty, patriotism, cultural heritage",
            tendency="authoritarian",
            weight=0.8,
            social_tendency=-0.65,
            social_weight=1.0,
            primary_keywords=["national identity", "sovereignty", "patriotism", "national heritage", "national pride"],
            secondary_keywords=["our way of life", "cultural heritage", "national unity", "defend our nation"],
            semantic_positive=[
                "We must defend national identity and sovereignty.",
                "Patriotism means preserving our traditions and heritage.",
                "National unity requires loyalty to our shared values.",
            ],
            semantic_negative=[
                "National identity should not be used to exclude minorities.",
                "Patriotism must not override civil liberties.",
            ],
            requires_context=["sovereign", "patriot", "heritage", "identity", "nation"],
            excludes_context=["thank you", "god bless", "applause"],
        )

        C["608"] = MarporCategory(
            code="608",
            label="Immigration Restriction / Anti-Multiculturalism",
            description="Assimilation, border control, immigration limits",
            tendency="authoritarian",
            weight=0.8,
            social_tendency=-0.60,
            social_weight=1.0,
            primary_keywords=["border security", "secure the border", "limit immigration", "deport", "assimilation"],
            secondary_keywords=["immigration restriction", "border control", "illegal immigration", "cultural unity"],
            semantic_positive=[
                "We need strict border control and limits on immigration.",
                "Immigrants must assimilate to our national culture.",
                "Illegal immigration threatens sovereignty and security.",
            ],
            semantic_negative=[
                "We should expand immigration and protect migrant rights.",
                "Border restrictions are harmful and should be relaxed.",
            ],
            requires_context=["border", "immig", "deport", "illegal", "assimil"],
        )

        C["404"] = MarporCategory(
            code="404",
            label="Economic Planning / Industrial Policy",
            description="State-led planning, industrial policy, strategic investment",
            tendency="authoritarian",
            weight=0.9,
            economic_tendency=-0.80,
            economic_weight=1.0,
            primary_keywords=["industrial policy", "economic planning", "state-led", "national development plan", "planning commission"],
            secondary_keywords=["strategic sectors", "public investment plan", "government-led development", "directed economy"],
            semantic_positive=[
                "Government should coordinate long-term industrial strategy.",
                "State planning can guide strategic investment and development.",
                "Industrial policy is needed to rebuild key sectors.",
            ],
            semantic_negative=[
                "Government should not plan the economy; markets should decide.",
            ],
            requires_context=["plan", "industrial", "strategy", "state-led", "development", "investment"],
        )

        C["412"] = MarporCategory(
            code="412",
            label="Controlled Economy / Heavy Regulation",
            description="Price controls, managed economy, strong regulation",
            tendency="authoritarian",
            weight=0.9,
            economic_tendency=-0.70,
            economic_weight=1.0,
            primary_keywords=["price controls", "wage controls", "controlled economy", "managed economy", "government control of prices"],
            secondary_keywords=["strong regulation", "regulatory control", "state intervention", "market controls"],
            semantic_positive=[
                "The government must control prices to protect consumers.",
                "A managed economy prevents exploitation and instability.",
                "Strong regulation is needed to correct market failures.",
            ],
            semantic_negative=[
                "Price controls distort markets and should be removed.",
                "Regulation should be reduced to allow market competition.",
            ],
            requires_context=["control", "regulat", "price", "wage", "managed"],
        )

        C["413"] = MarporCategory(
            code="413",
            label="Nationalization",
            description="Public ownership of industry; state ownership of key sectors",
            tendency="authoritarian",
            weight=0.9,
            economic_tendency=-0.85,
            economic_weight=1.0,
            primary_keywords=[
                "nationalization", "nationalise", "nationalize", "state ownership",
                "public ownership", "government ownership", "nationalize industry",
            ],
            secondary_keywords=[
                "public control", "state enterprise", "state-owned enterprise", "soe",
                "public enterprise", "bring into public hands", "public takeover",
            ],
            semantic_positive=[
                "Nationalize key sectors of the economy to serve the public interest.",
                "Bring utilities and infrastructure under public ownership.",
                "Transfer private monopolies into state ownership.",
                "The state should own strategic industries like energy and rail.",
                "Create state-owned enterprises to run essential services.",
            ],
            semantic_negative=[
                "Oppose nationalization and protect private ownership of industry.",
                "Privatize industries currently owned by the state.",
                "Reject state takeovers of private companies.",
            ],
            requires_context=["own", "industry", "sector", "utility", "state-owned", "public", "national"],
        )

        C["504"] = MarporCategory(
            code="504",
            label="Welfare State Expansion",
            description="Expand welfare programs, universal services, social safety net",
            tendency="authoritarian",
            weight=0.9,
            economic_tendency=-0.90,
            economic_weight=1.0,
            primary_keywords=[
                "expand welfare", "welfare expansion", "universal healthcare", "healthcare is a right",
                "social safety net", "public assistance", "unemployment benefits", "social programs",
                "social security", "medicare", "medicaid", "medicaid expansion",
                "affordable care act", "aca", "pre-existing conditions", "preexisting conditions",
                "child tax credit", "earned income tax credit",
                "snap", "food stamps",
                "paid family leave", "universal childcare", "child care",
            ],
            secondary_keywords=[
                "universal services", "expand benefits", "strong safety net", "public provision",
                "health insurance", "coverage", "lower prescription drug costs", "prescription drug costs",
            ],
            semantic_positive=[
                "We should expand the social safety net to protect vulnerable people.",
                "Universal healthcare should be guaranteed as a right.",
                "Government must provide essential services to all citizens.",
                "Social Security and Medicare must be protected and strengthened.",
                "People with pre-existing conditions must keep their health coverage.",
            ],
            semantic_negative=[
                "Welfare programs should be cut back or limited.",
                "Universal benefits are too costly and should not be expanded.",
                "Social programs should be privatized rather than expanded.",
            ],
            requires_context=["welfare", "benefit", "universal", "healthcare", "safety net", "assistance", "medicare", "medicaid", "social security"],
        )

        C["701"] = MarporCategory(
            code="701",
            label="Labour Groups / Pro-Union",
            description="Workers' rights, unions, collective bargaining",
            tendency="authoritarian",
            weight=0.8,
            social_tendency=+0.10,
            social_weight=0.2,
            economic_tendency=-0.70,
            economic_weight=1.0,
            primary_keywords=["collective bargaining", "right to organize", "trade unions", "labor unions", "workers' rights"],
            secondary_keywords=["union protections", "organized labor", "union contract", "labor movement"],
            semantic_positive=[
                "Strong unions protect workers from exploitation.",
                "Collective bargaining raises wages and improves conditions.",
                "Workers must have the right to organize without retaliation.",
            ],
            semantic_negative=[
                "Unions should be weakened and collective bargaining restricted.",
            ],
            requires_context=["union", "worker", "bargain", "organize", "labor"],
        )

        C["PROT"] = MarporCategory(
            code="PROT",
            label="Protectionism",
            description="Tariffs, trade barriers, economic nationalism",
            tendency="authoritarian",
            weight=0.85,
            economic_tendency=-0.40,
            economic_weight=1.0,
            primary_keywords=["tariffs", "protectionism", "trade barriers", "economic nationalism", "import restrictions"],
            secondary_keywords=["buy local", "buy domestic", "protect jobs", "protect industry", "raise tariffs"],
            semantic_positive=[
                "Tariffs protect domestic jobs and industries from foreign competition.",
                "We need trade barriers to defend national economic security.",
                "Import restrictions will rebuild domestic manufacturing.",
            ],
            semantic_negative=[
                "Tariffs harm consumers and should be reduced.",
                "Trade barriers should be removed to encourage competition.",
            ],
            requires_context=["tariff", "import", "trade", "domestic", "manufactur"],
        )

        C["POP"] = MarporCategory(
            code="POP",
            label="Populism",
            description="People vs elites, anti-establishment framing",
            tendency="authoritarian",
            weight=0.75,
            social_tendency=-0.20,
            social_weight=0.5,
            primary_keywords=["drain the swamp", "corrupt elite", "the establishment", "rigged system", "elites"],
            secondary_keywords=["the people", "ordinary citizens", "anti-establishment", "silent majority"],
            semantic_positive=[
                "The political elites have rigged the system against ordinary people.",
                "We will take power back from the establishment.",
                "This movement is the people versus the corrupt elite.",
            ],
            semantic_negative=[
                "Anti-elite rhetoric is misleading and should be rejected.",
            ],
            requires_context=["elite", "establishment", "rigged", "corrupt", "swamp"],
            requires_context_hard=True,
        )

        C["ENV_AUTH"] = MarporCategory(
            code="ENV_AUTH",
            label="Environmental Authoritarianism",
            description="Mandatory climate controls, strict enforcement, emergency measures",
            tendency="authoritarian",
            weight=0.7,
            social_tendency=-0.30,
            social_weight=0.5,
            economic_tendency=-0.35,
            economic_weight=0.6,
            primary_keywords=["climate emergency", "mandatory reductions", "strict environmental regulation", "climate mandate"],
            secondary_keywords=["enforcement of climate policy", "mandatory compliance", "emissions mandates", "forced transition"],
            semantic_positive=[
                "Climate action requires mandatory compliance and strict enforcement.",
                "The state must impose firm limits on emissions and industrial activity.",
                "Emergency climate measures require centralized coordination and control.",
            ],
            semantic_negative=[
                "Climate policy should rely on voluntary or market-based approaches.",
            ],
            requires_context=["climate", "mandatory", "enforce", "compliance", "regulation"],
        )

        C["CENT"] = MarporCategory(
            code="CENT",
            label="Moderation / Pragmatism (Non-ideology)",
            description="Balanced approach, pragmatism, compromise; NON-IDEOLOGICAL bucket",
            tendency="centrist",
            weight=0.6,
            primary_keywords=["middle ground", "balanced approach", "pragmatism", "moderation", "compromise"],
            secondary_keywords=["centrist", "pragmatic", "reasonable", "not ideological", "common sense solution"],
            semantic_positive=[
                "We need pragmatic solutions rather than ideological purity.",
                "A balanced approach considers multiple perspectives.",
                "Compromise is necessary to make progress.",
            ],
            excludes_context=["thank you", "god bless", "applause"],
        )

        C["BI"] = MarporCategory(
            code="BI",
            label="Bipartisan / Cross-party (Non-ideology)",
            description="Cross-party cooperation and unity; NON-IDEOLOGICAL bucket",
            tendency="centrist",
            weight=0.7,
            primary_keywords=["bipartisan", "across the aisle", "cross-party", "working together", "common ground"],
            secondary_keywords=["cooperation", "unity", "collaboration", "joint effort"],
            semantic_positive=[
                "We must work across parties to solve major problems.",
                "Bipartisan cooperation produces better outcomes.",
                "Both sides should come together for the common good.",
            ],
            excludes_context=["thank you", "god bless", "applause"],
        )

        C["REF"] = MarporCategory(
            code="REF",
            label="Reform / Improvement (Non-ideology)",
            description="Incremental improvement, modernization; NON-IDEOLOGICAL bucket",
            tendency="centrist",
            weight=0.7,
            primary_keywords=["reform", "modernize", "improve", "fix the system", "upgrade"],
            secondary_keywords=["incremental", "step by step", "practical reform", "better governance"],
            semantic_positive=[
                "We should reform institutions to make them work better.",
                "Incremental improvements can produce lasting change.",
                "Modernization is necessary to keep systems effective.",
            ],
            excludes_context=["thank you", "god bless", "applause"],
        )

    def _compile_patterns_for_code(self, code: str) -> None:
        cat = self.categories.get(code)
        if not cat:
            return
        pats: List[Tuple[str, re.Pattern]] = []
        for keyword in cat.get_all_keywords():
            kw = keyword.lower().strip()
            if not kw:
                continue
            if " " in kw:
                escaped = r"\s+".join(map(re.escape, kw.split()))
            else:
                escaped = re.escape(kw)
            pattern_str = r"\b" + escaped + r"\b"
            pats.append((keyword, re.compile(pattern_str, re.IGNORECASE)))
        self.patterns[code] = pats

    def _compile_patterns(self) -> None:
        self.patterns = {}
        for code in self.categories.keys():
            self._compile_patterns_for_code(code)

    def _ensure_semantic_ready(self) -> None:
        if self._semantic_ready:
            return
        self.semantic.ensure_loaded()
        if self.semantic.model is None:
            self._semantic_ready = True
            return
        self._precompute_semantic_embeddings()
        self._semantic_ready = True

    def _precompute_semantic_embeddings(self) -> None:
        self._sem_pos_emb.clear()
        self._sem_neg_emb.clear()
        self._sem_pos_texts.clear()
        self._sem_neg_texts.clear()

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

    def _is_non_substantive(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True

        if len(t) <= 120 and self._CEREMONIAL_RE.search(t):
            if self._ACTION_MARKER_RE.search(t):
                return False
            return True

        words = re.findall(r"[A-Za-z]{2,}", t)
        if len(words) <= 3 and self._CEREMONIAL_RE.search(t):
            return True
        return False

    def _compute_2d_axes(self, validated_ideol: List[Evidence], *, axis_min_total: float) -> Dict[str, Any]:
        s_lib = 0.0
        s_auth = 0.0
        e_left = 0.0
        e_right = 0.0

        for ev in validated_ideol:
            cat = self.categories.get(ev.code)
            if not cat:
                continue

            polm = _axis_pol_mult(int(ev.polarity))
            base = float(ev.strength)

            if float(cat.social_weight) > 0.0 and float(cat.social_tendency) != 0.0:
                contrib = base * float(cat.social_weight) * abs(float(cat.social_tendency)) * polm
                if cat.social_tendency > 0:
                    s_lib += contrib
                else:
                    s_auth += contrib

            if float(cat.economic_weight) > 0.0 and float(cat.economic_tendency) != 0.0:
                contrib = base * float(cat.economic_weight) * abs(float(cat.economic_tendency)) * polm
                if cat.economic_tendency > 0:
                    e_right += contrib
                else:
                    e_left += contrib

        social_total = s_lib + s_auth
        econ_total = e_left + e_right

        social_coord = _coord_from_masses(s_lib, s_auth)
        econ_coord = _coord_from_masses(e_right, e_left)

        social_conf = _axis_confidence(s_lib, s_auth) if social_total >= axis_min_total else 0.0
        econ_conf = _axis_confidence(e_right, e_left) if econ_total >= axis_min_total else 0.0
        overall_conf = float(_clamp((social_conf + econ_conf) / 2.0, 0.0, 1.0))

        magnitude = float(math.sqrt((social_coord ** 2) + (econ_coord ** 2)))

        axis_strengths = {
            "social": {
                "libertarian": float(round(s_lib, 6)),
                "authoritarian": float(round(s_auth, 6)),
                "total": float(round(social_total, 6)),
            },
            "economic": {
                "left": float(round(e_left, 6)),
                "right": float(round(e_right, 6)),
                "total": float(round(econ_total, 6)),
            },
        }

        return {
            "axis_labels": _axis_labels_block(),
            "axis_strengths": axis_strengths,
            "coordinates": {
                "social": float(round(social_coord, 3)),
                "economic": float(round(econ_coord, 3)),
            },
            "coordinates_xy": {
                "x": float(round(econ_coord, 3)),
                "y": float(round(social_coord, 3)),
            },
            "confidence_2d": {
                "social": float(round(social_conf, 3)),
                "economic": float(round(econ_conf, 3)),
                "overall": float(round(overall_conf, 3)),
            },
            "confidence": {
                "social": float(round(social_conf, 3)),
                "economic": float(round(econ_conf, 3)),
                "overall": float(round(overall_conf, 3)),
            },
            "quadrant_2d": {
                "magnitude": float(round(magnitude, 3)),
                "axis_directions": _axis_directions_from_masses(
                    s_lib=s_lib, s_auth=s_auth, e_left=e_left, e_right=e_right, axis_min_total=axis_min_total
                ),
            },
        }

    @staticmethod
    def _primary_family_from_axes(axes2d: Dict[str, Any], *, axis_min_total: float) -> str:
        axis = (axes2d.get("axis_strengths") or {}) if isinstance(axes2d, dict) else {}
        soc = axis.get("social") or {}
        eco = axis.get("economic") or {}

        s_total = float(soc.get("total", 0.0) or 0.0)
        e_total = float(eco.get("total", 0.0) or 0.0)

        s_lib = float(soc.get("libertarian", 0.0) or 0.0)
        s_auth = float(soc.get("authoritarian", 0.0) or 0.0)
        e_left = float(eco.get("left", 0.0) or 0.0)
        e_right = float(eco.get("right", 0.0) or 0.0)

        if s_total >= axis_min_total:
            return LIB_FAMILY if s_lib >= s_auth else AUTH_FAMILY
        if e_total >= axis_min_total:
            return ECON_RIGHT if e_right >= e_left else ECON_LEFT
        return CENTRIST_FAMILY

    @staticmethod
    def _axis_dominance_for_family(axes2d: Dict[str, Any], family: str) -> float:
        axis = axes2d.get("axis_strengths", {}) or {}
        soc = axis.get("social", {}) or {}
        eco = axis.get("economic", {}) or {}

        s_lib = float(soc.get("libertarian", 0.0) or 0.0)
        s_auth = float(soc.get("authoritarian", 0.0) or 0.0)
        e_left = float(eco.get("left", 0.0) or 0.0)
        e_right = float(eco.get("right", 0.0) or 0.0)

        if family in (LIB_FAMILY, AUTH_FAMILY):
            return float(_dominance(s_lib, s_auth))
        if family in (ECON_LEFT, ECON_RIGHT):
            return float(_dominance(e_left, e_right))
        return 0.0

    @staticmethod
    def _is_ideology_evidence_any_axis(
        *,
        evidence_count: int,
        topic_count: int,
        total_strength: float,
        axis_dom: float,
        family: str,
        axes2d: Dict[str, Any],
        evidence_min_total_strength: float,
        evidence_min_axis_dom: float,
        axis_min_total: float,
    ) -> bool:
        if family == CENTRIST_FAMILY:
            return False
        if evidence_count < EVIDENCE_MIN_EVIDENCE_COUNT:
            return False
        if topic_count < EVIDENCE_MIN_TOPIC_COUNT:
            return False
        if total_strength < evidence_min_total_strength:
            return False
        if axis_dom < evidence_min_axis_dom:
            return False

        axis = axes2d.get("axis_strengths", {}) or {}
        soc_total = float((axis.get("social", {}) or {}).get("total", 0.0) or 0.0)
        eco_total = float((axis.get("economic", {}) or {}).get("total", 0.0) or 0.0)

        if soc_total < axis_min_total and eco_total < axis_min_total:
            return False

        return True

    def classify_text_detailed(self, text: str, use_semantic: bool = True, threshold: float = 0.6) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text or self._is_non_substantive(text):
            out = self._empty_result()
            out["analysis_mode"] = "empty_or_ceremonial"
            out["timestamp"] = _utc_now_iso()
            return out

        total_chars = len(text)

        scale = _threshold_scale(float(threshold))
        keep_min = float(_clamp(MIN_EVIDENCE_STRENGTH_KEEP * scale, 0.08, 0.45))
        axis_min_total = float(_clamp(AXIS_MIN_TOTAL * scale, 0.08, 0.25))
        evidence_min_total_strength = float(_clamp(EVIDENCE_MIN_TOTAL_STRENGTH * scale, 0.10, 0.50))
        evidence_min_axis_dom = float(_clamp(EVIDENCE_MIN_AXIS_DOMINANCE * scale, 0.30, 0.55))

        lexical_evidence, lexical_max = self._find_lexical_evidence(text, total_chars=total_chars)
        lexical_validated = self._validate_evidence_with_context(
            text,
            lexical_evidence,
            total_chars=total_chars,
            keep_threshold=keep_min,
        )

        lexical_out = self._build_final_result(
            text,
            lexical_validated,
            axis_min_total=axis_min_total,
            evidence_min_total_strength=evidence_min_total_strength,
            evidence_min_axis_dom=evidence_min_axis_dom,
        )
        if bool(lexical_out.get("is_ideology_evidence", False)):
            lexical_out.setdefault("analysis_mode", "lexical_only")
            return lexical_out

        if not use_semantic:
            lexical_out.setdefault("analysis_mode", "lexical_only")
            return lexical_out

        self._ensure_semantic_ready()
        if self.semantic.model is None:
            lexical_out.setdefault("analysis_mode", "lexical_only_semantic_unavailable")
            return lexical_out

        semantic_only = self._find_semantic_evidence_only(
            text,
            lexical_max_strength=lexical_max,
            total_chars=total_chars,
        )
        combined = list(lexical_evidence) + list(semantic_only)
        combined_validated = self._validate_evidence_with_context(
            text,
            combined,
            total_chars=total_chars,
            keep_threshold=keep_min,
        )

        semantic_out = self._build_final_result(
            text,
            combined_validated,
            axis_min_total=axis_min_total,
            evidence_min_total_strength=evidence_min_total_strength,
            evidence_min_axis_dom=evidence_min_axis_dom,
        )
        semantic_out.setdefault("analysis_mode", "lexical_plus_semantic_fallback")
        return semantic_out

    def _find_lexical_evidence(self, text: str, *, total_chars: int) -> Tuple[List[Evidence], Dict[str, float]]:
        evidence: List[Evidence] = []
        text_lower = text.lower()
        lexical_max_strength: Dict[str, float] = defaultdict(float)

        for code, patterns in self.patterns.items():
            cat = self.categories[code]
            for keyword, pattern in patterns:
                for m in pattern.finditer(text):
                    if cat.anti_patterns:
                        window = text_lower[m.start() : min(len(text_lower), m.end() + 60)]
                        if any((anti and anti.lower() in window) for anti in cat.anti_patterns):
                            continue

                    match_type = "lexical_primary" if keyword in cat.primary_keywords else "lexical_secondary"
                    base = self.weights.get_evidence_quality(match_type)
                    lexical_max_strength[code] = max(float(lexical_max_strength[code]), float(base))

                    matched_text = text[m.start() : m.end()]

                    evidence.append(
                        Evidence(
                            code=code,
                            matched_text=matched_text,
                            span=(m.start(), m.end()),
                            match_type=match_type,
                            base_strength=float(base),
                            strength=float(base),
                            polarity=0,
                            polarity_reason="pending",
                            inside_quotes=self._is_inside_quotes(text, m.start()),
                        )
                    )

        return evidence, dict(lexical_max_strength)

    def _find_semantic_evidence_only(
        self,
        text: str,
        *,
        lexical_max_strength: Dict[str, float],
        total_chars: int,
    ) -> List[Evidence]:
        self._ensure_semantic_ready()
        if self.semantic.model is None:
            return []

        text_emb = self.semantic.encode_one(text)
        if text_emb is None:
            return []

        evidence: List[Evidence] = []

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
            if pos_emb is not None and getattr(pos_emb, "shape", (0,))[0] > 0:
                sims = self.semantic.cosine_similarity_matrix(text_emb, pos_emb)
                if sims is not None and getattr(sims, "size", 0) > 0:
                    topk_idx = np.argsort(sims)[::-1][: max(1, min(SEMANTIC_TOPK, int(sims.size)))]
                    topk_vals = sims[topk_idx]
                    best_pos = float(topk_vals[0])
                    second_pos = float(topk_vals[1]) if len(topk_vals) > 1 else float(-1.0)
                    best_pos_text = pos_texts[int(topk_idx[0])] if int(topk_idx[0]) < len(pos_texts) else None

            best_neg = None
            second_neg = None
            best_neg_text = None
            if neg_emb is not None and getattr(neg_emb, "shape", (0,))[0] > 0:
                sims = self.semantic.cosine_similarity_matrix(text_emb, neg_emb)
                if sims is not None and getattr(sims, "size", 0) > 0:
                    topk_idx = np.argsort(sims)[::-1][: max(1, min(SEMANTIC_TOPK, int(sims.size)))]
                    topk_vals = sims[topk_idx]
                    best_neg = float(topk_vals[0])
                    second_neg = float(topk_vals[1]) if len(topk_vals) > 1 else float(-1.0)
                    best_neg_text = neg_texts[int(topk_idx[0])] if int(topk_idx[0]) < len(neg_texts) else None

            chosen_polarity = 0
            chosen_sim = None
            chosen_example = None
            chosen_reason = None

            if (
                best_neg is not None
                and best_neg >= SEMANTIC_NEG_THRESHOLD
                and (best_neg - float(second_neg or -1.0)) >= SEMANTIC_MARGIN
            ):
                if best_pos is None or best_neg >= (best_pos + SEMANTIC_MARGIN):
                    chosen_polarity = -1
                    chosen_sim = best_neg
                    chosen_example = best_neg_text
                    chosen_reason = "semantic_oppose"

            if (
                chosen_polarity == 0
                and best_pos is not None
                and best_pos >= SEMANTIC_THRESHOLD
                and (best_pos - float(second_pos or -1.0)) >= SEMANTIC_MARGIN
            ):
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
                    matched_text=f"[Semantic: {(chosen_example or '')[:80]}...]",
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

        return evidence

    def _validate_evidence_with_context(
        self,
        text: str,
        evidence: List[Evidence],
        *,
        total_chars: int,
        keep_threshold: float,
    ) -> List[Evidence]:
        text_lower = text.lower()
        validated: List[Evidence] = []

        for ev in evidence:
            cat = self.categories.get(ev.code)
            if not cat:
                continue

            out = Evidence(**ev.__dict__)

            if cat.requires_context:
                hits = [c for c in cat.requires_context if c and c.lower() in text_lower]
                misses = [c for c in cat.requires_context if c and c.lower() not in text_lower]
                out.context_hits = hits
                out.context_misses = misses

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

            if out.polarity == 0:
                pol, reason = self._detect_polarity(text, out)
                out.polarity = int(pol)
                out.polarity_reason = str(reason)

            if out.polarity < 0:
                out.strength *= self.weights.OPPOSE_STRENGTH
            elif out.polarity == 0:
                out.strength *= self.weights.NEUTRAL_STRENGTH
            else:
                out.strength *= self.weights.SUPPORT_STRENGTH

            out.strength *= (self.weights.INSIDE_QUOTES if out.inside_quotes else self.weights.OUTSIDE_QUOTES)
            out.strength *= float(cat.weight)

            attr_hint, attr_reason = self._detect_attribution_hint(text, out)
            out.attribution_hint = bool(attr_hint)
            out.attribution_reason = attr_reason

            out.strength = float(min(1.0, max(0.0, out.strength)))
            if out.strength < float(keep_threshold):
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
        window = 90
        start = max(0, evidence.span[0] - window)
        end = min(len(text), evidence.span[1] + window)
        ctx = text_lower[start:end]

        for pat in self.opposition_patterns:
            if pat.search(ctx):
                return -1, f"opposition:{pat.pattern[:24]}"

        for pat in self.negation_patterns:
            for m in pat.finditer(ctx):
                distance = abs(m.start() - (evidence.span[0] - start))
                if distance < 60:
                    return -1, f"negation:{pat.pattern[:24]}"

        for pat in self.endorsement_patterns:
            if pat.search(ctx):
                return 1, f"support:{pat.pattern[:24]}"

        if evidence.match_type == "semantic" and evidence.polarity != 0:
            return evidence.polarity, evidence.polarity_reason or "semantic_assigned"

        return 0, "neutral_or_ambiguous"

    def _detect_attribution_hint(self, text: str, evidence: Evidence) -> Tuple[bool, Optional[str]]:
        text_lower = text.lower()
        window = 140
        start = max(0, evidence.span[0] - window)
        end = min(len(text), evidence.span[1] + window)
        ctx = text_lower[start:end]

        if evidence.inside_quotes:
            for pat in self.attribution_patterns:
                m = pat.search(ctx)
                if m:
                    return True, f"quoted_attribution:{pat.pattern[:32]}"

        for pat in self.attribution_patterns:
            m = pat.search(ctx)
            if m:
                return True, f"context_attribution:{pat.pattern[:32]}"

        return False, None

    @staticmethod
    def _is_inside_quotes(text: str, position: int) -> bool:
        qchars = {'"', "\u201c", "\u201d", "\u00ab", "\u00bb"}
        inside = False
        for ch in text[: max(0, position)]:
            if ch in qchars:
                inside = not inside
        return inside

    def _determine_subtype_evidence_level(
        self,
        code_strengths: Dict[str, float],
        evidence_codes: List[str],
        family: str,
    ) -> Optional[str]:
        fam_l = (family or "").strip().lower()
        if fam_l not in ("libertarian", "authoritarian"):
            return None

        def belongs(subtype_name: str) -> bool:
            s = (subtype_name or "").lower()
            return ("libertarian" in s) if fam_l == "libertarian" else ("authoritarian" in s)

        best_subtype: Optional[str] = None
        best_pc: float = 0.0
        best_strength: float = 0.0

        for subtype, subtype_codes in IDEOLOGY_SUBTYPES.items():
            if not belongs(subtype):
                continue

            total_s = float(sum(float(code_strengths.get(c, 0.0)) for c in subtype_codes))
            if total_s <= 0.0:
                continue

            pc = EvidenceConfidence.calculate_pattern_confidence_single(
                evidence_codes=evidence_codes,
                subtype_codes=subtype_codes,
            )
            pc_val = float(pc.get("pattern_confidence", 0.0) or 0.0)

            if (pc_val > best_pc + 1e-9) or (abs(pc_val - best_pc) < 1e-9 and total_s > best_strength):
                best_subtype = subtype
                best_pc = pc_val
                best_strength = total_s

        return best_subtype

    def _build_final_result(
        self,
        text: str,
        validated_all: List[Evidence],
        *,
        axis_min_total: float,
        evidence_min_total_strength: float,
        evidence_min_axis_dom: float,
    ) -> Dict[str, Any]:
        if not validated_all:
            out = self._empty_result()
            out["analysis_mode"] = "empty"
            out["timestamp"] = _utc_now_iso()
            return out

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

        centrist_strength = float(sum(float(e.strength) for e in validated_centrist))

        if not validated_ideol:
            out = self._empty_result()
            out["centrist_box"] = {
                "evidence": self._serialize_evidence(validated_centrist),
                "evidence_count": int(len(validated_centrist)),
                "marpor_codes": sorted(set(ev.code for ev in validated_centrist)),
                "marpor_breakdown": self._build_marpor_breakdown_from_validated(validated_centrist),
                "total_strength": float(centrist_strength),
            }
            out["jargon_box"] = dict(out["centrist_box"])
            out["analysis_mode"] = "centrist_only"
            out["timestamp"] = _utc_now_iso()
            return out

        axes2d = self._compute_2d_axes(validated_ideol, axis_min_total=axis_min_total)

        family = self._primary_family_from_axes(axes2d, axis_min_total=axis_min_total)
        axis_dom = self._axis_dominance_for_family(axes2d, family)

        evidence_count = len(validated_ideol)
        topic_count = len(set(ev.code for ev in validated_ideol))
        marpor_codes = sorted(set(ev.code for ev in validated_ideol))
        total_strength = float(sum(float(ev.strength) for ev in validated_ideol))

        is_ev = self._is_ideology_evidence_any_axis(
            evidence_count=evidence_count,
            topic_count=topic_count,
            total_strength=total_strength,
            axis_dom=axis_dom,
            family=family,
            axes2d=axes2d,
            evidence_min_total_strength=evidence_min_total_strength,
            evidence_min_axis_dom=evidence_min_axis_dom,
            axis_min_total=axis_min_total,
        )

        axis = axes2d.get("axis_strengths", {}) or {}
        soc_total = float((axis.get("social", {}) or {}).get("total", 0.0) or 0.0)
        eco_total = float((axis.get("economic", {}) or {}).get("total", 0.0) or 0.0)
        is_ev_2d = bool((soc_total >= axis_min_total) or (eco_total >= axis_min_total))

        subtype: Optional[str] = None
        code_strengths: DefaultDict[str, float] = defaultdict(float)
        for ev in validated_ideol:
            pol_w = 1.0 if ev.polarity > 0 else (0.7 if ev.polarity == 0 else 0.4)
            code_strengths[ev.code] += float(ev.strength) * pol_w

        if family in (LIB_FAMILY, AUTH_FAMILY):
            subtype = self._determine_subtype_evidence_level(
                code_strengths=dict(code_strengths),
                evidence_codes=marpor_codes,
                family=family,
            )

        pattern_conf: Dict[str, Any] = {}
        if subtype:
            subtype_codes = IDEOLOGY_SUBTYPES.get(subtype, [])
            if subtype_codes:
                pattern_conf = self.conf_calc.calculate_pattern_confidence_single(
                    evidence_codes=marpor_codes,
                    subtype_codes=subtype_codes,
                )

        avg_ev_conf = (
            sum(float((ev.evidence_confidence or {}).get("evidence_confidence", 0.0)) for ev in validated_ideol)
            / max(1, len(validated_ideol))
        )
        confidence_score = float(min(1.0, max(0.0, avg_ev_conf * (0.6 + 0.4 * axis_dom))))

        signal_strength = float((1.0 - math.exp(-max(0.0, total_strength))) * 100.0)
        signal_strength = float(min(100.0, max(0.0, signal_strength)))

        marpor_breakdown = self._build_marpor_breakdown_from_validated(validated_ideol)

        soc = axis.get("social", {}) or {}
        eco = axis.get("economic", {}) or {}

        s_lib = float(soc.get("libertarian", 0.0) or 0.0)
        s_auth = float(soc.get("authoritarian", 0.0) or 0.0)
        e_left = float(eco.get("left", 0.0) or 0.0)
        e_right = float(eco.get("right", 0.0) or 0.0)

        total_mass = s_lib + s_auth + e_left + e_right
        if total_mass > 0:
            scores = {
                LIB_FAMILY: round((s_lib / total_mass) * 100.0, 2),
                AUTH_FAMILY: round((s_auth / total_mass) * 100.0, 2),
                ECON_LEFT: round((e_left / total_mass) * 100.0, 2),
                ECON_RIGHT: round((e_right / total_mass) * 100.0, 2),
            }
        else:
            scores = {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, ECON_LEFT: 0.0, ECON_RIGHT: 0.0}

        return {
            "ideology_family": family,
            "ideology_subtype": subtype if family in (LIB_FAMILY, AUTH_FAMILY) else None,
            "scores": scores,
            "evidence": self._serialize_evidence(validated_ideol),
            "evidence_count": int(evidence_count),
            "support_evidence_count": int(sum(1 for ev in validated_ideol if ev.polarity > 0)),
            "marpor_codes": marpor_codes,
            "marpor_breakdown": marpor_breakdown,
            "marpor_code_analysis": marpor_breakdown,
            "total_evidence_strength": float(total_strength),
            "axis_dominance": float(axis_dom),
            "is_ideology_evidence": bool(is_ev),
            "is_ideology_evidence_2d": bool(is_ev_2d),
            "filtered_topic_count": int(topic_count),
            "confidence_score": float(confidence_score),
            "signal_strength": float(signal_strength),
            "ideology_2d": axes2d,
            "centrist_box": {
                "evidence": self._serialize_evidence(validated_centrist),
                "evidence_count": int(len(validated_centrist)),
                "marpor_codes": sorted(set(ev.code for ev in validated_centrist)),
                "marpor_breakdown": self._build_marpor_breakdown_from_validated(validated_centrist),
                "total_strength": float(centrist_strength),
            },
            "jargon_box": {
                "evidence": self._serialize_evidence(validated_centrist),
                "evidence_count": int(len(validated_centrist)),
                "marpor_codes": sorted(set(ev.code for ev in validated_centrist)),
                "marpor_breakdown": self._build_marpor_breakdown_from_validated(validated_centrist),
                "total_strength": float(centrist_strength),
            },
            "evidence_level_confidence": {
                "avg_evidence_confidence": round(float(avg_ev_conf), 4),
                "pattern_confidence": pattern_conf,
                "evidence_quality": self._analyze_evidence_quality(validated_ideol),
            },
            "code_strengths": dict(code_strengths),
            "analysis_level": "evidence_only",
            "analysis_mode": "ideology_classified",
            "timestamp": _utc_now_iso(),
        }

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

            polarity = {"support": 0, "oppose": 0, "neutral": 0}
            ev_conf: List[float] = []
            for e in evs:
                if e.polarity > 0:
                    polarity["support"] += 1
                elif e.polarity < 0:
                    polarity["oppose"] += 1
                else:
                    polarity["neutral"] += 1
                ev_conf.append(float((e.evidence_confidence or {}).get("evidence_confidence", 0.0)))

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
                "social_tendency": float(cat.social_tendency),
                "economic_tendency": float(cat.economic_tendency),
                "social_weight": float(cat.social_weight),
                "economic_weight": float(cat.economic_weight),
            }

        return dict(sorted(out.items(), key=lambda kv: kv[1]["percentage"], reverse=True))

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
            "total_evidence": int(len(evidence)),
            "lexical_primary_count": int(lexical_primary),
            "lexical_secondary_count": int(lexical_secondary),
            "semantic_count": int(len(semantic)),
            "avg_semantic_similarity": round(float(avg_sem), 3),
            "evidence_diversity": round(float(diversity), 3),
            "quality_score": round(float(quality), 3),
        }

    @staticmethod
    def _serialize_evidence(evidence: List[Evidence]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ev in evidence:
            out.append(
                {
                    "code": str(ev.code),
                    "matched_text": str(ev.matched_text),
                    "span": [int(ev.span[0]), int(ev.span[1])],
                    "match_type": str(ev.match_type),
                    "strength": round(float(ev.strength), 3),
                    "polarity": int(ev.polarity),
                    "polarity_reason": str(ev.polarity_reason),
                    "inside_quotes": bool(ev.inside_quotes),
                    "semantic_similarity": round(float(ev.semantic_similarity), 3) if ev.semantic_similarity is not None else None,
                    "context_hits": list(ev.context_hits),
                    "context_misses": list(ev.context_misses),
                    "evidence_confidence": dict(ev.evidence_confidence or {}),
                    "attribution_hint": bool(getattr(ev, "attribution_hint", False)),
                    "attribution_reason": getattr(ev, "attribution_reason", None),
                }
            )
        return out

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "ideology_family": CENTRIST_FAMILY,
            "ideology_subtype": None,
            "scores": {LIB_FAMILY: 0.0, AUTH_FAMILY: 0.0, ECON_LEFT: 0.0, ECON_RIGHT: 0.0},
            "evidence": [],
            "evidence_count": 0,
            "support_evidence_count": 0,
            "marpor_codes": [],
            "marpor_breakdown": {},
            "marpor_code_analysis": {},
            "total_evidence_strength": 0.0,
            "axis_dominance": 0.0,
            "is_ideology_evidence": False,
            "is_ideology_evidence_2d": False,
            "filtered_topic_count": 0,
            "confidence_score": 0.0,
            "signal_strength": 0.0,
            "ideology_2d": {
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
            },
            "centrist_box": {"evidence": [], "evidence_count": 0, "marpor_codes": [], "marpor_breakdown": {}, "total_strength": 0.0},
            "jargon_box": {"evidence": [], "evidence_count": 0, "marpor_codes": [], "marpor_breakdown": {}, "total_strength": 0.0},
            "evidence_level_confidence": {
                "avg_evidence_confidence": 0.0,
                "pattern_confidence": {},
                "evidence_quality": {"total_evidence": 0},
            },
            "code_strengths": {},
            "analysis_level": "evidence_only",
            "analysis_mode": "empty",
            "timestamp": _utc_now_iso(),
        }


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
    "ECON_LEFT",
    "ECON_RIGHT",
]