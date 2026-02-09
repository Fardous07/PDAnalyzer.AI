# backend/app/services/attribution_parser.py

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, Tuple

logger = logging.getLogger(__name__)

_SPACY_AVAILABLE = False
try:
    import spacy  # type: ignore

    _SPACY_AVAILABLE = True
except Exception:
    spacy = None  # type: ignore

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token  # noqa: F401


@dataclass
class AttributionResult:
    subject: str
    confidence: float
    commitment: float
    commitment_type: str
    inside_quotes: bool
    verb: Optional[str] = None
    subject_text: Optional[str] = None
    reasoning: str = ""
    span: Optional[Tuple[int, int]] = None
    subject_span: Optional[Tuple[int, int]] = None
    trigger_span: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ATTRIBUTION_VERBS: Set[str] = {
    "say",
    "claim",
    "argue",
    "state",
    "announce",
    "declare",
    "suggest",
    "tell",
    "explain",
    "insist",
    "assert",
    "predict",
    "warn",
    "promise",
    "admit",
    "deny",
    "confirm",
    "accuse",
    "blame",
    "criticize",
    "praise",
    "support",
    "oppose",
    "endorse",
    "back",
    "reject",
    "defend",
    "condemn",
    "believe",
    "think",
    "hope",
    "fear",
    "want",
    "prefer",
    "demand",
    "urge",
    "recommend",
    "propose",
    "contend",
    "maintain",
    "acknowledge",
    "concede",
    "allege",
    "charge",
    "report",
    "note",
}

COMMITMENT_MARKERS_STRONG: Set[str] = {
    "we will",
    "i will",
    "our plan",
    "we are going to",
    "i am going to",
    "we pledge",
    "i pledge",
    "i promise",
    "we promise",
    "we commit",
    "i commit",
    "we guarantee",
    "i guarantee",
    "we vow",
    "i vow",
    "we shall",
    "i shall",
}

COMMITMENT_MARKERS_MEDIUM: Set[str] = {
    "we must",
    "we should",
    "i should",
    "we need to",
    "we need",
    "let us",
    "we can",
    "we intend to",
    "i intend to",
    "we aim to",
    "i aim to",
    "we seek to",
    "i seek to",
    "we propose",
    "i propose",
    "we plan to",
}

BELIEF_MARKERS: Set[str] = {
    "i believe",
    "we believe",
    "i think",
    "we think",
    "i feel",
    "we feel",
    "i support",
    "we support",
    "i oppose",
    "we oppose",
    "i favor",
    "we favor",
    "i reject",
    "we reject",
    "i stand for",
    "we stand for",
    "i am committed",
    "we are committed",
    "i value",
    "we value",
    "i champion",
    "we champion",
}

HEDGE_MARKERS: Set[str] = {
    "might",
    "could",
    "may",
    "it is possible that",
    "it could be",
    "it might be",
    "perhaps",
    "maybe",
    "possibly",
    "potentially",
    "conceivably",
}

HYPOTHETICAL_MARKERS: Set[str] = {
    "if we",
    "if i",
    "if they",
    "if our",
    "if their",
    "if this",
    "if that",
    "suppose we",
    "imagine we",
    "imagine if",
    "were we to",
    "should we",
    "in the event that",
    "assuming that",
    "provided that",
}

OPPONENT_LEXICON: Set[str] = {
    "my opponent",
    "our opponents",
    "opponent",
    "rival",
    "rivals",
    "adversary",
    "biden",
    "trump",
    "democrats",
    "republicans",
    "the left",
    "the right",
    "conservatives",
    "liberals",
    "labour",
    "tories",
    "the opposition",
    "the other side",
    "their party",
    "his administration",
    "her administration",
}

THIRD_PARTY_LEXICON: Set[str] = {
    "experts say",
    "studies show",
    "research shows",
    "according to",
    "scientists",
    "economists",
    "analysts",
    "observers",
    "critics",
    "some say",
    "many believe",
    "it is said",
    "reports indicate",
}

SPEAKER_PRONOUNS: Set[str] = {"i", "we", "me", "us", "our", "ours", "my", "myself", "ourselves"}

NEGATION_WORDS: Set[str] = {
    "not",
    "never",
    "no",
    "none",
    "neither",
    "nor",
    "cannot",
    "won't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "wouldn't",
    "couldn't",
    "shouldn't",
}

_REPORTING_VERBS_RE = re.compile(
    r"\b(says|said|claims?|claimed|argues?|argued|stated|announced|declared|insisted|asserted|warned|predicted|reported|noted)\b",
    re.I,
)
_FIRST_PERSON_RE = re.compile(r"\b(i|we|my|our|us)\b", re.I)


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return 0.0 if v <= 0.0 else (1.0 if v >= 1.0 else v)


def _lower(s: str) -> str:
    return (s or "").lower()


def _has_any_phrase(text_lower: str, phrases: Set[str]) -> Optional[str]:
    for p in phrases:
        if p and p in text_lower:
            return p
    return None


class AttributionParser:
    def __init__(self, language: str = "en", nlp: Optional[Any] = None) -> None:
        self.language = language or "en"
        self._nlp: Optional[Any] = None

        if nlp is not None:
            self._nlp = nlp
        elif _SPACY_AVAILABLE and spacy is not None:
            try:
                self._nlp = spacy.load("en_core_web_sm", disable=["ner"])
                logger.info("AttributionParser: loaded spaCy model")
            except Exception as e:
                logger.warning("AttributionParser: spaCy load failed: %s", e)

        self._quote_re = re.compile(r'["\u201c\u201d\u00ab\u00bb]')

    def parse(self, text: str, evidence_span: Optional[Tuple[int, int]] = None) -> AttributionResult:
        text = (text or "").strip()
        if not text:
            return AttributionResult(
                subject="ambiguous",
                confidence=0.0,
                commitment=0.0,
                commitment_type="none",
                inside_quotes=False,
                reasoning="empty_text",
                span=evidence_span,
            )

        if evidence_span:
            start = max(0, min(len(text), int(evidence_span[0])))
            end = max(0, min(len(text), int(evidence_span[1])))
            evidence_span = (start, max(start, end))

        inside_quotes = self._is_inside_quotes(text, evidence_span)

        if self._nlp is None:
            return self._heuristic_parse(text, evidence_span, inside_quotes)

        try:
            doc = self._nlp(text)
        except Exception as e:
            logger.warning("spaCy parse failed: %s", e)
            return self._heuristic_parse(text, evidence_span, inside_quotes)

        sent = self._select_sentence(doc, evidence_span)
        return self._parse_with_spacy(doc, sent, evidence_span, inside_quotes)

    def _is_inside_quotes(self, text: str, evidence_span: Optional[Tuple[int, int]]) -> bool:
        if not evidence_span:
            return False
        start = max(0, min(len(text), evidence_span[0]))
        prefix = text[:start]
        count = len(self._quote_re.findall(prefix))
        return (count % 2) == 1

    def _select_sentence(self, doc: Any, evidence_span: Optional[Tuple[int, int]]) -> Any:
        if evidence_span:
            ev_start = max(0, evidence_span[0])
            try:
                for sent in doc.sents:
                    if sent.start_char <= ev_start < sent.end_char:
                        return sent
            except Exception:
                pass
        try:
            return list(doc.sents)[0]
        except Exception:
            return doc[:]

    def _score_commitment(self, sent_text: str) -> Tuple[float, str]:
        text = _lower(sent_text)
        score, ctype = 0.0, "none"

        for phrase in COMMITMENT_MARKERS_STRONG:
            if phrase in text:
                score, ctype = max(score, 0.92), "commitment"

        for phrase in COMMITMENT_MARKERS_MEDIUM:
            if phrase in text:
                score = max(score, 0.78)
                if ctype == "none":
                    ctype = "commitment"

        for phrase in BELIEF_MARKERS:
            if phrase in text:
                score = max(score, 0.68)
                if ctype == "none":
                    ctype = "commitment"

        hyp = any(phrase in text for phrase in HYPOTHETICAL_MARKERS)
        if hyp and score > 0:
            score, ctype = max(0.35, score - 0.28), "hypothetical"

        hedge = any(phrase in text for phrase in HEDGE_MARKERS)
        if hedge and score > 0:
            score = max(0.10, score - 0.22)

        return _clamp01(score), ctype

    def _find_candidate_verbs(self, sent: Any, evidence_span: Optional[Tuple[int, int]]) -> List[Any]:
        if not sent:
            return []
        ev_start = evidence_span[0] if evidence_span else None

        scored: List[Tuple[Any, int]] = []
        try:
            for tok in sent:
                try:
                    lemma = tok.lemma_.lower()
                    pos = tok.pos_
                except Exception:
                    continue

                if lemma in ATTRIBUTION_VERBS and pos in {"VERB", "AUX"}:
                    score = 1
                    if ev_start is not None:
                        try:
                            sub_start, sub_end = tok.idx, tok.idx + len(tok.text)
                            for t2 in tok.subtree:
                                sub_start = min(sub_start, t2.idx)
                                sub_end = max(sub_end, t2.idx + len(t2.text))
                            if sub_start <= ev_start < sub_end:
                                score = 2
                        except Exception:
                            pass
                    scored.append((tok, score))
        except Exception as e:
            logger.warning("Error finding candidate verbs: %s", e)
            return []

        scored.sort(key=lambda p: (p[1], p[0].idx), reverse=True)
        return [t for t, _ in scored]

    def _find_subject_token(self, verb: Any) -> Optional[Any]:
        try:
            for child in verb.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    return child
            if verb.head is not verb:
                for child in verb.head.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        return child
        except Exception:
            pass
        return None

    def _classify_subject_token(self, subject: Any) -> str:
        try:
            lower_txt = subject.text.lower().strip()
            if lower_txt in SPEAKER_PRONOUNS:
                return "speaker"

            subj_text = " ".join(tok.text.lower() for tok in subject.subtree)

            if _has_any_phrase(subj_text, OPPONENT_LEXICON):
                return "opponent"
            if _has_any_phrase(subj_text, THIRD_PARTY_LEXICON):
                return "third_party"

            if subject.pos_ == "PRON":
                return "third_party"
            if subject.pos_ == "PROPN":
                return "third_party"
        except Exception:
            pass

        return "ambiguous"

    def _check_negation(self, verb: Any) -> bool:
        try:
            for child in verb.children:
                if child.dep_ == "neg" or child.text.lower() in NEGATION_WORDS:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _token_span(token: Any) -> Tuple[int, int]:
        try:
            return (token.idx, token.idx + len(token.text))
        except Exception:
            return (0, 0)

    def _parse_with_spacy(
        self,
        doc: Any,
        sent: Any,
        evidence_span: Optional[Tuple[int, int]],
        inside_quotes: bool,
    ) -> AttributionResult:
        try:
            sent_text = sent.text
        except Exception:
            sent_text = str(sent)

        sent_lower = _lower(sent_text)

        commitment_score, commitment_type = self._score_commitment(sent_text)

        opp_hint = _has_any_phrase(sent_lower, OPPONENT_LEXICON)
        third_hint = _has_any_phrase(sent_lower, THIRD_PARTY_LEXICON)

        candidates = self._find_candidate_verbs(sent, evidence_span)
        verb_token = candidates[0] if candidates else None

        subject_token = None
        subject_label = "ambiguous"
        confidence = 0.45
        reasoning_parts: List[str] = ["spacy"]

        if verb_token:
            try:
                reasoning_parts.append(f"verb:{verb_token.lemma_}")
            except Exception:
                reasoning_parts.append("verb:unknown")
            confidence += 0.10

            subject_token = self._find_subject_token(verb_token)
            if subject_token:
                subject_label = self._classify_subject_token(subject_token)
                try:
                    reasoning_parts.append(f"dep:{subject_token.text}->{verb_token.lemma_}")
                except Exception:
                    reasoning_parts.append("dep:parsed")
                confidence += 0.22
            else:
                reasoning_parts.append("verb_no_subj")
                confidence += 0.05

            if subject_label == "ambiguous":
                if opp_hint:
                    subject_label = "opponent"
                    reasoning_parts.append("hint:opponent")
                    confidence = max(confidence, 0.55)
                elif third_hint:
                    subject_label = "third_party"
                    reasoning_parts.append("hint:third_party")
                    confidence = max(confidence, 0.55)

            if self._check_negation(verb_token) and commitment_type == "commitment":
                commitment_type = "negated"
                reasoning_parts.append("negated")

        else:
            reasoning_parts.append("no_attr_verb")

            if opp_hint:
                subject_label = "opponent"
                reasoning_parts.append("hint:opponent_noverb")
                confidence = max(confidence, 0.54)
            elif third_hint:
                subject_label = "third_party"
                reasoning_parts.append("hint:third_party_noverb")
                confidence = max(confidence, 0.54)
            else:
                if _FIRST_PERSON_RE.search(sent_lower):
                    subject_label = "speaker"
                    reasoning_parts.append("fp_in_sentence")
                    confidence = max(confidence, 0.56)

        if inside_quotes:
            reasoning_parts.append("quoted")
            if subject_label == "speaker":
                subject_label = "third_party"
                reasoning_parts.append("speaker->third_party_due_to_quotes")
            if commitment_type == "commitment":
                commitment_type = "quoted"
            confidence = max(0.40, confidence - 0.15)

        if subject_label != "speaker" and commitment_type == "commitment":
            commitment_type = "reported"
            commitment_score = max(0.10, commitment_score - 0.10)
            reasoning_parts.append("commitment->reported")

        if commitment_type == "hypothetical":
            confidence = max(0.40, confidence - 0.10)
            reasoning_parts.append("hypothetical")

        if subject_label == "ambiguous":
            confidence = min(confidence, 0.60)

        return AttributionResult(
            subject=subject_label,
            confidence=_clamp01(confidence),
            commitment=_clamp01(commitment_score),
            commitment_type=commitment_type,
            inside_quotes=inside_quotes,
            verb=verb_token.lemma_ if verb_token else None,
            subject_text=subject_token.text if subject_token else None,
            reasoning=";".join(reasoning_parts) or "default",
            span=evidence_span,
            subject_span=self._token_span(subject_token) if subject_token else None,
            trigger_span=self._token_span(verb_token) if verb_token else None,
        )

    def _heuristic_parse(self, text: str, evidence_span: Optional[Tuple[int, int]], inside_quotes: bool) -> AttributionResult:
        lower = _lower(text)
        commitment_score, commitment_type = self._score_commitment(text)

        reasoning_parts = ["heuristic"]
        subject_label = "ambiguous"
        confidence = 0.46

        opp_hint = _has_any_phrase(lower, OPPONENT_LEXICON)
        third_hint = _has_any_phrase(lower, THIRD_PARTY_LEXICON)
        has_first_person = bool(_FIRST_PERSON_RE.search(lower))

        if has_first_person and commitment_score >= 0.55:
            subject_label = "speaker"
            confidence = 0.66
            reasoning_parts.append("fp+commit")

        if subject_label != "speaker" and third_hint:
            subject_label = "third_party"
            confidence = max(confidence, 0.60)
            reasoning_parts.append(f"3rd:{third_hint[:18]}")

        if subject_label != "speaker" and opp_hint:
            if (
                _REPORTING_VERBS_RE.search(lower)
                or "according to" in lower
                or "as he said" in lower
                or "as she said" in lower
            ):
                subject_label = "opponent"
                confidence = max(confidence, 0.60)
                reasoning_parts.append(f"opp_report:{opp_hint[:18]}")
            else:
                if has_first_person:
                    subject_label = "speaker"
                    confidence = max(confidence, 0.56)
                    reasoning_parts.append(f"opp_mention_but_fp:{opp_hint[:18]}")
                else:
                    subject_label = "ambiguous"
                    confidence = max(confidence, 0.50)
                    reasoning_parts.append(f"opp_mention_weak:{opp_hint[:18]}")

        if subject_label == "ambiguous" and has_first_person:
            subject_label = "speaker"
            confidence = max(confidence, 0.54)
            reasoning_parts.append("fp_default")

        if inside_quotes:
            reasoning_parts.append("quoted")
            if subject_label == "speaker":
                subject_label = "third_party"
            if commitment_type == "commitment":
                commitment_type = "quoted"
            confidence = max(0.40, confidence - 0.12)

        if subject_label != "speaker" and commitment_type == "commitment":
            commitment_type = "reported"
            commitment_score = max(0.10, commitment_score - 0.10)
            reasoning_parts.append("commitment->reported")

        if commitment_type == "commitment" and any(
            re.search(rf"\b{re.escape(w)}\b", lower) for w in NEGATION_WORDS
        ):
            commitment_type = "negated"
            reasoning_parts.append("negated_lex")

        if commitment_type == "none":
            confidence = min(confidence, 0.60)

        return AttributionResult(
            subject=subject_label,
            confidence=_clamp01(confidence),
            commitment=_clamp01(commitment_score),
            commitment_type=commitment_type,
            inside_quotes=inside_quotes,
            verb=None,
            subject_text=None,
            reasoning=";".join(reasoning_parts),
            span=evidence_span,
            subject_span=None,
            trigger_span=None,
        )


_DEFAULT_ATTRIBUTION_PARSER: Optional[AttributionParser] = None


def get_attribution_parser() -> AttributionParser:
    global _DEFAULT_ATTRIBUTION_PARSER
    if _DEFAULT_ATTRIBUTION_PARSER is None:
        _DEFAULT_ATTRIBUTION_PARSER = AttributionParser()
    return _DEFAULT_ATTRIBUTION_PARSER


def parse_attribution(text: str, evidence_span: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    return get_attribution_parser().parse(text, evidence_span).to_dict()


__all__ = [
    "AttributionResult",
    "AttributionParser",
    "get_attribution_parser",
    "parse_attribution",
]