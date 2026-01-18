"""
SCIENTIFIC QUESTION GENERATOR FOR POLITICAL DISCOURSE ANALYSIS
==============================================================

LOCATION: backend/app/services/question_generator.py
ACTION: Replace existing file with this complete version

Key features:
-------------
- Two question modes:
  1) journalistic (public-facing)
  2) technical (research-facing)
- Uses user's selected LLM provider/model (via llm_router)
- Evidence-grounded: uses ideology_result + key_segments
- Hides raw MARPOR codes (never outputs 3-digit codes)
- Robust JSON parsing with safe fallbacks
- Works with both sync and async LLM routers

Expected input formats:
-----------------------
ideology_result can be produced by your scoring pipeline and may include either:
- "scores": {"Libertarian":.., "Authoritarian":.., "Centrist":..}
  OR
- "libertarianism_percentage", "authoritarianism_percentage", "neutral_percentage"

key_segments:
- list of dicts, each containing at least {"text": "..."} and optional metadata

IMPORTANT FIX INCLUDED:
----------------------
Your llm_router.get_llm_router(...) in the version you shared earlier accepts:
    get_llm_router(provider, model, temperature=0.3)
and does NOT accept max_tokens.

This file now handles both cases safely:
- If factory supports max_tokens, it passes it
- Otherwise, it sets max_tokens on the returned router instance (if available)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import inspect
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# OPTIONAL: MARPOR LABEL MAPPING (for natural-language indicator descriptions)
# =============================================================================

_MARPOR_AVAILABLE = False
try:
    from app.services.marpor_definitions import hybrid_marpor_analyzer  # type: ignore
    _MARPOR_AVAILABLE = True
except Exception:
    hybrid_marpor_analyzer = None  # type: ignore
    _MARPOR_AVAILABLE = False


# =============================================================================
# OPTIONAL: LLM ROUTER IMPORTS (multi-provider)
# =============================================================================

_ROUTER_AVAILABLE = False
_get_llm_router = None
_LLMRouter = None

try:
    # Prefer factory if present
    from app.services.llm_router import get_llm_router  # type: ignore
    _get_llm_router = get_llm_router
    _ROUTER_AVAILABLE = True
except Exception:
    pass

try:
    # Fallback: class-based router
    from app.services.llm_router import LLMRouter  # type: ignore
    _LLMRouter = LLMRouter
    _ROUTER_AVAILABLE = True
except Exception:
    pass


# =============================================================================
# UTILITIES
# =============================================================================

_RAW_CODE_PATTERN = re.compile(r"\b[1-7]\d{2}\b")  # e.g., 201, 401, 605


def _safe_percent(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _extract_family_scores(ideology_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Accepts either:
    - ideology_result["scores"] = {"Libertarian":..,"Authoritarian":..,"Centrist":..}
    OR:
    - ideology_result["libertarianism_percentage"], etc.
    """
    if isinstance(ideology_result.get("scores"), dict):
        s = ideology_result["scores"]
        return {
            "Libertarian": _safe_percent(s.get("Libertarian", 0.0)),
            "Authoritarian": _safe_percent(s.get("Authoritarian", 0.0)),
            "Centrist": _safe_percent(s.get("Centrist", 0.0)),
        }

    return {
        "Libertarian": _safe_percent(ideology_result.get("libertarianism_percentage", 0.0)),
        "Authoritarian": _safe_percent(ideology_result.get("authoritarianism_percentage", 0.0)),
        "Centrist": _safe_percent(ideology_result.get("neutral_percentage", 0.0)),
    }


def _truncate(text: str, max_chars: int = 220) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "..."


def _ensure_question_mark(q: str) -> str:
    q = (q or "").strip()
    if not q.endswith("?"):
        q += "?"
    return q


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    return s.strip("\"'“”‘’")


def _contains_raw_codes(s: str) -> bool:
    return bool(_RAW_CODE_PATTERN.search(s or ""))


def _unique_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def _code_to_natural_description(codes: List[str]) -> List[str]:
    """
    Convert MARPOR codes to natural-language indicators WITHOUT revealing codes.
    Uses marpor_definitions categories if available; otherwise uses a curated mapping.
    """
    codes = [str(c).strip() for c in (codes or []) if str(c).strip()]
    if not codes:
        return []

    # Prefer analyzer labels (no codes shown)
    if _MARPOR_AVAILABLE and hybrid_marpor_analyzer is not None:
        desc: List[str] = []
        for c in codes:
            cat = hybrid_marpor_analyzer.categories.get(c)
            if cat:
                desc.append(f"{cat.label} indicators")
        return _unique_preserve_order(desc)

    # Fallback static mapping
    code_map = {
        "201": "individual freedom and civil liberties indicators",
        "203": "constitutional limits and rule-of-law indicators",
        "301": "decentralization and local autonomy indicators",
        "401": "free market and economic freedom indicators",
        "407": "free trade and open market indicators",
        "414": "tax reduction and fiscal responsibility indicators",
        "505": "welfare limitation and self-reliance indicators",
        "507": "school choice and educational freedom indicators",
        "604": "personal autonomy and social freedom indicators",
        "702": "labor flexibility and individual bargaining indicators",
        "302": "centralization and strong state capacity indicators",
        "305": "political authority and strong leadership indicators",
        "404": "economic planning and state coordination indicators",
        "412": "state economic control indicators",
        "413": "nationalization and public ownership indicators",
        "504": "welfare expansion and social program indicators",
        "603": "traditional morality and values indicators",
        "605": "law-and-order and enforcement indicators",
        "701": "collective labor and union-strength indicators",
        "501": "environmental protection and sustainability indicators",
        "503": "equality and social justice indicators",
        "608": "assimilation / anti-multiculturalism indicators",
        "601": "national identity and tradition indicators",
    }

    out: List[str] = []
    for c in codes:
        if c in code_map:
            out.append(code_map[c])
    return _unique_preserve_order(out)


async def _call_llm_generate(llm: Any, prompt: str) -> str:
    """
    Works with both sync and async router implementations.
    """
    if llm is None:
        raise RuntimeError("LLM instance is None")

    # Async generate
    if hasattr(llm, "generate") and asyncio.iscoroutinefunction(llm.generate):
        return await llm.generate(prompt)

    # Sync generate
    if hasattr(llm, "generate") and callable(llm.generate):
        return llm.generate(prompt)

    # Alternative method names
    for name in ("invoke", "complete", "chat"):
        if hasattr(llm, name) and callable(getattr(llm, name)):
            out = getattr(llm, name)(prompt)
            return out if isinstance(out, str) else str(out)

    raise RuntimeError("LLM router does not expose a usable generate method.")


def _get_router(provider: str, model: str, temperature: float, max_tokens: int) -> Any:
    """
    Construct LLM router with user's chosen provider/model.

    Supports either:
    - get_llm_router(provider=..., model=..., temperature=...)  [your current router]
    - get_llm_router(provider=..., model=..., temperature=..., max_tokens=...) [future-compatible]
    - LLMRouter(provider=..., model=..., temperature=..., max_tokens=...)

    Also: if the factory does not accept max_tokens, we set it on the instance if possible.
    """
    if not _ROUTER_AVAILABLE:
        raise RuntimeError("llm_router not available in backend/app/services/llm_router.py")

    # Prefer factory if present
    if _get_llm_router is not None:
        try:
            sig = inspect.signature(_get_llm_router)
            kwargs: Dict[str, Any] = {"provider": provider, "model": model, "temperature": temperature}
            if "max_tokens" in sig.parameters:
                kwargs["max_tokens"] = max_tokens
            llm = _get_llm_router(**kwargs)
        except TypeError:
            # If signature inspection fails or params mismatch, try minimal call
            llm = _get_llm_router(provider=provider, model=model, temperature=temperature)

        # If factory does not support max_tokens, attempt to set on instance
        try:
            if hasattr(llm, "max_tokens"):
                setattr(llm, "max_tokens", max_tokens)
        except Exception:
            pass

        return llm

    # Fallback to class-based router
    if _LLMRouter is not None:
        return _LLMRouter(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)

    raise RuntimeError("No valid LLM router entrypoint found.")


# =============================================================================
# QUESTION GENERATOR
# =============================================================================

class QuestionGenerator:
    def __init__(self) -> None:
        logger.info("QuestionGenerator initialized")

    async def generate_questions_with_llm(
        self,
        question_type: str,
        speech_title: str,
        speaker: str,
        ideology_result: Dict[str, Any],
        key_segments: List[Dict[str, Any]],
        llm_provider: str,
        llm_model: str,
        max_questions: int = 3
    ) -> List[str]:
        """
        Generate questions using user's selected LLM (provider/model).

        max_questions:
        - default 3
        - min 1
        - max 8 (hard cap)
        """
        question_type = (question_type or "").strip().lower()
        if question_type not in ("journalistic", "technical"):
            question_type = "journalistic"

        max_questions = max(1, min(int(max_questions or 3), 8))

        # Extract scores and subtype labels
        scores = _extract_family_scores(ideology_result)
        lib_pct = scores["Libertarian"]
        auth_pct = scores["Authoritarian"]
        neu_pct = scores["Centrist"]

        ideology_family = ideology_result.get("ideology_family") or ideology_result.get("ideological_lean") or "Unknown"
        ideology_subtype = ideology_result.get("ideology_subtype") or ideology_result.get("libertarian_subtype") or ideology_result.get("authoritarian_subtype")

        confidence = ideology_result.get("confidence_score", ideology_result.get("confidence", 0.0))
        confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        if confidence > 1.0:
            confidence = confidence / 100.0  # if passed as 0-100

        marpor_codes = ideology_result.get("marpor_codes", ideology_result.get("primary_codes", [])) or []
        indicators = _code_to_natural_description(marpor_codes[:6])

        # Key statements (top 3)
        key_statements: List[str] = []
        for seg in (key_segments or [])[:3]:
            t = seg.get("text") or seg.get("full_text") or ""
            t = _truncate(str(t), 220)
            if t:
                key_statements.append(t)

        if question_type == "journalistic":
            prompt = self._build_journalistic_prompt(
                speech_title=speech_title,
                speaker=speaker,
                ideology_family=str(ideology_family),
                ideology_subtype=str(ideology_subtype) if ideology_subtype else None,
                lib_pct=lib_pct,
                auth_pct=auth_pct,
                neu_pct=neu_pct,
                confidence=confidence,
                key_statements=key_statements,
                indicators=indicators,
                max_questions=max_questions
            )
        else:
            prompt = self._build_technical_prompt(
                speech_title=speech_title,
                speaker=speaker,
                ideology_family=str(ideology_family),
                ideology_subtype=str(ideology_subtype) if ideology_subtype else None,
                lib_pct=lib_pct,
                auth_pct=auth_pct,
                neu_pct=neu_pct,
                confidence=confidence,
                key_statements=key_statements,
                indicators=indicators,
                max_questions=max_questions
            )

        # Generate via router
        try:
            llm = _get_router(
                provider=llm_provider,
                model=llm_model,
                temperature=0.7 if question_type == "journalistic" else 0.4,
                max_tokens=800
            )
            response = await _call_llm_generate(llm, prompt)
        except Exception as e:
            logger.error(f"LLM question generation failed: {e}", exc_info=True)
            return self._fallback_questions(
                question_type, speech_title, speaker, str(ideology_family),
                str(ideology_subtype) if ideology_subtype else None,
                scores, key_statements, max_questions
            )

        # Parse + validate
        questions = self._parse_questions_response(response, max_questions=max_questions)
        questions = self._validate_questions(
            questions,
            question_type=question_type,
            speaker=speaker,
            min_words=20 if question_type == "journalistic" else 25
        )

        if questions:
            return questions

        return self._fallback_questions(
            question_type, speech_title, speaker, str(ideology_family),
            str(ideology_subtype) if ideology_subtype else None,
            scores, key_statements, max_questions
        )

    # ---------------------------------------------------------------------
    # PROMPTS
    # ---------------------------------------------------------------------

    def _build_journalistic_prompt(
        self,
        *,
        speech_title: str,
        speaker: str,
        ideology_family: str,
        ideology_subtype: Optional[str],
        lib_pct: float,
        auth_pct: float,
        neu_pct: float,
        confidence: float,
        key_statements: List[str],
        indicators: List[str],
        max_questions: int
    ) -> str:
        ideology_desc = ideology_family
        if ideology_subtype and ideology_subtype.lower() != "none":
            ideology_desc = f"{ideology_family} (subtype: {ideology_subtype})"

        conf_label = "high" if confidence >= 0.80 else "moderate" if confidence >= 0.60 else "low"

        statements_block = ""
        if key_statements:
            statements_block = "\nKey statements:\n" + "\n".join([f"- \"{s}\"" for s in key_statements])

        indicator_block = ""
        if indicators:
            indicator_block = "\nTop ideological indicators detected:\n" + "\n".join([f"- {d}" for d in indicators])

        return f"""
You are a political journalist writing for a general audience.

Speech:
- Title: "{speech_title}"
- Speaker: {speaker}

Analysis summary (do NOT mention any coding schemes or numeric category codes):
- Classification: {ideology_desc}
- Breakdown: {lib_pct:.1f}% libertarian-leaning signals, {auth_pct:.1f}% authoritarian-leaning signals, {neu_pct:.1f}% centrist/neutral signals
- Confidence: {conf_label} ({confidence*100:.1f}%){indicator_block}{statements_block}

Task:
Generate {max_questions} journalistic questions that help a general audience understand the implications of this speech.

Rules:
1) Do not mention MARPOR or any numeric codes.
2) Each question must be at least 20 words and end with a question mark.
3) Each question must reference the speaker or the speech content (not generic).
4) Focus on implications: what this suggests, how it could affect people, politics, policy, or upcoming debates.

Return ONLY a JSON array of strings:
["Question 1?", "Question 2?", "Question 3?"]
""".strip()

    def _build_technical_prompt(
        self,
        *,
        speech_title: str,
        speaker: str,
        ideology_family: str,
        ideology_subtype: Optional[str],
        lib_pct: float,
        auth_pct: float,
        neu_pct: float,
        confidence: float,
        key_statements: List[str],
        indicators: List[str],
        max_questions: int
    ) -> str:
        ideology_desc = ideology_family
        if ideology_subtype and ideology_subtype.lower() != "none":
            ideology_desc = f"{ideology_family} (subtype: {ideology_subtype})"

        statements_block = ""
        if key_statements:
            statements_block = "\nKey statements (evidence excerpts):\n" + "\n".join([f"- \"{s}\"" for s in key_statements])

        indicator_block = ""
        if indicators:
            indicator_block = "\nTop ideological indicators detected (natural language only):\n" + "\n".join([f"- {d}" for d in indicators])

        return f"""
You are a political science researcher designing analytical questions about a speech classification.

Speech:
- Title: "{speech_title}"
- Speaker: {speaker}

Analysis summary:
- Classification: {ideology_desc}
- Breakdown: {lib_pct:.1f}% libertarian-leaning indicators, {auth_pct:.1f}% authoritarian-leaning indicators, {neu_pct:.1f}% centrist/neutral indicators
- Confidence: {confidence*100:.1f}%{indicator_block}{statements_block}

Task:
Generate {max_questions} technical questions that probe evidence strength, alternative interpretations, and methodological sensitivity.

Rules:
1) Do not mention any numeric category codes or “MARPOR”.
2) Each question must be at least 25 words and end with a question mark.
3) Each question must reference specific evidence cues (percentages, indicators, or key statement excerpts).
4) Prefer questions about alternative hypotheses, measurement validity, attribution/quotation ambiguity, and robustness.

Return ONLY a JSON array of strings:
["Question 1?", "Question 2?", "Question 3?"]
""".strip()

    # ---------------------------------------------------------------------
    # PARSING / VALIDATION
    # ---------------------------------------------------------------------

    def _parse_questions_response(self, response: str, max_questions: int) -> List[str]:
        txt = (response or "").strip()

        # 1) direct JSON
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return self._clean_questions(obj, max_questions)
        except Exception:
            pass

        # 2) extract first JSON array
        m = re.search(r"\[[\s\S]*\]", txt)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    return self._clean_questions(obj, max_questions)
            except Exception:
                pass

        # 3) line-based fallback
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        qs: List[str] = []
        for line in lines:
            line = re.sub(r"^\s*[\-\*\d\.\)\]]+\s*", "", line).strip()
            line = _strip_quotes(line)
            if "?" in line and len(line) > 20:
                qs.append(_ensure_question_mark(line))

        return self._clean_questions(qs, max_questions)

    def _clean_questions(self, questions: List[Any], max_questions: int) -> List[str]:
        out: List[str] = []
        for q in questions:
            if not isinstance(q, str):
                continue
            q = _strip_quotes(q)
            q = _ensure_question_mark(q)
            if not q:
                continue
            out.append(q)

        out = _unique_preserve_order(out)
        return out[:max_questions]

    def _validate_questions(self, questions: List[str], *, question_type: str, speaker: str, min_words: int) -> List[str]:
        valid: List[str] = []

        generic_bad = [
            "what does this mean",
            "is this good",
            "is this bad",
            "tell me more",
            "explain this",
        ]

        for q in questions:
            q0 = q.strip()
            ql = q0.lower()

            # must be question
            if "?" not in q0:
                continue

            # length (word count)
            if len(q0.split()) < int(min_words):
                continue

            # do not leak codes
            if _contains_raw_codes(q0):
                continue

            # avoid generic templates
            if any(p in ql for p in generic_bad):
                continue

            # journalistic should not be methodology-heavy
            if question_type == "journalistic":
                if any(t in ql for t in ["methodology", "robustness", "measurement", "sensitivity analysis"]):
                    continue

            # soft check: reference speaker/speech (do not hard-fail)
            # (kept permissive to avoid deleting good questions)

            valid.append(q0)

        return _unique_preserve_order(valid)

    # ---------------------------------------------------------------------
    # FALLBACKS
    # ---------------------------------------------------------------------

    def _fallback_questions(
        self,
        question_type: str,
        speech_title: str,
        speaker: str,
        ideology_family: str,
        ideology_subtype: Optional[str],
        scores: Dict[str, float],
        key_statements: List[str],
        max_questions: int
    ) -> List[str]:
        lib_pct = scores.get("Libertarian", 0.0)
        auth_pct = scores.get("Authoritarian", 0.0)

        subtype_part = f" (subtype: {ideology_subtype})" if ideology_subtype else ""
        key = _truncate(key_statements[0], 140) if key_statements else ""

        if question_type == "technical":
            qs = [
                _ensure_question_mark(
                    f"What specific speech evidence supports the {ideology_family}{subtype_part} classification, given the {lib_pct:.1f}% libertarian and {auth_pct:.1f}% authoritarian signal estimates"
                ),
                _ensure_question_mark(
                    "How sensitive is the classification to quotation or attribution ambiguity—could key passages be describing opponents rather than endorsing positions, and how would that change results"
                ),
            ]
            if key:
                qs.append(_ensure_question_mark(
                    f"How should the excerpt \"{key}\" be interpreted in terms of ideological commitment strength, and what alternative ideological reading could plausibly fit the same passage"
                ))
        else:
            qs = [
                _ensure_question_mark(
                    f"What does {speaker}'s {ideology_family.lower()}{subtype_part} pattern suggest about what they would prioritize if they had to choose between personal freedom and stronger government control"
                ),
                _ensure_question_mark(
                    "How might voters who care about everyday costs, safety, and rights interpret the balance between libertarian and authoritarian signals in this speech"
                ),
            ]
            if key:
                qs.append(_ensure_question_mark(
                    f"When {speaker} said \"{key}\", what real-world policy changes do you think they were preparing the public to accept or debate"
                ))

        return _unique_preserve_order(qs)[:max_questions]


# =============================================================================
# GLOBAL INSTANCE + EXPORTS
# =============================================================================

question_generator = QuestionGenerator()

__all__ = ["QuestionGenerator", "question_generator"]
