from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_MARPOR_AVAILABLE = False
try:
    from app.services.marpor_definitions import hybrid_marpor_analyzer  # type: ignore

    _MARPOR_AVAILABLE = True
except Exception:
    hybrid_marpor_analyzer = None  # type: ignore
    _MARPOR_AVAILABLE = False

_ROUTER_AVAILABLE = False
_get_llm_router = None
_LLMRouter = None

try:
    from app.services.llm_router import get_llm_router  # type: ignore

    _get_llm_router = get_llm_router
    _ROUTER_AVAILABLE = True
except Exception:
    pass

try:
    from app.services.llm_router import LLMRouter  # type: ignore

    _LLMRouter = LLMRouter
    _ROUTER_AVAILABLE = True
except Exception:
    pass

_RAW_CODE_PATTERN = re.compile(r"\b[1-7]\d{2}\b")


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _extract_family_scores(ideology_result: Dict[str, Any]) -> Dict[str, float]:
    if isinstance(ideology_result.get("scores"), dict):
        s = ideology_result["scores"]
        lib = _safe_float(s.get("Libertarian", 0.0))
        auth = _safe_float(s.get("Authoritarian", 0.0))
        cen = _safe_float(s.get("Centrist", 0.0))
    else:
        lib = _safe_float(ideology_result.get("libertarianism_percentage", 0.0))
        auth = _safe_float(ideology_result.get("authoritarianism_percentage", 0.0))
        cen = _safe_float(ideology_result.get("neutral_percentage", 0.0))

    mx = max(abs(lib), abs(auth), abs(cen))
    if 0.0 <= mx <= 1.05:
        lib *= 100.0
        auth *= 100.0
        cen *= 100.0

    return {"Libertarian": float(lib), "Authoritarian": float(auth), "Centrist": float(cen)}


def _truncate(text: str, max_chars: int = 220) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "..."


def _ensure_question_mark(q: str) -> str:
    q = (q or "").strip()
    return q if q.endswith("?") else (q + "?" if q else "")


def _strip_quotes(s: str) -> str:
    return (s or "").strip().strip("\"'“”‘’")


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
    codes = [str(c).strip() for c in (codes or []) if str(c).strip()]
    if not codes:
        return []

    if _MARPOR_AVAILABLE and hybrid_marpor_analyzer is not None:
        desc: List[str] = []
        for c in codes:
            cat = getattr(hybrid_marpor_analyzer, "categories", {}).get(c) if hybrid_marpor_analyzer else None
            if cat:
                label = str(getattr(cat, "label", "") or "").strip()
                if label:
                    desc.append(f"{label} indicators")
        return _unique_preserve_order(desc)

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
        d = code_map.get(c)
        if d:
            out.append(d)
    return _unique_preserve_order(out)


async def _call_llm_generate(llm: Any, prompt: str) -> str:
    if llm is None:
        raise RuntimeError("LLM instance is None")

    for name in ("generate", "invoke", "complete", "chat"):
        fn = getattr(llm, name, None)
        if callable(fn):
            out = fn(prompt)
            if inspect.isawaitable(out):
                out = await out
            return out if isinstance(out, str) else str(out)

    raise RuntimeError("LLM router does not expose a usable generate method.")


def _get_router(provider: str, model: str, temperature: float, max_tokens: int) -> Any:
    if not _ROUTER_AVAILABLE:
        raise RuntimeError("llm_router not available")

    if _get_llm_router is not None:
        llm = None
        try:
            sig = inspect.signature(_get_llm_router)
            kwargs: Dict[str, Any] = {"provider": provider, "model": model, "temperature": float(temperature)}
            if "max_tokens" in sig.parameters:
                kwargs["max_tokens"] = int(max_tokens)
            llm = _get_llm_router(**kwargs)
        except Exception:
            llm = _get_llm_router(provider=provider, model=model, temperature=float(temperature))

        try:
            if llm is not None and hasattr(llm, "max_tokens"):
                setattr(llm, "max_tokens", int(max_tokens))
        except Exception:
            pass

        if llm is None:
            raise RuntimeError("get_llm_router returned None")

        return llm

    if _LLMRouter is not None:
        return _LLMRouter(provider=provider, model=model, temperature=float(temperature), max_tokens=int(max_tokens))

    raise RuntimeError("No valid LLM router entrypoint found")


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
        max_questions: int = 3,
    ) -> List[str]:
        qt = (question_type or "").strip().lower()
        if qt not in ("journalistic", "technical"):
            qt = "journalistic"

        max_q = max(1, min(int(max_questions or 3), 8))

        scores = _extract_family_scores(ideology_result)
        lib_pct = float(scores["Libertarian"])
        auth_pct = float(scores["Authoritarian"])
        neu_pct = float(scores["Centrist"])

        ideology_family = ideology_result.get("ideology_family") or ideology_result.get("ideological_lean") or "Unknown"
        ideology_subtype = (
            ideology_result.get("ideology_subtype")
            or ideology_result.get("libertarian_subtype")
            or ideology_result.get("authoritarian_subtype")
        )

        confidence_raw = ideology_result.get("confidence_score", ideology_result.get("confidence", 0.0))
        conf = _safe_float(confidence_raw, 0.0)
        if conf > 1.0:
            conf = conf / 100.0
        conf = _clamp01(conf)

        marpor_codes = ideology_result.get("marpor_codes", ideology_result.get("primary_codes", [])) or []
        indicators = _code_to_natural_description(list(marpor_codes)[:6])

        key_statements: List[str] = []
        for seg in (key_segments or [])[:3]:
            t = seg.get("text") or seg.get("full_text") or ""
            t = _truncate(str(t), 220)
            if t:
                key_statements.append(t)

        if qt == "journalistic":
            prompt = self._build_journalistic_prompt(
                speech_title=str(speech_title or ""),
                speaker=str(speaker or ""),
                ideology_family=str(ideology_family),
                ideology_subtype=str(ideology_subtype) if ideology_subtype else None,
                lib_pct=lib_pct,
                auth_pct=auth_pct,
                neu_pct=neu_pct,
                confidence=conf,
                key_statements=key_statements,
                indicators=indicators,
                max_questions=max_q,
            )
            temperature = 0.7
            min_words = 20
        else:
            prompt = self._build_technical_prompt(
                speech_title=str(speech_title or ""),
                speaker=str(speaker or ""),
                ideology_family=str(ideology_family),
                ideology_subtype=str(ideology_subtype) if ideology_subtype else None,
                lib_pct=lib_pct,
                auth_pct=auth_pct,
                neu_pct=neu_pct,
                confidence=conf,
                key_statements=key_statements,
                indicators=indicators,
                max_questions=max_q,
            )
            temperature = 0.4
            min_words = 25

        try:
            llm = _get_router(
                provider=str(llm_provider or ""),
                model=str(llm_model or ""),
                temperature=temperature,
                max_tokens=800,
            )
            response = await _call_llm_generate(llm, prompt)
        except Exception as e:
            logger.error("LLM question generation failed: %s", e, exc_info=True)
            return self._fallback_questions(
                qt,
                str(speech_title or ""),
                str(speaker or ""),
                str(ideology_family),
                str(ideology_subtype) if ideology_subtype else None,
                scores,
                key_statements,
                max_q,
            )

        questions = self._parse_questions_response(response, max_questions=max_q)
        questions = self._validate_questions(questions, question_type=qt, min_words=min_words)

        if questions:
            return questions

        return self._fallback_questions(
            qt,
            str(speech_title or ""),
            str(speaker or ""),
            str(ideology_family),
            str(ideology_subtype) if ideology_subtype else None,
            scores,
            key_statements,
            max_q,
        )

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
        max_questions: int,
    ) -> str:
        ideology_desc = ideology_family
        if ideology_subtype and ideology_subtype.lower() != "none":
            ideology_desc = f"{ideology_family} (subtype: {ideology_subtype})"

        conf_label = "high" if confidence >= 0.80 else ("moderate" if confidence >= 0.60 else "low")

        statements_block = ""
        if key_statements:
            statements_block = "\nKey statements:\n" + "\n".join([f'- "{s}"' for s in key_statements])

        indicator_block = ""
        if indicators:
            indicator_block = "\nTop ideological indicators detected:\n" + "\n".join([f"- {d}" for d in indicators])

        return (
            f'You are a political journalist writing for a general audience.\n\n'
            f'Speech:\n- Title: "{speech_title}"\n- Speaker: {speaker}\n\n'
            f'Analysis summary (do NOT mention any coding schemes or numeric category codes):\n'
            f"- Classification: {ideology_desc}\n"
            f"- Breakdown: {lib_pct:.1f}% libertarian-leaning signals, {auth_pct:.1f}% authoritarian-leaning signals, {neu_pct:.1f}% centrist/neutral signals\n"
            f"- Confidence: {conf_label} ({confidence*100:.1f}%)"
            f"{indicator_block}{statements_block}\n\n"
            f"Task:\nGenerate {max_questions} journalistic questions that help a general audience understand the implications of this speech.\n\n"
            f"Rules:\n"
            f"1) Do not mention MARPOR or any numeric codes.\n"
            f"2) Each question must be at least 20 words and end with a question mark.\n"
            f"3) Each question must reference the speaker or the speech content (not generic).\n"
            f"4) Focus on implications: what this suggests, how it could affect people, politics, policy, or upcoming debates.\n\n"
            f'Return ONLY a JSON array of strings:\n["Question 1?", "Question 2?", "Question 3?"]'
        )

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
        max_questions: int,
    ) -> str:
        ideology_desc = ideology_family
        if ideology_subtype and ideology_subtype.lower() != "none":
            ideology_desc = f"{ideology_family} (subtype: {ideology_subtype})"

        statements_block = ""
        if key_statements:
            statements_block = "\nKey statements (evidence excerpts):\n" + "\n".join([f'- "{s}"' for s in key_statements])

        indicator_block = ""
        if indicators:
            indicator_block = "\nTop ideological indicators detected (natural language only):\n" + "\n".join([f"- {d}" for d in indicators])

        return (
            f"You are a political science researcher designing analytical questions about a speech classification.\n\n"
            f'Speech:\n- Title: "{speech_title}"\n- Speaker: {speaker}\n\n'
            f"Analysis summary:\n"
            f"- Classification: {ideology_desc}\n"
            f"- Breakdown: {lib_pct:.1f}% libertarian-leaning indicators, {auth_pct:.1f}% authoritarian-leaning indicators, {neu_pct:.1f}% centrist/neutral indicators\n"
            f"- Confidence: {confidence*100:.1f}%"
            f"{indicator_block}{statements_block}\n\n"
            f"Task:\nGenerate {max_questions} technical questions that probe evidence strength, alternative interpretations, and methodological sensitivity.\n\n"
            f"Rules:\n"
            f"1) Do not mention any numeric category codes or “MARPOR”.\n"
            f"2) Each question must be at least 25 words and end with a question mark.\n"
            f"3) Each question must reference specific evidence cues (percentages, indicators, or key statement excerpts).\n"
            f"4) Prefer questions about alternative hypotheses, measurement validity, attribution/quotation ambiguity, and robustness.\n\n"
            f'Return ONLY a JSON array of strings:\n["Question 1?", "Question 2?", "Question 3?"]'
        )

    def _parse_questions_response(self, response: str, max_questions: int) -> List[str]:
        txt = (response or "").strip()

        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return self._clean_questions(obj, max_questions)
        except Exception:
            pass

        m = re.search(r"\[[\s\S]*\]", txt)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    return self._clean_questions(obj, max_questions)
            except Exception:
                pass

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
            s = _strip_quotes(q)
            s = _ensure_question_mark(s)
            if s:
                out.append(s)
        out = _unique_preserve_order(out)
        return out[: max(1, int(max_questions))]

    def _validate_questions(self, questions: List[str], *, question_type: str, min_words: int) -> List[str]:
        bad_generic = {
            "what does this mean",
            "is this good",
            "is this bad",
            "tell me more",
            "explain this",
        }

        valid: List[str] = []
        for q in questions:
            q0 = (q or "").strip()
            if not q0 or "?" not in q0:
                continue
            if len(q0.split()) < int(min_words):
                continue
            if _contains_raw_codes(q0):
                continue

            ql = q0.lower()
            if any(p in ql for p in bad_generic):
                continue

            if question_type == "journalistic":
                if any(t in ql for t in ("methodology", "robustness", "measurement", "sensitivity analysis")):
                    continue

            valid.append(q0)

        return _unique_preserve_order(valid)

    def _fallback_questions(
        self,
        question_type: str,
        speech_title: str,
        speaker: str,
        ideology_family: str,
        ideology_subtype: Optional[str],
        scores: Dict[str, float],
        key_statements: List[str],
        max_questions: int,
    ) -> List[str]:
        lib_pct = float(scores.get("Libertarian", 0.0) or 0.0)
        auth_pct = float(scores.get("Authoritarian", 0.0) or 0.0)
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
                qs.append(
                    _ensure_question_mark(
                        f'How should the excerpt "{key}" be interpreted in terms of ideological commitment strength, and what alternative ideological reading could plausibly fit the same passage'
                    )
                )
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
                qs.append(
                    _ensure_question_mark(
                        f'When {speaker} said "{key}", what real-world policy changes do you think they were preparing the public to accept or debate'
                    )
                )

        return _unique_preserve_order(qs)[: max(1, int(max_questions))]


question_generator = QuestionGenerator()

__all__ = ["QuestionGenerator", "question_generator"]