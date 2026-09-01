from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class UserLanguage(str, Enum):
    """Supported user-facing language tags for Pakistani legal chat."""

    EN_PK = "en-PK"
    UR_PK = "ur-PK"
    PA_PK = "pa-PK"
    MIXED = "mixed"


class ResponseStyle(str, Enum):
    """How the model should write, including script choice."""

    ENGLISH = "english"
    URDU_SCRIPT = "urdu_script"
    ROMAN_URDU = "roman_urdu"
    PUNJABI_ROMAN = "punjabi_roman"
    PUNJABI_SHAHMUKHI = "punjabi_shahmukhi"
    MIXED = "mixed"


class ScriptKind(str, Enum):
    LATIN = "latin"
    ARABIC = "arabic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class LanguageDetection:
    """Lightweight detection result used before RAG, not for retrieval translation."""

    language: UserLanguage
    confidence: float
    script: ScriptKind
    style: ResponseStyle
    user_language: UserLanguage
    response_language: UserLanguage
    response_style: ResponseStyle
    retrieval_language: str = "original"
    explicit_override: bool = False
    override_phrase: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "language": self.language.value,
            "confidence": round(self.confidence, 3),
            "script": self.script.value,
            "style": self.style.value,
            "user_language": self.user_language.value,
            "response_language": self.response_language.value,
            "response_style": self.response_style.value,
            "retrieval_language": self.retrieval_language,
            "explicit_override": self.explicit_override,
            "override_phrase": self.override_phrase,
        }


# English legal / Pakistani-law loanwords must not decide the language.
_LEGAL_TERMS = frozenset(
    {
        "fir",
        "ppc",
        "crpc",
        "cpc",
        "qso",
        "pld",
        "scmr",
        "mld",
        "ylr",
        "clc",
        "pcrlj",
        "bail",
        "section",
        "sec",
        "article",
        "accused",
        "procedure",
        "application",
        "file",
        "filed",
        "filing",
        "case",
        "cases",
        "statute",
        "statutes",
        "offence",
        "offense",
        "court",
        "courts",
        "remand",
        "challan",
        "warrant",
        "petition",
        "appeal",
        "constitution",
        "ordinance",
        "act",
        "rule",
        "rules",
        "judgment",
        "judgement",
        "precedent",
        "citation",
        "advocate",
        "lawyer",
        "judge",
        "hearing",
        "arrest",
        "murder",
        "theft",
        "fraud",
        "contract",
        "agreement",
        "injunction",
        "limitation",
        "plaint",
        "notice",
        "writ",
        "revision",
        "stay",
        "evidence",
        "arbitration",
        "inheritance",
        "succession",
        "pakistan",
        "pakistani",
        "penal",
        "code",
        "order",
        "qanun",
        "shahadat",
        "trial",
        "sentence",
        "conviction",
        "acquittal",
        "high",
        "supreme",
        "sessions",
        "session",
        "magistrate",
        "ipc",
        "complainant",
        "prosecution",
        "defence",
        "defense",
        "law",
        "legal",
        "source",
        "sources",
        "principle",
        "principles",
        "chances",
        "chance",
        "sc",
        "lhc",
        "shc",
        "ihc",
        "phc",
        "bhc",
        "fsc",
        "crp",
        "u",
        "s",
        "us",
        "no",
        "nos",
        "v",
        "vs",
        "versus",
    }
)

_ENGLISH_FUNCTION = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "to",
        "for",
        "with",
        "from",
        "by",
        "at",
        "in",
        "on",
        "this",
        "that",
        "these",
        "those",
        "what",
        "how",
        "when",
        "where",
        "why",
        "which",
        "who",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "you",
        "your",
        "we",
        "our",
        "i",
        "me",
        "my",
        "not",
        "yes",
        "about",
        "into",
        "than",
        "then",
        "does",
        "do",
        "did",
        "done",
        "must",
        "may",
        "might",
        "under",
        "it",
        "its",
        "or",
        "and",
        "if",
        "as",
        "so",
        "but",
        "also",
        "any",
        "there",
        "here",
        "please",
    }
)

_ENGLISH_CONTENT = frozenset(
    {
        "explain",
        "explains",
        "explained",
        "describe",
        "tell",
        "discuss",
        "define",
        "summarize",
        "summarise",
        "simple",
        "simply",
        "english",
        "detail",
        "details",
        "meaning",
        "definition",
        "whether",
        "available",
        "possible",
        "briefly",
        "clearly",
        "professional",
        "answer",
        "question",
        "help",
        "need",
        "want",
        "give",
        "given",
        "provide",
        "provided",
    }
)

# High-signal Roman Urdu (not shared with English).
_ROMAN_URDU = {
    "mujhe": 2.0,
    "mujhay": 2.0,
    "mujhy": 2.0,
    "batao": 2.0,
    "bataen": 2.0,
    "bataiye": 2.0,
    "bataye": 2.0,
    "batayen": 2.0,
    "chahiye": 2.0,
    "chahye": 2.0,
    "yeh": 1.5,
    "woh": 1.5,
    "kya": 2.0,
    "kia": 1.0,
    "nahi": 1.5,
    "nahin": 1.5,
    "karo": 1.5,
    "karein": 1.5,
    "karen": 1.0,
    "sakti": 2.0,
    "sakta": 2.0,
    "sakte": 2.0,
    "hai": 2.0,
    "hain": 2.0,
    "hoga": 1.5,
    "hogi": 1.5,
    "honge": 1.5,
    "tarika": 2.0,
    "tareeka": 2.0,
    "tareeqa": 2.0,
    "qanoon": 2.0,
    "qanooni": 2.0,
    "zamanat": 2.0,
    "zamaanat": 2.0,
    "dafa": 2.0,
    "dafaa": 2.0,
    "muqadma": 2.0,
    "muqadama": 2.0,
    "samjhao": 2.0,
    "samjhaen": 2.0,
    "kaise": 2.0,
    "kaisay": 2.0,
    "kyun": 1.5,
    "kyunke": 1.5,
    "lekin": 1.0,
    "agar": 1.0,
    "toh": 1.0,
    "wala": 1.0,
    "wali": 1.0,
    "wale": 1.0,
    "liye": 1.5,
    "liyay": 1.5,
    "aap": 1.0,
    "aapko": 1.5,
    "tumhe": 1.5,
    "unhe": 1.0,
    "uska": 1.5,
    "iski": 1.5,
    "iska": 1.5,
    "uski": 1.5,
    "ka": 1.0,
    "ki": 0.8,
    "ke": 1.0,
    "ko": 0.8,
    "se": 0.6,
    "mein": 1.5,
    "mey": 1.0,
    "me": 0.4,
    "kahan": 2.0,
    "kaha": 1.0,
    "honi": 0.6,
    "hona": 0.6,
}

# Pakistani Punjabi (Majhi/Shahmukhi romanization), not Indian Gurmukhi assumptions.
_ROMAN_PUNJABI = {
    "eh": 2.5,
    "ae": 2.5,
    "kithay": 3.0,
    "kithe": 3.0,
    "kithey": 3.0,
    "menu": 2.5,
    "mennu": 2.5,
    "mainu": 2.5,
    "tusi": 2.5,
    "tussi": 2.5,
    "daso": 2.5,
    "dasso": 2.5,
    "dass": 2.0,
    "sakda": 3.0,
    "sakdi": 3.0,
    "sakde": 2.5,
    "karni": 1.5,
    "karna": 1.0,
    "banda": 2.0,
    "bahar": 1.0,
    "baahar": 1.5,
    "vekho": 2.5,
    "wekho": 2.5,
    "wich": 2.0,
    "vich": 2.5,
    "naal": 2.0,
    "nal": 1.0,
    "kinj": 2.5,
    "kiven": 2.5,
    "kiwen": 2.5,
    "hun": 1.0,
    "fer": 1.5,
    "phir": 0.5,
    "di": 1.5,
    "da": 1.2,
    "de": 0.6,
    "nu": 1.5,
    "noon": 1.0,
    "othe": 2.0,
    "ethe": 2.0,
    "ithe": 2.0,
    "honi": 1.2,
    "hona": 1.0,
    "hone": 1.0,
    "aa": 0.8,
    "te": 1.0,
    "ki": 0.4,
    "qanoon": 0.5,
    "bare": 1.5,
}

_ARABIC_PUNJABI_MARKERS = (
    "ایہ",
    "کتھے",
    "کتھے",
    "مینوں",
    "مینو",
    "دسو",
    "تسی",
    "سکدا",
    "ہونا اے",
    "کری اے",
)

_ARABIC_TOKEN_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+")
_ARABIC_CHAR_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")

_CITATION_RE = re.compile(
    r"\b(?:PLD|SCMR|MLD|YLR|CLC|PCr\.?LJ)\s+\d{4}\s+[A-Za-z]+\s+\d+\b",
    re.IGNORECASE,
)
_SECTION_ENTITY_RE = re.compile(
    r"\b(?:section|sec\.?|article|u/s)\s+[0-9A-Za-z\-]+\b",
    re.IGNORECASE,
)

# Explicit user instruction wins over automatic detection.
_OVERRIDE_PATTERNS: tuple[tuple[re.Pattern[str], UserLanguage, ResponseStyle | None], ...] = (
    (
        re.compile(r"\b(?:in|into)\s+roman\s+urdu\b", re.I),
        UserLanguage.UR_PK,
        ResponseStyle.ROMAN_URDU,
    ),
    (
        re.compile(r"\b(?:in|into)\s+simple\s+english\b", re.I),
        UserLanguage.EN_PK,
        ResponseStyle.ENGLISH,
    ),
    (
        re.compile(r"\bexplain\s+this\s+in\s+english\b", re.I),
        UserLanguage.EN_PK,
        ResponseStyle.ENGLISH,
    ),
    (
        re.compile(r"\b(?:in|into)\s+english\b", re.I),
        UserLanguage.EN_PK,
        ResponseStyle.ENGLISH,
    ),
    (
        re.compile(r"\benglish\s+mein\b", re.I),
        UserLanguage.EN_PK,
        ResponseStyle.ENGLISH,
    ),
    (
        re.compile(r"\b(?:in|into)\s+simple\s+urdu\b", re.I),
        UserLanguage.UR_PK,
        None,
    ),
    (
        re.compile(r"\b(?:in|into)\s+urdu\b", re.I),
        UserLanguage.UR_PK,
        None,
    ),
    (
        re.compile(r"\burdu\s+mein\b", re.I),
        UserLanguage.UR_PK,
        None,
    ),
    (
        re.compile(r"اردو\s+میں"),
        UserLanguage.UR_PK,
        ResponseStyle.URDU_SCRIPT,
    ),
    (
        re.compile(r"انگریزی\s+میں"),
        UserLanguage.EN_PK,
        ResponseStyle.ENGLISH,
    ),
    (
        re.compile(r"\b(?:in|into)\s+punjabi\b", re.I),
        UserLanguage.PA_PK,
        None,
    ),
    (
        re.compile(r"\bpunjabi\s+(?:mein|vich|wich)\b", re.I),
        UserLanguage.PA_PK,
        None,
    ),
)


def detect_language(text: str) -> LanguageDetection:
    """
    Detect dominant Pakistani legal-chat language without translating the query.

    Roman Urdu/Punjabi are lexicon-based. English legal terms are ignored so
    code-switched messages are not classified as English by accident.
    """
    original = text or ""
    stripped = original.strip()
    if not stripped:
        return LanguageDetection(
            language=UserLanguage.EN_PK,
            confidence=0.2,
            script=ScriptKind.UNKNOWN,
            style=ResponseStyle.ENGLISH,
            user_language=UserLanguage.EN_PK,
            response_language=UserLanguage.EN_PK,
            response_style=ResponseStyle.ENGLISH,
        )

    override_lang, override_style, override_phrase, remainder = _extract_override(
        stripped
    )
    script = _detect_script(stripped)
    scores = _score_languages(remainder)
    user_language, style, confidence = _decide_user_language(scores, script)

    response_language = override_lang or user_language
    if response_language == UserLanguage.MIXED and override_lang is None:
        response_language = _dominant_from_scores(scores, script)
        response_style = _style_for(response_language, script, scores)
        # Mixed replies should still mirror the user's code-switching.
        response_style = ResponseStyle.MIXED
        style = ResponseStyle.MIXED
    else:
        response_style = override_style or _style_for(
            response_language,
            script,
            scores,
        )
        if override_style is None and response_language == UserLanguage.UR_PK:
            response_style = _urdu_style_from_script(script, scores)
        if override_style is None and response_language == UserLanguage.PA_PK:
            response_style = (
                ResponseStyle.PUNJABI_SHAHMUKHI
                if script == ScriptKind.ARABIC
                else ResponseStyle.PUNJABI_ROMAN
            )

    return LanguageDetection(
        language=user_language,
        confidence=confidence,
        script=script,
        style=style,
        user_language=user_language,
        response_language=response_language,
        response_style=response_style,
        retrieval_language="original",
        explicit_override=override_lang is not None,
        override_phrase=override_phrase,
    )


def _extract_override(
    text: str,
) -> tuple[UserLanguage | None, ResponseStyle | None, str | None, str]:
    remainder = text
    for pattern, language, style in _OVERRIDE_PATTERNS:
        match = pattern.search(text)
        if match:
            remainder = pattern.sub(" ", text)
            remainder = re.sub(r"\s+", " ", remainder).strip(" :,-")
            return language, style, match.group(0), remainder
    return None, None, None, text


def _detect_script(text: str) -> ScriptKind:
    arabic = len(_ARABIC_CHAR_RE.findall(text))
    latin = len(_LATIN_CHAR_RE.findall(text))
    if arabic and latin:
        return ScriptKind.MIXED
    if arabic:
        return ScriptKind.ARABIC
    if latin:
        return ScriptKind.LATIN
    return ScriptKind.UNKNOWN


def _score_languages(text: str) -> dict[str, float]:
    cleaned = _CITATION_RE.sub(" ", text)
    cleaned = _SECTION_ENTITY_RE.sub(" ", cleaned)
    latin_tokens = [tok.lower() for tok in _LATIN_TOKEN_RE.findall(cleaned)]
    arabic_tokens = _ARABIC_TOKEN_RE.findall(cleaned)

    scores = {"en": 0.0, "ur": 0.0, "pa": 0.0, "en_content": 0.0}

    for token in latin_tokens:
        if token in _LEGAL_TERMS or token.isdigit():
            continue
        if token in _ROMAN_PUNJABI:
            scores["pa"] += _ROMAN_PUNJABI[token]
        if token in _ROMAN_URDU:
            scores["ur"] += _ROMAN_URDU[token]
        if token in _ENGLISH_CONTENT:
            scores["en"] += 1.6
            scores["en_content"] += 1.6
        elif token in _ENGLISH_FUNCTION:
            # "is" / "in" are also Roman Urdu; only count as English
            # when the token is not already a strong local marker.
            if token in {"is", "in", "me", "us", "no"} and (
                scores["ur"] > 0 or scores["pa"] > 0
            ):
                continue
            scores["en"] += 0.7
        elif len(token) >= 4 and token.isalpha():
            # Unknown Latin content word: weak English, unless local markers dominate.
            scores["en"] += 0.35

    if arabic_tokens:
        arabic_text = " ".join(arabic_tokens)
        if any(marker in arabic_text for marker in _ARABIC_PUNJABI_MARKERS):
            scores["pa"] += 3.0 + (0.8 * len(arabic_tokens))
        else:
            scores["ur"] += 2.5 + (0.9 * len(arabic_tokens))

    return scores


def _decide_user_language(
    scores: dict[str, float],
    script: ScriptKind,
) -> tuple[UserLanguage, ResponseStyle, float]:
    ur, pa, en = scores["ur"], scores["pa"], scores["en"]
    en_content = scores["en_content"]
    local = max(ur, pa)
    local_lang = UserLanguage.PA_PK if pa > ur else UserLanguage.UR_PK

    mixed = False
    if script == ScriptKind.MIXED and en_content >= 1.5 and local >= 2.0:
        mixed = True
    elif local >= 2.0 and en_content >= 1.5:
        mixed = True
    elif script == ScriptKind.MIXED and en >= 2.0 and local >= 2.0:
        mixed = True

    if mixed:
        confidence = min(0.88, 0.55 + (local + en) / 20)
        return UserLanguage.MIXED, ResponseStyle.MIXED, confidence

    if local >= 1.5 and local >= en + 0.4:
        style = _style_for(local_lang, script, scores)
        margin = local - en
        confidence = min(0.96, 0.62 + margin / 8 + min(local, 6) / 16)
        return local_lang, style, confidence

    if en >= 1.2 and en > local:
        confidence = min(0.95, 0.6 + (en - local) / 8)
        return UserLanguage.EN_PK, ResponseStyle.ENGLISH, confidence

    if script == ScriptKind.ARABIC:
        lang = (
            UserLanguage.PA_PK
            if pa > ur
            else UserLanguage.UR_PK
        )
        return lang, _style_for(lang, script, scores), 0.8

    if local > 0 and local >= en:
        return local_lang, _style_for(local_lang, script, scores), 0.55

    return UserLanguage.EN_PK, ResponseStyle.ENGLISH, 0.5


def _dominant_from_scores(
    scores: dict[str, float],
    script: ScriptKind,
) -> UserLanguage:
    ur, pa, en = scores["ur"], scores["pa"], scores["en"]
    if pa >= ur and pa >= en and pa > 0:
        return UserLanguage.PA_PK
    if ur >= pa and ur >= en and ur > 0:
        return UserLanguage.UR_PK
    if script in {ScriptKind.ARABIC, ScriptKind.MIXED} and ur + pa >= en:
        return UserLanguage.UR_PK if ur >= pa else UserLanguage.PA_PK
    return UserLanguage.EN_PK


def _style_for(
    language: UserLanguage,
    script: ScriptKind,
    scores: dict[str, float],
) -> ResponseStyle:
    if language == UserLanguage.EN_PK:
        return ResponseStyle.ENGLISH
    if language == UserLanguage.MIXED:
        return ResponseStyle.MIXED
    if language == UserLanguage.PA_PK:
        if script == ScriptKind.ARABIC:
            return ResponseStyle.PUNJABI_SHAHMUKHI
        return ResponseStyle.PUNJABI_ROMAN
    return _urdu_style_from_script(script, scores)


def _urdu_style_from_script(
    script: ScriptKind,
    scores: dict[str, float],
) -> ResponseStyle:
    if script in {ScriptKind.ARABIC, ScriptKind.MIXED}:
        return ResponseStyle.URDU_SCRIPT
    return ResponseStyle.ROMAN_URDU


INSUFFICIENT_EVIDENCE = {
    ResponseStyle.ENGLISH: (
        "No matching document or corpus passage was found for this question.\n\n"
        "I could not locate sufficient retrieved legal material to give a "
        "document-grounded answer. Please add a statute/section reference or "
        "upload a document, or ask again for general legal guidance that will "
        "be clearly marked as not authority-backed."
    ),
    ResponseStyle.ROMAN_URDU: (
        "Is sawaal ke liye koi matching document/corpus passage nahi mila.\n\n"
        "Document-grounded jawab ke liye kaafi retrieved material nahi. "
        "Statute/section ya upload dein, ya dobara poochhein — general "
        "guidance clearly mark ho gi ke yeh authority-backed nahi."
    ),
    ResponseStyle.URDU_SCRIPT: (
        "دستیاب قانونی ذرائع میں اس نکتہ پر حتمی جواب کے لیے کافی مستند "
        "مواد نہیں ملا۔\n\n"
        "موجودہ مواد ایک پُراعتماد قانونی نتیجہ کی تائید نہیں کرتا۔ براہ "
        "کرم قابل اطلاق قانون، قواعد یا پابند فیصلوں سے تصدیق کریں، یا مزید "
        "سیاق و سباق یا دستاویزات فراہم کریں۔ دفعہ نمبر اور کیس citations "
        "کو تبدیل نہ کریں۔"
    ),
    ResponseStyle.PUNJABI_ROMAN: (
        "Available legal sources wich eh gal da pukka jawab den layi kafi "
        "authoritative material nahi milya.\n\n"
        "Retrieved material te bharosa kar ke legal conclusion nahi ditta ja "
        "sakda. Applicable statute, rules ya binding case law to verify karo, "
        "ya hor context/documents deyo. Section numbers te citations wohi "
        "rehne do."
    ),
    ResponseStyle.PUNJABI_SHAHMUKHI: (
        "دستیاب قانونی ذرائع وچ اس گل دا پکا جواب دین لئی کافی مستند مواد "
        "نہیں ملیا۔\n\n"
        "براہ کرم قابل اطلاق قانون، قواعد یا پابند فیصلیاں توں تصدیق کرو۔ "
        "دفعہ نمبر تے citations نوں تبدیل نہ کرو۔"
    ),
    ResponseStyle.MIXED: (
        "Available legal sources mein/wich is point ka definitive jawab ke "
        "liye sufficient authoritative material nahi mila.\n\n"
        "Please verify against the applicable statute, rules, or binding "
        "case law. Section numbers and citations should stay unchanged."
    ),
}


def insufficient_evidence_answer(
    language: LanguageDetection | None = None,
) -> str:
    if language is None:
        return INSUFFICIENT_EVIDENCE[ResponseStyle.ENGLISH]
    return INSUFFICIENT_EVIDENCE.get(
        language.response_style,
        INSUFFICIENT_EVIDENCE[ResponseStyle.ENGLISH],
    )
