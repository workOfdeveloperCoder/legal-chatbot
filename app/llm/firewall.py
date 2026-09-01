from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.rag.language import LanguageDetection, ResponseStyle
from app.rag.models import Message


class FirewallDecision(str, Enum):
    ALLOW = "allow"
    BLOCK_CODE = "block_code"
    BLOCK_ILLEGAL = "block_illegal"
    BLOCK_IRRELEVANT = "block_irrelevant"
    BLOCK_JAILBREAK = "block_jailbreak"


@dataclass(frozen=True, slots=True)
class FirewallResult:
    allowed: bool
    decision: FirewallDecision
    reason: str
    refusal_message: str = ""

    def to_metadata(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "decision": self.decision.value,
            "reason": self.reason,
        }


_PROGRAMMING_LANGUAGE_WORD = re.compile(
    r"\b(?:python|javascript|typescript|nodejs|node\.js|java|golang|"
    r"c\+\+|c#|csharp|ruby|php|rust|kotlin|swift|scala|perl|haskell|"
    r"matlab|django|flask|fastapi|laravel)\b",
    re.IGNORECASE,
)

_CODE_INTENT = re.compile(
    r"(?:"
    r"\b(?:write|generate|create|debug|fix|implement|compile|refactor|"
    r"optimize)\b.{0,40}\b(?:code|script|function|algorithm|"
    r"source code|computer program|snippet|regex|unit test)\b"
    r"|"
    r"\b(?:code|script|function|algorithm|source code|computer program|"
    r"snippet)\b.{0,40}\b"
    r"(?:write|generate|create|debug|fix|implement)\b"
    r"|"
    r"\b(?:leetcode|hackerrank|programming assignment|coding challenge|"
    r"pull request)\b"
    r"|"
    r"```(?:python|javascript|js|ts|typescript|java|cpp|c\+\+|go|rust|"
    r"ruby|php|bash|sh|sql|html|css|json)\b"
    r")",
    re.IGNORECASE,
)

_LANGUAGE_CODE_INTENT = re.compile(
    r"(?:"
    r"\b(?:write|generate|create|debug|fix|implement|run)\b.{0,30}\b"
    r"(?:python|javascript|typescript|java|golang|ruby|php|rust|kotlin|"
    r"swift|sql|html|css|react|django|flask)\b"
    r"|"
    r"\b(?:python|javascript|typescript|java|golang|ruby|php|rust|"
    r"kotlin|swift)\b.{0,30}\b(?:code|script|function|program|class|"
    r"snippet)\b"
    r"|"
    r"\bin (?:python|javascript|typescript|java|golang|c\+\+|rust|php|"
    r"ruby|sql)\b"
    r")",
    re.IGNORECASE,
)

_LEGAL_CODE_CONTEXT = re.compile(
    r"\b(?:"
    r"penal code|pakistan penal code|code of civil procedure|"
    r"code of criminal procedure|criminal procedure code|"
    r"civil procedure code|qanun[\-\s]?e[\-\s]?shahadat|"
    r"building bye[\-\s]?laws|code of conduct|"
    r"ppc|crpc|cpc|qso"
    r")\b",
    re.IGNORECASE,
)

_ILLEGAL_HOW_TO = re.compile(
    r"(?:"
    r"\bhow (?:to|do i|can i|would i|could i) (?:kill|murder|poison|"
    r"shoot|stab|strangle)(?! (?:a |the )?case)\b"
    r"|"
    r"\b(?:kill|murder|poison) (?:him|her|them|someone|somebody|"
    r"a person|people|my (?:wife|husband|neighbour|neighbor))\b"
    r"|"
    r"\bhow .{0,50}get away with (?:murder|it|the crime|this)\b"
    r"|"
    r"\bhow .{0,50}without (?:getting )?caught\b"
    r"|"
    r"\bhow .{0,50}hide (?:the )?body\b"
    r"|"
    r"\bhow .{0,50}(?:destroy|tamper with) (?:the )?(?:evidence|cctv)\b"
    r"|"
    r"\b(?:make|build|assemble) (?:a |an )?(?:bomb|explosive|ied|"
    r"molotov)\b"
    r"|"
    r"\bhow (?:to|do i|can i) hack\b"
    r"|"
    r"\bhack into\b"
    r"|"
    r"\bhow (?:to|do i|can i) (?:steal|rob|kidnap|launder|"
    r"counterfeit|evade (?:the )?police)\b"
    r"|"
    r"\b(?:write|create|draft) (?:a |an )?(?:fake|forged|false) "
    r"(?:fir|affidavit|passport|cnic|degree)\b"
    r"|"
    r"\b(?:file|register|lodge) (?:a |an )?(?:fake|false|forged) "
    r"(?:fir|case|complaint)\b"
    r"|"
    r"\bhow (?:to|do i|can i) launder (?:money|funds|cash)\b"
    r"|"
    r"\bkisi ko (?:kaise )?maar"
    r"|"
    r"\bqatl kaise\b"
    r"|"
    r"\bbaghair pakr"
    r"|"
    r"\bbina pakde\b"
    r"|"
    r"\bbina pakray\b"
    r")",
    re.IGNORECASE,
)

_JAILBREAK = re.compile(
    r"(?:"
    r"\bignore (?:all |your |the )?(?:previous |prior |above )?"
    r"instructions\b"
    r"|"
    r"\byou are now\b.{0,40}\b(?:dan|jailbreak|unfiltered|unrestricted)\b"
    r"|"
    r"\b(?:developer mode|jailbreak|dan mode)\b"
    r"|"
    r"\bpretend (?:you are|you're) (?:not )?(?:a |an )?(?:legal|"
    r"lawyer)\b"
    r"|"
        r"\breveal (?:your )?(?:system )?prompt\b"
        r"|"
        r"\bdo not follow (?:your )?(?:rules|guidelines|restrictions)\b"
        r"|"
        r"\bignore (?:your |the )?(?:rules|restrictions|safety)\b"
        r"|"
        r"\b(?:pehle|purani) (?:instructions|hidayaat) ignore\b"
        r"|"
        r"\binstructions ignore karo\b"
        r"|"
        r"\bpretend (?:you are|you're) (?:a |an )?(?:dan|hitman|criminal|"
        r"unfiltered|unrestricted)(?: (?:ai|bot|model))?\b"
        r"|"
        r"\brole[\s\-]?play (?:as |being )?(?:a |an )?(?:dan|hitman|"
        r"criminal|unfiltered)\b"
        r"|"
        r"\bact as (?:if you (?:are|were) )?(?:a |an )?(?:dan|jailbreak|"
        r"unfiltered|unrestricted)\b"
        r")",
        re.IGNORECASE,
    )

_DEFINITE_IRRELEVANT = re.compile(
    r"(?:"
    r"\b(?:weather|forecast|temperature today|rain today)\b"
    r"|"
    r"\b(?:recipe|cook|cooking|ingredients for)\b"
    r"|"
    r"\b(?:cricket score|football score|match score|who won the|"
    r"premier league|psl score|world cup)\b"
    r"|"
    r"\b(?:netflix|movie recommendation|tell me a joke|tell a joke|"
    r"riddle|horoscope|zodiac)\b"
    r"|"
    r"\b(?:write (?:a |me )?(?:poem|song|story|novel|essay on))\b"
    r"|"
    r"\b(?:solve (?:this )?(?:equation|integral|derivative)|"
    r"calculus homework|math homework)\b"
    r"|"
    r"\b(?:bitcoin price|crypto price|stock price of)\b"
    r")",
    re.IGNORECASE,
)

_GREETING = re.compile(
    r"^\s*(?:"
    r"hi|hello|hey|salam|salaam|assalam|assalamu alaikum|"
    r"aoa|thanks|thank you|shukriya|ok|okay|yes|no|continue|"
    r"who are you|what (?:can|do) you (?:do|help with)|"
    r"help|help me"
    r")[\s!.?]*$",
    re.IGNORECASE,
)

_LEGAL_SIGNALS = (
    # English procedure / institutions
        "law",
    "legal",
    "laws",
    "rights",
    "liability",
    "jurisdiction",
    "tribunal",
    "pakistan",
    "pakistani",
    "ppc",
    "crpc",
    "cpc",
    "qanoon",
    "qanun",
    "court",
    "judge",
    "lawyer",
    "advocate",
    "justice",
    "hearing",
    "judgment",
    "judgement",
    "statute",
    "section",
    "sections",
    "article",
    "articles",
    "clause",
    "clauses",
    "provision",
    "provisions",
    "subsection",
    "sub-section",
    "schedule",
    "chapter",
    "preamble",
    "legislation",
    "legislative",
    "statutory",
    "enactment",
    "regulation",
    "regulations",
    "by-law",
    "bylaw",
    "bye-law",
    "jurisprudence",
    "doctrine",
    "ordinance",
    "constitution",
    "precedent",
    "citation",
    "citations",
    "gazette",
    "notification",
    "codified",
    "uncodified",
    "penal",
    "procedural",
    "substantive",
    "plaint",
    "petition",
    "appeal",
    "revision",
    "writ",
    "injunction",
    "stay order",
    "limitation",
    "contract",
    "agreement",
    "affidavit",
    "vakalatnama",
    "wakalatnama",
    "notice",
    "fir",
    "challan",
    "remand",
    "bail",
    "arrest",
    "offence",
    "offense",
    "accused",
    "complainant",
    "plaintiff",
    "defendant",
    "respondent",
    "petitioner",
    "witness",
    "evidence",
    "testimony",
    "prosecution",
    "defence",
    "defense",
    "conviction",
    "acquittal",
    "sentence",
    "punishment",
    "damages",
    "injunction",
    "arbitration",
    "mediation",
    "inheritance",
    "succession",
    "tenant",
    "landlord",
    "property",
    "title deed",
    "sale deed",
    "gift deed",
    "power of attorney",
    "divorce",
    "khula",
    "talaq",
    "maintenance",
    "custody",
    "dowry",
    "mehr",
    "waqf",
    "pre-emption",
    "shufa",
    "qisas",
    "diyat",
    "hudood",
    "double jeopardy",
    "res judicata",
    "estoppel",
    "mens rea",
    "actus reus",
    "habeas",
    "mandamus",
    "certiorari",
    "prohibition",
    "quo warranto",
    "natural justice",
    "locus standi",
    "cause of action",
    "class action",
    "written statement",
    "replication",
    "issues framed",
    "katcheri",
    "thana",
    "ppc",
    "crpc",
    "cpc",
    "qso",
    "pld",
    "scmr",
    "mld",
    "ylr",
    "clc",
    "pakistan penal code",
    "penal code",
    "civil procedure",
    "criminal procedure",
    "peca",
    "cyber crime",
    "money laundering",
    "anti money laundering",
    # Case / matter facts
    "this case",
    "my case",
    "this matter",
    "the case",
    "the matter",
    "uploaded document",
    "this document",
    "the document",
    "the article",
    "the plaint",
    "the fir",
    "kill a case",
    "infructuous",
    "parties",
    "next date",
    "next hearing",
    # Urdu / Roman Urdu
    "قانون",
    "عدالت",
    "مقدمہ",
    "ضمانت",
    "دفعہ",
    "فوجداری",
    "دیوانی",
    "وکالت",
    "qanoon",
    "qanun",
    "muqadma",
    "muqadama",
    "zamanat",
    "zamaanat",
    "dafa",
    "dafaa",
    "kachehri",
    "vakalat",
    "case",
)

_CASE_FACT_SIGNALS = (
    "plaintiff",
    "defendant",
    "accused",
    "complainant",
    "petitioner",
    "respondent",
    "witness",
    "party",
    "parties",
    "fir",
    "hearing",
    "next date",
    "facts",
    "incident",
    "allegation",
    "this case",
    "my case",
    "this matter",
    "uploaded",
    "attached",
    "the document",
    "the article",
        "the plaint",
        "according to",
        "filed",
        "suit",
        "claim",
        "prayer",
        "relief sought",
        "cause of action",
    )

_FOLLOW_UP = re.compile(
    r"^\s*(?:"
    r"and(?: then)?\??|what about.{0,80}|explain(?: more)?|"
    r"more detail(?:s)?|continue|go on|yes|please explain|"
    r"in (?:punjab|sindh|islamabad|kpk|balochistan|pakistan)\??|"
    r"aur\??|phir\??|mazid|مزید|اور کیا|پھر کیا"
    r")\s*$",
    re.IGNORECASE,
)

# Short replies that point at the prior turn ("what is this", "iska matlab").
_DEMONSTRATIVE_FOLLOWUP = re.compile(
    r"(?:"
    r"\bwhat(?:'s|\s+is|\s+are|\s+was|\s+were)?\s+"
    r"(?:this|that|it|these|those)\b|"
    r"\bwhat\s+does\s+(?:this|that|it)\s+mean\b|"
    r"\bwhat\s+do\s+you\s+mean\b|"
    r"\bexplain\s+(?:this|that|it|more|further)\b|"
    r"\bclarify(?:\s+(?:this|that|it))?\b|"
    r"\belaborate(?:\s+(?:on\s+)?(?:this|that|it))?\b|"
    r"\btell\s+me\s+more\b|"
    r"\bin\s+(?:simple|plain)\s+words\b|"
    r"\bsimplify(?:\s+(?:this|that|it))?\b|"
    r"\bsame\s+for\b|"
    r"\band\s+(?:this|that)\b|"
    r"\b(?:this|that|it|these|those)\b|"
    r"\b(?:yeh?|ye)\s+kya\s+hai\b|"
    r"\b(?:is|iska|uska)\s+(?:ka\s+)?matlab\b|"
    r"\baur\s+batao\b|"
    r"\bthora\s+(?:aur\s+)?(?:detail|samjhao)\b|"
    r"\bmazid\b|"
    r"\b(?:فوق|یہ کیا ہے|اس کا مطلب)\b"
    r")",
    re.IGNORECASE,
)

_FENCED_CODE_OUTPUT = re.compile(
    r"```(?:python|javascript|js|ts|typescript|java|cpp|c\+\+|go|rust|"
    r"ruby|php|bash|sh|sql|html|css)\b",
    re.IGNORECASE,
)

_OBVIOUS_CODE_OUTPUT = re.compile(
    r"(?:^|\n)(?:def |async def |function |import |from \w+ import |"
    r"public static void |console\.log\(|SELECT \*|DROP TABLE )",
)

_EVIDENCE_INJECTION = re.compile(
    r"(?im)^[ \t]*(?:ignore (?:all |previous |prior )?(?:instructions|rules).*"
    r"|you are now .{0,80}"
    r"|reveal (?:your )?(?:system )?prompt.*"
    r"|developer mode.*"
    r"|do not follow (?:your )?(?:rules|guidelines).*)"
)

_SECRET_LEAK = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9]{12,}|AIza[0-9A-Za-z\-_]{20,}"
    r"|api[_-]?key\s*[:=]\s*\S+|SECRET_KEY\s*[:=]\s*\S+"
    r"|Bearer [A-Za-z0-9\-._]{20,})",
)


def sanitize_evidence_text(text: str | None) -> str:
    """Strip instruction-like lines from retrieved/uploaded text."""
    if not text:
        return ""
    return _EVIDENCE_INJECTION.sub("[instruction omitted from evidence]", text)

_URDU_CODE = ("کوڈ لکھو", "پروگرام لکھو", "پائتھن", "پائيثن")
_URDU_ILLEGAL = (
    "قتل کیسے",
    "بم کیسے",
    "ہیک کیسے",
    "کسے ماروں",
    "کسی کو مار",
)

_OBFUSCATED_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"k[\s.\-_]*i[\s.\-_]*l[\s.\-_]*l", re.IGNORECASE), "kill"),
    (re.compile(r"m[\s.\-_]*u[\s.\-_]*r[\s.\-_]*d[\s.\-_]*e[\s.\-_]*r", re.IGNORECASE), "murder"),
    (re.compile(r"b[\s.\-_]*o[\s.\-_]*m[\s.\-_]*b", re.IGNORECASE), "bomb"),
)
_URDU_IRRELEVANT = ("موسم کیسا", "کھانا کیسے", "جوک سناؤ")


_REFUSALS: dict[FirewallDecision, dict[ResponseStyle, str]] = {
    FirewallDecision.BLOCK_CODE: {
        ResponseStyle.ENGLISH: (
            "I can only help with Pakistani legal research and questions "
            "about your case or uploaded legal documents. I cannot write, "
            "debug, or explain computer code or software.\n\n"
            "Please ask a legal question — for example a statute, "
            "procedure, drafted notice, or something in your matter."
        ),
        ResponseStyle.URDU_SCRIPT: (
            "میں صرف پاکستانی قانونی تحقیق اور آپ کے مقدمے یا اپ لوڈ شدہ "
            "دستاویزات سے متعلق سوالات کا جواب دے سکتا ہوں۔ کمپیوٹر کوڈ "
            "لکھنا یا ڈیبگ کرنا میرے دائرے میں نہیں ہے۔\n\n"
            "براہ کرم کوئی قانونی سوال پوچھیں — مثلاً دفعہ، کارروائی، "
            "لیگل نوٹس، یا اپنے کیس سے متعلق بات۔"
        ),
        ResponseStyle.ROMAN_URDU: (
            "Main sirf Pakistani qanooni research aur aap ke case / "
            "uploaded documents se related sawaalon ka jawab de sakta "
            "hoon. Computer code likhna ya debug karna mere scope mein "
            "nahi hai.\n\n"
            "Barah-e-karam koi qanooni sawaal poochhein — jaise dafa, "
            "procedure, legal notice, ya apne matter se mutaliq baat."
        ),
        ResponseStyle.PUNJABI_ROMAN: (
            "Main sirf Pakistani qanooni research te tuhade case / "
            "uploaded documents bare sawal da jawab de sakda haan. "
            "Computer code likhna mere kam da hissa nahi.\n\n"
            "Koi qanooni sawal puchho — dafa, procedure, notice, ya "
            "apne matter baare."
        ),
        ResponseStyle.PUNJABI_SHAHMUKHI: (
            "میں صرف پاکستانی قانونی تحقیق تے تہاڈے مقدمے نال سوالاں دا "
            "جواب دے سکدا واں۔ کمپیوٹر کوڈ لکھنا میرے کم وچ نہیں۔\n\n"
            "کوئی قانونی سوال پچھو۔"
        ),
        ResponseStyle.MIXED: (
            "Main sirf Pakistani legal research aur aap ke case / "
            "uploaded documents se related questions ka jawab de sakta "
            "hoon. Computer code likhna mere scope mein nahi hai.\n\n"
            "Please ask a legal question about a statute, procedure, "
            "or your matter."
        ),
    },
    FirewallDecision.BLOCK_ILLEGAL: {
        ResponseStyle.ENGLISH: (
            "I cannot help with requests that ask how to commit a crime "
            "or evade the law. I can discuss what the law is, criminal "
            "procedure, rights of an accused or victim, and how to "
            "pursue a lawful remedy in your case.\n\n"
            "If you have a legal question about an offence, charge, "
            "bail, or procedure, please rephrase it that way."
        ),
        ResponseStyle.URDU_SCRIPT: (
            "میں جرم کرنے یا قانون سے بچ نکلنے کے طریقے نہیں بتا سکتا۔ "
            "میں یہ بتا سکتا ہوں کہ قانون کیا ہے، فوجداری کارروائی کیا "
            "ہے، ملزم یا متاثرہ کے حقوق کیا ہیں، اور قانونی چارہ جوئی "
            "کیسے کی جائے۔\n\n"
            "اگر آپ کا سوال دفعہ، فرد جرم، ضمانت یا کارروائی کے بارے "
            "میں ہے تو اسے قانونی انداز میں دوبارہ پوچھیں۔"
        ),
        ResponseStyle.ROMAN_URDU: (
            "Main jurm karne ya qanoon se bach nikalne ke tareeqe nahi "
            "bata sakta. Main yeh bata sakta hoon ke qanoon kya kehta "
            "hai, faujdari procedure kya hai, aur legal remedy kaise "
            "hasil ki jaye.\n\n"
            "Agar sawaal dafa, charge, zamanat ya procedure ka hai to "
            "usay qanooni andaaz mein poochhein."
        ),
        ResponseStyle.PUNJABI_ROMAN: (
            "Main jurm karan ya qanoon ton bachann de tareeke nahi dass "
            "sakda. Main qanoon, faujdari procedure, te legal remedy "
            "bare dass sakda haan.\n\n"
            "Je sawal dafa, charge, zamanat ya procedure da hai te "
            "qanooni andaaz vich puchho."
        ),
        ResponseStyle.PUNJABI_SHAHMUKHI: (
            "میں جرم کرن جاں قانون توں بچن دے طریقے نہیں دس سکدا۔ قانونی "
            "سوال قانوونی انداز وچ پچھو۔"
        ),
        ResponseStyle.MIXED: (
            "Main crime commit karne ya law se evade karne ka how-to "
            "nahi de sakta. Legal position, procedure, rights, aur "
            "lawful remedies discuss kar sakta hoon.\n\n"
            "Agar sawaal offence, charge, bail, ya procedure ka hai, "
            "please rephrase it as a legal question."
        ),
    },
    FirewallDecision.BLOCK_IRRELEVANT: {
        ResponseStyle.ENGLISH: (
            "I am a legal research assistant for this firm. I can only "
            "answer questions about Pakistani law, court procedure, and "
            "your case or uploaded documents.\n\n"
            "Please ask something related to your matter — for example "
            "a section, limitation, bail, a draft, or a document you "
            "have uploaded."
        ),
        ResponseStyle.URDU_SCRIPT: (
            "میں اس فرم کا قانونی ریسرچ اسسٹنٹ ہوں۔ میں صرف پاکستانی "
            "قانون، عدالتی کارروائی، اور آپ کے مقدمے یا دستاویزات سے "
            "متعلق سوالات کا جواب دے سکتا ہوں۔\n\n"
            "براہ کرم اپنے کیس سے متعلق پوچھیں — مثلاً دفعہ، ضمانت، "
            "لیگل نوٹس، یا اپ لوڈ شدہ دستاویز۔"
        ),
        ResponseStyle.ROMAN_URDU: (
            "Main is firm ka legal research assistant hoon. Main sirf "
            "Pakistani qanoon, court procedure, aur aap ke case / "
            "uploaded documents se related sawaalon ka jawab de sakta "
            "hoon.\n\n"
            "Apne matter se mutaliq poochhein — dafa, zamanat, notice, "
            "ya uploaded document."
        ),
        ResponseStyle.PUNJABI_ROMAN: (
            "Main is firm da legal research assistant haan. Sirf "
            "Pakistani qanoon, court procedure, te tuhade case / "
            "documents bare sawal da jawab de sakda haan.\n\n"
            "Apne matter baare puchho."
        ),
        ResponseStyle.PUNJABI_SHAHMUKHI: (
            "میں ایس فرم دا قانونی اسسٹنٹ واں۔ صرف قانون، کارروائی تے "
            "تہاڈے مقدمے نال سوال پچھو۔"
        ),
        ResponseStyle.MIXED: (
            "Main is firm ka legal research assistant hoon. Sirf "
            "Pakistani law, court procedure, aur aap ke case / "
            "uploaded documents se related questions ka jawab de sakta "
            "hoon.\n\n"
            "Please ask something related to your matter."
        ),
    },
    FirewallDecision.BLOCK_JAILBREAK: {
        ResponseStyle.ENGLISH: (
            "I cannot ignore my role as this firm's legal research "
            "assistant. Ask a question about Pakistani law or your case "
            "and I will help within that scope."
        ),
        ResponseStyle.URDU_SCRIPT: (
            "میں اپنے کردار کو نظرانداز نہیں کر سکتا۔ پاکستانی قانون "
            "یا اپنے مقدمے سے متعلق سوال پوچھیں۔"
        ),
        ResponseStyle.ROMAN_URDU: (
            "Main apna role ignore nahi kar sakta. Pakistani qanoon ya "
            "apne case se related sawaal poochhein."
        ),
        ResponseStyle.PUNJABI_ROMAN: (
            "Main apna role ignore nahi kar sakda. Qanoon ya case baare "
            "sawal puchho."
        ),
        ResponseStyle.PUNJABI_SHAHMUKHI: (
            "میں اپنا کردار نظرانداز نہیں کر سکدا۔ قانون جاں مقدمے "
            "بارے سوال پچھو۔"
        ),
        ResponseStyle.MIXED: (
            "Main apna legal-assistant role ignore nahi kar sakta. "
            "Pakistani law ya apne case se related sawaal poochhein."
        ),
    },
}


class LLMFirewall:
    """
    Deterministic input/output guard for the legal chat LLM.

    Blocks programming, operational crime assistance, jailbreaks, and
    clearly off-topic chat. Allows Pakistani legal research and
    case/matter questions, including criminal-law discussion framed as
    law, procedure, or rights.
    """

    def screen(
        self,
        question: str,
        *,
        history: list[Message] | None = None,
        matter_id: str | None = None,
        document_id: str | None = None,
        has_uploaded_documents: bool = False,
        language: LanguageDetection | None = None,
    ) -> FirewallResult:
        text = self._normalize(question)
        if not text:
            return self._allow("empty")

        if self._is_code_request(text):
            return self._block(
                FirewallDecision.BLOCK_CODE,
                "programming or source-code request",
                language,
            )

        if self._is_illegal_assistance(text):
            return self._block(
                FirewallDecision.BLOCK_ILLEGAL,
                "operational crime or evasion request",
                language,
            )

        if history:
            for message in reversed(history[-6:]):
                if getattr(message, "role", "") != "user":
                    continue
                prior = self._normalize(getattr(message, "content", "") or "")
                if self._is_illegal_assistance(prior) and (
                    _FOLLOW_UP.match(text)
                    or text in {"ok", "okay", "yes", "continue", "go ahead", "do it"}
                ):
                    return self._block(
                        FirewallDecision.BLOCK_ILLEGAL,
                        "multi-turn operational crime request",
                        language,
                    )
                break

        if self._is_definite_irrelevant(text):
            return self._block(
                FirewallDecision.BLOCK_IRRELEVANT,
                "off-topic non-legal request",
                language,
            )

        stripped = self._strip_jailbreak(text)
        if stripped != text and not self._looks_legal_or_case(stripped):
            return self._block(
                FirewallDecision.BLOCK_JAILBREAK,
                "instruction-override with no legal question",
                language,
            )

        if _GREETING.match(text):
            return self._allow("greeting or identity")

        if self._looks_legal_or_case(text):
            return self._allow("legal or case-relevant question")

        in_case = bool(matter_id or document_id or has_uploaded_documents)
        if in_case and self._looks_like_case_followup(text):
            return self._allow("case or matter context")

        if self._is_legal_followup(text, history):
            return self._allow("legal conversation follow-up")

        return self._block(
            FirewallDecision.BLOCK_IRRELEVANT,
            "no legal or case signal",
            language,
        )

    def screen_output(
        self,
        answer: str,
        *,
        language: LanguageDetection | None = None,
    ) -> FirewallResult:
        text = answer or ""
        if _FENCED_CODE_OUTPUT.search(text) or _OBVIOUS_CODE_OUTPUT.search(text):
            return self._block(
                FirewallDecision.BLOCK_CODE,
                "model emitted programming code",
                language,
            )
        if _SECRET_LEAK.search(text):
            return self._block(
                FirewallDecision.BLOCK_JAILBREAK,
                "model emitted secret-like material",
                language,
            )
        return self._allow("output clean")

    def _is_code_request(self, text: str) -> bool:
        if any(marker in text for marker in _URDU_CODE):
            if not _LEGAL_CODE_CONTEXT.search(text):
                return True

        if _CODE_INTENT.search(text) or _LANGUAGE_CODE_INTENT.search(text):
            legal_code = bool(_LEGAL_CODE_CONTEXT.search(text))
            if legal_code and not _PROGRAMMING_LANGUAGE_WORD.search(text):
                return False
            return True

        return False

    def _is_illegal_assistance(self, text: str) -> bool:
        if any(marker in text for marker in _URDU_ILLEGAL):
            return True
        return bool(_ILLEGAL_HOW_TO.search(text))

    def _is_definite_irrelevant(self, text: str) -> bool:
        if any(marker in text for marker in _URDU_IRRELEVANT):
            return True
        if _DEFINITE_IRRELEVANT.search(text) and not self._looks_legal_or_case(text):
            return True
        return False

    def _looks_legal_or_case(self, text: str) -> bool:
        if not text:
            return False
        if self._contains_signal(text, _LEGAL_SIGNALS):
            return True
        if re.search(r"\b(?:section|sec\.?|article|u/s|دفعہ)\s+[0-9a-z\-]+", text):
            return True
        if re.search(r"\bdafaa?\s+[0-9a-z\-]+", text):
            return True
        if re.search(
            r"\b(?:PLD|SCMR|MLD|YLR|CLC|PCr\.LJ)\s+\d{4}\b",
            text,
            re.IGNORECASE,
        ):
            return True
        # Educational / conceptual questions about how statutes are structured.
        if re.search(
            r"\b(?:what\s+(?:is|are|does|do)|what(?:'s| is)\s+meant|"
            r"meaning\s+of|difference\s+between|explain|define|"
            r"definition\s+of)\b.{0,80}\b"
            r"(?:section|sections|article|articles|clause|clauses|"
            r"provision|provisions|statute|statutes|act|acts|"
            r"ordinance|code|codes|law|laws|schedule|chapter|"
            r"preamble|legislation)\b",
            text,
            re.IGNORECASE,
        ):
            return True
        return False

    def _looks_like_case_followup(self, text: str) -> bool:
        if self._contains_signal(text, _CASE_FACT_SIGNALS):
            return True
        return bool(_FOLLOW_UP.match(text))

    @staticmethod
    def _contains_signal(text: str, signals: tuple[str, ...]) -> bool:
        for signal in signals:
            if " " in signal or any(ord(ch) > 127 for ch in signal):
                if signal in text:
                    return True
                continue
            # Match simple English plurals: section→sections, article→articles.
            if re.search(rf"\b{re.escape(signal)}(?:es|s)?\b", text):
                return True
        return False

    def _is_legal_followup(
        self,
        text: str,
        history: list[Message] | None,
    ) -> bool:
        if not history:
            return False
        if not self._conversation_has_legal_context(history):
            return False
        if _FOLLOW_UP.match(text):
            return True
        if self._is_demonstrative_followup(text):
            return True
        # Short non-code replies in an ongoing legal thread.
        return len(text) <= 120 and not self._is_code_request(text)

    def _conversation_has_legal_context(
        self,
        history: list[Message],
    ) -> bool:
        """True if recent user or assistant turns look legal/case-related."""
        recent = history[-12:]
        for message in reversed(recent):
            content = self._normalize(getattr(message, "content", "") or "")
            if content and self._looks_legal_or_case(content):
                return True
        return False

    @staticmethod
    def _is_demonstrative_followup(text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned or len(cleaned) > 160:
            return False
        if not _DEMONSTRATIVE_FOLLOWUP.search(cleaned):
            return False
        words = cleaned.split()
        if len(words) <= 14:
            return True
        return bool(
            re.search(
                r"\b(?:what|why|how|explain|clarify|mean|matlab|kya|"
                r"simplify|elaborate)\b",
                cleaned,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _strip_jailbreak(text: str) -> str:
        stripped = _JAILBREAK.sub(" ", text)
        return re.sub(r"\s+", " ", stripped).strip()

    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = (text or "").translate(
            dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
        )
        cleaned = re.sub(r"\s+", " ", cleaned.strip()).lower()
        for pattern, replacement in _OBFUSCATED_WORDS:
            cleaned = pattern.sub(replacement, cleaned)
        return cleaned

    @staticmethod
    def _allow(reason: str) -> FirewallResult:
        return FirewallResult(
            allowed=True,
            decision=FirewallDecision.ALLOW,
            reason=reason,
        )

    def _block(
        self,
        decision: FirewallDecision,
        reason: str,
        language: LanguageDetection | None,
    ) -> FirewallResult:
        return FirewallResult(
            allowed=False,
            decision=decision,
            reason=reason,
            refusal_message=self._refusal(decision, language),
        )

    @staticmethod
    def _refusal(
        decision: FirewallDecision,
        language: LanguageDetection | None,
    ) -> str:
        messages = _REFUSALS[decision]
        if language is None:
            return messages[ResponseStyle.ENGLISH]
        return messages.get(
            language.response_style,
            messages[ResponseStyle.ENGLISH],
        )
