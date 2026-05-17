from __future__ import annotations

import re


# ══════════════════════════════════════════════════════════════
# NORMALIZE ARABIC
# ══════════════════════════════════════════════════════════════
def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    for src, dst in (
        ("أ", "ا"), ("إ", "ا"), ("آ", "ا"),
        ("ة", "ه"), ("ى", "ي"),
    ):
        text = text.replace(src, dst)
    return text


# ══════════════════════════════════════════════════════════════
# CLEAN INPUT
# ══════════════════════════════════════════════════════════════
def clean_input(text: str) -> str:
    if not text:
        return ""
    text = normalize_arabic(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
# CLEAN OUTPUT
# ══════════════════════════════════════════════════════════════


_ALLOWED = re.compile(
    r"[^\u0600-\u06FF"   
    r"\u200c\u200d"      
    r"a-zA-Z0-9"
    r"\s"
    r".,!?،؟؛:«»()\[\]{}'\"" + r"\-_%@#&*+=/\\|~^`" +
    r"]"
)


_AR_EN_BOUNDARY = re.compile(r"([\u0600-\u06FF])([A-Za-z0-9])")
_EN_AR_BOUNDARY = re.compile(r"([A-Za-z0-9])([\u0600-\u06FF])")


_DUP_PUNCT = re.compile(r"([.!؟،؛])\1+")


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.!؟،؛:])")


_MULTI_SPACE = re.compile(r"\s+")


def clean_output(text: str) -> str:
    if not text:
        return ""

    text = str(text)

    
    text = _MULTI_SPACE.sub(" ", text)

    
    text = _AR_EN_BOUNDARY.sub(r"\1 \2", text)
    text = _EN_AR_BOUNDARY.sub(r"\1 \2", text)

    
    text = _DUP_PUNCT.sub(r"\1", text)

    
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)


    text = _ALLOWED.sub("", text)

    # final space fix
    text = _MULTI_SPACE.sub(" ", text)

    return text.strip()


# ══════════════════════════════════════════════════════════════
# STREAMING BUFFER  
# ══════════════════════════════════════════════════════════════
class StreamingOutputCleaner:
    

    
    _WORD_BOUNDARY = re.compile(r"[\s.,!?،؟؛:\n]$")

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""

        self._buffer += chunk

        
        if self._WORD_BOUNDARY.search(self._buffer):
            out = clean_output(self._buffer)
            self._buffer = ""
            return out

        
        return ""

    def flush(self) -> str:
        
        if self._buffer:
            out = clean_output(self._buffer)
            self._buffer = ""
            return out
        return ""


# ══════════════════════════════════════════════════════════════
# SPECIALTY MAP
# ══════════════════════════════════════════════════════════════
SPECIALTY_RULES: dict[str, list[str]] = {
    "دكتور مخ واعصاب":  ["صداع", "دوخه", "تنميل", "تشنجات", "اعصاب"],
    "دكتور عظام":       ["ضهر", "ظهر", "ركبه", "مفاصل", "كتف", "عظام"],
    "دكتور جهاز هضمي": ["بطن", "قولون", "معده", "انتفاخ", "اسهال", "ترجيع"],
    "دكتور صدر":        ["كحه", "تنفس", "صدر", "ضيق نفس"],
    "دكتور قلب":        ["خفقان", "ضغط", "الم صدر"],
    "دكتور انف واذن":   ["ودن", "اذن", "زكام", "جيوب", "حلق"],
}


# ══════════════════════════════════════════════════════════════
# DETECT SPECIALTY
# ══════════════════════════════════════════════════════════════
def detect_specialty(question: str) -> str | None:
    q = clean_input(question)
    scores = {
        specialty: sum(1 for k in keywords if k in q)
        for specialty, keywords in SPECIALTY_RULES.items()
    }
    scores = {s: sc for s, sc in scores.items() if sc > 0}
    return max(scores, key=scores.get) if scores else None


# ══════════════════════════════════════════════════════════════
# ENFORCE SPECIALTY
# ══════════════════════════════════════════════════════════════
def enforce_specialty(answer: str, specialty: str | None) -> str:
    if not specialty:
        return answer
    answer = clean_output(answer)
    if specialty in answer or "يفضل تراجع" in answer:
        return answer
    return f"{answer}\n\n🩺 التوصية: يُفضل مراجعة {specialty}."