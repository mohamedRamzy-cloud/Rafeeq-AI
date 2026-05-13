import re


# ==========================================================
# NORMALIZE ARABIC
# ==========================================================
def normalize_arabic(text: str):

    if not text:
        return ""

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


# ==========================================================
# CLEAN INPUT
# ==========================================================
def clean_input(text: str):

    if not text:
        return ""

    text = normalize_arabic(text)
    text = text.strip()

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    return text.strip()


# ==========================================================
# CLEAN OUTPUT (FIXED + SMART)
# ==========================================================
def clean_output(text: str):

    if not text:
        return ""

    text = str(text)

    # normalize spaces first
    text = re.sub(r"\s+", " ", text)

    # 🔥 FIX: insert space between stuck Arabic words
    text = re.sub(r"([\u0600-\u06FF])([\u0600-\u06FF])", r"\1 \2", text)

    # fix Arabic-English sticking
    text = re.sub(r"([\u0600-\u06FF])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([\u0600-\u06FF])", r"\1 \2", text)

    # punctuation cleanup
    text = re.sub(r"([.!؟،])\1+", r"\1", text)
    text = re.sub(r"\s+([.!؟،])", r"\1", text)

    # remove weird chars safely
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s.,!?()%\-]", "", text)

    # final spacing fix
    text = re.sub(r"\s+", " ", text)

    return text.strip()
# ==========================================================
# SPECIALTY MAP
# ==========================================================
SPECIALTY_RULES = {

    "دكتور مخ وأعصاب": ["صداع", "دوخه", "تنميل", "تشنجات", "اعصاب"],
    "دكتور عظام": ["ضهر", "ظهر", "ركبه", "مفاصل", "كتف", "عظام"],
    "دكتور جهاز هضمي": ["بطن", "قولون", "معده", "انتفاخ", "اسهال", "ترجيع"],
    "دكتور صدر": ["كحه", "تنفس", "صدر", "ضيق نفس"],
    "دكتور قلب": ["خفقان", "ضغط", "الم صدر"],
    "دكتور انف واذن": ["ودن", "اذن", "زكام", "جيوب", "حلق"]
}


# ==========================================================
# DETECT SPECIALTY
# ==========================================================
def detect_specialty(question: str):

    q = clean_input(question)

    scores = {}

    for specialty, keywords in SPECIALTY_RULES.items():

        score = sum(1 for k in keywords if k in q)

        if score > 0:
            scores[specialty] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


# ==========================================================
# ENFORCE SPECIALTY (FIXED FORMATTING)
# ==========================================================
def enforce_specialty(answer: str, specialty: str):

    if not specialty:
        return answer

    answer = clean_output(answer)

    if specialty in answer:
        return answer

    if "يفضل تراجع" in answer:
        return answer

    return (
        f"{answer}\n\n"
        f"🩺 التوصية: يُفضل مراجعة {specialty}."
    )