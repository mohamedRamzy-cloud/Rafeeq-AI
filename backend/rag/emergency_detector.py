import re


class EmergencyDetector:

    def __init__(self):

        # ======================================================
        # HIGH RISK
        # ======================================================
        self.high_risk = {
            "ألم في الصدر": 5,
            "وجع في الصدر": 5,
            "ضيق نفس": 5,
            "مش قادر أتنفس": 6,
            "اختناق": 6,
            "إغماء": 5,
            "نزيف شديد": 6,
            "تشنجات": 5,
            "شلل": 6,
            "فقدان الوعي": 6
        }

        # ======================================================
        # MEDIUM RISK
        # ======================================================
        self.medium_risk = {
            "دوخة شديدة": 3,
            "سخونية عالية": 3,
            "ترجيع مستمر": 3,
            "هبوط": 3,
            "زرقة": 4,
            "عدم اتزان": 3
        }

        # ======================================================
        # REGEX PATTERNS
        # ======================================================
        self.patterns = [

            # breathing
            re.compile(r"(مش\s*قادر\s*أتنفس|ضيق\s*نفس|اختناق)"),

            # chest pain
            re.compile(r"(ألم|وجع)\s*في\s*الصدر"),

            # unconscious
            re.compile(r"(إغماء|فقدان\s*وعي)"),

            # seizures
            re.compile(r"(تشنجات|صرع)"),

            # bleeding
            re.compile(r"(نزيف\s*شديد)")
        ]

    # ==========================================================
    # NORMALIZE
    # ==========================================================
    def normalize(self, text: str):

        if not text:
            return ""

        text = text.lower()

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ==========================================================
    # SCORE CALCULATOR
    # ==========================================================
    def calculate_score(self, text: str):

        text = self.normalize(text)

        score = 0

        # regex priority
        for pattern in self.patterns:

            if pattern.search(text):
                score += 4

        # keyword weighting
        for k, v in self.high_risk.items():

            if k in text:
                score += v

        for k, v in self.medium_risk.items():

            if k in text:
                score += v

        return score

    # ==========================================================
    # MAIN CHECK
    # ==========================================================
    def check(self, text: str):

        score = self.calculate_score(text)

        return score >= 5

    # ==========================================================
    # RISK LEVEL
    # ==========================================================
    def risk_level(self, text: str):

        score = self.calculate_score(text)

        if score >= 10:
            return "critical"

        if score >= 5:
            return "high"

        if score >= 3:
            return "medium"

        return "low"

    # ==========================================================
    # RESPONSE
    # ==========================================================
    def response(self, text=""):

        level = self.risk_level(text)

        if level == "critical":

            return (
                "الأعراض دي ممكن تكون خطيرة جدًا وتحتاج تدخل طبي عاجل.\n"
                "الأفضل تتوجه للطوارئ فورًا أو تتصل بالإسعاف."
            )

        if level == "high":

            return (
                "الأعراض دي محتاجة تقييم طبي سريع.\n"
                "يفضل تراجع دكتور أو تروح طوارئ في أقرب وقت."
            )

        return (
            "لو الأعراض مستمرة أو بتزيد يفضل تراجع دكتور للاطمئنان."
        )