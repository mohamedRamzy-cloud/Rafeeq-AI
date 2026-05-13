import re


class MedicalEngine:

    def __init__(self):

        # =========================
        # SYMPTOM MAP
        # =========================
        self.symptoms = {
            "head": {
                "keywords": ["صداع", "وجع راس", "راس", "migraine"],
                "weight": 1
            },
            "chest": {
                "keywords": ["صدر", "وجع صدر", "ضيق نفس", "كحة"],
                "weight": 3
            },
            "stomach": {
                "keywords": ["بطن", "معدة", "قيء", "اسهال", "مغص"],
                "weight": 2
            },
            "neuro": {
                "keywords": ["دوخة", "تنميل", "فقدان توازن", "زغللة"],
                "weight": 3
            },
            "infection": {
                "keywords": ["حرارة", "سخونية", "التهاب", "حلق", "برد"],
                "weight": 2
            },
            "mental": {
                "keywords": ["قلق", "توتر", "اكتئاب", "نوم", "ارق"],
                "weight": 1
            }
        }

        # =========================
        # EMERGENCY CASES
        # =========================
        self.emergency_keywords = [
            "ألم صدر شديد",
            "صعوبة تنفس",
            "إغماء",
            "شلل",
            "نزيف شديد",
            "فقدان وعي"
        ]

    # =========================
    def normalize(self, text: str):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # =========================
    def is_emergency(self, text: str):
        for kw in self.emergency_keywords:
            if kw in text:
                return True
        return False

    # =========================
    def score(self, text: str):

        text = self.normalize(text)

        total = 0
        matched = []

        for category, data in self.symptoms.items():
            for kw in data["keywords"]:
                if kw in text:
                    total += data["weight"]
                    matched.append(category)
                    break

        return {
            "score": total,
            "matched": list(set(matched))
        }

    # =========================
    def severity(self, score: int):

        if score >= 5:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"

    # =========================
    def analyze(self, text: str):

        if self.is_emergency(text):
            return {
                "emergency": True,
                "severity": "critical",
                "action": "emergency"
            }

        result = self.score(text)

        return {
            "emergency": False,
            "severity": self.severity(result["score"]),
            "score": result["score"],
            "categories": result["matched"]
        }