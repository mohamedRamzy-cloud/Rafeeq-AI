import re
import random


class SafeFallback:

    def __init__(self):

        # =========================
        # ADVANCED SYMPTOM KNOWLEDGE BASE
        # =========================
        self.rules = {
            # ================= HEAD =================
            "head": [
                "صداع",
                "وجع راس",
                "راس بيوجع",
                "نص راس",
                "migraine"
            ],
            "chest": [
                "صدر",
                "وجع في الصدر",
                "كحة",
                "ضيق نفس",
                "heart"
            ],
            "stomach": [
                "بطن",
                "معدة",
                "مغص",
                "اسهال",
                "قيء",
                "ترجيع"
            ],
            "mental": [
                "قلق",
                "توتر",
                "اكتئاب",
                "نوم",
                "ارق",
                "stress"
            ],
            "general": [
                "دوخة",
                "ارهاق",
                "تعب",
                "حرارة",
                "سخونية",
                "برد"
            ],
            "infection": [
                "التهاب",
                "عدوى",
                "ميكروب",
                "حلق",
                "لوز"
            ]
        }

        # =========================
        # AI-STYLE RESPONSES (VARIATIONS)
        # =========================
        self.responses = {
            "head": [
                "غالبًا الصداع ده ممكن يكون بسبب إجهاد أو قلة نوم أو ضغط نفسي.",
                "حاول تقلل استخدام الشاشات وتشرب مياه كويس وتاخد راحة.",
                "لو الصداع مستمر أو بيزيد، الأفضل تراجع دكتور مخ وأعصاب."
            ],
            "chest": [
                "ألم الصدر ممكن يكون من إجهاد عضلي أو كحة أو توتر.",
                "لو فيه ضيق نفس أو ألم مستمر لازم كشف طبي بسرعة.",
                "حاول ترتاح وتجنب المجهود لحد ما تطمّن."
            ],
            "stomach": [
                "أعراض المعدة غالبًا مرتبطة بالأكل أو تهيج في الجهاز الهضمي.",
                "ابتعد عن الأكل التقيل والحراق واشرب سوائل دافية.",
                "لو الأعراض مستمرة يبقى محتاج فحص عند دكتور جهاز هضمي."
            ],
            "mental": [
                "التوتر أو اضطراب النوم ممكن يأثروا بشكل كبير على صحتك.",
                "حاول تقلل ضغط اليوم وتنظم نومك.",
                "لو الموضوع مستمر، ممكن تحتاج استشارة مختص نفسي."
            ],
            "general": [
                "الإرهاق أو الدوخة ممكن يكونوا بسبب قلة نوم أو أكل غير منتظم.",
                "اشرب مياه كويس واهتم بأكلك ونظامك اليومي.",
                "لو الأعراض متكررة الأفضل تعمل فحص للاطمئنان."
            ],
            "infection": [
                "ده ممكن يكون التهاب بسيط أو عدوى في الجهاز التنفسي.",
                "اشرب سوائل دافية وابتعد عن المهيجات.",
                "لو فيه حرارة أو استمرار للأعراض لازم كشف طبي."
            ]
        }

    # =========================
    # NORMALIZE (SMART)
    # =========================
    def normalize(self, text: str):
        if not text:
            return ""

        text = text.lower()
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # =========================
    # DETECT CATEGORY
    # =========================
    def detect_category(self, text: str):

        scores = {}

        for category, keywords in self.rules.items():

            score = 0

            for kw in keywords:
                if kw in text:
                    score += 1

            if score > 0:
                scores[category] = score

        if not scores:
            return None

        return max(scores, key=scores.get)

    # =========================
    # MAIN GET
    # =========================
    def get(self, text: str):

        if not text:
            return self.general()

        text = self.normalize(text)

        category = self.detect_category(text)

        if category and category in self.responses:
            return " ".join(random.sample(self.responses[category], k=3))

        return self.general()

    # =========================
    # GENERAL AI-LIKE RESPONSE
    # =========================
    def general(self):
        return (
            "الأعراض اللي بتوصفها ممكن تكون مرتبطة بالإجهاد أو النظام اليومي أو سبب طبي بسيط. "
            "حاول تهتم بنومك وأكلك وتشرب مياه بشكل كافي. "
            "ولو الأعراض مستمرة أو بتزيد، الأفضل تعمل كشف طبي عشان الاطمئنان."
        )