import logging
import re

logger = logging.getLogger(__name__)


class MedicalAIBrain:

    def __init__(self):

        self.emergency_keywords = [
            "ضيق تنفس",
            "ألم صدر",
            "فقدان وعي",
            "نزيف شديد",
            "stroke",
            "heart attack",
            "severe pain",
        ]

        self.specialties = {
            "heart": "cardiology",
            "صدر": "cardiology",
            "معدة": "gastroenterology",
            "بطن": "gastroenterology",
            "رأس": "neurology",
            "صداع": "neurology",
            "عظم": "orthopedics",
            "جلد": "dermatology",
            "عين": "ophthalmology",
        }

    # ======================================================
    # SAFE NORMALIZATION (IMPORTANT FIX)
    # ======================================================
    def _normalize(self, text: str) -> str:

        if not text:
            return ""

        text = text.lower().strip()

        replacements = {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ة": "ه",
            "ى": "ي",
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text

    # ======================================================
    # EMERGENCY DETECTION (IMPROVED)
    # ======================================================
    def _is_emergency(self, text: str) -> bool:

        text = self._normalize(text)

        for keyword in self.emergency_keywords:

            kw = self._normalize(keyword)

            if kw in text:
                return True

        return False

    # ======================================================
    # SPECIALTY DETECTION
    # ======================================================
    def _detect_specialty(self, text: str) -> str:

        text = self._normalize(text)

        for key, value in self.specialties.items():

            if self._normalize(key) in text:
                return value

        return "general"

    # ======================================================
    # INTENT DETECTION
    # ======================================================
    def _detect_intent(self, text: str) -> str:

        text = self._normalize(text)

        if "اعراض" in text or "سبب" in text:
            return "symptom_explanation"

        if "علاج" in text:
            return "treatment"

        if "خطير" in text:
            return "risk_assessment"

        return "general_question"

    # ======================================================
    # MAIN ANALYSIS
    # ======================================================
    def analyze(self, question: str) -> dict:

        try:

            if not question:
                return self._default()

            emergency = self._is_emergency(question)
            specialty = self._detect_specialty(question)
            intent = self._detect_intent(question)

            # IMPORTANT: keep consistent with pipeline
            result = {
                "emergency": emergency,
                "specialty": specialty,
                "intent": intent,
                "needs_rag": not emergency,
                "severity": "high" if emergency else "medium",
            }

            logger.info(f"[BRAIN] {result}")

            return result

        except Exception as e:
            logger.error(f"[BRAIN ERROR] {e}")
            return self._default()

    # ======================================================
    # SAFE DEFAULT
    # ======================================================
    def _default(self):

        return {
            "emergency": False,
            "specialty": "general",
            "intent": "general_question",
            "needs_rag": True,
            "severity": "low",
        }