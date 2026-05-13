import logging

from backend.ai.medical_brain import MedicalAIBrain

from backend.llm.manager import LLMManager
from backend.llm.models import FAST_MODEL, RAG_MODEL, STRICT_MODEL

logger = logging.getLogger(__name__)


class ResponseRouter:

    def __init__(self):

        self.brain = MedicalAIBrain()
        self.llm = LLMManager()

    # ======================================================
    # RULE ENGINE (FAST DECISION)
    # ======================================================
    def _rule_decision(self, analysis: dict) -> str:

        #Emergency override
        if analysis.get("emergency"):
            return "emergency"

        # fast simple questions
        if analysis.get("intent") in [
            "general_question",
            "symptom_explanation"
        ] and analysis.get("severity") == "low":
            return FAST_MODEL

        # RAG needed cases
        if analysis.get("needs_rag"):
            return RAG_MODEL

        return STRICT_MODEL

    # ======================================================
    # LLM JUDGE (HYBRID SMART LAYER)
    # ======================================================
    def _llm_judge(self, question: str, analysis: dict) -> str:

        try:

            judge_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a medical routing system. "
                        "Choose ONLY one option: fast, rag, strict, emergency. "
                        "Return only the label."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Question: {question}

Analysis:
{analysis}

Rules:
- emergency = life threatening
- fast = simple medical info
- rag = needs medical knowledge base
- strict = complex reasoning

Return only one word.
"""
                }
            ]

            # use cheap fast model for routing
            response = ""

            for chunk in self.llm.stream(judge_prompt, FAST_MODEL):
                response += chunk

            response = response.strip().lower()

            if response in ["fast", "rag", "strict", "emergency"]:
                return response

        except Exception as e:
            logger.warning(f"[ROUTER LLM FALLBACK ERROR] {e}")

        return None

    # ======================================================
    # MAIN ROUTE FUNCTION (HYBRID)
    # ======================================================
    def route(self, analysis: dict, question: str = None) -> str:

        try:

            # 1. rule-based first (fast path)
            decision = self._rule_decision(analysis)

            logger.info(f"[ROUTER RULE] → {decision}")

            # 2. only refine with LLM if not emergency
            if decision != "emergency" and question:

                llm_decision = self._llm_judge(question, analysis)

                if llm_decision:
                    logger.info(f"[ROUTER LLM] → {llm_decision}")
                    return llm_decision

            return decision

        except Exception as e:

            logger.error(f"[ROUTER ERROR] {e}")

            return RAG_MODEL