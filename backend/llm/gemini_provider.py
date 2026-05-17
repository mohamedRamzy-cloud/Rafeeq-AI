

from __future__ import annotations

import logging
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.core.config import settings
from backend.llm.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class GeminiNotInitializedError(RuntimeError):
    """بتتـ raise لو الـ GeminiProvider اتعمل بدون config صحيح."""


class GeminiProvider(BaseProvider):

    def __init__(self):
        self.llm: ChatGoogleGenerativeAI | None = None

        
        if not getattr(settings, "USE_GEMINI", False):
            logger.info("[GEMINI] USE_GEMINI معطل — Provider مش هيشتغل")
            return

        if not getattr(settings, "GOOGLE_API_KEY", None):
            logger.warning("[GEMINI] GOOGLE_API_KEY مش موجود — Provider مش هيشتغل")
            return

        
        model       = getattr(settings, "GEMINI_MODEL",       "gemini-2.5-flash-lite")
        temperature = getattr(settings, "GEMINI_TEMPERATURE", 0.2)
        max_tokens  = getattr(settings, "GEMINI_MAX_TOKENS",  None)

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                streaming=True,
                google_api_key=settings.GOOGLE_API_KEY,
            )
            logger.info("[GEMINI] Provider جاهز ✔ (model=%s)", model)

        except Exception as e:
            
            logger.exception("[GEMINI] فشل في إنشاء الـ LLM: %s", e)
            raise

    # ══════════════════════════════════════════════════════
    # STREAM
    # ══════════════════════════════════════════════════════
    def stream(self, messages):
        
        if not self.llm:
            raise GeminiNotInitializedError(
                "GeminiProvider مش initialized — "
                "تأكد إن USE_GEMINI=True وإن GOOGLE_API_KEY موجود في الـ settings"
            )

        
        try:
            for chunk in self.llm.stream(messages):
                content = getattr(chunk, "content", None)
                if content:
                    yield content

        except GeminiNotInitializedError:
            raise 
        except Exception as e:
            logger.exception("[GEMINI] خطأ أثناء الـ streaming: %s", e)
            raise  