from __future__ import annotations

import logging
from groq import Groq

from backend.core.config import settings
from backend.llm.base_provider import BaseProvider
from backend.llm.llm_utils import normalize_messages

logger = logging.getLogger(__name__)

_DEFAULT_MODELS    = ["openai/gpt-oss-20b"]
_DEFAULT_TEMP      = 0.2
_DEFAULT_MAX_TOKENS = 700


class GroqNotInitializedError(RuntimeError):
    pass

class GroqAllModelsFailedError(RuntimeError):
    pass


class GroqProvider(BaseProvider):

    def __init__(self):
        self.client: Groq | None = None

        self.models: list[str] = (
            getattr(settings, "GROQ_MODELS", None) or _DEFAULT_MODELS
        )
        self.temperature: float = getattr(settings, "GROQ_TEMPERATURE", _DEFAULT_TEMP)
        self.max_tokens:  int   = getattr(settings, "GROQ_MAX_TOKENS",  _DEFAULT_MAX_TOKENS)

        if not getattr(settings, "USE_GROQ", False):
            logger.info("[GROQ] USE_GROQ معطل — Provider مش هيشتغل")
            return

        if not getattr(settings, "GROQ_API_KEY", None):
            logger.warning("[GROQ] GROQ_API_KEY مش موجود — Provider مش هيشتغل")
            return

        try:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("[GROQ] Provider جاهز ✔ (models=%s)", self.models)
        except Exception as e:
            logger.exception("[GROQ] فشل في إنشاء الـ client: %s", e)
            raise

    def stream(self, messages):
        if not self.client:
            raise GroqNotInitializedError(
                "GroqProvider مش initialized — "
                "تأكد إن USE_GROQ=True وإن GROQ_API_KEY موجود"
            )

        normalized = normalize_messages(messages)
        errors: list[str] = []

        for model in self.models:
            try:
                has_output = False

                response = self.client.chat.completions.create(
                    model=model,
                    messages=normalized,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )

                for chunk in response:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        has_output = True
                        yield str(delta)

                if has_output:
                    return

                logger.warning("[GROQ] '%s' رد فاضي — بجرب التالي", model)
                errors.append(f"{model}: empty response")

            except GroqNotInitializedError:
                raise

            except Exception as e:
                logger.warning("[GROQ FAIL] '%s': %s", model, e)
                errors.append(f"{model}: {e}")
                continue

        raise GroqAllModelsFailedError(
            f"كل الـ Groq models فشلت: {errors}"
        )