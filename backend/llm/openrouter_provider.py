import logging
from openai import OpenAI

from backend.core.config import settings
from backend.llm.base_provider import BaseProvider
from backend.llm.llm_utils import normalize_messages

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseProvider):

    def __init__(self):

        self.client = None

        self.models = getattr(settings, "OPENROUTER_MODELS", [])

        if not self.models:
            self.models = ["openai/gpt-4o-mini"]

        if settings.USE_OPENROUTER and settings.OPENROUTER_API_KEY:

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.OPENROUTER_API_KEY
            )

    def stream(self, messages):

        if not self.client:
            raise Exception("OpenRouter not initialized")

        messages = normalize_messages(messages)

        for model in self.models:

            try:

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    stream=True
                )

                last = None

                for chunk in response:

                    delta = chunk.choices[0].delta.content

                    if not delta:
                        continue

                    if delta == last:
                        continue

                    last = delta
                    yield delta

                return

            except Exception as e:
                logger.warning(f"[OPENROUTER FAIL] {model}: {e}")
                continue

        raise Exception("All OpenRouter models failed")