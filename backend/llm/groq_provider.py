import logging
from groq import Groq

from backend.core.config import settings
from backend.llm.base_provider import BaseProvider
from backend.llm.llm_utils import normalize_messages

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):

    def __init__(self):

        self.client = None

        # ✅ SAFE DEFAULT
        self.models = getattr(settings, "GROQ_MODELS", [])

        if not self.models:
            self.models = ["openai/gpt-oss-20b"]

        if settings.USE_GROQ and settings.GROQ_API_KEY:
            self.client = Groq(api_key=settings.GROQ_API_KEY)

    def stream(self, messages):

        if not self.client:
            raise Exception("Groq not initialized")

        messages = normalize_messages(messages)

        for model in self.models:

            try:

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=700,
                    stream=True
                )

                last = None

                for chunk in response:

                    delta = getattr(chunk.choices[0].delta, "content", None)

                    if not delta:
                        continue

                    if delta == last:
                        continue

                    last = delta
                    yield str(delta)

                return

            except Exception as e:
                logger.warning(f"[GROQ FAIL] {model}: {e}")
                continue

        raise Exception("All Groq models failed")