import logging
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.core.config import settings
from backend.llm.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):

    def __init__(self):

        self.llm = None

        model = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash-lite")

        if settings.USE_GEMINI and settings.GOOGLE_API_KEY:

            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.2,
                streaming=True,
                google_api_key=settings.GOOGLE_API_KEY
            )

    def stream(self, messages):

        if not self.llm:
            raise Exception("Gemini not initialized")

        last = None

        for chunk in self.llm.stream(messages):

            content = getattr(chunk, "content", None)

            if not content:
                continue

            if content == last:
                continue

            last = content
            yield content