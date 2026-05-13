import logging
from backend.core.config import settings
from backend.llm.models import FAST_MODEL, RAG_MODEL, STRICT_MODEL

from backend.llm.groq_provider import GroqProvider
from backend.llm.gemini_provider import GeminiProvider
from backend.llm.openrouter_provider import OpenRouterProvider

logger = logging.getLogger(__name__)


class LLMManager:

    def __init__(self):

        self.route_map = {
            FAST_MODEL: "groq",
            RAG_MODEL: "groq",
            STRICT_MODEL: "gemini"
        }

        self.fallback_chain = ["groq", "gemini", "openrouter"]
        self.providers = {}

        logger.info("[LLM MANAGER] Ready ✔")

    # ======================================================
    def _get_provider(self, name: str):

        if not name:
            return None

        if name in self.providers:
            return self.providers[name]

        try:

            if name == "groq":
                if not settings.GROQ_API_KEY:
                    return None
                provider = GroqProvider()

            elif name == "gemini":
                if not settings.GOOGLE_API_KEY:
                    return None
                provider = GeminiProvider()

            elif name == "openrouter":
                if not settings.OPENROUTER_API_KEY:
                    return None
                provider = OpenRouterProvider()

            else:
                return None

            self.providers[name] = provider
            return provider

        except Exception as e:
            logger.exception(f"[PROVIDER INIT FAILED] {name}: {e}")
            return None

    # ======================================================
    def select_provider(self, route: str):

        name = self.route_map.get(route, "groq")
        return self._get_provider(name), name

    # ======================================================
    def _safe_stream(self, provider, messages):

        if not provider:
            return

        try:
            for chunk in provider.stream(messages):
                if chunk:
                    yield str(chunk)

        except Exception as e:
            logger.exception(f"[STREAM ERROR SAFE] {e}")

    # ======================================================
    def _fallback_stream(self, messages, failed):

        for name in self.fallback_chain:

            if name == failed:
                continue

            provider = self._get_provider(name)

            if not provider:
                continue

            try:
                for chunk in self._safe_stream(provider, messages):
                    yield chunk
                return

            except Exception:
                continue

        yield "الخدمة مشغولة حاليًا، حاول مرة تانية بعد شوية."

    # ======================================================
    def stream(self, messages, route: str):

        provider, name = self.select_provider(route)

        if not provider:
            yield from self._fallback_stream(messages, name)
            return

        try:

            has_output = False

            for chunk in self._safe_stream(provider, messages):
                has_output = True
                yield chunk

            if not has_output:
                yield from self._fallback_stream(messages, name)

        except Exception:
            yield from self._fallback_stream(messages, name)