
from __future__ import annotations

import logging
import time
from backend.core.config import settings
from backend.llm.models import FAST_MODEL, RAG_MODEL, STRICT_MODEL
from backend.llm.groq_provider import GroqProvider
from backend.llm.gemini_provider import GeminiProvider
from backend.llm.openrouter_provider import OpenRouterProvider

logger = logging.getLogger(__name__)


_DEFAULT_COOLDOWN = 300
RATE_LIMIT_COOLDOWN: int = getattr(settings, "LLM_RATE_LIMIT_COOLDOWN", _DEFAULT_COOLDOWN)


class LLMManager:

    def __init__(self):
        self.route_map: dict[str, str] = {
            FAST_MODEL:   "groq",
            RAG_MODEL:    "groq",
            STRICT_MODEL: "gemini",
        }

        self.fallback_chain: list[str] = ["groq", "gemini", "openrouter"]

        
        self._providers: dict[str, object] = {}
        self._disabled: dict[str, float] = {}

        logger.info("[LLM MANAGER] Ready ✔ (cooldown=%ds)", RATE_LIMIT_COOLDOWN)

    # ══════════════════════════════════════════════════════
    # CIRCUIT BREAKER
    # ══════════════════════════════════════════════════════
    def _is_disabled(self, name: str) -> bool:
        if name not in self._disabled:
            return False
        elapsed = time.monotonic() - self._disabled[name]
        if elapsed >= RATE_LIMIT_COOLDOWN:
            del self._disabled[name]
            
            self._providers.pop(name, None)
            logger.info("[CIRCUIT] '%s' cooldown انتهى — متاح تاني ✔", name)
            return False
        remaining = int(RATE_LIMIT_COOLDOWN - elapsed)
        logger.debug("[CIRCUIT] '%s' معطل — باقي %ds", name, remaining)
        return True

    def _disable(self, name: str) -> None:
        self._disabled[name] = time.monotonic()
        
        self._providers.pop(name, None)
        logger.warning(
            "[CIRCUIT] '%s' اتعطل لمدة %ds بسبب rate limit",
            name, RATE_LIMIT_COOLDOWN,
        )

    def _is_rate_limit(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "429" in msg
            or "rate_limit" in msg
            or "too many requests" in msg
            or getattr(exc, "status_code", None) == 429
        )

    # ══════════════════════════════════════════════════════
    # PROVIDER FACTORY
    # ══════════════════════════════════════════════════════
    def _get_provider(self, name: str) -> object | None:
        if not name:
            return None

        if self._is_disabled(name):
            logger.info("[CIRCUIT] '%s' معطل — skip", name)
            return None

        
        if name in self._providers:
            return self._providers[name]

        try:
            provider = self._build_provider(name)
            if provider is None:
                return None

            
            self._providers[name] = provider
            logger.info("[PROVIDER] '%s' اتعمل ✔", name)
            return provider

        except Exception as e:
            logger.exception("[PROVIDER INIT FAILED] %s: %s", name, e)
            return None

    def _build_provider(self, name: str) -> object | None:
        
        if name == "groq":
            if not getattr(settings, "GROQ_API_KEY", None):
                logger.warning("[PROVIDER] groq: GROQ_API_KEY مش موجود")
                return None
            return GroqProvider()

        if name == "gemini":
            if not getattr(settings, "GOOGLE_API_KEY", None):
                logger.warning("[PROVIDER] gemini: GOOGLE_API_KEY مش موجود")
                return None
            return GeminiProvider()

        if name == "openrouter":
            if not getattr(settings, "OPENROUTER_API_KEY", None):
                logger.warning("[PROVIDER] openrouter: OPENROUTER_API_KEY مش موجود")
                return None
            return OpenRouterProvider()

        logger.error("[PROVIDER] اسم غير معروف: '%s'", name)
        return None

    # ══════════════════════════════════════════════════════
    def select_provider(self, route: str) -> tuple[object | None, str]:
        
        if route not in self.route_map:
            logger.warning(
                "[ROUTE] route '%s' مش موجود في route_map — هنستخدم groq كـ default",
                route,
            )
        name = self.route_map.get(route, "groq")
        return self._get_provider(name), name

    # ══════════════════════════════════════════════════════
    # SAFE STREAM
    # ══════════════════════════════════════════════════════
    def _safe_stream(self, name: str, provider, messages):
        
        try:
            for chunk in provider.stream(messages):
                if chunk:
                    yield str(chunk)
        except Exception as e:
            if self._is_rate_limit(e):
                self._disable(name)
                logger.warning("[RATE LIMIT] '%s' وصل للحد — اتعطل", name)
            else:
                logger.exception("[STREAM ERROR] '%s': %s", name, e)
            raise

    # ══════════════════════════════════════════════════════
    # FALLBACK CHAIN
    # ══════════════════════════════════════════════════════
    def _fallback_stream(self, messages, failed: str):
        logger.warning("[FALLBACK] '%s' فشل — بجرب البدائل", failed)

        for name in self.fallback_chain:
            
            if self._is_disabled(name):
                logger.info("[FALLBACK] '%s' معطل — skip", name)
                continue

            
            if name == failed:
                logger.debug("[FALLBACK] '%s' هو اللي فشل — skip", name)
                continue

            provider = self._get_provider(name)
            if not provider:
                continue

            logger.info("[FALLBACK] بجرب '%s'", name)
            try:
                
                has_real_output = False
                for chunk in self._safe_stream(name, provider, messages):
                    if chunk.strip():
                        has_real_output = True
                    yield chunk

                if has_real_output:
                    logger.info("[FALLBACK] '%s' نجح ✔", name)
                    return
                else:
                    logger.warning("[FALLBACK] '%s' رد فاضي — بجرب التالي", name)

            except Exception:
                logger.warning("[FALLBACK] '%s' فشل برضه — بجرب التالي", name)
                continue

        yield "الخدمة مشغولة حاليًا، حاول مرة تانية بعد شوية. 🙏"

    # ══════════════════════════════════════════════════════
    # MAIN STREAM
    # ══════════════════════════════════════════════════════
    def stream(self, messages, route: str):
        provider, name = self.select_provider(route)

        if not provider:
            logger.warning("[LLM] '%s' مش متاح — fallback مباشرة", name)
            yield from self._fallback_stream(messages, name)
            return

        try:
            
            has_real_output = False
            for chunk in self._safe_stream(name, provider, messages):
                if chunk.strip():
                    has_real_output = True
                yield chunk

            if not has_real_output:
                logger.warning("[LLM] '%s' رد فاضي — fallback", name)
                yield from self._fallback_stream(messages, name)

        except Exception:
            logger.warning("[LLM] '%s' فشل — fallback", name)
            yield from self._fallback_stream(messages, name)