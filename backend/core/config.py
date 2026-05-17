from __future__ import annotations

import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Settings:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load()

    def _load(self):

        def get(key: str, default: str = "") -> str:
            return (os.getenv(key, default) or "").strip()

        def get_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except (ValueError, TypeError):
                logger.warning("[CONFIG] %s is not a valid int — using %s", key, default)
                return default

        def get_bool(key: str, default: bool = False) -> bool:
            raw = (os.getenv(key) or "").strip().lower()
            if raw in ("1", "true", "yes"):
                return True
            if raw in ("0", "false", "no"):
                return False
            return default

        def get_list(key: str, default: list[str]) -> list[str]:
            raw = (os.getenv(key) or "").strip()
            if not raw:
                return default
            return [v.strip() for v in raw.split(",") if v.strip()]

        self.APP_NAME         = get("APP_NAME", "Rafeeq Medical API")
        self.DEBUG            = get_bool("DEBUG", False)
        self.DEFAULT_LANGUAGE = get("DEFAULT_LANGUAGE", "ar")

        self.REDIS_HOST = get("REDIS_HOST", "redis_cache")
        self.REDIS_PORT = get_int("REDIS_PORT", 6379)

        self.QDRANT_HOST       = get("QDRANT_HOST", "qdrant_db")
        self.QDRANT_PORT       = get_int("QDRANT_PORT", 6333)
        self.QDRANT_URL        = get("QDRANT_URL") or f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
        self.QDRANT_COLLECTION = get("QDRANT_COLLECTION", "medical_rag")

        raw_origins = get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500")
        self.ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]

        self.GROQ_API_KEY             = get("GROQ_API_KEY")
        self.GOOGLE_API_KEY           = get("GOOGLE_API_KEY")
        self.OPENROUTER_API_KEY       = get("OPENROUTER_API_KEY")
        self.HUGGINGFACEHUB_API_TOKEN = get("HUGGINGFACEHUB_API_TOKEN")

        self.USE_GROQ       = get_bool("USE_GROQ",       True)
        self.USE_GEMINI     = get_bool("USE_GEMINI",     True)
        self.USE_OPENROUTER = get_bool("USE_OPENROUTER", True)

        
        self.GROQ_MODELS = get_list(
            "GROQ_MODELS",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        )
        self.GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2") or "0.2")
        self.GROQ_MAX_TOKENS  = get_int("GROQ_MAX_TOKENS", 700)

        self.GEMINI_MODEL       = get("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2") or "0.2")
        self.GEMINI_MAX_TOKENS  = get_int("GEMINI_MAX_TOKENS", 0) or None

        self.OPENROUTER_MODELS = get_list(
            "OPENROUTER_MODELS",
            ["openai/gpt-4o-mini"],
        )

        self.LLM_RATE_LIMIT_COOLDOWN = get_int("LLM_RATE_LIMIT_COOLDOWN", 300)

        self._validate()

    def _validate(self):
        provider_checks = [
            (self.USE_GROQ,       self.GROQ_API_KEY,       "GROQ_API_KEY"),
            (self.USE_GEMINI,     self.GOOGLE_API_KEY,     "GOOGLE_API_KEY"),
            (self.USE_OPENROUTER, self.OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
        ]

        
        active_providers = 0
        for enabled, key, name in provider_checks:
            if not enabled:
                logger.info("[CONFIG] %s disabled via flag", name)
                continue
            if not key:
                logger.warning("[CONFIG] %s enabled but key missing — provider will be skipped", name)
            else:
                active_providers += 1

        if active_providers == 0:
            raise RuntimeError("[CONFIG] No LLM provider has a valid API key — can't start")

        logger.info("[CONFIG] %d active LLM provider(s) ✔", active_providers)

        
        if not self.REDIS_HOST:
            logger.warning("[CONFIG] REDIS_HOST not set — cache may fail")

        if not self.QDRANT_URL:
            logger.warning("[CONFIG] QDRANT_URL not set — retrieval may fail")
        else:
            logger.info("[CONFIG] Qdrant → %s / collection=%s", self.QDRANT_URL, self.QDRANT_COLLECTION)

        if self.DEBUG:
            logger.info("[CONFIG] CORS origins: %s", self.ALLOWED_ORIGINS)
            logger.info("[CONFIG] Groq models: %s", self.GROQ_MODELS)
            logger.info("[CONFIG] Gemini model: %s", self.GEMINI_MODEL)
            logger.info("[CONFIG] OpenRouter models: %s", self.OPENROUTER_MODELS)


settings = Settings()