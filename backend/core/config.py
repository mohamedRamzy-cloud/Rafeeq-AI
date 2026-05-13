import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Settings:

    # Singleton — instance واحدة بس
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

    # ──────────────────────────────────────────────────────
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

        # ── App ───────────────────────────────────────────
        self.APP_NAME         = get("APP_NAME", "Rafeeq Medical API")
        self.DEBUG            = get_bool("DEBUG", False)
        self.DEFAULT_LANGUAGE = get("DEFAULT_LANGUAGE", "ar")

        # ── Infrastructure (Redis) ────────────────────────
        self.REDIS_HOST        = get("REDIS_HOST", "redis_cache")
        self.REDIS_PORT        = get_int("REDIS_PORT", 6379)

        # ── Infrastructure (Qdrant) ───────────────────────
        self.QDRANT_HOST = get("QDRANT_HOST", "qdrant_db")       # اسم الكونتينر
        self.QDRANT_PORT = get_int("QDRANT_PORT", 6333)       # البورت الافتراضي

        # لو QDRANT_URL موجود في env استخدمه، غير كده ابنِه تلقائيًا
        qdrant_url = get("QDRANT_URL")

        if qdrant_url:
          self.QDRANT_URL = qdrant_url
        else:
          self.QDRANT_URL = f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
          logger.info("[CONFIG] QDRANT_URL built automatically → %s", self.QDRANT_URL)

        self.QDRANT_COLLECTION = get("QDRANT_COLLECTION", "medical_rag")

        # ── CORS ──────────────────────────────────────────
        raw_origins = get(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:5500"
        )
        self.ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]

        # ── API Keys ──────────────────────────────────────
        self.GROQ_API_KEY             = get("GROQ_API_KEY")
        self.GOOGLE_API_KEY           = get("GOOGLE_API_KEY")
        self.OPENROUTER_API_KEY       = get("OPENROUTER_API_KEY")
        self.HUGGINGFACEHUB_API_TOKEN = get("HUGGINGFACEHUB_API_TOKEN")

        # ── Enable flags ──────────────────────────────────
        self.USE_GROQ       = get_bool("USE_GROQ", True)
        self.USE_GEMINI     = get_bool("USE_GEMINI", True)
        self.USE_OPENROUTER = get_bool("USE_OPENROUTER", True)

        # ── Models ────────────────────────────────────────
        self.GROQ_MODELS = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]

        self.GEMINI_MODEL = "gemini-2.5-flash-lite"

        self.OPENROUTER_MODELS = [
            "openai/gpt-4o-mini"
        ]

        # ── Validate ──────────────────────────────────────
        self._validate()

    # ──────────────────────────────────────────────────────
    def _validate(self):

        checks = [
            (self.USE_GROQ,       self.GROQ_API_KEY,       "GROQ_API_KEY"),
            (self.USE_GEMINI,     self.GOOGLE_API_KEY,     "GOOGLE_API_KEY"),
            (self.USE_OPENROUTER, self.OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
        ]

        active_providers = 0

        for enabled, key, name in checks:
            if not enabled:
                continue
            if not key:
                logger.warning(
                    "[CONFIG] %s is enabled but missing — provider will be skipped",
                    name
                )
            else:
                active_providers += 1

        if active_providers == 0:
            raise RuntimeError(
                "[CONFIG] No LLM provider has a valid API key."
            )

        logger.info("[CONFIG] %d active LLM provider(s) ready ✔", active_providers)

        if self.DEBUG:
            logger.info("[CONFIG] CORS origins: %s", self.ALLOWED_ORIGINS)


# ── Singleton instance ─────────────────────────────────────
settings = Settings()