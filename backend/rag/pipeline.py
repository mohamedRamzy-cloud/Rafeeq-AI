
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from backend.ai.medical_brain import MedicalAIBrain
from backend.rag.response_router import ResponseRouter
from backend.rag.preprocessing_query import preprocess_query
from backend.rag.retrieval import retrieve
from backend.rag.context_builder import build_context
from backend.rag.prompt_builder import build_prompt
from backend.llm.manager import LLMManager
from backend.memory.memory import ProMemory
from backend.rag.smart_cache import SmartCache
from backend.rag.fallback_model import SafeFallback
from backend.rag.utils import clean_input, clean_output, enforce_specialty

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# INIT — singletons created once at import time
# ═══════════════════════════════════════════════════════════
brain    = MedicalAIBrain()
router   = ResponseRouter()
llm      = LLMManager()
# ProMemory is stateful — must only be instantiated once.
# The double REDIS_HOST warning in logs means ChatService also
# creates its own instance. Fix: import this singleton from pipeline
# in chat_service.py instead of calling ProMemory() again.
memory   = ProMemory()
cache    = SmartCache()
fallback = SafeFallback()

executor = ThreadPoolExecutor(max_workers=4)


# ═══════════════════════════════════════════════════════════
# QUICK RESPONSES
# ═══════════════════════════════════════════════════════════
QUICK_RESPONSES: dict[str, str] = {
    # Islamic greetings
    "السلام عليكم":                        "وعليكم السلام ورحمة الله وبركاته 🌷",
    "سلام عليكم":                          "وعليكم السلام ورحمة الله وبركاته 🌷",
    "السلام عليكم ورحمه الله":             "وعليكم السلام ورحمة الله وبركاته 🌷",
    "السلام عليكم ورحمة الله وبركاته":    "وعليكم السلام ورحمة الله وبركاته 🌷",
    # Greetings
    "اهلا":          "أهلاً بحضرتك 🌷",
    "اهلا بيك":      "أهلاً وسهلاً بيك 🌷",
    "اهلين":         "أهلاً وسهلاً 👋",
    "مرحبا":         "مرحباً بحضرتك 👋",
    "هاي":           "هاي 👋 عامل ايه؟",
    "هلو":           "أهلاً بحضرتك 👋",
    "صباح الخير":    "صباح النور 🌷",
    "مساء الخير":    "مساء النور 🌙",
    "صباح النور":    "صباح الفل 🌷",
    "مساء النور":    "مساء الورد 🌷",
    # How are you
    "ازيك":          "الحمد لله 🌷 اتمنى تكون بخير",
    "عامل ايه":      "الحمد لله 🌷 تمام",
    "عامل اي":       "الحمد لله 🌷",
    "اخبارك":        "تمام الحمد لله 🌷",
    "اخبارك ايه":    "الحمد لله كله تمام 🌷",
    "كيفك":          "الحمد لله 🌷",
    # Thanks
    "شكرا":          "العفو 🌷",
    "شكراً":         "العفو 🌷",
    "متشكر":         "تحت أمرك 🌷",
    "تسلم":          "الله يسلمك 🌷",
    "ميرسي":         "العفو 👋",
    # Goodbye
    "باي":           "مع السلامة 👋",
    "سلام":          "مع السلامة 🌷",
    "اشوفك بعدين":   "في أمان الله 👋",
    "تصبح علي خير":  "وانت من أهله 🌙",
    # Identity
    "مين انت":       "أنا رفيق 🧠 مساعد طبي ذكي 👨‍⚕️",
    "انت مين":       "أنا رفيق 🧠 مساعد طبي ذكي 🌷",
    "اسمك ايه":      "أنا رفيق 👨‍⚕️",
    "بتعمل اي":      "أنا مساعد طبي ذكي بساعدك تفهم الأعراض بشكل مبسط وآمن 🌷",
    # Help
    "ساعدني":        "أكيد 🌷 اتفضل قول استفسارك",
    "محتاج مساعده":  "أكيد 🌷 احكيلي الأعراض أو السؤال اللي عندك",
    "عاوز مساعده":   "تحت أمرك 🌷 قول اللي مضايقك",
    "محتاج دكتور":   "قول الأعراض اللي عندك وأنا هحاول أوجهك للتخصص المناسب 🌷",
    # Misc polite
    "بتفهم":         "بحاول أساعد بأفضل شكل ممكن 🌷",
    "انت ذكي":       "شكراً 🌷 بحاول أكون مفيد",
    "انت جامد":      "تسلم 🌷",
    "الحمد لله":     "دايمًا يارب 🌷",
    "يارب":          "ربنا يطمّن قلبك 🌷",
    "ربنا يخليك":    "ويخليك 🌷",
    # Empty
    "":              "اتفضل اكتب استفسارك الطبي 🌷",
}


# ═══════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════
def norm(text: str) -> str:
    """Normalise Arabic text for quick-response lookup."""
    if not text:
        return ""
    text = text.lower().strip()
    for old, new in {"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"}.items():
        text = text.replace(old, new)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1", text)        # collapse repeated chars
    text = " ".join(text.split())
    return text


# ═══════════════════════════════════════════════════════════
# EMERGENCY — hard keyword guard (runs before any LLM call)
# ═══════════════════════════════════════════════════════════
# Why regex instead of a plain list?
#   Plain "اختناق in text" misses "عندي اختناق شوية".
#   Regex lets us match word roots loosely without an NLP library.
_EMERGENCY_RE = re.compile(
    # chest pain / tightness
    r"الم.*صدر|صدر.*الم|وجع.*صدر|صدر.*وجع"
    r"|ضيق.*صدر|صدر.*ضيق"
    # breathing
    r"|اختناق|ضيق.*تنف[سش]|مش.*قادر.*اتنف[سش]|تنف[سش].*صعب|صعوبه.*تنف[سش]"
    r"|مش.*لاقي.*نف[سس]|نف[سس].*مش.*طالع"
    # cardiac
    r"|قلب.*بيتوقف|قلب.*وقف|نوبه.*قلبيه|جلط[هة]"
    # consciousness
    r"|فقدان.*وعي|بفقد.*وعي|اغماء|بغمي.*عليه|طاح.*ارض"
    # severe bleeding / stroke
    r"|نزيف.*شديد|بينزف.*كتير|سكت[هة].*دماغيه"
    # neurological
    r"|شلل.*فجا|وجه.*عوج.*فجا|كلام.*مش.*واضح.*فجا"
    # poisoning / overdose
    r"|جرع[هة].*زياد[هة]|اكل.*سم|تسمم.*شديد"
    # self-harm
    r"|انتحار|اذي.*نفس[يه]?|عاوز.*اموت",
    re.IGNORECASE,
)

# Richer, human emergency reply (replaces the terse old one)
_EMERGENCY_REPLY = (
    "🚨 اللي بتقوله محتاج اهتمام طبي فوري.\n\n"
    "الأعراض دي — زي الاختناق أو الألم الشديد في الصدر — مش وقتها نستنى أو نجرب.\n\n"
    "📞 اتصل بالإسعاف دلوقتي (123 في مصر) أو اطلب من أي حد جنبك يودّيك "
    "أقرب طوارئ فورًا.\n\n"
    "متستناش تشوف لو بتتحسن — في الحالات دي الوقت بيفرق جداً. 🌷"
)


def _is_emergency(question: str, analysis: dict, route: str) -> bool:
    """
    Triple-layer emergency check — any layer firing is enough.

    Layer 1 — keyword regex on raw question (fastest, no LLM needed)
    Layer 2 — brain.analyze() flagged emergency=True
    Layer 3 — router returned "emergency" route
    Layer 4 — brain flagged severity="high" (extra safety net)
    """
    if _EMERGENCY_RE.search(question):
        logger.warning("[EMERGENCY] keyword match on: %.60s", question)
        return True
    if analysis.get("emergency"):
        logger.warning("[EMERGENCY] brain.analyze flagged emergency")
        return True
    if route == "emergency":
        logger.warning("[EMERGENCY] router returned emergency route")
        return True
    if analysis.get("severity") == "high":
        logger.warning("[EMERGENCY] brain.analyze flagged severity=high")
        return True
    return False


# ═══════════════════════════════════════════════════════════
# WEAK RESPONSE DETECTOR
# ═══════════════════════════════════════════════════════════
_WEAK_PATTERNS = re.compile(
    r"لا استطيع|لا يمكنني|غير قادر|حدث خطا|error|exception",
    re.IGNORECASE,
)


def _is_weak(text: str) -> bool:
    """Return True if the LLM response is empty, too short, or an error."""
    if not text or len(text.strip()) < 20:
        return True
    return bool(_WEAK_PATTERNS.search(text))


# ═══════════════════════════════════════════════════════════
# ANALYSIS RESULT VALIDATOR
# ═══════════════════════════════════════════════════════════
# "rag" added — ResponseRouter returns it for knowledge-retrieval questions
_VALID_ROUTES = {"normal", "emergency", "followup", "clarify", "rag"}

_ANALYSIS_DEFAULTS: dict = {
    "emergency": False,
    "specialty": "general",
    "intent": "general_question",
    "needs_rag": True,
    "severity": "low",
}


def _validate_analysis(raw) -> dict:
    """
    Ensure brain.analyze() returned a sane dict.
    Missing keys are filled with safe defaults so downstream code
    never has to do .get("key") with a default everywhere.
    """
    if not isinstance(raw, dict):
        logger.warning("[ANALYSIS] invalid type %s — using defaults", type(raw))
        return dict(_ANALYSIS_DEFAULTS)

    result = dict(_ANALYSIS_DEFAULTS)
    result.update({k: v for k, v in raw.items() if k in _ANALYSIS_DEFAULTS})

    # Coerce types
    result["emergency"] = bool(result["emergency"])
    result["needs_rag"] = bool(result["needs_rag"])
    result["severity"]  = str(result["severity"]).lower()
    result["specialty"] = str(result["specialty"]).lower() or "general"

    return result


# ═══════════════════════════════════════════════════════════
# CACHE GUARD
# ═══════════════════════════════════════════════════════════
_MIN_CACHE_LEN = 15   # anything shorter is likely an error fragment


def _cache_get(key: str) -> str | None:
    """
    Fetch from SmartCache with type + length validation.
    Returns None (cache miss) if the stored value looks corrupt.
    """
    try:
        value = cache.get(key)
        if isinstance(value, str) and len(value) >= _MIN_CACHE_LEN:
            return value
        if value is not None:
            logger.debug("[CACHE] evicting short/invalid entry for key=%s", key)
            cache.delete(key)           # evict silently if SmartCache supports it
    except Exception as exc:
        logger.warning("[CACHE GET] %s", exc)
    return None


def _cache_set(key: str, value: str) -> None:
    """Store in cache only if the value is worth keeping."""
    if not isinstance(value, str) or len(value) < _MIN_CACHE_LEN:
        return
    try:
        cache.set(key, value)
    except Exception as exc:
        logger.warning("[CACHE SET] %s", exc)


# ═══════════════════════════════════════════════════════════
# SAFE FUTURE
# ═══════════════════════════════════════════════════════════
def _safe_future(future, key: str, timeout: float = 8.0):
    try:
        return future.result(timeout=timeout)
    except Exception as exc:
        logger.warning("[FUTURE FAILED] %s: %s", key, exc)
        return None


# ═══════════════════════════════════════════════════════════
# PARALLEL WORKERS
# ═══════════════════════════════════════════════════════════
def _analysis_fn(question: str) -> dict:
    try:
        raw = brain.analyze(question)
        return _validate_analysis(raw)
    except Exception as exc:
        logger.exception("[ANALYSIS ERROR] %s", exc)
        return dict(_ANALYSIS_DEFAULTS)


def _retrieval_fn(question: str) -> str:
    try:
        processed = preprocess_query(question)
        if not processed:
            return ""
        docs = retrieve(processed, k=3)
        return build_context(docs) if docs else ""
    except Exception as exc:
        logger.exception("[RETRIEVAL ERROR] %s", exc)
        return ""


def _fallback_fn(question: str) -> str:
    """
    SafeFallback returns a pre-written answer for common questions.
    We validate it before using it — empty or weak answers are discarded.
    """
    try:
        result = fallback.get(question)
        if isinstance(result, str) and not _is_weak(result):
            return result
    except Exception as exc:
        logger.exception("[FALLBACK ERROR] %s", exc)
    return ""


def _memory_fn(session_id: str) -> list:
    try:
        result = memory.format(session_id)
        return result if isinstance(result, list) else []
    except Exception as exc:
        logger.exception("[MEMORY ERROR] %s", exc)
        return []


# ═══════════════════════════════════════════════════════════
# ROUTE VALIDATOR
# ═══════════════════════════════════════════════════════════
def _safe_route(analysis: dict, question: str) -> str:
    """
    Call ResponseRouter and normalise the result to a known value.
    Falls back to "normal" on any error or unknown return value.
    """
    try:
        route = router.route(analysis, question)
        route = str(route).lower().strip() if route else "normal"
        if route not in _VALID_ROUTES:
            logger.warning("[ROUTER] unknown route '%s' → using 'normal'", route)
            return "normal"
        return route
    except Exception as exc:
        logger.warning("[ROUTER ERROR] %s", exc)
        return "normal"


# ═══════════════════════════════════════════════════════════
# STREAM BUFFER — word-boundary flush
# ═══════════════════════════════════════════════════════════
_BOUNDARY_RE = re.compile(r'^(.*[\s\n،,\.؟?!:;\-–—])(.*?)$', re.DOTALL)


def _flush_words(buf: str) -> tuple[str, str]:
    """
    Split buf at the last safe word boundary.
    Returns (ready_to_send, leftover).

    '...الصداع م'  →  ('...الصداع ', 'م')   ← 'م' stays buffered
    '...الصداع '   →  ('...الصداع ', '')     ← all flushed
    'الصداع'        →  ('', 'الصداع')         ← no boundary yet
    """
    m = _BOUNDARY_RE.match(buf)
    if m:
        return m.group(1), m.group(2)
    return "", buf


# ═══════════════════════════════════════════════════════════
# POST-PROCESS (cache/memory only — never re-sent to client)
# ═══════════════════════════════════════════════════════════
def _post_process(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)           # max 1 blank line
    text = re.sub(r"[^\S\n]+", " ", text)            # collapse inline spaces
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def run_pipeline(question: str, session_id: str):
    """
    Main generator — yields text chunks to the caller (streaming).
    Every yield must be a clean, complete-word string.
    """
    try:
        # ── 0. Input sanitation ───────────────────────────
        question = clean_input(question)
        if not question:
            yield "اتفضل اكتب استفسارك 🌷"
            return

        normalized = norm(question)

        # ── 1. Quick reply (no LLM needed) ───────────────
        if normalized in QUICK_RESPONSES:
            yield QUICK_RESPONSES[normalized]
            return

        # ── 2. Emergency keyword check (BEFORE anything else) ──
        # We run this NOW — before cache, before LLM, before brain.
        # If the message is clearly an emergency, we must not delay.
        if _EMERGENCY_RE.search(question):
            logger.warning("[EMERGENCY] early keyword match — aborting to reply")
            yield _EMERGENCY_REPLY
            return

        # ── 3. Cache ──────────────────────────────────────
        cache_key = f"{session_id}:{normalized}"
        cached    = _cache_get(cache_key)
        if cached:
            logger.info("[CACHE HIT ✔]")
            yield cached
            return

        # ── 4. Memory (user turn) ────────────────────────
        try:
            memory.add(session_id, "user", question)
        except Exception as exc:
            logger.warning("[MEMORY ADD user] %s", exc)

        # ── 5. Parallel tasks ─────────────────────────────
        futures = {
            "analysis":  executor.submit(_analysis_fn,  question),
            "retrieval": executor.submit(_retrieval_fn, question),
            "fallback":  executor.submit(_fallback_fn,  question),
            "memory":    executor.submit(_memory_fn,    session_id),
        }
        results = {
            k: _safe_future(v, k, timeout=35.0 if k == 'retrieval' else 8.0)
            for k, v in futures.items()
        }

        analysis          = results["analysis"]  or dict(_ANALYSIS_DEFAULTS)
        context           = results["retrieval"] or ""
        fallback_response = results["fallback"]  or ""
        memory_messages   = results["memory"]    or []

        # ── 6. Routing ────────────────────────────────────
        route = _safe_route(analysis, question)

        # ── 7. Emergency — full check (all layers) ────────
        if _is_emergency(question, analysis, route):
            yield _EMERGENCY_REPLY
            return

        # ── 8. Build prompt ───────────────────────────────
        try:
            prompt   = build_prompt(
                question=question,
                context=context,
                fallback_context=fallback_response,
                memory_messages=memory_messages,
                emergency=False,
                first_message=(len(memory_messages) == 0),
            )
            messages = prompt.format_messages()
        except Exception as exc:
            logger.exception("[PROMPT ERROR] %s", exc)
            yield "حدث خطأ أثناء تجهيز الرد"
            return

        # ── 9. LLM stream ─────────────────────────────────
        raw_buffer  = ""
        word_buffer = ""
        has_output  = False

        try:
            stream = llm.stream(messages, route)
            if not stream:
                raise ValueError("llm.stream() returned empty/None")

            for chunk in stream:
                if not chunk:
                    continue

                word_buffer += str(chunk)
                to_yield, word_buffer = _flush_words(word_buffer)

                if to_yield:
                    raw_buffer += to_yield
                    has_output  = True
                    yield to_yield

            # Flush remaining partial token at end-of-stream
            if word_buffer:
                raw_buffer += word_buffer
                has_output  = True
                yield word_buffer

            if not has_output:
                raise ValueError("Stream produced no output")

        except Exception as exc:
            logger.exception("[LLM STREAM ERROR] %s", exc)
            # Try fallback before giving up
            if fallback_response:
                yield fallback_response
                return
            yield "الخدمة مشغولة حاليًا، حاول مرة تانية بعد شوية."
            return

        # ── 10. Post-process for cache/memory ────────────
        final_response = _post_process(clean_output(raw_buffer))

        # If LLM gave a weak answer, prefer the pre-written fallback
        if _is_weak(final_response) and fallback_response:
            final_response = fallback_response

        # Enforce medical specialty tone/disclaimer if needed
        try:
            final_response = enforce_specialty(
                final_response,
                analysis.get("specialty", "general"),
            )
        except Exception as exc:
            logger.warning("[ENFORCE SPECIALTY] %s", exc)

        if not final_response:
            final_response = fallback_response or "حاول مرة تانية بعد شوية."

        # ── 11. Persist ───────────────────────────────────
        _cache_set(cache_key, final_response)

        try:
            memory.add(session_id, "assistant", final_response)
        except Exception as exc:
            logger.warning("[MEMORY ADD assistant] %s", exc)

        logger.info("[PIPELINE DONE ✔]")

    except Exception as exc:
        logger.exception("[PIPELINE FATAL] %s", exc)
        yield "حصل خطأ مؤقت، حاول مرة أخرى"