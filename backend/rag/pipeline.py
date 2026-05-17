from __future__ import annotations

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
from backend.rag.utils import clean_input, clean_output, enforce_specialty, StreamingOutputCleaner

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# SINGLETONS
# ══════════════════════════════════════════════════════════════
brain    = MedicalAIBrain()
router   = ResponseRouter()
llm      = LLMManager()
memory   = ProMemory()
cache    = SmartCache()
fallback = SafeFallback()

executor = ThreadPoolExecutor(max_workers=4)


# ══════════════════════════════════════════════════════════════
# QUICK RESPONSES
# ══════════════════════════════════════════════════════════════
QUICK_RESPONSES: dict[str, str] = {
    "السلام عليكم":                        "وعليكم السلام ورحمة الله وبركاته 🌷",
    "سلام عليكم":                          "وعليكم السلام ورحمة الله وبركاته 🌷",
    "السلام عليكم ورحمه الله":             "وعليكم السلام ورحمة الله وبركاته 🌷",
    "السلام عليكم ورحمة الله وبركاته":    "وعليكم السلام ورحمة الله وبركاته 🌷",
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
    "ازيك":          "الحمد لله 🌷 اتمنى تكون بخير",
    "عامل ايه":      "الحمد لله 🌷 تمام",
    "عامل اي":       "الحمد لله 🌷",
    "اخبارك":        "تمام الحمد لله 🌷",
    "اخبارك ايه":    "الحمد لله كله تمام 🌷",
    "كيفك":          "الحمد لله 🌷",
    "شكرا":          "العفو 🌷",
    "شكراً":         "العفو 🌷",
    "متشكر":         "تحت أمرك 🌷",
    "تسلم":          "الله يسلمك 🌷",
    "ميرسي":         "العفو 👋",
    "باي":           "مع السلامة 👋",
    "سلام":          "مع السلامة 🌷",
    "اشوفك بعدين":   "في أمان الله 👋",
    "تصبح علي خير":  "وانت من أهله 🌙",
    "مين انت":       "أنا رفيق 🧠 مساعد طبي ذكي 👨‍⚕️",
    "انت مين":       "أنا رفيق 🧠 مساعد طبي ذكي 🌷",
    "اسمك ايه":      "أنا رفيق 👨‍⚕️",
    "بتعمل اي":      "أنا مساعد طبي ذكي بساعدك تفهم الأعراض بشكل مبسط وآمن 🌷",
    "ساعدني":        "أكيد 🌷 اتفضل قول استفسارك",
    "محتاج مساعده":  "أكيد 🌷 احكيلي الأعراض أو السؤال اللي عندك",
    "عاوز مساعده":   "تحت أمرك 🌷 قول اللي مضايقك",
    "محتاج دكتور":   "قول الأعراض اللي عندك وأنا هحاول أوجهك للتخصص المناسب 🌷",
    "بتفهم":         "بحاول أساعد بأفضل شكل ممكن 🌷",
    "انت ذكي":       "شكراً 🌷 بحاول أكون مفيد",
    "انت جامد":      "تسلم 🌷",
    "الحمد لله":     "دايمًا يارب 🌷",
    "يارب":          "ربنا يطمّن قلبك 🌷",
    "ربنا يخليك":    "ويخليك 🌷",
    "":              "اتفضل اكتب استفسارك الطبي 🌷",
}


# ══════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════
def _norm(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    for old, new in {"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"}.items():
        text = text.replace(old, new)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1", text)
    return " ".join(text.split())


# ══════════════════════════════════════════════════════════════
# EMERGENCY
# ══════════════════════════════════════════════════════════════
_EMERGENCY_RE = re.compile(
    r"الم.*صدر|صدر.*الم|وجع.*صدر|صدر.*وجع"
    r"|ضيق.*صدر|صدر.*ضيق"
    r"|اختناق|ضيق.*تنف[سش]|مش.*قادر.*اتنف[سش]|تنف[سش].*صعب|صعوبه.*تنف[سش]"
    r"|مش.*لاقي.*نف[سس]|نف[سس].*مش.*طالع"
    r"|قلب.*بيتوقف|قلب.*وقف|نوبه.*قلبيه|جلط[هة]"
    r"|فقدان.*وعي|بفقد.*وعي|اغماء|بغمي.*عليه|طاح.*ارض"
    r"|نزيف.*شديد|بينزف.*كتير|سكت[هة].*دماغيه"
    r"|شلل.*فجا|وجه.*عوج.*فجا|كلام.*مش.*واضح.*فجا"
    r"|جرع[هة].*زياد[هة]|اكل.*سم|تسمم.*شديد"
    r"|انتحار|اذي.*نفس[يه]?|عاوز.*اموت",
    re.IGNORECASE,
)

_EMERGENCY_REPLY = (
    "🚨 اللي بتقوله محتاج اهتمام طبي فوري.\n\n"
    "الأعراض دي — زي الاختناق أو الألم الشديد في الصدر — مش وقتها نستنى أو نجرب.\n\n"
    "📞 اتصل بالإسعاف دلوقتي (123 في مصر) أو اطلب من أي حد جنبك يودّيك "
    "أقرب طوارئ فورًا.\n\n"
    "متستناش تشوف لو بتتحسن — في الحالات دي الوقت بيفرق جداً. 🌷"
)


def _is_emergency(question: str, analysis: dict, route: str) -> bool:
    if _EMERGENCY_RE.search(question):
        logger.warning("[EMERGENCY] keyword match: %.60s", question)
        return True
    if analysis.get("emergency"):
        logger.warning("[EMERGENCY] brain flagged emergency")
        return True
    if route == "emergency":
        logger.warning("[EMERGENCY] router returned emergency")
        return True
    if analysis.get("severity") == "high":
        logger.warning("[EMERGENCY] severity=high")
        return True
    return False


# ══════════════════════════════════════════════════════════════
# WEAK RESPONSE DETECTOR
# ══════════════════════════════════════════════════════════════
_WEAK_RE = re.compile(
    r"لا استطيع|لا يمكنني|غير قادر|حدث خطا|error|exception",
    re.IGNORECASE,
)

def _is_weak(text: str) -> bool:
    return not text or len(text.strip()) < 20 or bool(_WEAK_RE.search(text))


# ══════════════════════════════════════════════════════════════
# ANALYSIS VALIDATOR
# ══════════════════════════════════════════════════════════════
_VALID_ROUTES = {"normal", "emergency", "followup", "clarify", "rag"}

_ANALYSIS_DEFAULTS: dict = {
    "emergency": False,
    "specialty": None,
    "intent":    "general_question",
    "needs_rag": True,
    "severity":  "low",
}


def _validate_analysis(raw) -> dict:
    if not isinstance(raw, dict):
        logger.warning("[ANALYSIS] invalid type %s — using defaults", type(raw))
        return dict(_ANALYSIS_DEFAULTS)
    result = dict(_ANALYSIS_DEFAULTS)
    result.update({k: v for k, v in raw.items() if k in _ANALYSIS_DEFAULTS})
    result["emergency"] = bool(result["emergency"])
    result["needs_rag"] = bool(result["needs_rag"])
    result["severity"]  = str(result["severity"]).lower()
    
    specialty = str(result["specialty"] or "").strip().lower()
    result["specialty"] = specialty if specialty and specialty != "general" else None
    return result


# ══════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════
_MIN_CACHE_LEN = 15


def _cache_get(key: str) -> str | None:
    try:
        value = cache.get(key)
        if isinstance(value, str) and len(value) >= _MIN_CACHE_LEN:
            return value
        if value is not None:
            logger.debug("[CACHE] evicting short entry key=%s", key)
            cache.delete(key)
    except Exception as exc:
        logger.warning("[CACHE GET] %s", exc)
    return None


def _cache_set(key: str, value: str) -> None:
    if not isinstance(value, str) or len(value) < _MIN_CACHE_LEN:
        return
    try:
        cache.set(key, value)
    except Exception as exc:
        logger.warning("[CACHE SET] %s", exc)



def _make_cache_key(question: str) -> str:
    return f"q:{question}"


# ══════════════════════════════════════════════════════════════
# SAFE FUTURE
# ══════════════════════════════════════════════════════════════
def _safe_future(future, key: str, timeout: float = 8.0):
    try:
        return future.result(timeout=timeout)
    except Exception as exc:
        logger.warning("[FUTURE FAILED] %s: %s", key, exc)
        return None


# ══════════════════════════════════════════════════════════════
# PARALLEL WORKERS
# ══════════════════════════════════════════════════════════════
def _analysis_fn(question: str) -> dict:
    try:
        return _validate_analysis(brain.analyze(question))
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


# ══════════════════════════════════════════════════════════════
# ROUTE VALIDATOR
# ══════════════════════════════════════════════════════════════
def _safe_route(analysis: dict, question: str) -> str:
    try:
        route = router.route(analysis, question)
        route = str(route).lower().strip() if route else "normal"
        if route not in _VALID_ROUTES:
            logger.warning("[ROUTER] unknown route '%s' → 'normal'", route)
            return "normal"
        return route
    except Exception as exc:
        logger.warning("[ROUTER ERROR] %s", exc)
        return "normal"


# ══════════════════════════════════════════════════════════════
# POST-PROCESS
# ══════════════════════════════════════════════════════════════
def _post_process(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(question: str, session_id: str):
    try:
        # ── 0. Input sanitation ───────────────────────────
        question = clean_input(question)
        if not question:
            yield "اتفضل اكتب استفسارك 🌷"
            return

        normalized = _norm(question)

        # ── 1. Quick reply ────────────────────────────────
        if normalized in QUICK_RESPONSES:
            yield QUICK_RESPONSES[normalized]
            return

        # ── 2. Emergency keyword check (early exit) ───────
        if _EMERGENCY_RE.search(question):
            logger.warning("[EMERGENCY] early keyword match")
            yield _EMERGENCY_REPLY
            return

        # ── 3. Cache ──────────────────────────────────────
        
        cache_key = _make_cache_key(normalized)
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
            k: _safe_future(v, k, timeout=12.0 if k == "retrieval" else 8.0)
            for k, v in futures.items()
        }

        analysis          = results["analysis"]  or dict(_ANALYSIS_DEFAULTS)
        context           = results["retrieval"] or ""
        fallback_response = results["fallback"]  or ""
        memory_messages   = results["memory"]    or []

        # ── 6. Routing ────────────────────────────────────
        route = _safe_route(analysis, question)

        # ── 7. Emergency full check ───────────────────────
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
            )
            messages = prompt.format_messages()
        except Exception as exc:
            logger.exception("[PROMPT ERROR] %s", exc)
            yield "حدث خطأ أثناء تجهيز الرد"
            return

        # ── 9. LLM stream ─────────────────────────────────
        
        cleaner    = StreamingOutputCleaner()
        raw_buffer = ""
        has_output = False

        try:
            for chunk in llm.stream(messages, route):
                if not chunk:
                    continue

                raw_buffer += str(chunk)
                has_output  = True

                cleaned = cleaner.feed(chunk)
                if cleaned:
                    yield cleaned

            
            remainder = cleaner.flush()
            if remainder:
                yield remainder

            if not has_output:
                raise ValueError("Stream produced no output")

        except Exception as exc:
            logger.exception("[LLM STREAM ERROR] %s", exc)
            yield fallback_response or "الخدمة مشغولة حاليًا، حاول مرة تانية بعد شوية."
            return

        # ── 10. Post-process & specialty ─────────────────
        final_response = _post_process(clean_output(raw_buffer))

        if _is_weak(final_response) and fallback_response:
            final_response = fallback_response

        
        specialty = analysis.get("specialty")
        if specialty:
            try:
                final_response = enforce_specialty(final_response, specialty)
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