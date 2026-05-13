
import json
import logging
from copy import deepcopy
from threading import RLock       # ← RLock بدل Lock
from typing import Dict, List, Optional

import redis
from cachetools import TTLCache

from backend.core.config import settings

logger = logging.getLogger(__name__)


class ProMemory:

    def __init__(self):
        self.redis_ok = False
        self.client   = None

        # RLock = Reentrant Lock — نفس الـ thread ممكن يدخله أكتر من مرة
        # من غير ما يحصل deadlock
        self.lock = RLock()

        self._cache = TTLCache(maxsize=500, ttl=3600)

        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=True,
            )
            self.client.ping()
            self.redis_ok = True
            logger.info("[MEMORY] Redis connected ✔")

        except Exception as e:
            logger.warning("[MEMORY] Redis disabled: %s", e)
            self.redis_ok = False

    # ──────────────────────────────────────────────────────
    def key(self, session_id: str) -> str:
        return f"state:{session_id}"

    def default_state(self) -> Dict:
        return {"messages": [], "summary": ""}

    def _safe_json_load(self, data: str) -> Dict:
        try:
            return json.loads(data)
        except Exception:
            return self.default_state()

    # ──────────────────────────────────────────────────────
    def load(self, session_id: str) -> Dict:
        # RLock — safe to call from within another locked method
        with self.lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                return deepcopy(cached)

            state = self.default_state()

            if self.redis_ok and self.client:
                try:
                    data = self.client.get(self.key(session_id))
                    if data:
                        state = self._safe_json_load(data)
                except Exception as e:
                    logger.error("[MEMORY LOAD ERROR] %s", e)
                    state = self.default_state()

            self._cache[session_id] = state
            return deepcopy(state)

    # ──────────────────────────────────────────────────────
    def save(self, session_id: str, state: dict):
        with self.lock:
            safe_state = deepcopy(state)
            self._cache[session_id] = safe_state

            if not (self.redis_ok and self.client):
                return

            try:
                self.client.set(
                    self.key(session_id),
                    json.dumps(safe_state, ensure_ascii=False),
                    ex=60 * 60 * 24,
                )
            except Exception as e:
                logger.error("[MEMORY SAVE ERROR] %s", e)

    # ──────────────────────────────────────────────────────
    def _is_duplicate(self, messages: List[Dict], role: str, message: str) -> bool:
        if not messages:
            return False
        message = (message or "").strip()
        if not message:
            return False
        for m in messages[-3:]:
            if m.get("role") == role and m.get("content", "").strip() == message:
                return True
        return False

    # ──────────────────────────────────────────────────────
    def _safe_text(self, text: Optional[str], max_chars: int = 1200) -> str:
        if not text:
            return ""
        return str(text).strip()[:max_chars]

    # ──────────────────────────────────────────────────────
    def _compress_old_messages(self, state: Dict) -> Dict:
        messages = state.get("messages", [])
        if len(messages) < 10:
            return state

        old_messages = messages[:-6]
        summary_parts = []

        for msg in old_messages:
            role    = msg.get("role", "")
            content = self._safe_text(msg.get("content", ""), max_chars=120)
            if content:
                summary_parts.append(f"{role}: {content}")

        summary_text = " | ".join(summary_parts).strip()
        existing     = state.get("summary", "").strip()
        merged       = (existing + " " + summary_text).strip()

        if len(merged) > 1500:
            merged = merged[-1500:]

        state["summary"]  = merged
        state["messages"] = messages[-6:]
        return state

    # ──────────────────────────────────────────────────────
    def add(self, session_id: str, role: str, message: str):
        message = self._safe_text(message)
        if not message:
            return

        # load() is called inside the same lock (RLock makes this safe)
        state    = self.load(session_id)
        messages = state.get("messages", [])

        if self._is_duplicate(messages, role, message):
            logger.debug("[MEMORY] duplicate skipped role=%s", role)
            return

        messages.append({"role": role, "content": message})
        state["messages"] = messages
        state = self._compress_old_messages(state)
        self.save(session_id, state)

    # ──────────────────────────────────────────────────────
    def format(self, session_id: str) -> List[Dict]:
        """
        Return the conversation history ready for prompt_builder.

        الملخص بيتحط كـ role="system" مش "assistant":
        - prompt_builder بيحول كل assistant message لـ ("assistant", ...)
          في الـ ChatPromptTemplate — يعني الملخص هيبان كأنه رد حقيقي
          من رفيق، وده غلط.
        - role="system" بيخلي prompt_builder يحطه كـ system note
          (سياق فقط) مش كجزء من المحادثة.
        """
        state    = self.load(session_id)
        summary  = state.get("summary", "")
        messages = state.get("messages", [])

        formatted = []

        if summary:
            formatted.append({
                "role":    "system",        # ← كان "assistant" — ده كان الخطأ
                "content": f"ملخص المحادثة السابقة: {summary}",
            })

        formatted.extend(deepcopy(messages[-6:]))
        return formatted

    # ──────────────────────────────────────────────────────
    def exists(self, session_id: str, role: str = None) -> bool:
        state    = self.load(session_id)
        messages = state.get("messages", [])
        if role:
            return any(m.get("role") == role for m in messages)
        return len(messages) > 0

    # ──────────────────────────────────────────────────────
    def clear(self, session_id: str):
        """مسح المحادثة كاملة — مفيد لزرار 'محادثة جديدة'."""
        with self.lock:
            self._cache.pop(session_id, None)
            if self.redis_ok and self.client:
                try:
                    self.client.delete(self.key(session_id))
                except Exception as e:
                    logger.warning("[MEMORY CLEAR ERROR] %s", e)