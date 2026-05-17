from __future__ import annotations

import json
import logging
from copy import deepcopy
from threading import RLock
from typing import Dict, List, Optional

import redis
from cachetools import TTLCache

from backend.core.config import settings

logger = logging.getLogger(__name__)


class ProMemory:

    def __init__(self):
        self.redis_ok = False
        self.client   = None
        self.lock      = RLock()
        self._cache    = TTLCache(maxsize=500, ttl=3600)

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

    def _key(self, session_id: str) -> str:
        return f"state:{session_id}"

    def _default_state(self) -> Dict:
        return {"messages": [], "summary": ""}

    def _safe_json_load(self, data: str) -> Dict:
        try:
            return json.loads(data)
        except Exception:
            return self._default_state()

    def _safe_text(self, text: Optional[str], max_chars: int = 1200) -> str:
        return str(text).strip()[:max_chars] if text else ""

    def _is_duplicate(self, messages: List[Dict], role: str, message: str) -> bool:
        if not messages or not message:
            return False
        for m in messages[-3:]:
            if m.get("role") == role and m.get("content", "").strip() == message:
                return True
        return False

    def _compress_old_messages(self, state: Dict) -> Dict:
        messages = state.get("messages", [])
        if len(messages) < 10:
            return state

        old_messages  = messages[:-6]
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

        
        return {
            **state,
            "summary":  merged,
            "messages": messages[-6:],
        }

    def _load_locked(self, session_id: str) -> Dict:
        
        cached = self._cache.get(session_id)
        if cached is not None:
            return deepcopy(cached)

        state = self._default_state()

        if self.redis_ok and self.client:
            try:
                data = self.client.get(self._key(session_id))
                if data:
                    state = self._safe_json_load(data)
            except Exception as e:
                logger.error("[MEMORY LOAD ERROR] %s", e)

        self._cache[session_id] = state
        return deepcopy(state)

    def load(self, session_id: str) -> Dict:
        with self.lock:
            return self._load_locked(session_id)

    def save(self, session_id: str, state: Dict) -> None:
        with self.lock:
            safe_state = deepcopy(state)
            self._cache[session_id] = safe_state

            if not (self.redis_ok and self.client):
                return

            try:
                self.client.set(
                    self._key(session_id),
                    json.dumps(safe_state, ensure_ascii=False),
                    ex=60 * 60 * 24,
                )
            except Exception as e:
                logger.error("[MEMORY SAVE ERROR] %s", e)

    def add(self, session_id: str, role: str, message: str) -> None:
        message = self._safe_text(message)
        if not message:
            return

        
        with self.lock:
            state    = self._load_locked(session_id)
            messages = state.get("messages", [])

            if self._is_duplicate(messages, role, message):
                logger.debug("[MEMORY] duplicate skipped role=%s", role)
                return

            messages.append({"role": role, "content": message})
            state["messages"] = messages
            state = self._compress_old_messages(state)

            safe_state = deepcopy(state)
            self._cache[session_id] = safe_state

            if self.redis_ok and self.client:
                try:
                    self.client.set(
                        self._key(session_id),
                        json.dumps(safe_state, ensure_ascii=False),
                        ex=60 * 60 * 24,
                    )
                except Exception as e:
                    logger.error("[MEMORY SAVE ERROR in add] %s", e)

    def format(self, session_id: str) -> List[Dict]:
        # ③ FIX: load جوه الـ lock
        with self.lock:
            state = self._load_locked(session_id)

        summary  = state.get("summary", "")
        messages = state.get("messages", [])
        formatted: List[Dict] = []

        if summary:
            formatted.append({
                "role":    "system",
                "content": f"ملخص المحادثة السابقة: {summary}",
            })

        formatted.extend(deepcopy(messages[-6:]))
        return formatted

    def exists(self, session_id: str, role: str | None = None) -> bool:
        with self.lock:
            state    = self._load_locked(session_id)
            messages = state.get("messages", [])
        if role:
            return any(m.get("role") == role for m in messages)
        return len(messages) > 0

    def clear(self, session_id: str) -> None:
        with self.lock:
            self._cache.pop(session_id, None)
            if self.redis_ok and self.client:
                try:
                    self.client.delete(self._key(session_id))
                except Exception as e:
                    logger.warning("[MEMORY CLEAR ERROR] %s", e)