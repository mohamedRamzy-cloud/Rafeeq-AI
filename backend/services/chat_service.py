from __future__ import annotations

import asyncio
import logging

from backend.rag.pipeline import run_pipeline

logger = logging.getLogger(__name__)


class ChatService:

    def stream_chat(self, question: str, session_id: str):
        return self._stream(question, session_id)

    async def _stream(self, question: str, session_id: str):
        try:
            result = run_pipeline(question, session_id)
        except Exception as exc:
            logger.exception("[CHAT SERVICE] pipeline init error: %s", exc)
            yield "حصل خطأ مؤقت، حاول مرة أخرى"
            return

        if result is None:
            logger.error("[CHAT SERVICE] run_pipeline returned None")
            yield "الخدمة مشغولة حاليًا، حاول مرة تانية بعد شوية."
            return

        try:
            for chunk in result:
                if not chunk:
                    continue
                out = str(chunk)
                yield out
                
                if len(out) < 40:
                    await asyncio.sleep(0)

        except Exception as exc:
            logger.exception("[CHAT SERVICE] stream error: %s", exc)
            yield "حصل خطأ مؤقت، حاول مرة أخرى"