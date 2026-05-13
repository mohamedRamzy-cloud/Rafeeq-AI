
import asyncio
import logging

from backend.rag.pipeline import run_pipeline
from backend.rag.pipeline import memory          # singleton — لا تعمل instance جديد

logger = logging.getLogger(__name__)


class ChatService:

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # بترجع async generator — main.py بيعمل عليها ensure_async_stream
    # ──────────────────────────────────────────────────────────
    def stream_chat(self, question: str, session_id: str):
        """
        Returns an async generator that yields text chunks.
        Callers (FastAPI WS / HTTP) iterate with ensure_async_stream().
        """
        return self._stream(question, session_id)

    # ──────────────────────────────────────────────────────────
    # CORE ASYNC GENERATOR
    # ──────────────────────────────────────────────────────────
    async def _stream(self, question: str, session_id: str):

        # ── Validate ─────────────────────────────────────────
        question   = (question or "").replace("\x00", "").strip()
        session_id = str(session_id or "").strip()

        if not question:
            yield "اتفضل اكتب استفسارك 🌷"
            return

        # ── Run pipeline ──────────────────────────────────────
        # pipeline.run_pipeline() هي اللي بتعمل:
        #   - memory.add(session_id, "user", question)
        #   - memory.add(session_id, "assistant", final_response)
        # علشان كده مش بنعملهم هنا تاني
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

        # ── Stream chunks ─────────────────────────────────────
        try:
            for chunk in result:
                if not chunk:
                    continue
                yield str(chunk)
                await asyncio.sleep(0)   

        except TypeError as exc:
            logger.error("[CHAT SERVICE] pipeline returned non-iterable: %s", exc)
            yield "حصل خطأ مؤقت، حاول مرة تانية"

        except Exception as exc:
            logger.exception("[CHAT SERVICE] stream error: %s", exc)
            yield "حصل خطأ مؤقت، حاول مرة أخرى"