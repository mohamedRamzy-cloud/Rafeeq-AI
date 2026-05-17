from __future__ import annotations

import asyncio
import inspect
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.core.config import settings
from backend.services.chat_service import ChatService
from backend.api.schemas import ChatRequest
from backend.rag.pipeline import brain, llm, memory
from backend.rag.preprocessing_query import preprocess_query
from backend.rag.retrieval import retrieve

WS_RECEIVE_TIMEOUT = 300
WS_PING_INTERVAL   = 30
HTTP_STREAM_MEDIA  = "text/plain; charset=utf-8"

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rafeeq-api")


class _ConnectionRegistry:
    def __init__(self):
        self._conns: dict[str, WebSocket] = {}

    def add(self, cid: str, ws: WebSocket) -> None:
        self._conns[cid] = ws

    def remove(self, cid: str) -> None:
        self._conns.pop(cid, None)

    async def close_all(self) -> None:
        for ws in list(self._conns.values()):
            try:
                await ws.close(code=1001)
            except Exception:
                pass
        self._conns.clear()

    @property
    def count(self) -> int:
        return len(self._conns)


registry = _ConnectionRegistry()


class _LazyWarmup:
    
    def __init__(self):
        self._done  = False
        self._lock  = asyncio.Lock()

    async def run_once(self) -> None:
        if self._done:
            return
        async with self._lock:
            if self._done:
                return
            await self._do_warmup()
            self._done = True

    async def _do_warmup(self) -> None:
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(None, brain.analyze, "عندي صداع خفيف")
            logger.info("[WARMUP] brain ✔")
        except Exception as exc:
            logger.warning("[WARMUP] brain failed: %s", exc)

        try:
            def _retrieval():
                q = preprocess_query("ما هو الصداع")
                retrieve(q, k=1)
            await loop.run_in_executor(None, _retrieval)
            logger.info("[WARMUP] retrieval ✔")
        except Exception as exc:
            logger.warning("[WARMUP] retrieval failed: %s", exc)

        try:
            def _llm():
                for _ in llm.stream([{"role": "user", "content": "قل مرحبا"}], "normal"):
                    break
            await loop.run_in_executor(None, _llm)
            logger.info("[WARMUP] llm ✔")
        except Exception as exc:
            logger.warning("[WARMUP] llm failed: %s", exc)

        logger.info("[WARMUP] done ✔")


_warmup = _LazyWarmup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Rafeeq API starting")
    yield
    logger.info("Shutting down — %d active WS", registry.count)
    await registry.close_all()
    logger.info("Shutdown complete ✔")


app = FastAPI(
    title="Rafeeq Medical API",
    version="1.0.0",
    description="Production Medical RAG API — Arabic streaming chatbot",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_service = ChatService()

_NO_LOG_PATHS = {"/health", "/"}


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    rid = str(uuid4())[:8]
    t0  = time.monotonic()

    if request.url.path not in _NO_LOG_PATHS:
        logger.info("[%s] %s %s", rid, request.method, request.url.path)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] unhandled error", rid)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "internal_server_error"},
        )

    elapsed = round(time.monotonic() - t0, 3)
    if request.url.path not in _NO_LOG_PATHS:
        logger.info("[%s] %ss status=%s", rid, elapsed, response.status_code)

    response.headers["X-Request-ID"] = rid
    return response


def _sanitize(text) -> str:
    return (text or "").replace("\x00", "").strip()


async def _iter_stream(stream):
 
    if inspect.isasyncgen(stream):
        async for chunk in stream:
            if chunk:
                yield str(chunk)
    else:
        for chunk in stream:
            if chunk:
                out = str(chunk)
                yield out
                if len(out) < 40:
                    await asyncio.sleep(0)


@app.get("/")
async def root():
    return {"status": "running", "version": "5.2", "ws_connections": registry.count}


@app.get("/health")
async def health():
    return {"status": "healthy", "ws_connections": registry.count}


@app.post("/chat/new")
async def new_chat(request: Request):
    try:
        body       = await request.json()
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            return JSONResponse(status_code=400, content={"error": "missing session_id"})
        memory.clear(session_id)
        logger.info("[NEW CHAT] session %s cleared", session_id[:8])
        return {"success": True, "session_id": session_id}
    except Exception as exc:
        logger.exception("[NEW CHAT ERROR] %s", exc)
        return JSONResponse(status_code=500, content={"error": "clear_failed"})


@app.post("/chat")
async def chat_http(request: ChatRequest):
    await _warmup.run_once()

    session_id = request.session_id or str(uuid4())
    question   = _sanitize(request.question)

    if not question:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "empty_question"},
        )

    logger.info("[HTTP] sid=%s q=%.50s", session_id[:8], question)

    async def _generate():
        try:
            async for batch in _iter_stream(chat_service.stream_chat(question, session_id)):
                if batch:
                    yield batch
        except Exception:
            logger.exception("[HTTP STREAM ERROR]")
            yield "حصل خطأ مؤقت، حاول مرة أخرى"

    return StreamingResponse(
        _generate(),
        media_type=HTTP_STREAM_MEDIA,
        headers={"X-Session-ID": session_id},
    )


@app.websocket("/chat/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    await _warmup.run_once()

    cid = str(uuid4())[:8]
    registry.add(cid, websocket)
    logger.info("[WS %s] connected (total=%d)", cid, registry.count)

    async def _keepalive():
        try:
            while True:
                await asyncio.sleep(WS_PING_INTERVAL)
                await websocket.send_json({"type": "ping"})
        except Exception:
            pass

    ping_task = asyncio.create_task(_keepalive())

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=WS_RECEIVE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.info("[WS %s] idle timeout", cid)
                await websocket.send_json({"type": "error", "error": "timeout"})
                break
            except WebSocketDisconnect:
                logger.info("[WS %s] client disconnected", cid)
                break
            except Exception as exc:
                logger.warning("[WS %s] receive error: %s", cid, exc)
                await websocket.send_json({"type": "error", "error": "invalid_request"})
                continue

            if data.get("type") == "pong":
                continue

            question   = _sanitize(data.get("question"))
            session_id = data.get("session_id") or str(uuid4())

            if not question:
                await websocket.send_json({"type": "error", "error": "empty_question"})
                continue

            logger.info("[WS %s] q=%.50s", cid, question)

            try:
                async for batch in _iter_stream(chat_service.stream_chat(question, session_id)):
                    if batch:
                        await websocket.send_json({
                            "type":       "chunk",
                            "content":    batch,
                            "session_id": session_id,
                        })

                await websocket.send_json({"type": "done", "session_id": session_id})

            except Exception:
                logger.exception("[WS %s] stream error", cid)
                await websocket.send_json({"type": "error", "error": "stream_failed"})

    except Exception:
        logger.exception("[WS %s] fatal error", cid)

    finally:
        ping_task.cancel()
        registry.remove(cid)
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("[WS %s] closed (total=%d)", cid, registry.count)