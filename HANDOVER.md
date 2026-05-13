# Rafeeq AI — Technical Handover Document

**Project:** Medical AI Assistant (RAG-based Arabic Chatbot)  
**Architecture:** Production-grade, streaming-first, fault-tolerant  
**Authors:** Mohamed Ramzy, Mohamed Reda  
**Status:** Production-ready MVP

---

> This system is strictly for educational and research purposes only.
> It does not provide real medical diagnosis or treatment.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Configuration Layer — config.py](#3-configuration-layer)
4. [Entry Point — main.py](#4-entry-point)
5. [Chat Orchestration — chat_service.py](#5-chat-orchestration)
6. [Core Pipeline — pipeline.py](#6-core-pipeline)
7. [Emergency Detection System](#7-emergency-detection-system)
8. [Prompt Builder — prompt_builder.py](#8-prompt-builder)
9. [Memory System — memory.py](#9-memory-system)
10. [Streaming Engine](#10-streaming-engine)
11. [LLM Routing](#11-llm-routing)
12. [Vector Retrieval](#12-vector-retrieval)
13. [Caching System](#13-caching-system)
14. [Parallel Execution](#14-parallel-execution)
15. [Fault Tolerance Map](#15-fault-tolerance-map)
16. [Performance Notes](#16-performance-notes)
17. [Known Issues and Limitations](#17-known-issues-and-limitations)

---

## 1. System Overview

Rafeeq is not a simple LLM wrapper. Every response goes through a structured pipeline that retrieves medical knowledge from a vector database, injects it into the prompt as grounded context, and streams the result back to the user in real time.

The fundamental difference from a plain chatbot:

```
Plain chatbot:   question  -->  LLM  -->  answer
Rafeeq:          question  -->  emergency check
                           -->  cache lookup
                           -->  parallel: [classify] [retrieve docs] [get fallback] [load memory]
                           -->  build structured prompt
                           -->  LLM stream
                           -->  validate and persist
```

This architecture reduces hallucination, adds a hard safety gate before any LLM call, and keeps latency low through parallel execution and caching.

---

## 2. Repository Structure

```
backend/
├── api/
│   ├── main.py                    FastAPI app, WebSocket handler, startup warmup
│   └── schemas.py                 Pydantic request/response models
│
├── services/
│   └── chat_service.py            Thin orchestration layer between API and pipeline
│
├── rag/
│   ├── pipeline.py                Main execution engine — run_pipeline()
│   ├── prompt_builder.py          Prompt assembly from system + context + history
│   ├── embeddings.py              Sentence-transformer embedding generation
│   ├── retrieval.py               Qdrant top-K semantic search
│   ├── context_builder.py         Formats retrieved docs into LLM-ready text
│   ├── emergency_detector.py      Regex + rule-based emergency keyword matching
│   ├── utils.py                   clean_input, clean_output, enforce_specialty
│   ├── medical_engine.py          Medical rules and specialty routing logic
│   ├── response_router.py         Maps analysis result to pipeline route
│   ├── smart_cache.py             Session+query keyed response cache
│   ├── preprocessing_query.py     Arabic query normalization before retrieval
│   └── fallback_model.py          Pre-written fallback responses for common queries
│
├── ai/
│   └── medical_brain.py           Intent, severity, and emergency classification
│
├── llm/
│   ├── manager.py                 Orchestrates provider selection and failover
│   ├── models.py                  Model name constants and configuration
│   ├── base_provider.py           Abstract base class all providers implement
│   ├── groq_provider.py           Groq API streaming implementation
│   ├── gemini_provider.py         Google Gemini streaming implementation
│   ├── openrouter_provider.py     OpenRouter streaming implementation
│   └── llm_utils.py               Shared utilities (retry, timeout, chunk cleaning)
│
├── memory/
│   └── memory.py                  ProMemory: Redis + TTLCache session store
│
├── vectorstore/
│   ├── qdrant_init.py             Connection setup and collection creation
│   ├── qdrant_service.py          Query interface used by retrieval.py
│   └── qdrant_upload.py           Bulk document ingestion and indexing
│
├── data/
│   ├── cleaning.py                Raw medical text normalization and deduplication
│   ├── loader.py                  Dataset loading from disk or HuggingFace Hub
│   └── documents.py               Document chunking and metadata tagging
│
└── core/
    └── config.py                  Singleton settings loaded from environment variables
```

### Module responsibilities in detail

**`rag/embeddings.py`**
Wraps the sentence-transformer model. Called by `retrieval.py` to embed the user query before vector search. The model is lazy-loaded on first call and cached in memory — this is what causes the 30-second cold start on the first request, which the startup warmup in `main.py` is designed to prevent.

**`rag/emergency_detector.py`**
Contains `_EMERGENCY_RE` and the `_is_emergency()` function. Separated from `pipeline.py` into its own module so the regex patterns can be updated, tested, and extended independently without touching the pipeline logic.

**`rag/utils.py`**
Three functions used at pipeline boundaries: `clean_input()` strips null bytes and normalizes whitespace on the way in; `clean_output()` removes LLM artifacts (stray asterisks, repeated punctuation) on the way out; `enforce_specialty()` appends specialty-appropriate disclaimers based on the classified medical specialty.

**`rag/medical_engine.py`**
Encapsulates medical domain rules that sit between the raw LLM output and the final response — specialty detection thresholds, response tone rules, and the logic that decides when a response needs a specialty-specific disclaimer.

**`llm/base_provider.py`**
Defines the interface that `groq_provider.py`, `gemini_provider.py`, and `openrouter_provider.py` all implement. `manager.py` works against this interface and never imports provider implementations directly — this is what makes adding a new provider a matter of creating one new file and registering it in `manager.py`.

**`llm/models.py`**
Centralizes all model name strings. Prevents the same model name from being hardcoded in multiple places with slightly different spellings across provider files.

**`llm/llm_utils.py`**
Shared utilities used by all providers: retry logic with exponential backoff, per-chunk timeout enforcement, and chunk text cleaning (removes encoding artifacts that some providers occasionally emit).

**`vectorstore/qdrant_service.py`**
The query-time interface to Qdrant. `retrieval.py` imports this rather than the Qdrant client directly, so the vector search logic (connection handling, error recovery, result formatting) is encapsulated in one place.

**`vectorstore/qdrant_upload.py`**
Used during data ingestion, not during inference. Handles bulk upsert of document embeddings into the Qdrant collection. Separate from `qdrant_service.py` because ingestion and query have very different access patterns and error handling needs.

**`data/cleaning.py`**
Normalizes raw medical text before embedding — removes HTML artifacts, normalizes Arabic Unicode variants, deduplicates near-identical passages. The quality of retrieval results depends directly on how clean the indexed text is.

**`data/documents.py`**
Chunks long documents into passage-sized units suitable for embedding (typically 256-512 tokens), and attaches metadata (source, specialty, document type) that `context_builder.py` uses when formatting the retrieved context for the prompt.
---

## 3. Configuration Layer

**File:** `backend/core/config.py`

### What it does

Loads all environment variables once at import time and exposes them as a validated singleton. This prevents the double-initialization problem that was causing two Redis connections to open on startup.

### Singleton pattern

```python
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
```

Every `from backend.core.config import settings` across the codebase returns the exact same object. Without this, `pipeline.py` and `chat_service.py` were each creating their own `ProMemory` instance (and therefore two separate Redis connections), which produced duplicate memory writes and the double warning in the logs.

### Startup validation

```python
def _validate(self):
    active_providers = 0
    for enabled, key, name in checks:
        if enabled and not key:
            logger.warning("provider enabled but key missing")
        elif enabled:
            active_providers += 1
    if active_providers == 0:
        raise RuntimeError("No LLM provider has a valid API key")
```

The application raises immediately at startup rather than failing silently on the first user request. This is deliberate: a configuration error should be visible at boot, not discovered in production traffic.

### CORS origins

```python
self.ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]
```

The `.strip()` call is important. Without it, a `.env` value like `http://localhost:3000, http://127.0.0.1:5500` would produce an origin with a leading space, which silently breaks CORS because the browser sends the URL without the space.

### Qdrant URL resolution

```python
self.QDRANT_URL = get("QDRANT_URL") or f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
```

If `QDRANT_URL` is explicitly set in the environment it takes priority. Otherwise it is built from `QDRANT_HOST` and `QDRANT_PORT`. This lets Docker Compose users configure via host/port while cloud deployments can supply a full URL directly.

---

## 4. Entry Point

**File:** `backend/api/main.py`

### Startup warmup — why it exists

The embedding model (`all-MiniLM-L6-v2`) takes approximately 30 seconds to load on first use. Without warmup, the first user request triggers model loading and times out. The warmup runs all heavy initialization before the server accepts any traffic.

```python
async def _warmup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, brain.analyze, "عندي صداع خفيف")
    await loop.run_in_executor(None, _retrieval_warmup)
    await loop.run_in_executor(None, _llm_warmup)
```

**Why `run_in_executor`?** `brain.analyze()` and `retrieve()` are synchronous (blocking) functions. Calling them directly inside an `async` function blocks the entire event loop — meaning no other requests can be processed during that time. `run_in_executor` offloads the blocking call to a thread pool, keeping the event loop free.

### WebSocket keepalive

```python
async def _keepalive():
    while True:
        await asyncio.sleep(WS_PING_INTERVAL)   # 30 seconds
        await websocket.send_json({"type": "ping"})
```

Many load balancers and reverse proxies (nginx, AWS ALB) close idle TCP connections after 60 seconds. Without an application-level ping, a user who stops typing for a minute loses their connection silently. The ping prevents that.

### Chunk batching

```python
async def _batched_stream(stream, batch_size=40):
    buf = ""
    async for chunk in _iter_stream(stream):
        buf += chunk
        if len(buf) >= batch_size:
            last_space = max(buf.rfind(" "), buf.rfind("\n"))
            if last_space > 0:
                yield buf[:last_space + 1]
                buf = buf[last_space + 1:]
            else:
                yield buf
                buf = ""
    if buf:
        yield buf
```

The LLM emits tokens of 2-5 characters each. Sending each token as a separate WebSocket frame creates hundreds of small network writes per response. Batching to ~40 characters reduces that by roughly 70% with no perceptible latency difference for the user.

The `rfind(" ")` ensures the batch boundary always falls on a word boundary. Without this, the API layer could split an Arabic word across two frames. The pipeline already does word-boundary buffering internally, but the batch step is an additional guard at the delivery layer.

### Connection registry

```python
class _ConnectionRegistry:
    def __init__(self):
        self._conns: dict[str, WebSocket] = {}

    async def close_all(self):
        for ws in list(self._conns.values()):
            try:
                await ws.close(code=1001)
            except Exception:
                pass
```

On application shutdown, FastAPI calls the lifespan cleanup. Without the registry, active WebSocket connections would be dropped abruptly. With it, every open connection receives a clean close frame (code 1001 = "going away") before the process exits.

### `/chat/new` endpoint

```python
@app.post("/chat/new")
async def new_chat(request: Request):
    memory.clear(session_id)
```

This endpoint is called by the frontend "new conversation" button. It removes the session's conversation history from both the TTL cache and Redis. Without this, starting a "new conversation" in the UI would still send old turns to the LLM as context.

---

## 5. Chat Orchestration

**File:** `backend/services/chat_service.py`

This layer is intentionally thin. Its only jobs are input validation, async generator wrapping, and error containment.

```python
async def _stream(self, question: str, session_id: str):
    question = (question or "").replace("\x00", "").strip()
    if not question:
        yield "اتفضل اكتب استفسارك"
        return

    result = run_pipeline(question, session_id)

    for chunk in result:
        if not chunk:
            continue
        yield str(chunk)
        await asyncio.sleep(0)
```

The `await asyncio.sleep(0)` inside the `for` loop is important. `run_pipeline` is a synchronous generator. Iterating it without yielding control would block the event loop for the entire duration of the response. `asyncio.sleep(0)` yields control back to the event loop after every chunk, allowing other coroutines (other user connections, health checks, keepalive pings) to run.

**What is not here:** memory saves and memory loads. Both are handled exclusively inside `pipeline.py`. An earlier version of this file was also calling `memory.add()`, which caused every user message and every assistant response to be saved twice — producing duplicate context in subsequent turns.

---

## 6. Core Pipeline

**File:** `backend/rag/pipeline.py`

This is the central execution engine. It is a synchronous generator function — it uses `yield` to stream response chunks to the caller.

### Execution order

```
clean_input()
norm()                          # normalize Arabic for cache lookup
QUICK_RESPONSES lookup          # instant return if matched
_EMERGENCY_RE.search()          # hard keyword guard — BEFORE cache and LLM
_cache_get()                    # return cached response if valid
memory.add(user turn)
submit parallel futures
safe_future() all results       # with per-task timeouts
_safe_route()                   # validate router output
_is_emergency()                 # full multi-layer check
build_prompt()
llm.stream()                    # yield chunks with word-boundary buffering
post_process()                  # for cache/memory only — not re-sent to client
_cache_set()
memory.add(assistant turn)
```

### Why the emergency check is split into two steps

Step 3 (regex) fires before the cache lookup. This ensures that even if there is a cached response for the same query, an emergency keyword in the current message still triggers the emergency reply. A cached "what causes headaches" response must not be returned to a user who just typed "اختناق والم في الصدر" even if it happens to match the same normalized key.

Step 7 (full check) fires after the parallel tasks complete, because at that point we have the `analysis` dict from `MedicalAIBrain` and the `route` from `ResponseRouter`, which together form the remaining three emergency layers.

### Analysis result validation

```python
def _validate_analysis(raw) -> dict:
    if not isinstance(raw, dict):
        return dict(_ANALYSIS_DEFAULTS)
    result = dict(_ANALYSIS_DEFAULTS)
    result.update({k: v for k, v in raw.items() if k in _ANALYSIS_DEFAULTS})
    result["emergency"] = bool(result["emergency"])
    result["severity"]  = str(result["severity"]).lower()
    return result
```

`brain.analyze()` is a model call that can return unexpected types or missing keys. Without validation, a missing `"emergency"` key would cause a `KeyError` deep inside the pipeline. The validator fills all missing keys with safe defaults and coerces types before any downstream code touches the result.

### Per-task timeouts

```python
results = {
    k: _safe_future(v, k, timeout=35.0 if k == "retrieval" else 8.0)
    for k, v in futures.items()
}
```

The retrieval task has a 35-second timeout instead of 8 seconds because on the first request after startup, the embedding model may not yet be fully loaded (warmup runs concurrently). Giving it 35 seconds ensures the model completes loading and the retrieval succeeds. On all subsequent requests the model is already in memory and retrieval takes under 100ms. The other three tasks (brain, fallback, memory) have no such cold-start problem and stay at 8 seconds.

---

## 7. Emergency Detection System

### Design principle

The system must never make an LLM call when a user is describing a medical emergency. LLMs can produce nuanced, hedged responses to emergency situations. The correct behavior is an immediate, unambiguous reply with emergency services information.

### Layer 1 — Regex hard guard

```python
_EMERGENCY_RE = re.compile(
    r"اختناق|ضيق.*تنف[سش]|مش.*قادر.*اتنف[سش]"
    r"|الم.*صدر|صدر.*الم|وجع.*صدر"
    r"|قلب.*بيتوقف|نوبه.*قلبيه|جلط[هة]"
    r"|فقدان.*وعي|اغماء|بغمي.*عليه"
    r"|نزيف.*شديد|سكت[هة].*دماغيه"
    r"|انتحار|اذي.*نفس",
    re.IGNORECASE
)
```

This regex runs before the cache lookup and before any model call. It uses pattern matching with `.*` to handle natural Arabic phrasing variations — for example `ضيق في التنفس` and `ضيق التنفس` both match `ضيق.*تنف[سش]`. The character class `[سش]` handles spelling variants of the same word.

### Why regex rather than ML for the first layer

Speed and reliability. A regex fires in microseconds with zero failure modes. An ML model can be unavailable, slow, or incorrect. For life-safety decisions, the first filter must be deterministic. The ML layers (brain, router) add nuance for cases the regex does not catch, but they are never the only gate.

### Layer 2 — MedicalAIBrain

Returns `{"emergency": True, ...}` from the analysis. This catches phrasing that the regex does not match — for example indirect descriptions of emergency symptoms without the exact keywords.

### Layer 3 — ResponseRouter

The router can return `route="emergency"` based on the full analysis object, including intent and specialty fields that the regex cannot inspect.

### Layer 4 — Severity scoring

```python
if analysis.get("severity") == "high":
    return True
```

Some conditions are high-severity but not immediately life-threatening in wording (for example, describing symptoms of diabetic crisis without using emergency keywords). The severity flag provides a catch-all for these cases.

### Emergency reply

```python
_EMERGENCY_REPLY = (
    "الأعراض دي محتاجة اهتمام طبي فوري.\n\n"
    "اتصل بالإسعاف دلوقتي (123) أو اطلب من حد جنبك يودّيك أقرب طوارئ فورا.\n\n"
    "متستناش تشوف لو بتتحسن — الوقت بيفرق جدا في الحالات دي."
)
```

The reply is a hardcoded string, not an LLM response. This is intentional. An LLM response can vary, can be vague, can include hedging language. The emergency reply must be consistent, direct, and always available even if all LLM providers are down.

---

## 8. Prompt Builder

**File:** `backend/rag/prompt_builder.py`

### The core fix — real chat turns instead of flat text

The original implementation injected conversation history as a single system message:

```python
# Old approach
("system", "المستخدم: سؤال\nرفيق: رد\nالمستخدم: سؤال تاني")
```

The model interprets this as instructions or background text — not as a conversation it participated in. It does not build on it naturally.

The current approach:

```python
for msg in memory_messages:
    if role == "user":
        messages.append(("human", content))
    elif role == "assistant":
        messages.append(("assistant", content))
```

Each historical turn is a real message in the `ChatPromptTemplate`. The model sees these as an actual conversation it had, not a description of one. This is why follow-up questions like "مقولتليش على نصايح" now produce answers that reference the specific advice given earlier, rather than generic new advice.

### Injection order

```
1. System prompt          (personality, rules, examples)
2. RAG context            (retrieved medical docs — placed before history so it acts as background)
3. Fallback context       (pre-written support — injected only if non-empty and non-weak)
4. Emergency override     (only if flagged — modifies tone for urgent routing)
5. Conversation history   (real human/assistant turns)
6. Current question       (human turn)
```

RAG context is placed before conversation history because the model reads top to bottom. Placing retrieved knowledge before the conversation means the model treats it as established background fact rather than a late addition it has to reconcile with what it already said.

### Summary message handling

`ProMemory.format()` injects a summary of old conversations as a `role="system"` message. The prompt builder checks for this and keeps it as a system note:

```python
if role == "assistant" and content.startswith("ملخص المحادثة السابقة:"):
    messages.append(("system", "[ملخص ما سبق]\n" + content))
    continue
```

If the summary were injected as an `assistant` message, the model would treat it as something it said in the conversation — and might try to respond to it or reference it as a prior statement. As a system message it is invisible to the conversation flow and acts as pure context.

---

## 9. Memory System

**File:** `backend/memory/memory.py`

### Storage layers

```
ProMemory.load()
    → TTLCache (in-process, 500 sessions, 1-hour TTL)
        → Redis (persistent, 24-hour TTL)
            → default_state() (empty, if both unavailable)
```

Every load checks the local cache first. Redis is only hit if the local cache misses. This means most memory reads never leave the process — Redis is primarily for persistence across restarts and for future horizontal scaling.

### RLock instead of Lock

```python
self.lock = RLock()   # Reentrant Lock
```

This is a subtle but important fix. `add()` acquires the lock, then calls `load()`, which also tries to acquire the lock. Python's standard `threading.Lock` is not reentrant — if the same thread tries to acquire it twice, it deadlocks waiting for itself.

`RLock` (Reentrant Lock) tracks which thread holds it and allows the same thread to acquire it again without blocking. The lock count is decremented each time the `with` block exits, and the lock is only fully released when the count reaches zero.

### Duplicate detection

```python
def _is_duplicate(self, messages, role, message):
    for m in messages[-3:]:
        if m.get("role") == role and m.get("content", "").strip() == message:
            return True
    return False
```

This prevents the same message from being saved multiple times if `memory.add()` is called redundantly. It checks only the last 3 messages rather than the full history for performance.

### History compression

```python
def _compress_old_messages(self, state):
    if len(messages) < 10:
        return state
    old_messages = messages[:-6]
    # summarize old_messages into state["summary"]
    state["messages"] = messages[-6:]
    return state
```

When a session accumulates more than 10 turns, the oldest turns are compressed into a summary string and removed from the active message list. Only the 6 most recent turns remain as full messages. This prevents the conversation history from growing unboundedly and eventually exceeding the LLM context window.

### Why `role="system"` for the summary in `format()`

```python
if summary:
    formatted.append({
        "role": "system",
        "content": f"ملخص المحادثة السابقة: {summary}",
    })
```

Earlier versions used `role="assistant"`. This caused the prompt builder to inject the summary as a fake assistant turn in the conversation. The model would see it as something it said and try to maintain consistency with it — which introduced subtle contradictions. As a system message, it is context that exists outside the conversation timeline.

---

## 10. Streaming Engine

### Problem: fragmented Arabic text

The LLM tokenizer operates on subword units. A single Arabic word may be split across multiple tokens:

```
"الصداع"  →  token 1: "ال"   token 2: "صد"   token 3: "اع"
```

Streaming each token as-is produces:

```
Client receives: "ال" then "صد" then "اع"
UI displays:     "ال صد اع" (with spaces between render updates)
```

### Solution: word-boundary buffering in pipeline.py

```python
_BOUNDARY_RE = re.compile(r'^(.*[\s\n،,\.؟?!:;\-])(.*?)$', re.DOTALL)

def _flush_words(buf):
    m = _BOUNDARY_RE.match(buf)
    if m:
        return m.group(1), m.group(2)
    return "", buf
```

The buffer accumulates chunks. The regex finds the last safe boundary character (space, newline, punctuation) in the buffer and splits there. Everything up to and including the boundary is yielded. The remainder stays in the buffer until more chunks arrive.

Result: complete words are always delivered together. A word that spans multiple tokens is held in the buffer until its final token (followed by a space) arrives.

### Batch buffering in main.py

A second layer of buffering at the delivery level:

```python
if len(buf) >= 40:
    last_space = max(buf.rfind(" "), buf.rfind("\n"))
    if last_space > 0:
        yield buf[:last_space + 1]
        buf = buf[last_space + 1:]
```

This groups multiple words into a single WebSocket frame before sending. The result is approximately 40 characters per frame rather than 2-5 characters. This reduces the number of WebSocket writes by roughly 70% and eliminates the UI "flickering" that occurs when many tiny frames arrive in rapid succession.

---

## 11. LLM Routing

**File:** `backend/llm/manager.py`

### Provider order

```
Groq  →  Google Gemini  →  OpenRouter
```

Groq is first because it has the lowest latency for streaming. Gemini is second because it has strong medical reasoning capability. OpenRouter is the final fallback and supports the widest range of available models.

### Route-to-model mapping

The `route` value from `ResponseRouter` can influence which provider or model is selected:

| Route | Preferred provider | Reason |
|---|---|---|
| `normal` | Groq | Fast response, conversational |
| `rag` | Groq or Gemini | Medical content, needs accuracy |
| `emergency` | Not reached | Emergency response is hardcoded |
| `followup` | Groq | Continues existing context |
| `clarify` | Groq | Short clarifying question |

### Startup validation

At import time, the config validator checks that at least one provider has a valid API key. If none do, the application refuses to start. This prevents a server that appears healthy but fails on every request.

---

## 12. Vector Retrieval

**File:** `backend/rag/retrieval.py`

### Pipeline

```
1. preprocess_query(question)       # normalize Arabic, remove noise
2. embed(processed_query)           # sentence-transformers embedding
3. qdrant.search(vector, top_k=3)   # cosine similarity search
4. build_context(docs)              # format docs into prompt-ready text
```

### First-request latency

The embedding model is lazy-loaded on first use. The warmup in `main.py` forces this load before any real request arrives. Without warmup, the first retrieval call blocks for ~30 seconds while the model downloads and initializes.

The retrieval future has a 35-second timeout (vs 8 seconds for other tasks) specifically to handle the case where warmup has not fully completed by the time the first user message arrives.

### Why top-k=3

Retrieving more documents increases the chance of finding relevant content but also increases prompt length and can introduce noise. Three documents is a balance point that keeps the context window manageable while covering most clinical topics present in the dataset.

---

## 13. Caching System

**File:** `backend/rag/smart_cache.py`

### Cache key

```python
cache_key = f"{session_id}:{normalized_query}"
```

The key combines session ID and normalized query. This means two different users asking the same question get separate cache entries — which is correct because their conversation context differs and the response may be personalized.

### Corruption guard

```python
def _cache_get(key):
    value = cache.get(key)
    if isinstance(value, str) and len(value) >= 15:
        return value
    if value is not None:
        cache.delete(key)   # evict invalid entry
    return None
```

If a stream is interrupted mid-way (server restart, timeout), the partial response may have been saved to cache. A minimum length of 15 characters filters out these fragments before they reach users. The invalid entry is deleted rather than just ignored, so it does not continue consuming cache space.

### What is not cached

- Emergency responses: the emergency check fires before the cache lookup, so emergency queries never write to cache
- Quick responses: these return before the cache check
- Weak responses: the weak response check runs after the LLM and before the cache write; weak responses fall through to the fallback instead

---

## 14. Parallel Execution

**File:** `backend/rag/pipeline.py` (executor section)

```python
executor = ThreadPoolExecutor(max_workers=4)

futures = {
    "analysis":  executor.submit(_analysis_fn,  question),
    "retrieval": executor.submit(_retrieval_fn, question),
    "fallback":  executor.submit(_fallback_fn,  question),
    "memory":    executor.submit(_memory_fn,    session_id),
}
```

All four tasks are submitted simultaneously and run in parallel threads. The results are collected with individual timeouts:

```python
results = {
    k: _safe_future(v, k, timeout=35.0 if k == "retrieval" else 8.0)
    for k, v in futures.items()
}
```

### Why ThreadPool and not asyncio tasks

All four functions (`brain.analyze`, `retrieve`, `fallback.get`, `memory.format`) are synchronous and blocking. `asyncio.create_task()` only works with coroutines. Wrapping synchronous functions in `asyncio.to_thread()` or `loop.run_in_executor()` would work but adds complexity inside the pipeline generator. `ThreadPoolExecutor.submit()` is the straightforward solution for running blocking code in parallel.

### Latency impact

Sequential execution of these four tasks would take approximately 3-6 seconds (dominated by retrieval on cold start and brain.analyze on first use). Parallel execution reduces this to the time of the slowest task — typically retrieval at 20-80ms after warmup.

---

## 15. Fault Tolerance Map

| Component | Failure mode | System behavior |
|---|---|---|
| Redis | Connection refused | Falls back to in-process TTLCache. Memory still works but does not persist across restarts. |
| Qdrant | Connection refused | `retrieval_fn` returns empty string. LLM generates without retrieved context. |
| LLM provider (primary) | API error / timeout | `LLMManager` tries next provider in chain. |
| All LLM providers | All unavailable | `SafeFallback` pre-written response returned if available. Otherwise static error message. |
| `brain.analyze()` | Exception | Returns `_ANALYSIS_DEFAULTS` (emergency=False, severity=low). Emergency regex still runs independently. |
| `SmartCache` | Get/set exception | Logged and silently skipped. Response is generated normally without caching. |
| Stream interrupted | WebSocket disconnect | Generator is abandoned. No crash. Memory is not saved for the incomplete response. |
| Cache entry corrupt | Short or wrong type | Entry is evicted. Fresh pipeline run executed. |

---

## 16. Performance Notes

### Latency breakdown per request

| Stage | Typical time |
|---|---|
| Cache hit | ~10ms |
| Quick response | ~1ms |
| Parallel tasks (after warmup) | ~80ms |
| LLM first token | ~400-900ms |
| Full response stream | 2-6 seconds |
| First request (cold, no warmup) | 30-35 seconds |

### Throughput

The async FastAPI server handles concurrent WebSocket connections without blocking. The ThreadPoolExecutor with 4 workers means up to 4 concurrent pipeline executions can run parallel tasks simultaneously. Additional requests queue but do not block the event loop.

### Memory per session

Each session stores up to 10 message turns (then compressed). At ~1200 characters per turn maximum (`_safe_text` truncation), a session uses at most ~12KB in Redis before compression. The TTL cache holds 500 sessions in process.

---

## 17. Known Issues and Limitations

### No fine-tuned medical model

The system uses general-purpose LLMs (Groq/Gemini/OpenRouter) with a medical system prompt and retrieved context. This is effective for common medical topics present in the dataset but may produce less accurate responses for rare conditions or highly technical clinical questions.

### Arabic normalization edge cases

The normalization function handles common variants (hamza forms, taa marbuta, alef maqsura) but does not handle all dialectal Arabic spelling variations. A query spelled differently from the indexed documents may fail to retrieve relevant results even if semantically equivalent.

### Emergency detection coverage

The regex covers common high-risk symptoms in Egyptian Arabic phrasing. It does not cover all possible phrasings, and it does not cover symptoms described in formal Arabic or other dialects. The ML layers (brain, router) add coverage, but the system should not be relied upon as a complete medical safety net.

### External LLM dependency

All response generation depends on third-party API availability. If all three providers (Groq, Gemini, OpenRouter) are unavailable simultaneously, the system returns pre-written fallback responses or a static error message. There is no local inference fallback.

### Session memory not encrypted

Redis stores conversation history in plain JSON. If the Redis instance is accessible outside the Docker network, session data is readable. For any deployment beyond a private development environment, Redis should be configured with authentication and the data should be considered sensitive.

---

## Contributors

- Mohamed Ramzy — AI Engineering Student
- Mohamed Reda — AI Engineering Student

---

## License

Educational and research use only. Not intended for clinical or medical diagnosis.
