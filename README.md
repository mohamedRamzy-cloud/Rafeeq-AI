<div align="center">

#  Rafeeq AI
### Arabic Medical Assistant — Production RAG System

[![Python](https://img.shields.io/badge/Python-3.11+-1A6B8A?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-27AE8F?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC143C?style=for-the-badge)](https://qdrant.tech)
[![Redis](https://img.shields.io/badge/Redis-Session_Memory-FF4438?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-Educational_Use-orange?style=for-the-badge)](LICENSE)

<br/>

> **Real-time Arabic medical assistant powered by RAG — grounding every answer in retrieved medical knowledge, not just model memory.**

<br/>

</div>

---

##  Disclaimer

> This system is intended for **educational and research purposes only**.
> It does **not** replace professional medical consultation.
> Always consult a qualified physician for medical decisions.

---

##  What Makes Rafeeq Different?

Traditional LLM chatbots rely purely on model weights — which can hallucinate.
Rafeeq uses a **Retrieval-Augmented Generation (RAG)** pipeline:

```
Question  →  Retrieve relevant medical docs  →  Inject as context  →  Generate grounded answer
```

Every response is backed by retrieved knowledge, not guesswork.

---

##  System Architecture

```
User
 │
 ▼
Frontend  (HTML / JS WebSocket Client)
 │
 ▼
FastAPI Server  (main.py)
 ├── POST   /chat          → HTTP Streaming
 ├── WS     /chat/ws       → Real-time WebSocket
 ├── POST   /chat/new      → Clear session memory
 └── GET    /health        → Health check
 │
 ▼
ChatService  (stream orchestration)
 │
 ▼
RAG Pipeline  (pipeline.py)
 ├──  Emergency Detector     (regex hard-guard — fires BEFORE anything else)
 ├──  Quick Response Layer    (greetings, identity — no LLM needed)
 ├──   Smart Cache            (session + query key)
 ├──  Parallel Executor      (ThreadPoolExecutor × 4)
 │    ├── MedicalAIBrain        (intent + severity classification)
 │    ├── Qdrant Retrieval      (semantic vector search)
 │    ├── SafeFallback          (pre-written fallback answers)
 │    └── ProMemory             (Redis session history)
 ├──  Prompt Builder         (system + context + real chat turns)
 └──  LLM Manager            (Groq → Gemini → OpenRouter)
      │
      ▼
     Word-boundary stream buffer  →  40-char batch  →  Client
```

---

##  Features

###  Core AI
| Feature | Description |
|---|---|
| RAG Pipeline | Retrieves top-K medical docs from Qdrant before every response |
| MedicalAIBrain | Classifies intent, specialty, severity, and emergency flag |
| Multi-Provider LLM | Auto-failover: Groq → Google Gemini → OpenRouter |
| Weak Response Fallback | Detects poor LLM output and replaces with pre-written answer |

###  Safety System (Multi-Layer)
| Layer | Method |
|---|---|
| Layer 1 | **Regex hard-guard** — fires before cache, before LLM |
| Layer 2 | **MedicalAIBrain** classification (`emergency=True`) |
| Layer 3 | **ResponseRouter** escalation (`route="emergency"`) |
| Layer 4 | **Severity scoring** (`severity="high"`) |

> Any single layer firing returns an immediate emergency response — no LLM call made.

###  Performance
| Optimization | Impact |
|---|---|
| Parallel pipeline (ThreadPoolExecutor) | ~2.5× speedup |
| Smart cache (session + query key) | 60–80% hit rate |
| Stream batch buffering (40 chars) | ~70% I/O reduction |
| Word-boundary flush | Zero mid-word text cuts |
| Startup warmup (async thread pool) | <1s first token after boot |

###  Real-Time Streaming
- WebSocket streaming with ping/pong keepalive (every 30s)
- HTTP `StreamingResponse` — first token in <1s
- Arabic word-boundary buffering — no split words delivered to client
- Connection registry with graceful shutdown

###  Memory & Context
- Redis-backed session memory with 24h TTL
- Local `TTLCache` fallback (500 sessions, 1h)
- Conversation injected as **real chat turns** (`human`/`assistant`) — not flat text
- Conversation summary compression after 10+ turns
- `RLock` (reentrant) prevents deadlock on nested memory calls

---

##  API Reference

### HTTP

```http
POST /chat
Content-Type: application/json

{
  "question":   "ما أسباب الصداع المستمر؟",
  "session_id": "user_123"
}
```

Returns: `text/plain` stream (chunked).

---

```http
POST /chat/new
Content-Type: application/json

{ "session_id": "user_123" }
```

Clears conversation memory for the session.

---

```http
GET /health
```

```json
{ "status": "healthy", "ws_connections": 4 }
```

---

### WebSocket

```
ws://localhost:8000/chat/ws
```

**Client → Server**
```json
{ "question": "عندي صداع من امبارح", "session_id": "user_123" }
```

**Server → Client**
```json
{ "type": "chunk",  "content": "متقلقش...", "session_id": "user_123" }
{ "type": "done",   "session_id": "user_123" }
{ "type": "ping" }
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn + WebSockets |
| Vector DB | Qdrant |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM Providers | Groq API, Google Gemini, OpenRouter |
| Memory | Redis + cachetools TTLCache |
| Containerization | Docker + Docker Compose |
| Language | Python 3.11+ (async) |

---

##  Project Structure

```
backend/
├── api/
│    └── main.py              # FastAPI routes, WS handler, warmup
|    └── schemas.py
├── services/
│    └── chat_service.py      # Stream orchestration
├── rag/
│    ├── pipeline.py          # Main RAG pipeline (run_pipeline)
│    ├── prompt_builder.py    # Prompt assembly
│    ├── embeddings.py
│    ├── retrieval.py
|    ├── context_builder.py
|    ├── emergency_detector
|    ├── utils.py
|    ├── medical_engine.py
│    ├── response_router.py
│    ├── smart_cache.py
|    ├── preprocessing_query.py
│    └── fallback_model.py
├── ai/
│    └── medical_brain.py     # Intent + emergency classification
├── llm/
│    └── manager.py           # Multi-provider LLM routing
|    └── models.py
|    └── base_provider.py
|    └── groq_provider.py
|    └── gemini_provider.py
|    └── openrouter_provider.py
|    └── llm_utils.py
├── memory/
│    └── memory.py            # ProMemory (Redis + local cache)
├── vectorstore/
│    └── qdrant_init.py
|    └── qdrant_service.py
|    └── qdrant_upload.py
├── data/           # PreProcessing Dataset
|    └── cleaning.py
|    └── loader.py
|    └── documents.py       
└── core/
     └── config.py            # Settings + validation
```

---

##  Setup & Deployment

### Prerequisites
- Docker + Docker Compose
- At least one API key: Groq, Google Gemini, or OpenRouter

### 1. Clone
```bash
git clone https://github.com/your-username/rafeeq-ai.git
cd rafeeq-ai
```

### 2. Configure environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
# At least one required
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Optional (defaults shown)
REDIS_HOST=redis
REDIS_PORT=6379
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=medical_rag
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:5500
DEBUG=false
```

### 3. Run
```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---

##  Environment Variables

| Variable | Default | Required |
|---|---|---|
| `GROQ_API_KEY` | — | One of three |
| `GOOGLE_API_KEY` | — | One of three |
| `OPENROUTER_API_KEY` | — | One of three |
| `REDIS_HOST` | `redis` | No |
| `REDIS_PORT` | `6379` | No |
| `QDRANT_HOST` | `qdrant` | No |
| `QDRANT_PORT` | `6333` | No |
| `QDRANT_URL` | auto-built | No |
| `QDRANT_COLLECTION` | `medical_rag` | No |
| `ALLOWED_ORIGINS` | localhost | No |
| `DEBUG` | `false` | No |

> If `QDRANT_URL` is set, it takes priority over `QDRANT_HOST`/`QDRANT_PORT`.

---

##  Pipeline Flow (Step by Step)

```
1. Sanitize input          — strip null bytes, normalize Arabic text
2. Quick response?         — return instantly, no LLM
3. Emergency check      — regex fires BEFORE cache or LLM
4. Cache hit?              — return cached response instantly
5. Add to memory           — save user turn
6. Run parallel tasks      — analysis + retrieval + fallback + memory
7. Route check             — normal / emergency / followup / rag
8. Build prompt            — system + context + chat history + question
9. Stream LLM              — word-boundary buffered chunks
10. Post-process           — clean → cache → save to memory
```

---

##  RAG Pipeline Design Decisions

| Decision | Reason |
|---|---|
| RAG over pure LLM | Reduces hallucination — answers grounded in retrieved docs |
| Streaming over batch | User sees first word in <1s |
| Real chat turns in prompt | `(human, ...) / (assistant, ...)` gives model true conversation context |
| Emergency before cache | Safety cannot be skipped even on cache hit |
| `RLock` in ProMemory | Prevents deadlock when `load()` is called inside `add()` |
| Singleton `Settings` | Prevents double Redis/Qdrant init from multiple imports |
| `run_in_executor` warmup | Keeps event loop free during heavy model loading |

---

##  Roadmap

- [ ] Fine-tuned Arabic medical LLM
- [ ] Medical knowledge graph integration
- [ ] Voice-to-text interface
- [ ] JWT authentication + user management
- [ ] Admin dashboard + analytics
- [ ] Cloud deployment (AWS / GCP)
- [ ] FHIR-compatible medical data ingestion

---

##  Authors

| Name | Role |
|---|---|
| **Mohamed Ramzy** | AI Engineering Student |
| **Mohamed Reda** | AI Engineering Student |

---

##  License

This project is for **educational and research use only**.
Not intended for clinical or medical diagnosis.

---

<div align="center">

Built with ❤️ — Rafeeq AI Team

</div>