import logging
import re
import hashlib

from cachetools import TTLCache
from backend.vectorstore.qdrant_service import get_client
from backend.rag.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medical_rag"

_client = None
_embedder = None

# BIG CACHE BOOST
_embedding_cache = TTLCache(maxsize=20000, ttl=7200)
_text_cache = TTLCache(maxsize=40000, ttl=7200)
_token_cache = TTLCache(maxsize=40000, ttl=7200)

# precompiled regex (fast path)
ARABIC = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
PUNCT = re.compile(r"[^\w\s]")
SPACE = re.compile(r"\s+")
ALEF = re.compile(r"[إأآا]")


def init():
    global _client, _embedder

    if _client is None:
        _client = get_client()

    if _embedder is None:
        _embedder = EmbeddingModel(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        _embedder.embed_query("warmup")


def get_client_safe():
    init()
    return _client


def get_embedder():
    init()
    return _embedder


def normalize_text(text: str):

    if not text:
        return ""

    cached = _text_cache.get(text)
    if cached:
        return cached

    text = text.lower().strip()
    text = ALEF.sub("ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = ARABIC.sub("", text)
    text = PUNCT.sub(" ", text)
    text = SPACE.sub(" ", text)

    _text_cache[text] = text
    return text


def tokenize(text: str):

    cached = _token_cache.get(text)
    if cached:
        return cached

    t = normalize_text(text)
    tokens = frozenset(w for w in t.split() if len(w) > 2)

    _token_cache[text] = tokens
    return tokens


def _key(text):
    return hashlib.md5(text.encode()).hexdigest()


def retrieve(query: str, k: int = 3):

    if not query:
        return []

    client = get_client_safe()
    embedder = get_embedder()

    q = normalize_text(query)
    key = _key(q)

    vector = _embedding_cache.get(key)

    if vector is None:
        vector = embedder.embed_query(q)
        _embedding_cache[key] = vector

    # LESS RESULTS = MUCH FASTER
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=6,   # reduced from dynamic scaling
        with_payload=True
    ).points

    if not results:
        return []

    q_tokens = tokenize(q)

    docs = []

    for r in results:

        p = getattr(r, "payload", None)
        if not p:
            continue

        text = p.get("text")
        if not text or len(text) < 80:
            continue

        score = float(getattr(r, "score", 0.0))

        if score < 0.25:
            continue

        t_tokens = tokenize(text)
        overlap = len(q_tokens & t_tokens)

        lexical = overlap / (len(q_tokens) + 1)

        final = (score * 0.92) + (lexical * 0.08)

        docs.append({
            "text": text,
            "score": round(final, 4),
            "source": p.get("source"),
            "specialty": p.get("specialty")
        })

    docs.sort(key=lambda x: x["score"], reverse=True)

    #fast dedup (no md5 per doc heavy slicing)
    seen = set()
    out = []

    for d in docs:
        h = hash(d["text"][:200])
        if h in seen:
            continue
        seen.add(h)
        out.append(d)

    return out[:k]