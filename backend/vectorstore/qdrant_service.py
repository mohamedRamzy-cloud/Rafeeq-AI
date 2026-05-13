import logging

from backend.vectorstore.qdrant_init import (
    get_client
)

from backend.rag.embeddings import (
    EmbeddingModel
)


logger = logging.getLogger(__name__)

# ==========================================================
# GLOBAL SINGLETON
# ==========================================================
_embedder = None

# ==========================================================
# MODEL CONFIG
# ==========================================================
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================================
# EMBEDDING MODEL
# ==========================================================
def get_embedder():

    global _embedder

    if _embedder is not None:
        return _embedder

    logger.info(
        "[EMBEDDER] Loading model..."
    )

    _embedder = EmbeddingModel(
        EMBEDDING_MODEL
    )

    logger.info(
        "[EMBEDDER] Ready "
    )

    return _embedder


# ==========================================================
# QDRANT CLIENT
# ==========================================================
def get_qdrant():

    return get_client()


# ==========================================================
# HEALTH CHECK
# ==========================================================
def health_check():

    try:

        emb = get_embedder()

        vec = emb.embed_query(
            "اختبار"
        )

        qdrant = get_qdrant()

        qdrant.get_collections()

        return {
            "status": "ok",
            "embedding_size": len(vec),
            "model": EMBEDDING_MODEL
        }

    except Exception as e:

        logger.exception(
            f"[HEALTH CHECK FAILED] {e}"
        )

        return {
            "status": "error",
            "error": str(e)
        }


# ==========================================================
# RUN TEST
# ==========================================================
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print(
        "[SERVICE] Testing..."
    )

    result = health_check()

    print(result)

    print(
        "[SERVICE] Done"
    )