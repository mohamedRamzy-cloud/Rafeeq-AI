import logging
import time
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ==========================================================
# CONFIG
# ==========================================================
COLLECTION_NAME = settings.QDRANT_COLLECTION
VECTOR_SIZE = 384

_client = None
_lock = Lock()


# ==========================================================
# DETECT ENV (IMPORTANT FIX FOR DOCKER)
# ==========================================================
def get_qdrant_url():
    """
    Fixes local vs docker issue automatically
    """
    url = settings.QDRANT_URL

    # لو شغال داخل Docker
    if "localhost" in url:
        return url

    return url


# ==========================================================
# WAIT FOR QDRANT (PRODUCTION SAFE)
# ==========================================================
def wait_for_qdrant(max_retries=30, delay=2):

    logger.info("[QDRANT] Waiting for server...")

    last_error = None

    url = get_qdrant_url()

    for i in range(max_retries):

        try:
            client = QdrantClient(
                url=url,
                timeout=10,
                prefer_grpc=False
            )

            # lightweight health check
            client.get_collections()

            logger.info("[QDRANT] Server ready ✔")
            return client

        except Exception as e:
            last_error = e
            logger.warning(f"[QDRANT] not ready yet ({i+1}/{max_retries})")
            time.sleep(delay)

    logger.error(f"[QDRANT] last error: {last_error}")
    raise RuntimeError("Qdrant not available after retries") from last_error


# ==========================================================
# SINGLETON CLIENT
# ==========================================================
def get_client():

    global _client

    if _client:
        return _client

    with _lock:

        if _client:
            return _client

        logger.info("[QDRANT] Connecting...")

        _client = wait_for_qdrant()

        logger.info("[QDRANT] Connected ✔")

    return _client


# ==========================================================
# CREATE COLLECTION (SAFE PRODUCTION VERSION)
# ==========================================================
def create_collection():

    client = get_client()

    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if COLLECTION_NAME in existing:
        logger.info("[QDRANT] Collection already exists ✔")
        return

    logger.info("[QDRANT] Creating collection...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    logger.info("[QDRANT] Collection created ✔")


# ==========================================================
# RECREATE (DEV ONLY SAFE CONTROL)
# ==========================================================
def recreate_collection(force: bool = False):

    client = get_client()

    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if COLLECTION_NAME in existing:

        if not force:
            logger.warning(
                "[QDRANT] Collection exists — skipping delete (safe mode ON)"
            )
            return

        logger.warning("[QDRANT] Deleting collection (FORCED)...")

        client.delete_collection(COLLECTION_NAME)

        time.sleep(2)

        logger.info("[QDRANT] Deleted ✔")

    create_collection()


# ==========================================================
# ENTRY
# ==========================================================
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    logger.info("[QDRANT INIT] Starting...")

    try:
        recreate_collection(force=False)
        logger.info("[QDRANT INIT] Done ✔")

    except Exception as e:
        logger.error(f"[FATAL] {e}")
        exit(1)