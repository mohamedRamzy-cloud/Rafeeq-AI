import hashlib
import logging
import uuid

from tqdm import tqdm
from qdrant_client.models import PointStruct

from backend.vectorstore.qdrant_service import (
    get_qdrant,
    get_embedder
)

from backend.vectorstore.qdrant_init import COLLECTION_NAME
from backend.data.loader import load_medical_dataset
from backend.data.cleaning import (
    parse_text,
    clean_example,
    filter_bad
)
from backend.data.documents import build_documents

logger = logging.getLogger(__name__)


# ==========================================================
# CONFIG
# ==========================================================
BATCH_SIZE = 64
EMBED_BATCH_SIZE = 32


# ==========================================================
# STABLE ID (PRODUCTION SAFE UUID5)
# ==========================================================
def stable_id(text: str, meta: str = "") -> str:

    raw = f"{text}-{meta}"

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            raw
        )
    )


# ==========================================================
# SAFE COLLECTION CHECK
# ==========================================================
def collection_has_data(
    client,
    collection_name: str
) -> bool:

    try:

        info = client.get_collection(
            collection_name=collection_name
        )

        collection_count = getattr(
            info,
            "points_count",
            0
        )

        logger.info(
            f"[UPLOAD] Existing points in Qdrant: "
            f"{collection_count}"
        )

        if collection_count > 0:

            logger.info(
                "[UPLOAD] Already exists in Qdrant "
                "✔ skipping ingestion"
            )

            return True

        return False

    except Exception as e:

        logger.warning(
            f"[COLLECTION CHECK ERROR] {e}"
        )

        return False


# ==========================================================
# DATA PREPARATION
# ==========================================================
def prepare_dataset():

    logger.info("[UPLOAD] Loading dataset...")

    data = load_medical_dataset()

    processed = []

    for item in tqdm(data, desc="Cleaning"):

        try:

            parsed = parse_text(item)

            cleaned = clean_example(parsed)

            if filter_bad(cleaned):
                processed.append(cleaned)

        except Exception as e:

            logger.warning(
                f"[CLEAN ERROR] {e}"
            )

    logger.info(
        f"[UPLOAD] Cleaned items: "
        f"{len(processed)}"
    )

    docs = build_documents(processed)

    # ======================================================
    # DEDUPLICATION
    # ======================================================
    seen = set()

    unique = []

    for doc in docs:

        text = (
            doc.page_content or ""
        ).strip()

        if not text:
            continue

        text_hash = hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()

        if text_hash in seen:
            continue

        seen.add(text_hash)

        unique.append(doc)

    logger.info(
        f"[UPLOAD] Final unique docs: "
        f"{len(unique)}"
    )

    return unique


# ==========================================================
# VECTOR VALIDATION
# ==========================================================
def validate_vectors(vectors) -> bool:

    # None check
    if vectors is None:

        logger.warning(
            "[UPLOAD] vectors is None"
        )

        return False

    # numpy array safe check
    if hasattr(vectors, "size"):

        if vectors.size == 0:

            logger.warning(
                "[UPLOAD] empty numpy vectors"
            )

            return False

    # generic len check
    try:

        if len(vectors) == 0:

            logger.warning(
                "[UPLOAD] empty vectors"
            )

            return False

    except Exception as e:

        logger.warning(
            f"[UPLOAD] vector length check failed: {e}"
        )

        return False

    return True


# ==========================================================
# POINTS BUILDER
# ==========================================================
def build_points(docs, vectors):

    points = []

    for i, doc in enumerate(docs):

        try:

            text = (
                doc.page_content or ""
            ).strip()

            if not text:
                continue

            meta = doc.metadata or {}

            vec = vectors[i]

            # numpy -> list
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            points.append(

                PointStruct(
                    id=stable_id(
                        text,
                        meta.get(
                            "specialization",
                            ""
                        )
                    ),

                    vector=vec,

                    payload={
                        "text": text,

                        "specialization":
                            meta.get(
                                "specialization",
                                ""
                            ),

                        "source":
                            meta.get(
                                "source",
                                "medical_dataset"
                            )
                    }
                )
            )

        except Exception as e:

            logger.warning(
                f"[POINT BUILD ERROR] {e}"
            )

    return points


# ==========================================================
# UPLOAD PIPELINE
# ==========================================================
def upload_data():

    logger.info(
        "[UPLOAD] Starting ingestion pipeline..."
    )

    client = get_qdrant()

    # ======================================================
    # CHECK COLLECTION FIRST
    # ======================================================
    if collection_has_data(
        client,
        COLLECTION_NAME
    ):

        logger.info(
            "[UPLOAD] Skipping dataset loading "
            "(already exists)"
        )

        return

    embedder = get_embedder()

    docs = prepare_dataset()

    if not docs:

        logger.warning(
            "[UPLOAD] No documents found"
        )

        return

    total_uploaded = 0

    # ======================================================
    # BATCH UPLOAD
    # ======================================================
    for i in tqdm(
        range(0, len(docs), BATCH_SIZE),
        desc="Uploading"
    ):

        batch = docs[i:i + BATCH_SIZE]

        texts = [
            d.page_content
            for d in batch
        ]

        try:

            # ==============================================
            # EMBEDDINGS
            # ==============================================
            vectors = embedder.embed_documents(
                texts,
                batch_size=EMBED_BATCH_SIZE
            )

            # ==============================================
            # SAFE VALIDATION
            # ==============================================
            if not validate_vectors(vectors):
                continue

            # ==============================================
            # VECTOR COUNT CHECK
            # ==============================================
            if len(vectors) != len(batch):

                logger.warning(
                    "[UPLOAD] vector mismatch "
                    "skip batch"
                )

                continue

            # ==============================================
            # BUILD POINTS
            # ==============================================
            points = build_points(
                batch,
                vectors
            )

            if not points:

                logger.warning(
                    "[UPLOAD] empty points "
                    "skip batch"
                )

                continue

            # ==============================================
            # UPSERT
            # ==============================================
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=False
            )

            total_uploaded += len(points)

            logger.info(
                f"[UPLOAD] Uploaded batch: "
                f"{len(points)} points"
            )

        except Exception as e:

            logger.exception(
                f"[UPLOAD FAILED] {e}"
            )

    logger.info(
        f"[UPLOAD] Done ✔ "
        f"total uploaded: {total_uploaded}"
    )


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    print(
        "[PIPELINE] Starting ingestion..."
    )

    upload_data()

    print(
        "[PIPELINE] Done "
    )