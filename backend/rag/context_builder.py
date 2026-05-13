import re
import hashlib


# ==========================================================
# CLEAN TEXT
# ==========================================================
def clean_text(text: str):

    if not text:
        return ""

    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# NORMALIZE
# ==========================================================
def normalize(text: str):

    text = clean_text(text)

    text = text.lower()

    # Arabic normalization
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


# ==========================================================
# DEDUPLICATION
# ==========================================================
def deduplicate_docs(docs):

    unique = []

    seen = set()

    for d in docs:

        text = d.get("text", "")

        normalized = normalize(text[:300])

        text_hash = hashlib.md5(
            normalized.encode("utf-8")
        ).hexdigest()

        if text_hash in seen:
            continue

        seen.add(text_hash)

        unique.append(d)

    return unique


# ==========================================================
# QUALITY FILTER
# ==========================================================
def is_good_chunk(text: str):

    if not text:
        return False

    text = clean_text(text)

    if len(text) < 60:
        return False

    garbage_patterns = [
        "؟؟",
        "###",
        "***",
        "http",
        ".com",
        "lorem ipsum"
    ]

    if any(g in text.lower() for g in garbage_patterns):
        return False

    return True


# ==========================================================
# SMART COMPRESSION
# ==========================================================
def compress_chunk(
    text: str,
    max_chars=500
):

    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    cut_points = [
        ".",
        "؟",
        "!",
        "،",
        "\n"
    ]

    truncated = text[:max_chars]

    best_cut = -1

    for c in cut_points:

        idx = truncated.rfind(c)

        if idx > best_cut:
            best_cut = idx

    if best_cut > 150:
        return truncated[:best_cut + 1]

    return truncated + "..."


# ==========================================================
# FORMAT CONTEXT BLOCK
# ==========================================================
def format_context_block(index, doc):

    text = doc.get("text", "")

    text = compress_chunk(text)

    specialty = doc.get(
        "specialty",
        ""
    )

    score = doc.get(
        "final_score",
        0
    )

    parts = [
        f"[Context {index}]",
        text
    ]

    if specialty:
        parts.append(
            f"(Specialty: {specialty})"
        )

    parts.append(
        f"(Relevance Score: {round(score, 3)})"
    )

    return "\n".join(parts)


# ==========================================================
# BUILD CONTEXT
# ==========================================================
def build_context(
    docs,
    max_docs=3
):

    if not docs:
        return ""

    # ======================================================
    # SORT BY SCORE
    # ======================================================
    docs = sorted(
        docs,
        key=lambda x: x.get(
            "final_score",
            x.get("score", 0)
        ),
        reverse=True
    )

    # ======================================================
    # REMOVE DUPLICATES
    # ======================================================
    docs = deduplicate_docs(docs)

    # ======================================================
    # LIMIT
    # ======================================================
    docs = docs[:max_docs]

    context_parts = []

    # ======================================================
    # BUILD CONTEXT
    # ======================================================
    for i, doc in enumerate(docs, 1):

        text = doc.get("text", "")

        text = clean_text(text)

        if not is_good_chunk(text):
            continue

        block = format_context_block(
            i,
            doc
        )

        context_parts.append(block)

    return "\n\n".join(context_parts)