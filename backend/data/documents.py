from langchain_core.documents import Document


def build_documents(dataset):
    docs = []

    for ex in dataset:
        if not ex.get("question") or not ex.get("answer"):
            continue

        content = f"""
التخصص: {ex.get('specialization','')}

السؤال: {ex.get('question','')}

الجواب: {ex.get('answer','')}
""".strip()

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "specialization": ex.get("specialization", "")
                }
            )
        )

    return docs



