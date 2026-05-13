import re


def clean_text(text):
    if not text:
        return ""

    text = str(text).strip()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_arabic(text):
    if not text:
        return ""

    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)

    return text


def parse_text(example):
    text = example.get("text", "")

    specialization, question, answer = "", "", ""

    for line in text.split("\n"):
        line = line.strip()

        if "التخصص:" in line:
            specialization = line.replace("التخصص:", "").strip()

        elif "السؤال:" in line:
            question = line.replace("السؤال:", "").strip()

        elif "الجواب:" in line:
            answer = line.replace("الجواب:", "").strip()

    return {
        "specialization": specialization,
        "question": question,
        "answer": answer
    }


def clean_example(example):
    return {
        "specialization": normalize_arabic(clean_text(example.get("specialization"))),
        "question": normalize_arabic(clean_text(example.get("question"))),
        "answer": normalize_arabic(clean_text(example.get("answer")))
    }


def filter_bad(example):
    return (
        example.get("question") and
        example.get("answer") and
        len(example.get("answer", "")) > 20
    )