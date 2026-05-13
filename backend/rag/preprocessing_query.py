from backend.data.cleaning import clean_text, normalize_arabic
import re

def preprocess_query(text: str) -> str:

    text = clean_text(text)
    text = normalize_arabic(text)

    # remove noise
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s]", "", text)

    text = text.lower()

    return text