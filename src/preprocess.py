import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning for sentiment analysis.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # keep only letters/numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text