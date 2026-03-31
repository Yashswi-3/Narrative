import html
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def clean_text(raw: str) -> str:
    text = raw or ""
    text = _URL_PATTERN.sub(" ", text)
    text = _ENTITY_PATTERN.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"(\*\*|__|~~)", "", text)
    text = re.sub(r"(?m)^\s*>+\s*", "", text)
    text = re.sub(r"(?m)^\s*#+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def filter_comments(comments: list[str], min_len: int = 30) -> list[str]:
    filtered: list[str] = []
    for comment in comments:
        cleaned = (comment or "").strip()
        if len(cleaned) < min_len:
            continue
        if re.fullmatch(r"(https?://\S+|www\.\S+)", cleaned, re.IGNORECASE):
            continue
        words = cleaned.split()
        if cleaned.endswith("?") and len(words) < 8:
            continue
        filtered.append(cleaned)
    return filtered


def chunk_by_tokens(text: str, tokenizer: Any, max_tokens: int = 900) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return []

    sentences = [s.strip() for s in normalized.split(". ") if s.strip()]
    chunks: list[str] = []
    current_sentences: list[str] = []

    def encode_len(value: str) -> int:
        return len(tokenizer.encode(value, add_special_tokens=False))

    for sentence in sentences:
        candidate = sentence if not current_sentences else ". ".join(current_sentences + [sentence])
        if encode_len(candidate) <= max_tokens:
            current_sentences.append(sentence)
            continue

        if current_sentences:
            chunk_text = ". ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if len(sentence_tokens) <= max_tokens:
            current_sentences = [sentence]
            continue

        start = 0
        while start < len(sentence_tokens):
            token_slice = sentence_tokens[start : start + max_tokens]
            part = tokenizer.decode(token_slice, skip_special_tokens=True).strip()
            if part:
                chunks.append(part)
            start += max_tokens
        current_sentences = []

    if current_sentences:
        last_chunk = ". ".join(current_sentences).strip()
        if last_chunk:
            chunks.append(last_chunk)

    return chunks


def score_sentence_relevance(sentences: list[str], query: str) -> list[tuple[str, float]]:
    if not sentences:
        return []

    corpus = sentences + [query]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)

    sentence_matrix = matrix[:-1]
    query_vector = matrix[-1]
    similarities = cosine_similarity(sentence_matrix, query_vector).ravel()

    scored = list(zip(sentences, similarities.tolist()))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def extractive_filter(text: str, query: str, keep_ratio: float = 0.6) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return ""

    sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(normalized) if s.strip()]
    if not sentences:
        return normalized

    scored = score_sentence_relevance(sentences, query)
    keep_count = max(3, int(len(sentences) * keep_ratio))
    keep_count = min(len(sentences), keep_count)

    selected = {sentence for sentence, _ in scored[:keep_count]}
    ordered = [sentence for sentence in sentences if sentence in selected]

    if len(ordered) < min(3, len(sentences)):
        ordered = sentences[: min(3, len(sentences))]

    return " ".join(ordered).strip()
