import logging
import re
from collections import Counter
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from backend import config
from backend.schemas import DiscussionPost, ProcessedPost
from backend.utils.text_utils import clean_text

logger = logging.getLogger(__name__)


_FALLBACK_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "will",
    "your",
    "about",
    "into",
    "just",
    "than",
    "then",
    "they",
    "them",
    "their",
    "there",
    "what",
    "when",
    "where",
    "which",
    "while",
    "would",
    "could",
    "should",
    "after",
    "before",
    "because",
    "been",
    "being",
    "also",
    "only",
    "over",
    "under",
    "more",
    "most",
    "such",
    "many",
    "some",
    "very",
    "much",
    "each",
    "other",
    "those",
    "these",
    "were",
    "here",
    "make",
}


def extract_topics(text: str, nlp) -> list[str]:
    try:
        if nlp is None:
            raise ValueError("spaCy model unavailable")
        doc = nlp(text)
        lemmas = [
            token.lemma_.strip().lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and len(token.lemma_.strip()) > 2
            and token.lemma_.strip().isalpha()
        ]
        counts = Counter(lemmas)
        return [lemma for lemma, _ in counts.most_common(10)]
    except Exception:
        logger.warning("spaCy not available, using regex fallback for topics")
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)]
        filtered = [word for word in words if word not in _FALLBACK_STOPWORDS]
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(10)]


def deduplicate_posts(posts: list[DiscussionPost], sentence_model: Any) -> list[DiscussionPost]:
    if len(posts) <= 1:
        return posts

    embeddings = sentence_model.encode([post.title for post in posts], convert_to_numpy=True)
    similarity_matrix = cosine_similarity(embeddings)

    visited: set[int] = set()
    deduped: list[DiscussionPost] = []

    for i in range(len(posts)):
        if i in visited:
            continue

        cluster = [i]
        queue = [i]
        visited.add(i)

        while queue:
            current = queue.pop()
            for j in range(len(posts)):
                if j in visited:
                    continue
                if similarity_matrix[current][j] > config.SIMILARITY_DEDUP_THRESHOLD:
                    visited.add(j)
                    queue.append(j)
                    cluster.append(j)

        best_idx = max(cluster, key=lambda idx: posts[idx].score)
        deduped.append(posts[best_idx])

    removed = len(posts) - len(deduped)
    logger.info("Removed %d duplicate posts, kept %d unique", removed, len(deduped))
    return deduped


def preprocess_posts(
    posts: list[DiscussionPost],
    query: str,
    nlp,
    sentence_model: Any,
) -> list[ProcessedPost]:
    deduplicated_posts = deduplicate_posts(posts, sentence_model)

    processed: list[ProcessedPost] = []
    for post in deduplicated_posts:
        cleaned_title = clean_text(post.title)
        cleaned_body = clean_text(post.body) if len(post.body) > 50 else ""
        comments_text = " ||| ".join(post.comments)
        combined = cleaned_title + (". " + cleaned_body if cleaned_body else "") + comments_text
        combined = combined.strip()

        vectors = sentence_model.encode([combined, query], convert_to_numpy=True)
        relevance_score = float(cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))[0][0])

        processed_post = ProcessedPost(
            id=post.id,
            title=cleaned_title,
            combined_text=combined,
            topics=extract_topics(combined, nlp),
            word_count=len(combined.split()),
            relevance_score=max(0.0, min(1.0, relevance_score)),
            source=post.source,
        )
        processed.append(processed_post)

    processed.sort(key=lambda item: item.relevance_score, reverse=True)
    return processed
