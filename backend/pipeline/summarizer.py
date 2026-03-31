import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from transformers import Pipeline

from backend import config
from backend.schemas import ProcessedPost, SummaryResult
from backend.utils.text_utils import chunk_by_tokens, extractive_filter

bart_tokenizer: Any = None


def set_bart_tokenizer(tokenizer: Any) -> None:
    global bart_tokenizer
    bart_tokenizer = tokenizer


def run_map_refine(chunks: list[str], bart_pipeline: Pipeline) -> str:
    if not chunks:
        return ""

    if len(chunks) == 1:
        return bart_pipeline(chunks[0])[0]["summary_text"]

    running_summary = bart_pipeline(
        chunks[0],
        max_length=200,
        min_length=50,
        do_sample=False,
    )[0]["summary_text"]

    for chunk in chunks[1:]:
        refine_input = f"Context so far: {running_summary} New information to add: {chunk}"

        if bart_tokenizer is not None:
            refine_tokens = bart_tokenizer.encode(refine_input, add_special_tokens=False)
            if len(refine_tokens) > config.CHUNK_MAX_TOKENS:
                refine_input = bart_tokenizer.decode(
                    refine_tokens[: config.CHUNK_MAX_TOKENS],
                    skip_special_tokens=True,
                )

        running_summary = bart_pipeline(
            refine_input,
            max_length=250,
            min_length=60,
            do_sample=False,
        )[0]["summary_text"]

    return running_summary


def detect_hallucination(summary: str) -> bool:
    matches = re.findall(r"\b(i|me|my|we|our|ours)\b", summary.lower())
    return len(matches) > 3


def _normalize_sentiment(label: str) -> str:
    normalized = label.lower()
    if "pos" in normalized:
        return "positive"
    if "neg" in normalized:
        return "negative"
    if "neu" in normalized:
        return "neutral"
    return "neutral"


def summarize(
    posts: list[ProcessedPost],
    query: str,
    bart_pipeline: Pipeline,
    sentiment_pipeline: Pipeline,
    nlp,
) -> SummaryResult:
    full_text = " ".join([post.combined_text for post in posts]).strip()
    total_words = len(full_text.split())

    if total_words < 100:
        confidence = "low"
    elif total_words < 500:
        confidence = "medium"
    else:
        confidence = "high"

    filtered_text = extractive_filter(full_text, query, keep_ratio=0.6)
    if bart_tokenizer is not None:
        chunks = chunk_by_tokens(filtered_text, bart_tokenizer, config.CHUNK_MAX_TOKENS)
    else:
        chunks = [filtered_text] if filtered_text else []

    if total_words < 100:
        summary = extractive_filter(full_text, query, keep_ratio=0.3)
    else:
        summary = run_map_refine(chunks, bart_pipeline)

    if detect_hallucination(summary):
        retry_input = filtered_text or full_text
        if bart_tokenizer is not None and retry_input:
            retry_tokens = bart_tokenizer.encode(retry_input, add_special_tokens=False)
            if len(retry_tokens) > config.CHUNK_MAX_TOKENS:
                retry_input = bart_tokenizer.decode(
                    retry_tokens[: config.CHUNK_MAX_TOKENS],
                    skip_special_tokens=True,
                )
        if retry_input:
            summary = bart_pipeline(
                retry_input,
                max_length=250,
                min_length=60,
                do_sample=False,
                num_beams=1,
            )[0]["summary_text"]

    if not summary or len(summary.strip()) < 20:
        summary = (
            "Summary could not be generated with confidence. "
            + "Top posts: "
            + " | ".join([post.title for post in posts[:3]])
        )

    sample_text = (full_text[:1000] or summary[:1000]).strip()
    sentiment_result = sentiment_pipeline(sample_text[:512])[0]
    sentiment = _normalize_sentiment(sentiment_result.get("label", "neutral"))
    sentiment_score = round(float(sentiment_result.get("score", 0.0)), 3)

    key_labels = ["PERSON", "ORG", "GPE", "EVENT"]
    counters = {label: Counter() for label in key_labels}

    if nlp is not None and full_text:
        doc = nlp(full_text[:5000])
        for ent in doc.ents:
            if ent.label_ in counters:
                counters[ent.label_][ent.text.strip()] += 1

    key_entities = {
        label: [name for name, _ in counters[label].most_common()]
        for label in key_labels
    }

    topic_counter = Counter(topic for post in posts for topic in post.topics)
    topics = [topic for topic, _ in topic_counter.most_common(15)]

    return SummaryResult(
        query=query,
        sources_used=[],
        post_count=len(posts),
        posts_after_dedup=len(posts),
        topics=topics,
        summary=summary.strip(),
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        key_entities=key_entities,
        top_posts=[
            {
                "title": post.title,
                "url": "",
                "score": 0,
                "source": post.source,
                "num_comments": 0,
            }
            for post in posts[:5]
        ],
        news_headlines=[],
        confidence=confidence,
        generated_at=datetime.now(timezone.utc),
        timing={
            "fetch_ms": 0,
            "preprocess_ms": 0,
            "summarize_ms": 0,
            "total_ms": 0,
        },
    )
