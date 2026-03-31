import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity

from backend import config
from backend.schemas import DiscussionPost, SearchRequest
from backend.utils.text_utils import clean_text, filter_comments

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10


def _wordnet_pos(tag: str) -> str | None:
    if tag.startswith("NN"):
        return wordnet.NOUN
    if tag.startswith("VB"):
        return wordnet.VERB
    return None


def expand_query(query: str) -> str:
    try:
        tokens = nltk.word_tokenize(query)
        tagged_tokens = nltk.pos_tag(tokens)
    except Exception:
        return query

    base_words = {w.lower() for w in tokens if w.isalpha()}
    synonyms: list[str] = []

    for word, tag in tagged_tokens:
        wn_pos = _wordnet_pos(tag)
        if wn_pos is None:
            continue

        synsets = wordnet.synsets(word, pos=wn_pos)
        if not synsets:
            continue

        chosen = None
        for lemma in synsets[0].lemma_names():
            normalized = lemma.replace("_", " ").strip().lower()
            if " " in normalized:
                continue
            if not normalized.isalpha():
                continue
            if normalized == word.lower() or normalized in base_words or normalized in synonyms:
                continue
            chosen = normalized
            break

        if chosen:
            synonyms.append(chosen)
            if len(synonyms) >= config.QUERY_EXPANSION_MAX_TERMS:
                break

    if not synonyms:
        return query

    return f"{query} {' '.join(synonyms)}"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return datetime.now(timezone.utc)


def _parse_unix_datetime(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _looks_technical(query: str) -> bool:
    technical_terms = {
        "api",
        "python",
        "java",
        "javascript",
        "node",
        "react",
        "sql",
        "database",
        "docker",
        "kubernetes",
        "code",
        "programming",
        "software",
        "bug",
        "deploy",
        "backend",
        "frontend",
        "linux",
    }
    tokens = {word.lower() for word in query.split()}
    return bool(tokens & technical_terms)


def _strip_html(value: str) -> str:
    try:
        return BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)
    except Exception:
        return clean_text(value or "")


def score_comments(comments: list[str], post_title: str, sentence_model: Any) -> list[str]:
    if not comments:
        return []

    try:
        embeddings = sentence_model.encode([post_title] + comments, convert_to_numpy=True)
        title_vector = embeddings[0].reshape(1, -1)
        comment_vectors = embeddings[1:]
        scores = cosine_similarity(comment_vectors, title_vector).ravel()
    except Exception:
        return comments[: config.MAX_COMMENTS_TO_KEEP]

    ranked = sorted(zip(comments, scores.tolist()), key=lambda item: item[1], reverse=True)
    return [comment for comment, _ in ranked[: config.MAX_COMMENTS_TO_KEEP]]


def fetch_hackernews(request: SearchRequest, sentence_model: Any) -> list[DiscussionPost]:
    expanded_query = expand_query(request.query)
    try:
        stories_response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "query": expanded_query,
                "tags": "story",
                "hitsPerPage": config.MAX_POSTS_PER_SOURCE,
            },
            timeout=REQUEST_TIMEOUT,
        )
        stories_response.raise_for_status()
        story_hits = stories_response.json().get("hits", [])
    except Exception as exc:
        logger.warning("Hacker News story fetch error: %s", exc)
        return []

    try:
        comments_response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": expanded_query, "tags": "comment", "hitsPerPage": 30},
            timeout=REQUEST_TIMEOUT,
        )
        comments_response.raise_for_status()
        comment_hits = comments_response.json().get("hits", [])
    except Exception as exc:
        logger.warning("Hacker News comment fetch error: %s", exc)
        comment_hits = []

    comments_by_story: dict[str, list[str]] = {}
    for comment in comment_hits:
        text = clean_text(comment.get("comment_text") or comment.get("text") or "")
        if not text:
            continue

        story_id = str(comment.get("story_id") or "")
        parent_id = str(comment.get("parent_id") or "")
        if story_id:
            comments_by_story.setdefault(story_id, []).append(text)
        if parent_id:
            comments_by_story.setdefault(parent_id, []).append(text)

    posts: list[DiscussionPost] = []
    for item in story_hits[: config.MAX_POSTS_PER_SOURCE]:
        post_id = str(item.get("objectID") or "")
        if not post_id:
            continue

        title = clean_text(item.get("story_title") or item.get("title") or "")
        if not title:
            title = "Untitled Hacker News story"

        body = clean_text(item.get("comment_text") or item.get("text") or "")
        raw_comments = filter_comments(
            comments_by_story.get(post_id, []),
            min_len=config.MIN_COMMENT_LENGTH,
        )
        scored_comments = score_comments(raw_comments, title, sentence_model)

        posts.append(
            DiscussionPost(
                id=post_id,
                title=title,
                url=item.get("story_url")
                or item.get("url")
                or f"https://news.ycombinator.com/item?id={post_id}",
                source="hackernews",
                score=_safe_int(item.get("points")),
                num_comments=_safe_int(item.get("num_comments"), len(raw_comments)),
                created_utc=_parse_iso_datetime(item.get("created_at")),
                body=body,
                comments=scored_comments,
            )
        )

    return posts


def _extract_bluesky_replies(node: dict[str, Any], collected: list[str]) -> None:
    if not isinstance(node, dict):
        return

    post = node.get("post") or {}
    record = post.get("record") or {}
    text = clean_text(record.get("text") or "")
    if text:
        collected.append(text)

    for reply in node.get("replies") or []:
        _extract_bluesky_replies(reply, collected)


def _fetch_bluesky_replies(uri: str) -> list[str]:
    try:
        response = requests.get(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread",
            params={"uri": uri, "depth": 3},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Bluesky replies fetch error: %s", exc)
        return []

    thread = payload.get("thread") or {}
    collected: list[str] = []
    for reply in thread.get("replies") or []:
        _extract_bluesky_replies(reply, collected)

    return collected


def fetch_bluesky(request: SearchRequest, sentence_model: Any) -> list[DiscussionPost]:
    try:
        response = requests.get(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
            params={"q": request.query, "limit": 25},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        posts_data = response.json().get("posts", [])
    except Exception as exc:
        logger.warning("Bluesky fetch error: %s", exc)
        return []

    posts: list[DiscussionPost] = []
    for item in posts_data[: config.MAX_POSTS_PER_SOURCE]:
        record = item.get("record") or {}
        body = clean_text(record.get("text") or "")
        if not body:
            continue

        uri = item.get("uri") or ""
        handle = ((item.get("author") or {}).get("handle") or "").strip()
        uri_parts = uri.rstrip("/").split("/")
        post_rkey = uri_parts[-1] if uri_parts else ""
        if handle and post_rkey:
            post_url = f"https://bsky.app/profile/{handle}/post/{post_rkey}"
        elif handle:
            post_url = f"https://bsky.app/profile/{handle}"
        else:
            post_url = "https://bsky.app"

        reply_count = _safe_int(item.get("replyCount"))
        raw_replies: list[str] = []
        if reply_count > 0 and uri:
            raw_replies = _fetch_bluesky_replies(uri)

        cleaned_replies = filter_comments(
            [clean_text(reply) for reply in raw_replies],
            min_len=config.MIN_COMMENT_LENGTH,
        )
        scored_comments = score_comments(cleaned_replies, body[:100], sentence_model)

        title = body[:100] + ("..." if len(body) > 100 else "")
        posts.append(
            DiscussionPost(
                id=uri or post_url,
                title=title,
                url=post_url,
                source="bluesky",
                score=_safe_int(item.get("likeCount")),
                num_comments=reply_count,
                created_utc=_parse_iso_datetime(record.get("createdAt")),
                body=body,
                comments=scored_comments,
            )
        )

    return posts


def _fetch_stackexchange_answers(question_ids: list[str], site: str) -> dict[str, list[str]]:
    if not question_ids:
        return {}

    id_fragment = ";".join(question_ids)
    params: dict[str, Any] = {
        "filter": "withbody",
        "site": site,
        "sort": "votes",
        "order": "desc",
        "pagesize": 30,
    }
    if config.STACKEXCHANGE_API_KEY:
        params["key"] = config.STACKEXCHANGE_API_KEY

    try:
        response = requests.get(
            f"https://api.stackexchange.com/2.3/questions/{id_fragment}/answers",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Stack Exchange answers fetch error (%s): %s", site, exc)
        return {}

    if payload.get("error_id") == 502:
        logger.warning("Stack Exchange quota exceeded while fetching answers")
        return {}

    grouped: dict[str, list[str]] = {}
    for answer in payload.get("items", []):
        parent_id = str(answer.get("question_id") or "")
        if not parent_id:
            continue
        text = clean_text(_strip_html(answer.get("body") or ""))
        if not text:
            continue
        grouped.setdefault(parent_id, []).append(text)

    return grouped


def fetch_stackexchange(request: SearchRequest, sentence_model: Any) -> list[DiscussionPost]:
    expanded_query = expand_query(request.query)
    sites = ["stackoverflow", "softwareengineering"] if _looks_technical(request.query) else [
        "stackoverflow",
        "politics",
        "skeptics",
    ]

    source_posts: list[DiscussionPost] = []
    for site in sites:
        params: dict[str, Any] = {
            "q": expanded_query,
            "site": site,
            "pagesize": config.MAX_POSTS_PER_SOURCE,
            "sort": "votes",
            "order": "desc",
            "filter": "withbody",
        }
        if config.STACKEXCHANGE_API_KEY:
            params["key"] = config.STACKEXCHANGE_API_KEY

        try:
            response = requests.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Stack Exchange fetch error (%s): %s", site, exc)
            continue

        if payload.get("error_id") == 502:
            logger.warning("Stack Exchange quota exceeded for site %s", site)
            return []

        items = payload.get("items", [])
        question_ids = [str(item.get("question_id")) for item in items if item.get("question_id")]
        answers_by_question = _fetch_stackexchange_answers(question_ids, site)

        for item in items:
            question_id = str(item.get("question_id") or "")
            if not question_id:
                continue

            title = clean_text(item.get("title") or "")
            if not title:
                continue

            body = clean_text(_strip_html(item.get("body") or ""))
            cleaned_answers = filter_comments(
                answers_by_question.get(question_id, []),
                min_len=config.MIN_COMMENT_LENGTH,
            )
            scored_comments = score_comments(cleaned_answers, title, sentence_model)

            source_posts.append(
                DiscussionPost(
                    id=question_id,
                    title=title,
                    url=item.get("link")
                    or f"https://{site}.stackexchange.com/questions/{question_id}",
                    source="stackexchange",
                    score=_safe_int(item.get("score")),
                    num_comments=_safe_int(item.get("answer_count")),
                    created_utc=_parse_unix_datetime(item.get("creation_date")),
                    body=body,
                    comments=scored_comments,
                )
            )

    source_posts.sort(key=lambda post: post.score, reverse=True)
    return source_posts[: config.MAX_POSTS_PER_SOURCE]


def fetch_posts(request: SearchRequest, sentence_model: Any) -> list[DiscussionPost]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(fetch_hackernews, request, sentence_model),
            executor.submit(fetch_bluesky, request, sentence_model),
            executor.submit(fetch_stackexchange, request, sentence_model),
        ]

        merged_posts: list[DiscussionPost] = []
        for future in futures:
            try:
                merged_posts.extend(future.result())
            except Exception as exc:
                logger.warning("Source fetch task failed: %s", exc)

    merged_posts.sort(key=lambda post: post.score, reverse=True)
    return merged_posts[: request.limit]
