import calendar
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import feedparser
import requests

from backend.config import GUARDIAN_API_KEY, NEWS_FETCH_TIMEOUT, RSS_FEEDS
from backend.schemas import NewsHeadline

logger = logging.getLogger(__name__)


def fetch_guardian(query: str, order_by: str) -> list[dict]:
    """
    order_by: 'newest' or 'relevance'
    Returns list of raw article dicts.
    """
    if not GUARDIAN_API_KEY:
        return []

    params = {
        "q": query,
        "order-by": order_by,
        "page-size": 10,
        "show-fields": "headline,publication,trailText",
        "api-key": GUARDIAN_API_KEY,
    }
    try:
        response = requests.get(
            "https://content.guardianapis.com/search",
            params=params,
            timeout=NEWS_FETCH_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("response", {}).get("results", [])
        return results
    except Exception as exc:
        logger.warning("Guardian API error: %s", exc)
        return []


def fetch_rss_headlines(query: str) -> list[dict]:
    """
    Parse all RSS feeds, filter entries that mention the query keywords,
    return list of normalized dicts.
    """
    keywords = [word.lower() for word in query.split() if len(word) > 2]
    results: list[dict] = []

    for feed_config in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_config["url"])
            for entry in feed.entries[:30]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                combined = (title + " " + summary).lower()

                if keywords and not any(keyword in combined for keyword in keywords):
                    continue

                relevance = sum(combined.count(keyword) for keyword in keywords) if keywords else 1
                results.append(
                    {
                        "publisher": feed_config["name"],
                        "headline": title,
                        "url": entry.get("link", ""),
                        "published_at": entry.get("published_parsed"),
                        "source": "rss",
                        "relevance": relevance,
                    }
                )
        except Exception as exc:
            logger.warning("RSS parsing error for %s: %s", feed_config["name"], exc)
            continue

    return results


def time_ago(dt: datetime | None) -> str:
    if dt is None:
        return "recently"
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = now - dt
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return f"{seconds} sec ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} hours ago"
    return f"{seconds // 86400} days ago"


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def struct_time_to_datetime(value) -> datetime | None:
    if value is None:
        return None
    try:
        ts = calendar.timegm(value)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def _dedupe_by_url(items: list[NewsHeadline]) -> list[NewsHeadline]:
    seen: set[str] = set()
    deduped: list[NewsHeadline] = []
    for item in items:
        key = item.url or f"{item.publisher}:{item.headline}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def fetch_news(query: str) -> list[NewsHeadline]:
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            guardian_latest_future = executor.submit(fetch_guardian, query, "newest")
            guardian_popular_future = executor.submit(fetch_guardian, query, "relevance")
            rss_future = executor.submit(fetch_rss_headlines, query)

        try:
            guardian_latest = guardian_latest_future.result()
        except Exception as exc:
            logger.warning("Guardian latest task failed: %s", exc)
            guardian_latest = []

        try:
            guardian_popular = guardian_popular_future.result()
        except Exception as exc:
            logger.warning("Guardian popular task failed: %s", exc)
            guardian_popular = []

        try:
            rss_items = rss_future.result()
        except Exception as exc:
            logger.warning("RSS task failed: %s", exc)
            rss_items = []
    except Exception as exc:
        logger.warning("News aggregation error: %s", exc)
        return []

    latest_headlines: list[NewsHeadline] = []
    for item in guardian_latest[:5]:
        published = parse_iso(item.get("webPublicationDate"))
        latest_headlines.append(
            NewsHeadline(
                publisher="The Guardian",
                headline=item.get("webTitle", ""),
                url=item.get("webUrl", ""),
                published_at=published,
                time_ago=time_ago(published),
                category="latest",
            )
        )

    rss_sorted_latest = sorted(
        rss_items,
        key=lambda item: struct_time_to_datetime(item.get("published_at"))
        or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )
    for item in rss_sorted_latest[:5]:
        published = struct_time_to_datetime(item.get("published_at"))
        latest_headlines.append(
            NewsHeadline(
                publisher=item.get("publisher", ""),
                headline=item.get("headline", ""),
                url=item.get("url", ""),
                published_at=published,
                time_ago=time_ago(published),
                category="latest",
            )
        )

    popular_headlines: list[NewsHeadline] = []
    for item in guardian_popular[:5]:
        published = parse_iso(item.get("webPublicationDate"))
        popular_headlines.append(
            NewsHeadline(
                publisher="The Guardian",
                headline=item.get("webTitle", ""),
                url=item.get("webUrl", ""),
                published_at=published,
                time_ago=time_ago(published),
                category="popular",
            )
        )

    rss_sorted_popular = sorted(
        rss_items,
        key=lambda item: int(item.get("relevance", 0)),
        reverse=True,
    )
    for item in rss_sorted_popular[:5]:
        published = struct_time_to_datetime(item.get("published_at"))
        popular_headlines.append(
            NewsHeadline(
                publisher=item.get("publisher", ""),
                headline=item.get("headline", ""),
                url=item.get("url", ""),
                published_at=published,
                time_ago=time_ago(published),
                category="popular",
            )
        )

    deduped = _dedupe_by_url(latest_headlines + popular_headlines)
    latest_out = [item for item in deduped if item.category == "latest"][:10]
    popular_out = [item for item in deduped if item.category == "popular"][:10]

    if len(latest_out) < 10:
        for item in latest_headlines:
            if len(latest_out) >= 10:
                break
            if item not in latest_out:
                latest_out.append(item)

    if len(popular_out) < 10:
        for item in popular_headlines:
            if len(popular_out) >= 10:
                break
            if item not in popular_out:
                popular_out.append(item)

    return (latest_out + popular_out)[:20]