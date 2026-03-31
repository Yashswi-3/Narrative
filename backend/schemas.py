from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    limit: int = Field(default=25, ge=5, le=100)
    time_filter: str = Field(default="week")


class DiscussionPost(BaseModel):
    """Unified schema for HN + Bluesky + StackExchange posts."""
    id: str
    title: str
    url: str
    source: str
    score: int
    num_comments: int
    created_utc: datetime
    body: str
    comments: list[str]


class ProcessedPost(BaseModel):
    id: str
    title: str
    combined_text: str
    topics: list[str]
    word_count: int
    relevance_score: float
    source: str


class NewsHeadline(BaseModel):
    publisher: str
    headline: str
    url: str
    published_at: Optional[datetime] = None
    time_ago: str
    category: str


class SummaryResult(BaseModel):
    query: str
    sources_used: list[str]
    post_count: int
    posts_after_dedup: int
    topics: list[str]
    summary: str
    sentiment: str
    sentiment_score: float
    key_entities: dict
    top_posts: list[dict]
    news_headlines: list[dict]
    confidence: str
    generated_at: datetime
    timing: dict
