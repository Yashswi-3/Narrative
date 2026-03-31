import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import spacy
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, pipeline

from backend import config
from backend.pipeline.fetcher import fetch_posts
from backend.pipeline.news_fetcher import fetch_news
from backend.pipeline.preprocessor import preprocess_posts
from backend.pipeline.summarizer import set_bart_tokenizer, summarize
from backend.schemas import SearchRequest, SummaryResult
from backend.utils.text_utils import chunk_by_tokens, extractive_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models_loaded = False

    app.state.bart_pipeline = pipeline("summarization", model=config.BART_MODEL)
    app.state.bart_tokenizer = AutoTokenizer.from_pretrained(config.BART_MODEL)
    app.state.sentiment_pipeline = pipeline("sentiment-analysis")
    app.state.sentence_model = SentenceTransformer(config.SENTENCE_MODEL)
    app.state.nlp = spacy.load(config.SPACY_MODEL)
    set_bart_tokenizer(app.state.bart_tokenizer)

    app.state.models_loaded = True
    logger.info("Models loaded")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    if isinstance(exc.detail, dict) and exc.detail.get("error") is True:
        payload = exc.detail
    else:
        payload = {
            "error": True,
            "message": "Request failed.",
            "detail": str(exc.detail),
        }
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(Exception)
async def generic_exception_handler(_, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Unexpected server error.",
            "detail": str(exc),
        },
    )


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": bool(getattr(app.state, "models_loaded", False)),
        "models": {
            "bart": config.BART_MODEL,
            "sentence": config.SENTENCE_MODEL,
            "spacy": config.SPACY_MODEL,
        },
    }


@app.get("/api/history")
async def history() -> list[dict]:
    if not config.OUTPUT_DIR.exists():
        return []

    history_items: list[SummaryResult] = []
    for file_path in config.OUTPUT_DIR.glob("*.json"):
        try:
            raw_data = json.loads(file_path.read_text(encoding="utf-8"))
            history_items.append(SummaryResult.model_validate(raw_data))
        except Exception:
            continue

    history_items.sort(key=lambda item: item.generated_at, reverse=True)
    return [item.model_dump(mode="json") for item in history_items]


@app.delete("/api/history")
async def clear_history() -> dict:
    deleted = 0
    for file_path in config.OUTPUT_DIR.glob("*.json"):
        file_path.unlink(missing_ok=True)
        deleted += 1
    return {"deleted": deleted}


def _safe_filename(query: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", query.strip().lower()).strip("_")
    return cleaned or "query"


@app.get("/api/search/stream")
async def search_stream(
    query: str = Query(..., min_length=2, max_length=200),
    limit: int = Query(default=25, ge=5, le=100),
    time_filter: str = Query(default="week", pattern=r"^(day|week|month|year|all)$"),
):
    request = SearchRequest(
        query=query,
        limit=limit,
        time_filter=time_filter,
    )

    async def event_generator():
        loop = asyncio.get_running_loop()
        timings = {
            "fetch_ms": 0,
            "preprocess_ms": 0,
            "summarize_ms": 0,
            "news_ms": 0,
            "total_ms": 0,
        }
        total_start = time.perf_counter()

        try:
            yield {
                "data": json.dumps(
                    {
                        "stage": 1,
                        "message": "Expanding query and discovering sources...",
                        "done": False,
                    }
                )
            }

            resolved_request = request.model_copy(update={"query": request.query.strip()})

            yield {
                "data": json.dumps(
                    {
                        "stage": 1,
                        "message": "Fetching from Hacker News, Bluesky, Stack Exchange...",
                        "done": False,
                    }
                )
            }

            fetch_start = time.perf_counter()
            fetch_posts_task = loop.run_in_executor(
                None,
                fetch_posts,
                resolved_request,
                app.state.sentence_model,
            )
            fetch_news_task = loop.run_in_executor(None, fetch_news, request.query)
            fetched_posts, news_headlines = await asyncio.gather(fetch_posts_task, fetch_news_task)
            elapsed_fetch = int((time.perf_counter() - fetch_start) * 1000)
            timings["fetch_ms"] = elapsed_fetch
            timings["news_ms"] = elapsed_fetch

            if not fetched_posts:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": True,
                        "message": "No relevant discussions found.",
                        "detail": "Try a broader query or different time range.",
                    },
                )

            yield {
                "data": json.dumps(
                    {
                        "stage": 2,
                        "message": f"Fetched {len(fetched_posts)} discussions. Deduplicating and ranking...",
                        "done": False,
                    }
                )
            }

            preprocess_start = time.perf_counter()
            processed_posts = await loop.run_in_executor(
                None,
                preprocess_posts,
                fetched_posts,
                request.query,
                app.state.nlp,
                app.state.sentence_model,
            )
            timings["preprocess_ms"] = int((time.perf_counter() - preprocess_start) * 1000)

            if not processed_posts:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": True,
                        "message": "No processable discussion content found.",
                        "detail": "Fetched posts did not meet preprocessing thresholds.",
                    },
                )

            yield {
                "data": json.dumps(
                    {
                        "stage": 2,
                        "message": f"Preprocessing {len(processed_posts)} unique posts...",
                        "done": False,
                    }
                )
            }

            yield {
                "data": json.dumps(
                    {
                        "stage": 3,
                        "message": "Running extractive filter...",
                        "done": False,
                    }
                )
            }

            full_text = " ".join(post.combined_text for post in processed_posts)
            filtered_preview = extractive_filter(full_text, request.query, keep_ratio=0.6)
            preview_chunks = chunk_by_tokens(
                filtered_preview,
                app.state.bart_tokenizer,
                config.CHUNK_MAX_TOKENS,
            )
            chunk_count = max(1, len(preview_chunks))

            yield {
                "data": json.dumps(
                    {
                        "stage": 3,
                        "message": f"Generating summary ({chunk_count} chunks)...",
                        "done": False,
                    }
                )
            }

            summarize_start = time.perf_counter()
            summary_result = await loop.run_in_executor(
                None,
                summarize,
                processed_posts,
                request.query,
                app.state.bart_pipeline,
                app.state.sentiment_pipeline,
                app.state.nlp,
            )
            timings["summarize_ms"] = int((time.perf_counter() - summarize_start) * 1000)

            yield {
                "data": json.dumps(
                    {
                        "stage": 4,
                        "message": "Fetching news headlines...",
                        "done": False,
                    }
                )
            }

            yield {
                "data": json.dumps(
                    {
                        "stage": 4,
                        "message": "Analyzing sentiment and entities...",
                        "done": False,
                    }
                )
            }

            timings["total_ms"] = int((time.perf_counter() - total_start) * 1000)
            sources_used = sorted({post.source for post in fetched_posts if post.source})
            top_posts = [
                {
                    "title": post.title,
                    "url": post.url,
                    "score": post.score,
                    "source": post.source,
                    "num_comments": post.num_comments,
                }
                for post in sorted(fetched_posts, key=lambda item: item.score, reverse=True)[:10]
            ]

            finalized_result = summary_result.model_copy(
                update={
                    "query": request.query,
                    "sources_used": sources_used,
                    "post_count": len(fetched_posts),
                    "posts_after_dedup": len(processed_posts),
                    "top_posts": top_posts,
                    "news_headlines": [headline.model_dump(mode="json") for headline in news_headlines],
                    "generated_at": datetime.now(timezone.utc),
                    "timing": timings,
                }
            )

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = config.OUTPUT_DIR / f"{_safe_filename(request.query)}_{timestamp}.json"
            output_path.write_text(
                json.dumps(finalized_result.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            yield {
                "data": json.dumps(
                    {
                        "stage": 4,
                        "message": "Complete.",
                        "done": True,
                        "result": finalized_result.model_dump(mode="json"),
                    }
                )
            }

        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            yield {
                "data": json.dumps(
                    {
                        "error": True,
                        "message": detail.get("message", "Search failed."),
                        "detail": detail.get("detail", str(exc.detail)),
                    }
                )
            }
        except Exception as exc:
            yield {
                "data": json.dumps(
                    {
                        "error": True,
                        "message": "Unexpected processing error.",
                        "detail": str(exc),
                    }
                )
            }

    return EventSourceResponse(event_generator())