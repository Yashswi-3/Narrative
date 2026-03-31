from datetime import datetime, timezone

import numpy as np
from fastapi.testclient import TestClient

from backend.pipeline.preprocessor import deduplicate_posts, extract_topics, preprocess_posts
from backend.pipeline.summarizer import set_bart_tokenizer, summarize
from backend.schemas import DiscussionPost, ProcessedPost
from backend.utils.text_utils import chunk_by_tokens, clean_text


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(tokens)


class DummySentenceModel:
    def encode(self, texts, convert_to_numpy=True):
        del convert_to_numpy
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                np.array(
                    [
                        len(lower),
                        sum(ch in "aeiou" for ch in lower),
                        sum(ch.isalpha() for ch in lower),
                    ],
                    dtype=float,
                )
            )
        return np.vstack(vectors)


class MockToken:
    def __init__(self, lemma: str):
        self.lemma_ = lemma
        self.is_alpha = lemma.isalpha()
        self.is_stop = False


class MockNLPForTopics:
    def __call__(self, _text):
        return [MockToken("running"), MockToken("technology"), MockToken("ai")]


class DummyEntity:
    def __init__(self, text: str, label: str):
        self.text = text
        self.label_ = label


class DummyDoc:
    def __init__(self, ents=None):
        self.ents = ents or []


class MockNLPForSummary:
    def __call__(self, _text):
        return DummyDoc(
            [
                DummyEntity("Alice", "PERSON"),
                DummyEntity("OpenAI", "ORG"),
                DummyEntity("India", "GPE"),
            ]
        )


def test_clean_text():
    raw = "Check this out https://reddit.com **bold** &amp; stuff"
    cleaned = clean_text(raw)
    assert "http" not in cleaned.lower()
    assert "**" not in cleaned
    assert "&amp;" not in cleaned


def test_extract_topics_full_words():
    topics = extract_topics("irrelevant", MockNLPForTopics())
    assert "running" in topics
    assert "technology" in topics
    assert "r" not in topics
    assert "t" not in topics
    assert "a" not in topics


def test_chunk_by_tokens_respects_limit():
    tokenizer = DummyTokenizer()
    text = " ".join([f"sentence{i}." for i in range(1, 401)])
    chunks = chunk_by_tokens(text, tokenizer, max_tokens=50)
    assert chunks
    assert all(len(tokenizer.encode(chunk, add_special_tokens=False)) <= 50 for chunk in chunks)


def test_preprocess_contract():
    posts = [
        DiscussionPost(
            id="1",
            title="India technology sector growth",
            url="https://reddit.com/r/india/post1",
            source="hackernews",
            score=100,
            num_comments=20,
            created_utc=datetime.now(timezone.utc),
            body="The technology market in India has expanded rapidly with startup growth and policy focus.",
            comments=[
                "Interesting growth trend across cloud and AI hiring.",
                "Policy changes are helping local companies scale operations.",
            ],
        )
    ]

    processed = preprocess_posts(posts, "india tech", MockNLPForTopics(), DummySentenceModel())

    assert isinstance(processed, list)
    assert processed
    assert all(isinstance(item, ProcessedPost) for item in processed)
    assert all(isinstance(item.combined_text, str) and item.combined_text.strip() for item in processed)
    assert all(isinstance(item.topics, list) for item in processed)
    assert all(all(isinstance(topic, str) for topic in item.topics) for item in processed)
    assert all(all(len(topic) > 1 for topic in item.topics) for item in processed)


def test_summarize_never_empty():
    set_bart_tokenizer(DummyTokenizer())
    posts = [
        ProcessedPost(
            id="1",
            title="Tiny post",
            combined_text="five words are right here",
            topics=["words", "tiny"],
            word_count=5,
            relevance_score=0.7,
            source="hackernews",
        )
    ]

    def bart_pipeline(_text, **kwargs):
        del kwargs
        return [{"summary_text": "Stub generated summary output."}]

    def sentiment_pipeline(_text, **kwargs):
        del kwargs
        return [{"label": "POSITIVE", "score": 0.95}]

    result = summarize(posts, "test", bart_pipeline, sentiment_pipeline, MockNLPForSummary())

    assert result.summary.strip()
    assert len(result.summary.strip()) > 10


def test_deduplication():
    posts = [
        DiscussionPost(
            id="1",
            title="Same title for dedup",
            url="https://reddit.com/r/test/1",
            source="bluesky",
            score=10,
            num_comments=2,
            created_utc=datetime.now(timezone.utc),
            body="Body one",
            comments=["Comment one"],
        ),
        DiscussionPost(
            id="2",
            title="Same title for dedup",
            url="https://reddit.com/r/test/2",
            source="stackexchange",
            score=20,
            num_comments=3,
            created_utc=datetime.now(timezone.utc),
            body="Body two",
            comments=["Comment two"],
        ),
    ]

    deduped = deduplicate_posts(posts, DummySentenceModel())
    assert len(deduped) == 1


def test_api_health(monkeypatch):
    import backend.main as main_module

    def fake_pipeline(task, model=None):
        del model
        if task == "summarization":
            return lambda _text, **kwargs: [{"summary_text": "summary"}]
        return lambda _text, **kwargs: [{"label": "POSITIVE", "score": 0.9}]

    monkeypatch.setattr(main_module, "pipeline", fake_pipeline)
    monkeypatch.setattr(main_module.AutoTokenizer, "from_pretrained", lambda _model: DummyTokenizer())
    monkeypatch.setattr(main_module, "SentenceTransformer", lambda _model: DummySentenceModel())
    monkeypatch.setattr(main_module.spacy, "load", lambda _model: MockNLPForSummary())

    with TestClient(main_module.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
