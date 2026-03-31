import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
OUTPUT_DIR.mkdir(exist_ok=True)

# API Keys (all optional - sources degrade gracefully if missing)
STACKEXCHANGE_API_KEY = os.getenv("STACKEXCHANGE_API_KEY", "")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "")

BART_MODEL = "sshleifer/distilbart-cnn-6-6"
SENTENCE_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Pipeline tuning
MAX_POSTS_PER_SOURCE = 15
MAX_COMMENTS_TO_KEEP = 5
MIN_COMMENT_LENGTH = 30
SIMILARITY_DEDUP_THRESHOLD = 0.82
QUERY_EXPANSION_MAX_TERMS = 3
CHUNK_MAX_TOKENS = 900

# News config
MAX_NEWS_HEADLINES = 20
NEWS_FETCH_TIMEOUT = 10

# RSS feeds (always free, no key needed)
RSS_FEEDS = [
    {"name": "BBC News", "url": "http://feeds.bbci.co.uk/news/rss.xml"},
    {"name": "Reuters", "url": "https://feeds.reuters.com/reuters/topNews"},
    {"name": "AP News", "url": "https://feeds.apnews.com/rss/apf-topnews"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
]
