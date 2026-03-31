# Pulse - Human Intelligence Platform

Search any topic. See what real people are discussing and what news is reporting - simultaneously.

## What it does
- Searches Hacker News, Bluesky, and Stack Exchange for real human discussions
- AI summarizes what humans are saying (not generating opinions itself)
- Fetches 20 news headlines (10 latest + 10 most covered) from Guardian + major RSS feeds
- Shows sentiment, key entities, topics, and source links

## Setup

### 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

### 2. Create your secrets file
cp .env.example .env

Edit .env and fill in:
- GUARDIAN_API_KEY - get free at https://open-platform.theguardian.com/access/
- STACKEXCHANGE_API_KEY - get free at https://stackapps.com/apps/oauth/register
  (optional but raises quota from 300 to 10,000 requests/day)

Neither Hacker News nor Bluesky require any key.

### 3. Run the backend
uvicorn backend.main:app --reload --port 8000

### 4. Open the frontend
Open frontend/index.html in your browser.

## Running tests
pytest tests/ -v