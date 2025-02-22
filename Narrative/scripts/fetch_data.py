import praw
import requests
import json
import ssl
from bs4 import BeautifulSoup
import re
import spacy

# Check if Spacy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[WARNING] Spacy model not found. Using basic regex-based keyword extraction.")
    nlp = None

# Reddit API Credentials
reddit = praw.Reddit(
    client_id="As44RYqjOK4m9FJ53EwC-g",
    client_secret="9jscdaZAODphthPzuVumkAgvH5EYJw",
    user_agent="Narative/0.1 by your_reddit_username"
)

# Fix SSL Issue
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Function to optimize user query
def optimize_query(query):
    if nlp:
        doc = nlp(query)
        keywords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(keywords)
    else:
        query = query.lower()
        query = re.sub(r"[^a-zA-Z0-9 ]", "", query)
        keywords = query.split()
        return " ".join(keywords)

# Function to find the best subreddit
def find_best_subreddit(query):
    search_url = f"https://www.reddit.com/search/?q={query}&type=sr"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        subreddit_links = soup.find_all('a', href=True)
        for link in subreddit_links:
            href = link['href']
            if '/r/' in href and not href.startswith("https://www.reddit.com/user/"):
                return href.split('/r/')[1].split('/')[0]
    return "all"

# Function to fetch Reddit posts
def fetch_reddit_posts(query, output_file=r"C:\Users\yashswi shukla\Desktop\Project\Narrative\Narrative\data\raw_reddit_data\reddit_results.json"):
    optimized_query = optimize_query(query)
    best_subreddit = find_best_subreddit(optimized_query)
    print(f"\n🔍 Searching in subreddit: r/{best_subreddit} for '{optimized_query}'\n")
    
    try:
        subreddit = reddit.subreddit(best_subreddit)
        top_posts = subreddit.search(optimized_query, limit=50)
        results = []
        
        for post in top_posts:
            post.comments.replace_more(limit=0)
            comments = [comment.body for comment in post.comments[:50]]
            results.append({
                "title": post.title,
                "url": post.url,
                "comments": comments
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Data saved to {output_file}")
        return results
    except Exception as e:
        print(f"⚠️ Error fetching posts: {e}")
        return []

# Main execution
if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    fetch_reddit_posts(user_query)
