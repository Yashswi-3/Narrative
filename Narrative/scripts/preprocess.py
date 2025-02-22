import re
import json
import spacy
from collections import Counter

# Load NLP model for topic detection
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[WARNING] Spacy model not found. Using basic regex-based keyword extraction.")
    nlp = None

# Define a helper function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s\.\,]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

# Extract key topics from text
def extract_topics(text, top_n=5):
    if nlp:
        doc = nlp(text)
        words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return [word[0] for word in Counter(words).most_common(top_n)]
    return []  # Fallback if Spacy is unavailable

# Function to process and extract key content from JSON data
def preprocess_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    processed_text = []
    for entry in data:
        key_topics = extract_topics(entry["title"])  # Extract key topics from the title
        text_chunk = f"Title: {clean_text(entry['title'])}. "
        insightful_comments = []
        for comment in entry["comments"]:
            comment_lower = comment.lower().strip()
            if comment_lower in ["[deleted]", "[removed]", "thanks", "lol", "nice"]:
                continue  # Skip irrelevant comments
            insightful_comments.append(clean_text(comment))
        text_chunk += " ".join(insightful_comments[:5])  # Take top 5 meaningful comments
        processed_text.append((text_chunk, key_topics))
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_text, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Processed data saved to {output_file}")

# Main execution
if __name__ == "__main__":
    input_path = "C:\\Users\\yashswi shukla\\Desktop\\Project\\Narrative\\Narrative\\data\\raw_reddit_data\\reddit_results.json"
    output_path = "C:\\Users\\yashswi shukla\\Desktop\\Project\\Narrative\\Narrative\\data\\processed_text\\processed_reddit_data.json"
    preprocess_data(input_path, output_path)
