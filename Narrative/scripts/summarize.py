import json
from transformers import pipeline

def chunk_text(text, max_chunk_size=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1  # +1 for space
        if current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def summarize_text(text_chunks, summarizer, max_length=200, min_length=50):
    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]["summary_text"])
    return summaries

def summarize_key_points(filename):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    combined_text = " ".join(data)[:4000]  # Truncate to approx. 1024 tokens
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text_chunks = chunk_text(combined_text)
    summary_result = summarize_text(text_chunks, summarizer)
    
    final_summary = " ".join(summary_result)
    return final_summary

if __name__ == "__main__":
    summary_result = summarize_key_points("C:\\Users\\yashswi shukla\\Desktop\\Project\\Narrative\\Narrative\\data\\processed_text\\processed_reddit_data.json")
    with open("C:\\Users\\yashswi shukla\\Desktop\\Project\\Narrative\\Narrative\\data\\summarized\\final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=4, ensure_ascii=False)
    print(r"C:\Users\yashswi shukla\Desktop\Project\Narrative\Narrative\data\summaries\final_summary.json")