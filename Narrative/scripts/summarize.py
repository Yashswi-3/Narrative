import json
import os
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

def summarize_key_points(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Ensure we extract the correct text content
    combined_text = " ".join(chunk for item in data if "processed_text" in item for chunk in item["processed_text"])[:4000]
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text_chunks = chunk_text(combined_text)
    summary_result = summarize_text(text_chunks, summarizer)
    
    final_summary = " ".join(summary_result)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save summary as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": final_summary}, f, indent=4, ensure_ascii=False)

    print(f"✅ Summary saved to {output_file}")

if __name__ == "__main__":
    input_path = r"C:\Users\yashswi shukla\Desktop\Project\Narrative\Narrative\data\processed_text\processed_reddit_data.json"
    output_path = r"C:\Users\yashswi shukla\Desktop\Project\Narrative\Narrative\data\summaries\reddit_summary.json"
    summarize_key_points(input_path, output_path)
