# Narrative: Reddit Data Processing Pipeline

## 📌 Overview
**Narrative** is a pipeline for fetching, preprocessing, and summarizing Reddit discussions based on a user-defined query. The project extracts relevant Reddit posts, processes them to filter meaningful content, and generates a concise summary using NLP techniques.

## 🚀 Features
- Fetches Reddit posts based on user input
- Cleans and preprocesses text for better summarization
- Extracts key topics from discussions
- Summarizes large amounts of text using state-of-the-art NLP models
- Stores structured results for further analysis

## 📂 Project Structure
```
Narrative/
│── data/
│   ├── raw_reddit_data/            # Stores raw JSON data from Reddit
│   ├── processed_text/             # Stores preprocessed Reddit data
│   ├── summaries/                  # Stores final summaries
│── fetch_data.py                   # Script to fetch Reddit posts
│── preprocess.py                    # Script for text preprocessing
│── summarize.py                     # Script to generate summaries
│── pipeline.py                      # Main script orchestrating the pipeline
│── README.md                        # Project documentation
```

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Yashswi-3/Narrative.git
cd Narrative
```

### 2️⃣ Install Dependencies
Ensure you have Python installed (>=3.7). Then, install the required libraries:
```sh
pip install praw transformers spacy
python -m spacy download en_core_web_sm
```

### 3️⃣ Set Up Reddit API Credentials
To fetch data from Reddit, set up a `praw.ini` file or define environment variables with your **client ID, client secret, and user agent**.

### 4️⃣ Run the Pipeline
```sh
python pipeline.py
```

## 📝 Usage
1. Run `pipeline.py` and enter a search query.
2. The script will fetch relevant posts, preprocess the text, and generate a summary.
3. The summarized results are saved in `data/summaries/reddit_summary.json`.
4. The summary is also printed in the terminal.

## 🏗️ How It Works
### 1️⃣ Fetching Data
- Uses **PRAW (Python Reddit API Wrapper)** to retrieve posts and comments.
- Saves raw data in `data/raw_reddit_data/reddit_results.json`.

### 2️⃣ Preprocessing Data
- Cleans text (removes URLs, special characters, etc.).
- Extracts key topics using **spaCy**.
- Filters out irrelevant comments.
- Saves processed data in `data/processed_text/processed_reddit_data.json`.

### 3️⃣ Summarization
- Splits long text into chunks.
- Uses **Facebook's BART model** for summarization.
- Generates and stores the final summary.

## 📊 Example Output
```
🔍 Final Summary:
Key Topics: Mental Health, Therapy, Stress Management.
Social media should be used to connect people in need. Majority of Indians...
```

## 🔥 Future Improvements
- Improve topic extraction using **BERT-based models**.
- Add **sentiment analysis** for better context.
- Enable multi-document summarization.
- Create a **web-based interface** for easier access.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository, submit issues, or create pull requests.

---
🚀 **Built with Python, NLP, and a passion for extracting insights from discussions!**
