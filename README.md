# 🤖 Chatur AI – Your Personalized AI Study Buddy

Chatur AI is an intelligent, student-friendly chatbot that helps you **learn smarter** by answering questions using:
- Your **uploaded PDFs**, **pasted text**, or **web URLs**
- Its own **AI knowledge**
- (Optionally) real-time **web search** *(coming soon)*

Built with **Streamlit**, **custom LangChain-like components**, and **Groq’s blazing fast LLaMA 3 models** – Chatur AI turns any study material into a personal tutor.

**Try this Chatur AI bot on**
- Telegram [Link](https://t.me/chatur_ai_bot)
- Streamli App [Link](https://chatur-ai.streamlit.app)

## 🚀 Features

- 📚 Upload study notes (PDF), paste text, or insert URLs
- 💬 Ask any academic question
- 🔍 Uses:
  - Vector-based semantic search (SentenceTransformers)
  - TF-IDF fallback for keyword-based relevance
  - Groq LLMs for contextual answers
- 🧠 Choose which sources to use:
  - Notes-only
  - General AI knowledge
  - Future: Live web search 🌐
- 📎 Tracks source references with similarity scores
- 🧼 Clear documents or chat history with one click
- 🛠 Custom VectorStore (no external DB needed)



## 🛠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Backend**: [Groq API](https://groq.com/)
- **Semantic Search**: [SentenceTransformers](https://www.sbert.net/)
- **Fallback Search**: TF-IDF (via `sklearn`)
- **File Handling**: PyPDF2 / pypdf + BeautifulSoup for web scraping
- **Deployment**: Localhost / Streamlit Cloud *(optional)*



## 📸 Screenshots
### 🔹 Main Chat Interface
![Chat Interface](https://github.com/AnshPradhan14/chatur_ai/blob/main/Screenshos/chat_interface.png)

### 🔹 Document Upload Sidebar
![Sidebar](https://github.com/AnshPradhan14/chatur_ai/blob/main/Screenshos/upload_sidebar.png)

### 🔹Telegram Bot
![Chat Interface](https://github.com/AnshPradhan14/chatur_ai/blob/main/Screenshos/IMG_20250717_003708.jpg)



---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/AnSHPradhan14/chatur-ai.git
cd chatur-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API Key
Create a .streamlit/secrets.toml file in the root directory:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Run the app
```bash
streamlit run app.py
```

## 🧠 How it works (worksflow)
**1. Document Processing**
- Extracts and chunks text from PDFs / URLs / pasted text
- Stores chunks in a custom Vector Store using sentence embeddings

**2. Search Mechanism**
- For each query, searches most relevant chunks using cosine similarity
- If semantic search fails, fallback to TF-IDF

**3. Response Generation**

- Relevant chunks + question are sent to Groq’s LLM (LLaMA 3)
- Generates a helpful answer with sources

## 🛣️ Roadmap
- ☑️Multi-source document upload (PDF, text, URL)
- ☑️Semantic + TF-IDF fallback search
- ☑️Groq LLM integration
- Live Web Search with SerpAPI or Bing
- Clickable source citations
- Downloadable chat history
- Streamlit Cloud / HuggingFace Spaces deployment


## 🙏 Acknowledgments
- [Groq](https://groq.com/) for their lightning-fast LLaMA API
- [Streamlit](https://streamlit.io/) for the rapid UI development
- [SBERT](https://www.sbert.net/) for Sentence Embeddings
- All the amazing open-source contributors 🙌


## 📄 License
This project is licensed under the MIT License.
