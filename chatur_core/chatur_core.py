import io
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import PyPDF2
    PDF_LIBRARY = "pypdf2"
except ImportError:
    try:
        import pypdf
        PDF_LIBRARY = "pypdf"
    except ImportError:
        PDF_LIBRARY = None


class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def extract_pdf_text(self, file_bytes, filename="uploaded.pdf", progress_callback=None) -> List[Dict]:
        documents = []
        if PDF_LIBRARY == "pypdf2":
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                chunks = self._chunk_text(text)
                for j, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'source': f"{filename} - Page {i + 1}, Chunk {j + 1}",
                        'type': 'pdf',
                        'metadata': {'page': i + 1}
                    })
                if progress_callback:
                    progress_callback(i + 1)
        return documents

    def process_text_input(self, text, source_name="manual") -> List[Dict]:
        chunks = self._chunk_text(text)
        return [{
            'content': chunk,
            'source': f"{source_name} - Chunk {i + 1}",
            'type': 'text'
        } for i, chunk in enumerate(chunks)]

    def _chunk_text(self, text):
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks


class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_matrix = None
        self.documents = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def add_documents(self, docs: List[Dict]):
        texts = [doc['content'] for doc in docs]
        embeddings = self.embedding_model.encode(texts)
        if self.embeddings_matrix is None:
            self.embeddings_matrix = embeddings
        else:
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings])
        self.documents.extend(docs)

        all_texts = [doc['content'] for doc in self.documents]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

    def search(self, query, k=3) -> List[Dict]:
        if not self.documents:
            return []
        query_embedding = self.embedding_model.encode([query])
        sims = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        indices = np.argsort(sims)[::-1][:k]
        results = []
        for idx in indices:
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(sims[idx])
            results.append(doc)
        return results


class ChaturAI:
    def __init__(self, groq_api_key: str):
        self.vector_store = VectorStore()
        self.processor = DocumentProcessor()
        self.groq_client = Groq(api_key=groq_api_key)

    def add_text(self, text: str):
        docs = self.processor.process_text_input(text)
        self.vector_store.add_documents(docs)

    def add_pdf(self, file_bytes: bytes, filename="uploaded.pdf"):
        docs = self.processor.extract_pdf_text(file_bytes, filename)
        self.vector_store.add_documents(docs)

    def answer(self, question: str) -> str:
        docs = self.vector_store.search(question, k=3)
        context = "\n".join([f"{i+1}. {doc['content']}" for i, doc in enumerate(docs)])

        base_system_prompt = """
You are Chatur AI — an expert, friendly AI tutor created to help students understand and master topics simply and clearly.

Behavior Rules:
- Always explain things clearly and step by step.
- Break down complex ideas into simple parts.
- Use these tutor-style teaching cues where appropriate:
    - “Let me explain that step by step:”
    - “Here's a simplified version of that concept:”
    - “Think of it like this…”
- Provide clear definitions, examples, and analogies.
- Respond in a warm, student-friendly tone.
- Format key insights using:
   - Definitions
   - Bullet points
   - Real-life analogies
   - Summaries like “Here’s what you'd write in an exam:”
- Do not use asterisk to highlight any point.
"""

        if context:
            user_prompt = f"""
Context from user-provided notes:
    {context}

Question: {question}

Goal: Help the student understand this concept thoroughly and prepare for exams. Explain using basic language, step-by-step flow, and learning-friendly phrasing.
"""
        else:
            user_prompt = f"""
The user hasn't provided any study material. Please answer the following using your general knowledge.

Question: {question}

Goal: Help the student understand this concept thoroughly and prepare for exams. Explain using basic language, step-by-step flow, and learning-friendly phrasing.
"""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": base_system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=720,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"
