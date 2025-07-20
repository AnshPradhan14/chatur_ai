import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from groq import Groq
import re
import hashlib
from typing import List, Dict, Tuple, Optional
import io
import time

#pdf library handling
try:
    import PyPDF2 # type: ignore
    PDF_LIBRARY = "pypdf2"
except ImportError:
    try:
        import pypdf
        PDF_LIBRARY = "pypdf"
    except ImportError:
        PDF_LIBRARY = None

if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
#ensure groq_initialized is set, important for the new flow
if 'groq_initialized' not in st.session_state:
    st.session_state.groq_initialized = False


class DocumentProcessor:
    """Handles PDF and URL document processing"""

    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def extract_pdf_text(self, pdf_file) -> List[Dict]:
        """Extract text from uploaded PDF file with multiple fallback methods"""
        if PDF_LIBRARY is None:
            st.error("No PDF library available. Please install: pip install PyPDF2")
            st.info("Alternative: Copy and paste your text directly, or use URLs instead of PDFs")
            return []

        try:
            #reset file pointer to beginning
            pdf_file.seek(0)

            documents = []

            if PDF_LIBRARY == "pypdf2":
                #method 1: pypdf2
                try:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)

                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()

                        if text.strip(): #only add non-empty pages
                            chunks = self._chunk_text(text)
                            for i, chunk in enumerate(chunks):
                                documents.append({
                                    'content': chunk,
                                    'source': f"{pdf_file.name} - Page {page_num + 1}, Chunk {i + 1}",
                                    'type': 'pdf',
                                    'metadata': {
                                        'page': page_num + 1,
                                        'chunk': i + 1,
                                        'filename': pdf_file.name
                                    }
                                })
                except Exception as e:
                    st.error(f"PyPDF2 extraction failed: {str(e)}")
                    return self._extract_pdf_fallback(pdf_file)

            elif PDF_LIBRARY == "pypdf":
                #method 2: pypdf (newer version)
                try:
                    import pypdf
                    pdf_reader = pypdf.PdfReader(pdf_file)

                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()

                        if text.strip(): #only add non-empty pages
                            chunks = self._chunk_text(text)
                            for i, chunk in enumerate(chunks):
                                documents.append({
                                    'content': chunk,
                                    'source': f"{pdf_file.name} - Page {page_num + 1}, Chunk {i + 1}",
                                    'type': 'pdf',
                                    'metadata': {
                                        'page': page_num + 1,
                                        'chunk': i + 1,
                                        'filename': pdf_file.name
                                    }
                                })
                except Exception as e:
                    st.error(f"pypdf extraction failed: {str(e)}")
                    return self._extract_pdf_fallback(pdf_file)

            return documents

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return self._extract_pdf_fallback(pdf_file)

    def _extract_pdf_fallback(self, pdf_file) -> List[Dict]:
        """Fallback method for PDF extraction using simple text extraction"""
        try:
            #reset file pointer
            pdf_file.seek(0)

            #try to read as bytes and extract any readable text
            content = pdf_file.read()

            #simple text extraction (this is very basic)
            text_content = str(content, errors='ignore')

            #clean up the extracted text
            text_content = re.sub(r'[^\x00-\x7F]+', ' ', text_content) #remove non-ascii
            text_content = re.sub(r'\s+', ' ', text_content) #normalize whitespace

            if len(text_content.strip()) > 100: #only if we got meaningful text
                chunks = self._chunk_text(text_content)
                documents = []

                for i, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'source': f"{pdf_file.name} - Chunk {i + 1} (fallback extraction)",
                        'type': 'pdf',
                        'metadata': {
                            'chunk': i + 1,
                            'filename': pdf_file.name,
                            'extraction_method': 'fallback'
                        }
                    })
                return documents
            else:
                st.warning(f"Could not extract readable text from {pdf_file.name}. The PDF might be image-based or encrypted.")
                return []

        except Exception as e:
            st.error(f"Fallback PDF extraction also failed: {str(e)}")
            return []

    def extract_url_text(self, url: str) -> List[Dict]:
        """Extract text from URL with improved content targeting"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15) #increased timeout
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            #remove script, style, header, footer, nav elements to reduce noise
            for unwanted_tag in soup(["script", "style", "header", "footer", "nav", "aside", ".ads"]):
                unwanted_tag.decompose()

            #try to find main article content using common tags and classes
            article_content_tags = soup.find_all(['article', 'main', 'div'], class_=[
                'article-content', 'article-body', 'entry-content', 'post-content', 'main-content',
                re.compile(r'body'), re.compile(r'content') #more generic pattern for div classes
            ])

            chunks_text = ""
            if article_content_tags:
                #prioritize larger chunks of text
                article_content_tags.sort(key=lambda tag: len(tag.get_text(separator=' ', strip=True)), reverse=True)
                
                for tag in article_content_tags:
                    text_from_tag = tag.get_text(separator=' ', strip=True)
                    if len(text_from_tag) > 100: #only consider substantial text
                        chunks_text += text_from_tag + "\n\n"
                    if len(chunks_text) > 2000: #stop after a reasonable amount to avoid too much noise
                        break
                
            if not chunks_text.strip():
                #fallback: if specific article content is not found, get text from body
                body_tag = soup.find('body')
                if body_tag:
                    chunks_text = body_tag.get_text(separator=' ', strip=True)
                else:
                    chunks_text = soup.get_text(separator=' ', strip=True) #last resort, get all text

            #clean up text
            lines = (line.strip() for line in chunks_text.splitlines())
            chunks_text = '\n'.join(chunk for chunk in lines if chunk)
            chunks_text = re.sub(r'\s+', ' ', chunks_text).strip() #normalize whitespace and strip again

            if not chunks_text.strip() or len(chunks_text) < 50: #minimum length for meaningful content
                st.warning(f"Could not extract meaningful text from URL: {url}. It might be heavily JavaScript-rendered or have an unusual structure.")
                return []

            chunks = self._chunk_text(chunks_text)
            documents = []

            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'source': f"{url} - Chunk {i + 1}",
                    'type': 'url',
                    'metadata': {
                        'url': url,
                        'chunk': i + 1
                    }
                })

            return documents

        except requests.exceptions.RequestException as e:
            st.error(f"Network error or invalid URL {url}: {str(e)}. Please check the URL and your internet connection.")
            return []
        except Exception as e:
            st.error(f"Error processing URL {url}: {str(e)}")
            return []

    def process_text_input(self, text: str, source_name: str = "Manual Input") -> List[Dict]:
        """Process manually entered text"""
        if not text.strip():
            return []

        chunks = self._chunk_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            documents.append({
                'content': chunk,
                'source': f"{source_name} - Chunk {i + 1}",
                'type': 'text',
                'metadata': {
                    'chunk': i + 1,
                    'source_name': source_name
                }
            })

        return documents

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            #try to break at sentence boundary
            chunk = text[start:end]
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')

            break_point = max(last_period, last_newline)

            if break_point > start + self.chunk_size // 2:
                end = start + break_point + 1

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return [chunk.strip() for chunk in chunks if chunk.strip()]

class VectorStore:
    """Handles vector embeddings and similarity search using scikit-learn"""

    def __init__(self):
        self.embedding_model = None
        self.embeddings_matrix = None
        self.documents = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    @st.cache_resource
    def load_embedding_model(_self):
        """Load sentence transformer model"""
        return SentenceTransformer('all-MiniLM-L6-v2')

    def initialize(self):
        """Initialize the embedding model"""
        if self.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                self.embedding_model = self.load_embedding_model()

    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store"""
        if not documents:
            return

        self.initialize()

        #extract text content for embedding
        texts = [doc['content'] for doc in documents]

        with st.spinner(f"Creating embeddings for {len(texts)} chunks..."):
            #generate embeddings using sentencetransformer
            new_embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

            #store or update embeddings matrix
            if self.embeddings_matrix is None:
                self.embeddings_matrix = new_embeddings
                self.documents = documents
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_embeddings])
                self.documents.extend(documents)

            #also create tf-idf matrix as backup search method
            all_texts = [doc['content'] for doc in self.documents]
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents using cosine similarity"""
        if self.embeddings_matrix is None or not self.documents:
            return []

        self.initialize()

        #method 1: semantic search using embeddings
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

        #get top k results
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1: #minimum similarity threshold
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(similarities[idx])
                doc['search_method'] = 'semantic'
                results.append(doc)

        #if semantic search doesn't yield good results, try tf-idf
        if len(results) < k // 2 and self.tfidf_vectorizer is not None:
            try:
                query_tfidf = self.tfidf_vectorizer.transform([query])
                tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

                #get additional results from tf-idf
                tfidf_top_indices = np.argsort(tfidf_similarities)[::-1][:k]

                for idx in tfidf_top_indices:
                    if tfidf_similarities[idx] > 0.05 and idx not in [r['metadata'].get('original_idx', -1) for r in results]:
                        doc = self.documents[idx].copy()
                        doc['similarity_score'] = float(tfidf_similarities[idx])
                        doc['search_method'] = 'tfidf'
                        doc['metadata']['original_idx'] = idx
                        results.append(doc)

                        if len(results) >= k:
                            break
            except:
                pass #tf-idf search failed, continue with semantic results

        #sort by similarity score and return top k
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:k]
        return results

    def clear(self):
        """Clear the vector store"""
        self.embeddings_matrix = None
        self.documents = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

class ChatBot:
    """Main chatbot logic with Groq integration"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.groq_client = None

    def initialize_groq(self, api_key: str):
        """Initialize Groq client"""
        try:
            self.groq_client = Groq(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Error initializing Groq: {str(e)}")
            return False

    def web_search(self, query: str) -> str:
        """Simple web search simulation (replace with actual API)"""
        #this is a placeholder - replace with actual web search api
        return f"Web search results for '{query}' would appear here. Please integrate with SerpAPI or Bing API for actual web search functionality."

    def generate_response(self, query: str, search_modes: Dict[str, bool], context_docs: List[Dict] = None) -> str:
        """Generate response using Groq LLM"""
        #we now check st.session_state.groq_client directly
        if st.session_state.groq_client is None:
            return "Error: Groq API client not initialized. Please ensure your API key is configured correctly."

        #build context from retrieved documents
        context = ""
        if context_docs:
            context = "\n\nRelevant information from your documents:\n"
            for i, doc in enumerate(context_docs, 1):
                context += f"\n{i}. From {doc['source']}:\n{doc['content']}\n"

        #add web search context if enabled
        web_context = ""
        if search_modes.get('web_search', False):
            web_results = self.web_search(query)
            web_context = f"\n\nWeb search context:\n{web_results}\n"

        #build system prompt based on search modes
        system_prompt = """You are Chatur AI — an expert academic assistant designed to help students learn effectively, combining clarity, context-awareness, and accuracy.

Response & Behavior Guidelines:
- Your answers should always be clear, concise, and educational.
- Structure all responses so the most important information comes first, with step-by-step explanations as needed.
- Use concrete examples, relatable analogies, and definitions when helpful, but do not over-explain.
- Prioritize information from provided notes or user-uploaded documents; only use your general knowledge if this is insufficient.
- If the user greets you (“hi”, “hello”, etc.), respond only with:  
  "Hello! I’m Chatur AI.\nHow can I help you today?" (with nothing extra).
- If asked about "Ansh Pradhan," the creator or owner of this bot, reply:  
  "Ansh Pradhan is the creator and owner of this bot and the developer behind Chatur AI."

Formatting Rules:
- Use headings, bullet points (never asterisks), and concise lists for clarity.
- Keep answers focused; avoid unnecessary length and avoid over-explaining basic topics.
- Never use asterisk (*) characters for lists or emphasis.

General Principles:
- Be supportive and patient; encourage further questions.
- Confirm understanding when relevant.
- Cite sources or show document context if it strengthens your explanation.

Your goal: Help users master study concepts, clarify doubts, and have a positive learning experience!


"""

        if search_modes.get('use_notes', True) and context_docs:
            system_prompt += "Prioritize information from the student's uploaded documents when answering questions. "

        if search_modes.get('general_ai', True):
            system_prompt += "You can supplement with your general knowledge when the uploaded documents don't contain sufficient information. "

        if search_modes.get('web_search', False):
            system_prompt += "Consider web search results when available. "

        #create the full prompt
        full_prompt = f"{context}{web_context}\n\nStudent Question: {query}"

        try:
            response = st.session_state.groq_client.chat.completions.create( #use st.session_state.groq_client
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                model="llama3-8b-8192", #groq's llama model
                temperature=0.7,
                max_tokens=1024,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating response from Groq: {str(e)}"

    def chat(self, query: str, search_modes: Dict[str, bool]) -> Tuple[str, List[Dict]]:
        """Main chat function"""
        context_docs = []

        #search uploaded documents if enabled
        if search_modes.get('use_notes', True):
            context_docs = self.vector_store.search(query, k=3)

        #generate response
        response = self.generate_response(query, search_modes, context_docs)

        return response, context_docs

def main():
    st.set_page_config(
        page_title="Chatur AI",
        page_icon="icon.png",
        layout="wide"
    )

    st.title("🤖 Chatur AI Chatbot")
    st.markdown("*Upload your study materials and ask questions with flexible search options*")

    #initialize components
    doc_processor = DocumentProcessor()
    chatbot = ChatBot()

    creator_info = """
Who is Ansh Pradhan?
Ansh Pradhan is creator of ChaturAI. He is a 4th year Computer Engineering student at the Institute of Advanced Research - Gandhinagar, specializing in Artificial Intelligence and Machine Learning.
A passionate Python developer and AI enthusiast, he is committed to creating intelligent, accessible tools that solve real-world problems and enhance learning experiences.

Who is the creator of this bot? \ Who created this bot?
Chatur AI was built and deployed by Ansh Pradhan. Originally developed using Python, LangChain, Streamlit, and Groq's LLMs, it now runs as a fully functional Telegram chatbot enhanced with Vector Database-powered memory and retrieval for personalized, document-based responses.

Who is the owner of this bot?
The owner and developer of this bot is Ansh Pradhan, a student and developer with deep interest in AI and educational tools.

Details about the owner:
- Full Name: Ansh Pradhan
- Education: B.Tech in Computer Engineering at IAR Gandhinagar
- Areas of Interest: Artificial Intelligence, Machine Learning, Data Science, Python Development
- Profession: AI & Python Developer | Hackathon Winner | Tech Community Leader
- Skills: Python, Scikit-learn, Pandas, LangChain, Streamlit, Git, ML Algorithms, APIs
- Projects: Chatur AI (intelligent AI tutor bot), BMI predictor, Weather App, and more
- Motto: Building tools that are helpful, accessible, and intelligent — just like Chatur AI!
- Connect via linkedIn - https://www.linkedin.com/in/anshpradhan14/

How can I contact the owner?
You can connect with Ansh Pradhan on LinkedIn:
https://www.linkedin.com/in/anshpradhan14/

What Powers Chatur AI?
- Python + LangChain for LLM orchestration
- Groq LLMs for lightning-fast, token-efficient responses
- Streamlit for initial UI deployment
- Telegram Bot API for real-time chat integration
- FAISS / Chroma Vector DB for contextual memory and document-based Q&A
- OpenAI Functions / Tools for modular capabilities and intelligent routing
Chatur AI is designed to assist students, developers, and curious minds by combining conversational intelligence with custom document understanding — making it a powerful study buddy, AI tutor, or knowledge assistant.

About the Creator
- Name: Ansh Pradhan
- Education: B.Tech in Computer Engineering (AI Major), IAR Gandhinagar
- Skills: Python, ML Algorithms, LangChain, LLMs, Streamlit, FAISS, OpenCV, Git, APIs
- Projects:
  - Chatur AI – AI-powered chatbot with context-aware retrieval, accessible via Telegram and Streamlit web app.
  - Age & Gender Detection using CNN
  - Build Live International Space Station Tracker & Deployed on Streamlit
  - Stock Price Predictor
  - Oil Spill Detection System
  - Aircraft Damage Predictor
  - Weather Forecast App
  - BMI Predictor with Health Insights
"""
    custom_docs = doc_processor.process_text_input(creator_info, source_name="Bot Owner Info")
    chatbot.vector_store.add_documents(custom_docs)

    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.secrets.get("GROQ_API_KEY")

        if not st.session_state.groq_initialized:
            if groq_api_key:
                if chatbot.initialize_groq(groq_api_key):
                    st.session_state.groq_client = chatbot.groq_client
                    st.session_state.groq_initialized = True
                    st.success("Groq API connected!")
                else:
                    st.session_state.groq_initialized = False
                    st.error("Failed to connect Groq API. Check your key in .streamlit/secrets.toml")
            else:
                st.warning("Groq API Key not found in .streamlit/secrets.toml. Please configure it.")
                st.info("Create `.streamlit/secrets.toml` in your project root and add `GROQ_API_KEY=\"your_key_here\"`")
        else:
            st.success("Groq API is configured and ready.")

        st.write("")

        st.header("📚 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload your study materials (PDFs)",
            disabled=(PDF_LIBRARY is None)
        )

        manual_text = st.text_area(
            "Or paste your text directly:",
            placeholder="Paste your study notes, articles, or any text content here...",
            height=70,
            help="Alternative to PDF upload - paste your content directly"
        )

        text_source_name = st.text_input(
            "Source name for pasted text:",
            value="Manual Input",
            help="Give a name to your pasted content"
        )

        urls_text = st.text_area(
            "Enter URLs (one per line)",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Enter URLs to web pages you want to include"
        )

        if st.button("📥 Process Documents"):
            all_documents = []
            if uploaded_files and PDF_LIBRARY:
                for uploaded_file in uploaded_files:
                    st.info(f"Processing {uploaded_file.name}...")
                    pdf_docs = doc_processor.extract_pdf_text(uploaded_file)
                    all_documents.extend(pdf_docs)
                    if pdf_docs:
                        st.success(f"Extracted {len(pdf_docs)} chunks from {uploaded_file.name}")

            if manual_text.strip():
                st.info("Processing manual text input...")
                text_docs = doc_processor.process_text_input(manual_text, text_source_name)
                all_documents.extend(text_docs)
                if text_docs:
                    st.success(f"Processed {len(text_docs)} chunks from manual input")

            if urls_text.strip():
                urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                for url in urls:
                    st.info(f"Processing {url}...")
                    url_docs = doc_processor.extract_url_text(url)
                    all_documents.extend(url_docs)
                    if url_docs:
                        st.success(f"Extracted {len(url_docs)} chunks from {url}")
                    else:
                        st.warning(f"Failed to extract any meaningful chunks from {url}. It might be heavily JavaScript-rendered or have an unusual structure.")

            if all_documents:
                chatbot.vector_store.add_documents(all_documents)
                st.session_state.documents = all_documents
                st.success(f"Total: {len(all_documents)} document chunks processed!")
            else:
                st.warning("No documents were processed. Please check your inputs.")

        if st.button("🗑️ Clear All Documents"):
            chatbot.vector_store.clear()
            st.session_state.documents = []
            st.session_state.chat_history = []
            st.success("Documents cleared!")

        if st.session_state.documents:
            st.info(f"{len(st.session_state.documents)} chunks loaded")

        st.divider()

        #search options
        st.header("Search Options")
        use_notes = st.checkbox("📚 Use Notes", value=True, help="Search in uploaded documents")
        web_search = st.checkbox("🌐 Web Search", value=False, help="Include web search results")
        general_ai = st.checkbox("🧠 General AI", value=True, help="Use AI's general knowledge")

        search_modes = {
            'use_notes': use_notes,
            'web_search': web_search,
            'general_ai': general_ai
        }

        active_modes = [mode for mode, enabled in search_modes.items() if enabled]
        if active_modes:
            st.success(f"Active: {', '.join(active_modes)}")
        else:
            st.warning("No search modes selected!")

    #main chat interface
    st.header("💬 Chat with Your Study Buddy")

    chat_container = st.container()

    with chat_container:
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}...", expanded=(i == len(st.session_state.chat_history) - 1)):
                st.write("**Question:**", question)
                st.write("**Answer:**", answer)

                if sources:
                    st.write("**Sources:**")
                    for j, source in enumerate(sources, 1):
                        st.write(f"{j}. {source['source']} (Score: {source['similarity_score']:.3f})")

    #modified section for query input
    query = st.text_input(
        "Ask a question:",
        placeholder="What would you like to know about your study materials?",
        key="query_input",
        on_change=lambda: process_query(st.session_state.query_input, search_modes, chatbot) #call a function when input changes
    )

    col_ask, col_middle_spacer, col_clear = st.columns([2, 4, 1])

    with col_ask:
        pass

    with col_clear:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

#function for processing query
def process_query(query_text: str, search_modes: Dict[str, bool], chatbot_instance: ChatBot):
    if query_text: #only process if query is not empty
        if not any(search_modes.values()):
            st.warning("Please select at least one search mode!")
            #this warning might disappear quickly as the page reruns,
            #so ideally search modes would be chosen before asking.
        elif st.session_state.groq_client is None:
            st.warning("Groq API client is not initialized. Please ensure your API key is correctly set in `.streamlit/secrets.toml`.")
        else:
            with st.spinner("Thinking..."):
                response, context_docs = chatbot_instance.chat(query_text, search_modes)
                st.session_state.chat_history.append((query_text, response, context_docs))
                st.session_state.query_input = "" #clear the text input widget's value

if __name__ == "__main__":
    main()

st.markdown(
    "<hr style='margin-top:30px;margin-bottom:10px;'>"
    "<div style='text-align:center; color:gray;'>© 2025 Ansh Pradhan. All rights reserved.</div>",
    unsafe_allow_html=True,
)
