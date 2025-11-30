import os
import io
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

# --- Framework Imports ---
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from bson import ObjectId
from passlib.context import CryptContext
import jwt
from groq import Groq

# --- Math & NLP Imports (For Keywords) ---
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Database & Logic Imports ---
from db import users_col, chats_col, messages_col, docs_col, embeddings_col
from embedder import get_embedding
# We import DocumentProcessor to handle PDF/Text parsing logic
from app import DocumentProcessor 

# ============================================================================
# 1. CONFIGURATION & SETUP
# ============================================================================

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# API Keys & Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is missing in .env file!")

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET is missing in .env file!")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- OWNER CONTEXT (Hardcoded Knowledge) ---
OWNER_INFO_CONTEXT = """
[SYSTEM MEMORY: CREATOR & IDENTITY INFO]
Who is Ansh Pradhan?
Ansh Pradhan is the creator and owner of ChaturAI. He is a 4th year Computer Engineering student at the Institute of Advanced Research - Gandhinagar, specializing in Artificial Intelligence and Machine Learning.
- Profession: Python Developer, AI Enthusiast.
- Projects: Chatur AI, BMI Predictor, Weather App.
- Skills: Python, LangChain, Streamlit, MongoDB, Vector Databases.
- Contact: https://www.linkedin.com/in/anshpradhan14/
- Motto: Building tools that are helpful, accessible, and intelligent.

Who created this bot? / Who owns this bot?
Chatur AI was built and deployed by Ansh Pradhan. It uses Groq LLMs, MongoDB Atlas Vector Search, and FastAPI backend.
"""

# ============================================================================
# 2. AUTHENTICATION HELPERS
# ============================================================================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(days=1)) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def get_current_user(Authorization: str = Header(None)):
    """
    Dependency to verify JWT token and retrieve the current user.
    """
    if Authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authentication Token")
    
    try:
        token = Authorization.replace("Bearer ", "")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid Token Payload")
            
        user = users_col.find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
            
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid Authentication Token")

def extract_json_from_response(text: str):
    """
    Clean up LLM response to get valid JSON. 
    Removes markdown code blocks if present.
    """
    try:
        # If the model outputs ```json ... ```
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Fallback: try to find list start/end
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except:
                pass
        return {"error": "Failed to parse JSON", "raw": text}
    
# ============================================================================
# 3. PYDANTIC DATA MODELS
# ============================================================================

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ChatCreate(BaseModel):
    title: str = "New Chat"

class RenameChatRequest(BaseModel):
    title: str

class ChatRequest(BaseModel):
    chat_id: str
    query: str
    mode: str = "hybrid"  # "strict" | "hybrid"

class UrlRequest(BaseModel):
    url: str
    chat_id: str

class TextRequest(BaseModel):
    text: str
    chat_id: str

class KeywordRequest(BaseModel):
    text: str

class QuizRequest(BaseModel):
    chat_id: str
    question_count: int = 5

# ============================================================================
# 4. APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Chatur AI Production Backend",
    description="API for RAG Chatbot with MongoDB, Groq, and Auth",
    version="2.5.0"
)

# CORS Middleware (Allow connection from Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace '*' with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 5. AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/register")
def register_user(user: UserCreate):
    if users_col.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = {
        "name": user.name,
        "email": user.email,
        "password": hash_password(user.password),
        "created_at": datetime.utcnow()
    }
    result = users_col.insert_one(new_user)
    return {"status": "success", "user_id": str(result.inserted_id), "message": "User created successfully"}

@app.post("/auth/login")
def login_user(user: UserLogin):
    db_user = users_col.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(data={"user_id": str(db_user["_id"]), "email": db_user["email"]})
    
    return {
        "status": "success",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(db_user["_id"]),
            "name": db_user["name"],
            "email": db_user["email"]
        }
    }

# ============================================================================
# 6. CHAT MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/chats/my")
def get_user_chats(user: dict = Depends(get_current_user)):
    """Get all chats belonging to the logged-in user."""
    chats = list(chats_col.find({"user_id": str(user["_id"])}).sort("created_at", -1))
    # Convert ObjectId to string
    for chat in chats:
        chat["_id"] = str(chat["_id"])
    return chats

@app.post("/chats/create")
def create_chat(payload: ChatCreate, user: dict = Depends(get_current_user)):
    """Create a new empty chat session."""
    new_chat = {
        "user_id": str(user["_id"]),
        "title": payload.title,
        "created_at": datetime.utcnow()
    }
    result = chats_col.insert_one(new_chat)
    return {"chat_id": str(result.inserted_id), "title": new_chat["title"]}

@app.put("/chats/{chat_id}")
def rename_chat(chat_id: str, payload: RenameChatRequest, user: dict = Depends(get_current_user)):
    """Rename an existing chat."""
    result = chats_col.update_one(
        {"_id": ObjectId(chat_id), "user_id": str(user["_id"])},
        {"$set": {"title": payload.title}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
    
    return {"status": "success", "chat_id": chat_id, "new_title": payload.title}

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, user: dict = Depends(get_current_user)):
    """
    Hard Delete: Removes Chat, Messages, Documents, and Embeddings.
    """
    # 1. Verify Ownership
    chat = chats_col.find_one({"_id": ObjectId(chat_id), "user_id": str(user["_id"])})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
    
    # 2. Cascading Delete
    chats_col.delete_one({"_id": ObjectId(chat_id)})
    messages_col.delete_many({"chat_id": chat_id})
    docs_col.delete_many({"chat_id": chat_id})
    embeddings_col.delete_many({"chat_id": chat_id})
    
    logger.info(f"User {user['_id']} deleted chat {chat_id} and all associated data.")
    return {"status": "success", "message": "Chat and all data deleted permanently."}

@app.get("/chats/{chat_id}/messages")
def get_chat_history(chat_id: str, user: dict = Depends(get_current_user)):
    """Retrieve message history for a specific chat."""
    # Check access
    if not chats_col.find_one({"_id": ObjectId(chat_id), "user_id": str(user["_id"])}):
        raise HTTPException(status_code=403, detail="Access denied")
        
    msgs = list(messages_col.find({"chat_id": chat_id}).sort("timestamp", 1))
    return [
        {"role": m["role"], "content": m["content"], "timestamp": m["timestamp"]}
        for m in msgs
    ]

# ============================================================================
# 7. DOCUMENT UPLOAD ENDPOINTS (PDF, URL, TEXT)
# ============================================================================

def save_document_and_embeddings(chunks: List[Dict], chat_id: str, user_id: str, title: str, doc_type: str):
    """Helper function to save docs and embeddings to MongoDB to avoid code duplication."""
    # 1. Save Document Metadata
    doc_entry = {
        "user_id": user_id,
        "chat_id": chat_id,
        "title": title,
        "doc_type": doc_type,
        "uploaded_at": datetime.utcnow(),
        "chunk_count": len(chunks)
    }
    doc_result = docs_col.insert_one(doc_entry)
    doc_id = str(doc_result.inserted_id)

    # 2. Generate & Save Embeddings
    embeddings_to_insert = []
    for chunk in chunks:
        try:
            vector = get_embedding(chunk['content'])
            embeddings_to_insert.append({
                "doc_id": doc_id,
                "chat_id": chat_id,
                "user_id": user_id,
                "chunk_text": chunk['content'],
                "embedding": vector,
                "source": title
            })
        except Exception as e:
            logger.error(f"Embedding generation failed for chunk: {e}")

    if embeddings_to_insert:
        embeddings_col.insert_many(embeddings_to_insert)
        
    return doc_id, len(embeddings_to_insert)

@app.post("/documents/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chat_id: str = Form(...),
    user: dict = Depends(get_current_user)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
    try:
        content = await file.read()
        pdf_file = io.BytesIO(content)
        
        processor = DocumentProcessor()
        chunks = processor.extract_pdf_text(pdf_file, file.filename, str(user["_id"]), chat_id)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
            
        doc_id, count = save_document_and_embeddings(chunks, chat_id, str(user["_id"]), file.filename, "pdf")
        return {"status": "success", "doc_id": doc_id, "chunks_processed": count}
        
    except Exception as e:
        logger.error(f"PDF Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload/url")
def upload_url(payload: UrlRequest, user: dict = Depends(get_current_user)):
    try:
        processor = DocumentProcessor()
        chunks = processor.extract_url_text(payload.url, str(user["_id"]), payload.chat_id)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not scrape text from URL.")
            
        doc_id, count = save_document_and_embeddings(chunks, payload.chat_id, str(user["_id"]), payload.url, "url")
        return {"status": "success", "doc_id": doc_id, "chunks_processed": count}
        
    except Exception as e:
        logger.error(f"URL Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload/text")
def upload_text(payload: TextRequest, user: dict = Depends(get_current_user)):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
        
    try:
        processor = DocumentProcessor()
        # We treat raw text as a "User Note"
        chunks = processor.process_text_input(payload.text, "User Note", str(user["_id"]), payload.chat_id)
        
        doc_id, count = save_document_and_embeddings(chunks, payload.chat_id, str(user["_id"]), "Raw Text Note", "text")
        return {"status": "success", "doc_id": doc_id, "chunks_processed": count}
        
    except Exception as e:
        logger.error(f"Text Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 8. MAIN CHAT ENDPOINT (RAG LOGIC)
# ============================================================================

@app.post("/chat")
def chat_endpoint(payload: ChatRequest, user: dict = Depends(get_current_user)):
    chat_id = payload.chat_id
    query = payload.query
    mode = payload.mode # "strict" or "hybrid"

    # 1. Verify Chat Access
    if not chats_col.find_one({"_id": ObjectId(chat_id), "user_id": str(user["_id"])}):
        raise HTTPException(status_code=403, detail="Access denied to this chat")

    # 2. Log User Message
    messages_col.insert_one({
        "chat_id": chat_id,
        "role": "user",
        "content": query,
        "timestamp": datetime.utcnow()
    })

    # 3. Vector Search (Retrieve Context)
    context_text = ""
    try:
        query_vector = get_embedding(query)
        results = list(embeddings_col.aggregate([
            {
                "$vectorSearch": {
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": 50,
                    "limit": 6, # Fetch top 20 chunks to avoid context fragmentation
                    "index": "vector_index",
                    "filter": {"chat_id": chat_id}
                }
            },
            {
                "$project": {
                    "chunk_text": 1,
                    "source": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]))
        
        # Format Context
        if results:
            context_text = "\n\n".join([f"--- Source: {r.get('source', 'Unknown')} ---\n{r['chunk_text']}" for r in results])
            
    except Exception as e:
        logger.error(f"Vector Search Failed: {e}")
        # In Hybrid mode, we continue. In Strict mode, this is critical.

    # 4. Retrieve Conversation History (Last 6 messages)
    history_cursor = messages_col.find({"chat_id": chat_id}).sort("timestamp", -1).limit(6) 
    history_msgs = list(history_cursor)[::-1]
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history_msgs])

    # 5. Construct System Prompt based on Mode
    
    formatting_rules = """
    FORMATTING RULES:
    - Use standard Markdown for lists.
    - For numbered lists, use sequential numbers (1., 2., 3...) not all 1s.
    - Bold the names of key concepts (e.g., **1. Supervised Learning**).
    """

    if mode == "strict":
        system_instructions = f"""
        You are Chatur AI, a strict academic assistant.
        
        {OWNER_INFO_CONTEXT}
        
        **STRICT MODE RULES:**
        1. Answer ONLY using the context provided below.
        2. If the answer is NOT in the context, explicitly state: "I cannot find this information in the uploaded documents."
        3. Do not use general knowledge to answer questions about the document content.
        
        {formatting_rules}
        
        CONTEXT FROM DOCUMENTS:
        {context_text}
        
        CONVERSATION HISTORY:
        {history_text}
        """
    else: # Hybrid Mode
        system_instructions = f"""
        You are Chatur AI, a helpful and intelligent assistant.
        
        {OWNER_INFO_CONTEXT}
        
        **HYBRID MODE INSTRUCTIONS:**
        1. Prioritize the 'CONTEXT' provided below for specific details.
        2. If the context is insufficient, use your General AI Knowledge to provide a complete answer by adding points apart from the context provided by the user.
        3. Clearly mention if you are answering from general knowledge vs. the documents.
        4. Answer should contain more details about the context which have provided by the user and add external context too.

        {formatting_rules}

        CONTEXT FROM DOCUMENTS:
        {context_text}
        
        CONVERSATION HISTORY:
        {history_text}
        """

    # 6. Generate Response via Groq
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": query}
            ],
            temperature=0.6,
            max_tokens=1500
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = "I apologize, but I encountered an error while processing your request."
        logger.error(f"Groq API Error: {e}")

    # 7. Save Assistant Response
    messages_col.insert_one({
        "chat_id": chat_id,
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.utcnow()
    })

    return {
        "answer": answer,
        "mode_used": mode,
        "context_retrieved": bool(context_text)
    }

# ============================================================================
# 9. UTILITY ENDPOINTS (SUMMARY, QUIZ, KEYWORDS)
# ============================================================================

def get_chat_combined_text(chat_id: str, limit_chars: int = 15000) -> str:
    """Retrieve all text chunks for a chat to generate summaries/quizzes."""
    cursor = embeddings_col.find({"chat_id": chat_id}).limit(60)
    full_text = "\n".join([doc['chunk_text'] for doc in cursor])
    return full_text[:limit_chars]

@app.post("/api/summarize")
def generate_summary(payload: dict, user: dict = Depends(get_current_user)):
    """Generates a Markdown summary with bullet points."""
    chat_id = payload.get("chat_id")
    # Fetch text using the helper function
    text_context = get_chat_combined_text(chat_id)
    
    if len(text_context) < 50:
        return {"summary": "Not enough content to generate a summary. Please upload documents first."}

    system_prompt = """
    You are an expert academic summarizer. 
    Format your response in **Markdown**. 
    
    **Formatting Rules:**
    1. Use **Bold** for key terms.
    2. Use Bullet points for lists.
    3. Use structured Headings (#, ##).
    4. **Tables:** If comparing items, use a standard Markdown Table.
       - **CRITICAL:** You MUST put a newline character after every row in the table. 
       - Do NOT write the table on a single line.
    """
    
    query = f"Summarize these notes in a structured way:\n{text_context}"
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=4096
        )
        return {"summary": resp.choices[0].message.content}
    except Exception as e:
        return {"summary": f"Error: {str(e)}"}

@app.post("/api/quiz")
def generate_quiz(payload: QuizRequest, user: dict = Depends(get_current_user)):
    """Generates Quiz with user-specified question count."""
    chat_id = payload.chat_id
    count = payload.question_count
    
    # Cap the count to prevent timeouts/abuse
    if count < 1: count = 5
    if count > 20: count = 20
    
    text_context = get_chat_combined_text(chat_id)
    if len(text_context) < 50: return {"quiz_questions": []}

    system_prompt = f"""
    Generate exactly {count} Multiple Choice Questions based on the context.
    
    CRITICAL INSTRUCTIONS:
    1. Return strictly a JSON Array of objects.
    2. Do NOT repeat questions.
    3. Ensure distractor options (wrong answers) are plausible.
    4. Format:
    [
        {{
            "question": "Question text?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Option B" 
        }}
    ]
    """
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{text_context}"}
            ],
            max_tokens=4096 
        )
        content = resp.choices[0].message.content
        data = extract_json_from_response(content)
        return {"quiz_questions": data}
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")
        return {"quiz_questions": []}

@app.post("/api/flashcards")
def generate_flashcards(payload: dict, user: dict = Depends(get_current_user)):
    """Generates Flashcards (LLM decides count based on content density)."""
    chat_id = payload.get("chat_id")
    text_context = get_chat_combined_text(chat_id)

    if len(text_context) < 50: return {"flashcards": []}

    system_prompt = """
    Analyze the context and generate an appropriate number of flashcards (between 5 and 15) based on the density of information.
    - If the text is short, generate fewer cards.
    - If the text is dense/long, generate more cards to cover key concepts.
    
    Return strictly a JSON Array of objects:
    [
        {"front": "Concept/Question", "back": "Definition/Answer"}
    ]
    """
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{text_context}"}
            ],
            max_tokens=4096
        )
        content = resp.choices[0].message.content
        data = extract_json_from_response(content)
        return {"flashcards": data}
    except Exception as e:
        logger.error(f"Flashcard generation failed: {e}")
        return {"flashcards": []}

@app.post("/api/keywords")
def extract_keywords(payload: KeywordRequest):
    """
    Extracts keywords using TF-IDF (Scikit-Learn).
    Useful for highlighting important terms in the frontend.
    """
    text = payload.text
    if not text or len(text) < 50:
        return {"keywords": []}
        
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=15)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        
        # Sort by score
        keyword_scores = zip(feature_names, scores)
        sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
        
        return {"keywords": [k[0] for k in sorted_keywords]}
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return {"keywords": []}

@app.get("/")
def root():
    return {"status": "Running", "service": "Chatur AI Production Backend", "version": "2.5.0"}