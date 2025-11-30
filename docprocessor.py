import requests
from bs4 import BeautifulSoup
import io
from typing import List, Dict

# --- PDF Library Handling ---
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
    """
    Handles PDF and URL document processing and text chunking.
    Returns clean list of chunks with metadata to be saved by the API.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _log_error(self, message: str):
        print(f"[ERROR] {message}")

    def extract_pdf_text(self, pdf_file: io.BytesIO, filename: str, user_id: str, chat_id: str) -> List[Dict]:
        """Extract text from PDF and tag with user/chat ID."""
        if PDF_LIBRARY is None:
            self._log_error("No PDF library available. Please install: pip install PyPDF2")
            return []

        try:
            pdf_file.seek(0)
            documents = []
            pdf_reader = None

            # Initialize reader based on available library
            if PDF_LIBRARY == "pypdf2":
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            elif PDF_LIBRARY == "pypdf":
                import pypdf
                pdf_reader = pypdf.PdfReader(pdf_file)

            if pdf_reader:
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        chunks = self._chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'content': chunk,
                                'source': f"{filename}",
                                'type': 'pdf',
                                'metadata': {
                                    'page': page_num + 1,
                                    'chunk': i + 1,
                                    'filename': filename,
                                    'user_id': user_id,
                                    'chat_id': chat_id
                                }
                            })
                return documents
            return []
        except Exception as e:
            self._log_error(f"Error processing PDF {filename}: {str(e)}")
            return []

    def extract_url_text(self, url: str, user_id: str, chat_id: str) -> List[Dict]:
        """Scrape text from URL and tag with user/chat ID."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove junk elements
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            if not text:
                return []

            chunks = self._chunk_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'source': url,
                    'type': 'url',
                    'metadata': {
                        'url': url,
                        'chunk': i + 1,
                        'user_id': user_id,
                        'chat_id': chat_id
                    }
                })
            return documents
        except Exception as e:
            self._log_error(f"Error processing URL {url}: {str(e)}")
            return []

    def process_text_input(self, text: str, source_name: str, user_id: str, chat_id: str) -> List[Dict]:
        """Process raw text input directly."""
        if not text.strip():
            return []

        chunks = self._chunk_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'content': chunk,
                'source': source_name,
                'type': 'text',
                'metadata': {
                    'chunk': i + 1,
                    'source_name': source_name,
                    'user_id': user_id,
                    'chat_id': chat_id
                }
            })
        return documents

    def _chunk_text(self, text: str) -> List[str]:
        """Splits text into overlapping chunks for better RAG context."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunks.append(text[start:])
                break
            
            # Try to find a natural break point (newline or period) to avoid cutting sentences mid-word
            chunk = text[start:end]
            break_point = max(chunk.rfind('.'), chunk.rfind('\n'))
            
            if break_point > self.chunk_size // 2:
                end = start + break_point + 1
                
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
        return [c.strip() for c in chunks if c.strip()]