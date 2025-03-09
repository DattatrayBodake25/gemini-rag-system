import fitz  # PyMuPDF for PDF processing
import re
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFProcessor:
    def __init__(self, file_path: str, max_pages: Optional[int] = None):
        """
        Initialize PDFProcessor.
        
        :param file_path: Path to the PDF file.
        :param max_pages: Optional limit on the number of pages to process.
        """
        self.file_path = file_path
        self.max_pages = max_pages  # Limit the number of pages to process

    def extract_text(self, start_page: int = 0, end_page: Optional[int] = None) -> str:
        """
        Extract text from a PDF file within a specified page range.
        
        :param start_page: Page number to start extraction from (default is 0).
        :param end_page: Page number to stop extraction at (default is None, meaning last page).
        :return: Extracted and preprocessed text.
        """
        text = ""
        try:
            with fitz.open(self.file_path) as doc:
                total_pages = len(doc)
                end_page = min(end_page or total_pages, total_pages)  # Ensure valid range
                if self.max_pages:
                    end_page = min(start_page + self.max_pages, total_pages)  # Apply page limit

                logging.info(f"Extracting text from pages {start_page} to {end_page - 1}.")
                for page_num in range(start_page, end_page):
                    text += doc[page_num].get_text("text")

        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            return ""  # Return empty text on failure
        
        return self.preprocess_text(text)

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        :param text: Raw extracted text.
        :return: Cleaned and normalized text.
        """
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and newlines
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        :param text: Text to be split.
        :param chunk_size: Maximum size of each chunk in words.
        :param overlap: Overlapping words between chunks for better context preservation.
        :return: List of text chunks.
        """
        words = text.split()
        if not words:
            logging.warning("No text found to chunk.")
            return []

        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
        logging.info(f"Text chunked into {len(chunks)} chunks.")
        return chunks