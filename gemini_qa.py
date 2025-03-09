import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GeminiQA:
    def __init__(self, api_key: str):
        """
        Initialize GeminiQA with the API key and model configuration.
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logging.info("GeminiQA initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize GeminiQA: {e}")
            raise RuntimeError("Error initializing GeminiQA.")


    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer using Gemini's LLM based on retrieved document context."""
        try:
            if not context.strip():
                logging.warning("Context is empty; response may be inaccurate.")
                return "Information not found."
            
            prompt = (
                "You are an expert in analyzing documents and providing precise answers.\n\n"
                "Here is some extracted text from a document:\n"
                f"{context}\n\n"
                "Based on this, answer the following question concisely and accurately:\n"
                f"{question}\n\n"
                "If the information is not present in the document, simply state 'Information not found'."
            )
            
            response = self.model.generate_content(prompt)
            return response.text.strip() if response and hasattr(response, 'text') else "Error generating response."
        
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "An error occurred while generating the answer."