import streamlit as st
import os
import logging
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from embedding_manager import EmbeddingManager
from retrieval_engine import RetrievalEngine
from gemini_qa import GeminiQA

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Error: Gemini API key not found. Please set it in the .env file.")
    logging.error("Gemini API key not found. Application cannot proceed.")
    st.stop()

# Streamlit App Configuration
st.set_page_config(
    page_title="AI-Powered Q&A System",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for Better UI
def set_custom_css():
    st.markdown(
        """
        <style>
            .stTextInput>div>div>input { border-radius: 10px; padding: 10px; }
            .stTextArea>div>div>textarea { border-radius: 10px; }
            .stButton>button { border-radius: 10px; font-size: 18px; }
            .stAlert { border-radius: 10px; }
            .block-container { padding-top: 20px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_css()

# Sidebar - PDF Upload
with st.sidebar:
    st.title("📄 Upload PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], help="Only PDF files are supported.")
    
    if uploaded_file:
        temp_pdf_path = "temp.pdf"
        try:
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract Text
            pdf_processor = PDFProcessor(temp_pdf_path)
            text = pdf_processor.extract_text()
            
            if not text.strip():
                st.warning("⚠️ No text extracted. The document might be empty or contain images only.")
                logging.warning("No text extracted from PDF.")
            
            # Show preview
            st.subheader("📜 Preview Extracted Text")
            st.text_area("Extracted Content", text, height=300)
        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")
            logging.error(f"Error processing PDF: {e}")

# Main UI
st.title("🤖 RAG-Based Q&A System Using Gemini API")
st.markdown("Ask questions based on the uploaded document using AI-powered retrieval.")

if uploaded_file:
    try:
        pdf_processor = PDFProcessor(temp_pdf_path)
        text = pdf_processor.extract_text()
        chunks = pdf_processor.chunk_text(text)
        
        if not chunks:
            st.error("⚠️ No text chunks generated. Ensure the document contains readable text.")
            logging.error("No chunks generated.")
            st.stop()
        
        # Embeddings & Retrieval Setup
        embedding_manager = EmbeddingManager(GEMINI_API_KEY)
        embeddings = embedding_manager.generate_embeddings(chunks)
        embedding_manager.build_faiss_index(embeddings)
        retrieval_engine = RetrievalEngine(embedding_manager)
        gemini_qa = GeminiQA(GEMINI_API_KEY)

        # User Input
        st.subheader("🔍 Ask a Question")
        question = st.text_input("Enter your question:", placeholder="What is the key insight from the document?")
        
        if question:
            with st.spinner("🔍 Searching for relevant content..."):
                relevant_chunks = retrieval_engine.retrieve_chunks(question, chunks)
                context = "\n".join(relevant_chunks)

            if not context.strip():
                st.warning("⚠️ No relevant content found in the document to answer this question.")
                logging.warning("No relevant chunks retrieved.")
            else:
                with st.spinner("🤖 Generating answer..."):
                    answer = gemini_qa.generate_answer(context, question)
                
                st.subheader("🎯 Smart Answer:")
                st.success(answer)
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        logging.error(f"Error: {e}")
else:
    st.info("📂 Please upload a PDF to start your Q&A session.")