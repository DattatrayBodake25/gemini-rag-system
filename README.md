# RAG-Based Q&A System Using Google's Gemini API

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** based Q&A system using **Google's Gemini API**. It extracts text from a PDF, processes it into chunks, generates embeddings, stores them in a **FAISS vector database**, and retrieves relevant content to generate answers based on user queries.

## 🚀 Features
- **PDF Processing & Text Extraction**: Supports **PyMuPDF (fitz)** for text extraction.
- **Document Chunking**: Splits extracted text into overlapping chunks (~500 tokens each).
- **Embedding & Vector Search**: Uses **Gemini's embedding API** & **FAISS** for efficient retrieval.
- **LLM-based Answer Generation**: Passes retrieved text to **Google's Gemini API** to generate accurate answers.
- **Streamlit UI**: Provides an interactive interface for uploading PDFs and querying.

---

## 🛠️ Tech Stack
- **Python** (Main Programming Language)
- **Streamlit** (Web Interface)
- **FAISS** (Vector Search for Embeddings)
- **Google Gemini API** (LLM for Answer Generation)
- **PyMuPDF (fitz) / pdfplumber** (PDF Processing)
- **NumPy, Scikit-learn** (Vector Operations & Similarity Calculation)

---

## 📂 Project Structure
```
rag_system/
│
├── .env                    # Stores the Gemini API key
├── pdf_processor.py        # Handles PDF text extraction and chunking
├── embedding_manager.py    # Manages embeddings and FAISS vector storage
├── retrieval_engine.py     # Retrieves relevant chunks
├── gemini_qa.py           # Generates answers using Gemini API
├── app.py                 # Streamlit app for user interaction
└── requirements.txt       # List of dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DattatrayBodake25/gemini-rag-system
cd rag_system
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up API Key
Create a `.env` file and add your **Google Gemini API Key**:
```plaintext
GEMINI_API_KEY=your_api_key_here
```

---

## 🚀 Usage

### Running the Streamlit App
```bash
streamlit run app.py
```

### Using the Application
1. **Upload a PDF** via the Streamlit sidebar.
2. **View Extracted Text** preview in the UI.
3. **Enter a Question** related to the document.
4. **Get AI-Generated Answers** based on retrieved content.

---

## 🏗️ Module Breakdown

### 1️⃣ `pdf_processor.py` - PDF Text Extraction & Chunking
- Extracts text from PDFs using **PyMuPDF**.
- Splits the text into **overlapping chunks** (~500 tokens each).
- Cleans and normalizes the extracted text.

### 2️⃣ `embedding_manager.py` - Embedding Generation & FAISS Indexing
- Generates **text embeddings** using **Gemini's embedding API**.
- Stores embeddings in a **FAISS index** for fast retrieval.

### 3️⃣ `retrieval_engine.py` - Chunk Retrieval
- Searches the FAISS index to retrieve **top-3 most relevant chunks**.
- Uses **cosine similarity** for ranking relevant chunks.

### 4️⃣ `gemini_qa.py` - Answer Generation
- Constructs a **prompt** using the retrieved document context.
- Passes it to **Google Gemini API** for answer generation.
- Returns **precise, document-based responses**.

### 5️⃣ `app.py` - Streamlit UI
- Provides an **interactive interface** for document upload & Q&A.
- Handles **PDF processing, chunking, retrieval, and answer generation**.

---

## 🔍 Example Workflow
1. **Upload a PDF** (e.g., a research paper, legal document, or report).
2. The system extracts and chunks text into **smaller passages**.
3. **Embeddings** are generated and stored in **FAISS**.
4. The user enters a **question** related to the document.
5. The system retrieves **top-3 relevant chunks** based on embeddings.
6. **Google Gemini API** generates an answer based on the retrieved content.

---

## 📝 Requirements
To ensure smooth execution, install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
Some key dependencies include:
- `faiss-cpu`
- `streamlit`
- `PyMuPDF`
- `google-generativeai`
- `pdfplumber`
- `scikit-learn`

---

## 📜 License
This project is licensed under the **MIT License**.

**🚀 Happy Coding! 🎯**
