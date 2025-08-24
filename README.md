# DocuRAG - Document Retrieval-Augmented Generation

DocuRAG is a simple RAG (Retrieval-Augmented Generation) project that allows you to query documents and get intelligent responses using **local embeddings** and **OpenAI’s GPT API**.  

## Features
- Local embeddings with **Sentence-Transformers** (Hugging Face)  
- Chroma vector store for document retrieval  
- Retrieval-Augmented Generation using **OpenAI GPT**  
- Easy to extend for multiple documents  

## Project Structure
- `books/` - Contains source documents (e.g., `odyssey.txt`)  
- `db/chroma_db/` - Persistent vector store directory  
- `src/docurag_rag.py` - Main Python script  
- `.env` - Environment variables (OpenAI API key)  
- `requirements.txt` - Python dependencies  

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/DocuRAG.git
cd DocuRAG
