# DocChat AI

DocChat AI is an AI-powered document question-answering application that enables users to upload multiple documents and interact with them through natural language conversations. It uses Retrieval-Augmented Generation (RAG) to provide context-aware answers based on the uploaded documents while maintaining conversational memory.

The application is built with Streamlit, LangChain, ChromaDB, Hugging Face Embeddings, and Groq's Llama 3 model.

## Features

- Upload multiple documents simultaneously
- Supports PDF, DOCX, TXT, Markdown, and CSV files
- Retrieval-Augmented Generation (RAG)
- Semantic document search using vector embeddings
- Conversational memory across the chat session
- Source references for every response
- Multiple document knowledge base
- Modern dark and light mode interface
- Fast inference using Groq Llama 3
- Responsive Streamlit web application

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API (Llama 3)
- Hugging Face Sentence Transformers
- ChromaDB
- Recursive Character Text Splitter

## Project Workflow

1. Upload one or more documents.
2. Documents are loaded and processed.
3. Text is split into smaller chunks.
4. Chunks are converted into vector embeddings.
5. Embeddings are stored in ChromaDB.
6. User asks questions.
7. Relevant document chunks are retrieved using semantic search.
8. Retrieved context is sent to the LLM.
9. The AI generates an accurate, context-aware response with source references.

## Supported File Types

- PDF
- DOCX
- TXT
- Markdown (.md)
- CSV

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/docchat-ai.git
```

Navigate to the project folder:

```bash
cd docchat-ai
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configure Groq API

### Local Environment

Windows PowerShell

```powershell
$env:GROQ_API_KEY="YOUR_API_KEY"
```

Windows CMD

```cmd
set GROQ_API_KEY=YOUR_API_KEY
```

Linux/macOS

```bash
export GROQ_API_KEY=YOUR_API_KEY
```

### Streamlit Community Cloud

Create:

```
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY="YOUR_API_KEY"
```

## Run the Application

```bash
streamlit run rag_app_cloud.py
```

## Project Structure

```
DocChat-AI/
│
├── rag_app_cloud.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml
```

## Future Improvements

- Chat history export
- Conversation summarization
- OCR support for scanned PDFs
- Image-based document support
- Multi-user authentication
- Cloud vector database integration
- Citation highlighting
- Hybrid search (keyword + semantic)
- Streaming responses

## Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Large Language Model (LLM) integration
- Prompt Engineering
- Vector Databases
- Semantic Search
- Hugging Face Embeddings
- LangChain
- ChromaDB
- Streamlit
- Conversational AI
- Document Intelligence

## Author

**Farmaan A**

Final Year B.Tech Artificial Intelligence & Data Science

Interested in Artificial Intelligence, Generative AI, Machine Learning, and AI-powered Web Applications.