# CLAUDE.md - RAG Document Q&A Project

## Project Overview

A Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over PDF documents using OpenAI, LangChain, and ChromaDB.

## Tech Stack

- **Python 3.8+**
- **LangChain 1.0+** - LLM framework with LCEL patterns
- **LangChain Chroma** - Vector database for document retrieval (new package: langchain-chroma)
- **LangChain Ollama** - Ollama integration (new package: langchain-ollama)
- **pypdf** - PDF parsing
- **tenacity** - Retry logic

## Key Files

| File | Purpose |
|------|---------|
| `rag_script.py` | Main RAG pipeline - loads PDFs, creates embeddings, builds vector store |
| `query.py` | CLI interface for querying documents |
| `llm_config.py` | Multi-provider LLM configuration module |
| `documents/` | Place PDF files here |
| `chroma_db/` | Persistent vector store (auto-generated) |
| `logs/` | Log files (auto-generated) |
| `.env` | API keys and configuration |

## Commands

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Build/update vector store (run after adding new PDFs)
.venv\Scripts\python.exe rag_script.py

# Query documents (CLI)
.venv\Scripts\python.exe query.py "Your question here"

# Interactive mode
.venv\Scripts\python.exe query.py
```

## Supported LLM Providers

The system supports multiple LLM providers. Configure via `LLM_PROVIDER` in `.env`:

| Provider | Value | API Key Env | Model Examples |
|----------|-------|-------------|----------------|
| **OpenAI** | `openai` | `OPENAI_API_KEY` | gpt-3.5-turbo, gpt-4 |
| **Anthropic** | `anthropic` | `ANTHROPIC_API_KEY` | claude-3-haiku, claude-3-sonnet |
| **Google** | `google` | `GOOGLE_API_KEY` | gemini-pro |
| **ZhipuAI** | `zhipuai` | `ZHIPUAI_API_KEY` | glm-4, glm-3-turbo |
| **Azure** | `azure` | `AZURE_OPENAI_API_KEY` | gpt-4, gpt-35-turbo |
| **Ollama** | `ollama` | (none) | llama2, mistral |

## Configuration

All settings can be configured via environment variables in `.env`:

### Required
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

### OpenAI (default)
```bash
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0
```

### Anthropic (Claude)
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

### Google (Gemini)
```bash
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-pro
```

### ZhipuAI (GLM)
```bash
LLM_PROVIDER=zhipuai
ZHIPUAI_API_KEY=your_key_here
ZHIPUAI_MODEL=glm-4
```

### Azure OpenAI
```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo
```

### Ollama (Local - Recommended for zero API costs)
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:3b  # Recommended: fast & lightweight (~1.9GB)
# Other options: llama2, mistral, phi3, qwen2.5-coder:7b
OLLAMA_BASE_URL=http://localhost:11434
```

### Recommended Ollama Models
| Model | Size | Speed |
|-------|------|-------|
| `qwen2.5-coder:3b` | ~1.9GB | Very fast |
| `phi3` | ~2.3GB | Very fast |
| `mistral` | ~4GB | Fast |
| `qwen2.5-coder:7b` | ~4.7GB | Medium |

### Retrieval Configuration
```bash
RETRIEVAL_K=1  # Number of chunks to retrieve (1-3 recommended for speed)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Directory Configuration
```bash
DOCUMENTS_DIR=./documents
CHROMA_DIR=./chroma_db
```

### Logging Configuration
```bash
LOG_DIR=./logs
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Retry Configuration
```bash
MAX_RETRIES=3
RETRY_DELAY=1.0
RETRY_BACKOFF=2.0
```

## Usage (Python API)

```python
from query import ask_question, load_vector_store, create_rag_chain

# Load vector store and create chain
vectorstore = load_vector_store()
rag_chain, retriever = create_rag_chain(vectorstore)

# Ask a question
result = ask_question(rag_chain, retriever, "How do I charge the vehicle?")
print(result['result'])  # Answer
print(result['source_documents'])  # Source chunks
```

## Important Notes

- Run `rag_script.py` to rebuild the vector store after adding/changing PDFs
- The `.env` file must contain the appropriate API key for your chosen provider
- ChromaDB persists the vector store in `chroma_db/` directory
- Logs are written to `logs/query_YYYYMMDD.log` and `logs/rag_script_YYYYMMDD.log`
- Retry logic handles rate limits and temporary errors with exponential backoff
- Embeddings use OpenAI ada-002 by default (works with most providers)
