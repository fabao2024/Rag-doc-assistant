# Quick Reference Guide

## Common Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

### Running the System

```bash
# Build vector store (first time or when adding new documents)
.venv\Scripts\python.exe rag_script.py

# Query your documents
.venv\Scripts\python.exe query.py "Your question here"

# Interactive mode
.venv\Scripts\python.exe query.py
```

### Testing

```bash
# Verify all imports work
.venv\Scripts\python.exe test_imports.py

# Test vector store
.venv\Scripts\python.exe test_vectorstore.py
```

## Project Files Quick Reference

| File | Purpose |
|------|---------|
| `rag_script.py` | Main RAG pipeline - builds vector store |
| `query.py` | CLI for querying documents |
| `test_imports.py` | Verify dependencies |
| `requirements.txt` | Python dependencies |
| `.env` | Your API keys (gitignored) |
| `.env.example` | Template for .env |
| `documents/` | Place PDFs here |
| `chroma_db/` | Vector database (auto-generated) |

## Configuration Quick Reference

### Chunk Size
```python
# In rag_script.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust for your needs
    chunk_overlap=200
)
```

### LLM Model
```python
# In query.py or rag_script.py
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4"
    temperature=0
)
```

### Retrieval Count
```python
# In query.py
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top 3 chunks
```

## Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| Import errors | Update to new package names (see README) |
| LangSmith errors | Set `LANGCHAIN_TRACING_V2=false` in .env |
| Empty documents | Add PDFs to `documents/` folder |
| API rate limits | Use gpt-3.5-turbo, add delays |

## Publishing Checklist

- [ ] Update README with your info
- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Personalize Medium article
- [ ] Add screenshots
- [ ] Publish Medium article
- [ ] Share on social media

## Useful Links

- [LangChain Docs](https://python.langchain.com/)
- [OpenAI Platform](https://platform.openai.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [GitHub Repo](https://github.com/fabao2024/Rag-doc-assistant)
