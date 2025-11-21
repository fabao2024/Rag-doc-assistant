# Building a Production-Ready RAG System: From PDFs to Intelligent Q&A

## Introduction

Imagine having a personal AI assistant that can instantly answer questions about any PDF document you throw at it — whether it's a 300-page technical manual, research papers, or legal documents. That's exactly what I built, and in this article, I'll show you how.

**What you'll learn:**
- How to build a Retrieval-Augmented Generation (RAG) system from scratch
- Modern LangChain patterns (LCEL) for production applications
- Best practices for document processing and vector storage
- Common pitfalls and how to avoid them

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines the power of large language models with your own data. Instead of relying solely on the LLM's training data, RAG:

1. **Retrieves** relevant information from your documents
2. **Augments** the LLM's prompt with this context
3. **Generates** accurate, grounded answers

This approach dramatically reduces hallucinations and enables the LLM to answer questions about information it was never trained on.

## The Architecture

```
PDF Documents → Text Chunks → Embeddings → Vector DB → Retrieval → LLM → Answer
```

Let's break down each component:

### 1. Document Loading & Chunking

First, we load PDFs and split them into manageable chunks:

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDFs
loader = PyPDFDirectoryLoader("./documents")
docs_raw = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200  # Prevents context loss at boundaries
)
docs_chunked = splitter.split_documents(docs_raw)
```

**Why chunking matters:** LLMs have token limits. Chunking ensures we can process large documents while maintaining context.

### 2. Embeddings & Vector Storage

Next, we convert text chunks into vector embeddings and store them:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs_chunked,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

**Why ChromaDB?** It's lightweight, fast, and perfect for prototyping. For production, consider Pinecone or Weaviate.

### 3. The Modern RAG Chain (LCEL)

Here's where it gets interesting. LangChain 1.0 introduced LCEL (LangChain Expression Language), a cleaner way to build chains:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer based on the context below.
    
Context:
{context}"""),
    ("human", "{question}")
])

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)

# Use it!
answer = rag_chain.invoke("How do I charge the vehicle?")
```

**Why LCEL?** It's more readable, composable, and supports streaming natively.

## Real-World Results

I tested this system with a 257-page vehicle owner's manual (4MB PDF). Here's what I found:

- **Setup time:** ~30 seconds to process and embed
- **Query time:** 2-3 seconds per question
- **Accuracy:** Consistently accurate with proper source citations
- **Cost:** ~$0.02 per 100 queries (using gpt-3.5-turbo)

### Example Query

**Question:** "What is the vehicle's range?"

**Answer:** 
> The vehicle has a NEDC range of up to 400 km on a single charge. Under WLTP testing conditions, the range is approximately 310-360 km depending on driving conditions and climate control usage.

**Sources:** Pages 15, 23, 47

## Lessons Learned

### 1. Import Hell is Real

LangChain 1.0 reorganized everything. If you see `ModuleNotFoundError`, update your imports:

```python
# ❌ Old (deprecated)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ✅ New (correct)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Use LCEL instead of RetrievalQA
```

### 2. Chunk Size Matters

I experimented with different chunk sizes:
- **500 tokens:** Too small, lost context
- **2000 tokens:** Too large, irrelevant info
- **1000 tokens (sweet spot):** Best balance

### 3. Overlap Prevents Context Loss

Always use overlap (I use 200 characters). Without it, important information at chunk boundaries gets lost.

### 4. LangSmith Errors are Harmless

You might see "Forbidden" errors from LangSmith. These are just telemetry failures and don't affect functionality. Suppress them:

```python
import warnings
warnings.filterwarnings('ignore')
```

## Production Considerations

### Security
- ✅ Never commit `.env` files
- ✅ Use environment variables for API keys
- ✅ Implement rate limiting

### Performance
- ✅ Cache embeddings (ChromaDB handles this)
- ✅ Use async for concurrent queries
- ✅ Consider batch processing for large document sets

### Cost Optimization
- ✅ Use gpt-3.5-turbo for most queries
- ✅ Reserve GPT-4 for complex questions
- ✅ Implement caching for repeated queries

## Next Steps

Want to enhance this system? Try:

1. **Multi-modal RAG:** Add image/table extraction
2. **Hybrid Search:** Combine vector + keyword search
3. **Query Routing:** Route simple questions to cheaper models
4. **Streaming Responses:** Use LCEL's native streaming
5. **Evaluation:** Implement RAG evaluation metrics (RAGAS)

## Conclusion

Building a RAG system is easier than ever with modern tools like LangChain and ChromaDB. The key is understanding the fundamentals:

- Proper chunking strategy
- Quality embeddings
- Efficient retrieval
- Clean prompt engineering

The complete code is available on [GitHub](https://github.com/fabao2024/Rag-doc-assistant). Give it a star if you found it helpful!

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [My GitHub Repo](https://github.com/fabao2024/Rag-doc-assistant)

---

**Questions?** Drop them in the comments below. I'd love to hear about your RAG implementations!

**Connect with me:**
- GitHub: [@fabao2024](https://github.com/fabao2024)
- LinkedIn: [Fabio Pettian](https://www.linkedin.com/in/fabiopettian/)
- Project: [Rag-doc-assistant](https://github.com/fabao2024/Rag-doc-assistant)

*If you enjoyed this article, please clap 👏 and share it with others!*
