"""
RAG with Prompt Caching

This script demonstrates a Retrieval-Augmented Generation (RAG) system using OpenAI and LangChain.
"""
import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from functools import wraps
from typing import Callable, Any

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Create logs directory if it doesn't exist
log_path = Path(LOG_DIR)
log_path.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path / f"rag_script_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Retry Configuration
# ============================================================================
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))


# ============================================================================
# Retry Logic
# ============================================================================
def with_retry(func: Callable) -> Callable:
    """Decorator to add retry logic with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if it's a retryable error
                retryable = any(keyword in error_msg for keyword in [
                    "rate limit", "timeout", "connection", "temporary", "service unavailable",
                    "429", "500", "502", "503", "504"
                ])

                if not retryable or attempt == MAX_RETRIES - 1:
                    raise

                delay = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise last_exception
    return wrapper

# ============================================================================
# Retry Configuration
# ============================================================================
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))


# ============================================================================
# Retry Logic Decorator
# ============================================================================
def with_retry(func: Callable) -> Callable:
    """Decorator to add retry logic with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if it's a retryable error
                retryable = any(keyword in error_msg for keyword in [
                    "rate limit", "timeout", "connection", "temporary", "service unavailable",
                    "429", "500", "502", "503", "504"
                ])

                if not retryable or attempt == MAX_RETRIES - 1:
                    raise

                delay = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise last_exception
    return wrapper

# ============================================================================
# 1. Load Environment Variables
# ============================================================================
logger.info("Loading environment variables...")

# Disable LangSmith tracing completely (avoid 403 errors)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

load_dotenv()

# Verify API key is loaded
from llm_config import get_provider, check_api_key, get_required_api_key

provider = get_provider()
required_key = get_required_api_key(provider)

if required_key and not check_api_key(provider):
    error_msg = (
        f"Provider '{provider}' requires '{required_key}' environment variable. "
        f"Please check your .env file."
    )
    logger.error(error_msg)
    raise ValueError(error_msg)

logger.info(f"Environment variables loaded successfully, provider: {provider}")

# ============================================================================
# 2. Configuration
# ============================================================================
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

logger.info(f"Configuration: documents={DOCUMENTS_DIR}, chroma={CHROMA_DIR}, chunk_size={CHUNK_SIZE}")

# ============================================================================
# 3. Import Required Libraries
# ============================================================================
logger.info("Importing libraries...")
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm_config import get_llm, get_embeddings

logger.info("Libraries imported successfully")

# ============================================================================
# 4. Load and Process Documents
# ============================================================================
logger.info(f"Loading PDFs from {DOCUMENTS_DIR}...")

documents_path = Path(DOCUMENTS_DIR)
if not documents_path.exists():
    error_msg = f"Documents directory not found: {DOCUMENTS_DIR}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

loader = PyPDFDirectoryLoader(str(documents_path))
docs_raw = loader.load()

if not docs_raw:
    error_msg = f"No PDF documents found in {DOCUMENTS_DIR}"
    logger.error(error_msg)
    raise ValueError(error_msg)

logger.info(f"Loaded {len(docs_raw)} document(s)")

# Split documents into chunks
logger.info("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
docs_chunked = splitter.split_documents(docs_raw)

logger.info(f"Split into {len(docs_chunked)} chunks")

# ============================================================================
# 5. Create Vector Store
# ============================================================================
logger.info("Creating embeddings and vector store...")
start_time = datetime.now()

embeddings = get_embeddings()

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs_chunked,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

duration = (datetime.now() - start_time).total_seconds()
logger.info(f"Vector store created in {duration:.2f}s, persisted to {CHROMA_DIR}")

# ============================================================================
# 6. Set Up Custom Prompt Template
# ============================================================================
logger.info("Configuring custom prompt template...")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide detailed and comprehensive answers based on the context.

Context:
{context}"""),
    ("human", "{question}")
])

PROMPT = prompt_template

logger.info("Custom prompt template configured")

# ============================================================================
# 7. Initialize LLM and RAG Chain
# ============================================================================
logger.info("Initializing LLM...")
llm = get_llm()

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

logger.info("RAG chain initialized successfully")

# ============================================================================
# 8. Interactive Query Function
# ============================================================================
@with_retry
def ask_question(question: str):
    """
    Ask a question to the RAG system and display the answer with sources.
    """
    logger.info(f"Processing question: {question[:100]}...")

    # Get the answer
    answer = rag_chain.invoke(question)

    # Get source documents
    source_docs = retriever.invoke(question)

    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
    print("\n" + "="*80)
    print(f"Retrieved {len(source_docs)} source document(s)")
    print("="*80)

    logger.info(f"Answer generated, retrieved {len(source_docs)} sources")

    return {"result": answer, "source_documents": source_docs}

# ============================================================================
# 9. Example Query
# ============================================================================
if __name__ == "__main__":
    logger.info("="*50)
    logger.info("RAG System Ready - Starting example query")
    logger.info("="*50)

    # Example query
    query = "What is the main topic of the documents?"

    print("\n" + "="*80)
    print("RAG SYSTEM READY")
    print("="*80)

    print("\n" + "="*80)
    print("QUESTION:")
    print("="*80)
    print(query)

    result = ask_question(query)

    print("\n" + "="*80)
    print("SOURCE DOCUMENTS:")
    print("="*80)
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\nSource {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

    print("\n" + "="*80)
    print("You can now use ask_question('your question') to query the system")
    print("="*80)

    logger.info("RAG script completed successfully")
