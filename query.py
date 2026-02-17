"""
Interactive query script for the RAG system
Usage: .venv\\Scripts\\python.exe query.py "Your question here"
"""
import sys
import os
import logging
import warnings
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
        logging.FileHandler(log_path / f"query_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# Disable LangSmith tracing completely
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Load environment
load_dotenv()

# Import required modules
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm_config import get_llm, get_embeddings, get_provider, check_api_key, get_required_api_key

# Configuration
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

# Retry configuration
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
# Provider Validation
# ============================================================================
def validate_provider_config():
    """Validate that the LLM provider is properly configured."""
    provider = get_provider()
    required_key = get_required_api_key(provider)

    if required_key and not check_api_key(provider):
        error_msg = (
            f"Provider '{provider}' requires '{required_key}' environment variable. "
            f"Please set it in your .env file."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Using LLM provider: {provider}")
    return provider


def load_vector_store():
    """Load the vector store with error handling."""
    chroma_path = Path(CHROMA_DIR)

    if not chroma_path.exists():
        error_msg = f"Vector store not found at {CHROMA_DIR}. Please run rag_script.py first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    logger.info(f"Vector store loaded from {CHROMA_DIR}")
    return vectorstore


def create_rag_chain(vectorstore):
    """Create the RAG chain with configuration."""
    # Initialize LLM
    llm = get_llm()

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide detailed and comprehensive answers based on the context.

Context:
{context}"""),
        ("human", "{question}")
    ])

    # Create retriever with configurable k
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # Helper function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    provider = get_provider()
    logger.debug(f"RAG chain created with provider={provider}, k={RETRIEVAL_K}")
    return rag_chain, retriever


@with_retry
def ask_question_with_retry(rag_chain, retriever, question: str):
    """Ask a question with retry logic."""
    logger.info(f"Processing question: {question[:100]}...")
    start_time = datetime.now()

    # Get the answer
    answer = rag_chain.invoke(question)

    # Get source documents
    source_docs = retriever.invoke(question)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Answer generated in {duration:.2f}s, retrieved {len(source_docs)} sources")

    return {"result": answer, "source_documents": source_docs}


def ask_question(rag_chain, retriever, question: str):
    """
    Ask a question to the RAG system with error handling.
    Returns dict with 'result' and 'source_documents'.
    """
    try:
        return ask_question_with_retry(rag_chain, retriever, question)

    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str:
            error_msg = "Rate limit exceeded"
            logger.error(error_msg)
            raise RuntimeError(f"{error_msg}. Please wait a moment and try again.") from e
        logger.error(f"Error processing question: {e}")
        raise RuntimeError(f"Error processing question: {e}") from e


def display_result(result: dict):
    """Display the query result in a formatted way."""
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(result['result'])

    source_docs = result['source_documents']
    print("\n" + "="*80)
    print(f"SOURCE DOCUMENTS ({len(source_docs)} retrieved):")
    print("="*80)
    for i, doc in enumerate(source_docs, 1):
        print(f"\nSource {i}:")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content preview: {doc.page_content[:150]}...")

    print("\n" + "="*80)


def main():
    """Main entry point."""
    logger.info("="*50)
    logger.info("RAG Query Interface Started")
    logger.info("="*50)

    # Validate provider configuration
    try:
        validate_provider_config()
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)

    # Get question from command line or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("\nEnter your question: ").strip()

    if not question:
        logger.warning("Empty question provided")
        print("Error: Empty question provided.")
        sys.exit(1)

    print("\n" + "="*80)
    print("QUESTION:")
    print("="*80)
    print(question)

    try:
        # Load vector store
        print("\nLoading vector store...")
        vectorstore = load_vector_store()

        # Create RAG chain
        print("Initializing RAG chain...")
        rag_chain, retriever = create_rag_chain(vectorstore)

        # Get answer with error handling
        print("Processing question...")
        result = ask_question(rag_chain, retriever, question)

        # Display result
        display_result(result)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nError: {e}")
        sys.exit(1)

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)

    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

    logger.info("Query completed successfully")


if __name__ == "__main__":
    main()
