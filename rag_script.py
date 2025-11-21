"""
RAG with Prompt Caching

This script demonstrates a Retrieval-Augmented Generation (RAG) system using OpenAI and LangChain.
"""

import os
from dotenv import load_dotenv

# ============================================================================
# 1. Load Environment Variables
# ============================================================================
print("Loading environment variables...")
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

print("✓ Environment variables loaded successfully\n")

# ============================================================================
# 2. Import Required Libraries
# ============================================================================
print("Importing libraries...")
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("✓ Libraries imported successfully\n")

# ============================================================================
# 3. Load and Process Documents
# ============================================================================
print("Loading PDFs from documents directory...")
loader = PyPDFDirectoryLoader("./documents")
docs_raw = loader.load()

print(f"✓ Loaded {len(docs_raw)} document(s)\n")

# Split documents into chunks
print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs_chunked = splitter.split_documents(docs_raw)

print(f"✓ Split into {len(docs_chunked)} chunks\n")

# ============================================================================
# 4. Create Vector Store
# ============================================================================
print("Creating embeddings and vector store...")
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs_chunked,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✓ Vector store created successfully\n")

# ============================================================================
# 5. Set Up Custom Prompt Template
# ============================================================================
print("Configuring custom prompt template...")
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

print("✓ Custom prompt template configured\n")

# ============================================================================
# 6. Initialize LLM and RAG Chain
# ============================================================================
print("Initializing LLM and RAG chain...")
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

print("✓ RAG chain initialized successfully\n")

# ============================================================================
# 7. Interactive Query Function
# ============================================================================
def ask_question(question: str):
    """
    Ask a question to the RAG system and display the answer with sources.
    """
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
    
    return {"result": answer, "source_documents": source_docs}

# ============================================================================
# 8. Example Query
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RAG SYSTEM READY")
    print("="*80)
    
    # Example query
    query = "What is the main topic of the documents?"
    
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
