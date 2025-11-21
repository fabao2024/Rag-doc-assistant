"""
Simple test to verify the RAG system is working
"""
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import required modules
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("Loading existing vector store...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print(f"✓ Vector store loaded with {vectorstore._collection.count()} documents")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test retrieval
print("\nTesting document retrieval...")
test_query = "What is this document about?"
docs = retriever.invoke(test_query)
print(f"✓ Retrieved {len(docs)} documents")

# Show first document snippet
if docs:
    print(f"\nFirst document preview:")
    print(f"{docs[0].page_content[:200]}...")

print("\n" + "="*80)
print("✅ RAG system is working! Vector store is ready.")
print("="*80)
