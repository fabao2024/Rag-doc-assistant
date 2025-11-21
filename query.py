"""
Interactive query script for the RAG system
Usage: .venv\\Scripts\\python.exe query.py "Your question here"
"""
import sys
import os
import warnings
from dotenv import load_dotenv

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# Load environment
load_dotenv()

# Import required modules
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load vector store
print("Loading vector store...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

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

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

# Get question from command line or prompt
if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
else:
    question = input("\nEnter your question: ")

print("\n" + "="*80)
print("QUESTION:")
print("="*80)
print(question)

print("\n" + "="*80)
print("ANSWER:")
print("="*80)

# Get answer
answer = rag_chain.invoke(question)
print(answer)

# Get source documents
source_docs = retriever.invoke(question)

print("\n" + "="*80)
print(f"SOURCE DOCUMENTS ({len(source_docs)} retrieved):")
print("="*80)
for i, doc in enumerate(source_docs, 1):
    print(f"\nSource {i}:")
    print(f"Page: {doc.metadata.get('page', 'N/A')}")
    print(f"Content preview: {doc.page_content[:150]}...")

print("\n" + "="*80)
