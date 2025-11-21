"""
Test script to verify all LangChain imports work correctly
"""

print("Testing LangChain imports...")
print("="*80)

try:
    print("1. Testing langchain_community.document_loaders...")
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    print("   ✓ PyPDFDirectoryLoader imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("2. Testing langchain_text_splitters...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("   ✓ RecursiveCharacterTextSplitter imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("3. Testing langchain_community.vectorstores...")
    from langchain_community.vectorstores import Chroma
    print("   ✓ Chroma imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("4. Testing langchain_openai...")
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    print("   ✓ OpenAIEmbeddings and ChatOpenAI imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("5. Testing langchain_core.prompts...")
    from langchain_core.prompts import ChatPromptTemplate
    print("   ✓ ChatPromptTemplate imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("6. Testing langchain_core.output_parsers...")
    from langchain_core.output_parsers import StrOutputParser
    print("   ✓ StrOutputParser imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("7. Testing langchain_core.runnables...")
    from langchain_core.runnables import RunnablePassthrough
    print("   ✓ RunnablePassthrough imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("="*80)
print("\n✅ ALL IMPORTS SUCCESSFUL!")
print("\nYou can now use these imports in your notebook:")
print("-" * 80)
print("""
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
""")
print("-" * 80)
