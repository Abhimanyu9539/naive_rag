#!/usr/bin/env python3
"""
Debug script to test document addition step by step.
"""

import os
import sys
import logging
from typing import List
from langchain.schema import Document
import dotenv

dotenv.load_dotenv()

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the components
from embeddings.embedder_factory import EmbedderFactory
from vector_stores.pinecone_store import PineconeStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_documents() -> List[Document]:
    """Create test documents."""
    return [
        Document(
            page_content="This is a test document about machine learning.",
            metadata={"source": "test.txt", "topic": "ml"}
        ),
        Document(
            page_content="This is another test document about AI.",
            metadata={"source": "test2.txt", "topic": "ai"}
        )
    ]


def main():
    """Main debug function."""
    print("Debug Test - Document Addition")
    print("=" * 40)
    
    # Check environment variables
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_ENVIRONMENT"):
        print("❌ Missing Pinecone environment variables")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OpenAI API key")
        return
    
    try:
        # Step 1: Create embeddings
        print("Step 1: Creating embeddings...")
        embeddings = EmbedderFactory.get_embedder("openai")
        print("✅ Embeddings created successfully")
        
        # Step 2: Create vector store
        print("Step 2: Creating vector store...")
        vector_store = PineconeStore(
            index_name="debug-test-index",
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        print("✅ Vector store created successfully")
        
        # Step 3: Create index
        print("Step 3: Creating index...")
        dimension = embeddings.get_embedding_dimensions()
        print(f"   Embedding dimension: {dimension}")
        vector_store.create_index(dimension=dimension)
        print("✅ Index created successfully")
        
        # Step 4: Create test documents
        print("Step 4: Creating test documents...")
        documents = create_test_documents()
        print(f"   Created {len(documents)} documents")
        for i, doc in enumerate(documents):
            print(f"   Document {i+1}: {doc.page_content[:50]}...")
        print("✅ Documents created successfully")
        
        # Step 5: Test document embedding
        print("Step 5: Testing document embedding...")
        try:
            embedded_docs = embeddings.embed_documents(documents)
            print(f"   Successfully embedded {len(embedded_docs)} documents")
            print("✅ Document embedding successful")
        except Exception as e:
            print(f"   ❌ Document embedding failed: {str(e)}")
            return
        
        # Step 6: Add documents to vector store
        print("Step 6: Adding documents to vector store...")
        try:
            vector_store.add_documents(documents, embeddings, namespace="debug")
            print("✅ Documents added successfully")
        except Exception as e:
            print(f"   ❌ Adding documents failed: {str(e)}")
            return
        
        print("\n🎉 All steps completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Error in debug test: {str(e)}")


if __name__ == "__main__":
    main()
