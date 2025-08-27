#!/usr/bin/env python3
"""
Simple test script to verify that the retrieval module can be imported correctly.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from retrieval import RetrieverFactory
    print("✅ Successfully imported RetrieverFactory")
    
    from retrieval import BaseRetriever
    print("✅ Successfully imported BaseRetriever")
    
    from retrieval import SimpleRetriever
    print("✅ Successfully imported SimpleRetriever")
    
    from retrieval import HybridRetriever
    print("✅ Successfully imported HybridRetriever")
    
    from retrieval import MultiQueryRetriever
    print("✅ Successfully imported MultiQueryRetriever")
    
    from retrieval import ContextualRetriever
    print("✅ Successfully imported ContextualRetriever")
    
    from retrieval import RerankRetriever
    print("✅ Successfully imported RerankRetriever")
    
    from retrieval import TimeAwareRetriever
    print("✅ Successfully imported TimeAwareRetriever")
    
    from retrieval import EnsembleRetriever
    print("✅ Successfully imported EnsembleRetriever")
    
    print("\n🎉 All imports successful! The retrieval module is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
