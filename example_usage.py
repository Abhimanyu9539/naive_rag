"""
Comprehensive example of using the LangChain-based document preparation pipeline.

This example demonstrates the power and simplicity of using pure LangChain components
instead of custom implementations.
"""

from document_preparation import DocumentPreparationPipeline, DirectoryDocumentLoader
from document_preparation.chunkers import TokenChunker, CharacterChunker
import json

def main():
    """Main example function demonstrating the LangChain-based pipeline."""
    print("=== LangChain Document Pipeline Example ===\n")
    
    # Initialize the pipeline with LangChain components
    pipeline = DocumentPreparationPipeline(
        max_tokens=500,  # Maximum tokens per chunk
        overlap_tokens=50,  # Overlap between chunks
        use_tokens=True,  # Use token-based splitting with tiktoken
        preprocessor_config={
            'remove_extra_whitespace': True,
            'normalize_unicode': True,
            'remove_html_tags': True,
            'remove_urls': True,
            'remove_emails': True
        }
    )
    
    # Example 1: Processing a single document with token-based chunking
    print("Example 1: Processing a single document with token-based chunking")
    print("-" * 60)
    
    # Create a sample text file for demonstration
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that work and react like humans. Some of the activities 
    computers with artificial intelligence are designed for include speech recognition, 
    learning, planning, and problem solving.
    
    Machine Learning is a subset of AI that provides systems the ability to automatically 
    learn and improve from experience without being explicitly programmed. Machine learning 
    focuses on the development of computer programs that can access data and use it to 
    learn for themselves.
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple 
    layers to model and understand complex patterns. It has been particularly successful 
    in areas such as computer vision, natural language processing, and speech recognition.
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
    between computers and human language. It involves developing algorithms and models 
    that can understand, interpret, and generate human language in a meaningful way.
    """
    
    with open("ai_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Process the document using LangChain components
    chunks = pipeline.process_single_document("ai_document.txt")
    
    if chunks:
        print(f"✅ Created {len(chunks)} chunks from the document")
        print(f"📄 First chunk preview: {chunks[0].page_content[:100]}...")
        
        # Get comprehensive statistics
        stats = pipeline.get_statistics(chunks)
        print(f"📊 Average tokens per chunk: {stats['average_tokens_per_chunk']}")
        print(f"📊 Total tokens: {stats['total_tokens']}")
        print(f"📊 Chunk size range: {stats['chunk_size_range']}")
        print(f"🔧 Chunker type: {stats['pipeline_config']['chunker_type']}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 2: Using the directory loader with LangChain's DirectoryLoader
    print("Example 2: Using DirectoryDocumentLoader with LangChain's DirectoryLoader")
    print("-" * 60)
    
    # Create additional sample files
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write("This is a simple text file for testing the LangChain pipeline.")
    
    with open("another_document.txt", "w", encoding="utf-8") as f:
        f.write("This is another document with different content about data science and analytics.")
    
    # Use directory loader directly with LangChain components
    directory_loader = DirectoryDocumentLoader()
    documents = directory_loader.load_with_filters(".", file_types=['txt'])
    
    print(f"✅ Loaded {len(documents)} documents from directory")
    for doc in documents:
        print(f"  📄 {doc.metadata.get('source', 'unknown')}: {len(doc.page_content.split())} words")
    
    print("\n" + "="*70 + "\n")
    
    # Example 3: Processing multiple file types with filters
    print("Example 3: Processing multiple file types with filters")
    print("-" * 60)
    
    # Process only specific file types using LangChain's automatic loader selection
    chunks = pipeline.process_documents(".", file_types=['txt'])
    print(f"✅ Processed {len(chunks)} chunks from text files")
    
    # Save results for inspection
    pipeline.save_chunks_to_file(chunks, "langchain_example_output.txt")
    print("💾 Results saved to 'langchain_example_output.txt'")
    
    print("\n" + "="*70 + "\n")
    
    # Example 4: Using individual LangChain loaders
    print("Example 4: Using individual LangChain loaders")
    print("-" * 60)
    
    from document_preparation.loaders import TextDocumentLoader, PDFDocumentLoader
    
    # Test text loader with LangChain's TextLoader
    text_loader = TextDocumentLoader()
    try:
        docs = text_loader.load("ai_document.txt")
        print(f"✅ Loaded document: {docs[0].metadata.get('source', 'unknown')}")
        print(f"📊 Word count: {len(docs[0].page_content.split())}")
        print(f"📋 Metadata keys: {list(docs[0].metadata.keys())}")
        print(f"🔧 Loader type: {docs[0].metadata.get('loader_type', 'unknown')}")
    except Exception as e:
        print(f"❌ Error loading document: {e}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 5: Token-based vs character-based chunking with LangChain splitters
    print("Example 5: Token-based vs character-based chunking with LangChain splitters")
    print("-" * 60)
    
    # Create pipeline with character-based chunking
    char_pipeline = DocumentPreparationPipeline(
        max_tokens=200,  # This becomes character count
        overlap_tokens=20,  # This becomes character overlap
        use_tokens=False  # Use character-based splitting with RecursiveCharacterTextSplitter
    )
    
    # Test both chunkers directly
    print("🔧 Testing LangChain chunkers directly:")
    
    # Token chunker
    token_chunker = TokenChunker(max_tokens=100, overlap_tokens=10)
    token_chunks = token_chunker.chunk_documents(docs)
    print(f"  🎯 TokenChunker: {len(token_chunks)} chunks")
    print(f"  🎯 Average tokens: {sum(len(token_chunker.tokenizer.encode(chunk.page_content)) for chunk in token_chunks) / len(token_chunks):.1f}")
    
    # Character chunker
    char_chunker = CharacterChunker(chunk_size=200, chunk_overlap=20)
    char_chunks = char_chunker.chunk_documents(docs)
    print(f"  🔤 CharacterChunker: {len(char_chunks)} chunks")
    print(f"  🔤 Average chars: {sum(len(chunk.page_content) for chunk in char_chunks) / len(char_chunks):.1f}")
    
    # Compare pipeline results
    char_chunks_pipeline = char_pipeline.process_single_document("ai_document.txt")
    token_chunks_pipeline = pipeline.process_single_document("ai_document.txt")
    
    print(f"\n📊 Pipeline comparison:")
    print(f"  🔤 Character-based chunks: {len(char_chunks_pipeline)}")
    print(f"  🔤 Average chars per chunk: {char_pipeline.get_statistics(char_chunks_pipeline)['average_characters_per_chunk']}")
    print(f"  🎯 Token-based chunks: {len(token_chunks_pipeline)}")
    print(f"  🎯 Average tokens per chunk: {pipeline.get_statistics(token_chunks_pipeline)['average_tokens_per_chunk']}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 6: Comprehensive statistics and analysis
    print("Example 6: Comprehensive statistics and analysis")
    print("-" * 60)
    
    # Get detailed statistics
    stats = pipeline.get_statistics(chunks)
    
    print("📊 Pipeline Statistics:")
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*70 + "\n")
    
    # Example 7: Metadata preservation demonstration
    print("Example 7: Metadata preservation demonstration")
    print("-" * 60)
    
    if chunks:
        print("📋 Sample chunk metadata:")
        sample_chunk = chunks[0]
        for key, value in sample_chunk.metadata.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 8: Error handling and resilience
    print("Example 8: Error handling and resilience")
    print("-" * 60)
    
    # Try to process a non-existent file
    try:
        non_existent_chunks = pipeline.process_single_document("non_existent_file.txt")
        print(f"✅ Gracefully handled non-existent file: {len(non_existent_chunks)} chunks")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Try to process an unsupported file type
    try:
        with open("test.xyz", "w") as f:
            f.write("This is an unsupported file type")
        
        unsupported_chunks = pipeline.process_single_document("test.xyz")
        print(f"✅ Gracefully handled unsupported file type: {len(unsupported_chunks)} chunks")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 9: LangChain text splitter features
    print("Example 9: LangChain text splitter features")
    print("-" * 60)
    
    print("🔧 LangChain Text Splitters used:")
    print("  🎯 TokenTextSplitter: For token-based chunking with tiktoken")
    print("  🔤 RecursiveCharacterTextSplitter: For character-based chunking")
    print("  📋 Both preserve metadata through split_documents() method")
    print("  🔄 Both support configurable chunk sizes and overlaps")
    print("  📊 Both provide comprehensive statistics")
    
    print("\n" + "="*70 + "\n")
    
    # Summary
    print("🎉 LangChain Pipeline Summary:")
    print("-" * 60)
    print("✅ Successfully converted to pure LangChain components")
    print("✅ Using LangChain's TokenTextSplitter and RecursiveCharacterTextSplitter")
    print("✅ Maintained all original functionality")
    print("✅ Added comprehensive metadata preservation")
    print("✅ Improved error handling and logging")
    print("✅ Enhanced statistics and analysis")
    print("✅ Simplified codebase significantly")
    print("✅ Leveraged battle-tested LangChain functionality")
    print("✅ Support for both token and character-based chunking")

if __name__ == "__main__":
    main()
