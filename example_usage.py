"""
Simple example of using the document preparation pipeline with LangChain loaders.
"""

from document_preparation import DocumentPreparationPipeline, DirectoryDocumentLoader

def main():
    """Main example function."""
    print("=== Document Preparation Example with LangChain ===\n")
    
    # Initialize the pipeline
    pipeline = DocumentPreparationPipeline(
        max_tokens=500,  # Maximum tokens per chunk
        overlap_tokens=50,  # Overlap between chunks
        preprocessor_config={
            'remove_extra_whitespace': True,
            'normalize_unicode': True,
            'remove_html_tags': True,
            'remove_urls': True,
            'remove_emails': True
        }
    )
    
    # Example 1: Process a single document
    print("Example 1: Processing a single document")
    print("-" * 40)
    
    # Create a simple text file for demonstration
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
    """
    
    with open("ai_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Process the document
    chunks = pipeline.process_single_document("ai_document.txt")
    
    if chunks:
        print(f"Created {len(chunks)} chunks from the document")
        print(f"First chunk preview: {chunks[0].content[:100]}...")
        
        # Get statistics
        stats = pipeline.get_statistics(chunks)
        print(f"Average tokens per chunk: {stats['average_tokens_per_chunk']}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Using the directory loader directly
    print("Example 2: Using DirectoryDocumentLoader directly")
    print("-" * 40)
    
    # Create additional sample files
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write("This is a simple text file for testing.")
    
    with open("another_document.txt", "w", encoding="utf-8") as f:
        f.write("This is another document with different content about data science and analytics.")
    
    # Use directory loader directly
    directory_loader = DirectoryDocumentLoader()
    documents = directory_loader.load_with_filters(".", file_types=['txt'])
    
    print(f"Loaded {len(documents)} documents from directory")
    for doc in documents:
        print(f"  - {doc.title}: {doc.word_count} words")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Processing multiple file types with filters
    print("Example 3: Processing multiple file types with filters")
    print("-" * 40)
    
    # Process only specific file types
    chunks = pipeline.process_documents(".", file_types=['txt'])
    print(f"Processed {len(chunks)} chunks from text files")
    
    # Save results for inspection
    pipeline.save_chunks_to_file(chunks, "example_output.txt")
    print("Results saved to 'example_output.txt'")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Using individual LangChain loaders
    print("Example 4: Using individual LangChain loaders")
    print("-" * 40)
    
    from document_preparation.loaders import TextLoader, PDFLoader
    
    # Test text loader
    text_loader = TextLoader()
    try:
        doc = text_loader.load("ai_document.txt")
        print(f"Loaded document: {doc.title}")
        print(f"Word count: {doc.word_count}")
        print(f"Metadata keys: {list(doc.metadata.keys())}")
    except Exception as e:
        print(f"Error loading document: {e}")

if __name__ == "__main__":
    main()
