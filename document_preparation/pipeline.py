"""
Main pipeline for document preparation.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from .loaders import PDFLoader, TextLoader, DocxLoader, WebLoader, DirectoryDocumentLoader
from .preprocessors import TextPreprocessor
from .chunkers import TokenChunker
from .utils import Document, Chunk, is_supported_file


class DocumentPreparationPipeline:
    """Main pipeline for document preparation."""
    
    def __init__(self, 
                 max_tokens: int = 500,
                 overlap_tokens: int = 50,
                 preprocessor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document preparation pipeline.
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlapping tokens between chunks
            preprocessor_config: Configuration for text preprocessor
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Initialize components
        self.loaders = {
            'pdf': PDFLoader(),
            'txt': TextLoader(),
            'docx': DocxLoader(),
            'web': WebLoader(),
            'directory': DirectoryDocumentLoader()
        }
        
        # Initialize preprocessor with default or custom config
        if preprocessor_config is None:
            preprocessor_config = {
                'remove_extra_whitespace': True,
                'normalize_unicode': True,
                'remove_html_tags': True,
                'remove_urls': True,
                'remove_emails': True
            }
        
        self.preprocessor = TextPreprocessor(**preprocessor_config)
        self.chunker = TokenChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    
    def process_documents(self, 
                         input_path: str, 
                         recursive: bool = True,
                         file_types: Optional[List[str]] = None) -> List[Chunk]:
        """
        Process documents from a directory or single file.
        
        Args:
            input_path: Path to directory or file
            recursive: Whether to search subdirectories recursively
            file_types: List of file extensions to process (e.g., ['pdf', 'txt'])
            
        Returns:
            List of processed chunks
        """
        # Load documents
        documents = self._load_documents(input_path, recursive, file_types)
        
        if not documents:
            print(f"No documents found in {input_path}")
            return []
        
        print(f"Loaded {len(documents)} documents")
        
        # Preprocess documents
        processed_documents = self._preprocess_documents(documents)
        print(f"Preprocessed {len(processed_documents)} documents")
        
        # Chunk documents
        chunks = self._chunk_documents(processed_documents)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def process_single_document(self, file_path: str) -> List[Chunk]:
        """
        Process a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed chunks
        """
        # Load document
        document = self._load_single_document(file_path)
        if not document:
            return []
        
        print(f"Loaded document: {document.title}")
        
        # Preprocess document
        processed_document = self.preprocessor.preprocess_document(document)
        print(f"Preprocessed document: {processed_document.title}")
        
        # Chunk document
        chunks = self.chunker.chunk_document(processed_document)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def _load_documents(self, 
                       input_path: str, 
                       recursive: bool = True,
                       file_types: Optional[List[str]] = None) -> List[Document]:
        """Load documents from the input path."""
        path = Path(input_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")
        
        documents = []
        
        if path.is_file():
            # Single file
            document = self._load_single_document(str(path))
            if document:
                documents.append(document)
        else:
            # Directory - use the new directory loader
            documents = self.loaders['directory'].load_with_filters(
                str(path), 
                file_types=file_types, 
                recursive=recursive
            )
        
        return documents
    
    def _load_single_document(self, file_path: str) -> Optional[Document]:
        """Load a single document."""
        if not is_supported_file(file_path):
            print(f"Unsupported file type: {file_path}")
            return None
        
        try:
            # Determine the appropriate loader
            extension = Path(file_path).suffix.lower()
            
            if extension == '.pdf':
                return self.loaders['pdf'].load(file_path)
            elif extension == '.txt':
                return self.loaders['txt'].load(file_path)
            elif extension in ['.docx', '.doc']:
                return self.loaders['docx'].load(file_path)
            else:
                # Try web loader for URLs
                try:
                    return self.loaders['web'].load(file_path)
                except:
                    print(f"No suitable loader found for: {file_path}")
                    return None
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess all documents."""
        return self.preprocessor.preprocess_documents(documents)
    
    def _chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk all documents."""
        return self.chunker.chunk_documents(documents)
    
    def get_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about the processed chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary containing statistics
        """
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_words = sum(chunk.word_count for chunk in chunks)
        total_chars = sum(chunk.char_count for chunk in chunks)
        
        # Token counts
        tokenizer = self.chunker.tokenizer
        total_tokens = sum(len(tokenizer.encode(chunk.content)) for chunk in chunks)
        
        # Average statistics
        avg_words = total_words / total_chunks
        avg_chars = total_chars / total_chunks
        avg_tokens = total_tokens / total_chunks
        
        # Document sources
        sources = set(chunk.document_id for chunk in chunks)
        
        return {
            'total_chunks': total_chunks,
            'total_words': total_words,
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'average_words_per_chunk': round(avg_words, 2),
            'average_characters_per_chunk': round(avg_chars, 2),
            'average_tokens_per_chunk': round(avg_tokens, 2),
            'unique_documents': len(sources),
            'chunk_size_range': {
                'min_tokens': min(len(tokenizer.encode(chunk.content)) for chunk in chunks),
                'max_tokens': max(len(tokenizer.encode(chunk.content)) for chunk in chunks)
            }
        }
    
    def save_chunks_to_file(self, chunks: List[Chunk], output_file: str):
        """
        Save chunks to a text file for inspection.
        
        Args:
            chunks: List of chunks to save
            output_file: Path to output file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== Chunk {i+1} ===\n")
                f.write(f"Document: {chunk.document_id}\n")
                f.write(f"Chunk ID: {chunk.chunk_id}\n")
                f.write(f"Words: {chunk.word_count}\n")
                f.write(f"Characters: {chunk.char_count}\n")
                f.write(f"Content:\n{chunk.content}\n\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"Saved {len(chunks)} chunks to {output_file}")
