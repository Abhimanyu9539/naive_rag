"""
Main pipeline for document preparation using LangChain components.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from .loaders import (
    TextDocumentLoader, PDFDocumentLoader, DocxDocumentLoader, 
    WebDocumentLoader, DirectoryDocumentLoader
)
from .preprocessors import TextPreprocessor
from .chunkers import TokenChunker, CharacterChunker
from .utils import is_supported_file, get_document_statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreparationPipeline:
    """Main pipeline for document preparation using LangChain components."""
    
    def __init__(self, 
                 max_tokens: int = 500,
                 overlap_tokens: int = 50,
                 use_tokens: bool = True,
                 preprocessor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document preparation pipeline.
        
        Args:
            max_tokens: Maximum tokens/characters per chunk
            overlap_tokens: Overlapping tokens/characters between chunks
            use_tokens: Whether to use token-based chunking (True) or character-based (False)
            preprocessor_config: Configuration for text preprocessor
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.use_tokens = use_tokens
        
        # Initialize components
        self.loaders = {
            'pdf': PDFDocumentLoader(),
            'txt': TextDocumentLoader(),
            'docx': DocxDocumentLoader(),
            'web': WebDocumentLoader(),
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
        
        # Initialize appropriate chunker based on configuration
        if self.use_tokens:
            self.chunker = TokenChunker(
                max_tokens=max_tokens, 
                overlap_tokens=overlap_tokens
            )
        else:
            self.chunker = CharacterChunker(
                chunk_size=max_tokens,  # Use as character count
                chunk_overlap=overlap_tokens  # Use as character overlap
            )
        
        logger.info(f"Initialized pipeline with max_tokens={max_tokens}, "
                   f"overlap_tokens={overlap_tokens}, use_tokens={use_tokens}")
    
    def process_documents(self, 
                         input_path: str, 
                         recursive: bool = True,
                         file_types: Optional[List[str]] = None) -> List[Document]:
        """
        Process documents from a directory or single file.
        
        Args:
            input_path: Path to directory or file
            recursive: Whether to search subdirectories recursively
            file_types: List of file extensions to process (e.g., ['pdf', 'txt'])
            
        Returns:
            List of processed LangChain Document chunks
        """
        # Load documents
        documents = self._load_documents(input_path, recursive, file_types)
        
        if not documents:
            logger.warning(f"No documents found in {input_path}")
            return []
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Preprocess documents
        processed_documents = self._preprocess_documents(documents)
        logger.info(f"Preprocessed {len(processed_documents)} documents")
        
        # Chunk documents using LangChain text splitters
        chunks = self._chunk_documents(processed_documents)
        logger.info(f"Created {len(chunks)} chunks using {'token' if self.use_tokens else 'character'}-based chunking")
        
        return chunks
    
    def process_single_document(self, file_path: str) -> List[Document]:
        """
        Process a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed LangChain Document chunks
        """
        # Load document
        documents = self._load_single_document(file_path)
        if not documents:
            return []
        
        logger.info(f"Loaded document: {documents[0].metadata.get('source', 'unknown')}")
        
        # Preprocess document
        processed_documents = self.preprocessor.preprocess_documents(documents)
        logger.info(f"Preprocessed document")
        
        # Chunk document using LangChain text splitters
        chunks = self.chunker.chunk_documents(processed_documents)
        logger.info(f"Created {len(chunks)} chunks using {'token' if self.use_tokens else 'character'}-based chunking")
        
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
            documents = self._load_single_document(str(path))
        else:
            # Directory - use the directory loader
            documents = self.loaders['directory'].load_with_filters(
                str(path), 
                file_types=file_types, 
                recursive=recursive
            )
        
        return documents
    
    def _load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document."""
        if not is_supported_file(file_path):
            logger.warning(f"Unsupported file type: {file_path}")
            return []
        
        try:
            # Determine the appropriate loader
            extension = Path(file_path).suffix.lower()
            
            if extension == '.pdf':
                return self.loaders['pdf'].load(file_path)
            elif extension in ['.txt', '.md', '.rst', '.log']:
                return self.loaders['txt'].load(file_path)
            elif extension in ['.docx', '.doc']:
                return self.loaders['docx'].load(file_path)
            else:
                # Try web loader for URLs
                if self.loaders['web'].can_load(file_path):
                    return self.loaders['web'].load(file_path)
                else:
                    logger.warning(f"No suitable loader found for: {file_path}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess all documents."""
        return self.preprocessor.preprocess_documents(documents)
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk all documents using LangChain text splitters."""
        return self.chunker.chunk_documents(documents)
    
    def get_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the processed chunks.
        
        Args:
            chunks: List of LangChain Document chunks to analyze
            
        Returns:
            Dictionary containing statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_words': 0,
                'total_characters': 0,
                'total_tokens': 0,
                'average_words_per_chunk': 0,
                'average_characters_per_chunk': 0,
                'average_tokens_per_chunk': 0,
                'unique_sources': 0,
                'file_type_distribution': {},
                'source_distribution': {},
                'pipeline_config': {
                    'max_tokens': self.max_tokens,
                    'overlap_tokens': self.overlap_tokens,
                    'use_tokens': self.use_tokens,
                    'chunker_type': 'TokenChunker' if self.use_tokens else 'CharacterChunker'
                }
            }
        
        # Basic statistics
        stats = get_document_statistics(chunks)
        
        # Add chunking-specific statistics
        if self.use_tokens:
            # Token-based statistics
            tokenizer = self.chunker.tokenizer
            total_tokens = sum(len(tokenizer.encode(chunk.page_content)) for chunk in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            
            stats.update({
                'total_tokens': total_tokens,
                'average_tokens_per_chunk': round(avg_tokens, 2),
                'chunk_size_range': {
                    'min_tokens': min(len(tokenizer.encode(chunk.page_content)) for chunk in chunks),
                    'max_tokens': max(len(tokenizer.encode(chunk.page_content)) for chunk in chunks)
                }
            })
        else:
            # Character-based statistics
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            avg_chars = total_chars / len(chunks) if chunks else 0
            
            stats.update({
                'total_characters': total_chars,
                'average_characters_per_chunk': round(avg_chars, 2),
                'chunk_size_range': {
                    'min_chars': min(len(chunk.page_content) for chunk in chunks),
                    'max_chars': max(len(chunk.page_content) for chunk in chunks)
                }
            })
        
        # Add pipeline configuration
        stats['pipeline_config'] = {
            'max_tokens': self.max_tokens,
            'overlap_tokens': self.overlap_tokens,
            'use_tokens': self.use_tokens,
            'chunker_type': 'TokenChunker' if self.use_tokens else 'CharacterChunker'
        }
        
        return stats
    
    def save_chunks_to_file(self, chunks: List[Document], output_file: str):
        """
        Save chunks to a text file for inspection.
        
        Args:
            chunks: List of LangChain Document chunks to save
            output_file: Path to output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== LangChain Document Pipeline Output ===\n\n")
                f.write(f"Total chunks: {len(chunks)}\n")
                f.write(f"Pipeline config: {self.get_statistics(chunks)['pipeline_config']}\n\n")
                f.write("=" * 60 + "\n\n")
                
                for i, chunk in enumerate(chunks):
                    f.write(f"=== Chunk {i+1} ===\n")
                    f.write(f"Source: {chunk.metadata.get('source', 'unknown')}\n")
                    f.write(f"File type: {chunk.metadata.get('file_type', 'unknown')}\n")
                    f.write(f"Chunk ID: {chunk.metadata.get('chunk_id', f'chunk_{i:04d}')}\n")
                    f.write(f"Words: {len(chunk.page_content.split())}\n")
                    f.write(f"Characters: {len(chunk.page_content)}\n")
                    
                    if self.use_tokens:
                        tokenizer = self.chunker.tokenizer
                        tokens = len(tokenizer.encode(chunk.page_content))
                        f.write(f"Tokens: {tokens}\n")
                    
                    f.write(f"Content:\n{chunk.page_content}\n\n")
                    f.write("-" * 50 + "\n\n")
            
            logger.info(f"Saved {len(chunks)} chunks to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving chunks to file: {e}")
    
    def run_pipeline(self, 
                    input_path: str, 
                    file_types: Optional[List[str]] = None,
                    recursive: bool = True) -> List[Document]:
        """
        Run the complete document processing pipeline.
        
        Args:
            input_path: Path to file, directory, or URL
            file_types: List of file extensions to process
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of processed LangChain Document chunks
        """
        logger.info(f"Starting pipeline for: {input_path}")
        
        # Process documents
        chunks = self.process_documents(input_path, recursive, file_types)
        
        logger.info(f"Pipeline completed. Created {len(chunks)} chunks")
        return chunks
