"""
Token-based document chunker using LangChain's TokenTextSplitter.
"""

from typing import List
import tiktoken

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from .base_chunker import BaseChunker


class TokenChunker(BaseChunker):
    """Token-based document chunker using LangChain's TokenTextSplitter."""
    
    def __init__(self, 
                 max_tokens: int = 500,
                 overlap_tokens: int = 50,
                 tokenizer_name: str = "cl100k_base"):
        """
        Initialize the token chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlapping tokens between chunks
            tokenizer_name: Tokenizer to use (default: cl100k_base for GPT models)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # Call parent constructor after setting attributes
        super().__init__(chunk_size=max_tokens, chunk_overlap=overlap_tokens)
    
    def _create_text_splitter(self):
        """Create a token-based text splitter."""
        return TokenTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=self.overlap_tokens,
            encoding_name=self.tokenizer_name
        )
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a document using token-based splitting.
        
        Args:
            document: LangChain Document to chunk
            
        Returns:
            List of chunked Document objects
        """
        try:
            # Use LangChain's text splitter to preserve metadata
            chunks = self.text_splitter.split_documents([document])
            
            # Add chunk metadata
            document_id = document.metadata.get('source', 'unknown')
            chunks = self.add_chunk_metadata(chunks, document_id)
            
            return chunks
            
        except Exception as e:
            print(f"Error chunking document: {e}")
            return []
    
    def get_token_count(self, text: str) -> int:
        """
        Get the token count for a text using the configured tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def get_chunk_token_counts(self, chunks: List[Document]) -> List[int]:
        """
        Get token counts for a list of chunks.
        
        Args:
            chunks: List of chunked documents
            
        Returns:
            List of token counts
        """
        return [self.get_token_count(chunk.page_content) for chunk in chunks]
