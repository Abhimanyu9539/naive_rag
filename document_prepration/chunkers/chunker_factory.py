import logging
from typing import Dict, Type, Any

from .base_chunker import BaseChunker
from .recursive_character_chunker import RecursiveCharacterChunker
from .character_chunker import CharacterChunker
from .token_chunker import TokenChunker
from .sentence_chunker import SentenceChunker
from .markdown_chunker import MarkdownChunker

logger = logging.getLogger(__name__)


class ChunkerFactory:
    """Factory class for creating different types of chunkers."""
    
    # Registry of available chunkers
    CHUNKERS: Dict[str, Type[BaseChunker]] = {
        'recursive_character': RecursiveCharacterChunker,
        'character': CharacterChunker,
        'token': TokenChunker,
        'sentence': SentenceChunker,
        'markdown': MarkdownChunker
    }
    
    @classmethod
    def get_available_chunkers(cls) -> list:
        """Get list of available chunker types."""
        return list(cls.CHUNKERS.keys())
    
    @classmethod
    def create_chunker(cls, chunker_type: str, **kwargs) -> BaseChunker:
        """
        Create a chunker instance based on the specified type.
        
        Args:
            chunker_type: Type of chunker to create
            **kwargs: Additional arguments to pass to the chunker constructor
            
        Returns:
            BaseChunker instance
            
        Raises:
            ValueError: If chunker_type is not supported
        """
        if chunker_type not in cls.CHUNKERS:
            available = cls.get_available_chunkers()
            raise ValueError(f"Unknown chunker type '{chunker_type}'. Available types: {available}")
        
        try:
            chunker_class = cls.CHUNKERS[chunker_type]
            chunker = chunker_class(**kwargs)
            logger.info(f"Created {chunker_type} chunker with parameters: {kwargs}")
            return chunker
        except Exception as e:
            logger.error(f"Error creating {chunker_type} chunker: {str(e)}")
            raise
    
    @classmethod
    def register_chunker(cls, name: str, chunker_class: Type[BaseChunker]):
        """
        Register a new chunker type.
        
        Args:
            name: Name for the chunker type
            chunker_class: Chunker class to register
        """
        if not issubclass(chunker_class, BaseChunker):
            raise ValueError(f"Chunker class must inherit from BaseChunker")
        
        cls.CHUNKERS[name] = chunker_class
        logger.info(f"Registered new chunker type: {name}")
