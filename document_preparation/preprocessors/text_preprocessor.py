"""
Text preprocessor for cleaning and normalizing documents.
"""

import re
import unicodedata
from typing import List, Optional

from .base_preprocessor import BasePreprocessor


class TextPreprocessor(BasePreprocessor):
    """Text preprocessor for cleaning and normalizing documents."""
    
    def __init__(self, 
                 remove_extra_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 remove_special_chars: bool = False,
                 lowercase: bool = False,
                 remove_numbers: bool = False,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_html_tags: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_extra_whitespace: Whether to remove extra whitespace
            normalize_unicode: Whether to normalize unicode characters
            remove_special_chars: Whether to remove special characters
            lowercase: Whether to convert text to lowercase
            remove_numbers: Whether to remove numbers
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_html_tags: Whether to remove HTML tags
        """
        super().__init__()
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html_tags = remove_html_tags
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text by applying various cleaning operations.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Normalize unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Remove HTML tags
        if self.remove_html_tags:
            text = self._remove_html_tags(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self._remove_emails(text)
        
        # Remove special characters
        if self.remove_special_chars:
            text = self._remove_special_chars(text)
        
        # Remove numbers
        if self.remove_numbers:
            text = self._remove_numbers(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = self._remove_extra_whitespace(text)
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKC', text)
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Simple HTML tag removal
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # URL pattern matching
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        # Email pattern matching
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters from text."""
        # Keep alphanumeric characters, spaces, and basic punctuation
        return re.sub(r'[^a-zA-Z0-9\s.,!?;:()"\'-]', '', text)
    
    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace from text."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        return text
    
    def clean_text_basic(self, text: str) -> str:
        """
        Apply basic text cleaning (most common operations).
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Create a basic preprocessor with common settings
        basic_preprocessor = TextPreprocessor(
            remove_extra_whitespace=True,
            normalize_unicode=True,
            remove_special_chars=False,
            lowercase=False,
            remove_numbers=False,
            remove_urls=True,
            remove_emails=True,
            remove_html_tags=True
        )
        return basic_preprocessor.preprocess(text)
