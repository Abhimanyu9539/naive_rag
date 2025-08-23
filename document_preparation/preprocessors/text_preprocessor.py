"""
Text preprocessor using LangChain components for comprehensive text cleaning.
"""

import re
import unicodedata
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from .base_preprocessor import BasePreprocessor


class TextPreprocessor(BasePreprocessor):
    """Comprehensive text preprocessor for cleaning and normalizing text."""
    
    def __init__(self, 
                 remove_extra_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 remove_html_tags: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_special_chars: bool = False,
                 lowercase: bool = False,
                 remove_numbers: bool = False):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_extra_whitespace: Remove extra whitespace and normalize spacing
            normalize_unicode: Normalize unicode characters
            remove_html_tags: Remove HTML tags from text
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses from text
            remove_special_chars: Remove special characters (keep only alphanumeric and spaces)
            lowercase: Convert text to lowercase
            remove_numbers: Remove numbers from text
        """
        super().__init__()
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_html_tags = remove_html_tags
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with comprehensive cleaning.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return text
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove HTML tags
        if self.remove_html_tags:
            text = self._remove_html_tags(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self._remove_emails(text)
        
        # Remove numbers
        if self.remove_numbers:
            text = self._remove_numbers(text)
        
        # Remove special characters
        if self.remove_special_chars:
            text = self._remove_special_chars(text)
        
        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = self._normalize_whitespace(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Simple HTML tag removal
        html_pattern = re.compile(r'<[^>]+>')
        return html_pattern.sub('', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # URL pattern matching
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        # Email pattern matching
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        # Remove standalone numbers and numbers within words
        number_pattern = re.compile(r'\b\d+\b')
        return number_pattern.sub('', text)
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and spaces."""
        # Keep only letters, numbers, and spaces
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace by removing extra spaces and newlines."""
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
