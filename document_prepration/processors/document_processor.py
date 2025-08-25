"""
This module contains the DocumentProcessor class, which is used to process documents loaded by the loaders.
"""

from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
import re
import logging

class DocumentProcessor:
    """
    This class is used to process documents loaded by the loaders.
    It provides functionality for text cleaning and basic preprocessing.
    """
    
    def __init__(self, 
                 remove_extra_whitespace: bool = True,
                 remove_special_chars: bool = True,
                 lowercase: bool = False,
                 remove_numbers: bool = False,
                 remove_urls: bool = False,
                 remove_emails: bool = False):
        """
        Initialize the DocumentProcessor.
        
        Args:
            remove_extra_whitespace (bool): Whether to remove extra whitespace
            remove_special_chars (bool): Whether to remove special characters
            lowercase (bool): Whether to convert text to lowercase
            remove_numbers (bool): Whether to remove numbers
            remove_urls (bool): Whether to remove URLs
            remove_emails (bool): Whether to remove email addresses
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean the text based on the configured preprocessing options.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            self.logger.debug("Empty text provided for cleaning")
            return ""
        
        original_length = len(text)
        self.logger.debug(f"Cleaning text of length {original_length}")
        
        # Remove URLs if requested
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            self.logger.debug("Removed URLs from text")
        
        # Remove email addresses if requested
        if self.remove_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
            self.logger.debug("Removed email addresses from text")
        
        # Remove numbers if requested
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
            self.logger.debug("Removed numbers from text")
        
        # Remove special characters if requested
        if self.remove_special_chars:
            # Keep alphanumeric, spaces, and basic punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
            self.logger.debug("Removed special characters from text")
        
        # Remove extra whitespace if requested
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            self.logger.debug("Normalized whitespace in text")
        
        # Convert to lowercase if requested
        if self.lowercase:
            text = text.lower()
            self.logger.debug("Converted text to lowercase")
        
        final_length = len(text)
        self.logger.debug(f"Text cleaning completed. Original length: {original_length}, Final length: {final_length}")
        
        return text
    
    def process_documents(self, documents: List[Document], 
                         add_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a list of documents by cleaning text and adding metadata.
        
        Args:
            documents (List[Document]): List of documents to process
            add_metadata (Dict[str, Any], optional): Additional metadata to add to all documents
            
        Returns:
            List[Document]: Processed documents
        """
        self.logger.info(f"Starting to process {len(documents)} documents")
        processed_documents = []
        
        for i, doc in enumerate(documents):
            try:
                self.logger.debug(f"Processing document {i+1}/{len(documents)}")
                
                # Clean the text content
                cleaned_content = self.clean_text(doc.page_content)
                
                # Create new document with cleaned content
                processed_doc = Document(
                    page_content=cleaned_content,
                    metadata=doc.metadata.copy()
                )
                
                # Add custom metadata if provided
                if add_metadata:
                    self.logger.debug(f"Adding custom metadata to document {i+1}")
                    processed_doc.metadata.update(add_metadata)
                
                # Add processing metadata
                processed_doc.metadata.update({
                    'processed': True,
                    'original_length': len(doc.page_content),
                    'processed_length': len(cleaned_content),
                    'document_index': i,
                    'preprocessing_applied': {
                        'remove_extra_whitespace': self.remove_extra_whitespace,
                        'remove_special_chars': self.remove_special_chars,
                        'lowercase': self.lowercase,
                        'remove_numbers': self.remove_numbers,
                        'remove_urls': self.remove_urls,
                        'remove_emails': self.remove_emails
                    }
                })
                
                processed_documents.append(processed_doc)
                self.logger.debug(f"Successfully processed document {i+1}")
                
            except Exception as e:
                self.logger.warning(f"Error processing document {i}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_documents)} out of {len(documents)} documents")
        return processed_documents
    
    def filter_documents(self, documents: List[Document],
                        min_length: int = 10,
                        max_length: Optional[int] = None,
                        filter_keywords: Optional[List[str]] = None) -> List[Document]:
        """
        Filter documents based on various criteria.
        
        Args:
            documents (List[Document]): List of documents to filter
            min_length (int): Minimum length of document content
            max_length (int, optional): Maximum length of document content
            filter_keywords (List[str], optional): Keywords to filter out documents containing them
            
        Returns:
            List[Document]: Filtered documents
        """
        self.logger.info(f"Starting to filter {len(documents)} documents with criteria: min_length={min_length}, max_length={max_length}, filter_keywords={filter_keywords}")
        filtered_documents = []
        
        for i, doc in enumerate(documents):
            content = doc.page_content
            content_length = len(content)
            
            # Check minimum length
            if content_length < min_length:
                self.logger.debug(f"Document {i+1} filtered out due to length {content_length} < {min_length}")
                continue
            
            # Check maximum length
            if max_length and content_length > max_length:
                self.logger.debug(f"Document {i+1} filtered out due to length {content_length} > {max_length}")
                continue
            
            # Check for filter keywords
            if filter_keywords:
                content_lower = content.lower()
                if any(keyword.lower() in content_lower for keyword in filter_keywords):
                    self.logger.debug(f"Document {i+1} filtered out due to containing filter keywords")
                    continue
            
            filtered_documents.append(doc)
        
        self.logger.info(f"Filtered {len(documents)} documents to {len(filtered_documents)} documents")
        return filtered_documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the documents.
        
        Args:
            documents (List[Document]): List of documents to analyze
            
        Returns:
            Dict[str, Any]: Statistics about the documents
        """
        self.logger.info(f"Calculating statistics for {len(documents)} documents")
        
        if not documents:
            self.logger.warning("No documents provided for statistics calculation")
            return {
                'total_documents': 0,
                'total_characters': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0
            }
        
        lengths = [len(doc.page_content) for doc in documents]
        
        stats = {
            'total_documents': len(documents),
            'total_characters': sum(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }
        
        self.logger.info(f"Statistics calculated: {stats}")
        return stats
    
    def update_metadata(self, documents: List[Document],
                       metadata_updates: Dict[str, Any]) -> List[Document]:
        """
        Update metadata for all documents.
        
        Args:
            documents (List[Document]): List of documents to update
            metadata_updates (Dict[str, Any]): Metadata to add/update
            
        Returns:
            List[Document]: Documents with updated metadata
        """
        self.logger.info(f"Updating metadata for {len(documents)} documents with updates: {metadata_updates}")
        updated_documents = []
        
        for i, doc in enumerate(documents):
            updated_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, **metadata_updates}
            )
            updated_documents.append(updated_doc)
            self.logger.debug(f"Updated metadata for document {i+1}")
        
        self.logger.info(f"Successfully updated metadata for {len(updated_documents)} documents")
        return updated_documents
