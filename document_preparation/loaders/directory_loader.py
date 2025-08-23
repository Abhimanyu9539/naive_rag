"""
Directory loader for handling multiple file types using LangChain.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from .base_loader import BaseLoader
from ..utils import Document


class DirectoryDocumentLoader(BaseLoader):
    """Loader for directories with multiple file types using LangChain."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the directory loader.
        
        Args:
            encoding: Text encoding to use when reading files
        """
        super().__init__(encoding)
        
        # Define loaders for different file types
        self.loaders = {
            "**/*.txt": (TextLoader, {"encoding": encoding}),
            "**/*.pdf": (PyPDFLoader, {}),
            "**/*.docx": (Docx2txtLoader, {}),
            "**/*.doc": (Docx2txtLoader, {})
        }
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can handle the given directory."""
        return Path(file_path).is_dir()
    
    def load(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not self.can_load(directory_path):
            raise ValueError(f"Path {directory_path} is not a directory")
        
        documents = []
        
        # Load documents using LangChain's DirectoryLoader for each file type
        for glob_pattern, (loader_class, loader_kwargs) in self.loaders.items():
            try:
                langchain_loader = DirectoryLoader(
                    directory_path,
                    glob=glob_pattern,
                    loader_cls=loader_class,
                    loader_kwargs=loader_kwargs
                )
                langchain_docs = langchain_loader.load()
                
                # Convert LangChain documents to our format
                for langchain_doc in langchain_docs:
                    source = langchain_doc.metadata.get('source', 'unknown')
                    document = self._convert_langchain_document(langchain_doc, source)
                    
                    # Add file metadata
                    file_metadata = self._get_file_metadata(source)
                    document.metadata.update(file_metadata)
                    
                    documents.append(document)
                    
            except Exception as e:
                print(f"Error loading files with pattern {glob_pattern}: {e}")
                continue
        
        return documents
    
    def load_with_filters(self, 
                         directory_path: str, 
                         file_types: Optional[List[str]] = None,
                         recursive: bool = True) -> List[Document]:
        """
        Load documents with specific file type filters.
        
        Args:
            directory_path: Path to the directory
            file_types: List of file extensions to include (e.g., ['txt', 'pdf'])
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        
        # Filter loaders based on file types
        filtered_loaders = self.loaders
        if file_types:
            filtered_loaders = {}
            for pattern, (loader_class, loader_kwargs) in self.loaders.items():
                for file_type in file_types:
                    if f"*.{file_type}" in pattern:
                        filtered_loaders[pattern] = (loader_class, loader_kwargs)
                        break
        
        # Load documents using filtered loaders
        for glob_pattern, (loader_class, loader_kwargs) in filtered_loaders.items():
            try:
                # Adjust glob pattern for recursive/non-recursive search
                if not recursive:
                    glob_pattern = glob_pattern.replace("**/", "")
                
                langchain_loader = DirectoryLoader(
                    directory_path,
                    glob=glob_pattern,
                    loader_cls=loader_class,
                    loader_kwargs=loader_kwargs
                )
                langchain_docs = langchain_loader.load()
                
                # Convert LangChain documents to our format
                for langchain_doc in langchain_docs:
                    source = langchain_doc.metadata.get('source', 'unknown')
                    document = self._convert_langchain_document(langchain_doc, source)
                    
                    # Add file metadata
                    file_metadata = self._get_file_metadata(source)
                    document.metadata.update(file_metadata)
                    
                    documents.append(document)
                    
            except Exception as e:
                print(f"Error loading files with pattern {glob_pattern}: {e}")
                continue
        
        return documents
