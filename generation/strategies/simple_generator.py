"""
Simple generator strategy using basic prompt template.
"""

import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from ..base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class SimpleGenerator(BaseGenerator):
    """
    Simple generator that uses a basic prompt template for generation.
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 **kwargs):
        """
        Initialize the simple generator.
        
        Args:
            llm: Language model instance
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm, **kwargs)
        
        # Set default configuration
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.temperature = kwargs.get('temperature', 0.7)
        self.include_sources = kwargs.get('include_sources', True)
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=self._get_prompt_template()
        )
    
    def _get_prompt_template(self) -> str:
        """
        Get the prompt template for simple generation.
        
        Returns:
            Prompt template string
        """
        template = """You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        if self.include_sources:
            template += "\n\nSources: Please cite the relevant documents when providing your answer."
        
        return template
    
    def generate(self, 
                query: str,
                retrieved_docs: List[Document],
                **kwargs) -> str:
        """
        Generate a response using simple prompt template.
        
        Args:
            query: User's query
            retrieved_docs: Retrieved documents
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        if not self.validate_inputs(query, retrieved_docs):
            return "I'm sorry, but I couldn't process your request. Please provide a valid query and ensure documents are available."
        
        try:
            # Preprocess inputs
            processed_query = self.preprocess_query(query)
            processed_docs = self.preprocess_documents(retrieved_docs)
            
            # Create context
            context = self.create_context(processed_query, processed_docs)
            
            # Create prompt
            prompt = self.prompt_template.format(
                context=context,
                question=processed_query
            )
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            logger.info(f"Generated response for query: {processed_query[:50]}...")
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def generate_with_metadata(self, 
                             query: str,
                             retrieved_docs: List[Document],
                             **kwargs) -> Dict[str, Any]:
        """
        Generate a response with additional metadata.
        
        Args:
            query: User's query
            retrieved_docs: Retrieved documents
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing response and metadata
        """
        response = self.generate(query, retrieved_docs, **kwargs)
        
        # Extract sources from documents
        sources = []
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        
        return {
            'response': response,
            'query': query,
            'num_documents_used': len(retrieved_docs),
            'sources': sources,
            'generator_type': 'simple',
            'config': {
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'include_sources': self.include_sources
            }
        }
