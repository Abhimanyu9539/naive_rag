"""
Contextual generator strategy with enhanced context awareness.
"""

import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from ..base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class ContextualGenerator(BaseGenerator):
    """
    Contextual generator that provides sophisticated context-aware generation.
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 **kwargs):
        """
        Initialize the contextual generator.
        
        Args:
            llm: Language model instance
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm, **kwargs)
        
        # Set default configuration
        self.max_tokens = kwargs.get('max_tokens', 1500)
        self.temperature = kwargs.get('temperature', 0.5)
        self.include_sources = kwargs.get('include_sources', True)
        self.context_window = kwargs.get('context_window', 4000)
        self.enable_reasoning = kwargs.get('enable_reasoning', True)
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "reasoning_instruction"],
            template=self._get_prompt_template()
        )
    
    def _get_prompt_template(self) -> str:
        """
        Get the prompt template for contextual generation.
        
        Returns:
            Prompt template string
        """
        template = """You are an expert AI assistant with deep knowledge and analytical capabilities. 
Your task is to provide comprehensive, accurate, and well-reasoned answers based on the provided context.

Context Information:
{context}

User Question: {question}

{reasoning_instruction}

Please provide a detailed answer that:
1. Directly addresses the user's question
2. Uses specific information from the provided context
3. Acknowledges any limitations or uncertainties
4. Provides additional insights when relevant

Answer:"""
        
        if self.include_sources:
            template += "\n\nSources: Please cite specific documents and page numbers when referencing information."
        
        return template
    
    def _create_enhanced_context(self, 
                                query: str,
                                documents: List[Document]) -> str:
        """
        Create enhanced context with better organization and relevance scoring.
        
        Args:
            query: User's query
            documents: Retrieved documents
            
        Returns:
            Enhanced context string
        """
        context_parts = [f"Query: {query}\n\n"]
        
        # Group documents by source for better organization
        source_groups = {}
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        # Add documents organized by source
        for source, docs in source_groups.items():
            context_parts.append(f"Source: {source}")
            
            for i, doc in enumerate(docs, 1):
                score = doc.metadata.get('score', 0)
                relevance_indicator = "⭐" * min(int(score * 5) + 1, 5) if score > 0 else ""
                
                context_parts.append(f"  Document {i} {relevance_indicator}:")
                context_parts.append(f"  {doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _get_reasoning_instruction(self) -> str:
        """
        Get reasoning instruction based on configuration.
        
        Returns:
            Reasoning instruction string
        """
        if self.enable_reasoning:
            return """Before providing your answer, please:
1. Analyze the relevance of each document to the question
2. Identify the most important information from the context
3. Consider any potential contradictions or gaps in the information
4. Structure your response logically with clear reasoning"""
        else:
            return ""
    
    def generate(self, 
                query: str,
                retrieved_docs: List[Document],
                **kwargs) -> str:
        """
        Generate a response using contextual generation.
        
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
            processed_docs = self.preprocess_documents(
                retrieved_docs, 
                max_length=self.context_window
            )
            
            # Create enhanced context
            context = self._create_enhanced_context(processed_query, processed_docs)
            reasoning_instruction = self._get_reasoning_instruction()
            
            # Create prompt
            prompt = self.prompt_template.format(
                context=context,
                question=processed_query,
                reasoning_instruction=reasoning_instruction
            )
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            logger.info(f"Generated contextual response for query: {processed_query[:50]}...")
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {str(e)}")
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
        
        # Extract detailed source information
        sources = []
        source_scores = {}
        
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown')
            score = doc.metadata.get('score', 0)
            
            if source not in sources:
                sources.append(source)
                source_scores[source] = []
            
            source_scores[source].append(score)
        
        # Calculate average scores per source
        avg_scores = {}
        for source, scores in source_scores.items():
            avg_scores[source] = sum(scores) / len(scores) if scores else 0
        
        return {
            'response': response,
            'query': query,
            'num_documents_used': len(retrieved_docs),
            'sources': sources,
            'source_scores': avg_scores,
            'generator_type': 'contextual',
            'config': {
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'include_sources': self.include_sources,
                'context_window': self.context_window,
                'enable_reasoning': self.enable_reasoning
            }
        }
