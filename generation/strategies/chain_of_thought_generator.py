"""
Chain of Thought generator strategy with step-by-step reasoning.
"""

import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from ..base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class ChainOfThoughtGenerator(BaseGenerator):
    """
    Chain of Thought generator that implements step-by-step reasoning.
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 **kwargs):
        """
        Initialize the chain of thought generator.
        
        Args:
            llm: Language model instance
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm, **kwargs)
        
        # Set default configuration
        self.max_tokens = kwargs.get('max_tokens', 2000)
        self.temperature = kwargs.get('temperature', 0.3)
        self.include_sources = kwargs.get('include_sources', True)
        self.reasoning_steps = kwargs.get('reasoning_steps', 3)
        self.show_reasoning = kwargs.get('show_reasoning', True)
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "reasoning_steps"],
            template=self._get_prompt_template()
        )
    
    def _get_prompt_template(self) -> str:
        """
        Get the prompt template for chain of thought generation.
        
        Returns:
            Prompt template string
        """
        template = """You are an expert AI assistant that uses step-by-step reasoning to provide accurate answers.

Context Information:
{context}

Question: {question}

Please follow this {reasoning_steps}-step reasoning process:

Step 1: Analyze the question and identify what information is needed
Step 2: Review the provided context and identify relevant information
Step 3: Synthesize the information and form a logical conclusion
Step 4: Provide a comprehensive answer with clear reasoning

Let me think through this step by step:

"""
        
        if self.include_sources:
            template += "\nSources: Please cite specific documents when referencing information."
        
        return template
    
    def _create_structured_context(self, 
                                 query: str,
                                 documents: List[Document]) -> str:
        """
        Create structured context for chain of thought reasoning.
        
        Args:
            query: User's query
            documents: Retrieved documents
            
        Returns:
            Structured context string
        """
        context_parts = [f"Question: {query}\n\n"]
        context_parts.append("Available Information:")
        
        # Organize documents by relevance and source
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', f'Document {i}')
            score = doc.metadata.get('score', 0)
            relevance = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            
            context_parts.append(f"\n{i}. Source: {source} (Relevance: {relevance})")
            context_parts.append(f"   Content: {doc.page_content}")
        
        return "\n".join(context_parts)
    
    def generate(self, 
                query: str,
                retrieved_docs: List[Document],
                **kwargs) -> str:
        """
        Generate a response using chain of thought reasoning.
        
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
            
            # Create structured context
            context = self._create_structured_context(processed_query, processed_docs)
            
            # Create prompt
            prompt = self.prompt_template.format(
                context=context,
                question=processed_query,
                reasoning_steps=self.reasoning_steps
            )
            
            # Generate response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Format the response to show reasoning steps clearly
            if self.show_reasoning:
                formatted_response = self._format_reasoning_response(response_text)
            else:
                # Extract only the final answer
                formatted_response = self._extract_final_answer(response_text)
            
            logger.info(f"Generated chain of thought response for query: {processed_query[:50]}...")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating chain of thought response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def _format_reasoning_response(self, response: str) -> str:
        """
        Format the response to clearly show reasoning steps.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Formatted response with clear reasoning steps
        """
        # Add clear section headers if not present
        if "Step 1:" not in response:
            lines = response.split('\n')
            formatted_lines = []
            
            step_count = 1
            for line in lines:
                if line.strip() and not line.startswith('Step'):
                    formatted_lines.append(f"Step {step_count}: {line}")
                    step_count += 1
                else:
                    formatted_lines.append(line)
            
            response = '\n'.join(formatted_lines)
        
        return response
    
    def _extract_final_answer(self, response: str) -> str:
        """
        Extract only the final answer from the reasoning response.
        
        Args:
            response: Full reasoning response
            
        Returns:
            Final answer only
        """
        # Look for common patterns that indicate the final answer
        answer_indicators = [
            "Final Answer:",
            "Answer:",
            "Conclusion:",
            "Therefore,",
            "In conclusion,"
        ]
        
        lines = response.split('\n')
        answer_start = -1
        
        for i, line in enumerate(lines):
            for indicator in answer_indicators:
                if indicator.lower() in line.lower():
                    answer_start = i
                    break
            if answer_start != -1:
                break
        
        if answer_start != -1:
            return '\n'.join(lines[answer_start:])
        else:
            # If no clear indicator, return the last few lines
            return '\n'.join(lines[-3:])
    
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
        
        # Extract sources and reasoning information
        sources = []
        reasoning_quality = self._assess_reasoning_quality(response)
        
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        
        return {
            'response': response,
            'query': query,
            'num_documents_used': len(retrieved_docs),
            'sources': sources,
            'reasoning_steps': self.reasoning_steps,
            'reasoning_quality': reasoning_quality,
            'generator_type': 'chain_of_thought',
            'config': {
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'include_sources': self.include_sources,
                'reasoning_steps': self.reasoning_steps,
                'show_reasoning': self.show_reasoning
            }
        }
    
    def _assess_reasoning_quality(self, response: str) -> str:
        """
        Assess the quality of reasoning in the response.
        
        Args:
            response: Generated response
            
        Returns:
            Quality assessment string
        """
        step_count = response.lower().count('step')
        if step_count >= self.reasoning_steps:
            return "Excellent"
        elif step_count >= self.reasoning_steps - 1:
            return "Good"
        elif step_count >= 2:
            return "Fair"
        else:
            return "Poor"
