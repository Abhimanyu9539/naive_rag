"""
Multi-agent generator strategy with collaborative generation.
"""

import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from ..base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class MultiAgentGenerator(BaseGenerator):
    """
    Multi-agent generator that simulates collaborative generation using multiple specialized agents.
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 **kwargs):
        """
        Initialize the multi-agent generator.
        
        Args:
            llm: Language model instance
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm, **kwargs)
        
        # Set default configuration
        self.max_tokens = kwargs.get('max_tokens', 2000)
        self.temperature = kwargs.get('temperature', 0.6)
        self.include_sources = kwargs.get('include_sources', True)
        self.num_agents = kwargs.get('num_agents', 3)
        self.show_agent_contributions = kwargs.get('show_agent_contributions', True)
        
        # Define agent roles
        self.agent_roles = self._define_agent_roles()
        
        # Create prompt templates for each agent
        self.agent_prompts = self._create_agent_prompts()
    
    def _define_agent_roles(self) -> Dict[str, str]:
        """
        Define the roles and responsibilities of each agent.
        
        Returns:
            Dictionary mapping agent names to their roles
        """
        return {
            'researcher': 'Analyzes the context and extracts key information',
            'analyst': 'Evaluates the relevance and credibility of information',
            'synthesizer': 'Combines insights and creates the final response'
        }
    
    def _create_agent_prompts(self) -> Dict[str, PromptTemplate]:
        """
        Create prompt templates for each agent.
        
        Returns:
            Dictionary mapping agent names to their prompt templates
        """
        prompts = {}
        
        # Researcher agent prompt
        researcher_template = """You are a Research Agent. Your role is to analyze the provided context and extract key information relevant to the user's question.

Context:
{context}

Question: {question}

Your task:
1. Identify the most relevant pieces of information from the context
2. Extract key facts, data, and insights
3. Note any important details that directly address the question
4. Highlight any gaps or limitations in the available information

Research Findings:"""
        
        prompts['researcher'] = PromptTemplate(
            input_variables=["context", "question"],
            template=researcher_template
        )
        
        # Analyst agent prompt
        analyst_template = """You are an Analysis Agent. Your role is to evaluate the research findings and assess their relevance and credibility.

Research Findings:
{research_findings}

Question: {question}

Your task:
1. Evaluate the relevance of each finding to the question
2. Assess the credibility and reliability of the information
3. Identify any potential biases or limitations
4. Prioritize the findings by importance and relevance

Analysis:"""
        
        prompts['analyst'] = PromptTemplate(
            input_variables=["research_findings", "question"],
            template=analyst_template
        )
        
        # Synthesizer agent prompt
        synthesizer_template = """You are a Synthesis Agent. Your role is to combine the research and analysis to create a comprehensive, well-structured response.

Research Findings:
{research_findings}

Analysis:
{analysis}

Question: {question}

Your task:
1. Combine the research findings and analysis
2. Create a comprehensive, well-structured response
3. Ensure the answer directly addresses the user's question
4. Provide clear reasoning and cite sources when appropriate

Final Response:"""
        
        prompts['synthesizer'] = PromptTemplate(
            input_variables=["research_findings", "analysis", "question"],
            template=synthesizer_template
        )
        
        return prompts
    
    def generate(self, 
                query: str,
                retrieved_docs: List[Document],
                **kwargs) -> str:
        """
        Generate a response using multi-agent collaboration.
        
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
            
            # Step 1: Research Agent
            research_prompt = self.agent_prompts['researcher'].format(
                context=context,
                question=processed_query
            )
            research_response = self.llm.invoke(research_prompt)
            research_findings = research_response.content if hasattr(research_response, 'content') else str(research_response)
            
            # Step 2: Analysis Agent
            analysis_prompt = self.agent_prompts['analyst'].format(
                research_findings=research_findings,
                question=processed_query
            )
            analysis_response = self.llm.invoke(analysis_prompt)
            analysis = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
            
            # Step 3: Synthesis Agent
            synthesis_prompt = self.agent_prompts['synthesizer'].format(
                research_findings=research_findings,
                analysis=analysis,
                question=processed_query
            )
            synthesis_response = self.llm.invoke(synthesis_prompt)
            final_response = synthesis_response.content if hasattr(synthesis_response, 'content') else str(synthesis_response)
            
            # Format the response based on configuration
            if self.show_agent_contributions:
                formatted_response = self._format_multi_agent_response(
                    research_findings, analysis, final_response
                )
            else:
                formatted_response = final_response
            
            logger.info(f"Generated multi-agent response for query: {processed_query[:50]}...")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating multi-agent response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def _format_multi_agent_response(self, 
                                   research_findings: str,
                                   analysis: str,
                                   final_response: str) -> str:
        """
        Format the response to show contributions from each agent.
        
        Args:
            research_findings: Output from research agent
            analysis: Output from analysis agent
            final_response: Output from synthesis agent
            
        Returns:
            Formatted multi-agent response
        """
        formatted_parts = [
            "🤖 Multi-Agent Collaborative Response\n",
            "=" * 50,
            "\n🔍 RESEARCH AGENT FINDINGS:",
            research_findings,
            "\n" + "=" * 30,
            "\n📊 ANALYSIS AGENT EVALUATION:",
            analysis,
            "\n" + "=" * 30,
            "\n🎯 SYNTHESIS AGENT FINAL RESPONSE:",
            final_response
        ]
        
        return "\n".join(formatted_parts)
    
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
        
        # Extract sources and agent information
        sources = []
        agent_contributions = {}
        
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        
        # Track agent contributions
        for agent_name, role in self.agent_roles.items():
            agent_contributions[agent_name] = {
                'role': role,
                'contribution_type': 'collaborative'
            }
        
        return {
            'response': response,
            'query': query,
            'num_documents_used': len(retrieved_docs),
            'sources': sources,
            'num_agents': self.num_agents,
            'agent_contributions': agent_contributions,
            'generator_type': 'multi_agent',
            'config': {
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'include_sources': self.include_sources,
                'num_agents': self.num_agents,
                'show_agent_contributions': self.show_agent_contributions
            }
        }
