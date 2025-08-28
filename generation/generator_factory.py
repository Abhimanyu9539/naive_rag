"""
Factory class for creating different types of generators.
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseLanguageModel

from .base_generator import BaseGenerator
from .strategies.simple_generator import SimpleGenerator
from .strategies.contextual_generator import ContextualGenerator
from .strategies.chain_of_thought_generator import ChainOfThoughtGenerator
from .strategies.multi_agent_generator import MultiAgentGenerator

logger = logging.getLogger(__name__)


class GeneratorFactory:
    """
    Factory class for creating different types of generators.
    """
    
    _generators = {
        'simple': SimpleGenerator,
        'contextual': ContextualGenerator,
        'chain_of_thought': ChainOfThoughtGenerator,
        'multi_agent': MultiAgentGenerator
    }
    
    @classmethod
    def create_generator(cls, 
                        generator_type: str,
                        llm: BaseLanguageModel,
                        **kwargs) -> BaseGenerator:
        """
        Create a generator instance based on the specified type.
        
        Args:
            generator_type: Type of generator to create
            llm: Language model instance
            **kwargs: Additional configuration parameters
            
        Returns:
            Generator instance
            
        Raises:
            ValueError: If generator type is not supported
        """
        generator_type = generator_type.lower()
        
        if generator_type not in cls._generators:
            available_types = list(cls._generators.keys())
            raise ValueError(
                f"Unsupported generator type: {generator_type}. "
                f"Available types: {available_types}"
            )
        
        generator_class = cls._generators[generator_type]
        
        try:
            generator = generator_class(llm=llm, **kwargs)
            logger.info(f"Created {generator_type} generator")
            return generator
        except Exception as e:
            logger.error(f"Error creating {generator_type} generator: {str(e)}")
            raise
    
    @classmethod
    def get_available_generators(cls) -> Dict[str, str]:
        """
        Get a dictionary of available generator types and their descriptions.
        
        Returns:
            Dictionary mapping generator types to descriptions
        """
        return {
            'simple': 'Simple generation using basic prompt template',
            'contextual': 'Context-aware generation with document context',
            'chain_of_thought': 'Step-by-step reasoning generation',
            'multi_agent': 'Multi-agent collaborative generation'
        }
    
    @classmethod
    def register_generator(cls, 
                          name: str, 
                          generator_class: type):
        """
        Register a new generator type.
        
        Args:
            name: Name of the generator type
            generator_class: Generator class to register
        """
        if not issubclass(generator_class, BaseGenerator):
            raise ValueError(f"Generator class must inherit from BaseGenerator")
        
        cls._generators[name.lower()] = generator_class
        logger.info(f"Registered new generator type: {name}")
    
    @classmethod
    def get_generator_config(cls, generator_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a generator type.
        
        Args:
            generator_type: Type of generator
            
        Returns:
            Default configuration dictionary
        """
        generator_type = generator_type.lower()
        
        if generator_type not in cls._generators:
            raise ValueError(f"Unsupported generator type: {generator_type}")
        
        # Default configurations for different generator types
        default_configs = {
            'simple': {
                'max_tokens': 1000,
                'temperature': 0.7,
                'include_sources': True
            },
            'contextual': {
                'max_tokens': 1500,
                'temperature': 0.5,
                'include_sources': True,
                'context_window': 4000
            },
            'chain_of_thought': {
                'max_tokens': 2000,
                'temperature': 0.3,
                'include_sources': True,
                'reasoning_steps': 3
            },
            'multi_agent': {
                'max_tokens': 2000,
                'temperature': 0.6,
                'include_sources': True,
                'num_agents': 3
            }
        }
        
        return default_configs.get(generator_type, {})
