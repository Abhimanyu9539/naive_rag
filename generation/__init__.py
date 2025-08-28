"""
Generation module for RAG system.

This module provides various text generation strategies for creating responses
based on retrieved documents and user queries.
"""

from .base_generator import BaseGenerator
from .generator_factory import GeneratorFactory
from .strategies.simple_generator import SimpleGenerator
from .strategies.contextual_generator import ContextualGenerator
from .strategies.chain_of_thought_generator import ChainOfThoughtGenerator
from .strategies.multi_agent_generator import MultiAgentGenerator

__all__ = [
    'BaseGenerator',
    'GeneratorFactory',
    'SimpleGenerator',
    'ContextualGenerator',
    'ChainOfThoughtGenerator',
    'MultiAgentGenerator'
]
