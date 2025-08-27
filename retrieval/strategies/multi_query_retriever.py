"""
Multi-query retriever that generates multiple queries from the original query and combines results.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from collections import defaultdict

from ..base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class MultiQueryRetriever(BaseRetriever):
    """
    Multi-query retriever that generates multiple queries from the original query
    and combines the results to improve retrieval diversity and coverage.
    """
    
    def __init__(self, 
                 embeddings,
                 vector_store,
                 num_queries: int = 3,
                 query_generation_strategy: str = "synonym",
                 namespace: Optional[str] = None,
                 **kwargs):
        """
        Initialize the multi-query retriever.
        
        Args:
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            num_queries: Number of queries to generate
            query_generation_strategy: Strategy for generating queries ('synonym', 'paraphrase', 'keyword')
            namespace: Optional namespace for the vector store
            **kwargs: Additional configuration parameters
        """
        super().__init__(embeddings, vector_store, **kwargs)
        self.num_queries = num_queries
        self.query_generation_strategy = query_generation_strategy
        self.namespace = namespace
        
        logger.info(f"Initialized MultiQueryRetriever with {num_queries} queries using {query_generation_strategy} strategy")
    
    def generate_synonym_queries(self, query: str) -> List[str]:
        """
        Generate queries using synonym substitution.
        
        Args:
            query: Original query
            
        Returns:
            List of generated queries
        """
        # Simple synonym dictionary (in practice, you'd use a proper thesaurus)
        synonyms = {
            'good': ['excellent', 'great', 'fine', 'nice'],
            'bad': ['poor', 'terrible', 'awful', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'leisurely', 'gradual', 'delayed'],
            'important': ['significant', 'crucial', 'essential', 'vital'],
            'help': ['assist', 'support', 'aid', 'guide'],
            'problem': ['issue', 'trouble', 'difficulty', 'challenge'],
            'solution': ['answer', 'resolution', 'fix', 'remedy']
        }
        
        queries = [query]
        words = query.lower().split()
        
        for i, word in enumerate(words):
            if word in synonyms:
                for synonym in synonyms[word][:2]:  # Use first 2 synonyms
                    new_words = words.copy()
                    new_words[i] = synonym
                    new_query = ' '.join(new_words)
                    queries.append(new_query)
                    
                    if len(queries) >= self.num_queries:
                        break
            if len(queries) >= self.num_queries:
                break
        
        return queries[:self.num_queries]
    
    def generate_paraphrase_queries(self, query: str) -> List[str]:
        """
        Generate queries using paraphrasing techniques.
        
        Args:
            query: Original query
            
        Returns:
            List of generated queries
        """
        queries = [query]
        
        # Simple paraphrasing rules
        paraphrases = [
            # Question variations
            (r'what is (.+)', r'how does \1 work'),
            (r'how to (.+)', r'what is the best way to \1'),
            (r'why (.+)', r'what causes \1'),
            (r'when (.+)', r'at what time \1'),
            (r'where (.+)', r'in what location \1'),
            
            # Statement variations
            (r'i need (.+)', r'i want \1'),
            (r'i want (.+)', r'i am looking for \1'),
            (r'explain (.+)', r'describe \1'),
            (r'tell me about (.+)', r'give me information about \1'),
        ]
        
        import re
        for pattern, replacement in paraphrases:
            if len(queries) >= self.num_queries:
                break
            try:
                new_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                if new_query != query:
                    queries.append(new_query)
            except:
                continue
        
        # If we don't have enough queries, add some keyword-based variations
        if len(queries) < self.num_queries:
            keywords = self.extract_keywords(query)
            for keyword in keywords[:self.num_queries - len(queries)]:
                queries.append(keyword)
        
        return queries[:self.num_queries]
    
    def generate_keyword_queries(self, query: str) -> List[str]:
        """
        Generate queries using keyword extraction.
        
        Args:
            query: Original query
            
        Returns:
            List of generated queries
        """
        keywords = self.extract_keywords(query)
        queries = [query]
        
        # Add individual keywords as queries
        for keyword in keywords[:self.num_queries - 1]:
            queries.append(keyword)
        
        return queries[:self.num_queries]
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        import re
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def generate_queries(self, query: str) -> List[str]:
        """
        Generate multiple queries based on the strategy.
        
        Args:
            query: Original query
            
        Returns:
            List of generated queries
        """
        if self.query_generation_strategy == "synonym":
            return self.generate_synonym_queries(query)
        elif self.query_generation_strategy == "paraphrase":
            return self.generate_paraphrase_queries(query)
        elif self.query_generation_strategy == "keyword":
            return self.generate_keyword_queries(query)
        else:
            logger.warning(f"Unknown query generation strategy: {self.query_generation_strategy}")
            return [query]
    
    def retrieve(self, 
                query: str, 
                k: int = 4,
                **kwargs) -> List[Document]:
        """
        Retrieve documents using multiple queries.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of relevant Document objects
        """
        if not self.validate_query(query):
            return []
        
        try:
            # Get results with scores
            results_with_scores = self.retrieve_with_scores(query, k, **kwargs)
            
            # Return only documents (without scores)
            results = [doc for doc, _ in results_with_scores]
            
            logger.info(f"Retrieved {len(results)} documents using multi-query search")
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-query retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 4,
                           **kwargs) -> List[tuple]:
        """
        Retrieve documents with scores using multiple queries.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of tuples containing (Document, score)
        """
        if not self.validate_query(query):
            return []
        
        try:
            # Generate multiple queries
            queries = self.generate_queries(query)
            logger.debug(f"Generated {len(queries)} queries: {queries}")
            
            # Get namespace from kwargs or use default
            namespace = kwargs.get('namespace', self.namespace)
            
            # Collect results from all queries
            all_results = []
            for i, sub_query in enumerate(queries):
                try:
                    # Get results for this sub-query
                    sub_results = self.vector_store.similarity_search_with_score(
                        query=sub_query,
                        embeddings=self.embeddings,
                        k=k,
                        namespace=namespace
                    )
                    
                    # Adjust scores based on query similarity to original
                    adjusted_results = []
                    for doc, score in sub_results:
                        # Boost score for results from original query
                        if i == 0:
                            adjusted_score = score * 1.2
                        else:
                            adjusted_score = score * 0.8
                        adjusted_results.append((doc, adjusted_score))
                    
                    all_results.extend(adjusted_results)
                    
                except Exception as e:
                    logger.warning(f"Error processing sub-query '{sub_query}': {str(e)}")
                    continue
            
            # Aggregate and deduplicate results
            doc_scores = defaultdict(list)
            for doc, score in all_results:
                content_hash = hash(doc.page_content)
                doc_scores[content_hash].append((doc, score))
            
            # Calculate final scores (average of all scores for each document)
            final_results = []
            for content_hash, doc_score_list in doc_scores.items():
                doc = doc_score_list[0][0]  # Take the first document instance
                avg_score = sum(score for _, score in doc_score_list) / len(doc_score_list)
                final_results.append((doc, avg_score))
            
            # Sort by final score and return top k
            final_results.sort(key=lambda x: x[1], reverse=True)
            final_results = final_results[:k]
            
            logger.info(f"Retrieved {len(final_results)} documents using multi-query search")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in multi-query retrieval with scores: {str(e)}")
            return []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the multi-query retriever.
        
        Returns:
            Dictionary containing the retriever configuration
        """
        config = super().get_config()
        config.update({
            'num_queries': self.num_queries,
            'query_generation_strategy': self.query_generation_strategy,
            'namespace': self.namespace
        })
        return config
