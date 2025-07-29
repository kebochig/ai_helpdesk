"""
Knowledge retrieval agent for searching relevant information from the knowledge base.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from src.models.schemas import (
    HelpDeskRequest,
    ClassificationResult,
    RetrievalResult,
    AgentResponse,
    RequestCategory
)
from src.services.vector_store import vector_store_service
from src.services.llm_service import llm_service

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """Agent responsible for retrieving relevant knowledge from the vector store."""
    
    def __init__(self):
        """Initialize the knowledge agent."""
        self.vector_store = vector_store_service
        self.llm = llm_service
        self.max_results = 5
        self.min_score_threshold = 0.3
    
    def retrieve_knowledge(
        self, 
        request: HelpDeskRequest, 
        classification: ClassificationResult
    ) -> AgentResponse:
        """
        Retrieve relevant knowledge for a classified request.
        
        Args:
            request: The original help desk request
            classification: The classification result
            
        Returns:
            AgentResponse with RetrievalResult
        """
        start_time = time.time()
        
        try:
            # Generate enhanced search queries
            search_queries = self._generate_search_queries(request, classification)
            
            # Perform multiple searches and combine results
            all_results = []
            for query in search_queries:
                results = self.vector_store.search_similar(
                    query=query,
                    k=self.max_results,
                    category_filter=classification.category.value if classification.category != RequestCategory.UNKNOWN else None,
                    score_threshold=self.min_score_threshold
                )
                all_results.extend(zip(results.documents, results.relevance_scores))
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_and_rank_results(all_results)
            
            # Create final retrieval result
            final_documents = [doc for doc, score in unique_results[:self.max_results]]
            final_scores = [score for doc, score in unique_results[:self.max_results]]
            
            retrieval_result = RetrievalResult(
                documents=final_documents,
                relevance_scores=final_scores,
                query=request.request,
                total_results=len(final_documents)
            )
            
            # Enhance with semantic analysis if needed
            if len(final_documents) < 3:
                enhanced_result = self._enhance_with_broader_search(request, classification)
                if enhanced_result.total_results > 0:
                    retrieval_result = self._merge_retrieval_results(retrieval_result, enhanced_result)
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                success=True,
                data=retrieval_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error retrieving knowledge: {str(e)}"
            logger.error(error_msg)
            
            return AgentResponse(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _generate_search_queries(
        self, 
        request: HelpDeskRequest, 
        classification: ClassificationResult
    ) -> List[str]:
        """Generate multiple search queries for comprehensive retrieval."""
        queries = []
        
        # Original request as primary query
        queries.append(request.request)
        
        # Category-specific query
        if classification.category != RequestCategory.UNKNOWN:
            category_query = f"{classification.category.value.replace('_', ' ')} {request.request}"
            queries.append(category_query)
        
        # Keyword-based queries
        if classification.keywords:
            keyword_query = " ".join(classification.keywords)
            queries.append(keyword_query)
            
            # Individual important keywords
            for keyword in classification.keywords[:3]:  # Top 3 keywords
                queries.append(keyword)
        
        # Generate semantic variations using LLM
        try:
            semantic_queries = self._generate_semantic_variations(request.request)
            queries.extend(semantic_queries)
        except Exception as e:
            logger.warning(f"Could not generate semantic variations: {str(e)}")
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in queries:
            if query.lower() not in seen:
                unique_queries.append(query)
                seen.add(query.lower())
        
        return unique_queries[:5]  # Limit to 5 queries to avoid over-searching
    
    def _generate_semantic_variations(self, original_query: str) -> List[str]:
        """Generate semantic variations of the original query."""
        prompt = f"""
        Generate 2-3 alternative ways to express this help desk request, focusing on different technical terms or synonyms:
        
        Original: "{original_query}"
        
        Generate variations that might match different documentation styles. Return only the variations, one per line, no numbering or bullets.
        """
        
        try:
            response = self.llm.generate_response_sync(
                prompt=prompt,
                temperature=0.3
            )
            
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return variations[:3]  # Limit to 3 variations
            
        except Exception as e:
            logger.error(f"Error generating semantic variations: {str(e)}")
            return []
    
    def _deduplicate_and_rank_results(
        self, 
        results: List[tuple]
    ) -> List[tuple]:
        """Deduplicate results and rank by relevance score."""
        # Group by document ID and keep highest score
        doc_scores = {}
        for doc, score in results:
            if doc.id not in doc_scores or score > doc_scores[doc.id][1]:
                doc_scores[doc.id] = (doc, score)
        
        # Sort by score (higher is better)
        ranked_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        
        return ranked_results
    
    def _enhance_with_broader_search(
        self, 
        request: HelpDeskRequest, 
        classification: ClassificationResult
    ) -> RetrievalResult:
        """Perform broader search when initial results are insufficient."""
        try:
            # Extract key terms from request
            key_terms = self.llm.extract_keywords(request.request)
            
            # Search without category filter
            broader_query = " ".join(key_terms[:5])  # Use top 5 keywords
            
            results = self.vector_store.search_similar(
                query=broader_query,
                k=self.max_results,
                category_filter=None,  # No category filter for broader search
                score_threshold=0.2  # Lower threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in broader search: {str(e)}")
            return RetrievalResult(
                documents=[],
                relevance_scores=[],
                query=request.request,
                total_results=0
            )
    
    def _merge_retrieval_results(
        self, 
        primary: RetrievalResult, 
        secondary: RetrievalResult
    ) -> RetrievalResult:
        """Merge two retrieval results, prioritizing the primary."""
        try:
            # Combine documents and scores
            all_docs = primary.documents + secondary.documents
            all_scores = primary.relevance_scores + [score * 0.8 for score in secondary.relevance_scores]  # Reduce secondary scores
            
            # Deduplicate
            combined_results = list(zip(all_docs, all_scores))
            unique_results = self._deduplicate_and_rank_results(combined_results)
            
            # Extract final results
            final_docs = [doc for doc, score in unique_results[:self.max_results]]
            final_scores = [score for doc, score in unique_results[:self.max_results]]
            
            return RetrievalResult(
                documents=final_docs,
                relevance_scores=final_scores,
                query=primary.query,
                total_results=len(final_docs)
            )
            
        except Exception as e:
            logger.error(f"Error merging retrieval results: {str(e)}")
            return primary  # Return primary if merge fails
    
    def evaluate_retrieval_quality(self, retrieval_result: RetrievalResult) -> Dict[str, Any]:
        """
        Evaluate the quality of retrieval results.
        
        Args:
            retrieval_result: The retrieval result to evaluate
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            if not retrieval_result.documents:
                return {
                    "quality_score": 0.0,
                    "coverage": 0.0,
                    "relevance": 0.0,
                    "diversity": 0.0,
                    "recommendation": "No relevant documents found"
                }
            
            # Calculate metrics
            avg_relevance = sum(retrieval_result.relevance_scores) / len(retrieval_result.relevance_scores)
            coverage = min(1.0, len(retrieval_result.documents) / 3)  # Ideal is 3+ documents
            
            # Diversity: check if documents come from different sources
            sources = set(doc.source for doc in retrieval_result.documents)
            diversity = min(1.0, len(sources) / len(retrieval_result.documents))
            
            # Overall quality score
            quality_score = (avg_relevance * 0.5 + coverage * 0.3 + diversity * 0.2)
            
            # Generate recommendation
            if quality_score >= 0.8:
                recommendation = "Excellent retrieval quality"
            elif quality_score >= 0.6:
                recommendation = "Good retrieval quality"
            elif quality_score >= 0.4:
                recommendation = "Moderate retrieval quality - consider escalation"
            else:
                recommendation = "Poor retrieval quality - escalation recommended"
            
            return {
                "quality_score": quality_score,
                "coverage": coverage,
                "relevance": avg_relevance,
                "diversity": diversity,
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval quality: {str(e)}")
            return {
                "quality_score": 0.5,
                "coverage": 0.5,
                "relevance": 0.5,
                "diversity": 0.5,
                "recommendation": "Could not evaluate quality"
            }
    
    def get_related_documents(
        self, 
        primary_documents: List[str], 
        category: RequestCategory,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get documents related to the primary retrieved documents.
        
        Args:
            primary_documents: List of primary document IDs
            category: Request category
            limit: Maximum number of related documents
            
        Returns:
            List of related documents with metadata
        """
        try:
            related_docs = []
            
            # Search for documents in the same category
            category_results = self.vector_store.semantic_search_with_metadata(
                query=category.value.replace('_', ' '),
                filters={"category": category.value},
                k=limit * 2
            )
            
            # Filter out primary documents
            for doc, score, metadata in category_results:
                if doc.id not in primary_documents:
                    related_docs.append({
                        "document": doc,
                        "relevance_score": score,
                        "relationship": "same_category",
                        "metadata": metadata
                    })
                    
                    if len(related_docs) >= limit:
                        break
            
            return related_docs
            
        except Exception as e:
            logger.error(f"Error getting related documents: {str(e)}")
            return []
    
    def update_retrieval_feedback(
        self, 
        query: str, 
        retrieved_docs: List[str], 
        helpful_docs: List[str],
        unhelpful_docs: List[str]
    ) -> bool:
        """
        Update retrieval system based on feedback (placeholder for future ML improvements).
        
        Args:
            query: Original query
            retrieved_docs: Documents that were retrieved
            helpful_docs: Documents marked as helpful
            unhelpful_docs: Documents marked as unhelpful
            
        Returns:
            True if feedback was processed successfully
        """
        try:
            # Log feedback for future improvements
            feedback_data = {
                "query": query,
                "retrieved": retrieved_docs,
                "helpful": helpful_docs,
                "unhelpful": unhelpful_docs,
                "timestamp": time.time()
            }
            
            logger.info(f"Retrieval feedback logged: {feedback_data}")
            
            # In a production system, this would update ML models or ranking algorithms
            return True
            
        except Exception as e:
            logger.error(f"Error processing retrieval feedback: {str(e)}")
            return False


# Global knowledge agent instance
knowledge_agent = KnowledgeAgent()