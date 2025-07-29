"""
Vector store service for knowledge base management.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.models.schemas import KnowledgeDocument, RetrievalResult
from src.services.llm_service import llm_service
from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector store operations."""
    
    def __init__(self):
        """Initialize the vector store service."""
        self.embeddings = llm_service.embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.system_config.vector_store.chunk_size,
            chunk_overlap=settings.system_config.vector_store.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store: Optional[Chroma] = None
        self._initialize_chroma_client()
    
    def _initialize_chroma_client(self):
        """Initialize ChromaDB client."""
        try:
            # Ensure persist directory exists
            os.makedirs(settings.system_config.vector_store.persist_directory, exist_ok=True)
            
            # Initialize Chroma vector store
            self.vector_store = Chroma(
                collection_name=settings.system_config.vector_store.collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.system_config.vector_store.persist_directory,
                client_settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[KnowledgeDocument]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of knowledge documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return True
            
            # Convert to LangChain documents and split
            langchain_docs = []
            for doc in documents:
                # Create base document
                base_doc = Document(
                    page_content=doc.content,
                    metadata={
                        "id": doc.id,
                        "title": doc.title,
                        "category": doc.category.value if doc.category else None,
                        "tags": doc.tags,
                        "source": doc.source,
                        "last_updated": doc.last_updated.isoformat()
                    }
                )
                
                # Split document into chunks
                chunks = self.text_splitter.split_documents([base_doc])
                
                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": f"{doc.id}_chunk_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                
                langchain_docs.extend(chunks)
            
            # Add to vector store
            self.vector_store.add_documents(langchain_docs)
            self.vector_store.persist()
            
            logger.info(f"Added {len(langchain_docs)} document chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def search_similar(
        self, 
        query: str, 
        k: int = 5,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> RetrievalResult:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            category_filter: Optional category to filter by
            score_threshold: Minimum similarity score
            
        Returns:
            RetrievalResult with documents and scores
        """
        try:
            # Prepare filter if category specified
            where_filter = {}
            if category_filter:
                where_filter["category"] = category_filter
            
            # Perform similarity search with scores
            if where_filter:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, where=where_filter
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= score_threshold
            ]
            
            # Convert to KnowledgeDocument objects
            documents = []
            scores = []
            
            for doc, score in filtered_results:
                knowledge_doc = KnowledgeDocument(
                    id=doc.metadata.get("id", "unknown"),
                    title=doc.metadata.get("title", "Untitled"),
                    content=doc.page_content,
                    category=doc.metadata.get("category"),
                    tags=doc.metadata.get("tags", []),
                    source=doc.metadata.get("source", "unknown"),
                    last_updated=doc.metadata.get("last_updated", "unknown")
                )
                documents.append(knowledge_doc)
                scores.append(float(score))
            
            result = RetrievalResult(
                documents=documents,
                relevance_scores=scores,
                query=query,
                total_results=len(documents)
            )
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query}")
            logger.info(result)
            return result
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return RetrievalResult(
                documents=[],
                relevance_scores=[],
                query=query,
                total_results=0
            )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "document_count": count,
                "collection_name": settings.system_config.vector_store.collection_name,
                "embedding_model": settings.system_config.vector_store.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"document_count": 0}
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete by metadata filter
            for doc_id in document_ids:
                self.vector_store.delete(where={"id": doc_id})
            
            self.vector_store.persist()
            logger.info(f"Deleted {len(document_ids)} documents from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset the entire collection (delete all documents).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Reset the collection
            self.vector_store._client.reset()
            
            # Reinitialize
            self._initialize_chroma_client()
            
            logger.info("Vector store collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False
    
    def update_document(self, document: KnowledgeDocument) -> bool:
        """
        Update an existing document in the vector store.
        
        Args:
            document: Updated document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing document
            self.delete_documents([document.id])
            
            # Add updated document
            return self.add_documents([document])
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False
    
    def semantic_search_with_metadata(
        self, 
        query: str,
        filters: Dict[str, Any] = None,
        k: int = 5
    ) -> List[Tuple[KnowledgeDocument, float, Dict[str, Any]]]:
        """
        Advanced semantic search with metadata filtering.
        
        Args:
            query: Search query
            filters: Metadata filters
            k: Number of results
            
        Returns:
            List of tuples (document, score, metadata)
        """
        try:
            # Perform search with filters
            where_filter = filters or {}
            results = self.vector_store.similarity_search_with_score(
                query, k=k, where=where_filter
            )
            
            # Convert results
            output = []
            for doc, score in results:
                knowledge_doc = KnowledgeDocument(
                    id=doc.metadata.get("id", "unknown"),
                    title=doc.metadata.get("title", "Untitled"),
                    content=doc.page_content,
                    category=doc.metadata.get("category"),
                    tags=doc.metadata.get("tags", []),
                    source=doc.metadata.get("source", "unknown"),
                    last_updated=doc.metadata.get("last_updated", "unknown")
                )
                output.append((knowledge_doc, float(score), doc.metadata))
            
            return output
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []


# Global vector store service instance
vector_store_service = VectorStoreService()