"""
Main application for the AI Help Desk System.
"""
import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

from config.settings import settings
from src.models.schemas import HelpDeskRequest, HelpDeskResponse
from src.workflows.helpdesk_workflow import helpdesk_workflow
from src.services.document_processor import document_processor
from src.services.vector_store import vector_store_service


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('helpdesk.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HelpDeskSystem:
    """Main Help Desk System orchestrator."""
    
    def __init__(self):
        """Initialize the help desk system."""
        self.workflow = helpdesk_workflow
        self.document_processor = document_processor
        self.vector_store = vector_store_service
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the help desk system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing AI Help Desk System...")
            
            # Validate settings
            settings.validate_settings()
            logger.info("Settings validated successfully")
            
            # Initialize knowledge base
            await self._initialize_knowledge_base()
            
            self.is_initialized = True
            logger.info("AI Help Desk System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize help desk system: {str(e)}")
            return False
    
    def initialize_sync(self) -> bool:
        """
        Synchronous version of initialize.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing AI Help Desk System...")
            
            # Validate settings
            settings.validate_settings()
            logger.info("Settings validated successfully")
            
            # Initialize knowledge base
            self._initialize_knowledge_base_sync()
            
            self.is_initialized = True
            logger.info("AI Help Desk System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize help desk system: {str(e)}")
            return False
    
    async def _initialize_knowledge_base(self):
        """Initialize the knowledge base."""
        try:
            logger.info("Loading knowledge base documents...")
            
            # Load all documents from knowledge base directory
            documents = self.document_processor.load_all_documents()
            
            if documents:
                # Add documents to vector store
                success = self.vector_store.add_documents(documents)
                if success:
                    logger.info(f"Successfully loaded {len(documents)} documents into knowledge base")
                else:
                    logger.warning("Failed to add some documents to vector store")
            else:
                logger.warning("No documents found in knowledge base directory")
                logger.info("You can add documents to the 'data/knowledge_base' directory")
            
            # Get vector store statistics
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Knowledge base stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    def _initialize_knowledge_base_sync(self):
        """Synchronous version of knowledge base initialization."""
        try:
            logger.info("Loading knowledge base documents...")
            
            # Load all documents from knowledge base directory
            documents = self.document_processor.load_all_documents()
            
            if documents:
                # Add documents to vector store
                success = self.vector_store.add_documents(documents)
                if success:
                    logger.info(f"Successfully loaded {len(documents)} documents into knowledge base")
                else:
                    logger.warning("Failed to add some documents to vector store")
            else:
                logger.warning("No documents found in knowledge base directory")
                logger.info("You can add documents to the 'data/knowledge_base' directory")
            
            # Get vector store statistics
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Knowledge base stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    async def process_request(self, request_text: str, user_id: str = None) -> HelpDeskResponse:
        """
        Process a help desk request.
        
        Args:
            request_text: The user's request text
            user_id: Optional user identifier
            
        Returns:
            HelpDeskResponse with the result
        """
        if not self.is_initialized:
            raise RuntimeError("Help desk system not initialized. Call initialize() first.")
        
        try:
            # Create request object
            request = HelpDeskRequest(
                id=str(uuid.uuid4()),
                user_id=user_id,
                request=request_text,
                timestamp=datetime.now()
            )
            
            logger.info(f"Processing request {request.id}: {request_text[:100]}...")
            
            # Process through workflow
            response = await self.workflow.process_request(request)
            
            logger.info(f"Request {request.id} processed in {response.processing_time:.2f}s")
            logger.info(f"Category: {response.category.value}, Confidence: {response.confidence:.2f}")
            logger.info(f"Escalation: {'Yes' if response.escalation.should_escalate else 'No'}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise
    
    def process_request_sync(self, request_text: str, user_id: str = None) -> HelpDeskResponse:
        """
        Synchronous version of process_request.
        
        Args:
            request_text: The user's request text
            user_id: Optional user identifier
            
        Returns:
            HelpDeskResponse with the result
        """
        if not self.is_initialized:
            raise RuntimeError("Help desk system not initialized. Call initialize_sync() first.")
        
        try:
            # Create request object
            request = HelpDeskRequest(
                id=str(uuid.uuid4()),
                user_id=user_id,
                request=request_text,
                timestamp=datetime.now()
            )
            
            logger.info(f"Processing request {request.id}: {request_text[:100]}...")
            
            # Process through workflow
            response = self.workflow.process_request_sync(request)
            
            logger.info(f"Request {request.id} processed in {response.processing_time:.2f}s")
            logger.info(f"Category: {response.category.value}, Confidence: {response.confidence:.2f}")
            logger.info(f"Escalation: {'Yes' if response.escalation.should_escalate else 'No'}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise
    
    def batch_process_requests(self, requests: List[Dict[str, str]]) -> List[HelpDeskResponse]:
        """
        Process multiple requests in batch.
        
        Args:
            requests: List of request dictionaries with 'request' and optional 'user_id' keys
            
        Returns:
            List of HelpDeskResponse objects
        """
        if not self.is_initialized:
            raise RuntimeError("Help desk system not initialized. Call initialize_sync() first.")
        
        responses = []
        
        for i, req_data in enumerate(requests):
            try:
                logger.info(f"Processing batch request {i+1}/{len(requests)}")
                
                response = self.process_request_sync(
                    request_text=req_data['request'],
                    user_id=req_data.get('user_id')
                )
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error processing batch request {i+1}: {str(e)}")
                # Create error response
                from src.models.schemas import EscalationDecision, PriorityLevel, RequestCategory
                error_response = HelpDeskResponse(
                    request_id=str(uuid.uuid4()),
                    category=RequestCategory.UNKNOWN,
                    response=f"Error processing request: {str(e)}",
                    confidence=0.0,
                    knowledge_sources=[],
                    escalation=EscalationDecision(
                        should_escalate=True,
                        reasoning=f"Processing error: {str(e)}",
                        priority=PriorityLevel.HIGH,
                        suggested_department="it_support",
                        estimated_complexity="error"
                    ),
                    processing_time=0.0
                )
                responses.append(error_response)
        
        return responses
    
    def refresh_knowledge_base(self) -> bool:
        """
        Refresh the knowledge base with updated documents.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Refreshing knowledge base...")
            
            # Reset vector store
            success = self.vector_store.reset_collection()
            if not success:
                logger.error("Failed to reset vector store collection")
                return False
            
            # Reload documents
            self._initialize_knowledge_base_sync()
            
            logger.info("Knowledge base refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status and metrics.
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Get workflow metrics
            workflow_metrics = self.workflow.get_workflow_metrics()
            
            return {
                "initialized": self.is_initialized,
                "timestamp": datetime.now().isoformat(),
                "knowledge_base": vector_stats,
                "workflow_metrics": workflow_metrics,
                "configuration": {
                    "model": settings.system_config.llm.model_name,
                    "temperature": settings.system_config.llm.temperature,
                    "escalation_threshold": settings.system_config.escalation_threshold,
                    "min_confidence_threshold": settings.system_config.min_confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "initialized": self.is_initialized,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


def create_sample_requests() -> List[Dict[str, str]]:
    """Create sample requests for testing."""
    return [
        {
            "request": "I forgot my password and can't log into my computer. How do I reset it?",
            "user_id": "user001"
        },
        {
            "request": "I need to install Microsoft Office on my new laptop. Can you help?",
            "user_id": "user002"
        },
        {
            "request": "My internet connection is not working. I can't access any websites.",
            "user_id": "user003"
        },
        {
            "request": "I think I received a phishing email. It looks suspicious and asks for my login details.",
            "user_id": "user004"
        },
        {
            "request": "My printer is not working. It shows an error message but I can't read it clearly.",
            "user_id": "user005"
        }
    ]


def main():
    """Main function to demonstrate the help desk system."""
    # Initialize system
    system = HelpDeskSystem()
    
    print("=== AI Help Desk System ===")
    print("Initializing system...")
    
    if not system.initialize_sync():
        print("Failed to initialize system. Check logs for details.")
        return
    
    print("System initialized successfully!")
    
    # Get system status
    status = system.get_system_status()
    print(f"\nKnowledge Base: {status['knowledge_base']['document_count']} documents loaded")
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Enter help desk requests (type 'quit' to exit):")
    
    while True:
        try:
            request_text = input("\nYour request: ").strip()
            
            if request_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not request_text:
                continue
            
            response = system.process_request_sync(request_text)
            
            print(f"\nCategory: {response.category.value}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"\nResponse:\n{response.response}")
            
            if response.escalation.should_escalate:
                print(f"\n⚠️  This request will be escalated to: {response.escalation.suggested_department}")
                print(f"Priority: {response.escalation.priority.value}")
                print(f"Reason: {response.escalation.reasoning}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nThank you for using the AI Help Desk System!")


if __name__ == "__main__":
    main()