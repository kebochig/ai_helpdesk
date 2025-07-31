"""
LangGraph workflow for orchestrating the multi-agent help desk system.
"""
import time
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.models.schemas import (
    WorkflowState,
    HelpDeskRequest,
    HelpDeskResponse,
    RequestCategory
)
from src.agents.classifier_agent import classifier_agent
from src.agents.knowledge_agent import knowledge_agent
from src.agents.response_agent import response_agent
from src.agents.escalation_agent import escalation_agent

logger = logging.getLogger(__name__)


class HelpDeskWorkflow:
    """LangGraph workflow for processing help desk requests."""
    
    def __init__(self):
        """Initialize the workflow."""
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.workflow = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        graph = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        graph.add_node("classify", self._classify_node)
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge_node)
        graph.add_node("generate_response", self._generate_response_node)
        graph.add_node("determine_escalation", self._determine_escalation_node)
        graph.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow edges
        graph.set_entry_point("classify")
        
        # Classification -> Knowledge Retrieval (if successful)
        graph.add_conditional_edges(
            "classify",
            self._should_continue_after_classification,
            {
                "continue": "retrieve_knowledge",
                "error": "handle_error"
            }
        )
        
        # Knowledge Retrieval -> Response Generation
        graph.add_conditional_edges(
            "retrieve_knowledge",
            self._should_continue_after_retrieval,
            {
                "continue": "generate_response",
                "error": "handle_error"
            }
        )
        
        # Response Generation -> Escalation Decision
        graph.add_conditional_edges(
            "generate_response",
            self._should_continue_after_response,
            {
                "continue": "determine_escalation",
                "error": "handle_error"
            }
        )
        
        # Escalation Decision -> End
        graph.add_edge("determine_escalation", END)
        graph.add_edge("handle_error", END)
        
        return graph
    
    async def process_request(
        self, 
        request: HelpDeskRequest,
        config: Dict[str, Any] = None
    ) -> HelpDeskResponse:
        """
        Process a help desk request through the workflow.
        
        Args:
            request: The help desk request to process
            config: Optional configuration for the workflow
            
        Returns:
            HelpDeskResponse with the final result
        """
        start_time = time.time()
        
        try:
            # Initialize workflow state
            initial_state = WorkflowState(
                request=request,
                classification=None,
                retrieval_result=None,
                response=None,
                escalation=None,
                errors=[]
            )
            
            # Run the workflow
            config = config or {"configurable": {"thread_id": request.id}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Create the final response
            processing_time = time.time() - start_time
            
            return HelpDeskResponse(
                request_id=request.id,
                category=final_state.classification.category if final_state.classification else RequestCategory.UNKNOWN,
                response=final_state.response or "Unable to generate response",
                confidence=final_state.classification.confidence if final_state.classification else 0.0,
                knowledge_sources=[doc.source for doc in final_state.retrieval_result.documents] if final_state.retrieval_result else [],
                escalation=final_state.escalation,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing request {request.id}: {str(e)}")
            
            # Return error response
            from src.models.schemas import EscalationDecision, PriorityLevel
            error_escalation = EscalationDecision(
                should_escalate=True,
                reasoning=f"Workflow error: {str(e)}",
                priority=PriorityLevel.HIGH,
                suggested_department="it_support",
                estimated_complexity="error"
            )
            
            return HelpDeskResponse(
                request_id=request.id,
                category=RequestCategory.UNKNOWN,
                response="I apologize, but I encountered an error processing your request. I'm escalating this to our support team for immediate assistance.",
                confidence=0.0,
                knowledge_sources=[],
                escalation=error_escalation,
                processing_time=processing_time
            )
    
    def process_request_sync(
        self,
        request: HelpDeskRequest,
        config: Dict[str, Any] = None
    ) -> HelpDeskResponse:
        """
        Synchronous version of process_request.
        
        Args:
            request: The help desk request to process
            config: Optional configuration for the workflow
            
        Returns:
            HelpDeskResponse with the final result
        """
        start_time = time.time()
        
        try:
            # Initialize workflow state
            state = WorkflowState(
                request=request,
                classification=None,
                retrieval_result=None,
                response=None,
                escalation=None,
                errors=[]
            )
            
            # Execute workflow steps synchronously
            state = self._classify_node(state)
            if state.errors:
                state = self._handle_error_node(state)
            else:
                state = self._retrieve_knowledge_node(state)
                if state.errors:
                    state = self._handle_error_node(state)
                else:
                    state = self._generate_response_node(state)
                    if state.errors:
                        state = self._handle_error_node(state)
                    else:
                        state = self._determine_escalation_node(state)
            
            # Create the final response
            processing_time = time.time() - start_time
            
            return HelpDeskResponse(
                request_id=request.id,
                category=state.classification.category if state.classification else RequestCategory.UNKNOWN,
                response=state.response or "Unable to generate response",
                confidence=state.classification.confidence if state.classification else 0.0,
                knowledge_sources=[doc.source for doc in state.retrieval_result.documents] if state.retrieval_result else [],
                escalation=state.escalation,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing request {request.id}: {str(e)}")
            
            # Return error response
            from src.models.schemas import EscalationDecision, PriorityLevel
            error_escalation = EscalationDecision(
                should_escalate=True,
                reasoning=f"Workflow error: {str(e)}",
                priority=PriorityLevel.HIGH,
                suggested_department="it_support",
                estimated_complexity="error"
            )
            
            return HelpDeskResponse(
                request_id=request.id,
                category=RequestCategory.UNKNOWN,
                response="I apologize, but I encountered an error processing your request. I'm escalating this to our support team for immediate assistance.",
                confidence=0.0,
                knowledge_sources=[],
                escalation=error_escalation,
                processing_time=processing_time
            )
    
    def _classify_node(self, state: WorkflowState) -> WorkflowState:
        """Classification node."""
        try:
            logger.info(f"Classifying request: {state.request.id}")
            
            result = classifier_agent.classify_request(state.request)
            
            if result.success:
                state.classification = result.data
                logger.info(f"Request classified as: {state.classification.category.value} (confidence: {state.classification.confidence:.2f})")
            else:
                state.errors.append(f"Classification failed: {result.error}")
                logger.error(f"Classification failed for request {state.request.id}: {result.error}")
            
            return state
            
        except Exception as e:
            error_msg = f"Error in classification node: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            return state
    
    def _retrieve_knowledge_node(self, state: WorkflowState) -> WorkflowState:
        """Knowledge retrieval node."""
        try:
            logger.info(f"Retrieving knowledge for request: {state.request.id}")
            
            result = knowledge_agent.retrieve_knowledge(state.request, state.classification)
            
            if result.success:
                state.retrieval_result = result.data
                logger.info(f"Retrieved {state.retrieval_result.total_results} relevant documents")
            else:
                state.errors.append(f"Knowledge retrieval failed: {result.error}")
                logger.error(f"Knowledge retrieval failed for request {state.request.id}: {result.error}")
            
            return state
            
        except Exception as e:
            error_msg = f"Error in knowledge retrieval node: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            return state
    
    def _generate_response_node(self, state: WorkflowState) -> WorkflowState:
        """Response generation node."""
        try:
            logger.info(f"Generating response for request: {state.request.id}")
            
            result = response_agent.generate_response(
                state.request,
                state.classification,
                state.retrieval_result
            )
            
            if result.success:
                state.response = result.data
                logger.info(f"Response generated successfully")
            else:
                state.errors.append(f"Response generation failed: {result.error}")
                logger.error(f"Response generation failed for request {state.request.id}: {result.error}")
            
            return state
            
        except Exception as e:
            error_msg = f"Error in response generation node: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            return state
    
    def _determine_escalation_node(self, state: WorkflowState) -> WorkflowState:
        """Escalation determination node."""
        try:
            logger.info(f"Determining escalation for request: {state.request.id}")
            
            # Evaluate response quality if response exists
            response_quality = None
            if state.response:
                response_quality = response_agent.evaluate_response_quality(
                    state.response, state.classification
                )
            
            result = escalation_agent.determine_escalation(
                state.request,
                state.classification,
                state.retrieval_result,
                response_quality
            )
            
            if result.success:
                state.escalation = result.data
                logger.info(f"Escalation decision: {'Escalate' if state.escalation.should_escalate else 'No escalation'} (Priority: {state.escalation.priority.value})")
            else:
                state.errors.append(f"Escalation determination failed: {result.error}")
                logger.error(f"Escalation determination failed for request {state.request.id}: {result.error}")
                # Use the fallback escalation from the agent response
                if hasattr(result, 'data') and result.data:
                    state.escalation = result.data
            
            return state
            
        except Exception as e:
            error_msg = f"Error in escalation determination node: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            return state
    
    def _handle_error_node(self, state: WorkflowState) -> WorkflowState:
        """Error handling node."""
        try:
            logger.error(f"Handling errors for request {state.request.id}: {state.errors}")
            
            # Generate fallback response
            error_summary = "; ".join(state.errors[-3:])  # Last 3 errors
            state.response = f"I apologize, but I encountered some difficulties processing your request ({error_summary}). I'm escalating this to our support team for immediate assistance."
            
            # Create escalation decision for errors
            from src.models.schemas import EscalationDecision, PriorityLevel
            state.escalation = EscalationDecision(
                should_escalate=True,
                reasoning=f"System errors encountered: {error_summary}",
                priority=PriorityLevel.HIGH,
                suggested_department="it_support",
                estimated_complexity="system_error"
            )
            
            # Provide minimal classification if missing
            if not state.classification:
                from src.models.schemas import ClassificationResult
                state.classification = ClassificationResult(
                    category=RequestCategory.UNKNOWN,
                    confidence=0.0,
                    reasoning="Classification failed due to system error",
                    keywords=[]
                )
            
            return state
            
        except Exception as e:
            logger.critical(f"Error in error handling node: {str(e)}")
            return state
    
    # Conditional edge functions
    def _should_continue_after_classification(self, state: WorkflowState) -> str:
        """Determine next step after classification."""
        return "error" if state.errors else "continue"
    
    def _should_continue_after_retrieval(self, state: WorkflowState) -> str:
        """Determine next step after knowledge retrieval."""
        return "error" if state.errors else "continue"
    
    def _should_continue_after_response(self, state: WorkflowState) -> str:
        """Determine next step after response generation."""
        return "error" if state.errors else "continue"
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about workflow performance.
        
        Returns:
            Dictionary with workflow metrics
        """
        try:
            # TODO: Implement Redis DB to persist and retrieve states
            return {
                "total_requests_processed": 0,
                "avg_processing_time": 0.0,
                "success_rate": 0.0,
                "step_success_rates": {
                    "classification": 0.0,
                    "knowledge_retrieval": 0.0,
                    "response_generation": 0.0,
                    "escalation_determination": 0.0
                },
                "error_categories": {},
                "performance_by_category": {}
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow metrics: {str(e)}")
            return {}


# Global workflow instance
helpdesk_workflow = HelpDeskWorkflow()