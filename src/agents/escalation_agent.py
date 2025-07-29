"""
Escalation agent for determining when human intervention is needed.
"""
import time
import logging
from typing import Dict, Any, Optional
from src.models.schemas import (
    HelpDeskRequest,
    ClassificationResult,
    RetrievalResult,
    EscalationDecision,
    PriorityLevel,
    RequestCategory,
    AgentResponse
)
from src.services.llm_service import llm_service
from config.settings import settings, ESCALATION_RULES

logger = logging.getLogger(__name__)


class EscalationAgent:
    """Agent responsible for determining escalation decisions."""
    
    def __init__(self):
        """Initialize the escalation agent."""
        self.llm = llm_service
        self.escalation_threshold = settings.system_config.escalation_threshold
        self.escalation_rules = ESCALATION_RULES
    
    def determine_escalation(
        self,
        request: HelpDeskRequest,
        classification: ClassificationResult,
        retrieval_result: RetrievalResult,
        response_quality: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Determine if a request should be escalated to human support.
        
        Args:
            request: The original help desk request
            classification: The classification result
            retrieval_result: The knowledge retrieval result
            response_quality: Optional response quality metrics
            
        Returns:
            AgentResponse with EscalationDecision
        """
        start_time = time.time()
        
        try:
            # Check category-specific auto-escalation rules
            category_rule = self.escalation_rules.get(classification.category.value, {})
            
            if category_rule.get("auto_escalate", False):
                decision = self._create_auto_escalation_decision(classification, category_rule)
            else:
                # Analyze multiple factors to make escalation decision
                decision = self._analyze_escalation_factors(
                    request, classification, retrieval_result, response_quality
                )
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                success=True,
                data=decision,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error determining escalation: {str(e)}"
            logger.error(error_msg)
            
            # Default to escalation on error for safety
            fallback_decision = EscalationDecision(
                should_escalate=True,
                reasoning="Error in escalation analysis - defaulting to human review",
                priority=PriorityLevel.MEDIUM,
                suggested_department="it_support",
                estimated_complexity="unknown"
            )
            
            return AgentResponse(
                success=False,
                error=error_msg,
                data=fallback_decision,
                processing_time=processing_time
            )
    
    def _create_auto_escalation_decision(
        self,
        classification: ClassificationResult,
        category_rule: Dict[str, Any]
    ) -> EscalationDecision:
        """Create escalation decision for auto-escalation categories."""
        
        priority_map = {
            "low": PriorityLevel.LOW,
            "medium": PriorityLevel.MEDIUM,
            "high": PriorityLevel.HIGH,
            "critical": PriorityLevel.CRITICAL
        }
        
        return EscalationDecision(
            should_escalate=True,
            reasoning=f"Auto-escalation triggered for {classification.category.value} category due to security/business impact requirements",
            priority=priority_map.get(category_rule.get("priority", "medium"), PriorityLevel.MEDIUM),
            suggested_department=category_rule.get("department", "it_support"),
            estimated_complexity=f"Category: {classification.category.value}, Response time: {category_rule.get('response_time', 'standard')}"
        )
    
    def _analyze_escalation_factors(
        self,
        request: HelpDeskRequest,
        classification: ClassificationResult,
        retrieval_result: RetrievalResult,
        response_quality: Optional[Dict[str, Any]]
    ) -> EscalationDecision:
        """Analyze multiple factors to determine escalation need."""
        
        escalation_factors = []
        escalation_score = 0.0
        
        # Factor 1: Classification confidence
        if classification.confidence < 0.5:
            escalation_factors.append("Low classification confidence")
            escalation_score += 0.3
        elif classification.confidence < 0.7:
            escalation_score += 0.1
        
        # Factor 2: Knowledge retrieval quality
        if retrieval_result.total_results == 0:
            escalation_factors.append("No relevant knowledge base articles found")
            escalation_score += 0.4
        elif retrieval_result.total_results < 2 or (retrieval_result.relevance_scores and max(retrieval_result.relevance_scores) < 0.6):
            escalation_factors.append("Limited relevant knowledge available")
            escalation_score += 0.2
        
        # Factor 3: Response quality (if provided)
        if response_quality:
            quality_score = response_quality.get("quality_score", 0.5)
            if quality_score < 0.6:
                escalation_factors.append("Generated response quality below threshold")
                escalation_score += 0.2
        
        # Factor 4: Request complexity analysis
        complexity_analysis = self._analyze_request_complexity(request)
        if complexity_analysis["complexity_level"] == "high":
            escalation_factors.append("High complexity request detected")
            escalation_score += 0.3
        elif complexity_analysis["complexity_level"] == "medium":
            escalation_score += 0.1
        
        # Factor 5: Sentiment and urgency analysis
        sentiment_analysis = self.llm.analyze_text_sentiment(request.request)
        if sentiment_analysis.get("urgency") in ["high", "critical"]:
            escalation_factors.append("High urgency indicated by user")
            escalation_score += 0.2
        
        if sentiment_analysis.get("sentiment") == "negative" and sentiment_analysis.get("tone") in ["frustrated", "angry"]:
            escalation_factors.append("User appears frustrated or upset")
            escalation_score += 0.1
        
        # Factor 6: Category-specific thresholds
        category_rule = self.escalation_rules.get(classification.category.value, {})
        category_threshold = 0.5  # Default threshold
        if category_rule.get("priority") == "high":
            category_threshold = 0.4  # Lower threshold for high-priority categories
        elif category_rule.get("priority") == "critical":
            category_threshold = 0.3  # Even lower for critical
        
        # Make escalation decision
        should_escalate = escalation_score >= category_threshold
        
        # Determine priority level
        if escalation_score >= 0.8:
            priority = PriorityLevel.CRITICAL
        elif escalation_score >= 0.6:
            priority = PriorityLevel.HIGH
        elif escalation_score >= 0.4:
            priority = PriorityLevel.MEDIUM
        else:
            priority = PriorityLevel.LOW
        
        # Override priority based on category rules
        category_priority = category_rule.get("priority")
        if category_priority:
            priority_map = {
                "low": PriorityLevel.LOW,
                "medium": PriorityLevel.MEDIUM,
                "high": PriorityLevel.HIGH,
                "critical": PriorityLevel.CRITICAL
            }
            rule_priority = priority_map.get(category_priority, priority)
            # Use the higher of the calculated or rule-based priority
            if list(PriorityLevel).index(rule_priority) > list(PriorityLevel).index(priority):
                priority = rule_priority
        
        # Generate reasoning
        if should_escalate:
            reasoning = f"Escalation recommended based on: {', '.join(escalation_factors)}. Escalation score: {escalation_score:.2f}"
        else:
            reasoning = f"No escalation needed. AI can handle this request. Escalation score: {escalation_score:.2f}"
            if escalation_factors:
                reasoning += f". Minor concerns: {', '.join(escalation_factors)}"
        
        return EscalationDecision(
            should_escalate=should_escalate,
            reasoning=reasoning,
            priority=priority,
            suggested_department=category_rule.get("department", "it_support"),
            estimated_complexity=complexity_analysis["complexity_level"]
        )
    
    def _analyze_request_complexity(self, request: HelpDeskRequest) -> Dict[str, Any]:
        """Analyze the complexity of a request."""
        try:
            prompt = f"""
            Analyze the complexity of this IT support request:
            
            Request: "{request.request}"
            
            Consider these factors:
            1. Technical depth required
            2. Number of systems potentially involved
            3. Likelihood of requiring specialized knowledge
            4. Risk of causing system downtime
            5. Whether it's a common/routine request
            
            Respond with a JSON object:
            {{
                "complexity_level": "low/medium/high",
                "technical_depth": "basic/intermediate/advanced",
                "systems_involved": ["system1", "system2"],
                "specialist_knowledge_required": true/false,
                "routine_request": true/false,
                "risk_level": "low/medium/high",
                "reasoning": "explanation"
            }}
            
            Only return the JSON, no additional text.
            """
            
            response = self.llm.generate_response_sync(prompt, temperature=0.1)
            
            # Parse JSON response
            import json
            analysis = json.loads(response)
            
            # Validate and provide defaults
            analysis.setdefault("complexity_level", "medium")
            analysis.setdefault("technical_depth", "intermediate")
            analysis.setdefault("systems_involved", [])
            analysis.setdefault("specialist_knowledge_required", False)
            analysis.setdefault("routine_request", False)
            analysis.setdefault("risk_level", "medium")
            analysis.setdefault("reasoning", "Analysis completed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing request complexity: {str(e)}")
            return {
                "complexity_level": "medium",
                "technical_depth": "intermediate",
                "systems_involved": [],
                "specialist_knowledge_required": False,
                "routine_request": False,
                "risk_level": "medium",
                "reasoning": "Could not analyze complexity - defaulting to medium"
            }
    
    def get_escalation_recommendations(
        self,
        escalation_decision: EscalationDecision,
        classification: ClassificationResult
    ) -> Dict[str, Any]:
        """
        Get detailed escalation recommendations.
        
        Args:
            escalation_decision: The escalation decision
            classification: The original classification
            
        Returns:
            Dictionary with escalation recommendations
        """
        try:
            category_rule = self.escalation_rules.get(classification.category.value, {})
            
            recommendations = {
                "escalation_required": escalation_decision.should_escalate,
                "priority": escalation_decision.priority.value,
                "suggested_department": escalation_decision.suggested_department,
                "estimated_response_time": category_rule.get("response_time", "standard"),
                "next_steps": [],
                "required_information": [],
                "escalation_path": []
            }
            
            if escalation_decision.should_escalate:
                # Define next steps based on category and priority
                if escalation_decision.priority == PriorityLevel.CRITICAL:
                    recommendations["next_steps"] = [
                        "Immediately assign to senior technician",
                        "Notify department manager",
                        "Set up direct communication with user",
                        "Monitor progress every 30 minutes"
                    ]
                    recommendations["escalation_path"] = [
                        "Level 2 Support → Senior Technician → Department Manager"
                    ]
                elif escalation_decision.priority == PriorityLevel.HIGH:
                    recommendations["next_steps"] = [
                        "Assign to experienced technician",
                        "Set priority flag in ticketing system",
                        "Schedule follow-up within 2 hours"
                    ]
                    recommendations["escalation_path"] = [
                        "Level 2 Support → Senior Technician"
                    ]
                else:
                    recommendations["next_steps"] = [
                        "Route to appropriate department queue",
                        "Assign standard priority",
                        "Follow normal SLA timeline"
                    ]
                    recommendations["escalation_path"] = [
                        "Level 2 Support"
                    ]
                
                # Category-specific required information
                if classification.category == RequestCategory.SECURITY_INCIDENT:
                    recommendations["required_information"] = [
                        "Detailed timeline of events",
                        "Affected systems and data",
                        "Current containment status",
                        "User contact information for immediate response"
                    ]
                elif classification.category == RequestCategory.HARDWARE_FAILURE:
                    recommendations["required_information"] = [
                        "Device model and serial number",
                        "Error messages or symptoms",
                        "Warranty status",
                        "Business impact assessment"
                    ]
                elif classification.category == RequestCategory.NETWORK_CONNECTIVITY:
                    recommendations["required_information"] = [
                        "Affected locations or users",
                        "Network diagnostics results",
                        "Timeline of connectivity issues",
                        "Business services impacted"
                    ]
                else:
                    recommendations["required_information"] = [
                        "Detailed problem description",
                        "Steps already attempted",
                        "Business impact",
                        "Preferred resolution timeline"
                    ]
            else:
                recommendations["next_steps"] = [
                    "Provide AI-generated response to user",
                    "Monitor for user satisfaction",
                    "Follow up if no response within 24 hours"
                ]
                recommendations["escalation_path"] = ["Self-service resolution"]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating escalation recommendations: {str(e)}")
            return {
                "escalation_required": True,
                "priority": "medium",
                "suggested_department": "it_support",
                "next_steps": ["Manual review required"],
                "escalation_path": ["Level 2 Support"]
            }
    
    def update_escalation_rules(
        self,
        category: RequestCategory,
        new_rules: Dict[str, Any]
    ) -> bool:
        """
        Update escalation rules for a specific category.
        
        Args:
            category: The category to update rules for
            new_rules: New rule parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate new rules
            valid_keys = {"auto_escalate", "priority", "department", "response_time"}
            if not all(key in valid_keys for key in new_rules.keys()):
                logger.error("Invalid rule keys provided")
                return False
            
            # Update rules (in a production system, this would persist to database)
            current_rules = self.escalation_rules.get(category.value, {})
            current_rules.update(new_rules)
            self.escalation_rules[category.value] = current_rules
            
            logger.info(f"Updated escalation rules for {category.value}: {new_rules}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating escalation rules: {str(e)}")
            return False
    
    def get_escalation_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about escalation decisions.
        
        Returns:
            Dictionary with escalation metrics
        """
        try:
            # In a production system, this would query actual escalation data
            return {
                "total_requests_processed": 0,  # Would be actual count
                "escalation_rate": 0.0,  # Percentage of requests escalated
                "avg_resolution_time_ai": 0.0,  # Average time for AI-resolved requests
                "avg_resolution_time_human": 0.0,  # Average time for human-resolved requests
                "escalation_accuracy": 0.0,  # Percentage of escalations that were appropriate
                "category_escalation_rates": {},  # Escalation rates by category
                "priority_distribution": {
                    "low": 0,
                    "medium": 0,
                    "high": 0,
                    "critical": 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting escalation metrics: {str(e)}")
            return {}


# Global escalation agent instance
escalation_agent = EscalationAgent()