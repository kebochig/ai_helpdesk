"""
Response generation agent for creating contextual help desk responses.
"""
import time
import logging
from typing import Dict, Any, List, Optional
from src.models.schemas import (
    HelpDeskRequest,
    ClassificationResult,
    RetrievalResult,
    AgentResponse,
    RequestCategory
)
from src.services.llm_service import llm_service
from config.settings import settings

logger = logging.getLogger(__name__)


class ResponseAgent:
    """Agent responsible for generating contextual responses to help desk requests."""
    
    def __init__(self):
        """Initialize the response agent."""
        self.llm = llm_service
        self.max_response_length = 1000
        self.min_confidence_for_direct_answer = 0.7
    
    def generate_response(
        self,
        request: HelpDeskRequest,
        classification: ClassificationResult,
        retrieval_result: RetrievalResult
    ) -> AgentResponse:
        """
        Generate a contextual response based on the request, classification, and retrieved knowledge.
        
        Args:
            request: The original help desk request
            classification: The classification result
            retrieval_result: The knowledge retrieval result
            
        Returns:
            AgentResponse with generated response string
        """
        start_time = time.time()
        
        try:
            # Determine response strategy based on available information
            response_strategy = self._determine_response_strategy(
                classification, retrieval_result
            )
            
            # Generate response based on strategy
            if response_strategy == "direct_answer":
                response = self._generate_direct_answer(request, classification, retrieval_result)
            elif response_strategy == "guided_solution":
                response = self._generate_guided_solution(request, classification, retrieval_result)
            elif response_strategy == "general_guidance":
                response = self._generate_general_guidance(request, classification)
            else:  # fallback
                response = self._generate_fallback_response(request, classification)
            
            # Post-process response
            final_response = self._post_process_response(response, classification)
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                success=True,
                data=final_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            
            return AgentResponse(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _determine_response_strategy(
        self,
        classification: ClassificationResult,
        retrieval_result: RetrievalResult
    ) -> str:
        """Determine the best response strategy based on available information."""
        
        # High confidence classification with good retrieval results
        if (classification.confidence >= self.min_confidence_for_direct_answer and 
            retrieval_result.total_results >= 2 and
            len(retrieval_result.relevance_scores) > 0 and
            max(retrieval_result.relevance_scores) >= 0.7):
            return "direct_answer"
        
        # Medium confidence with some relevant documents
        if (classification.confidence >= 0.5 and
            retrieval_result.total_results >= 1):
            return "guided_solution"
        
        # Low confidence but recognized category
        if classification.category != RequestCategory.UNKNOWN:
            return "general_guidance"
        
        # Fallback for unclear requests
        return "fallback"
    
    def _generate_direct_answer(
        self,
        request: HelpDeskRequest,
        classification: ClassificationResult,
        retrieval_result: RetrievalResult
    ) -> str:
        """Generate a direct, specific answer using retrieved knowledge."""
        
        # Prepare context from retrieved documents
        context = self._prepare_knowledge_context(retrieval_result)
        
        prompt = f"""
You are a helpful IT support specialist providing direct assistance to users.

User Request: "{request.request}"
Category: {classification.category.value.replace('_', ' ').title()}
Confidence: {classification.confidence:.2f}

Relevant Knowledge Base Information:
{context}

Instructions:
1. Provide a clear, step-by-step solution to the user's specific problem
2. Use the knowledge base information to give accurate guidance. If sufficient knowledge isn't present, suggest based on general knowledge and escalate for further support.
3. Include specific technical details when available
4. Format the response professionally but in a friendly tone
5. If multiple solutions exist, present the most straightforward one first
6. Include any relevant warnings or prerequisites
7. Keep the response concise but complete (under {self.max_response_length} words)

Generate a helpful response:
"""
        
        try:
            response = self.llm.generate_response_sync(
                prompt=prompt,
                temperature=0.2
            )
            return response
        except Exception as e:
            logger.error(f"Error generating direct answer: {str(e)}")
            return self._generate_fallback_response(request, classification)
    
    def _generate_guided_solution(
        self,
        request: HelpDeskRequest,
        classification: ClassificationResult,
        retrieval_result: RetrievalResult
    ) -> str:
        """Generate a guided solution with multiple options or clarifications."""
        
        context = self._prepare_knowledge_context(retrieval_result)
        
        prompt = f"""
You are an IT support specialist helping a user with a technical issue.

User Request: "{request.request}"
Category: {classification.category.value.replace('_', ' ').title()}

Available Information:
{context}

Instructions:
1. Acknowledge the user's request
2. Provide 2-3 possible solutions or troubleshooting steps
3. Explain when each approach might be most appropriate
4. Include any additional information requests if needed for clarification
5. Maintain a helpful and professional tone
6. If the knowledge base doesn't have complete information, say so honestly
7. Suggest escalation if the issue seems complex

Generate a guided response:
"""
        
        try:
            response = self.llm.generate_response_sync(
                prompt=prompt,
                temperature=0.3
            )
            return response
        except Exception as e:
            logger.error(f"Error generating guided solution: {str(e)}")
            return self._generate_fallback_response(request, classification)
    
    def _generate_general_guidance(
        self,
        request: HelpDeskRequest,
        classification: ClassificationResult
    ) -> str:
        """Generate general guidance for recognized categories without specific knowledge."""
        
        category_guidance = {
            RequestCategory.PASSWORD_RESET: {
                "steps": [
                    "Try using the self-service password reset portal",
                    "Check if your account is locked due to multiple failed attempts",
                    "Verify you're using the correct username/email format",
                    "Contact IT support if self-service options don't work"
                ],
                "common_resources": "password reset portal, IT help desk"
            },
            RequestCategory.EMAIL_CONFIGURATION: {
                "steps": [
                    "Verify your email server settings (SMTP/IMAP/POP)",
                    "Check your internet connection",
                    "Ensure your email client is up to date",
                    "Try removing and re-adding your email account"
                ],
                "common_resources": "email setup guides, IT configuration team"
            },
            RequestCategory.NETWORK_CONNECTIVITY: {
                "steps": [
                    "Check if other devices can connect to the network",
                    "Restart your network adapter or Wi-Fi connection",
                    "Verify you're connecting to the correct network",
                    "Check with network administrators about any ongoing issues"
                ],
                "common_resources": "network diagnostics, network support team"
            },
            RequestCategory.SOFTWARE_INSTALLATION: {
                "steps": [
                    "Check if you have administrative privileges",
                    "Verify system requirements for the software",
                    "Download software from approved/official sources only",
                    "Contact IT for assistance with licensed software installation"
                ],
                "common_resources": "software catalog, IT approval process"
            },
            RequestCategory.HARDWARE_FAILURE: {
                "steps": [
                    "Document the specific symptoms and error messages",
                    "Try basic troubleshooting (restart, check connections)",
                    "Check if the device is still under warranty",
                    "Contact hardware support or IT for replacement/repair"
                ],
                "common_resources": "hardware support team, warranty information"
            },
            RequestCategory.SECURITY_INCIDENT: {
                "steps": [
                    "Do not click on suspicious links or download attachments",
                    "Disconnect from the network if you suspect malware",
                    "Document what happened and when",
                    "Report to the security team immediately"
                ],
                "common_resources": "security team, incident response procedures"
            },
            RequestCategory.POLICY_QUESTION: {
                "steps": [
                    "Check the company policy documentation or intranet",
                    "Consult with your manager or HR representative",
                    "Review any relevant training materials",
                    "Contact the appropriate department for clarification"
                ],
                "common_resources": "policy documents, HR department, management"
            }
        }
        
        guidance = category_guidance.get(classification.category, {
            "steps": ["Contact IT support for assistance with this request"],
            "common_resources": "IT help desk"
        })
        
        prompt = f"""
Create a helpful response for this IT support request:

User Request: "{request.request}"
Category: {classification.category.value.replace('_', ' ').title()}

Suggested steps: {guidance['steps']}
Resources: {guidance['common_resources']}

Instructions:
1. Acknowledge the user's request
2. Provide the suggested troubleshooting steps in a clear format
3. Mention available resources
4. Maintain a professional but friendly tone
5. Suggest contacting support if the steps don't resolve the issue

Generate the response:
"""
        
        try:
            response = self.llm.generate_response_sync(
                prompt=prompt,
                temperature=0.3
            )
            return response
        except Exception as e:
            logger.error(f"Error generating general guidance: {str(e)}")
            return self._generate_fallback_response(request, classification)
    
    def _generate_fallback_response(
        self,
        request: HelpDeskRequest,
        classification: ClassificationResult
    ) -> str:
        """Generate a fallback response when other strategies fail."""
        
        fallback_template = f"""
Thank you for contacting IT support. I understand you're experiencing an issue with: "{request.request}"

While I don't have specific information about your exact situation in our knowledge base, I want to help you get this resolved quickly.

Here's what I recommend:

1. **Immediate Steps:**
   - Document any error messages you're seeing
   - Note when the issue started and what you were doing at the time
   - Try restarting the application or device if applicable

2. **Next Steps:**
   - This request appears to be related to {classification.category.value.replace('_', ' ')}
   - I'm escalating your request to our specialized support team
   - You should receive a response within the standard timeframe for this type of issue

3. **Additional Information:**
   - Please have your employee ID and device information ready
   - If this is urgent, you can also call our IT help desk directly

Your request has been logged and assigned a tracking number. You'll receive updates as we work on resolving this issue.

Is there any additional information you can provide that might help us understand the problem better?
"""
        
        return fallback_template.strip()
    
    def _prepare_knowledge_context(self, retrieval_result: RetrievalResult) -> str:
        """Prepare context from retrieved knowledge base documents."""
        
        if not retrieval_result.documents:
            return "No specific knowledge base articles found for this request."
        
        context_parts = []
        
        for i, (doc, score) in enumerate(zip(retrieval_result.documents, retrieval_result.relevance_scores)):
            # Only include high-relevance documents in context
            if score >= 0.5:
                context_parts.append(f"""
Document {i+1} (Relevance: {score:.2f}):
Title: {doc.title}
Content: {doc.content[:500]}{'...' if len(doc.content) > 500 else ''}
""")
        
        if not context_parts:
            return "No highly relevant knowledge base articles found for this specific request."
        
        return "\n".join(context_parts)
    
    def _post_process_response(self, response: str, classification: ClassificationResult) -> str:
        """Post-process the generated response for quality and consistency."""
        
        try:
            # Clean up formatting
            response = response.strip()
            
            # Ensure reasonable length
            if len(response) > self.max_response_length:
                # Truncate at the last complete sentence
                sentences = response.split('. ')
                truncated = ''
                for sentence in sentences:
                    if len(truncated + sentence + '. ') <= self.max_response_length:
                        truncated += sentence + '. '
                    else:
                        break
                response = truncated.strip()
                if not response.endswith('.'):
                    response += '.'
            
            # Add category-specific closing if needed
            if classification.category == RequestCategory.SECURITY_INCIDENT:
                if "security team" not in response.lower():
                    response += "\n\nPlease report this to the security team immediately if you haven't already done so."
            
            return response
            
        except Exception as e:
            logger.error(f"Error post-processing response: {str(e)}")
            return response  # Return original if post-processing fails
    
    def evaluate_response_quality(self, response: str, classification: ClassificationResult) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated response.
        
        Args:
            response: The generated response
            classification: The original classification
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            metrics = {
                "length_appropriate": 50 <= len(response) <= self.max_response_length,
                "contains_actionable_steps": any(word in response.lower() for word in 
                    ['step', 'try', 'check', 'verify', 'contact', 'click', 'go to']),
                "professional_tone": not any(word in response.lower() for word in 
                    ['dunno', 'maybe', 'i guess', 'whatever']),
                "category_relevant": classification.category.value.replace('_', ' ') in response.lower(),
                "includes_escalation": any(word in response.lower() for word in 
                    ['escalat', 'contact', 'support', 'help desk'])
            }
            
            quality_score = sum(metrics.values()) / len(metrics)
            
            return {
                "quality_score": quality_score,
                "metrics": metrics,
                "recommendation": "Good response" if quality_score >= 0.8 else 
                    "Review needed" if quality_score >= 0.6 else "Poor quality"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating response quality: {str(e)}")
            return {"quality_score": 0.5, "metrics": {}, "recommendation": "Could not evaluate"}


# Global response agent instance
response_agent = ResponseAgent()