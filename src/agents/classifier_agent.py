"""
Classification agent for categorizing help desk requests.
"""
import json
import time
import logging
from typing import Dict, Any
from src.models.schemas import (
    HelpDeskRequest, 
    ClassificationResult, 
    RequestCategory,
    AgentResponse
)
from src.services.llm_service import llm_service
from config.settings import CLASSIFICATION_EXAMPLES

logger = logging.getLogger(__name__)


class ClassifierAgent:
    """Agent responsible for classifying help desk requests."""
    
    def __init__(self):
        """Initialize the classifier agent."""
        self.llm = llm_service
        self.categories = list(RequestCategory)
        self.examples = CLASSIFICATION_EXAMPLES
    
    def classify_request(self, request: HelpDeskRequest) -> AgentResponse:
        """
        Classify a help desk request into predefined categories.
        
        Args:
            request: The help desk request to classify
            
        Returns:
            AgentResponse with ClassificationResult
        """
        start_time = time.time()
        
        try:
            # Prepare the classification prompt
            prompt = self._build_classification_prompt(request.request)
            
            # Get LLM response
            response = self.llm.generate_response_sync(
                prompt=prompt,
                # system_message="You are an expert help desk classifier. Analyze requests and provide accurate classifications.",
                system_message = '''
                    You are an expert IT help desk classifier. Your job is to categorize help desk requests 
                                    into specific categories and provide detailed analysis.
                                    
                                    Categories:
                                    - password_reset: Password recovery, account lockout, login issues
                                    - software_installation: Installing, updating, or configuring software
                                    - hardware_failure: Computer, printer, monitor, or device issues
                                    - network_connectivity: Internet, WiFi, VPN, or network access problems
                                    - email_configuration: Email setup, Outlook issues, email access
                                    - security_incident: Malware, suspicious activity, data breaches
                                    - policy_question: Company policies, procedures, guidelines
                                    
                                    Always respond with valid JSON that matches the expected format.
                    ''',
                temperature=0.1
            )
            
            # Parse the response
            classification_result = self._parse_classification_response(response, request.request)
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                success=True,
                data=classification_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error classifying request: {str(e)}"
            logger.error(error_msg)
            
            return AgentResponse(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _build_classification_prompt(self, request_text: str) -> str:
        """Build the classification prompt."""
        categories_str = "\n".join([f"- {cat.value}" for cat in self.categories])
        
        # Build examples section
        examples_section = ""
        for category, examples in self.examples.items():
            examples_section += f"\n{category}:\n"
            for example in examples[:3]:  # Limit to 3 examples per category
                examples_section += f"  - \"{example}\"\n"
        
        prompt = f"""
Classify the following help desk request into one of these categories:

Categories:
{categories_str}

Examples of each category:
{examples_section}

Request to classify: "{request_text}"

Analyze the request and provide your classification in the following JSON format:
{{
    "category": "category_name",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of why you chose this category",
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Important guidelines:
1. Choose the MOST SPECIFIC category that fits the request
2. If unsure between categories, choose the one with higher business impact
3. Security-related issues should always be classified as "security_incident"
4. Password issues should be "password_reset" unless they involve security concerns
5. Confidence should be:
   - 0.9-1.0: Very clear classification
   - 0.7-0.8: Reasonably confident
   - 0.5-0.6: Somewhat uncertain
   - Below 0.5: Very uncertain (consider "unknown")

Only return the JSON, no additional text.
"""
        return prompt
    
    def _parse_classification_response(
        self, 
        response: str, 
        original_request: str
    ) -> ClassificationResult:
        """Parse the LLM classification response."""
        try:
            # Clean the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Parse JSON
            parsed = json.loads(response)
            
            # Validate category
            category_str = parsed.get("category", "unknown")
            try:
                category = RequestCategory(category_str)
            except ValueError:
                logger.warning(f"Invalid category '{category_str}', defaulting to UNKNOWN")
                category = RequestCategory.UNKNOWN
            
            # Extract other fields
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
            
            reasoning = parsed.get("reasoning", "No reasoning provided")
            keywords = parsed.get("keywords", [])
            
            # Validate keywords
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(kw) for kw in keywords if kw][:10]  # Limit to 10 keywords
            
            return ClassificationResult(
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                keywords=keywords
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            # Fallback classification
            return self._fallback_classification(original_request)
        except Exception as e:
            logger.error(f"Error parsing classification response: {str(e)}")
            return self._fallback_classification(original_request)
    
    def _fallback_classification(self, request_text: str) -> ClassificationResult:
        """Provide fallback classification using keyword matching."""
        try:
            request_lower = request_text.lower()
            
            # Simple keyword-based classification
            category_keywords = {
                RequestCategory.PASSWORD_RESET: ["password", "login", "forgot", "reset", "account", "locked"],
                RequestCategory.EMAIL_CONFIGURATION: ["email", "outlook", "mail", "smtp", "pop", "imap"],
                RequestCategory.NETWORK_CONNECTIVITY: ["internet", "network", "wifi", "connection", "vpn"],
                RequestCategory.SOFTWARE_INSTALLATION: ["install", "software", "application", "program", "app"],
                RequestCategory.HARDWARE_FAILURE: ["computer", "printer", "monitor", "keyboard", "mouse", "broken"],
                RequestCategory.SECURITY_INCIDENT: ["virus", "malware", "suspicious", "hack", "phishing", "security"],
                RequestCategory.POLICY_QUESTION: ["policy", "procedure", "allowed", "guideline", "rule"]
            }
            
            best_category = RequestCategory.UNKNOWN
            best_score = 0
            matched_keywords = []
            
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in request_lower)
                if score > best_score:
                    best_score = score
                    best_category = category
                    matched_keywords = [kw for kw in keywords if kw in request_lower]
            
            confidence = min(0.6, best_score * 0.15)  # Conservative confidence
            
            return ClassificationResult(
                category=best_category,
                confidence=confidence,
                reasoning=f"Fallback classification based on keyword matching. Found {best_score} relevant keywords.",
                keywords=matched_keywords[:5]
            )
            
        except Exception as e:
            logger.error(f"Error in fallback classification: {str(e)}")
            return ClassificationResult(
                category=RequestCategory.UNKNOWN,
                confidence=0.1,
                reasoning="Classification failed, defaulting to unknown",
                keywords=[]
            )
    
    def validate_classification(self, classification: ClassificationResult) -> bool:
        """
        Validate a classification result.
        
        Args:
            classification: The classification to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if category is valid
            if classification.category not in RequestCategory:
                return False
            
            # Check confidence range
            if not (0.0 <= classification.confidence <= 1.0):
                return False
            
            # Check if reasoning is provided
            if not classification.reasoning or len(classification.reasoning.strip()) < 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating classification: {str(e)}")
            return False
    
    def get_classification_confidence_threshold(self, category: RequestCategory) -> float:
        """
        Get the minimum confidence threshold for a specific category.
        
        Args:
            category: The category to get threshold for
            
        Returns:
            Minimum confidence threshold
        """
        # Different categories may have different confidence requirements
        thresholds = {
            RequestCategory.SECURITY_INCIDENT: 0.8,  # High confidence required for security
            RequestCategory.HARDWARE_FAILURE: 0.6,   # Medium confidence for hardware
            RequestCategory.PASSWORD_RESET: 0.5,     # Lower threshold for common requests
            RequestCategory.UNKNOWN: 0.0             # Any confidence for unknown
        }
        
        return thresholds.get(category, 0.6)  # Default threshold


# Global classifier agent instance
classifier_agent = ClassifierAgent()