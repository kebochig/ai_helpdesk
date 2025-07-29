"""
LLM service for interacting with Google Gemini API.
"""
import time
import logging
from typing import Optional, Dict, Any
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations using Google Gemini."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self._setup_genai()
        self.llm = GoogleGenerativeAI(
            model=settings.system_config.llm.model_name,
            temperature=settings.system_config.llm.temperature,
            max_tokens=settings.system_config.llm.max_tokens,
            google_api_key=settings.google_api_key
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.system_config.vector_store.embedding_model,
            google_api_key=settings.google_api_key
        )
        
    def _setup_genai(self):
        """Setup Google Generative AI."""
        genai.configure(api_key=settings.google_api_key)
    
    async def generate_response(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            temperature: Optional temperature override
            
        Returns:
            Generated response text
        """
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            # Generate response
            if temperature:
                # Create a temporary LLM instance with different temperature
                temp_llm = GoogleGenerativeAI(
                    model=settings.system_config.llm.model_name,
                    temperature=temperature,
                    max_tokens=settings.system_config.llm.max_tokens,
                    google_api_key=settings.google_api_key
                )
                response = await temp_llm.agenerate([messages])
            else:
                response = await self.llm.agenerate([messages])
            
            processing_time = time.time() - start_time
            logger.info(f"LLM response generated in {processing_time:.2f}s")
            
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    def generate_response_sync(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Synchronous version of generate_response.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            temperature: Optional temperature override
            
        Returns:
            Generated response text
        """
        try:
            start_time = time.time()
            
            # Prepare full prompt
            full_prompt = ""
            if system_message:
                full_prompt += f"System: {system_message}\n\n"
            full_prompt += f"User: {prompt}\n\nAssistant:"
            
            # Generate response
            if temperature:
                temp_llm = GoogleGenerativeAI(
                    model=settings.system_config.llm.model_name,
                    temperature=temperature,
                    max_tokens=settings.system_config.llm.max_tokens,
                    google_api_key=settings.google_api_key
                )
                response = temp_llm.invoke(full_prompt)
            else:
                response = self.llm.invoke(full_prompt)
            
            processing_time = time.time() - start_time
            logger.info(f"LLM response generated in {processing_time:.2f}s")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            start_time = time.time()
            embeddings = self.embeddings.embed_documents(texts)
            processing_time = time.time() - start_time
            logger.info(f"Generated embeddings for {len(texts)} texts in {processing_time:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment and urgency of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        prompt = f"""
        Analyze the following help desk request for sentiment, urgency, and tone:
        
        Request: "{text}"
        
        Provide your analysis in the following JSON format:
        {{
            "sentiment": "positive/neutral/negative",
            "urgency": "low/medium/high/critical",
            "tone": "frustrated/calm/confused/angry/polite",
            "confidence": 0.0-1.0,
            "reasoning": "explanation of analysis"
        }}
        
        Only return the JSON, no additional text.
        """
        
        try:
            response = self.generate_response_sync(prompt, temperature=0.1)
            # Parse JSON response
            import json
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return {
                "sentiment": "neutral",
                "urgency": "medium",
                "tone": "unknown",
                "confidence": 0.5,
                "reasoning": "Analysis failed"
            }
    
    def extract_keywords(self, text: str) -> list[str]:
        """
        Extract key terms and entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of keywords
        """
        prompt = f"""
        Extract the most important keywords and technical terms from this help desk request:
        
        Request: "{text}"
        
        Return only a comma-separated list of keywords, no additional text.
        Focus on:
        - Technical terms
        - Software/hardware names
        - Actions requested
        - Problem symptoms
        """
        
        try:
            response = self.generate_response_sync(prompt, temperature=0.1)
            keywords = [kw.strip() for kw in response.split(',')]
            return [kw for kw in keywords if kw and len(kw) > 1]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []


# Global LLM service instance
llm_service = LLMService()