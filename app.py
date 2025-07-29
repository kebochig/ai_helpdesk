"""
FastAPI application for the AI Help Desk System.
"""
import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from src.models.schemas import HelpDeskRequest, HelpDeskResponse
from src.workflows.helpdesk_workflow import helpdesk_workflow
from src.services.document_processor import document_processor
from src.services.vector_store import vector_store_service

from main import HelpDeskSystem


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


# Pydantic models for API requests/responses
class HelpDeskRequestAPI(BaseModel):
    """API model for help desk requests."""
    request: str = Field(..., description="The user's help desk request text", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class BatchRequestAPI(BaseModel):
    """API model for batch requests."""
    requests: List[HelpDeskRequestAPI] = Field(..., description="List of help desk requests")


class SystemStatusResponse(BaseModel):
    """API model for system status response."""
    initialized: bool
    timestamp: str
    knowledge_base: Dict[str, Any]
    workflow_metrics: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RefreshResponse(BaseModel):
    """API model for refresh response."""
    success: bool
    message: str
    timestamp: str


# Global system instance
help_desk_system = HelpDeskSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI Help Desk System...")
    success = await help_desk_system.initialize()
    if not success:
        logger.error("Failed to initialize help desk system")
        raise RuntimeError("System initialization failed")
    
    logger.info("AI Help Desk System started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down AI Help Desk System...")


# Create FastAPI app with lifespan events
app = FastAPI(
    title="AI Help Desk System",
    description="An intelligent help desk system powered by AI for automated support ticket processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_system() -> HelpDeskSystem:
    """Dependency to get the help desk system instance."""
    return help_desk_system


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint returning basic API information."""
    return {
        "message": "AI Help Desk System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check(system: HelpDeskSystem = Depends(get_system)):
    """Health check endpoint."""
    if not system.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "status": "healthy",
        "initialized": system.is_initialized,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status", response_model=SystemStatusResponse, tags=["System"])
async def get_system_status(system: HelpDeskSystem = Depends(get_system)):
    """Get detailed system status and metrics."""
    status = system.get_system_status()
    return SystemStatusResponse(**status)


@app.post("/process", response_model=HelpDeskResponse, tags=["Help Desk"])
def process_request(
    request: HelpDeskRequestAPI,
    system: HelpDeskSystem = Depends(get_system)
):
    """
    Process a single help desk request.
    
    Args:
        request: The help desk request containing the user's question and optional user ID
        
    Returns:
        HelpDeskResponse with the AI-generated response and metadata
    """
    try:
        response =  system.process_request_sync(
            request_text=request.request,
            user_id=request.user_id
        )
        print(f"Category: {response.category.value}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Response: {response.response[:200]}...")
        print(f"Escalation: {'Yes' if response.escalation.should_escalate else 'No'}")
        print(f"Priority: {response.escalation.priority.value}")
        print(f"Processing Time: {response.processing_time:.2f}s")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/process/batch", response_model=List[HelpDeskResponse], tags=["Help Desk"])
def process_batch_requests(
    batch_request: BatchRequestAPI,
    background_tasks: BackgroundTasks,
    system: HelpDeskSystem = Depends(get_system)
):
    """
    Process multiple help desk requests in batch.
    
    Args:
        batch_request: List of help desk requests
        background_tasks: FastAPI background tasks
        
    Returns:
        List of HelpDeskResponse objects
    """
    try:
        # Convert API models to dictionaries
        requests_data = [
            {
                "request": req.request,
                "user_id": req.user_id
            }
            for req in batch_request.requests
        ]
        
        responses = system.batch_process_requests(requests_data)
        return responses
    except Exception as e:
        logger.error(f"Error processing batch requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing batch requests: {str(e)}")


@app.post("/knowledge-base/refresh", response_model=RefreshResponse, tags=["Knowledge Base"])
async def refresh_knowledge_base(
    background_tasks: BackgroundTasks,
    system: HelpDeskSystem = Depends(get_system)
):
    """
    Refresh the knowledge base with updated documents.
    This operation runs in the background to avoid timeout issues.
    """
    async def refresh_task():
        try:
            success = await system.refresh_knowledge_base()
            if success:
                logger.info("Knowledge base refresh completed successfully")
            else:
                logger.error("Knowledge base refresh failed")
        except Exception as e:
            logger.error(f"Error during knowledge base refresh: {str(e)}")
    
    # Add refresh task to background tasks
    background_tasks.add_task(refresh_task)
    
    return RefreshResponse(
        success=True,
        message="Knowledge base refresh started in background",
        timestamp=datetime.now().isoformat()
    )


@app.get("/knowledge-base/stats", tags=["Knowledge Base"])
async def get_knowledge_base_stats(system: HelpDeskSystem = Depends(get_system)):
    """Get knowledge base statistics."""
    try:
        stats = system.vector_store.get_collection_stats()
        return {
            "knowledge_base_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting knowledge base stats: {str(e)}")


@app.get("/sample-requests", tags=["Help Desk"])
async def get_sample_requests():
    """Get sample requests for testing the system."""
    return {
        "sample_requests": [
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
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Custom internal server error handler."""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )