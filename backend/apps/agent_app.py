from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import logging
from typing import List, Optional
from services.nexent_client_service import NexentClientService

router = APIRouter(prefix="/agent", tags=["Agent"])
logger = logging.getLogger("agent_app")

# Global instance of NexentClientService
nexent_service = NexentClientService()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@router.on_event("startup")
async def initialize_agent_client():
    """Initialize agent client on application startup"""
    logger.info("Agent router initialized")

@router.post("/ask", response_model=AnswerResponse)
async def ask_agent_question(request: QuestionRequest):
    """Ask a question to the agent"""
    try:
        # This is a simplified endpoint that delegates to the main nexent service
        # In a more complex system, this could route to specialized agents
        logger.info(f"Agent question received: {request.question}")
        return AnswerResponse(answer="Agent functionality is integrated with the main Nexent service. Please use /nexent/ask endpoint.")
    except Exception as e:
        logger.error(f"Error processing agent question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing question")