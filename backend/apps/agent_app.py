from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import os
from services.nexent_client_service import NexentClientService

router = APIRouter(prefix="/nexent", tags=["Nexent"])
logger = logging.getLogger("nexent_app")

# Global instance of NexentClientService
nexent_service = NexentClientService()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@router.on_event("startup")
async def initialize_nexent_client():
    """Initialize Nexent client on application startup"""
    try:
        # 从环境变量获取值
        api_key = os.getenv("NEXENT_API_KEY", "your-api-key")
        model_endpoint = os.getenv("NEXENT_MODEL_ENDPOINT", "https://nexent.modelengine.com/api/v1")
        
        nexent_service.initialize_nexent_client(api_key, model_endpoint)
        nexent_service.attach_pathology_model()
        nexent_service.setup_pathology_qa_agent()
        
        logger.info("Nexent client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Nexent client: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize Nexent client")

@router.post("/ask", response_model=AnswerResponse)
async def ask_pathology_question(request: QuestionRequest):
    """Ask a pathology-related question"""
    try:
        answer = nexent_service.ask_pathology_question(request.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing question")