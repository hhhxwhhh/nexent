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
    
@router.post("/knowledge_base/mount",
            summary="Mount a knowledge base",
            description="Attach a knowledge base to the current pathology QA model to enhance its capabilities.")
async def mount_knowledge_base(knowledge_base_id: int):
    """Mount a knowledge base to the current pathology QA model"""
    try:
        # 尝试使用新方法挂载知识库
        success = nexent_service.mount_knowledge_base_to_agent(knowledge_base_id)
        if success:
            return {
                "success": True,
                "message": f"Knowledge base {knowledge_base_id} mounted successfully"
            }
        else:
            # 如果新方法失败，回退到旧方法
            success = nexent_service.mount_knowledge_base_to_current_model(knowledge_base_id)
            if success:
                return {
                    "success": True,
                    "message": f"Knowledge base {knowledge_base_id} mounted successfully (using legacy method)"
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to mount knowledge base {knowledge_base_id}"
                }
    except Exception as e:
        logger.error(f"Error mounting knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error mounting knowledge base: {str(e)}")