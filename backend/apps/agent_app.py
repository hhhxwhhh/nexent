from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import logging
import os
import base64
from typing import List, Optional
from services.nexent_client_service import NexentClientService

router = APIRouter(prefix="/nexent", tags=["Nexent"])
logger = logging.getLogger("nexent_app")

# Global instance of NexentClientService
nexent_service = NexentClientService()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

class ImageAnalysisRequest(BaseModel):
    image_data: str  # Base64 encoded image
    task: str

class ImageAnalysisResponse(BaseModel):
    result: str

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
        
        # 初始化本地视觉语言模型
        model_path = os.getenv("LOCAL_VL_MODEL_PATH", "/Users/wang/model_engine/Qwen2.5-VL-7B-Instruct")
        nexent_service.initialize_local_vl_model(model_path)
        nexent_service.setup_vl_agent()
        
        logger.info("Nexent client initialized successfully with both remote and local models")
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

@router.post("/analyze_image", response_model=ImageAnalysisResponse)
async def analyze_medical_image(request: ImageAnalysisRequest):
    """Analyze a medical image using the local VL model"""
    try:
        result = nexent_service.analyze_medical_image(request.image_data, request.task)
        return ImageAnalysisResponse(result=result)
    except Exception as e:
        logger.error(f"Error analyzing medical image: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing medical image")

@router.post("/upload_image")
async def upload_and_analyze_image(
    file: UploadFile = File(...),
    task: str = "Please describe this medical image"
):
    """Upload and analyze a medical image"""
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to base64
        image_data = base64.b64encode(contents).decode('utf-8')
        
        # Analyze the image
        result = nexent_service.analyze_medical_image(image_data, task)
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "analysis_result": result
        }
    except Exception as e:
        logger.error(f"Error uploading and analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")