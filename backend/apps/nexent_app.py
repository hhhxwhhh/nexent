from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import logging
import os
import base64
from typing import List, Optional, Dict, Any

from services.nexent_client_service import NexentClientService

router = APIRouter(prefix="/nexent", tags=["Nexent"])
logger = logging.getLogger("nexent_app")

# Global instance of NexentClientService
nexent_service = NexentClientService()

# Request and Response Models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

class ImageAnalysisRequest(BaseModel):
    image_data: str  # Base64 encoded image
    task: str

class ImageAnalysisResponse(BaseModel):
    result: str

class ModelSelectionRequest(BaseModel):
    model_name: str

class ModelSelectionResponse(BaseModel):
    success: bool
    message: str

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class KnowledgeBaseCreateRequest(BaseModel):
    knowledge_base_name: str

class KnowledgeBaseCreateResponse(BaseModel):
    success: bool
    knowledge_base_id: Optional[int] = None
    message: str

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

# Core Question Answering Endpoints
@router.post("/ask", response_model=AnswerResponse, 
             summary="Ask a pathology-related question",
             description="Submit a pathology-related question to the intelligent agent for analysis and response.")
async def ask_pathology_question(request: QuestionRequest):
    """Ask a pathology-related question"""
    try:
        answer = nexent_service.ask_pathology_question(request.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing question")

# Medical Image Analysis Endpoints
@router.post("/analyze_image", response_model=ImageAnalysisResponse,
             summary="Analyze a medical image",
             description="Submit a base64-encoded medical image for analysis by the local vision-language model.")
async def analyze_medical_image(request: ImageAnalysisRequest):
    """Analyze a medical image using the local VL model"""
    try:
        result = nexent_service.analyze_medical_image(request.image_data, request.task)
        return ImageAnalysisResponse(result=result)
    except Exception as e:
        logger.error(f"Error analyzing medical image: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing medical image")

@router.post("/upload_image",
             summary="Upload and analyze a medical image",
             description="Upload a medical image file and automatically analyze it with the vision-language model.")
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

# System Health and Management Endpoints
@router.get("/health", response_model=HealthCheckResponse,
            summary="Check service health",
            description="Perform a health check on the Nexent service and its connections to backend systems.")
async def health_check():
    """Check the health status of the Nexent service"""
    try:
        is_connected = nexent_service.check_model_engine_connectivity()
        if is_connected:
            return HealthCheckResponse(
                status="healthy",
                message="Nexent service is running and connected to ModelEngine"
            )
        else:
            return HealthCheckResponse(
                status="degraded",
                message="Nexent service is running but not connected to ModelEngine"
            )
    except Exception as e:
        logger.error(f"Error during health check: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during health check")

@router.post("/select_model", response_model=ModelSelectionResponse,
             summary="Select a model",
             description="Choose and configure a specific model for use in subsequent operations.")
async def select_model(request: ModelSelectionRequest):
    """Select and configure a specific model"""
    try:
        success = nexent_service.select_model(request.model_name)
        if success:
            return ModelSelectionResponse(
                success=True,
                message=f"Successfully selected model: {request.model_name}"
            )
        else:
            return ModelSelectionResponse(
                success=False,
                message=f"Failed to select model: {request.model_name}"
            )
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error selecting model: {str(e)}")

@router.get("/models",
            summary="List available models",
            description="Retrieve a list of models available from the ModelEngine service.")
async def list_models():
    """List available models from ModelEngine"""
    try:
        models = nexent_service.list_available_models()
        return {
            "models": models
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing models")

# Knowledge Base Management Endpoints
@router.post("/documents/upload",
             summary="Upload pathology documents",
             description="Upload pathology documents to create a new knowledge base for enhanced question answering.")
async def upload_documents(
    files: List[UploadFile] = File(...),
    knowledge_base_name: str = "default_knowledge_base"
):
    """Upload pathology documents and create a knowledge base"""
    try:
        result = await nexent_service.upload_pathology_documents(
            files=files,
            knowledge_base_name=knowledge_base_name
        )
        return result
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")

@router.post("/knowledge_base/mount",
             summary="Mount a knowledge base",
             description="Attach a knowledge base to the current pathology QA model to enhance its capabilities.")
async def mount_knowledge_base(knowledge_base_id: int):
    """Mount a knowledge base to the current pathology QA model"""
    try:
        success = nexent_service.mount_knowledge_base_to_current_model(knowledge_base_id)
        if success:
            return {
                "success": True,
                "message": f"Knowledge base {knowledge_base_id} mounted successfully"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to mount knowledge base {knowledge_base_id}"
            }
    except Exception as e:
        logger.error(f"Error mounting knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error mounting knowledge base: {str(e)}")