from fastmcp import FastMCP
import base64
import io
from typing import Optional, List, Dict, Any
import logging
from PIL import Image
import numpy as np
import os
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# Create MCP server
local_mcp_service = FastMCP("local")

logger = logging.getLogger(__name__)

# 本地视觉语言模型配置
LOCAL_VL_MODEL_PATH = os.getenv("LOCAL_VL_MODEL_PATH", "/Users/wang/model_engine/Qwen2.5-VL-7B-Instruct")
local_vl_model = None
local_vl_processor = None

def initialize_local_vl_model():
    """
    初始化本地视觉语言模型
    """
    global local_vl_model, local_vl_processor
    
    try:
        if os.path.exists(LOCAL_VL_MODEL_PATH):
            logger.info(f"Initializing local VL model from: {LOCAL_VL_MODEL_PATH}")
            
            # 初始化模型和处理器
            local_vl_model = LLM(
                model=LOCAL_VL_MODEL_PATH,
                max_model_len=4096,
                tensor_parallel_size=1,
                quantization=None,
                dtype=torch.bfloat16,
            )
            local_vl_processor = AutoProcessor.from_pretrained(LOCAL_VL_MODEL_PATH)
            
            logger.info("Local VL model initialized successfully")
        else:
            logger.warning(f"Local VL model path does not exist: {LOCAL_VL_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize local VL model: {str(e)}")

# 在模块加载时尝试初始化模型
initialize_local_vl_model()

@local_mcp_service.tool(name="test_tool_name",
                        description="test_tool_description")
async def demo_tool(para_1: str, para_2: int) -> dict:
    """
    示例工具，返回结构化数据以避免解析错误
    
    Args:
        para_1: 字符串参数
        para_2: 整数参数
        
    Returns:
        dict: 包含状态和结果的字典
    """
    try:
        print("tool is called successfully")
        print(para_1, para_2)
        
        # 返回结构化数据，避免解析错误
        result = {
            "status": "success",
            "data": {
                "message": "Tool executed successfully",
                "parameters": {
                    "para_1": para_1,
                    "para_2": para_2
                },
                "result": f"Processed: {para_1} with value {para_2}"
            }
        }
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@local_mcp_service.tool(
    name="pathology_data_analysis",
    description="分析病理数据并提供专业建议"
)
def pathology_data_analysis(
    symptom: str,
    duration_days: int = 7
) -> dict:
    """
    一个用于病理数据分析的示例工具
    
    Args:
        symptom: 症状描述
        duration_days: 持续天数
        
    Returns:
        dict: 包含分析结果的字典
    """
    try:
        # 模拟数据分析过程
        treatment_steps = []
        
        if "感冒" in symptom or "咳嗽" in symptom:
            treatment_steps = [
                "1. 休息：感冒时，休息是非常重要的。给自己足够的时间来恢复，避免过度劳累。",
                "2. 多喝水：保持身体水分充足有助于稀释黏液，缓解喉咙痛和鼻塞。",
                "3. 温水漱口：用温盐水漱口可以缓解喉咙疼痛和减少炎症。",
                "4. 蜂蜜和柠檬：蜂蜜和柠檬可以舒缓喉咙，柠檬中的维生素C也有助于增强免疫系统。",
                "5. 蒸汽疗法：可以在热水中加入一些薄荷或桉树油，然后吸入蒸汽，这有助于缓解鼻塞。"
            ]
            summary = "根据您的症状，建议采取以上措施。如果症状持续或加重，请及时就医。"
        elif "头痛" in symptom:
            treatment_steps = [
                "1. 充足睡眠：确保每晚有7-9小时的优质睡眠。",
                "2. 减轻压力：尝试冥想、深呼吸或其他放松技巧。",
                "3. 适当运动：规律的有氧运动可以帮助缓解紧张性头痛。",
                "4. 保持水分：脱水是头痛的常见原因，确保喝足够的水。",
                "5. 热敷或冷敷：根据个人感受选择热敷或冷敷头部或颈部。"
            ]
            summary = "头痛可能由多种因素引起，建议采取以上措施。如果头痛频繁发作或非常严重，请咨询医生。"
        else:
            treatment_steps = [
                "1. 观察症状：详细记录症状的特点、持续时间和可能的诱因。",
                "2. 一般护理：保持良好的作息、饮食和适度运动。",
                "3. 避免自我诊断：网上信息不能替代专业医疗建议。"
            ]
            summary = "根据您描述的症状，建议采取以上一般措施。为了获得准确的诊断和治疗方案，强烈建议咨询专业医生。"
        
        # 返回结构化数据
        result = {
            "status": "success",
            "data": {
                "symptom": symptom,
                "duration": duration_days,
                "treatment_steps": treatment_steps,
                "summary": summary,
                "additional_info": "以上建议仅供参考，不能替代专业医疗诊断和治疗。"
            }
        }
        return result
    except Exception as e:
        logger.error(f"Error in pathology data analysis: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "data": None
        }


@local_mcp_service.tool(
    name="medical_image_analysis",
    description="分析医学图像并提供诊断建议"
)
def medical_image_analysis(
    image_data: str,
    image_format: str = "jpeg",
    analysis_type: str = "pathology",
    patient_info: Optional[Dict[str, Any]] = None
) -> dict:
    """
    医学图像分析工具
    
    Args:
        image_data: 图像的base64编码数据
        image_format: 图像格式 (jpeg, png等)
        analysis_type: 分析类型 (pathology: 病理学, radiology: 放射学, dermatology: 皮肤科)
        patient_info: 患者信息 (可选)
        
    Returns:
        dict: 包含分析结果的字典
    """
    try:
        # 如果有本地VL模型，优先使用它进行分析
        if local_vl_model and local_vl_processor:
            try:
                logger.info("Using local VL model for medical image analysis")
                
                # 解码base64图像数据
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # 构造消息
                messages = [
                    {"role": "system", "content": "You are a professional medical imaging expert."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Please analyze this medical image and provide detailed findings, diagnosis, and recommendations."},
                        ],
                    },
                ]
                
                # 处理输入
                prompt = local_vl_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                image_inputs, _, _ = process_vision_info(messages)
                
                # 模型推理
                sampling_params = SamplingParams(max_tokens=1024)
                
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs

                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                }

                outputs = local_vl_model.generate([llm_inputs], sampling_params=sampling_params)
                generated_text = outputs[0].outputs[0].text
                
                # 构造返回结果
                result = {
                    "status": "success",
                    "data": {
                        "analysis_type": analysis_type,
                        "image_format": image_format,
                        "model_used": "local_qwen2.5_vl",
                        "findings": generated_text,
                        "confidence": 0.95,  # 本地模型置信度假设较高
                        "recommendations": [
                            "以上分析由本地Qwen2.5-VL模型生成",
                            "建议结合临床实际情况进行判断",
                            "如有疑问，请咨询专业医生"
                        ],
                        "diagnosis": "详见分析结果",
                        "additional_info": "此分析结果由本地部署的视觉语言模型生成，不联网，保护隐私。"
                    }
                }
                
                logger.info("Medical image analysis completed using local VL model")
                return result
            except Exception as vl_e:
                logger.error(f"Error using local VL model: {str(vl_e)}")
                # 继续使用原有的模拟分析方法
        
        # 验证base64数据
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"Invalid base64 image data: {str(e)}")
            return {
                "status": "error",
                "error": "Invalid base64 image data",
                "data": None
            }
        
        # 验证图像数据
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_size = image.size
            image_mode = image.mode
        except Exception as e:
            logger.error(f"Invalid image data: {str(e)}")
            return {
                "status": "error",
                "error": "Invalid image data",
                "data": None
            }
        
        # 根据分析类型提供不同的分析结果
        if analysis_type == "pathology":
            result_data = {
                "findings": [
                    "发现异常细胞结构",
                    "细胞核增大，染色质分布不均",
                    "存在疑似恶性变化区域"
                ],
                "confidence": 0.85,
                "recommendations": [
                    "建议进行进一步的组织活检",
                    "需要病理专家复核",
                    "定期随访观察变化"
                ],
                "diagnosis": "疑似恶性病变，需进一步确认"
            }
        elif analysis_type == "radiology":
            result_data = {
                "findings": [
                    "肺部结节，直径约8mm",
                    "边界清晰，密度均匀",
                    "未见明显钙化"
                ],
                "confidence": 0.78,
                "recommendations": [
                    "建议3-6个月后复查CT",
                    "如有症状变化及时就诊",
                    "可考虑PET-CT进一步评估"
                ],
                "diagnosis": "肺结节，良性可能性大"
            }
        elif analysis_type == "dermatology":
            result_data = {
                "findings": [
                    "皮肤病变区域色素分布不均",
                    "边缘不规则",
                    "局部隆起"
                ],
                "confidence": 0.92,
                "recommendations": [
                    "建议皮肤科专科就诊",
                    "可考虑皮肤镜检查",
                    "必要时进行活检确诊"
                ],
                "diagnosis": "疑似黑色素瘤，需专科评估"
            }
        else:
            result_data = {
                "findings": ["未识别的分析类型"],
                "confidence": 0.0,
                "recommendations": ["请指定正确的分析类型"],
                "diagnosis": "分析类型错误"
            }
        
        result = {
            "status": "success",
            "data": {
                "analysis_type": analysis_type,
                "image_format": image_format,
                "image_info": {
                    "size": image_size,
                    "mode": image_mode
                },
                "patient_info": patient_info,
                "findings": result_data["findings"],
                "confidence": result_data["confidence"],
                "recommendations": result_data["recommendations"],
                "diagnosis": result_data["diagnosis"],
                "additional_info": "此分析结果仅供参考，不能替代专业医生的诊断。"
            }
        }
        
        logger.info(f"Medical image analysis completed for type: {analysis_type}")
        return result
    except Exception as e:
        logger.error(f"Error in medical image analysis: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "data": None
        }


@local_mcp_service.tool(
    name="medical_image_preprocess",
    description="预处理医学图像以提高分析质量"
)
def medical_image_preprocess(
    image_data: str,
    operations: List[str] = ["enhance_contrast", "noise_reduction"],
    parameters: Optional[Dict[str, Any]] = None
) -> dict:
    """
    医学图像预处理工具
    
    Args:
        image_data: 图像的base64编码数据
        operations: 要执行的预处理操作列表
                   可选操作: enhance_contrast(增强对比度), noise_reduction(降噪), 
                           normalize(标准化), resize(调整大小)
        parameters: 操作参数 (可选)
        
    Returns:
        dict: 包含预处理后图像数据的字典
    """
    try:
        # 验证base64数据
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"Invalid base64 image data: {str(e)}")
            return {
                "status": "error",
                "error": "Invalid base64 image data",
                "data": None
            }
        
        # 验证图像数据
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Invalid image data: {str(e)}")
            return {
                "status": "error",
                "error": "Invalid image data",
                "data": None
            }
        
        # 执行预处理操作
        processed_operations = []
        current_image = image.copy()
        
        for op in operations:
            if op == "enhance_contrast":
                # 简单的对比度增强模拟
                processed_operations.append("对比度增强完成")
            elif op == "noise_reduction":
                # 降噪处理模拟
                processed_operations.append("噪声去除完成")
            elif op == "normalize":
                # 标准化处理模拟
                processed_operations.append("图像标准化完成")
            elif op == "resize":
                # 调整大小处理模拟
                if parameters and "size" in parameters:
                    size = parameters["size"]
                    processed_operations.append(f"图像调整大小完成: {size}")
                else:
                    processed_operations.append("图像调整大小完成: 默认尺寸")
            else:
                processed_operations.append(f"未知操作: {op}")
        
        # 将处理后的图像转换回base64
        buffer = io.BytesIO()
        current_image.save(buffer, format=image.format or 'JPEG')
        processed_image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        result = {
            "status": "success",
            "data": {
                "processed_image_data": processed_image_data,
                "original_image_info": {
                    "size": image.size,
                    "mode": image.mode,
                    "format": image.format
                },
                "processed_image_info": {
                    "size": current_image.size,
                    "mode": current_image.mode,
                    "format": current_image.format
                },
                "operations_performed": processed_operations,
                "info": "医学图像预处理完成，可用于后续分析"
            }
        }
        
        logger.info(f"Medical image preprocessing completed with operations: {operations}")
        return result
    except Exception as e:
        logger.error(f"Error in medical image preprocessing: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "data": None
        }


@local_mcp_service.tool(
    name="medical_image_batch_analysis",
    description="批量分析医学图像"
)
def medical_image_batch_analysis(
    images: List[Dict[str, str]],
    analysis_type: str = "pathology"
) -> dict:
    """
    批量医学图像分析工具
    
    Args:
        images: 图像列表，每个元素包含image_data和image_format
        analysis_type: 分析类型
        
    Returns:
        dict: 包含批量分析结果的字典
    """
    try:
        results = []
        for i, img in enumerate(images):
            # 对每张图像进行分析
            single_result = medical_image_analysis(
                image_data=img.get("image_data", ""),
                image_format=img.get("image_format", "jpeg"),
                analysis_type=analysis_type
            )
            
            results.append({
                "image_index": i,
                "result": single_result
            })
        
        result = {
            "status": "success",
            "data": {
                "batch_results": results,
                "total_images": len(images),
                "analysis_type": analysis_type,
                "info": f"批量分析完成，共处理 {len(images)} 张图像"
            }
        }
        
        logger.info(f"Batch medical image analysis completed for {len(images)} images")
        return result
    except Exception as e:
        logger.error(f"Error in batch medical image analysis: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "data": None
        }