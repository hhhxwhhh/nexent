from fastmcp import FastMCP

# Create MCP server
local_mcp_service = FastMCP("local")

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
        return {
            "status": "error",
            "error": str(e),
            "data": None
        }