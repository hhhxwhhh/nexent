import logging
from typing import List, Optional, Dict, Any
from nexent.core.agents.code_agent import CodeAgent
from sdk.nexent.core.agents.nexent_agent import NexentAgent
from sdk.nexent.core.agents.agent_model import ModelConfig, AgentConfig, ToolConfig
from nexent.core.utils.observer import MessageObserver, ProcessType
from threading import Event
from .pathology_document_service import PathologyDocumentService

logger = logging.getLogger("nexent_client_service")

from consts.const import MODEL_ENGINE_HOST, MODEL_ENGINE_APIKEY


class NexentClientService:
    """Service for initializing and managing Nexent client and models"""

    def __init__(self):
        self.agent = None
        self.model_configs = []
        self.pathology_service = PathologyDocumentService()
        self.model_engine_host = MODEL_ENGINE_HOST
        self.model_engine_apikey = MODEL_ENGINE_APIKEY

    def initialize_nexent_client(self, api_key: str = None, model_endpoint: str = None) -> None:
        """
        Initialize the Nexent client with basic configuration
        
        Args:
            api_key: API key for accessing the model (defaults to env var)
            model_endpoint: Endpoint URL for the model service (defaults to env var)
        """
        logger.info("Initializing Nexent client")
        
        # 使用传入的参数或环境变量
        actual_api_key = api_key or self.model_engine_apikey
        actual_model_endpoint = model_endpoint or self.model_engine_host
        
        if not actual_api_key or not actual_model_endpoint:
            raise ValueError("API key and model endpoint must be provided either as arguments or environment variables")
        
        # Create model configuration for pathology QA model
        pathology_model_config = ModelConfig(
            cite_name="pathology_qa_model",
            api_key=actual_api_key,
            model_name="gpt-3.5-turbo",  # Default model, can be changed
            url=actual_model_endpoint,
            temperature=0.2,
            top_p=0.95
        )
        
        self.model_configs.append(pathology_model_config)
        logger.info("Nexent client initialized with model config")

    def initialize_local_vl_model(self, model_path: str = None) -> None:
        """
        Initialize the local vision-language model (Qwen2.5-VL)
        
        Args:
            model_path: Path to the downloaded Qwen2.5-VL model
        """
        logger.info("Initializing local vision-language model")
        
        if not model_path:
            model_path = "/Users/wang/model_engine/Qwen2.5-VL-7B-Instruct"
        
        # Create model configuration for local VL model
        vl_model_config = ModelConfig(
            cite_name="local_vl_model",
            api_key="",  # Local model doesn't need API key
            model_name="qwen2.5-vl-7b-instruct",  # Model identifier
            url=f"local://{model_path}",  # Special URL scheme for local models
            temperature=0.1,
            top_p=0.9
        )
        
        self.model_configs.append(vl_model_config)
        logger.info(f"Local VL model initialized with path: {model_path}")

    def check_model_engine_connectivity(self) -> bool:
        """
        Check connectivity to ModelEngine platform
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            import aiohttp
            import asyncio
            
            # 检查必要的配置是否存在
            if not self.model_engine_apikey or not self.model_engine_host:
                logger.error("ModelEngine API key or host not configured")
                return False
            
            async def _check_connectivity():
                headers = {'Authorization': f'Bearer {self.model_engine_apikey}'}
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.model_engine_host}/open/router/v1/health", headers=headers) as response:
                        return response.status == 200
            
            # Run the async function in a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(_check_connectivity())
            if result:
                logger.info("Successfully connected to ModelEngine")
            else:
                logger.warning("Failed to connect to ModelEngine")
            return result
        except Exception as e:
            logger.error(f"Failed to check ModelEngine connectivity: {str(e)}")
            return False

        
    async def list_available_models_async(self) -> List[Dict[str, Any]]:
        """
        Asynchronously list available models from ModelEngine
        
        Returns:
            List of available models
        """
        try:
            import aiohttp
            
            headers = {'Authorization': f'Bearer {self.model_engine_apikey}'}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.model_engine_host}/open/router/v1/models", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', [])
                    else:
                        logger.error(f"Failed to get models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Failed to get models from ModelEngine: {str(e)}")
            return []
            
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Synchronously list available models from ModelEngine
        
        Returns:
            List of available models
        """
        try:
            import aiohttp
            import asyncio
            
            async def _get_models():
                headers = {'Authorization': f'Bearer {self.model_engine_apikey}'}
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.model_engine_host}/open/router/v1/models", headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get('data', [])
                        else:
                            logger.error(f"Failed to get models: {response.status}")
                            return []
            
            # Run the async function in a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(_get_models())
            return result
        except Exception as e:
            logger.error(f"Failed to get models from ModelEngine: {str(e)}")
            return []

    def select_model(self, model_name: str) -> bool:
        """
        Select and configure a specific model from ModelEngine
        
        Args:
            model_name: Name of the model to select
            
        Returns:
            bool: True if model selection successful, False otherwise
        """
        try:
            # 获取可用模型列表
            available_models = self.list_available_models()
            
            # 查找指定的模型
            selected_model = None
            for model in available_models:
                if model.get('name') == model_name:
                    selected_model = model
                    break
            
            if not selected_model:
                logger.error(f"Model {model_name} not found in available models")
                return False
            
            # 更新模型配置
            for config in self.model_configs:
                if config.cite_name == "pathology_qa_model":
                    config.model_name = model_name
                    # 可以根据模型类型设置其他参数
                    if selected_model.get('type') == 'chat':
                        config.temperature = 0.7
                        config.top_p = 0.9
                    break
                    
            logger.info(f"Selected model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to select model {model_name}: {str(e)}")
            return False

    def attach_pathology_model(self, model_name: Optional[str] = None) -> None:
        """
        Attach a model for pathology question answering
        
        Args:
            model_name: Specific model name to use, defaults to gpt-3.5-turbo
        """
        if not self.model_configs:
            raise RuntimeError("Nexent client not initialized. Call initialize_nexent_client first.")
        
        model_to_use = model_name if model_name else "gpt-3.5-turbo"
        
        # Update model name in config
        for config in self.model_configs:
            if config.cite_name == "pathology_qa_model":
                config.model_name = model_to_use
                break
                
        logger.info(f"Attached pathology model: {model_to_use}")

    def setup_pathology_qa_agent(self) -> None:
        """
        Setup the pathology question answering agent with appropriate prompt template
        """
        if not self.model_configs:
            raise RuntimeError("Nexent client not initialized. Call initialize_nexent_client first.")
        
        # Create observer for the agent
        observer = MessageObserver()
        
        # Create stop event
        stop_event = Event()
        
        # Define prompt template for pathology QA
        prompt_templates = {
            "default": """
You are a professional pathology expert. Please answer the following question based on medical literature and clinical guidelines.
Requirements:
1. Be concise and accurate
2. Reference authoritative sources such as WHO classification, white papers, etc.
3. If you're unsure, please indicate so honestly

Question: {question}
"""
        }
        
        # Create tool configurations (empty for now, can be extended)
        tool_configs: List[ToolConfig] = []
        
        # Create agent configuration
        agent_config = AgentConfig(
            name="PathologyQA_Agent",
            description="Professional pathology question answering agent",
            prompt_templates=prompt_templates,
            tools=tool_configs,
            max_steps=5,
            model_name="pathology_qa_model",
            provide_run_summary=False,
            managed_agents=[]
        )
        
        # Create the agent
        self.nexent_agent = NexentAgent(
            observer=observer,
            model_config_list=self.model_configs,
            stop_event=stop_event
        )
        
        # Create the core agent using the configuration
        core_agent = self.nexent_agent.create_single_agent(agent_config)
        self.nexent_agent.set_agent(core_agent)
        
        logger.info("Pathology QA agent setup completed")

    def setup_vl_agent(self) -> None:
        """
        Setup the vision-language agent with appropriate prompt template for image analysis
        """
        if not self.model_configs:
            raise RuntimeError("Nexent client not initialized. Call initialize_nexent_client first.")
        
        # Create observer for the agent
        observer = MessageObserver()
        
        # Create stop event
        stop_event = Event()
        
        # Define prompt template for VL tasks
        prompt_templates = {
            "default": """
You are a professional medical imaging expert. Please analyze the provided medical image and provide a detailed description.
Requirements:
1. Describe the image content accurately
2. Identify any abnormalities or notable features
3. Provide relevant medical insights
4. If you're unsure, please indicate so honestly

Image Analysis Task: {task}
"""
        }
        
        # Create tool configurations for VL tasks
        tool_configs: List[ToolConfig] = [
            ToolConfig(
                class_name="MedicalImageAnalysisTool",
                name="medical_image_analysis",
                description="Analyze medical images and provide diagnostic suggestions",
                inputs="image_data: str, image_format: str, analysis_type: str",
                output_type="dict",
                params={},
                source="local",
                usage=None
            )
        ]
        
        # Create agent configuration
        agent_config = AgentConfig(
            name="VisionLanguage_Agent",
            description="Professional vision-language agent for medical image analysis",
            prompt_templates=prompt_templates,
            tools=tool_configs,
            max_steps=5,
            model_name="local_vl_model",  # Use the local VL model
            provide_run_summary=False,
            managed_agents=[]
        )
        
        # Create the agent
        self.vl_agent = NexentAgent(
            observer=observer,
            model_config_list=self.model_configs,
            stop_event=stop_event
        )
        
        # Create the core agent using the configuration
        core_agent = self.vl_agent.create_single_agent(agent_config)
        self.vl_agent.set_agent(core_agent)
        
        logger.info("Vision-Language agent setup completed")

    def ask_pathology_question(self, question: str) -> str:
        """
        Ask a pathology-related question and get an answer
        
        Args:
            question: The pathology question to ask
            
        Returns:
            Answer from the model
        """
        if not hasattr(self, 'nexent_agent') or not self.nexent_agent.agent:
            raise RuntimeError("Agent not initialized. Call setup_pathology_qa_agent first.")
            
        logger.info(f"Asking pathology question: {question}")
        
        try:
            # Execute the agent with the question
            self.nexent_agent.agent_run_with_observer(query=question)
            
            # Get the final answer from observer
            final_answer = self.nexent_agent.observer.get_final_answer()
            
            if final_answer:
                return final_answer
            else:
                return "未能获取到答案，请稍后重试"
        except Exception as e:
            logger.error(f"Error while asking pathology question: {str(e)}")
            return f"处理问题时发生错误: {str(e)}"

    def analyze_medical_image(self, image_data: str, task: str = "Please describe this medical image") -> str:
        """
        Analyze a medical image using the local VL model
        
        Args:
            image_data: Base64 encoded image data
            task: Description of the analysis task
            
        Returns:
            Analysis result from the model
        """
        if not hasattr(self, 'vl_agent') or not self.vl_agent.agent:
            raise RuntimeError("VL Agent not initialized. Call setup_vl_agent first.")
            
        logger.info(f"Analyzing medical image with task: {task}")
        
        try:
            # Execute the agent with the image and task
            full_query = f"Image data: {image_data[:50]}... Task: {task}"  # Truncate image data for logging
            self.vl_agent.agent_run_with_observer(query=full_query)
            
            # Get the final answer from observer
            final_answer = self.vl_agent.observer.get_final_answer()
            
            if final_answer:
                return final_answer
            else:
                return "未能获取到图像分析结果，请稍后重试"
        except Exception as e:
            logger.error(f"Error while analyzing medical image: {str(e)}")
            return f"处理图像时发生错误: {str(e)}"

    async def upload_pathology_documents(
        self,
        files: List[Any],
        knowledge_base_name: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload pathology documents and create a knowledge base
        
        Args:
            files: List of file objects to upload
            knowledge_base_name: Name for the new knowledge base
            tenant_id: Tenant ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Dictionary with upload results
        """
        # Use the default model from configs for embedding
        embedding_model_name = "text-embedding-ada-002"  # Default embedding model
        for config in self.model_configs:
            if config.cite_name == "pathology_qa_model":
                embedding_model_name = config.model_name
                break
                
        return await self.pathology_service.upload_pathology_documents(
            files=files,
            knowledge_base_name=knowledge_base_name,
            model_name=embedding_model_name,
            tenant_id=tenant_id,
            user_id=user_id
        )

    def mount_knowledge_base_to_current_model(self, knowledge_base_id: int) -> bool:
        """
        Mount a knowledge base to the current pathology QA model
        
        Args:
            knowledge_base_id: ID of the knowledge base to mount
            
        Returns:
            True if successful, False otherwise
        """
        # Find the pathology QA model config
        pathology_model_config = None
        for config in self.model_configs:
            if config.cite_name == "pathology_qa_model":
                pathology_model_config = config
                break
                
        if not pathology_model_config:
            logger.error("Pathology QA model not found")
            return False
            
        return self.pathology_service.mount_knowledge_base_to_model(
            knowledge_base_id=knowledge_base_id,
            model_config=pathology_model_config
        )