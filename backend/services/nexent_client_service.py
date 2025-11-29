import logging
from typing import List, Optional, Dict, Any
from nexent.core.agents.code_agent import CodeAgent
from sdk.nexent.core.agents.nexent_agent import NexentAgent
from sdk.nexent.core.agents.agent_model import ModelConfig, AgentConfig, ToolConfig
from nexent.core.utils.observer import MessageObserver, ProcessType
from threading import Event
from .pathology_document_service import PathologyDocumentService

from .mcp_tools import PathologyImageAnalysisTool

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
""",
            "complex_case": """
You are an experienced pathology expert. Please analyze the following complex case step by step:

1. First, identify the key symptoms and clinical findings
2. Consider differential diagnoses based on current medical literature
3. Evaluate the likelihood of each potential diagnosis
4. Recommend appropriate diagnostic tests if needed
5. Suggest initial treatment approaches based on evidence-based medicine

Case Details: {question}
"""
        }
        pathology_image_tool = PathologyImageAnalysisTool()
        
            # Create tool configurations with more advanced tools
        tool_configs: List[ToolConfig] = [
            ToolConfig(
                class_name="PathologyDataAnalysisTool",
                name="pathology_data_analysis",
                description="Analyze pathology data and provide professional recommendations",
                inputs="symptom: str, duration_days: int = 7",
                output_type="dict",
                params={},
                source="local",
                usage=None
            ),
            ToolConfig(
                class_name="LiteratureSearchTool",
                name="literature_search",
                description="Search latest medical literature for evidence-based answers",
                inputs="query: str, limit: int = 5",
                output_type="dict",
                params={},
                source="local",
                usage=None
            ),
            ToolConfig(
                class_name="PathologyImageAnalysisTool",
                name="pathology_image_analysis",
                description="Analyze pathology images and provide diagnostic suggestions",
                inputs="image_data: str, analysis_type: str = 'diagnosis'",
                output_type="dict",
                params={},
                source="local",
                usage=None
            )
        ]
        
        # Create agent configuration
        agent_config = AgentConfig(
            name="PathologyQA_Agent",
            description="Professional pathology question answering agent",
            prompt_templates=prompt_templates,
            tools=tool_configs,
            max_steps=10,  # Increase steps for complex cases
            model_name="pathology_qa_model",
            provide_run_summary=True,  # Enable summaries
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
            # Determine question complexity to choose appropriate prompt template
            complexity = self._assess_question_complexity(question)
            template_name = "complex_case" if complexity > 0.7 else "default"
            
            # Execute the agent with the question and appropriate template
            self.nexent_agent.agent_run_with_observer(
                query=question,
                prompt_template_name=template_name
            )
            
            # Get the final answer from observer
            final_answer = self.nexent_agent.observer.get_final_answer()
            
            if final_answer:
                return final_answer
            else:
                return "未能获取到答案，请稍后重试"
        except Exception as e:
            logger.error(f"Error while asking pathology question: {str(e)}")
            return f"处理问题时发生错误: {str(e)}"
        
    def _assess_question_complexity(self, question: str) -> float:
        """
        Assess the complexity of a question to determine which prompt template to use
        
        Args:
            question: The question to assess
            
        Returns:
            Complexity score between 0 and 1
        """
        # Simple heuristic for question complexity
        complex_keywords = ['分析', '诊断', '治疗方案', '鉴别诊断', '病理机制', '综合']
        word_count = len(question.split())
        
        complexity_score = min(word_count / 20.0, 1.0)  # Normalize by length
        
        # Add complexity for keywords
        for keyword in complex_keywords:
            if keyword in question:
                complexity_score += 0.2
                
        return min(complexity_score, 1.0)


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
            final_answer = self.vl_agent.observer .get_final_answer()
            
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
        user_id: Optional[str] = None,
        auto_summarize: bool = True
    ) -> Dict[str, Any]:
        """
        Upload pathology documents and create a knowledge base
        
        Args:
            files: List of file objects to upload
            knowledge_base_name: Name for the new knowledge base
            tenant_id: Tenant ID (optional)
            user_id: User ID (optional)
            auto_summarize: Whether to automatically summarize documents (default: True)
            
        Returns:
            Dictionary with upload results
        """
        # Use the default model from configs for embedding
        embedding_model_name = "text-embedding-ada-002"  # Default embedding model
        for config in self.model_configs:
            if config.cite_name == "pathology_qa_model":
                embedding_model_name = config.model_name
                break
                
        result = await self.pathology_service.upload_pathology_documents(
            files=files,
            knowledge_base_name=knowledge_base_name,
            model_name=embedding_model_name,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # if auto_summarize is True, trigger auto summarization
        if auto_summarize and result.get("success") and result.get("knowledge_base_id"):
            try:
                kb_id = result["knowledge_base_id"]
                # trigger auto summarization task (async execution)
                await self._retry_auto_summarize(kb_id)
                result["auto_summarize_triggered"] = True
            except Exception as e:
                logger.error(f"Failed to trigger auto summarization: {str(e)}")
                result["auto_summarize_triggered"] = False
                result["auto_summarize_error"] = str(e)
                
        return result

    async def _retry_auto_summarize(self, knowledge_base_id: int, max_retries: int = 3) -> None:
        """
        Retry auto summarization with exponential backoff
        
        Args:
            knowledge_base_id: ID of the knowledge base to summarize
            max_retries: Maximum number of retry attempts
        """
        import asyncio
        
        for attempt in range(max_retries):
            try:
                await self.pathology_service.auto_summarize_knowledge_base(knowledge_base_id)
                logger.info(f"Auto summarization successful for KB {knowledge_base_id}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Auto summarization failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Auto summarization failed after {max_retries} attempts: {str(e)}")
                    raise



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