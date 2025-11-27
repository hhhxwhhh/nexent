import logging
from typing import List, Optional
from nexent.core.agents.code_agent import CodeAgent
from nexent.sdk.nexent.core.agents.nexent_agent import NexentAgent
from nexent.core.agents.agent_model import ModelConfig, AgentConfig, ToolConfig
from nexent.core.utils.observer import MessageObserver, ProcessType
from threading import Event

logger = logging.getLogger("nexent_client_service")


class NexentClientService:
    """Service for initializing and managing Nexent client and models"""

    def __init__(self):
        self.agent = None
        self.model_configs = []

    def initialize_nexent_client(self, api_key: str, model_endpoint: str) -> None:
        """
        Initialize the Nexent client with basic configuration
        
        Args:
            api_key: API key for accessing the model
            model_endpoint: Endpoint URL for the model service
        """
        logger.info("Initializing Nexent client")
        
        # Create model configuration for pathology QA model
        pathology_model_config = ModelConfig(
            cite_name="pathology_qa_model",
            api_key=api_key,
            model_name="gpt-3.5-turbo",  # Default model, can be changed
            url=model_endpoint,
            temperature=0.2,
            top_p=0.95
        )
        
        self.model_configs.append(pathology_model_config)
        logger.info("Nexent client initialized with model config")

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
        observer = MessageObserver(process_type=ProcessType.AGENT_RUN)
        
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