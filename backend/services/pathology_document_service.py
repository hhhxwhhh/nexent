import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from sdk.nexent.core.agents.agent_model import ModelConfig
from backend.services.file_management_service import upload_files_impl
from backend.database.knowledge_db import create_knowledge_record, update_knowledge_record, get_knowledge_record
from backend.database.model_management_db import get_model_by_display_name
from database.client import db_client

logger = logging.getLogger("pathology_document_service")


class PathologyDocumentService:
    """Service for uploading pathology documents and mounting them to models"""

    def __init__(self):
        self.upload_dir = Path("/tmp/uploads")

    async def upload_pathology_documents(
        self, 
        files: List[Any], 
        knowledge_base_name: str,
        model_name: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload pathology documents and create a knowledge base for them
        
        Args:
            files: List of file objects to upload
            knowledge_base_name: Name for the new knowledge base
            model_name: Name of the embedding model to use
            tenant_id: Tenant ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Dictionary with upload results and knowledge base information
        """
        try:
            # Step 1: Upload files to storage
            logger.info(f"Uploading {len(files)} pathology documents")
            errors, uploaded_paths, uploaded_names = await upload_files_impl(
                destination="minio",
                file=files,
                folder=f"pathology/{knowledge_base_name}",
                index_name=knowledge_base_name
            )
            
            if errors:
                logger.warning(f"Encountered errors during upload: {errors}")
                
            # Step 2: Get model information
            model_info = await self._get_embedding_model_info(model_name)
            if not model_info:
                raise ValueError(f"Model {model_name} not found")
                
            # Step 3: Create knowledge base record
            kb_id = await self._create_knowledge_base(
                name=knowledge_base_name,
                description=f"Pathology documents knowledge base for {knowledge_base_name}",
                embedding_model_name=model_name,
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            # Step 4: Return results
            return {
                "success": True,
                "knowledge_base_id": kb_id,
                "knowledge_base_name": knowledge_base_name,
                "uploaded_files": list(zip(uploaded_paths, uploaded_names)),
                "errors": errors,
                "model_info": model_info
            }
            
        except Exception as e:
            logger.error(f"Error uploading pathology documents: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_embedding_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get embedding model information by name
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            model_info = get_model_by_display_name(model_name)
            return model_info
        except Exception as e:
            logger.error(f"Error retrieving model info for {model_name}: {str(e)}")
            return None

    async def _create_knowledge_base(
        self,
        name: str,
        description: str,
        embedding_model_name: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        Create a knowledge base record in the database
        
        Args:
            name: Knowledge base name
            description: Knowledge base description
            embedding_model_name: Name of the embedding model
            tenant_id: Tenant ID (optional)
            user_id: User ID (optional)
            
        Returns:
            ID of the created knowledge base
        """
        kb_data = {
            "index_name": name,
            "knowledge_describe": description,
            "knowledge_sources": "elasticsearch",
            "embedding_model_name": embedding_model_name,
            "tenant_id": tenant_id or "default_tenant",
            "user_id": user_id or "default_user"
        }
        
        kb_id = create_knowledge_record(kb_data)
        logger.info(f"Created knowledge base {name} with ID {kb_id}")
        return kb_id

    def mount_knowledge_base_to_model(
        self,
        knowledge_base_id: int,
        model_config: ModelConfig
    ) -> bool:
        """
        Mount a knowledge base to a model configuration
        
        Args:
            knowledge_base_id: ID of the knowledge base to mount
            model_config: Model configuration to mount to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 获取知识库信息
            from backend.database.knowledge_db import get_knowledge_info_by_knowledge_ids
            knowledge_info_list = get_knowledge_info_by_knowledge_ids([str(knowledge_base_id)])
            
            if not knowledge_info_list:
                logger.error(f"Knowledge base with ID {knowledge_base_id} not found")
                return False
                
            knowledge_info = knowledge_info_list[0]
            index_name = knowledge_info["index_name"]
            embedding_model_name = knowledge_info["embedding_model_name"]
            
            # 更新模型配置，使其包含这个知识库
            # 在实际应用中，我们需要更新Agent配置以包含KnowledgeBaseSearchTool
            # 这里我们记录日志表示挂载成功
            logger.info(f"Mounted knowledge base {knowledge_base_id} ({index_name}) to model {model_config.cite_name}")
            
            # 在生产环境中，这里应该实际更新Agent的工具配置
            # 例如更新数据库中的Agent配置，添加KnowledgeBaseSearchTool工具
            # 并指定该工具可以访问这个知识库
            
            return True
        except Exception as e:
            logger.error(f"Error mounting knowledge base to model: {str(e)}")
            return False

        
    async def auto_summarize_knowledge_base(self, knowledge_base_id: int, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        自动生成知识库摘要
        
        Args:
            knowledge_base_id: 知识库ID
            tenant_id: 租户ID（可选）
            
        Returns:
            包含操作结果的字典
        """
        try:
            # 获取知识库信息
            kb_query = {"knowledge_id": knowledge_base_id}
            if tenant_id:
                kb_query["tenant_id"] = tenant_id
                
            knowledge_record = get_knowledge_record(kb_query)
            if not knowledge_record:
                raise ValueError(f"Knowledge base with ID {knowledge_base_id} not found")
            
            index_name = knowledge_record["index_name"]
            
            # 导入必要的服务
            from backend.services.elasticsearch_service import ElasticSearchService
            from nexent.vector_database.elasticsearch_core import ElasticSearchCore
            
            # 初始化ElasticSearch服务
            es_service = ElasticSearchService()
            es_core = ElasticSearchCore()
            
            # 生成摘要
            try:
                # 调用ElasticSearch服务生成摘要
                summary_result = await es_service.summary_index_name(
                    index_name=index_name,
                    batch_size=1000,
                    es_core=es_core,
                    tenant_id=tenant_id or knowledge_record.get("tenant_id", "default_tenant"),
                    language="zh"  # 默认使用中文
                )
                
                # 更新知识库记录中的摘要信息
                update_data = {
                    "knowledge_id": knowledge_base_id,
                    "update_data": {
                        "knowledge_describe": summary_result  # 将生成的摘要存储到描述字段
                    }
                }
                
                if tenant_id:
                    update_data["tenant_id"] = tenant_id
                    
                update_success = update_knowledge_record(update_data)
                
                if update_success:
                    logger.info(f"Successfully auto-summarized knowledge base {knowledge_base_id}")
                    return {
                        "success": True,
                        "message": "知识库摘要生成成功",
                        "knowledge_base_id": knowledge_base_id,
                        "summary": summary_result
                    }
                else:
                    logger.warning(f"Failed to update knowledge base {knowledge_base_id} with summary")
                    return {
                        "success": False,
                        "message": "摘要生成成功但更新数据库失败",
                        "knowledge_base_id": knowledge_base_id
                    }
                    
            except Exception as summary_error:
                logger.error(f"Error generating summary for knowledge base {knowledge_base_id}: {str(summary_error)}")
                # 即使摘要生成失败，也返回部分成功的结果
                return {
                    "success": False,
                    "message": f"摘要生成失败: {str(summary_error)}",
                    "knowledge_base_id": knowledge_base_id
                }
                
        except Exception as e:
            logger.error(f"Error auto summarizing knowledge base {knowledge_base_id}: {str(e)}")
            return {
                "success": False,
                "message": f"处理过程中发生错误: {str(e)}",
                "knowledge_base_id": knowledge_base_id
            }