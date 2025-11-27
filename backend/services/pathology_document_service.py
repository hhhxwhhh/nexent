import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from sdk.nexent.core.agents.agent_model import ModelConfig
from backend.services.file_management_service import upload_files_impl
from backend.database.knowledge_db import create_knowledge_record
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
            # In the current implementation, the mounting is achieved by ensuring
            # the knowledge base is created with the correct embedding model name
            # which is already handled in _create_knowledge_base
            
            # For future enhancements, this could involve updating agent configurations
            # or establishing explicit relationships in the database
            logger.info(f"Mounted knowledge base {knowledge_base_id} to model {model_config.cite_name}")
            return True
        except Exception as e:
            logger.error(f"Error mounting knowledge base to model: {str(e)}")
            return False