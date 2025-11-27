import uvicorn
import logging
import warnings
import asyncio
import os
from consts.const import APP_VERSION

warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
load_dotenv()

from apps.base_app import app
from utils.logging_utils import configure_logging, configure_elasticsearch_logging
from services.tool_configuration_service import initialize_tools_on_startup
from services.nexent_client_service import NexentClientService

configure_logging(logging.INFO)
configure_elasticsearch_logging()
logger = logging.getLogger("main_service")

# Initialize Nexent client service
nexent_service = NexentClientService()

async def startup_initialization():
    """
    Perform initialization tasks during server startup
    """
    logger.info("Starting server initialization...")
    logger.info(f"APP version is: {APP_VERSION}")
    try:
        # Initialize tools on startup - service layer handles detailed logging
        await initialize_tools_on_startup()
        
        # Initialize Nexent client
        try:
            # These values should come from environment variables in a production setup
            api_key = os.getenv("NEXENT_API_KEY", "your-default-api-key")
            model_endpoint = os.getenv("NEXENT_MODEL_ENDPOINT", "https://nexent.modelengine.com/api/v1")
            
            nexent_service.initialize_nexent_client(api_key, model_endpoint)
            nexent_service.attach_pathology_model()
            nexent_service.setup_pathology_qa_agent()
            logger.info("Nexent client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Nexent client: {str(e)}")
            logger.warning("Continuing server startup without Nexent client")
        
        logger.info("Server initialization completed successfully!")
            
    except Exception as e:
        logger.error(f"Server initialization failed: {str(e)}")
        # Don't raise the exception to allow server to start even if initialization fails
        logger.warning("Server will continue to start despite initialization issues")


if __name__ == "__main__":
    asyncio.run(startup_initialization())
    uvicorn.run(app, host="0.0.0.0", port=5010, log_level="info")