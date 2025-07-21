"""
LLM client module for Meetings AI application.
Follows instructions.md requirements - uses global variables for LLM setup.
"""
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Global variables as per instructions.md
# These are initialized in the main application startup
access_token = None
embedding_model = None
llm = None


def get_access_token():
    """Get access token - placeholder for organization-specific implementation."""
    # This function should be implemented based on your organization's requirements
    # For development with OpenAI key: return None or key
    # For Azure AD token-based: return token
    import os
    return os.environ.get('OPENAI_API_KEY', None)


def get_embedding_model(token):
    """Get embedding model instance - follows instructions.md pattern."""
    try:
        from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
        import os
        
        # Check if using Azure
        if os.environ.get('AZURE_OPENAI_ENDPOINT'):
            return AzureOpenAIEmbeddings(
                azure_deployment=os.environ.get('AZURE_EMBEDDING_DEPLOYMENT', 'text-embedding-3-large'),
                azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                api_key=token or os.environ.get('AZURE_OPENAI_API_KEY'),
                api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-01')
            )
        else:
            # Use OpenAI directly
            return OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=token or os.environ.get('OPENAI_API_KEY')
            )
    except Exception as e:
        logger.error(f"Error creating embedding model: {e}")
        return None


def get_llm(token):
    """Get LLM instance - follows instructions.md pattern."""
    try:
        from langchain_openai import ChatOpenAI, AzureChatOpenAI
        import os
        
        # Check if using Azure
        if os.environ.get('AZURE_OPENAI_ENDPOINT'):
            return AzureChatOpenAI(
                azure_deployment=os.environ.get('AZURE_LLM_DEPLOYMENT', 'gpt-4o'),
                azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                api_key=token or os.environ.get('AZURE_OPENAI_API_KEY'),
                api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-01'),
                temperature=0.3
            )
        else:
            # Use OpenAI directly
            return ChatOpenAI(
                model="gpt-4o",
                api_key=token or os.environ.get('OPENAI_API_KEY'),
                temperature=0.3
            )
    except Exception as e:
        logger.error(f"Error creating LLM: {e}")
        return None


def initialize_ai_clients():
    """
    Initialize global AI clients as per instructions.md.
    This should be called once during application startup.
    """
    global access_token, embedding_model, llm
    
    try:
        logger.info("Initializing AI clients...")
        
        # Get access token
        access_token = get_access_token()
        
        # Initialize embedding model
        embedding_model = get_embedding_model(access_token)
        if embedding_model:
            logger.info("Embedding model initialized successfully")
        else:
            logger.error("Failed to initialize embedding model")
        
        # Initialize LLM
        llm = get_llm(access_token)
        if llm:
            logger.info("LLM initialized successfully")
        else:
            logger.error("Failed to initialize LLM")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing AI clients: {e}")
        return False


def get_embedding_client():
    """
    Get the global embedding model instance.
    
    Returns:
        Embedding model instance or None
    """
    global embedding_model
    if embedding_model is None:
        logger.warning("Embedding model not initialized. Call initialize_ai_clients() first.")
    return embedding_model


def get_llm_client():
    """
    Get the global LLM instance.
    
    Returns:
        LLM instance or None
    """
    global llm
    if llm is None:
        logger.warning("LLM not initialized. Call initialize_ai_clients() first.")
    return llm


def generate_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors or None if failed
    """
    try:
        embedding_client = get_embedding_client()
        if not embedding_client:
            logger.error("Embedding client not available")
            return None
        
        embeddings = embedding_client.embed_documents(texts)
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None


def generate_response(prompt: str, **kwargs) -> Optional[str]:
    """
    Generate a response using the LLM.
    
    Args:
        prompt: Input prompt
        **kwargs: Additional parameters for the LLM
        
    Returns:
        Generated response or None if failed
    """
    try:
        llm_client = get_llm_client()
        if not llm_client:
            logger.error("LLM client not available")
            return None
        
        response = llm_client.invoke(prompt)
        
        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
            
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return None


def check_ai_availability() -> Dict[str, bool]:
    """
    Check availability of AI services.
    
    Returns:
        Dictionary with availability status
    """
    return {
        'embedding_model': embedding_model is not None,
        'llm': llm is not None,
        'access_token': access_token is not None
    }