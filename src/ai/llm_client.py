"""
LLM client module for Meetings AI application.
Follows instructions.md requirements - uses global variables from meeting_processor.
"""
import logging
from typing import Optional, Dict, Any, List

# Import safe functions for lazy loading as per instructions.md requirements
from meeting_processor import (
    access_token, embedding_model, llm,  # Keep original imports for compatibility
    get_llm_safe, get_embedding_model_safe, get_access_token_safe, ensure_ai_clients_initialized
)

logger = logging.getLogger(__name__)


def initialize_ai_clients():
    """
    Initialize global AI client instances following instructions.md.
    
    Uses lazy loading - clients are initialized only on first use.
    This significantly improves application startup time.
    
    Returns:
        bool: True if clients are available, False otherwise
    """
    try:
        logger.info("Checking AI clients availability...")
        
        # Use lazy initialization
        if not ensure_ai_clients_initialized():
            logger.error("Failed to initialize AI clients - check API configuration")
            return False
            
        logger.info("AI clients are available and ready")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AI clients: {e}")
        return False


def get_embedding_client():
    """Get the global embedding model instance with lazy loading."""
    return get_embedding_model_safe()


def get_llm_client():
    """Get the global LLM instance with lazy loading."""
    return get_llm_safe()


def get_access_token_client():
    """Get the global access token with lazy loading."""
    return get_access_token_safe()


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using the global embedding model.
    
    Args:
        texts: List of text strings to generate embeddings for
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
        
    Raises:
        Exception: If embedding model is not initialized or generation fails
    """
    try:
        embedding_client = get_embedding_model_safe()
        if embedding_client is None:
            raise Exception("Embedding model not available - check API configuration")
            
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings using the global embedding model
        embeddings = []
        for text in texts:
            if not text or not text.strip():
                # Handle empty text
                logger.warning("Empty text provided for embedding")
                embeddings.append([0.0] * 3072)  # text-embedding-3-large dimension
                continue
                
            try:
                # Use the LangChain embedding model to generate embeddings
                text_embeddings = embedding_client.embed_documents([text.strip()])
                embedding_vector = text_embeddings[0]
                embeddings.append(embedding_vector)
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for text: {e}")
                # Fallback to zero vector
                embeddings.append([0.0] * 3072)
        
        logger.debug(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error in generate_embeddings: {e}")
        raise


def generate_response(prompt: str) -> Optional[str]:
    """
    Generate a text response using the global LLM client.
    
    Args:
        prompt: The prompt text to send to the LLM
        
    Returns:
        Generated response text or None if generation fails
        
    Raises:
        Exception: If LLM is not initialized or generation fails
    """
    try:
        llm_client = get_llm_safe()
        if llm_client is None:
            raise Exception("LLM not available - check API configuration")
            
        logger.debug(f"Generating response for prompt (length: {len(prompt)})")
        
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided for response generation")
            return None
            
        try:
            # Use the LLM client to generate response
            response = llm_client.invoke(prompt.strip())
            
            # Extract content from response
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return None
                
            logger.debug(f"Successfully generated response (length: {len(response_text)})")
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        raise