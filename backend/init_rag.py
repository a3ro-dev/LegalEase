#!/usr/bin/env python
import os
import sys
import logging
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from rag_system import EnhancedLegalRAGSystem, MistralTextEmbeddings  # use our Mistral embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    # Try to find .env file automatically
    env_file = find_dotenv()
    if env_file:
        load_dotenv(dotenv_path=env_file)
    else:
        logger.warning("No .env file found. Will try to use environment variables.")

def init_rag_system(index_name=None, namespace="indian-law"):
    """Initialize the RAG system with an existing Pinecone index"""
    
    # Initialize Mistral embeddings to match our Pinecone index dimensions
    embeddings = MistralTextEmbeddings(model=os.getenv("EMBEDDINGS_MODEL", "mistral-embed"))
    
    try:
        # Connect to Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # If no specific index name provided, find available indexes
        if not index_name:
            available_indexes = pc.list_indexes().names()
            if not available_indexes:
                logger.error("No indexes found in your Pinecone account")
                return None
                
            # Use the first available index
            index_name = available_indexes[0]
            logger.info(f"Using first available index: {index_name}")
        
        # Check if the index exists
        available_indexes = pc.list_indexes().names()
        if index_name not in available_indexes:
            logger.error(f"Index '{index_name}' not found. Available indexes: {available_indexes}")
            
            # Try to use any available index
            if available_indexes:
                index_name = available_indexes[0]
                logger.info(f"Falling back to available index: {index_name}")
            else:
                return None
            
        logger.info(f"Connecting to existing Pinecone index: {index_name}")
        index = pc.Index(index_name)
        
        # Get available namespaces
        stats = index.describe_index_stats()
        available_namespaces = list(stats.get('namespaces', {}).keys())
        
        if namespace not in available_namespaces and available_namespaces:
            logger.warning(f"Namespace '{namespace}' not found. Available namespaces: {available_namespaces}")
            if available_namespaces:
                namespace = available_namespaces[0]
                logger.info(f"Using namespace: {namespace}")
        
        # Create vector store
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=namespace if namespace in available_namespaces else None
        )
        
        # Initialize RAG system
        rag_system = EnhancedLegalRAGSystem(vectorstore, enable_web=True)
        logger.info(f"RAG system successfully initialized with index '{index_name}'!")
        if namespace in available_namespaces:
            logger.info(f"Using namespace: '{namespace}'")
        
        return rag_system
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Try to use the specific index from the URL if provided
    # Extract index name from the URL: https://llama-text-embed-v2-index-i8e9zom.svc.aped-4627-b74a.pinecone.io
    specific_index = "llama-text-embed-v2-index-i8e9zom"
    
    rag_system = init_rag_system(index_name=specific_index)
    
    if rag_system:
        # Test with a simple query
        import asyncio
        
        async def test_query():
            question = "What is the procedure for filing RTI in India?"
            logger.info(f"Testing RAG system with query: {question}")
            
            response = await rag_system.query(question)
            logger.info(f"Response received: {response.content[:200]}...")
            
        asyncio.run(test_query())
    else:
        sys.exit(1)
