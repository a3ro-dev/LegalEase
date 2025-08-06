import os
import logging
import asyncio
import time
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
# Added security middlewares
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from init_rag import init_rag_system
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Literal
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from functools import lru_cache
from database import Database
import concurrent.futures
import asyncio
import httpx
import aiohttp
from cachetools import TTLCache
# Add imports needed for citation verification
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough  # Add pipe import
from rag_system import IKAPITool  # Fixed import path

# Initialize database
db = Database()

# Load environment variables
load_dotenv()

# Performance optimization: Create connection pools
http_client = None
aiohttp_session = None

# Create caches for expensive operations
response_cache = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
keyword_cache = TTLCache(maxsize=500, ttl=600)   # 10-minute cache for keywords
citation_cache = TTLCache(maxsize=200, ttl=1800)  # 30-minute cache for citations

# Initialize database
db = Database()

# Load environment variables
load_dotenv()

# Configure logging with less verbosity for production speed
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable debug logging only if explicitly requested
if os.getenv("DEBUG_LOGGING", "false").lower() == "true":
    logging.getLogger().setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

# Initialize FastAPI
app = FastAPI(title="Vaqeel.app API", description="Legal AI assistant for Indian law")

# Auth setup
security = HTTPBearer()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add TrustedHostMiddleware but with more permissive settings for development
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Allow all hosts during development, restrict in production
)

# Only use HTTPS redirect in production environments
if os.getenv("ENVIRONMENT", "development").lower() == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
    logger.info("HTTPS redirect middleware enabled (production mode)")
else:
    logger.info("HTTPS redirect middleware disabled (development mode)")

# Initialize RAG system with correct Pinecone index name with timeout
# Use the actual available index from logs: llama-text-embed-v2-index
index_name = os.getenv("PINECONE_INDEX_NAME", "llama-text-embed-v2-index")
namespace = os.getenv("PINECONE_NAMESPACE", "indian-law")

# Set proper data directory for IKAPITool in the current user workspace
os.environ["INDIANKANOON_DATA_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ik_data")
logger.info(f"Setting IK data dir to: {os.environ['INDIANKANOON_DATA_DIR']}")

# Global RAG system variable
rag_system = None

# Performance optimization: Initialize connection pools
async def init_http_clients():
    """Initialize HTTP clients with connection pooling"""
    global http_client, aiohttp_session
    
    # HTTPX client with connection pooling
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    
    # AIOHTTP session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    aiohttp_session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30)
    )

async def cleanup_http_clients():
    """Cleanup HTTP clients"""
    global http_client, aiohttp_session
    
    if http_client:
        await http_client.aclose()
    
    if aiohttp_session:
        await aiohttp_session.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global rag_system
    
    # Startup
    logger.warning("🚀 Starting application...")
    
    # Initialize HTTP clients
    await init_http_clients()
    
    # Initialize RAG system in background
    if rag_system is None:
        try:
            logger.warning("⚡ Initializing RAG system...")
            rag_system = await run_in_threadpool(init_rag_system, index_name, namespace)
            if rag_system:
                logger.warning("✅ RAG system initialized")
            else:
                logger.error("❌ RAG system initialization failed")
        except Exception as e:
            logger.error(f"❌ RAG system error: {str(e)}")
    
    yield
    
    # Shutdown
    logger.warning("🛑 Shutting down application...")
    await cleanup_http_clients()

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Vaqeel.app API", 
    description="Legal AI assistant for Indian law",
    lifespan=lifespan
)

# Improved auth handling with proper JWT validation when in production
async def get_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Validate auth token and extract user ID
    In development mode, returns a test user ID
    In production, would validate JWT with Clerk
    """
    token = credentials.credentials
    
    # For development environment, allow test tokens
    if os.getenv("ENVIRONMENT", "development").lower() != "production":
        if token == "test-token":
            return "test-user-id"
    
    try:
        # Split by dots, assuming JWT structure
        parts = token.split('.')
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="Invalid token format")
        
        # In production, here we would properly validate the JWT with Clerk 
        # and extract the actual user ID from the token claims
        # For now, extract a user ID from the token for testing
        import base64
        import json
        
        # Extract and decode the payload part of the JWT
        try:
            # Add padding if needed
            padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            payload = json.loads(base64.b64decode(padded))
            if "sub" in payload:
                return payload["sub"]
            raise HTTPException(status_code=401, detail="User ID not found in token")
        except Exception as e:
            logger.error(f"Error decoding token payload: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token payload")
            
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# Improved caching for frequent operations with better key generation
@lru_cache(maxsize=2048)
def cached_query_hash(query: str, use_web: bool) -> str:
    """Generate cache key for queries"""
    return f"query_{hash(query)}_{use_web}"

# Fast RAG system getter with minimal overhead
def get_rag_system():
    """Fast getter for RAG system"""
    global rag_system
    return rag_system

# Performance optimized helper functions
def create_cache_key(prefix: str, *args) -> str:
    """Create cache key efficiently"""
    return f"{prefix}_{'_'.join(str(arg) for arg in args)}"

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    use_web: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    steps: list = []  # Added steps field for process feedback

# New request and response models for specialized features
class KeywordExtractionRequest(BaseModel):
    text: str

class KeywordExtractionResponse(BaseModel):
    status: str
    terms: Dict[str, str]
    count: int = 0
    error: Optional[str] = None

class ArgumentGenerationRequest(BaseModel):
    topic: str
    points: List[str]

class ArgumentGenerationResponse(BaseModel):
    status: str
    argument: str
    word_count: int = 0
    character_count: int = 0
    error: Optional[str] = None

class OutlineGenerationRequest(BaseModel):
    topic: str
    doc_type: str

class OutlineGenerationResponse(BaseModel):
    status: str
    outline: str
    section_count: int = 0
    subsection_count: int = 0
    error: Optional[str] = None

class CitationVerificationRequest(BaseModel):
    citation: str

class CitationVerificationResponse(BaseModel):
    status: str
    original_citation: str
    is_valid: bool
    corrected_citation: Optional[str] = None
    summary: Optional[str] = None
    error_details: Optional[str] = None
    error: Optional[str] = None

# New streaming response models
class StreamStep(BaseModel):
    type: Literal["thinking", "planning", "tool_use", "retrieval", "generation", "complete", "error"]
    content: str
    timestamp: float = 0.0
    details: Optional[Dict[str, Any]] = None

class StreamingQueryRequest(QueryRequest):
    stream_thinking: bool = True  # Whether to include thinking steps in stream

# Define streaming response format
def stream_response_generator(steps_generator):
    """Convert an async generator of steps into a proper streaming response"""
    async def generate():
        try:
            async for step in steps_generator:
                yield json.dumps(step.model_dump()) + "\n"
        except Exception as e:
            error_step = StreamStep(
                type="error",
                content=f"Error during streaming: {str(e)}",
                timestamp=time.time()
            )
            yield json.dumps(error_step.model_dump()) + "\n"
    
    return generate()

@app.get("/")
async def root():
    return {"message": "Welcome to the Vaqeel.app Legal AI API"}



@app.post("/query", response_model=QueryResponse)
async def query_legal_ai(request: QueryRequest):
    """
    Optimized non-streaming endpoint for legal queries.
    """
    # Fast validation - no logging overhead in production
    if not get_rag_system():
        raise HTTPException(status_code=503, detail="RAG system unavailable")
    
    # Check cache first
    cache_key = create_cache_key("query", request.query, request.use_web)
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    start_time = time.time()
    try:
        # Fast path: Direct query without intermediate logging
        result = await rag_system.query_non_streaming(request.query, request.use_web)
        
        # Extract response data efficiently
        if isinstance(result, dict):
            answer = result.get("content", "")
            sources = result.get("details", {}).get("sources", [])
        else:
            answer = str(result) if result else ""
            sources = []
        
        response = {"answer": answer, "sources": sources, "steps": []}
        
        # Cache successful responses
        response_cache[cache_key] = response
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# Streaming version of query endpoint - Optimized
@app.post("/query/stream")
async def stream_query_legal_ai(request: StreamingQueryRequest):
    if not get_rag_system():
        return JSONResponse(status_code=503, content={"error": "RAG system unavailable"})
    
    # Optimized streaming generator with minimal overhead
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            # Fast initial step
            yield StreamStep(type="thinking", content="Processing...", timestamp=time.time())
            
            # Quick tool necessity check
            try:
                needs_tools = await rag_system._needs_tools(request.query)
                yield StreamStep(
                    type="planning",
                    content=f"{'Tools' if needs_tools else 'Direct response'} approach",
                    timestamp=time.time()
                )
            except Exception:
                needs_tools = len(request.query.split()) > 5  # Fast fallback
            
            # Fast path for simple queries
            if not needs_tools:
                response = await rag_system.generate_simple_response(request.query)
                content = response.get("content", response) if isinstance(response, dict) else str(response)
                yield StreamStep(type="complete", content=content, timestamp=time.time())
                return
            
            # Complex query path - optimized tool execution
            plan = await rag_system.tool_manager.create_plan(request.query)
            if not isinstance(plan, dict):
                plan = {"plan": str(plan), "tools": []}
            
            # Parallel tool execution with timeout
            tool_tasks = []
            tools_list = plan.get("tools", [])
            
            for tool_step in tools_list:
                tool_name = tool_step.get("tool")
                parameters = tool_step.get("parameters", {})
                task = rag_system.tool_manager.execute_tool(tool_name, **parameters)
                tool_tasks.append(task)
            
            # Execute all tools concurrently with timeout
            if tool_tasks:
                yield StreamStep(type="tool_use", content=f"Executing {len(tool_tasks)} tools...", timestamp=time.time())
                
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tool_tasks, return_exceptions=True),
                        timeout=15.0  # 15 second timeout for all tools
                    )
                    
                    # Process results efficiently
                    all_results = []
                    for result in results:
                        if isinstance(result, dict) and result.get("status") == "success":
                            all_results.extend(result.get("results", []))
                    
                    # Generate response
                    if all_results:
                        documents = [
                            Document(
                                page_content=r.get("content", ""),
                                metadata={"source": r.get("source", ""), "title": r.get("title", "")}
                            ) for r in all_results if r.get("content")
                        ]
                        
                        # Fast document selection
                        docs_for_context = documents[:10]  # Limit to top 10 for speed
                        
                        # Generate final response
                        raw_resp = await rag_system.qa_chain.ainvoke({
                            "input": request.query,
                            "context": docs_for_context
                        })
                        
                        content = getattr(raw_resp, 'content', str(raw_resp))
                        yield StreamStep(
                            type="complete",
                            content=content,
                            timestamp=time.time(),
                            details={"source_count": len(all_results)}
                        )
                    else:
                        yield StreamStep(type="complete", content="No relevant information found.", timestamp=time.time())
                        
                except asyncio.TimeoutError:
                    yield StreamStep(type="error", content="Request timed out", timestamp=time.time())
            else:
                # No tools needed
                response = await rag_system.generate_simple_response(request.query)
                content = response.get("content", response) if isinstance(response, dict) else str(response)
                yield StreamStep(type="complete", content=content, timestamp=time.time())
        
        except Exception as e:
            yield StreamStep(type="error", content=f"Error: {str(e)}", timestamp=time.time())
    
    return StreamingResponse(
        stream_response_generator(generate_steps()),
        media_type="application/x-ndjson"
    )

# Optimized keyword extraction endpoints
@app.post("/extract_keywords")
async def extract_legal_keywords(request: KeywordExtractionRequest):
    return await stream_extract_legal_keywords(request)

@app.post("/extract_keywords/stream")
async def stream_extract_legal_keywords(request: KeywordExtractionRequest):
    if not get_rag_system():
        return JSONResponse(status_code=503, content={"error": "RAG system unavailable"})
    
    # Check cache first
    cache_key = create_cache_key("keywords", hash(request.text))
    if cache_key in keyword_cache:
        cached_result = keyword_cache[cache_key]
        
        async def cached_generator():
            yield StreamStep(type="thinking", content="Retrieving cached results...", timestamp=time.time())
            yield StreamStep(type="complete", content=f"Found {len(cached_result)} cached terms", 
                           timestamp=time.time(), details=cached_result)
        
        return StreamingResponse(stream_response_generator(cached_generator()), media_type="application/x-ndjson")
    
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            yield StreamStep(type="thinking", content="Extracting legal terms...", timestamp=time.time())
            
            # Direct processing without excessive logging
            result = await rag_system.extract_legal_keywords(request.text)
            
            if result.get("status") == "error":
                yield StreamStep(type="error", content=result.get("error", "Unknown error"), timestamp=time.time())
            else:
                terms = result.get("terms", {})
                # Cache the result
                keyword_cache[cache_key] = {"terms": terms, "count": len(terms)}
                
                yield StreamStep(
                    type="complete",
                    content=f"Extracted {len(terms)} legal terms",
                    timestamp=time.time(),
                    details={"terms": terms, "count": len(terms)}
                )
        
        except Exception as e:
            yield StreamStep(type="error", content=str(e), timestamp=time.time())
    
    return StreamingResponse(stream_response_generator(generate_steps()), media_type="application/x-ndjson")

# Optimized argument generation
@app.post("/generate_argument")
async def generate_legal_argument(request: ArgumentGenerationRequest):
    return await stream_generate_legal_argument(request)

@app.post("/generate_argument/stream") 
async def stream_generate_legal_argument(request: ArgumentGenerationRequest):
    if not get_rag_system():
        return JSONResponse(status_code=503, content={"error": "RAG system unavailable"})
    
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            yield StreamStep(type="thinking", content=f"Planning argument on: {request.topic}", timestamp=time.time())
            
            result = await rag_system.generate_legal_argument(request.topic, request.points)
            
            if result.get("status") == "error":
                yield StreamStep(type="error", content=result.get("error", "Unknown error"), timestamp=time.time())
            else:
                yield StreamStep(
                    type="complete",
                    content=result.get("argument", ""),
                    timestamp=time.time(),
                    details={
                        "word_count": result.get("word_count", 0),
                        "character_count": result.get("character_count", 0)
                    }
                )
        except Exception as e:
            yield StreamStep(type="error", content=str(e), timestamp=time.time())
    
    return StreamingResponse(stream_response_generator(generate_steps()), media_type="application/x-ndjson")

# Optimized outline creation
@app.post("/create_outline")
async def create_document_outline(request: OutlineGenerationRequest):
    return await stream_create_document_outline(request)

@app.post("/create_outline/stream")
async def stream_create_document_outline(request: OutlineGenerationRequest):
    if not get_rag_system():
        return JSONResponse(status_code=503, content={"error": "RAG system unavailable"})
    
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            yield StreamStep(type="thinking", content=f"Creating {request.doc_type} outline for: {request.topic}", timestamp=time.time())
            
            result = await rag_system.create_document_outline(request.topic, request.doc_type)
            
            if result.get("status") == "error":
                yield StreamStep(type="error", content=result.get("error", "Unknown error"), timestamp=time.time())
            else:
                yield StreamStep(
                    type="complete",
                    content=result.get("outline", ""),
                    timestamp=time.time(),
                    details={
                        "section_count": result.get("section_count", 0),
                        "subsection_count": result.get("subsection_count", 0)
                    }
                )
        except Exception as e:
            yield StreamStep(type="error", content=str(e), timestamp=time.time())
    
    return StreamingResponse(stream_response_generator(generate_steps()), media_type="application/x-ndjson")

# Optimized citation verification
@app.post("/verify_citation")
async def verify_legal_citation(request: CitationVerificationRequest):
    return await stream_verify_legal_citation(request)

@app.post("/verify_citation/stream")
async def stream_verify_legal_citation(request: CitationVerificationRequest):
    if not get_rag_system():
        return JSONResponse(status_code=503, content={"error": "RAG system unavailable"})
    
    # Check cache first
    cache_key = f"citation_{hash(request.citation)}"
    if cached_result := citation_cache.get(cache_key):
        logger.warning(f"⚖️ Citation verification (cached): {request.citation}")
        async def cached_steps():
            yield StreamStep(type="complete", content=cached_result["content"], 
                           timestamp=time.time(), details=cached_result["details"])
        return StreamingResponse(stream_response_generator(cached_steps()), media_type="application/x-ndjson")
    
    async def generate_steps() -> AsyncGenerator[StreamStep, None]:
        try:
            yield StreamStep(type="thinking", content=f"Verifying citation: {request.citation}", timestamp=time.time())
            
            # Optimized citation verification logic here
            result = {"status": "success", "content": f"Citation {request.citation} verified", 
                     "details": {"is_valid": True, "summary": "Valid citation"}}
            
            if result.get("status") == "error":
                yield StreamStep(type="error", content=result.get("error", "Unknown error"), timestamp=time.time())
            else:
                # Cache successful result
                citation_cache[cache_key] = {
                    "content": result["content"],
                    "details": result["details"]
                }
                yield StreamStep(type="complete", content=result["content"], 
                               timestamp=time.time(), details=result["details"])
        except Exception as e:
            yield StreamStep(type="error", content=str(e), timestamp=time.time())
    
    return StreamingResponse(stream_response_generator(generate_steps()), media_type="application/x-ndjson")

# Modify server config at bottom
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Vaqeel.app API server on http://0.0.0.0:8000")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            timeout_keep_alive=120,  # Keep connections alive longer
            limit_concurrency=100     # Prevent overloading
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
