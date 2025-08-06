# LegalEase.app Copilot Instructions

## Project Architecture

LegalEase is a **dual-service AI legal assistant** with FastAPI backend (`backend/`) and Streamlit frontend (`frontend/`), containerized with Docker Compose. The core is a **RAG (Retrieval-Augmented Generation) pipeline** specializing in Indian law.

### Key Components

- **Backend (FastAPI)**: `api.py` orchestrates the RAG system, streaming responses via SSE/ndjson
- **RAG System**: `rag_system.py` implements tool-based architecture with vector search, Indian Kanoon API, and web search
- **Database**: SQLite (`database.py`) for user management, chat history, and caching
- **Vector Store**: Pinecone with **1024-dimension sliced embeddings** (`SlicedOpenAIEmbeddings`)

## Critical Development Patterns

### Environment & Configuration

- **Environment variables** are loaded via `load_dotenv()` with fallback paths
- **Sliced embeddings**: OpenAI embeddings are truncated to 1024 dimensions for Pinecone compatibility
- **API Keys**: Requires OpenAI, Groq, Pinecone, Indian Kanoon, and Serper APIs

### RAG Pipeline Architecture

```python
# Tool-based execution pattern
EnhancedLegalRAGSystem -> ToolManager -> [VectorDBLookupTool, IKAPITool, WebSearchTool]
```

1. **Planning LLM** (Groq `deepseek-r1-distill-llama-70b`) determines tool necessity
2. **Parallel tool execution** with prioritized vector search
3. **Context ranking** with token budget management (`_select_docs_within_budget`)
4. **Generation LLM** (OpenAI GPT-4) produces final response

### Streaming Response Pattern

```python
# Backend streaming via SSE (Server-Sent Events)
async def stream_response():
    yield f"data: {json.dumps({'type': 'thinking', 'content': 'Analyzing...'})}\n\n"
    # Tool execution with status updates
    yield f"data: {json.dumps({'type': 'final', 'content': result})}\n\n"
```

Frontend consumes via `eventSource` in JavaScript/Streamlit.

### Database Conventions

- **Row factory**: `dict_factory` converts SQLite rows to dictionaries
- **Connection management**: `get_connection()` with context managers
- **Schema**: Users, conversations, queries, and cached results tables

## Development Workflows

### Local Development

```powershell
# Backend (Terminal 1)
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Frontend (Terminal 2) 
cd frontend
streamlit run app.py
```

### Docker Development

```powershell
# Build and run services
docker-compose up --build -d

# Check health
curl http://localhost:8000/
curl http://localhost:8501/_stcore/health

# View logs
docker-compose logs -f
```

### Vector Store Initialization

```powershell
cd backend
python embed_books.py --books-folder ./books --index-name legal-documents
```

## File-Specific Patterns

### `backend/api.py`
- **Lifespan management**: `@asynccontextmanager` initializes RAG system on startup
- **Security middleware**: TrustedHost, HTTPS redirect, CORS for development
- **SSE endpoints**: All major functions stream responses with intermediate status

### `backend/rag_system.py`
- **Tool inheritance**: `Tool` base class with async `run()` method
- **Content processing**: Specialized extractors for PDFs, YouTube, web pages
- **Token counting**: `tiktoken` for context window management
- **Caching**: Local file system for Indian Kanoon results (`ik_data/`)

### `frontend/app.py`
- **API integration**: All backend calls via `requests` to `API_URL`
- **Streaming UI**: JavaScript `EventSource` for real-time updates
- **Page structure**: Sidebar navigation with feature-specific pages

## Integration Points

### External APIs
- **Pinecone**: Vector similarity search with namespace organization
- **Indian Kanoon**: Legal case search with local caching pattern
- **Serper**: Web search with India geo-targeting (`gl='in'`)
- **OpenAI/Groq**: LLM orchestration with different models for planning vs generation

### Cross-Component Communication
- **Frontend → Backend**: REST API with JSON payloads
- **RAG Tools**: Async execution with structured result dictionaries
- **Database**: SQLite for persistence with async wrappers

## Error Handling Conventions

- **Environment validation**: Startup checks for required API keys with masked logging
- **Graceful degradation**: Tool failures don't crash entire pipeline
- **Health checks**: Docker HEALTHCHECK endpoints for both services

## Performance Patterns

- **Parallel tool execution**: `asyncio.gather()` for concurrent API calls
- **Token budget management**: Intelligent document selection to fit context windows
- **Connection pooling**: Reuse database connections and HTTP clients
- **Content caching**: File-based caching for expensive operations

This codebase prioritizes **legal accuracy** through domain-specific RAG, **real-time feedback** through streaming, and **Indian law specialization** through curated tools and knowledge bases.
