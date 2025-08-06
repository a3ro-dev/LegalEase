import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader, YoutubeLoader
from langchain_community.document_transformers import Html2TextTransformer
from bs4 import BeautifulSoup
import httpx
import re
import tldextract
from typing import List, Dict, Any, Optional, Union, Callable
import json
import asyncio
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
import tempfile
import fitz  # PyMuPDF
from pytube import YouTube
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from ikapi import IKApi, FileStorage
import logging
import requests  # Added for Serper API and Mistral API
import urllib.parse  # Added for URL encoding
from langchain.chains.combine_documents import create_stuff_documents_chain
import tiktoken  # Add this import for token counting
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.embeddings import Embeddings
from mistralai import Mistral  # Import Mistral client

# Load environment variables - try absolute path first, then fallback to auto-find
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Pinecone client
def get_pinecone_client():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc

# Replace the embeddings initialization with Mistral embeddings
class MistralTextEmbeddings(Embeddings):
    """Mistral embeddings that provide 1024-dimension vectors."""
    
    def __init__(self, api_key: str = None, model: str = "mistral-embed"):
        """Initialize Mistral embeddings client.
        
        Args:
            api_key: Mistral API key. If None, will use MISTRAL_API_KEY from environment.
            model: Mistral embedding model name. Default is 'mistral-embed'.
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        self.client = Mistral(api_key=self.api_key)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text and return 1024-dimensional vector"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                inputs=[text]
            )
            # Mistral embeddings are already 1024 dimensions
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents and return 1024-dimensional vectors"""
        try:
            # Process in chunks to avoid API limits
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    model=self.model,
                    inputs=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

embeddings = MistralTextEmbeddings(model=os.getenv("EMBEDDINGS_MODEL"))

# Helper function to create LLMs from environment variables
def create_llm_from_env(llm_type: str) -> ChatGroq:
    """Create a ChatGroq LLM instance from environment variables.
    
    Args:
        llm_type: Type of LLM ('general', 'specialist', 'planning', 'evaluation')
    
    Returns:
        Configured ChatGroq instance
    """
    env_prefix = llm_type.upper() + "_LLM"
    model = os.getenv(f"{env_prefix}_MODEL")
    temperature = float(os.getenv(f"{env_prefix}_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv(f"{env_prefix}_MAX_TOKENS", "2048"))
    
    return ChatGroq(
        temperature=temperature,
        model_name=model,
        max_tokens=max_tokens
    )

# Document processing pipeline
def create_vector_store(doc_paths: List[str], index_name: str = os.getenv("DEFAULT_INDEX_NAME"), namespace: str = os.getenv("DEFAULT_NAMESPACE"), books_folder: str = None):
    """
    Create or update a production vector store with documents, prioritizing books folder PDFs
    """
    # First prioritize books if specified
    all_docs = []
    books_docs = []
    other_docs = []
    
    # Process books folder first if specified
    if books_folder and os.path.isdir(books_folder):
        logger.info(f"Processing books folder: {books_folder}")
        books_loader = DirectoryLoader(books_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        books_docs = books_loader.load()
        logger.info(f"Loaded {len(books_docs)} documents from books folder")
    
    # Process other document paths
    for doc_path in doc_paths:
        if os.path.isdir(doc_path):
            loader = DirectoryLoader(doc_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        else:
            # Automatically detect file type
            if doc_path.endswith('.pdf'):
                loader = PyPDFLoader(doc_path)
            elif doc_path.endswith(('.docx', '.doc', '.txt', '.html', '.htm')):
                loader = UnstructuredFileLoader(doc_path)
            else:
                # Unsupported file types are explicitly rejected
                raise ValueError(f"Unsupported file type: {doc_path}")
        
        docs = loader.load()
        other_docs.extend(docs)
    
    # Combine with books first
    all_docs = books_docs + other_docs
    logger.info(f"Total documents loaded: {len(all_docs)} (Books: {len(books_docs)}, Other: {len(other_docs)})")
    
    # Split documents with proper parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)
    
    # Process metadata
    for i, doc in enumerate(splits):
        if not doc.metadata.get('doc_id'):
            doc.metadata['doc_id'] = f"doc_{i}"
        doc.metadata['chunk_id'] = i
        doc.metadata['source_type'] = 'document'
        # Mark books for better retrieval
        if i < len(books_docs):
            doc.metadata['priority'] = 'high'
            doc.metadata['source_type'] = 'book'
    
    # Initialize Pinecone
    pc = get_pinecone_client()
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pinecone_region = os.getenv("PINECONE_REGION", "gcp-starter")
        logger.info(f"Creating Pinecone index '{index_name}' in region '{pinecone_region}'")
        
        try:
            if '-' in pinecone_region:
                cloud = pinecone_region.split('-')[0]
                region_val = '-'.join(pinecone_region.split('-')[1:])
                # If region is "starter", then omit the region key
                if region_val.lower() == "starter":
                    spec_config = {"serverless": {"cloud": cloud}}
                else:
                    spec_config = {"serverless": {"cloud": cloud, "region": region_val}}
                pc.create_index(
                    name=index_name,
                    dimension=1024,
                    metric="cosine",
                    spec=spec_config
                )
            else:
                pc.create_index(
                    name=index_name,
                    dimension=1024,
                    metric="cosine",
                    spec={"serverless": {"cloud": pinecone_region}}
                )
        except Exception as e:
            logger.warning(f"Error creating index with region-specific config: {e}")
            # Try alternative configurations
            try:
                # Try with simplified config (just cloud, no region)
                pc.create_index(
                    name=index_name,
                    dimension=1024,
                    metric="cosine",
                    spec={"serverless": {"cloud": "gcp"}}
                )
                logger.info("Successfully created index with simplified GCP config")
            except Exception as e2:
                # Final fallback - try AWS
                logger.warning(f"Error with GCP config: {e2}, trying AWS...")
                try:
                    pc.create_index(
                        name=index_name,
                        dimension=1024,
                        metric="cosine",
                        spec={"serverless": {"cloud": "aws"}}
                    )
                    logger.info("Successfully created index with AWS config")
                except Exception as e3:
                    logger.error(f"All index creation attempts failed: {e3}")
                    raise Exception(f"Unable to create Pinecone index: {e3}")
    
    # Get the index
    index = pc.Index(index_name)
    
    # Set up vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=namespace
    )
    
    # Add documents to vector store in batches to avoid payload exceeding size limits
    batch_size = 50  # adjust as needed
    for i in range(0, len(splits), batch_size):
        batch = splits[i: i+batch_size]
        vectorstore.add_documents(batch)
    
    return vectorstore

# Tool system architecture
class Tool:
    """Base class for all tools in the RAG system"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run the tool and return results"""
        raise NotImplementedError("Tool must implement run method")
    
    def to_dict(self) -> Dict[str, str]:
        """Return tool metadata as dict"""
        return {
            "name": self.name,
            "description": self.description
        }

class IKAPITool(Tool):
    """Tool for searching Indian Kanoon legal documents"""
    
    def __init__(self):
        super().__init__(
            name="indian_kanoon_search",
            description="Search for Indian legal documents, cases, and judgments. Provides authoritative legal information from Indian Kanoon."
        )
        token = os.getenv("INDIANKANOON_API_TOKEN")
        data_dir = os.getenv("INDIANKANOON_DATA_DIR", "/tmp/ik_data")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        self.storage = FileStorage(data_dir)
        
        # Configure minimal args with proper type annotations
        class Args:
            token: str = None
            maxcites: int = 5
            maxcitedby: int = 5
            orig: bool = False
            maxpages: int = 1
            pathbysrc: bool = True
            addedtoday: bool = False
            numworkers: int = 1
            fromdate: Optional[str] = None
            todate: Optional[str] = None
            sortby: Optional[str] = None
        
        args = Args()
        args.token = token
        args.maxcites = 5
        args.maxcitedby = 5
        args.orig = False
        args.maxpages = 1
        args.pathbysrc = True
        args.addedtoday = False
        args.numworkers = 1
        args.fromdate = None
        args.todate = None
        args.sortby = None
        
        self.ikapi = IKApi(args, self.storage)
    
    async def run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search case law in Indian Kanoon"""
        start_time = time.time()
        try:
            results = self.ikapi.search(query, 0, 1)  # First page only
            result_obj = json.loads(results)
            
            if 'docs' not in result_obj:
                return {"status": "success", "results": [], "source": "indian_kanoon", "time_taken": time.time() - start_time}
                
            docs = result_obj['docs'][:max_results]
            processed_docs = []
            
            # Process docs concurrently for speed
            with ThreadPoolExecutor(max_workers=min(len(docs), 3)) as executor:
                future_to_doc = {executor.submit(self._fetch_document, doc): doc for doc in docs}
                for future in as_completed(future_to_doc):
                    try:
                        processed_doc = future.result()
                        if processed_doc:
                            processed_docs.append(processed_doc)
                    except Exception as e:
                        logger.error(f"Error processing document: {e}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"IK API search completed in {elapsed_time:.2f}s with {len(processed_docs)} results")
            
            return {
                "status": "success", 
                "results": processed_docs, 
                "source": "indian_kanoon",
                "time_taken": elapsed_time
            }
        except Exception as e:
            logger.error(f"Error searching Indian Kanoon: {e}")
            return {"status": "error", "error": str(e), "results": [], "source": "indian_kanoon", "time_taken": time.time() - start_time}
    
    def _fetch_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Helper to fetch and process a single document"""
        try:
            doc_id = doc.get('tid')  # Use .get() instead of ['tid'] for safety
            if doc_id is None:
                return None
                
            doc_json = self.ikapi.fetch_doc(doc_id)  # type: ignore  # Ignore type error for ikapi
            doc_data = json.loads(doc_json)
            
            return {
                "title": doc_data.get('title', ''),
                "content": doc_data.get('doc', ''),
                "source": f"indiankanoon.org/doc/{doc_id}/",
                "domain": "indiankanoon.org",
                "docid": doc_id,
                "citation": doc_data.get('citation', ''),
                "type": "legal_document"
            }
        except Exception as e:
            logger.error(f"Error fetching document {doc.get('tid', 'unknown')}: {e}")
            return None

class WebSearchTool(Tool):
    """Tool for searching and retrieving web content"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information about legal topics, laws, and procedures. Returns content from various websites."
        )
        # Increase timeout for better reliability
        self.http_client = httpx.AsyncClient(timeout=15, verify=False)
        # Track already processed URLs to avoid duplicates
        self.processed_urls = set()
        # Domain priorities for ranking results
        self.domain_priorities = {
            # Government and legal sites (highest priority)
            "indiankanoon.org": "very_high",
            "supremecourt.gov.in": "very_high",
            "sci.gov.in": "very_high",
            "main.sci.gov.in": "very_high",
            "doj.gov.in": "very_high",
            "legalservices.gov.in": "very_high",
            "indiacode.nic.in": "very_high",
            "lawmin.gov.in": "very_high",
            "legislative.gov.in": "very_high",
            "highcourtofkerala.nic.in": "very_high",
            "bombayhighcourt.nic.in": "very_high", 
            "karnatakajudiciary.kar.nic.in": "very_high",
            "delhihighcourt.nic.in": "very_high",
            # Any .gov.in domain
            "gov.in": "high",
            "nic.in": "high",
            # Educational and organizational domains
            "nalsar.ac.in": "high",
            "nls.ac.in": "high",
            "nludelhi.ac.in": "high",
            "jgu.edu.in": "high",
            # Legal resource sites
            "scconline.com": "high",
            "livelaw.in": "high",
            "barandbench.com": "high",
            "manupatra.com": "high",
            "niti.gov.in": "high",
            # Educational sites
            "edu": "medium",
            "ac.in": "medium",
            # Other organizational sites
            "org": "medium",
            # Commercial sites
            "com": "low",
            # Video sites
            "youtube.com": "medium",
            "youtu.be": "medium"
        }
        # Set thresholds
        self.min_results_threshold = 3
        self.max_search_queries = 3  # Allow up to 3 refined search prompts
        self.max_urls_per_query = 3
        self.concurrent_requests = 3  # How many urls to process simultaneously
    
    async def run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web and process results with improved efficiency"""
        start_time = time.time()
        search_results = await self._search_web(query, max_results * 2)  # Get more results than needed for filtering
        
        if not search_results:
            return {"status": "error", "error": "No search results found", "results": [], "time_taken": time.time() - start_time}
        
        # Filter out already processed URLs
        filtered_results = [r for r in search_results if r["url"] not in self.processed_urls]
        
        # Sort by domain priority
        prioritized_results = self._prioritize_results(filtered_results)
        
        # Take top N results
        results_to_process = prioritized_results[:min(max_results, self.max_urls_per_query)]
        
        # Update processed URLs set
        for result in results_to_process:
            self.processed_urls.add(result["url"])
        
        # Process each URL concurrently
        processed_results = []
        if results_to_process:
            tasks = []
            for result in results_to_process:
                tasks.append(self._process_content(result["url"]))
            
            # Process in batches for better resource management
            for i in range(0, len(tasks), self.concurrent_requests):
                batch = tasks[i:i+self.concurrent_requests]
                try:
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.error(f"Error processing content: {result}")
                        elif result:
                            processed_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Web search completed in {elapsed_time:.2f}s with {len(processed_results)} results")
        
        return {
            "status": "success", 
            "results": processed_results, 
            "source": "web_search",
            "time_taken": elapsed_time
        }
    
    def _prioritize_results(self, results: List[Dict]) -> List[Dict]:
        """Sort results by domain priority"""
        # Calculate priority score for each result
        for result in results:
            url = result.get("url", "")
            domain = tldextract.extract(url).registered_domain
            
            # Default priority is lowest
            priority_score = 0
            
            # Check for exact domain matches first
            if domain in self.domain_priorities:
                priority = self.domain_priorities[domain]
                if priority == "very_high":
                    priority_score = 100
                elif priority == "high":
                    priority_score = 75 
                elif priority == "medium":
                    priority_score = 50
                elif priority == "low":
                    priority_score = 25
            else:
                # Check for partial domain matches (.gov.in etc)
                for key, value in self.domain_priorities.items():
                    if key in url:
                        if value == "very_high":
                            priority_score = 100
                            break
                        elif value == "high":
                            priority_score = 75
                            break
                        elif value == "medium":
                            priority_score = 50
                            break
                        elif value == "low":
                            priority_score = 25
                            break
            
            result["priority_score"] = priority_score
        
        # Sort by priority score (descending)
        return sorted(results, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    async def _search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Perform a web search using Serper API with Indian geolocation
        """
        # Add warnings filter to suppress SSL verification warnings
        import warnings
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        
        try:
            # Use Serper API as primary search method if available
            serper_api_key = os.getenv("SERPER_API_KEY")
            if (serper_api_key):
                logger.info(f"Searching with Serper API: {query}")
                url = "https://google.serper.dev/search"
                
                payload = json.dumps({
                    "q": query,
                    "gl": "in",  # Set geolocation to India
                    "num": max_results * 2  # Request more results to have backup options
                })
                
                headers = {
                    'X-API-KEY': serper_api_key,
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(url, headers=headers, data=payload, timeout=10)
                data = response.json()
                
                results = []
                
                # Process organic search results
                if "organic" in data:
                    for item in data["organic"][:max_results]:
                        # Skip sites that commonly cause SSL errors
                        if any(domain in item.get("link", "") for domain in ["rtionline.up.gov.in"]):
                            continue
                            
                        results.append({
                            "url": item.get("link", ""),
                            "title": item.get("title", ""),
                            "description": item.get("snippet", "")
                        })
                
                # Process knowledge graph if available
                if "knowledgeGraph" in data and len(results) < max_results:
                    kg = data["knowledgeGraph"]
                    if "descriptionLink" in kg:
                        results.append({
                            "url": kg.get("descriptionLink", ""),
                            "title": kg.get("title", ""),
                            "description": kg.get("description", "")
                        })
                
                return results[:max_results]
                
            # Fall back to DuckDuckGo if Serper API key not available
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = await self.http_client.get(url)
            data = response.json()
            
            results = []
            # Extract organic results
            for result in data.get("Results", [])[:max_results]:
                results.append({
                    "url": result.get("FirstURL", ""),
                    "title": result.get("Text", ""),
                    "description": ""
                })
            
            # Also include related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if "FirstURL" in topic:
                    results.append({
                        "url": topic.get("FirstURL", ""),
                        "title": topic.get("Text", ""),
                        "description": ""
                    })
            
            return results
                
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            # Fallback to a simple Google search page scraping as last resort
            try:
                search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                response = await self.http_client.get(search_url, headers=headers)
                
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                
                for g in soup.find_all('div', class_='g'):
                    anchors = g.find_all('a')
                    if anchors:
                        link = anchors[0]['href']
                        title = g.find('h3').text if g.find('h3') else "No title"
                        results.append({
                            "url": link,
                            "title": title,
                            "description": ""
                        })
                        
                        if len(results) >= max_results:
                            break
                
                return results
            except Exception as e2:
                logger.error(f"Error in fallback search: {e2}")
                return []
    
    async def _process_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Process different content types from the web"""
        try:
            # Determine content type
            if "youtube.com" in url or "youtu.be" in url:
                return await self._extract_youtube(url)
            elif url.endswith(".pdf"):
                return await self._extract_pdf(url)
            else:
                return await self._extract_webpage(url)
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None
    
    async def _extract_youtube(self, url: str) -> Dict[str, Any]:
        """Extract content from YouTube videos"""
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True, language=["en", "hi"]
            )
            docs = loader.load()
            
            if not docs:
                # Fallback to pytube
                yt = YouTube(url)
                transcript = yt.captions.get_by_language_code('en')
                
                if transcript:
                    content = transcript.generate_srt_captions()
                else:
                    content = yt.description
                
                return {
                    "content": content,
                    "source": url,
                    "domain": "youtube.com",
                    "title": yt.title,
                    "type": "video"
                }
            
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            metadata = docs[0].metadata if docs else {}
            
            return {
                "content": combined_text,
                "source": url,
                "domain": "youtube.com",
                "title": metadata.get("title", "YouTube Video"),
                "type": "video"
            }
        except Exception as e:
            logger.error(f"Error extracting YouTube content: {e}")
            return {
                "content": f"Failed to extract content from {url}",
                "source": url,
                "domain": "youtube.com",
                "title": "YouTube Video",
                "type": "video"
            }
    
    async def _extract_pdf(self, url: str) -> Dict[str, Any]:
        """Extract content from PDF documents with OCR"""
        try:
            response = await self.http_client.get(url)
            
            # Download the PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            text_content = ""
            try:
                # First try to extract text directly
                doc = fitz.open(tmp_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text_content += page.get_text()
                doc.close()
                
                # If no text extracted, use OCR
                if not text_content.strip():
                    doc = fitz.open(tmp_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img = Image.open(BytesIO(pix.tobytes()))
                        text_content += pytesseract.image_to_string(img)
                    doc.close()
            finally:
                os.unlink(tmp_path)
            
            domain = tldextract.extract(url).domain
            
            return {
                "content": text_content,
                "source": url,
                "domain": domain,
                "title": f"PDF Document from {domain}",
                "type": "pdf"
            }
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return {
                "content": f"Failed to extract content from {url}",
                "source": url,
                "domain": tldextract.extract(url).domain,
                "title": "PDF Document",
                "type": "pdf"
            }
    
    async def _extract_webpage(self, url: str) -> Dict[str, Any]:
        """Extract content from webpages"""
        try:
            # Skip specific problematic domains
            if any(domain in url for domain in ["rtionline.up.gov.in"]):
                return None
                
            # Use a custom loader with SSL verification disabled for government sites
            loader = AsyncHtmlLoader([url], verify_ssl=False)
            docs = await loader.aload()
            
            if not docs:
                return None
                
            transformer = Html2TextTransformer()
            docs_transformed = transformer.transform_documents(docs)
            
            if not docs_transformed:
                return None
                
            content = docs_transformed[0].page_content
            metadata = docs_transformed[0].metadata
            
            # Extract title and other metadata using BeautifulSoup
            soup = BeautifulSoup(docs[0].page_content, "html.parser")
            title = soup.title.text if soup.title else url
            
            domain = tldextract.extract(url).domain
            
            return {
                "content": content,
                "source": url,
                "domain": domain,
                "title": title,
                "type": "webpage"
            }
        except Exception as e:
            logger.error(f"Error extracting webpage content: {e}")
            return None
    
    async def search_and_process(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web and process the results in one step"""
        search_results = await self.search_web(query, max_results)
        if not search_results:
            return []
            
        # Process results concurrently
        processed_results = []
        tasks = []
        
        for result in search_results[:max_results]:
            if result.get("url"):
                tasks.append(self.process_content(result["url"]))
                
        # Process in batches for better resource management
        for i in range(0, len(tasks), self.concurrent_requests):
            batch = tasks[i:i+self.concurrent_requests]
            try:
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing content: {result}")
                    elif result:
                        processed_results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                
        return processed_results

class VectorDBLookupTool(Tool):
    """Tool for retrieving information from the vector database"""
    
    def __init__(self, vectorstore):
        super().__init__(
            name="vector_db_lookup",
            description="Search the legal knowledge base for relevant information. Contains high-quality legal documents, acts, and precedents."
        )
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
    
    async def run(self, query: str, k: int = 8) -> Dict[str, Any]:
        """Retrieve documents from vector database"""
        start_time = time.time()
        try:
            # Set a reasonable default
            k = min(max(1, k), 10)
            
            # Update retrieval parameters
            self.retriever.search_kwargs["k"] = k
            
            # Get documents
            docs = self.retriever.invoke(query)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "vector database"),
                    "title": doc.metadata.get("title", "Document"),
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "type": doc.metadata.get("source_type", "document")
                })
            
            elapsed_time = time.time() - start_time
            logger.info(f"Vector DB lookup completed in {elapsed_time:.2f}s with {len(results)} documents")
            
            return {
                "status": "success", 
                "results": results, 
                "source": "vector_db",
                "time_taken": elapsed_time
            }
        except Exception as e:
            logger.error(f"Error in vector DB lookup: {e}")
            return {"status": "error", "error": str(e), "results": [], "source": "vector_db", "time_taken": time.time() - start_time}

class PineconeIndexingTool(Tool):
    """Tool for indexing new content into Pinecone"""
    
    def __init__(self, embeddings):
        super().__init__(
            name="pinecone_indexer",
            description="Index new valuable content into the Pinecone vector database for future retrieval"
        )
        self.embeddings = embeddings
        self.pc = get_pinecone_client()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    async def run(self, content: List[Dict], index_name: str = "llama-text-embed-v2-index", namespace: str = "indian-law") -> Dict[str, Any]:
        """Index content into Pinecone"""
        start_time = time.time()
        try:
            # Convert to Document objects
            docs = []
            for i, item in enumerate(content):
                if not item or not item.get("content"):
                    continue
                    
                docs.append(
                    Document(
                        page_content=item["content"],
                        metadata={
                            "source": item.get("source", ""),
                            "title": item.get("title", ""),
                            "domain": item.get("domain", ""),
                            "type": item.get("type", "web"),
                            "doc_id": f"dynamic_{int(time.time())}_{i}"
                        }
                    )
                )
            
            if not docs:
                return {
                    "status": "error", 
                    "error": "No valid content to index", 
                    "indexed_count": 0,
                    "time_taken": time.time() - start_time
                }
            
            # Split documents
            splits = self.text_splitter.split_documents(docs)
            
            # Process metadata
            for i, doc in enumerate(splits):
                if not doc.metadata.get('doc_id'):
                    doc.metadata['doc_id'] = f"doc_{int(time.time())}_{i}"
                doc.metadata['chunk_id'] = i
                doc.metadata['source_type'] = doc.metadata.get('type', 'web')
                doc.metadata['indexed_date'] = time.strftime("%Y-%m-%d")
            
            # Get the index
            if index_name not in self.pc.list_indexes().names():
                return {
                    "status": "error", 
                    "error": f"Index {index_name} does not exist", 
                    "indexed_count": 0,
                    "time_taken": time.time() - start_time
                }
                
            index = self.pc.Index(index_name)
            
            # Set up vector store
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text",
                namespace=namespace
            )
            
            # Add documents to vector store in batches
            batch_size = 50
            total_indexed = 0
            
            for i in range(0, len(splits), batch_size):
                batch = splits[i: i+batch_size]
                vectorstore.add_documents(batch)
                total_indexed += len(batch)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Indexing completed in {elapsed_time:.2f}s - indexed {total_indexed} documents")
            
            return {
                "status": "success", 
                "indexed_count": total_indexed,
                "time_taken": elapsed_time
            }
        except Exception as e:
            logger.error(f"Error in indexing: {e}")
            return {"status": "error", "error": str(e), "indexed_count": 0, "time_taken": time.time() - start_time}

class ToolManager:
    """Manages tool registration, selection and execution"""
    
    def __init__(self, planning_llm):
        self.tools = {}
        self.planning_llm = planning_llm  # planning LLM passed from parent
        self.evaluation_llm = create_llm_from_env("evaluation")
        
        # Planning prompt template
        self.planning_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specializing in Indian law, tasked with planning how to answer legal queries about the Indian judicial system.
        
        Available tools:
        {tools}
        
        User query: {query}
        
        First, analyze the query to understand what information is needed about Indian law. Then create a plan for using the tools to get that information.
        
        When crafting your plan:
        1. Break down complex legal queries into 3-4 specific search queries that target different aspects of Indian law
        2. Use precise legal terminology relevant to Indian statutes, acts, and court systems
        3. Include queries that target specific acts, sections, or landmark judgments of the Supreme Court of India or High Courts when relevant
        4. Consider both the letter of the law (statutory provisions) and judicial interpretations (case law)
        
        For each tool, decide if it should be used, in what order, and with what parameters. The search strategy should:
        1. Always start with vector_db_lookup to check our existing knowledge base about Indian law
        2. Use indian_kanoon_search for queries that need authoritative legal precedents or specific case citations
        3. Only resort to web_search if the above tools don't yield sufficient information, with queries precisely tailored to Indian legal contexts
        
        Your output must be a valid JSON object with this structure:
        {
            "plan": "Detailed explanation of your search strategy for this Indian legal query",
            "search_queries": [
                "specific query 1 targeting relevant Indian legal information",
                "specific query 2 focusing on another aspect",
                "specific query 3 addressing case law or precedents",
                "specific query 4 (if needed)"
            ],
            "tools": [
                {
                    "tool": "tool_name",
                    "parameters": {"query": "precise legal query focusing on Indian context", "other_params": "values"},
                    "reason": "Why you're using this tool for this specific Indian legal information"
                },
                ...
            ]
        }
        
        Return only the JSON, nothing else.
        """)
        
        # Evaluation prompt template
        self.evaluation_prompt = ChatPromptTemplate.from_template("""
        You need to evaluate if the information below would be valuable to add to our legal knowledge base.
        
        Information:
        {content}
        
        Source: {source}
        Type: {type}
        
        Evaluate the information based on:
        1. Accuracy & reliability - Is it from an authoritative source?
        2. Relevance to Indian law and legal topics
        3. Completeness and depth
        4. Uniqueness - Does it contain information not commonly available?
        
        Output just a JSON object with:
        {
            "should_index": true/false,
            "reason": "Brief explanation of your decision",
            "quality_score": score from 1-10
        }
        
        Return only the JSON, nothing else.
        """)
    
    def register_tool(self, tool: Tool):
        """Register a tool with the manager"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def list_tools(self) -> List[Dict[str, str]]:
        """Get a list of all registered tools"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        """Generate a formatted description of available tools"""
        result = ""
        for name, tool in self.tools.items():
            result += f"- {name}: {tool.description}\n"
        return result
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool by name"""
        if tool_name not in self.tools:
            return {"status": "error", "error": f"Tool '{tool_name}' not found"}
            
        try:
            return await self.tools[tool_name].run(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def create_plan(self, query: str) -> Dict[str, Any]:
        """Create a plan for answering the query"""
        try:
            prompt_val = self.planning_prompt.format_messages(
                tools=self.get_tools_description(),
                query=query
            )
            
            response = await self.planning_llm.ainvoke(prompt_val)
            plan_text = response.content
            
            # Extract JSON from response (in case the model adds extra text)
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                plan_text = json_match.group(0)
            else:
                # Handle cases where the response might not be valid JSON at all
                logger.error(f"Could not extract JSON from planning response: {plan_text}")
                raise ValueError("Planning response did not contain valid JSON.")

            # More robust JSON parsing - strip leading/trailing whitespace
            plan_text = plan_text.strip()

            # Handle case where the JSON might have single quotes instead of double quotes
            try:
                plan = json.loads(plan_text)
            except json.JSONDecodeError:
                try:
                    # Attempt to fix single quotes
                    plan_text_fixed = plan_text.replace("'", '"')
                    plan = json.loads(plan_text_fixed)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse plan JSON even after fixing quotes: {plan_text}")
                    raise e # Re-raise the error after logging

            # Ensure the plan has the expected structure or create a valid structure
            if not isinstance(plan, dict):
                raise ValueError("Plan must be a dictionary")
                
            # Handle minimal schema validation
            if "plan" not in plan:
                plan["plan"] = "Generated plan"
                
            if "tools" not in plan:
                # Set a default tool configuration
                plan["tools"] = [
                    {
                        "tool": "vector_db_lookup",
                        "parameters": {"query": query, "k": 5},
                        "reason": "Default knowledge base search"
                    }
                ]
                
            # Add search queries array if not present but was requested in the prompt
            if "search_queries" not in plan:
                plan["search_queries"] = [query]
                
            return plan
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            # Return a clear default plan using vector DB and web search due to planning error
            return {
                "plan": "Default plan due to planning error: using vector DB lookup and web search",
                "search_queries": [query],
                "tools": [
                    {
                        "tool": "vector_db_lookup",
                        "parameters": {"query": query, "k": 5},
                        "reason": "Fallback: search existing knowledge base"
                    },
                    {
                        "tool": "web_search",
                        "parameters": {"query": query, "max_results": 3},
                        "reason": "Fallback: fetch information online"
                    }
                ]
            }
    
    async def evaluate_content(self, content: Dict) -> Dict[str, Any]:
        """Evaluate if content should be indexed"""
        try:
            prompt_val = self.evaluation_prompt.format_messages(
                content=content.get("content", ""),
                source=content.get("source", "unknown"),
                type=content.get("type", "unknown")
            )
            response = await self.evaluation_llm.ainvoke(prompt_val)
            eval_text = response.content if hasattr(response, 'content') else str(response)
            # Extract JSON substring safely
            json_match = re.search(r'\{.*\}', eval_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                logger.warning(f"Could not extract JSON from evaluation response: {eval_text}")
                json_text = "{}" # Default to empty JSON if extraction fails

            try:
                evaluation = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse evaluation JSON: {json_text} - Error: {e}")
                # Default to not indexing on parsing error
                evaluation = {
                    "should_index": False,
                    "reason": f"JSON parsing error: {str(e)}",
                    "quality_score": 0
                }
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating content: {e}")
            # Default to not indexing on error
            return {
                "should_index": False,
                "reason": f"Evaluation error: {str(e)}",
                "quality_score": 0
            }

class LegalRAGSystem:
    """Base RAG system for legal queries with specialized legal document assistance functions"""
    
    def __init__(self, vectorstore):
        # Store the vectorstore reference directly
        self.vector_store = vectorstore
        
        # Use vectorstore retriever
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        # Configure models - all using Groq now
        self.general_llm = create_llm_from_env("general")
        
        # Switch to Groq for specialist as well
        self.legal_specialist = create_llm_from_env("specialist")
        
        # Setup improved prompt with better context handling
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an Indian legal assistant specializing exclusively in Indian law and the Indian judicial system.
Analyze the following legal context and provide a detailed, accurate response with proper citations to Indian statutes, acts, and case law.

Always refer to relevant sections of Indian legislation, landmark judgments of the Supreme Court of India or High Courts, and follow Indian legal conventions and terminology.

If information is not available in the context, clearly state that rather than making up an answer or referring to non-Indian legal systems.

Use markdown formatting for better readability.
After every sentence, include an inline citation in square brackets (e.g., [1], [2]).
At the end of your answer, include a "References:" section enumerating full citations (title and URL) for each cited source.
Ensure citations are precise and formatted consistently.
Keep your response concise and focused on answering the user's question directly according to Indian law.
"""),
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])
        
        # Specialized prompts for different use cases
        self.keyword_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal terminology expert. Extract and define all key legal terms from the provided text, using the context for better definitions.
Format your response as a valid JSON object where each key is a legal term and each value is its concise definition.
Only include terms that have specific legal meaning and organize them by importance.
Return only the JSON object, nothing else.
"""),
            ("human", "Context:\n{context}\n\nText: {input}") # Added {context}
        ])

        self.composition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal writing assistant helping to compose structured legal arguments.
            Create a well-structured argument based on the topic and points provided. 
            
            The output should be formatted in markdown and ready to be directly copied into a document.
            Format headings with proper hierarchy, use bold for emphasis, and include proper citations format.
            Organize the argument logically with introduction, main points, and conclusion.
            """),
            ("human", "Topic: {topic}\n\nPoints to include: {points}\n\nContext: {context}")
        ])
        
        self.outline_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal document structure expert. Create a comprehensive outline for a legal document.
            
            The outline should follow standard legal document structure appropriate for the document type.
            Use a clear hierarchical structure with proper markdown formatting:
            # for main sections
            ## for subsections
            ### for sub-subsections
            
            Include placeholders in [brackets] where specific information would need to be added.
            The output should be ready to be directly copied into a document editor.
            """),
            ("human", "Document type: {doc_type}\n\nTopic: {topic}\n\nContext: {context}")
        ])
        
        self.citation_verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal citation verification specialist. Analyze the given citation and verify its accuracy.
            
            Check the following:
            1. Is the citation format correct?
            2. Does the case exist?
            3. Is the citation to the appropriate authority?
            4. Are there any errors in the citation?
            
            Provide the corrected citation if needed and a brief summary of the cited case/law.
            
            Format your response as JSON:
            {
                "original_citation": "the citation as provided",
                "is_valid": true/false,
                "corrected_citation": "properly formatted citation if correction needed",
                "summary": "brief summary of what is being cited",
                "error_details": "explanation of any errors found"
            }
            """),
            ("human", "Citation: {citation}\n\nContext: {context}")
        ])
        
        # Create chains for each specialized function
        self.qa_chain = create_stuff_documents_chain(
            self.legal_specialist, 
            self.qa_prompt
        )
        
        # Set up token counter with fallback mechanism
        try:
            # Try to use the cl100k_base tokenizer which is more reliably available
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Using cl100k_base tokenizer")
        except (ImportError, Exception) as e:
            logger.warning(f"Error loading tiktoken: {e}")
            # Define a simple fallback tokenizer that estimates tokens
            self.tokenizer = None
            logger.info("Using fallback token estimator")
        
        # Set maximum tokens for input context (leaving room for model output and prompt)
        self.max_input_tokens = 15000  # Lower limit for safety
    
    # Helper function to count tokens in text
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting error: {e}")
                # Fall back to simple estimation
                return len(text.split()) * 1.5  # Rough estimate
        else:
            # Simple estimation: 1 token ~= 4 characters in English
            return len(text) // 4
    
    async def extract_legal_keywords(self, text: str) -> Dict[str, Any]:
        """
        Extract key legal terms and their definitions from text.

        Args:
            text: The legal text to analyze

        Returns:
            Dictionary with extracted terms and their definitions
        """
        try:
            # First retrieve relevant context for better definitions
            raw_docs = self.retriever.invoke(text[:500])  # Use first part of text for query
            logger.info(f"extract_legal_keywords: Retriever invoked. Raw result type: {type(raw_docs)}")

            # Robustly ensure raw_docs is a list and wrap items
            if not isinstance(raw_docs, list):
                raw_docs = [raw_docs]
            docs = []
            for i, item in enumerate(raw_docs):
                if isinstance(item, Document):
                    docs.append(item)
                elif isinstance(item, dict) and 'page_content' in item:
                    docs.append(Document(page_content=item['page_content'], metadata=item.get('metadata', {})))
                elif isinstance(item, str):
                    logger.warning(f"extract_legal_keywords: Retriever returned a string at index {i} - wrapping.")
                    docs.append(Document(page_content=item, metadata={'source': 'retriever_string_result'}))
                else:
                    logger.warning(f"extract_legal_keywords: Retriever returned unexpected type {type(item)} at index {i} - converting to string.")
                    docs.append(Document(page_content=str(item), metadata={'source': 'retriever_unknown_result'}))

            context_docs = self._select_docs_within_budget(docs, 3000)  # Keep context reasonable
            logger.info(f"extract_legal_keywords: Selected {len(context_docs)} documents for context.")

            # Create chain for keyword extraction
            keyword_chain = create_stuff_documents_chain(
                self.legal_specialist,
                self.keyword_extraction_prompt,
                document_variable_name="context" # Explicitly set document variable name
            )

            # Get response - pass the list of Documents
            response = await keyword_chain.ainvoke({
                "input": text,
                "context": context_docs # Pass the list of Documents
            })

            # Extract JSON from response (response should be a string from the chain)
            content = response if isinstance(response, str) else str(response)
            logger.info(f"extract_legal_keywords: Raw response content: {content[:200]}...") # Log raw response

            # Try to parse as JSON (clean up if needed)
            try:
                # Find JSON block within potential markdown code fences
                json_match = re.search(r'```(?:json)?\n({.*?})\n```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Fallback: find first '{' and last '}'
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1 and start < end:
                        json_str = content[start:end+1]
                    else:
                        json_str = content # Assume the whole string might be JSON

                results = json.loads(json_str)
                return {
                    "status": "success",
                    "terms": results,
                    "count": len(results)
                }
            except json.JSONDecodeError as json_err:
                logger.error(f"extract_legal_keywords: Failed to parse JSON response: {json_err}. Raw content: {content}")
                return {
                    "status": "error",
                    "error": f"Could not parse response as JSON: {json_err}",
                    "raw_response": content,
                    "terms": {}
                }

        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "terms": {}
            }

    async def generate_legal_argument(self, topic: str, points: List[str]) -> Dict[str, Any]:
        """
        Generate a structured legal argument ready for insertion into documents.
        
        Args:
            topic: The main topic of the legal argument
            points: Key points to include in the argument
            
        Returns:
            Dictionary with formatted argument text
        """
        try:
            # Get relevant context for the topic and wrap into Document objects
            raw_docs = self.retriever.invoke(topic)
            logger.info(f"Retriever invoked for topic '{topic}'. Raw result type: {type(raw_docs)}") # Log type
            
            # Robustly ensure raw_docs is a list and flatten if necessary
            if not isinstance(raw_docs, list):
                raw_docs = [raw_docs]
            
            docs = []
            for i, item in enumerate(raw_docs):
                if isinstance(item, Document):
                    docs.append(item)
                elif isinstance(item, dict) and 'page_content' in item:
                    docs.append(Document(page_content=item['page_content'], metadata=item.get('metadata', {})))
                elif isinstance(item, str):
                    logger.warning(f"generate_legal_argument: Retriever returned a string at index {i}: '{item[:100]}...' - wrapping.")
                    docs.append(Document(page_content=item, metadata={'source': 'retriever_string_result'}))
                else:
                    # Attempt to convert unexpected types to string
                    logger.warning(f"generate_legal_argument: Retriever returned unexpected type {type(item)} at index {i}. Converting to string.")
                    docs.append(Document(page_content=str(item), metadata={'source': 'retriever_unknown_result'}))

            logger.info(f"Processed retriever results into {len(docs)} Document objects.")

            # Pass the guaranteed list of Document objects to the budget selector
            context_docs = self._select_docs_within_budget(docs, 4000)

            # Create chain for argument generation
            composition_chain = create_stuff_documents_chain(
                self.legal_specialist,
                self.composition_prompt,
                document_variable_name="context" # Explicitly set document variable name
            )

            # Format points as string
            points_str = "\n".join([f"- {point}" for point in points])

            # === START DEBUG LOGGING ===
            logger.info(f"[DEBUG] Type of context_docs before ainvoike: {type(context_docs)}")
            if isinstance(context_docs, list):
                logger.info(f"[DEBUG] Number of items in context_docs: {len(context_docs)}")
                valid_docs = True
                for i, item in enumerate(context_docs):
                    item_type = type(item)
                    logger.info(f"[DEBUG] Item {i} type in context_docs: {item_type}")
                    if not isinstance(item, Document):
                        logger.error(f"[CRITICAL] Item {i} in context_docs is NOT a Document: {item}")
                        valid_docs = False
                if not valid_docs:
                     logger.error("[CRITICAL] context_docs contains non-Document items!")
            else:
                logger.error("[CRITICAL] context_docs is not a list!")
            # === END DEBUG LOGGING ===

            # Get response - Pass the list of Document objects directly
            response = await composition_chain.ainvoke({
                "topic": topic,
                "points": points_str,
                "context": context_docs # Pass the list of Documents
            })

            # Clean up the response
            content = response if isinstance(response, str) else str(response)

            return {
                "status": "success",
                "argument": content,
                "word_count": len(content.split()),
                "character_count": len(content)
            }
            
        except Exception as e:
            logger.error(f"Error generating legal argument: {e}", exc_info=True) # Add exc_info for full traceback
            return {
                "status": "error",
                "error": str(e),
                "argument": ""
            }
    
    async def create_document_outline(self, topic: str, doc_type: str) -> Dict[str, Any]:
        """
        Create a pre-writing document outline for legal documents.

        Args:
            topic: The subject matter of the document
            doc_type: Type of legal document (e.g., 'brief', 'memo', 'contract', 'petition')

        Returns:
            Dictionary with formatted document outline
        """
        try:
            # Get relevant context
            raw_docs = self.retriever.invoke(f"{doc_type} {topic}")
            logger.info(f"create_document_outline: Retriever invoked. Raw result type: {type(raw_docs)}")

            # Robustly ensure raw_docs is a list and wrap items
            if not isinstance(raw_docs, list):
                raw_docs = [raw_docs]
            docs = []
            for i, item in enumerate(raw_docs):
                if isinstance(item, Document):
                    docs.append(item)
                elif isinstance(item, dict) and 'page_content' in item:
                    docs.append(Document(page_content=item['page_content'], metadata=item.get('metadata', {})))
                elif isinstance(item, str):
                    logger.warning(f"create_document_outline: Retriever returned a string at index {i} - wrapping.")
                    docs.append(Document(page_content=item, metadata={'source': 'retriever_string_result'}))
                else:
                    logger.warning(f"create_document_outline: Retriever returned unexpected type {type(item)} at index {i} - converting to string.")
                    docs.append(Document(page_content=str(item), metadata={'source': 'retriever_unknown_result'}))

            context_docs = self._select_docs_within_budget(docs, 3000)
            logger.info(f"create_document_outline: Selected {len(context_docs)} documents for context.")

            # Create chain for outline generation
            outline_chain = create_stuff_documents_chain(
                self.legal_specialist,
                self.outline_prompt,
                document_variable_name="context" # Explicitly set document variable name
            )

            # Get response - pass the list of Documents
            response = await outline_chain.ainvoke({
                "topic": topic,
                "doc_type": doc_type,
                "context": context_docs # Pass the list of Documents
            })

            # Response should be the outline string
            content = response if isinstance(response, str) else str(response)

            # Structure analysis for reporting
            section_count = content.count('\n# ') + (1 if content.startswith('# ') else 0)
            subsection_count = content.count('\n## ') + (1 if content.startswith('## ') else 0)

            return {
                "status": "success",
                "outline": content,
                "section_count": section_count,
                "subsection_count": subsection_count
            }

        except Exception as e:
            logger.error(f"Error creating document outline: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "outline": ""
            }

    async def verify_citation(self, citation: str) -> Dict[str, Any]:
        """
        Verify legal citations for accuracy and provide summary information.
        
        Args:
            citation: The legal citation to verify
            
        Returns:
            Dictionary with verification results and correction if needed
        """
        try:
            # First check IK API for the citation
            ik_tool = IKAPITool()
            
            # Clean citation to use as search query
            clean_citation = citation.replace(',', ' ').replace('vs.', 'vs').replace('v.', 'v')
            
            # Search for the citation
            ik_results = await ik_tool.run(clean_citation, max_results=2)
            
            # Get context from search results
            context = ""
            if ik_results.get("status") == "success" and ik_results.get("results"):
                for result in ik_results.get("results", []):
                    context += f"\nTitle: {result.get('title', '')}\n"
                    context += f"Content excerpt: {result.get('content', '')[:500]}...\n"
                    context += f"Source: {result.get('source', '')}\n\n"
            
            # Also search vector DB
            raw_docs = self.retriever.invoke(clean_citation)
            
            # Add vector DB results to context with better error handling
            context += "\n\nVector DB Results:\n"
            try:
                # First ensure raw_docs is a list
                if not isinstance(raw_docs, list):
                    raw_docs = [raw_docs]
                
                # Process each document safely
                for i, doc in enumerate(raw_docs[:2]):  # Just use top 2 results
                    if isinstance(doc, Document):
                        # Already a Document object
                        context += f"\n{doc.page_content[:500]}...\n"
                    elif hasattr(doc, 'page_content'):
                        # Has page_content attribute but may not be Document instance
                        context += f"\n{doc.page_content[:500]}...\n"
                    elif isinstance(doc, str):
                        # String content, add directly
                        context += f"\n{doc[:500]}...\n"
                    elif isinstance(doc, dict) and "page_content" in doc:
                        # Dictionary with page_content
                        context += f"\n{doc['page_content'][:500]}...\n"
                    else:
                        # Unknown format, convert to string
                        context += f"\n{str(doc)[:500]}...\n"
            except Exception as e:
                logger.error(f"Error processing vector DB results: {e}")
                context += "\nError retrieving additional context from vector DB.\n"
            
            # Create chain for citation verification
            verification_chain = create_stuff_documents_chain(
                self.legal_specialist,
                self.citation_verification_prompt
            )
            
            # Get response
            response = await verification_chain.ainvoke({
                "citation": citation,
                "context": context
            })
            
            # Extract JSON from response
            content = response.get("content", "{}")
            
            # Try to parse as JSON
            try:
                import re
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                
                results = json.loads(content)
                results["status"] = "success"
                return results
            except json.JSONDecodeError:
                # If not valid JSON, return the raw text
                logger.warning("Failed to parse citation verification response as JSON")
                return {
                    "status": "error",
                    "error": "Could not parse response",
                    "raw_response": content,
                    "is_valid": False
                }
                
        except Exception as e:
            logger.error(f"Error verifying citation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "is_valid": False
            }

    def _select_docs_within_budget(self, documents: List[Document], token_budget: int) -> List[Document]:
        """Select documents to include within token budget"""
        docs_for_context = []
        current_tokens = 0
        
        # Ensure all items are Document instances
        documents = [d if isinstance(d, Document) else Document(page_content=str(d), metadata={}) for d in documents]
        # First sort documents by relevance/priority if metadata score exists
        documents.sort(
            key=lambda x: x.metadata.get("relevance_score", 0) + x.metadata.get("domain_score", 0), 
            reverse=True
        )
        
        for doc in documents:
            doc_tokens = self._count_tokens(doc.page_content)
            
            # Skip extremely large documents
            if doc_tokens > token_budget * 0.8:
                # Truncate very large documents
                try:
                    if self.tokenizer:
                        max_tokens = int(token_budget * 0.5)  # Use at most 50% of budget for a single doc
                        truncated_content = self.tokenizer.decode(self.tokenizer.encode(doc.page_content)[:max_tokens])
                        doc.page_content = truncated_content + "... [content truncated due to length]"
                    else:
                        # Simple truncation for fallback case
                        max_chars = int(token_budget * 0.5 * 4)  # Rough char estimate
                        doc.page_content = doc.page_content[:max_chars] + "... [content truncated due to length]"
                except Exception as e:
                    logger.warning(f"Error truncating document: {e}")
                    # Simple truncation fallback
                    doc.page_content = doc.page_content[:5000] + "... [content truncated]"
                
                doc_tokens = self._count_tokens(doc.page_content)
            
            # Check if adding this document would exceed our budget
            if current_tokens + doc_tokens > token_budget:
                # If we already have some docs, stop here
                if docs_for_context:
                    break
                # If this is the first doc, truncate it to fit
                else:
                    try:
                        if self.tokenizer:
                            max_tokens = token_budget - 100  # Leave a small buffer
                            truncated_content = self.tokenizer.decode(self.tokenizer.encode(doc.page_content)[:max_tokens])
                            doc.page_content = truncated_content + "... [content truncated due to length]"
                        else:
                            # Simple truncation for fallback case
                            max_chars = int((token_budget - 100) * 4)  # Rough char estimate
                            doc.page_content = doc.page_content[:max_chars] + "... [content truncated]"
                    except Exception as e:
                        logger.warning(f"Error truncating document: {e}")
                        # Simple truncation fallback
                        doc.page_content = doc.page_content[:3000] + "... [content truncated]"
                    
                    doc_tokens = self._count_tokens(doc.page_content)
            
            docs_for_context.append(doc)
            current_tokens += doc_tokens
            
            # If we've exceeded 70% of our budget, stop adding more
            if current_tokens > token_budget * 0.7:
                break
        
        return docs_for_context

class EnhancedLegalRAGSystem(LegalRAGSystem):
    """Advanced RAG system with tool-based architecture"""
    
    def __init__(self, vectorstore, enable_web=True):
        super().__init__(vectorstore)
        
        # Initialize tool-based system
        self.planning_llm = create_llm_from_env("planning")
        
        self.tool_manager = ToolManager(self.planning_llm)
        
        # Register tools
        self.tool_manager.register_tool(VectorDBLookupTool(vectorstore))
        self.tool_manager.register_tool(IKAPITool())
        self.tool_manager.register_tool(WebSearchTool())
        self.tool_manager.register_tool(PineconeIndexingTool(embeddings))
        
        # Cache for results
        self.cache = {}
        
        # Add classifier prompt for determining tool necessity
        self.tool_necessity_prompt = ChatPromptTemplate.from_template("""
        Determine if the user's query requires specialized tools to answer or is a simple greeting/chitchat.
        
        User query: {query}
        
        First, analyze if this query:
        1. Is a greeting (like "hi", "hello", "good morning")
        2. Is simple chitchat (like "how are you", "what's up")
        3. Requires no factual information to answer
        4. Can be handled with general knowledge without research
        
        If ANY of the above are true, respond with "NO_TOOLS_NEEDED".
        If the query requires legal information, research, or specific knowledge, respond with "TOOLS_REQUIRED".
        
        Respond with ONLY one of these two options and no other text.
        """)
        # Prompt for refining web search queries
        self.search_refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that takes a legal question about Indian law and generates up to 3 concise web search queries to retrieve relevant information. Output only a JSON array of queries, max 3."),
            ("human", "User query: {query}")
        ])
    
    async def _needs_tools(self, query: str) -> bool:
        """Determine if a query needs tools or can be answered directly"""
        try:
            # Skip classifier for obviously complex queries
            if len(query.split()) > 10:
                return True
                
            prompt_val = self.tool_necessity_prompt.format_messages(query=query)
            response = await self.planning_llm.ainvoke(prompt_val)
            result = response.content.strip().upper()
            
            logger.info(f"Query tool necessity classifier result: {result}")
            return "TOOLS_REQUIRED" in result
        except Exception as e:
            logger.warning(f"Error in tools necessity check: {e}, defaulting to simple response")
            # If very short query and classifier failed, likely simple
            return len(query.split()) > 5
    
    async def generate_simple_response(self, query: str) -> Dict[str, str]:
        """Generate a response for simple queries without using tools"""
        try:
            # Simple template for non-research queries
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly legal assistant. For simple greetings and chitchat, 
                respond politely but keep it brief. Don't pretend to use any tools or research for simple interactions.
                If the user is asking a more complex question that would benefit from research, indicate that you 
                could help them with that topic if they'd like to explore it further."""),
                ("human", "{query}")
            ])
            
            prompt_val = simple_prompt.format_messages(query=query)
            response = await self.legal_specialist.ainvoke(prompt_val)
            
            # Ensure we always return a dict with 'content' key
            if response and hasattr(response, 'content'):
                return {"content": response.content}
            else:
                logger.warning("Received empty or invalid response from language model")
                return {"content": "Hello! How can I assist you with legal information today?"}
        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            return {"content": "Hello! How can I assist you with legal information today?"}
    
    async def query(self, question: str, use_web: bool = True):
        try:
            # 1. First check if the query even needs tools
            if not await self._needs_tools(question):
                logger.info(f"Query classified as simple, skipping tools: '{question}'")
                return await self.generate_simple_response(question)
                
            # 2. Planning phase - determine which tools to use
            plan = await self.tool_manager.create_plan(question)
            logger.info(f"Created plan: {plan['plan']}")
            
            # 3. Tool execution phase
            all_results = []
            valuable_content = []
            
            for tool_step in plan.get("tools", []):
                tool_name = tool_step.get("tool")
                parameters = tool_step.get("parameters", {})
                
                if tool_name == "web_search":
                    # refine the web search queries using general_llm
                    refinement = await self.general_llm.ainvoke(
                        self.search_refinement_prompt.format_messages(query=question)
                    )
                    try:
                        refined = json.loads(refinement.content)
                    except Exception:
                        refined = [question]
                    # execute web_search for each refined query up to 3, max 3 results each
                    for sub_q in refined[:3]:
                        logger.info(f"Executing web_search with refined query: {sub_q}")
                        sub_result = await self.tool_manager.execute_tool(
                            "web_search", query=sub_q, max_results=3
                        )
                        tool_result = sub_result
                        # process sub_result below
                        if tool_result.get("status") == "success":
                            results = tool_result.get("results", [])
                            if results:
                                all_results.extend(results)
                                for result in results:
                                    if not result.get("content") or len(result.get("content", "")) < 200:
                                        continue
                                    evaluation = await self.tool_manager.evaluate_content(result)
                                    if evaluation.get("should_index", False) and evaluation.get("quality_score", 0) >= 7:
                                        valuable_content.append(result)
                                        logger.info(f"Marked content from {result.get('source', 'unknown')} for indexing (score: {evaluation.get('quality_score', 0)})")
                    continue
                # default tool execution
                logger.info(f"Executing tool: {tool_name}")
                tool_result = await self.tool_manager.execute_tool(tool_name, **parameters)
                 
                if tool_result.get("status") == "success":
                    results = tool_result.get("results", [])
                     
                    if results:
                        all_results.extend(results)
                        logger.info(f"Got {len(results)} results from {tool_name}")
                        
                        # Evaluate if results should be indexed (if not from vector DB)
                        if tool_name != "vector_db_lookup" and tool_result.get("source") != "vector_db":
                            for result in results:
                                # Skip empty or very short content
                                if not result.get("content") or len(result.get("content", "")) < 200:
                                    continue
                                    
                                evaluation = await self.tool_manager.evaluate_content(result)
                                if evaluation.get("should_index", False) and evaluation.get("quality_score", 0) >= 7:
                                    valuable_content.append(result)
                                    logger.info(f"Marked content from {result.get('source', 'unknown')} for indexing (score: {evaluation.get("quality_score", 0)})")
            
            # 4. Index valuable content if any
            if valuable_content:
                logger.info(f"Indexing {len(valuable_content)} valuable pieces of content")
                indexing_result = await self.tool_manager.execute_tool("pinecone_indexer", content=valuable_content)
                logger.info(f"Indexing result: {indexing_result}")
            
            # 5. Convert to Document objects for final response generation
            documents = []
            for result in all_results:
                if not result.get("content"):
                    continue
                    
                documents.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "source": result.get("source", ""),
                            "title": result.get("title", ""),
                            "domain": result.get("domain", ""),
                            "type": result.get("type", "unknown")
                        }
                    )
                )
            
            # 6. Manage token count
            token_budget = self.max_input_tokens
            question_tokens = self._count_tokens(question)
            token_budget -= question_tokens
            
            # Calculate tokens for prompt template (fixed approximation)
            system_prompt = "You are a legal assistant specializing in Indian law. Analyze the following legal context and provide a detailed, accurate response with proper citations."
            human_prompt_template = "Context:\n{context}\n\nQuestion: {input}"
            prompt_template_tokens = self._count_tokens(system_prompt) + self._count_tokens(human_prompt_template)
            token_budget -= prompt_template_tokens
            
            # Reserve tokens for the model's response
            token_budget -= 2000  # Response tokens
            
            # Select and format documents to include within token budget
            docs_for_context = self._select_docs_within_budget(documents, token_budget)
            # Build numbered references and content blocks for precise citations
            ref_list = []
            content_blocks = []
            for idx, doc in enumerate(docs_for_context, start=1):
                title = doc.metadata.get("title", "")
                src = doc.metadata.get("source", "")
                ref_list.append(f"[{idx}] {title} - {src}")
                # Include content under same index
                content_blocks.append(f"[{idx}] {doc.page_content}")
            # Combine into single context string
            context_str = "References:\n" + "\n".join(ref_list) + "\n\n" + "\n\n".join(content_blocks)
            # 7. Generate response with numbered citations
            logger.info(f"Generating response with {len(docs_for_context)} documents and formatted context")
            resp = await self.qa_chain.ainvoke({
                "input": question,
                "context": context_str
            })
            # Ensure response is a dict
            if isinstance(resp, dict):
                return resp
            # Wrap raw or str response
            content = resp.content if hasattr(resp, 'content') else str(resp)
            return {"content": content}
        except Exception as e:
            logger.error(f"Error in tool-based query: {e}")
            # Fallback to simpler approach
            try:
                # Get some documents from vector DB
                docs = self.retriever.invoke(question)
                return await self.qa_chain.ainvoke({
                    "input": question,
                    "context": docs[:3]  # Use only top 3 docs
                })
            except Exception as e2:
                logger.error(f"Error in fallback query: {e2}")
                return {"content": f"I'm having trouble answering this question due to a technical error. Please try asking in a different way or contact support. Error: {str(e2)}"}
    
    async def query_non_streaming(self, question: str, use_web: bool = True):
        """
        Generate a complete response without streaming for a legal query.
        This is similar to the query method but returns complete results at once.
        
        Args:
            question: The legal question to answer
            use_web: Whether to use web search if needed
            
        Returns:
            Dict containing the complete response
        """
        try:
            # Track execution steps for debugging/transparency
            steps_log = []
            
            # 1. First check if the query even needs tools
            if not await self._needs_tools(question):
                logger.info(f"Query classified as simple, skipping tools: '{question}'")
                steps_log.append({"type": "classification", "content": "Query classified as simple"})
                return await self.generate_simple_response(question)
                
            # 2. Planning phase - determine which tools to use
            plan = await self.tool_manager.create_plan(question)
            logger.info(f"Created plan: {plan['plan']}")
            steps_log.append({"type": "planning", "content": plan['plan']})
            
            # 3. Tool execution phase
            all_results = []
            valuable_content = []
            tool_results = {}  # Store results by tool type
            
            for tool_step in plan.get("tools", []):
                tool_name = tool_step.get("tool")
                parameters = tool_step.get("parameters", {})
                
                if tool_name == "web_search":
                    # refine web search queries
                    refinement = await self.general_llm.ainvoke(
                        self.search_refinement_prompt.format_messages(query=question)
                    )
                    try:
                        refined = json.loads(refinement.content)
                    except Exception:
                        refined = [question]
                    
                    web_results = {"status": "success", "results": []}  # Initialize web results
                    steps_log.append({"type": "web_search_refinement", "content": refined})
                    
                    for sub_q in refined[:3]:
                        logger.info(f"Executing web_search with refined query: {sub_q}")
                        sub_result = await self.tool_manager.execute_tool(
                            "web_search", query=sub_q, max_results=3
                        )
                        
                        if sub_result.get("status") == "success":
                            results = sub_result.get("results", [])
                            if results:
                                all_results.extend(results)
                                web_results["results"].extend(results)  # Add to web results
                                for result in results:
                                    if not result.get("content") or len(result.get("content", "")) < 200:
                                        continue
                                    evaluation = await self.tool_manager.evaluate_content(result)
                                    if evaluation.get("should_index", False) and evaluation.get("quality_score", 0) >= 7:
                                        valuable_content.append(result)
                                        logger.info(f"Marked content from {result.get('source', 'unknown')} for indexing (score: {evaluation.get('quality_score', 0)})")
                    
                    # Store the combined web results
                    tool_results["web_search"] = web_results
                    steps_log.append({"type": "web_search", "content": f"Found {len(web_results['results'])} web results"})
                    continue
                    
                # default tool execution
                logger.info(f"Executing tool: {tool_name}")
                tool_result = await self.tool_manager.execute_tool(tool_name, **parameters)
                 
                if tool_result.get("status") == "success":
                    results = tool_result.get("results", [])
                    
                    # Store results by tool name
                    tool_results[tool_name] = tool_result
                    steps_log.append({"type": tool_name, "content": f"Found {len(results)} results"})
                     
                    if results:
                        all_results.extend(results)
                        logger.info(f"Got {len(results)} results from {tool_name}")
                        
                        # Evaluate if results should be indexed (if not from vector DB)
                        if tool_name != "vector_db_lookup" and tool_result.get("source") != "vector_db":
                            for result in results:
                                # Skip empty or very short content
                                if not result.get("content") or len(result.get("content", "")) < 200:
                                    continue
                                    
                                evaluation = await self.tool_manager.evaluate_content(result)
                                if evaluation.get("should_index", False) and evaluation.get("quality_score", 0) >= 7:
                                    valuable_content.append(result)
                                    logger.info(f"Marked content from {result.get('source', 'unknown')} for indexing (score: {evaluation.get('quality_score', 0)})")
            
            # 4. Index valuable content if any
            if valuable_content:
                logger.info(f"Indexing {len(valuable_content)} valuable pieces of content")
                indexing_result = await self.tool_manager.execute_tool("pinecone_indexer", content=valuable_content)
                logger.info(f"Indexing result: {indexing_result}")
                steps_log.append({"type": "indexing", "content": f"Indexed {len(valuable_content)} valuable pieces of content"})
            
            # 5. Convert to Document objects for final response generation
            documents = []
            for result in all_results:
                if not result.get("content"):
                    continue
                    
                documents.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "source": result.get("source", ""),
                            "title": result.get("title", ""),
                            "domain": result.get("domain", ""),
                            "type": result.get("type", "unknown")
                        }
                    )
                )

            # Select relevant documents within token budget
            docs_for_context = self._select_docs_within_budget(documents, self.max_input_tokens)
            logger.info(f"Generating response with {len(docs_for_context)} documents and formatted context")
            steps_log.append({"type": "context_selection", "content": f"Selected {len(docs_for_context)} documents for context"})

            # Format context string for human-readable display and tracking
            sources_list = []
            source_map = {}
            source_counter = 1
            # Extract sources for the response metadata but don't modify the docs themselves
            for i, doc in enumerate(docs_for_context):
                source_url = doc.metadata.get("source", f"source_{i+1}")
                if source_url not in source_map:
                    source_map[source_url] = source_counter
                    sources_list.append({
                        "id": source_counter, 
                        "url": source_url, 
                        "title": doc.metadata.get("title", source_url)
                    })
                    source_counter += 1
                # Add source reference to metadata for citation purposes
                doc.metadata["source_id"] = source_map[source_url]
            
            # Generate final response using the QA chain - pass the Document objects directly
            try:
                response = await self.qa_chain.ainvoke({
                    "input": question,
                    "context": docs_for_context  # Pass Document objects list, not a string
                })
                # Extract response content
                answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as chain_error:
                logger.error(f"Error in QA chain: {chain_error}", exc_info=True)
                # Fallback to simpler chain if complex one fails
                try:
                    # Create a simpler format for the context as fallback
                    simple_context = "\n\n".join([f"[{doc.metadata.get('source_id', 'unknown')}] {doc.page_content}" for doc in docs_for_context])
                    simple_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an Indian legal assistant. Provide a helpful response based on the context."),
                        ("human", "Context:\n{context}\n\nQuestion: {input}")
                    ])
                    simple_chain = simple_prompt | self.legal_specialist
                    simple_response = await simple_chain.ainvoke({
                        "input": question,
                        "context": simple_context
                    })
                    answer = simple_response.content
                    logger.info("Used fallback simple chain for response")
                    steps_log.append({"type": "recovery", "content": "Used fallback chain due to error"})
                except Exception as fallback_error:
                    logger.error(f"Error in fallback chain: {fallback_error}")
                    answer = f"I apologize, but I encountered technical difficulties processing your request about BNS Section 77. Please try again or rephrase your question."

            # Structure the final response
            final_response = {
                "answer": answer,
                "sources": sources_list,
                "steps": steps_log
            }
            return final_response

        except Exception as e:
            logger.error(f"Error in non-streaming query: {e}", exc_info=True)
            # Fallback response in case of error
            return {
                "answer": f"Sorry, I encountered an error processing your request: {str(e)}. Please try again later.",
                "sources": [],
                "steps": [{"type": "error", "content": str(e)}]
            }
    
    def _select_docs_within_budget(self, documents: List[Document], token_budget: int) -> List[Document]:
        """Select documents to include within token budget"""
        docs_for_context = []
        current_tokens = 0
        
        # Ensure all items are Document instances
        documents = [d if isinstance(d, Document) else Document(page_content=str(d), metadata={}) for d in documents]
        # First sort documents by relevance/priority if metadata score exists
        documents.sort(
            key=lambda x: x.metadata.get("relevance_score", 0) + x.metadata.get("domain_score", 0), 
            reverse=True
        )
        
        for doc in documents:
            doc_tokens = self._count_tokens(doc.page_content)
            
            # Skip extremely large documents
            if doc_tokens > token_budget * 0.8:
                # Truncate very large documents
                try:
                    if self.tokenizer:
                        max_tokens = int(token_budget * 0.5)  # Use at most 50% of budget for a single doc
                        truncated_content = self.tokenizer.decode(self.tokenizer.encode(doc.page_content)[:max_tokens])
                        doc.page_content = truncated_content + "... [content truncated due to length]"
                    else:
                        # Simple truncation for fallback case
                        max_chars = int(token_budget * 0.5 * 4)  # Rough char estimate
                        doc.page_content = doc.page_content[:max_chars] + "... [content truncated due to length]"
                except Exception as e:
                    logger.warning(f"Error truncating document: {e}")
                    # Simple truncation fallback
                    doc.page_content = doc.page_content[:5000] + "... [content truncated]"
                
                doc_tokens = self._count_tokens(doc.page_content)
            
            # Check if adding this document would exceed our budget
            if current_tokens + doc_tokens > token_budget:
                # If we already have some docs, stop here
                if docs_for_context:
                    break
                # If this is the first doc, truncate it to fit
                else:
                    try:
                        if self.tokenizer:
                            max_tokens = token_budget - 100  # Leave a small buffer                            truncated_content = self.tokenizer.decode(self.tokenizer.encode(doc.page_content)[:max_tokens])                            doc.page_content = truncated_content + "... [content truncated due to length]"                        else:
                            # Simple truncation for fallback case
                            max_chars = int((token_budget - 100) * 4)  # Rough char estimate
                            doc.page_content = doc.page_content[:max_chars] + "... [content truncated]"
                    except Exception as e:
                        logger.warning(f"Error truncating document: {e}")
                        # Simple truncation fallback
                        doc.page_content = doc.page_content[:3000] + "... [content truncated]"
                    
                    doc_tokens = self._count_tokens(doc.page_content)
            
            docs_for_context.append(doc)
            current_tokens += doc_tokens
            
            # If we've exceeded 70% of our budget, stop adding more
            if current_tokens > token_budget * 0.7:
                break
        
        return docs_for_context