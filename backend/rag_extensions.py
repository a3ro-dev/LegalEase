"""
Extended methods for the EnhancedLegalRAGSystem to support unified chat functionality.
"""

import json
import logging
from typing import Dict, Any, List, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from rag_system import IKAPITool

logger = logging.getLogger(__name__)


async def query_with_tools(self, query: str, use_web: bool = True) -> Dict[str, Any]:
    """Main query method with tools - wrapper for the existing query method"""
    return await self.query(query, use_web=use_web)


async def stream_query_with_tools(self, query: str, use_web: bool = True) -> AsyncGenerator[Any, None]:
    """Stream query results with real-time updates"""
    import time
    
    # Initial step
    yield {"type": "thinking", "content": "Analyzing your query...", "timestamp": time.time()}
    
    # Check if tools are needed
    needs_tools = await self._needs_tools(query)
    yield {
        "type": "planning", 
        "content": f"Determined that {'external tools are' if needs_tools else 'no external tools are'} needed for this query.",
        "timestamp": time.time(),
        "details": {"needs_tools": needs_tools}
    }
    
    if not needs_tools:
        # Simple response
        result = await self.generate_simple_response(query)
        yield {
            "type": "complete",
            "content": result.get("content", ""),
            "timestamp": time.time()
        }
        return
    
    # Planning phase
    yield {"type": "planning", "content": "Creating execution plan...", "timestamp": time.time()}
    plan = await self.tool_manager.create_plan(query)
    
    # Tool execution
    yield {"type": "tool_use", "content": "Executing research tools...", "timestamp": time.time()}
    result = await self.query(query, use_web=use_web)
    
    # Final response
    content = result.get("content", "") if isinstance(result, dict) else str(result)
    yield {
        "type": "complete",
        "content": content,
        "timestamp": time.time(),
        "details": {"sources": result.get("details", {}).get("sources", []) if isinstance(result, dict) else []}
    }


async def extract_keywords_advanced(self, text: str) -> Dict[str, Any]:
    """Extract legal keywords from text using the LLM"""
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal keyword extraction expert specializing in Indian law. 
            Extract important legal terms and concepts from the given text. For each term, provide a brief definition in the context of Indian law.
            
            Return your response as a JSON object with the following structure:
            {
                "terms": {
                    "term1": "definition1",
                    "term2": "definition2"
                },
                "count": number_of_terms
            }"""),
            ("human", "Text: {text}")
        ])
        
        messages = prompt.format_messages(text=text)
        response = await self.legal_specialist.ainvoke(messages)
        
        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "terms": {"error": "Failed to parse keywords"},
                "count": 0
            }
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return {"terms": {"error": str(e)}, "count": 0}


async def generate_legal_argument_advanced(self, topic: str, points: List[str]) -> Dict[str, Any]:
    """Generate a structured legal argument"""
    try:
        # First, gather relevant context from vector DB
        relevant_docs = self.retriever.invoke(topic)
        context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
        
        points_text = "\n".join([f"- {point}" for point in points]) if points else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal writing expert specializing in Indian law. 
            Create a well-structured legal argument based on the topic and key points provided.
            Use proper legal citation format and include relevant legal principles and precedents.
            Structure your argument with clear sections and logical flow."""),
            ("human", """Topic: {topic}
            
            Key Points:
            {points}
            
            Relevant Legal Context:
            {context}
            
            Please create a comprehensive legal argument.""")
        ])
        
        messages = prompt.format_messages(topic=topic, points=points_text, context=context)
        response = await self.legal_specialist.ainvoke(messages)
        
        argument = response.content
        
        return {
            "argument": argument,
            "word_count": len(argument.split()),
            "character_count": len(argument)
        }
    except Exception as e:
        logger.error(f"Error generating legal argument: {e}")
        return {"argument": f"Error generating argument: {str(e)}", "word_count": 0, "character_count": 0}


async def create_document_outline_advanced(self, topic: str, doc_type: str) -> Dict[str, Any]:
    """Create a structured document outline"""
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a legal document drafting expert specializing in Indian law.
            Create a detailed outline for a {doc_type} on the topic: {topic}
            
            Include proper sections, subsections, and key points that should be covered.
            Follow standard Indian legal document formatting conventions."""),
            ("human", "Create a comprehensive outline for: {topic}")
        ])
        
        messages = prompt.format_messages(topic=topic)
        response = await self.legal_specialist.ainvoke(messages)
        
        outline = response.content
        
        # Count sections and subsections (rough estimation)
        section_count = outline.count('\n#') + outline.count('\nSection') + outline.count('\nI.') + outline.count('\n1.')
        subsection_count = outline.count('\n##') + outline.count('\nSubsection') + outline.count('\nA.') + outline.count('\na.')
        
        return {
            "outline": outline,
            "section_count": section_count,
            "subsection_count": subsection_count
        }
    except Exception as e:
        logger.error(f"Error creating outline: {e}")
        return {"outline": f"Error creating outline: {str(e)}", "section_count": 0, "subsection_count": 0}


async def verify_citation_advanced(self, citation: str) -> Dict[str, Any]:
    """Verify a legal citation using Indian Kanoon and vector database"""
    try:
        # Use IK API tool to search for the citation
        ik_tool = IKAPITool()
        search_result = await ik_tool.run(citation, max_results=3)
        
        is_valid = False
        summary = ""
        corrected_citation = None
        
        if search_result.get("status") == "success" and search_result.get("results"):
            results = search_result.get("results", [])
            if results:
                is_valid = True
                # Use the first result for summary
                first_result = results[0]
                summary = first_result.get("content", "")[:500] + "..." if len(first_result.get("content", "")) > 500 else first_result.get("content", "")
                corrected_citation = first_result.get("citation", citation)
        
        # If not found in IK, try vector database
        if not is_valid:
            docs = self.retriever.invoke(citation)
            if docs:
                is_valid = True
                summary = docs[0].page_content[:500] + "..." if len(docs[0].page_content) > 500 else docs[0].page_content
        
        return {
            "original_citation": citation,
            "is_valid": is_valid,
            "corrected_citation": corrected_citation,
            "summary": summary if summary else "Citation could not be verified in available databases."
        }
    except Exception as e:
        logger.error(f"Error verifying citation: {e}")
        return {
            "original_citation": citation,
            "is_valid": False,
            "corrected_citation": None,
            "summary": f"Error verifying citation: {str(e)}"
        }


def extend_enhanced_rag_system():
    """Add extended methods to EnhancedLegalRAGSystem"""
    from rag_system import EnhancedLegalRAGSystem
    
    # Add methods to the class
    EnhancedLegalRAGSystem.query_with_tools = query_with_tools
    EnhancedLegalRAGSystem.stream_query_with_tools = stream_query_with_tools
    EnhancedLegalRAGSystem.extract_keywords_advanced = extract_keywords_advanced
    EnhancedLegalRAGSystem.generate_legal_argument_advanced = generate_legal_argument_advanced
    EnhancedLegalRAGSystem.create_document_outline_advanced = create_document_outline_advanced
    EnhancedLegalRAGSystem.verify_citation_advanced = verify_citation_advanced
