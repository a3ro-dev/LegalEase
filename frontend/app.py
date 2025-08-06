import os
import streamlit as st
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the app
st.set_page_config(
    page_title="LegalEase - Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API endpoint (FastAPI backend)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #3B82F6;
    }
    .source-title {
        font-weight: bold;
        color: #1E3A8A;
    }
    .source-url {
        color: #3B82F6;
        font-size: 0.9rem;
    }
    .response-container {
        background-color: #F0F9FF;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #BAE6FD;
    }
    .help-text {
        font-size: 0.85rem;
        color: #64748B;
    }
    .keyword-box {
        background-color: #F0FDF4;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #22C55E;
    }
    .term {
        font-weight: bold;
        color: #166534;
    }
    .definition {
        color: #374151;
    }
    .outline-container {
        background-color: #FEFCE8;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #FEF08A;
    }
    .citation-container {
        background-color: #FDF2F8;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #FBCFE8;
    }
    .citation-valid {
        background-color: #ECFDF5;
        border-left: 3px solid #10B981;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    .citation-invalid {
        background-color: #FEF2F2;
        border-left: 3px solid #EF4444;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    .tabs-font {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "General Chat"

# App header
st.markdown("<h1 class='main-header'>LegalEase.app</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your AI Legal Assistant for Indian Law</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1159/1159356.png", width=100)
    st.markdown("## About LegalEase")
    st.markdown("""
    LegalEase is an AI legal assistant trained on Indian law. 
    It can help with legal research, explain legal concepts, and 
    provide information on laws and cases.
    
    **Disclaimer:** LegalEase provides information for educational purposes only. 
    It is not a substitute for professional legal advice.
    """)
    
    st.markdown("---")
    st.markdown("### Choose Function")
    
    function_options = [
        "General Chat",
        "Legal Keyword Extraction", 
        "Legal Argument Composer", 
        "Document Outline Generator", 
        "Citation Verifier"
    ]
    
    selected_function = st.radio(
        "Select a function", 
        function_options,
        key="function_selector",
        index=function_options.index(st.session_state.current_tab)
    )
    st.session_state.current_tab = selected_function

# Main content area based on selected function
if st.session_state.current_tab == "General Chat":
    st.markdown("### General Legal Chat")
    st.markdown("""
    Ask any legal question about Indian law. The system will intelligently decide whether to search 
    the knowledge base, legal databases, or the web to provide you with the most accurate answer.
    """)
    
    # Add a hint for better user experience
    with st.expander("ℹ️ How to use this feature", expanded=False):
        st.markdown("""
        1. Type your legal question in the chat box below
        2. Click 'Ask' and the system will research your question
        3. View the comprehensive answer with sources
        4. Continue the conversation with follow-up questions
        """)
    
    # Chat interface
    user_query = st.text_area("Ask your legal question:", 
                             height=100, 
                             key="general_chat_input",
                             help="Type any question about Indian law, legal procedures, or cases")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ask_button = st.button("Ask", type="primary", 
                              help="Click to get an AI-powered answer to your legal question")
    with col2:
        use_web = st.checkbox("Include web search", value=True, 
                             help="Allow the system to search the web for recent information")
    
    if ask_button and user_query:
        with st.spinner("Researching your question..."):
            try:
                response = requests.post(
                    f"{API_URL}/query/stream",
                    json={
                        "query": user_query,
                        "use_web": use_web
                    },
                    stream=True,
                    timeout=120
                )
                thinking_placeholder = st.empty()
                final_answer = ""
                sources = []
                
                for line in response.iter_lines():
                    if not line: continue
                    try:
                        step = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if step.get('type') in ['thinking','planning','tool_use','retrieval','generation']:
                            thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                        elif step.get('type') == 'complete':
                            final_answer = step.get('content', '')
                            details = step.get('details', {})
                            sources = details.get('sources', [])
                            break
                        elif step.get('type') == 'error':
                            thinking_placeholder.text(f"Error: {step['content']}")
                            break
                    except json.JSONDecodeError:
                        continue
                
                thinking_placeholder.empty()
                
                # Display the answer
                if final_answer:
                    st.markdown("### Answer")
                    st.markdown(f"""
                    <div class='response-container'>
                        {final_answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources if available
                    if sources:
                        st.markdown("### Sources")
                        for i, source in enumerate(sources, 1):
                            source_title = source.get('title', 'Legal Document')
                            source_url = source.get('source', 'N/A')
                            st.markdown(f"""
                            <div class='source-box'>
                                <div class='source-title'>{i}. {source_title}</div>
                                <div class='source-url'>{source_url}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_query,
                        'answer': final_answer,
                        'sources': sources
                    })
                else:
                    st.info("I couldn't generate an answer to your question. Please try rephrasing it.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        with st.expander(f"📝 Chat History ({len(st.session_state.chat_history)} conversations)", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):  # Show last 5 conversations
                st.markdown(f"**Q{i}:** {chat['question'][:100]}...")
                st.markdown(f"**A{i}:** {chat['answer'][:200]}...")
                if chat.get('sources'):
                    st.markdown(f"*Sources: {len(chat['sources'])} references*")
                st.markdown("---")

elif st.session_state.current_tab == "Legal Keyword Extraction":
    st.markdown("### Legal Keyword Extraction")
    st.markdown("""
    Extract key legal terms and their definitions from your text. 
    Perfect for understanding complex legal documents or preparing study materials.
    """)
    
    # Add a hint for better user experience
    with st.expander("ℹ️ How to use this feature", expanded=False):
        st.markdown("""
        1. Paste or type any legal text in the area below
        2. Click 'Extract Keywords' and wait for processing
        3. Review the extracted terms and their definitions
        4. Download the results in JSON format if needed
        """)
    
    legal_text = st.text_area("Paste legal text for keyword extraction:", 
                             height=200, 
                             key="legal_text_input",
                             help="Paste any legal document text here for analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        extract_button = st.button("Extract Keywords", type="primary", 
                                 help="Click to analyze the text and extract legal terms")
    
    if extract_button and legal_text:
        with st.spinner("Extracting legal keywords..."):
            try:
                response = requests.post(
                    f"{API_URL}/extract_keywords/stream",
                    json={
                        "text": legal_text
                    },
                    stream=True,
                    timeout=60
                )
                thinking_placeholder = st.empty()
                terms = {}
                for line in response.iter_lines():
                    if not line: continue
                    try:
                        step = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if step.get('type') in ['thinking','retrieval','generation']:
                            thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                        elif step.get('type') == 'complete':
                            content = step.get('content', '')
                            details = step.get('details', {})
                            try:
                                # Try to parse the content as JSON (keywords)
                                terms = json.loads(content) if content else {}
                            except json.JSONDecodeError:
                                # If content is not JSON, check details for terms
                                terms = details.get('terms', {})
                            break
                        elif step.get('type') == 'error':
                            thinking_placeholder.text(f"Error: {step['content']}")
                            break
                    except json.JSONDecodeError:
                        continue
                
                # Display results
                if terms:
                    st.success(f"Found {len(terms)} legal terms!")
                    
                    # Only create download button once, outside the loop
                    terms_json = json.dumps(terms, indent=2)
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.download_button(
                            label="Download Terms as JSON",
                            data=terms_json,
                            file_name="legal_terms.json",
                            mime="application/json",
                            key="download_terms_json",  # Add a unique key
                            help="Save extracted terms to a JSON file"
                        )
                    
                    # Display terms in a nice format
                    st.markdown("### Extracted Legal Terms")
                    for term, definition in terms.items():
                        st.markdown(f"""
                        <div class='keyword-box'>
                            <div class='term'>{term}</div>
                            <div class='definition'>{definition}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specialized legal terms were found in the text.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Update Argument Composer to streaming
elif st.session_state.current_tab == "Legal Argument Composer":
    with st.expander("ℹ️ How to use this feature", expanded=False):
        st.markdown("1. Enter the main legal topic or issue you need an argument for.\n2. List key points you want included.\n3. Click 'Generate Argument' to compose your legal argument.")
    st.markdown("### Legal Argument Composer")
    st.markdown("""
    Generate structured legal arguments ready to paste into your documents.
    Provide a topic and key points to include, and LegalEase will compose a coherent legal argument.
    """)
    
    topic = st.text_input("Main legal topic or issue:", key="argument_topic", help="Enter the central legal question or topic for your argument.")
    
    # Dynamically add points with help
    if 'argument_points' not in st.session_state:
        st.session_state.argument_points = ["", "", ""]
    
    st.markdown("#### Key points to include:")
    
    # Display existing points
    for i, point in enumerate(st.session_state.argument_points):
        st.session_state.argument_points[i] = st.text_input(
            f"Point {i+1}", value=point, key=f"point_{i}", 
            help="Enter a key point or argument heading to include in the legal argument."
        )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Add Point", help="Click to add another key point field.") and len(st.session_state.argument_points) < 10:
            st.session_state.argument_points.append("")
            st.rerun()
    with col2:
        if st.button("Remove Point", help="Click to remove the last key point field.") and len(st.session_state.argument_points) > 1:
            st.session_state.argument_points.pop()
            st.rerun()
    
    if st.button("Generate Argument", type="primary") and topic:
        # Filter out empty points
        points = [p for p in st.session_state.argument_points if p.strip()]
        
        if not points:
            st.warning("Please add at least one key point for your argument.")
        else:
            with st.spinner("Composing legal argument..."):
                try:
                    response = requests.post(
                        f"{API_URL}/generate_argument/stream",
                        json={
                            "topic": topic,
                            "points": points
                        },
                        stream=True,
                        timeout=90
                    )
                    thinking_placeholder = st.empty()
                    argument = ""
                    details = {}
                    for line in response.iter_lines():
                        if not line: continue
                        try:
                            step = json.loads(line.decode('utf-8').replace('data: ', ''))
                            if step.get('type') in ['thinking','retrieval','generation','planning']:
                                thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                            elif step.get('type') == 'complete':
                                argument = step.get('content', '')
                                details = step.get('details', {})
                                break
                            elif step.get('type') == 'error':
                                thinking_placeholder.text(f"Error: {step['content']}")
                                break
                        except json.JSONDecodeError:
                            continue
                
                    thinking_placeholder.empty()
                    
                    if argument:
                        st.markdown("### Generated Legal Argument")
                        st.markdown(argument)
                        st.download_button(label="Download as Markdown", data=argument, file_name="legal_argument.md", mime="text/markdown")
                        word_count = details.get('word_count', len(argument.split()))
                        st.info(f"Argument length: {word_count} words")
                    else:
                        st.info("Could not generate an argument with the provided information.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Update Outline Generator to streaming
elif st.session_state.current_tab == "Document Outline Generator":
    with st.expander("ℹ️ How to use this feature", expanded=False):
        st.markdown("1. Select the type of legal document you want to create an outline for.\n2. Enter the topic or subject of the document.\n3. Click 'Generate Outline' to receive a structured outline.")
    st.markdown("### Document Outline Generator")
    st.markdown("""
    Get a professional document structure for your legal documents. 
    Select the document type and enter your topic to receive a comprehensive outline.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        doc_type = st.selectbox(
            "Document Type", 
            ["Legal Brief", "Legal Memo", "Legal Opinion", "Contract", "Petition", 
             "Complaint", "Settlement Agreement", "Legal Notice", "Affidavit", "Will"],
            help="Choose the type of legal document you need an outline for."
        )
    with col2:
        outline_topic = st.text_input("Document Topic:", key="outline_topic", 
                                     help="Enter the subject or issue the document will address.")
    
    if st.button("Generate Outline", type="primary") and outline_topic and doc_type:
        with st.spinner("Generating document outline..."):
            try:
                response = requests.post(
                    f"{API_URL}/create_outline/stream",
                    json={
                        "topic": outline_topic,
                        "doc_type": doc_type
                    },
                    stream=True,
                    timeout=60
                )
                thinking_placeholder = st.empty()
                outline_text = ""
                details = {}
                for line in response.iter_lines():
                    if not line: continue
                    try:
                        step = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if step.get('type') in ['thinking','retrieval','generation','planning']:
                            thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                        elif step.get('type') == 'complete':
                            outline_text = step.get('content', '')
                            details = step.get('details', {})
                            break
                        elif step.get('type') == 'error':
                            thinking_placeholder.text(f"Error: {step['content']}")
                            break
                    except json.JSONDecodeError:
                        continue
                if outline_text:
                    st.markdown("### Document Outline")
                    st.markdown(f"<div class='outline-container'>{outline_text}</div>", unsafe_allow_html=True)
                    st.download_button(label="Download Outline as Markdown", data=outline_text, file_name=f"{doc_type.lower().replace(' ','_')}_outline.md", mime="text/markdown")
                    st.info(f"Outline structure: {details.get('section_count',0)} main sections with {details.get('subsection_count',0)} subsections")
                else:
                    st.info("Could not generate an outline with the provided information.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Update Citation Verifier to streaming
elif st.session_state.current_tab == "Citation Verifier":
    with st.expander("ℹ️ How to use this feature", expanded=False):
        st.markdown("1. Paste or type the legal citation you want to verify.\n2. Click 'Verify Citation'.\n3. Review whether the citation is valid, corrections, and summary of the law.")
    st.markdown("### Citation Verifier")
    st.markdown("""
    Verify the accuracy of legal citations, get corrections for improper formats, 
    and check if citations refer to valid cases or statutes.
    """)
    
    citation = st.text_input(
        "Enter legal citation to verify:", 
        key="citation_input", 
        placeholder="e.g., AIR 1950 SC 27 or (2019) 1 SCC 1",
        help="Type the citation you wish to check for format and validity."
    )
    
    if st.button("Verify Citation", type="primary") and citation:
        with st.spinner("Verifying citation..."):
            try:
                response = requests.post(
                    f"{API_URL}/verify_citation/stream",
                    json={
                        "citation": citation
                    },
                    stream=True,
                    timeout=60
                )
                thinking_placeholder = st.empty()
                result_data = {}
                for line in response.iter_lines():
                    if not line: continue
                    try:
                        step = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if step.get('type') in ['thinking','tool_use','retrieval','generation']:
                            thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                        elif step.get('type') == 'complete':
                            result_data = step.get('details', {})
                            break
                        elif step.get('type') == 'error':
                            thinking_placeholder.text(f"Error: {step['content']}")
                            break
                    except json.JSONDecodeError:
                        continue
                if result_data:
                    # Display verification result
                    original = result_data.get("original_citation", citation)
                    is_valid = result_data.get("is_valid", False)
                    corrected = result_data.get("corrected_citation", "")
                    summary = result_data.get("summary", "")
                    error_details = result_data.get("error_details", "")
                    
                    st.markdown("<div class='citation-container'>", unsafe_allow_html=True)
                    
                    # Original citation
                    st.markdown(f"**Original Citation:** {original}")
                    
                    # Valid or Invalid indicator
                    if is_valid:
                        st.markdown("""
                        <div class='citation-valid'>
                            <strong>✓ Valid Citation</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='citation-invalid'>
                            <strong>✗ Invalid Citation</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Corrected citation if available
                    if corrected and corrected != original:
                        st.markdown(f"**Corrected Citation:** {corrected}")
                    
                    # Summary of cited case/law
                    if summary:
                        st.markdown("**Summary:**")
                        st.markdown(summary)
                    
                    # Error details if invalid
                    if not is_valid and error_details:
                        st.markdown("**Issues:**")
                        st.markdown(error_details)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("No verification data received.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p class='help-text'>LegalEase.app is powered by advanced AI technology and a comprehensive Indian legal database.</p>", unsafe_allow_html=True)
