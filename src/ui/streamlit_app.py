"""
Streamlit chat interface for the RAG system.
Provides user-friendly interface for document upload and chat functionality.
"""

import streamlit as st
import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Local RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health() -> bool:
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(file) -> Dict[str, Any]:
    """Upload document to the API"""
    files = {"file": (file.name, file, "application/pdf")}
    response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
    return response.json()

def get_documents() -> List[Dict[str, Any]]:
    """Get list of documents from API"""
    response = requests.get(f"{API_BASE_URL}/documents")
    return response.json()

def get_document_stats() -> Dict[str, Any]:
    """Get document statistics"""
    response = requests.get(f"{API_BASE_URL}/documents/stats")
    return response.json()

def search_documents(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search documents"""
    payload = {
        "query": query,
        "max_results": max_results,
        "similarity_threshold": 0.5
    }
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    return response.json()

def send_chat_message(message: str, session_id: int = None) -> Dict[str, Any]:
    """Send chat message to API"""
    payload = {
        "message": message,
        "session_id": session_id,
        "max_chunks": 5,
        "similarity_threshold": 0.5
    }
    response = requests.post(f"{API_BASE_URL}/chat", json=payload)
    return response.json()

def get_chat_sessions() -> List[Dict[str, Any]]:
    """Get list of chat sessions"""
    response = requests.get(f"{API_BASE_URL}/chat/sessions")
    return response.json()

def get_chat_history(session_id: int) -> List[Dict[str, Any]]:
    """Get chat history for a session"""
    response = requests.get(f"{API_BASE_URL}/chat/sessions/{session_id}/history")
    return response.json()

def main():
    """Main Streamlit application"""
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running. Please start the FastAPI backend first.")
        st.code("python -m src.api.main")
        return
    
    st.title("📚 Local RAG System")
    st.markdown("Chat with your documents using local embeddings and Claude API")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Document statistics
        try:
            stats = get_document_stats()
            st.metric("Documents", stats.get('total_documents', 0))
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Completed", stats.get('completed_documents', 0))
        except:
            st.error("Failed to load stats")
        
        st.header("📄 Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF document to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Uploading document..."):
                    try:
                        result = upload_document(uploaded_file)
                        st.success(result.get('message', 'Document uploaded successfully'))
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
        
        # Document list
        st.subheader("📋 Documents")
        try:
            documents = get_documents()
            for doc in documents[-5:]:  # Show last 5 documents
                status_emoji = {
                    'completed': '✅',
                    'processing': '⏳',
                    'failed': '❌',
                    'pending': '⏸️'
                }.get(doc['status'], '❓')
                
                st.write(f"{status_emoji} {doc['filename']}")
                st.caption(f"Chunks: {doc['chunk_count']}")
        except:
            st.error("Failed to load documents")
    
    # Main content area - tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🔍 Search"])
    
    with tab1:
        # Chat interface
        st.header("Chat with Your Documents")
        
        # Session management
        col1, col2 = st.columns([3, 1])
        with col1:
            if 'current_session_id' not in st.session_state:
                st.session_state.current_session_id = None
        
        with col2:
            if st.button("New Chat"):
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.rerun()
        
        # Initialize chat messages
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Load chat history if session exists
        if st.session_state.current_session_id and not st.session_state.messages:
            try:
                history = get_chat_history(st.session_state.current_session_id)
                st.session_state.messages = history
            except:
                pass
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and message.get("retrieved_chunks"):
                    with st.expander("📚 Sources"):
                        for chunk in message["retrieved_chunks"]:
                            st.write(f"**{chunk['filename']}** (Page {chunk['page_number']})")
                            st.write(f"Similarity: {chunk.get('similarity_score', 0):.3f}")
                            st.caption(chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text'])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = send_chat_message(prompt, st.session_state.current_session_id)
                        
                        # Update session ID if new
                        if not st.session_state.current_session_id:
                            st.session_state.current_session_id = response['session_id']
                        
                        # Display response
                        st.markdown(response['response'])
                        
                        # Show sources
                        if response['retrieved_chunks']:
                            with st.expander("📚 Sources"):
                                for chunk in response['retrieved_chunks']:
                                    st.write(f"**{chunk['filename']}** (Page {chunk['page_number']})")
                                    st.write(f"Similarity: {chunk.get('similarity_score', 0):.3f}")
                                    st.caption(chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text'])
                        
                        # Add assistant message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response['response'],
                            "retrieved_chunks": response['retrieved_chunks']
                        })
                        
                        # Show token usage
                        if response.get('tokens_used'):
                            st.caption(f"Tokens used: {response['tokens_used']}")
                        
                    except Exception as e:
                        st.error(f"Failed to get response: {str(e)}")
    
    with tab2:
        # Search interface
        st.header("Search Documents")
        
        search_query = st.text_input("Enter search query:")
        max_results = st.slider("Maximum results", 1, 20, 5)
        
        if search_query:
            if st.button("Search"):
                with st.spinner("Searching..."):
                    try:
                        results = search_documents(search_query, max_results)
                        
                        st.write(f"Found {len(results['results'])} results:")
                        
                        for i, result in enumerate(results['results'], 1):
                            with st.expander(f"{i}. {result['filename']} (Page {result['page_number']}) - Similarity: {result.get('similarity_score', 0):.3f}"):
                                st.write(result['chunk_text'])
                                st.caption(f"Chunk ID: {result['id']}")
                    
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")

if __name__ == "__main__":
    main()