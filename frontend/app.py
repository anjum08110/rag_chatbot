import streamlit as st
import requests
import json
from datetime import datetime
from typing import List,Dict
import os

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="images/chatbot_icon.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .source-box {
        background-color: #f0f4f8;
        padding: 12px;
        border-left: 4px solid #0066cc;
        border-radius: 4px;
        margin-top: 10px;
    }
    
    .answer-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

#Backend URL
BACKEND_URL="http://localhost:8000"

#Helper functions

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False
    
def upload_document(file):
    """Upload a PDF file to the backend"""
    try:
        files = {"file":( file.name, file.getbuffer(), "application/pdf")}
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            return True, response.json().get("message", "File uploaded successfully")
        else:
            return False, response.json().get("detail","Upload failed")
    
    except requests.exceptions.ConnectionError:
        return False," Cannot connect to backend. Make sure its running on port 8000"
    except Exception as e:
        return False, f"Error: {str(e)}"
    
def query_rag(question: str) -> Dict:
    """Send a query to the RAG pipeline"""

    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question},
            timeout=120
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Error: {response.json().get('detail', 'Unknown error')}",
                "sources": []
            }
    except requests.exceptions.Timeout:
        return {
            "answer": "Request timed out. The AI model is taking too long — this usually means the API quota is being retried. Please wait a moment and try again.",
            "sources": []
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "Cannot connect to backend. Make sure it's running on port 8000.",
            "sources": []
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }
    
def get_chat_history():
    """Fetch chat history from backend"""
    try: 
        response = requests.get(f"{BACKEND_URL}/chat-history", timeout=10)
        if response.status_code==200:
            return response.json().get("messages",[])
        return[]
    except:
        return[]
    
def get_uploaded_files():
    """Fetch list of uploaded files"""
    try:
        response = requests.get(f"{BACKEND_URL}/uploaded-files", timeout=10)
        if response.status_code==200:
            return response.json().get("files",[])
        return []
    except:
        return[]
    
def get_stats():
    """Fetch pipeline statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def reset_pipeline():
    """Reset the entire pipeline"""
    try:
        response = requests.delete(f"{BACKEND_URL}/reset", timeout=10)
        return response.status_code == 200
    except:
        return False
    

#Session State

if "chat-messages" not in st.session_state:
    st.session_state.chat_messages = []

if "backend-ready" not in st.session_state:
    st.session_state.backend_ready= check_backend_health()


#Sidebar

with st.sidebar:
    st.title("RAG Chatbot")
    st.markdown("---")

    st.subheader("Status")
    if st.session_state.backend_ready:
        st.success("Backend Connected")
    else:
        st.error("Backend Not Running")
        st.info("Start backend with: `python backend/main.py`")

    st.markdown("---")

    st.subheader("Upload Documents")

    uploaded_files= st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help= "Upload PDF documents to create a knowledge base"
    )

    if uploaded_files is not None:
        if st.button("Upload PDF", use_container_width=True):
            with st.spinner("Uploading and proceesing..."):
                success, message = upload_document(uploaded_files)
                if success:
                    st.success(message)
                    st.session_state.backend_ready = check_backend_health()
                else:
                    st.error(message)
    st.markdown("---")

    st.subheader("Uploaded Files")
    files = get_uploaded_files()

    if files: 
        for file in files: 
            col1, col2 = st.columns([3,1])
            with col1:
                st.caption(f"{file['name']}")
            with col2:
                st.caption(f"{file['chunks']} chunks")
    else:
        st.info("No files uploaded yet")

    st.markdown("---")

    st.subheader("Danger Zone")

    if st.button("Reset Everything", use_container_width=True, type="secondary"):
        if st.checkbox("I understand this will delete all data"):
            if reset_pipeline():
                st.success("Pipeline reset successfully")
                st.session_state.chat_messages = []
            else:
                st.error("Failed to reset pipeline")               


#Main Content
st.title("AI Research Assistant")
st.markdown("Upload PDFs and ask questions using RAG")

st.markdown("---")

if not st.session_state.backend_ready:
    st.error("""Backend is not running. Please start the backend server:
```
cd backend
python main.py
```""")
else:
    tab1, tab2 = st.tabs(["Chat", "History"])

    with tab1:
        st.subheader("Ask Questions About Your Documents")

        # Display chat history
        history = get_chat_history()

        if history:
            for message in history:
                with st.chat_message("user"):
                    st.write(message["question"])
                with st.chat_message("assistant"):
                    st.write(message["answer"])
                    if message.get("sources"):
                        with st.expander("Sources"):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(f"**Source {i}:** {source['source']}\n\n{source['content']}")
        else:
            st.info("No conversation yet. Upload a PDF from the sidebar, then ask a question below.")

        st.markdown("---")

        # Chat input — always visible
        col1, col2 = st.columns([6, 1])
        with col1:
            question = st.text_input(
                "question",
                placeholder="e.g. What are the main topics discussed?",
                label_visibility="collapsed",
                key="question_input"
            )
        with col2:
            submit = st.button("Ask", type="primary", use_container_width=True)

        if submit and question:
            with st.spinner("Thinking..."):
                result = query_rag(question)
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(result["answer"])
                if result.get("sources"):
                    with st.expander("Sources"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {i}:** {source['source']}\n\n{source['content']}")
            st.rerun()
        elif submit:
            st.warning("Please enter a question.")

    with tab2:
        st.subheader("Conversation History")

        history = get_chat_history()

        if history:
            for i, message in enumerate(history, 1):
                with st.expander(f"Message {i} — {message.get('timestamp', 'Unknown time')}"):
                    st.markdown("**Question:**")
                    st.write(message["question"])
                    st.markdown("**Answer:**")
                    st.write(message["answer"])
                    if message.get("sources"):
                        st.markdown("**Sources:**")
                        for source in message["sources"]:
                            st.markdown(f"- **{source['source']}** (Page {source.get('page', 'Unknown')})\n\n  {source['content']}")
        else:
            st.info("No messages yet. Start by uploading a document and asking a question!")

#Footer
st.markdown("---")

st.subheader("Technologies Used")

col1,col2,col3,col4= st.columns(4)

with col1:
    st.caption("**Backend**")
    st.caption("° FastAPI")
    st.caption("°Uvicorn")
    st.caption("• Python")

with col2:
    st.caption("**Frontend**")
    st.caption("• Streamlit")
    st.caption("• Requests")

with col3:
    st.caption("**AI & ML**")
    st.caption("• LangChain")
    st.caption("• Google Gemini")
    st.caption("• HuggingFace")

with col4:
    st.caption("**Data & Storage**")
    st.caption("• ChromaDB")
    st.caption("• PyPDF")

st.markdown("---")