import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

#Initialize FastAPI app
app= FastAPI(
    title="RAG Chatbot API",
    descriptions="AI Research Assistant with RAG",
    version="1.0.0"
)

#Add Cors Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods= ["*"],
    allow_headers=["*"]
)

#Create Upload Dir if it doesn't exist
UPLOAD_DIR="./uploads"
os.makedirs(UPLOAD_DIR, exist_ok= True)

print("Initializing RAG Pipeline...")
rag_pipeline= RAGPipeline()
print("RAG Pipeline is Ready")

#PYdantics Models
class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str

class RenameTitleRequest(BaseModel):
    """Request model for renaming a chat history entry"""
    title: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


class StatsResponse(BaseModel):
    """Statistics response"""
    total_messages: int
    uploaded_files: int
    total_chunks: int
    files: List[dict]

#API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoints: verify server is running
    Returns:
        Health Status
    """
    return{
        "status":"healthy",
        "message":"RAG Chatbot is running"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File()):
    """Upload a PDF document to teh knowledge base
    
    Args:
        file: PDF file to upload
    
    Return:
        Success message with file info
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                details="Only PDF files are supported"
            )
        
        #Save file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"File uplaoded: {file.filename}")

        success = rag_pipeline.load_documents(file_path,file.filename)

        if not success:
            raise HTTPException(
                status_code= 500,
                detail= "Failed to process document"
            )
        
        return {
            "status":"success",
            "message":f"Document '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "filesize":file.size
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code= 500,
            details= f"Error uploading file: {str(e)}"
        )
    
@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the RAG pipeline with a question

    Args:
        request: QueryRequest with question field

    Returns:
        Answer with source citation
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return rag_pipeline.query(request.question)
    
@app.get("/chat-history")
async def get_chat_history():
    """Get all the chat history
    
    Returns:
    List of all messages in conversation
    """

    try:
        history = rag_pipeline.get_chat_history()
        return{
            "status": "success",
            "total_messages": len(history),
            "messages": history
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            details= f"Error retrieving history: {str(e)}"
        )

@app.delete("/chat-history/{index}")
async def delete_chat_message(index: int):
    """Delete a specific chat history entry by index"""
    success = rag_pipeline.delete_chat_message(index)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"status": "success", "message": f"Message {index} deleted"}

@app.patch("/chat-history/{index}/title")
async def rename_chat_message(index: int, request: RenameTitleRequest):
    """Rename the title of a specific chat history entry"""
    if not request.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    success = rag_pipeline.rename_chat_message(index, request.title)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"status": "success", "message": f"Message {index} renamed"}

@app.get("/uploaded-files")
async def get_uploaded_files():
    """Get list of uploaded files
    
    Returns:
        List of uploaded documents with metadata
    """
    try:
        files= rag_pipeline.get_uploaded_files()
        return{
            "status": "success",
            "total_files": len(files),
            "files": files
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving files: {str(e)}"
        )
    
@app.get("/stats", response_model= StatsResponse)
async def get_statistics():
    """Get pipeline statistics for dashboard
    
    Returns:
        Statistics about uploaded documents and messages
    """

    try:
        stats= rag_pipeline.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving stats: {str(e)}"
        )
    
@app.delete("/reset")
async def reset_pipeline():
    """Reset the entire pipeline - WARNING: This deletes all data..
    
    Returns:
        Success message
    """
    try:
        rag_pipeline.reset()

        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

        return{
            "status": "success",
            "message": "RAG pipeline reset successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting pipeline: {str(e)}"
        )
    
@app.get("/")
async def root():
    """Root endpoint with API info"""

    return{
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "description": "AI Research Assistant with RAG",
        "endpoints": {
            "health": "GET /health",
            "upload": "POST /upload",
            "query": "POST /query",
            "chat_history": "GET/chat-history",
            "files": "GET /uploaded-files",
            "stats": "GET /stats",
            "reset": "DELETE /reset",
            "docs": "GET /docs"
        },
        "docs": "Visit http://localhost:8000/docs for interactive API documentation"
    }

#MAIN

if __name__ == "__main__":
    print("\n" + "="*50)
    print("RAG Chatbot Backend Starting...")
    print("="*50)
    print("Server: http://localhost:8000")
    print("Docs: https://localhost:8000/docs")
    print("="*50 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
