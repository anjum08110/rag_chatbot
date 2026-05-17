import os
from typing import List,Dict,Any
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loader import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import shutil

class DocumentProcessor:
    """Handles PDF loading and text chunking"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor
        
        Args:
            chunk_size: Number of tokens per chunk (1000 = ~750 words)
            chunk_overlap: Overlap between chunks(helps with the context)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splintter = RecursiveCharacterTextSplitter(
            chunk_size= chunk_size,
            chunk_overlap= chunk_overlap,
            separators= ["\n\n","\n", " ", ""]
        )
    
    def load_pdf(self, file_path:str) -> List:
        """
        Load a PDF and splits it into chunks
        
        Args: 
            file_path: Path to the PDF file
            
        Returns:
            List of Documents chunks
        """

        try:
            loader = PyPDFLoader(file_path)
            documents= loader.load()

            chunks = self.splintter.split_documents(documents)

            print(f"Loaded {file_path}: {len(chunks)} chunks created")
            return chunks
        
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            raise
        

class VectorStore:
    """Manages ChromaDB vector store for document embeddings"""

    def __init__(self, db_path: str = "./chroma_data"):
        """
        Initialize the vector store
                
        Args:
            db_path= Path where embeddings will be stored
        """
        self.db_path= db_path

        #Initilaizing embedding

        self.embeddings= HuggingFaceEmbeddings(
            model_name ="all-MiniLM-L6-v2", #fast and good quality
            model_kwargs={"device": "cpu"}
        )

        self.vector_store = Chroma(
            persist_directory= db_path,
            embedding_function= self.embeddings
        )

        print(f"Vector store initialized at {db_path}")

    def add_documents(self, chunks):
        """
        Add documents chunks to vector store
        
        Args:
            chunks: List of document chunks from DocumentProcessor
        """
        try:
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
            print(f"Added {len(chunks)} documents to vector store")

        except Exception as e:
            print(f"Error adding docuemnts: {str(e)}")
            raise

    def get_retriever(self, k:int =4):
        """
        Get a retriever for similartity search
        
        Args:
            k: Number of Documents to retrieve(default 4)
            
        Returns:
            A retriever object for searching
        """
        return self.vector_store.as_retriever(search_kwargs={"k":k})

    def reset(self):
        """Clear all data from vector store"""

        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
            self.vector_store= Chroma(
                persist_directory=self.db_path,
                embedding_function= self.embeddings
            )
            print("Vector store reset")
        except Exception as e:
            print(f"Error resetting store: {str(e)}")
        

class RAGPipeline:
    """Main RAG pipeline: Retrieval + Generation"""

    def __init__(self, google_api_key:str, db_path: str= "./chroma_data"):
        """
        Initialize the RAG Pipeline
        
        Args:
            google_api_key= Your Google Gemini API key
            db_path: Path to Chroma database
        """
        self.document_processor = DocumentProccesor(
            chunk_size= 1000,
            chunk_overlap= 200
        )

        self.vector_store= VectorStore(db_path=db_path)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            temparature=0.7,
            top_p= 0.9
        )

        #Chat history and file tracking
        self.chat_history = []
        self.uploaded_files = []

        #Promp template for Rag
        self.prompt_template = PromptTemplate(
            input_variables= ["context", "question"],
            template = """You are a helpful assistant that answers that questions based on the provided documents.

    Context from documents: {context}

    Question: {question}

    Instruction:
    1. Answer based Only on the provided context
    2. If the answer is not in the context, say "I don't have enough information to answer this"
    3. Be concise and Clear
    4. Cite which part of thr document you're referencing when relavent
        
    Answer: """    
            )
        
    def load_documents(self, file_path:str, file_name:str)-> bool:
        """
        Load a PDF documents and add to vector store

        Args:
            file_path: Path to the PDF file
            file_name: Name of the file(for tracking)

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Processing {file_name}...")

            chunks= self.document_processor.load_pdf(file_path)

            for chunk in chunks:
                chunk.metadata["source"] = file_name


            self.vector_store.add_documents(chunks)

            self.uploaded_files.append({
                "name": file_name,
                "timestamp": datetime.now().isoformat(),
                "chunks":len(chunks)
            })

            print(f"Successfully loaded {file_name}")
            return True
        
        except Exception as e:
            print(f"Error loading document: {str(e)}")

    def query(self,question:str) -> Dict[str, Any]:
        """
        Query the RAG pipeline
        
        Args:
            questions: User's Question
            
        Return:
            Dictionary with answer, sources and metadata"""
        
        try:
            print(f"Processing query: {question}")

            retriever= self.vector_store.get_retriever(k=4)

            retrieved_docs= retriever.get_relavent_documents(question)

            if not retrieved_docs:
                return{
                    "answer": "no relavant documents found. Please upload documents first.",
                    "sources" : [],
                    "timestamp": datetime.now().isoformat()
                }
            
            context = "\n\n---\n\n".join([
                f"[{doc.metadata.get('source','Unknown')}]\n {doc.page_content}"
                for doc in retrieved_docs
                ])
            
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )

            response=self.llm.invoke(prompt)
            answer = response.content

            sources = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "page": doc.metadata.get("page",0),
                    "source": doc.metadata.get("source","Unkonown")
                }
                for doc in retrieved_docs
            ]

            self.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })

            print(f"Query processed successfully")

            return {
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()

            }
        
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources":[],
                "timestamp": datetime.now().isoformat()
            }
        
    def get_chat_history(self) -> List[Dict]:
        """Get all chat history"""
        return self.chat_history

    def get_uploaded_files(self) -> List[Dict]:
        """Get list of uploaded files"""
        return self.uploaded_files

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return{
            "total_messages": len(self.chat_history),
            "uploaded_files": len(self.uploaded_files),
            "total_chunks": sum([f["chunks"] for f in self.uploaded_files]),
            "files": self.uploaded_files
        }

    def reset(self):
        """Reset everthing - clear all data"""
        self.vector_store.reset()
        self.chat_history =[]
        self.uploaded_files =[]
        print("RAG pipeline reset") 