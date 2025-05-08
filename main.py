from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Company Chatbot")

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize the Hugging Face model and tokenizer
MODEL_NAME = "facebook/opt-350m"  # Using a smaller model for local deployment
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Create the Hugging Face pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

# Initialize the LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize vector store
vectorstore = None

class ChatRequest(BaseModel):
    message: str
    user_type: str  # "customer" or "employee"
    chat_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str

def load_documents():
    """Load and process documents from the data directory"""
    documents = []
    data_dir = Path("data")
    
    if not data_dir.exists():
        return []
    
    for file_path in data_dir.glob("**/*"):
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
    
    if not documents:
        return []
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.create_documents(documents)
    
    # Create vector store
    return Chroma.from_documents(texts, embeddings)

@app.on_event("startup")
async def startup_event():
    """Initialize the vector store on startup"""
    global vectorstore
    vectorstore = load_documents()

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface"""
    with open("templates/index.html", "r") as f:
        return f.read()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests"""
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    # Create conversation chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    # Prepare context based on user type
    context = "You are a helpful assistant for a company. "
    if request.user_type == "customer":
        context += "You are speaking with a customer. Be professional and helpful."
    else:
        context += "You are speaking with an employee. You can provide more detailed information."
    
    # Combine context with user message
    full_message = f"{context}\nUser: {request.message}"
    
    # Get response
    result = qa_chain({"question": full_message})
    
    return ChatResponse(response=result["answer"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
