import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# LLM(to process the data)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, max_tokens=170)

# Indian Evidence Act PDF URL
INDIAN_EVIDENCE_ACT_URL = "https://www.indiacode.nic.in/bitstream/123456789/15351/1/iea_1872.pdf"
INDIAN_EVIDENCE_ACT_FILENAME = "indian_evidence_act_1872.pdf"

def download_pdf(url, filename="temp.pdf"):
    """Download PDF from URL"""
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def ensure_indian_evidence_act():
    """Ensure Indian Evidence Act PDF is downloaded and available"""
    if not os.path.exists(INDIAN_EVIDENCE_ACT_FILENAME):
        try:
            download_pdf(INDIAN_EVIDENCE_ACT_URL, INDIAN_EVIDENCE_ACT_FILENAME)
            print(f"Downloaded Indian Evidence Act PDF: {INDIAN_EVIDENCE_ACT_FILENAME}")
        except Exception as e:
            print(f"Error downloading Indian Evidence Act PDF: {e}")
            return None
    return INDIAN_EVIDENCE_ACT_FILENAME

def load_and_split_pdf(pdf_path):
    """Load and split PDF into chunks"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks

def get_embeddings():
    """Get embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_faiss_index_with_iea(chunks, embeddings, index_path="faiss_index"):
    """Create FAISS index with Indian Evidence Act always included"""

    iea_path = ensure_indian_evidence_act()
    
    if iea_path and os.path.exists(iea_path):

        iea_chunks = load_and_split_pdf(iea_path)
        

        all_chunks = chunks + iea_chunks
        

        for chunk in iea_chunks:
            if hasattr(chunk, 'metadata'):
                chunk.metadata['source'] = 'Indian Evidence Act, 1872'
                chunk.metadata['document_type'] = 'reference_law'
        
        print(f"Added {len(iea_chunks)} chunks from Indian Evidence Act to the knowledge base")
    else:
        all_chunks = chunks
        print("Warning: Could not include Indian Evidence Act in knowledge base")
    

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def create_faiss_index(chunks, embeddings, index_path="faiss_index"):
    """Create FAISS index with support for custom paths - now includes Indian Evidence Act"""
    return create_faiss_index_with_iea(chunks, embeddings, index_path)

def query_faiss(query, embeddings):
    """Query FAISS index"""
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=3)
    return results

def initialize_knowledge_base_with_iea(embeddings, index_path="faiss_index"):
    """Initialize knowledge base with only Indian Evidence Act (for chat without uploads)"""
    iea_path = ensure_indian_evidence_act()
    
    if iea_path and os.path.exists(iea_path):
        iea_chunks = load_and_split_pdf(iea_path)
        for chunk in iea_chunks:
            if hasattr(chunk, 'metadata'):
                chunk.metadata['source'] = 'Indian Evidence Act, 1872'
                chunk.metadata['document_type'] = 'reference_law'
        
        vectorstore = FAISS.from_documents(iea_chunks, embeddings)
        vectorstore.save_local(index_path)
        print(f"Initialized knowledge base with {len(iea_chunks)} chunks from Indian Evidence Act")
        return vectorstore
    else:
        print("Error: Could not initialize knowledge base with Indian Evidence Act")
        return None





