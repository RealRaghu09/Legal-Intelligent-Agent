import requests

def download_pdf(url, filename="temp.pdf"):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks

from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_community.vectorstores import FAISS

def create_faiss_index(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def query_faiss(query, embeddings):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=3)
    return results



pdf_file = download_pdf("https://www.indiacode.nic.in/bitstream/123456789/15351/1/iea_1872.pdf")


chunks = load_and_split_pdf(pdf_file)


embeddings = get_embeddings()


create_faiss_index(chunks, embeddings)


query = "Is it about Indian Laws ?"
results = query_faiss(query, embeddings)

for i, res in enumerate(results, 1):
    print(f"\nResult {i}:\n{res.page_content}")

