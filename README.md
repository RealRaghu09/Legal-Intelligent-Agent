# Legal-Laws: AI-Powered Legal Knowledge Base

A  legal document analysis and Q&A system built with Streamlit, LangChain, and Google Gemini AI. This application allows users to upload legal documents (PDFs) and interact with them through an intelligent chat interface.
##   Knowledge Base 
    https://www.indiacode.nic.in/bitstream/123456789/15351/1/iea_1872.pdf
## 🔑 API Keys Required

### Google Gemini API
1. Visit [Google AI Studio]
2. Create a new API key

## 🔄 Data Flow Architecture

### 1. Document Upload & Processing Flow
```
User Upload → PDF Processing → Text Extraction → Chunking → Vectorization → FAISS Index Storage
```

```
User Query → Query Vectorization → Similarity Search → Context Retrieval → LLM Processing → Response Generation
```

**Step-by-Step Process:**
1. **Query Input**: User types question in chat interface
2. **Query Vectorization**: Question is converted to vector using same embedding model
3. **Similarity Search**: FAISS finds top 3 most similar document chunks
4. **Context Assembly**: Retrieved chunks are combined with user query
5. **LLM Processing**: Google Gemini processes query + context
6. **Response Generation**: LLM generates contextual answer based on retrieved information

### 3. Thread Management Flow
```
Thread Creation → Session State Management → Thread-Specific Storage → Thread Switching → Cleanup
```

**Step-by-Step Process:**
1. **Thread Creation**: New thread generates unique UUID and creates dedicated storage
2. **State Management**: Streamlit session state tracks threads, messages, and attachments
3. **Storage Isolation**: Each thread has separate FAISS index and message history
4. **Thread Switching**: Users can switch between threads while maintaining context
5. **Cleanup**: Thread deletion removes associated files and session data

### 4. Real-Time Processing Flow
```
Upload Trigger → Progress Tracking → Background Processing → Status Updates → UI Refresh
```

**Step-by-Step Process:**
1. **Upload Trigger**: File upload initiates processing pipeline
2. **Progress Tracking**: Progress bar updates during document processing
3. **Background Processing**: Threading allows UI to remain responsive
4. **Status Updates**: Real-time status messages inform user of processing stage
5. **UI Refresh**: Interface updates when processing completes

### 5. Memory Management Flow
```
Temporary Files → Processing → Cleanup → Persistent Storage → Index Management
```

**Step-by-Step Process:**
1. **Temporary Storage**: Uploaded files stored temporarily during processing
2. **Processing**: Documents processed in memory with chunking
3. **Cleanup**: Temporary files removed after processing
4. **Persistent Storage**: FAISS indices saved to disk for future use
5. **Index Management**: Thread-specific indices managed separately

## 🛠️ Customization

### Model Configuration
- **LLM Model**: Change `gemini-1.5-flash` to other Gemini models in `Backend.py`
- **Embedding Model**: Modify `sentence-transformers/all-MiniLM-L6-v2` in `Backend.py`
- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` in the text splitter

### UI Customization
- Modify `st.set_page_config()` in `Frontend.py` for page settings
- Customize the sidebar and main interface layout
- Add additional Streamlit components as needed

### Performance Optimization
- Use smaller chunk sizes for faster processing
- Limit the number of concurrent threads
- Consider using GPU-accelerated FAISS for large document sets

## 📊 Data Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Upload   │───>│  PDF Processing  │───>│  Text Chunking  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───>│ Query Processing │<───│  Vectorization  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
--------------------Results(LLM response + Context)--------------------
```

**Step-by-Step Process:**
1. **File Upload**: User uploads PDF through Streamlit interface
2. **PDF Loading**: `PyPDFLoader` extracts text content from PDF
3. **Text Chunking**: `RecursiveCharacterTextSplitter` divides text into manageable chunks (1000 chars with 200 char overlap)
4. **Threads**creates a new thread that have context of knowledgebase and pdf uploaded if any 
4. **Embedding Generation**: Each chunk is converted to vector using `sentence-transformers/all-MiniLM-L6-v2`
5. **Index Creation**: Vectors are stored in FAISS index for fast similarity search
6. **Thread Storage**: Each conversation thread maintains its own FAISS index in `faiss_index/thread_[id]/`
7. **Generates Response**: it generates the response with context and pdf uploads
```
