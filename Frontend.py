import streamlit as st
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path
import uuid
from datetime import datetime
import shutil

from Backend import load_and_split_pdf, get_embeddings, create_faiss_index, query_faiss,initialize_knowledge_base_with_iea



from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Legal Knowledge Base",
    layout="wide",
    initial_sidebar_state="expanded"
)

# For threads and session states
if 'threads' not in st.session_state:
    st.session_state.threads = []
if 'current_thread_id' not in st.session_state:
    st.session_state.current_thread_id = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = {}
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"
if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = 0
if 'thread_attachments' not in st.session_state:
    st.session_state.thread_attachments = {}
if 'llm_ready' not in st.session_state:
    st.session_state.llm_ready = False
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
# llm init
def initialize_llm():
    """Initialize the Gemini LLM"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_tokens=1000)
        st.session_state.llm_ready = True
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        st.session_state.llm_ready = False
        return None
# for every new thread creation
def create_new_thread(title="New Thread"):
    """Create a new thread"""
    thread_id = str(uuid.uuid4())
    thread = {
        'id': thread_id,
        'title': title,
        'created_at': datetime.now(),
        'last_activity': datetime.now()
    }
    st.session_state.threads.append(thread)
    st.session_state.chat_messages[thread_id] = []
    st.session_state.thread_attachments[thread_id] = []
    return thread_id
# for adding message to a specific thread
def add_message_to_thread(thread_id, message, sender="user", message_type="text"):
    """Add a message to a specific thread"""
    if thread_id not in st.session_state.chat_messages:
        st.session_state.chat_messages[thread_id] = []
    
    msg = {
        'id': str(uuid.uuid4()),
        'content': message,
        'sender': sender,
        'timestamp': datetime.now(),
        'type': message_type
    }
    st.session_state.chat_messages[thread_id].append(msg)
    for thread in st.session_state.threads:# check this one error?
        if thread['id'] == thread_id:
            thread['last_activity'] = datetime.now()
            break

def process_pdf_with_threading(pdf_file, thread_id, progress_callback=None):
    """Process PDF with threading support for thread-specific knowledge base"""
    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_file.seek(0)
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
            
        if progress_callback:
            progress_callback(20)
        

        chunks = load_and_split_pdf(tmp_path)
        
        if progress_callback:
            progress_callback(50)
        
        embeddings = get_embeddings()
        
        if progress_callback:
            progress_callback(70)
        

        thread_faiss_path = f"faiss_index/thread_{thread_id}"
        

        os.makedirs(thread_faiss_path, exist_ok=True)
        

        if os.path.exists(f"{thread_faiss_path}/index.faiss"):

        # ndian Evidence Act will be included
            vectorstore = FAISS.load_local(thread_faiss_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(chunks)
        else:
            vectorstore = create_faiss_index(chunks, embeddings, thread_faiss_path)
        
        vectorstore.save_local(thread_faiss_path)
        
        os.unlink(tmp_path)
        
        if progress_callback:
            progress_callback(100)
        
        attachment = {
            'filename': pdf_file.name,
            'status': 'processed',
            'processed_at': datetime.now(),
            'chunks_count': len(chunks),
            'file_size': len(pdf_file.getvalue()) if hasattr(pdf_file, 'getvalue') else 0
        }
        st.session_state.thread_attachments[thread_id].append(attachment)
        
        return True, f"Successfully processed {pdf_file.name} (with Indian Evidence Act included)"
        
    except Exception as e:
        return False, f"Error processing {pdf_file.name}: {str(e)}"

def update_progress(progress):
    """Update progress in session state"""
    st.session_state.processing_progress = progress

def process_files_threaded(uploaded_files, thread_id):
    """Process multiple files using threading"""
    st.session_state.processing_status = "processing"
    st.session_state.processing_progress = 0
    st.session_state.processing_results = []
    
    results = []
    

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(process_pdf_with_threading, file, thread_id, update_progress): file 
            for file in uploaded_files
        }
        

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                success, message = future.result()
                results.append((file.name, success, message))
                st.session_state.processing_results.append((file.name, success, message))
            except Exception as e:
                error_result = (file.name, False, f"Exception: {str(e)}")
                results.append(error_result)
                st.session_state.processing_results.append(error_result)
    
    st.session_state.processing_status = "completed"
    return results

def query_faiss_thread_specific(query, embeddings, thread_index_path):
    """Query thread-specific FAISS index"""
    vectorstore = FAISS.load_local(thread_index_path, embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=5)
    return results

def query_knowledge_base(query, thread_id):
    """Query the knowledge base and return results with Gemini AI enhancement"""
    try:
        thread_faiss_path = f"faiss_index/thread_{thread_id}"
        
        if not os.path.exists(f"{thread_faiss_path}/index.faiss"):

            embeddings = get_embeddings()
            initialize_knowledge_base_with_iea(embeddings, thread_faiss_path)
            
            system_msg = "📚 **System**: Knowledge base initialized with Indian Evidence Act, 1872. You can now ask questions about evidence law."
            add_message_to_thread(thread_id, system_msg, "assistant", "system")
        
        embeddings = get_embeddings()
        faiss_results = query_faiss_thread_specific(query, embeddings, thread_faiss_path)
        
        faiss_response = "**🔍 FAISS Search Results:**\n\n"
        for i, result in enumerate(faiss_results, 1):
            faiss_response += f"**Result {i}:**\n{result.page_content}\n\n"
            if hasattr(result, 'metadata'):
                source = result.metadata.get('source', 'Unknown')
                page = result.metadata.get('page', 'Unknown')
                faiss_response += f"*Source: {source}*\n"
                faiss_response += f"*Page: {page}*\n\n"
        
        add_message_to_thread(thread_id, faiss_response, "assistant", "faiss_results")
        
        if st.session_state.llm_ready:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_tokens=1000)
                
                context = f"""Based on the following search results from legal documents (including Indian Evidence Act, 1872) in this thread, please provide a comprehensive and accurate answer to: {query}

Search Results:
"""
                for i, result in enumerate(faiss_results, 1):
                    context += f"Result {i}: {result.page_content}\n\n"
                
                context += "\nPlease provide a clear, well-structured response that directly addresses the query using the information from the search results. If the query relates to evidence law, prioritize information from the Indian Evidence Act, 1872."
                
                gemini_response = llm.invoke(context)
                
                ai_response = "** AI Analysis:**\n\n"
                ai_response += gemini_response.content
                
                add_message_to_thread(thread_id, ai_response, "assistant", "ai_analysis")
                
                return "Both search results and AI analysis have been added to the chat."
                
            except Exception as e:
                error_msg = f"Error getting AI response: {str(e)}"
                add_message_to_thread(thread_id, error_msg, "assistant", "error")
                return "Search results added, but AI analysis failed."
        
        else:
            return "Search results added to chat. AI analysis not available."
            
    except Exception as e:
        error_msg = f"Error searching knowledge base: {str(e)}"
        add_message_to_thread(thread_id, error_msg, "assistant", "error")
        return error_msg

def delete_thread(thread_id):
    """Delete a thread and its associated knowledge base"""

    st.session_state.threads = [t for t in st.session_state.threads if t['id'] != thread_id]
    

    if thread_id in st.session_state.chat_messages:
        del st.session_state.chat_messages[thread_id]
    

    if thread_id in st.session_state.thread_attachments:
        del st.session_state.thread_attachments[thread_id]
    

    thread_faiss_path = f"faiss_index/thread_{thread_id}"
    if os.path.exists(thread_faiss_path):
        shutil.rmtree(thread_faiss_path)
    

    if st.session_state.current_thread_id == thread_id:
        st.session_state.current_thread_id = None

def main():
    st.title("⚖️ Legal Knowledge Base with Threads & AI")
    st.markdown("Organize your legal research with threads and chat with your documents using AI")
    

    if not st.session_state.llm_ready:
        initialize_llm()
    

    col1, col2 = st.columns([1, 2])
    

    with col1:
        st.header("🧵 Threads")
        

        if st.button("➕ New Thread", type="primary"):
            new_thread_id = create_new_thread()
            st.session_state.current_thread_id = new_thread_id
            st.rerun()
        
        st.markdown("---")
        

        if st.session_state.threads:
            for thread in st.session_state.threads:

                is_selected = st.session_state.current_thread_id == thread['id']
                

                with st.container():
                    if is_selected:
                        st.markdown(f"** {thread['title']}**")
                    else:
                        st.markdown(f"** {thread['title']}")
                    

                    st.caption(f"Created: {thread['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"Last activity: {thread['last_activity'].strftime('%Y-%m-%d %H:%M')}")
                    

                    attachments = st.session_state.thread_attachments.get(thread['id'], [])
                    st.caption(f" {len(attachments)} documents")
                    

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"Open", key=f"open_{thread['id']}"):
                            st.session_state.current_thread_id = thread['id']
                            st.rerun()
                    with col_b:
                        if st.button(f"Delete", key=f"delete_{thread['id']}"):
                            delete_thread(thread['id'])
                            st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("No threads yet. Create your first thread!")
    

    with col2:
        if st.session_state.current_thread_id:
            current_thread = next((t for t in st.session_state.threads if t['id'] == st.session_state.current_thread_id), None)
            
            if current_thread:
                st.header(f"{current_thread['title']}")
                
                with st.expander("�� Upload Documents", expanded=False):
                    uploaded_files = st.file_uploader(
                        "Choose PDF files",
                        type=['pdf'],
                        accept_multiple_files=True,
                        key=f"uploader_{st.session_state.current_thread_id}",
                        help="Upload PDF documents to this thread"
                    )
                    
                    if uploaded_files:

                        st.success(f"{len(uploaded_files)} file(s) selected")
                        

                        for file in uploaded_files:
                            st.write(f" {file.name}")
                        
                        st.markdown("---")
                        
                        if st.button(" Process Documents", type="primary", key=f"process_{st.session_state.current_thread_id}"):

                            current_thread_id = st.session_state.current_thread_id
                            processing_thread = threading.Thread(
                                target=lambda: process_files_threaded(uploaded_files, current_thread_id)
                            )
                            processing_thread.start()
                

                if st.session_state.current_thread_id in st.session_state.thread_attachments:
                    attachments = st.session_state.thread_attachments[st.session_state.current_thread_id]
                    if attachments:
                        st.subheader(" Attachments")
                        for attachment in attachments:
                            status_icon = "✅" if attachment['status'] == 'processed' else "⏳"
                            st.write(f"{status_icon} {attachment['filename']}")
                            if attachment['status'] == 'processed':
                                st.caption(f"Chunks: {attachment['chunks_count']} | Size: {attachment['file_size']} bytes")
                

                if st.session_state.processing_status == "processing":
                    st.info(" Processing documents...")
                    progress_bar = st.progress(st.session_state.processing_progress / 100)
                    st.write(f"Progress: {st.session_state.processing_progress}%")
                    time.sleep(0.1)
                    st.rerun()
                

                if st.session_state.processing_status == "completed" and st.session_state.processing_results:
                    st.success(" Processing completed!")
                    
                    for filename, success, message in st.session_state.processing_results:
                        if success:
                            st.success(f"✅ {filename} - {message}")
                        else:
                            st.error(f"❌ {filename} - {message}")
                    
                    st.session_state.processing_results = []
                    st.session_state.processing_status = "idle"
                    time.sleep(2)
                    st.rerun()


                st.markdown("---")
                st.subheader(" Chat with Documents")
                

                chat_container = st.container()
                with chat_container:
                    if st.session_state.current_thread_id in st.session_state.chat_messages:
                        messages = st.session_state.chat_messages[st.session_state.current_thread_id]
                        
                        for message in messages:
                            if message['sender'] == 'user':
                                st.markdown(f"** You:** {message['content']}")
                            elif message['type'] == 'faiss_results':
                                st.markdown(f"** Search Results:**")
                                st.markdown(message['content'])
                            elif message['type'] == 'ai_analysis':
                                st.markdown(f"** AI Analysis:**")
                                st.markdown(message['content'])
                            elif message['type'] == 'error':
                                st.error(f"**❌ Error:** {message['content']}")
                            elif message['type'] == 'system':
                                st.markdown(f"** System:** {message['content']}")
                            else:
                                st.markdown(f"** Assistant:** {message['content']}")
                            
                            st.caption(f"{message['timestamp'].strftime('%H:%M')}")
                            st.markdown("---")
                

                with st.container():
                    user_input = st.text_input(
                        "Ask about your documents:",
                        placeholder="What would you like to know about your documents?",
                        key=f"chat_input_{st.session_state.current_thread_id}"
                    )
                    
                    col_send, col_clear = st.columns([1, 1])
                    with col_send:
                        if st.button("Send", type="primary", key=f"send_{st.session_state.current_thread_id}"):
                            if user_input:

                                thread_faiss_path = f"faiss_index/thread_{st.session_state.current_thread_id}"
                                if not os.path.exists(f"{thread_faiss_path}/index.faiss"):
                                    st.error("Please upload and process documents to this thread first!")
                                else:

                                    add_message_to_thread(st.session_state.current_thread_id, user_input, "user")
                                    with st.spinner("Searching knowledge base and generating AI response..."):
                                        response = query_knowledge_base(user_input, st.session_state.current_thread_id)
                                    
                                    st.rerun()
                    
                    with col_clear:
                        if st.button("Clear Chat", key=f"clear_{st.session_state.current_thread_id}"):
                            st.session_state.chat_messages[st.session_state.current_thread_id] = []
                            st.rerun()
        else:
            st.header(" Chat")
            st.info("Select a thread from the left panel to start chatting!")
    

    with st.expander("System Status"):
        st.subheader("System Information")
        st.write(f"**LLM Status:** {' Ready' if st.session_state.llm_ready else 'Not Ready'}")
        st.write(f"**Processing Status:** {st.session_state.processing_status}")
        st.write(f"**Active Threads:** {len(st.session_state.threads)}")
        

        if st.session_state.current_thread_id:
            thread_faiss_path = f"faiss_index/thread_{st.session_state.current_thread_id}"
            has_knowledge_base = os.path.exists(f"{thread_faiss_path}/index.faiss")
            st.write(f"**Current Thread Knowledge Base:** {' Ready' if has_knowledge_base else ' No Documents'}")
            
            if has_knowledge_base:
                attachments = st.session_state.thread_attachments.get(st.session_state.current_thread_id, [])
                st.write(f"**Documents in Thread:** {len(attachments)}")
        

        if not st.session_state.llm_ready:
            if st.button(" Initialize LLM"):
                initialize_llm()
                st.rerun()
        
        if st.button(" Reset All Knowledge Bases"):
            faiss_index_path = Path("faiss_index")
            if faiss_index_path.exists():
                shutil.rmtree(faiss_index_path)
                st.session_state.processing_status = "idle"
                st.success("All knowledge bases reset successfully!")
                st.rerun()

if __name__ == "__main__":
    main()
