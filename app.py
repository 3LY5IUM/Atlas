import streamlit as st
import os
import sys
from dotenv import load_dotenv

from pathlib import Path

import time

PDF_DIR = os.getenv("PDF_INPUT_DIR", "./pdfs")

# add the dir to python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
# Load .env into environment
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Disable Streamlit's file watcher to avoid torch.classes inspection
import torch

torch.classes.__path__ = []
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
def sync_pdf_folder_with_vectorstore():
    """Sync PDF folder with vector store - add new, remove deleted"""
    try:
        # Get current PDFs in folder
        current_pdfs = set()
        for pdf_path in Path(PDF_DIR).rglob("*.pdf"):
            current_pdfs.add(str(pdf_path.resolve()))
        
        # Get PDFs currently in vector store
        stored_docs = st.session_state.vector_store.get(include=["metadatas"])
        stored_pdfs = set()
        
        if stored_docs.get('metadatas'):
            for metadata in stored_docs['metadatas']:
                if metadata and 'source' in metadata:
                    stored_pdfs.add(metadata['source'])
        
        # Find new PDFs to add
        new_pdfs = current_pdfs - stored_pdfs
        
        # Find deleted PDFs to remove
        deleted_pdfs = stored_pdfs - current_pdfs
        
        # Process new PDFs
        if new_pdfs:
            st.info(f"Adding {len(new_pdfs)} new PDF(s)")
            for pdf_path in new_pdfs:
                try:
                    elements = st.session_state.pdf_processor.process_pdf(
                        pdf_path, 
                        st.session_state.vector_store
                    )
                    if elements:
                        add_documents(st.session_state.vector_store, elements)
                        st.success(f"Added {Path(pdf_path).name}")
                except Exception as e:
                    st.error(f"Error adding {Path(pdf_path).name}: {str(e)}")
        
        # Remove deleted PDFs
        if deleted_pdfs:
            st.info(f"🗑️ Removing {len(deleted_pdfs)} deleted PDF(s)")
            for pdf_path in deleted_pdfs:
                try:
                    results = st.session_state.vector_store.get(
                        where={"source": pdf_path}
                    )
                    if results.get('ids'):
                        st.session_state.vector_store.delete(ids=results['ids'])
                        st.success(f"Removed {Path(pdf_path).name}")
                except Exception as e:
                    st.error(f"Error removing {Path(pdf_path).name}: {str(e)}")
        
        # Update processed flag
        if current_pdfs:
            st.session_state.documents_processed = True
        
        return len(new_pdfs), len(deleted_pdfs)
        
    except Exception as e:
        st.error(f"❌ Sync error: {str(e)}")
        return 0, 0

def display_vectorstore_status():
    """Display current PDFs in vector store"""
    try:
        # Get all documents from vector store
        all_docs = st.session_state.vector_store.get(include=["metadatas"])
        
        # Extract unique PDFs
        unique_pdfs = {}
        if all_docs.get('metadatas'):
            for metadata in all_docs['metadatas']:
                if metadata and 'source' in metadata:
                    source = metadata['source']
                    file_name = Path(source).name
                    
                    if file_name not in unique_pdfs:
                        unique_pdfs[file_name] = {
                            'source': source,
                            'count': 0,
                            'hash': metadata.get('file_hash', 'unknown')
                        }
                    unique_pdfs[file_name]['count'] += 1
        
        # Display in sidebar
        with st.sidebar:
            st.markdown("---")
            st.header("Vector Store Contents")
            
            if unique_pdfs:
                st.success(f"{len(unique_pdfs)} PDF(s) embedded")
                
                for file_name, info in unique_pdfs.items():
                    # Check if file still exists in folder
                    file_exists = Path(info['source']).exists()
                    status_icon = "✅" if file_exists else "❌"
                    
                    with st.expander(f"{status_icon} {file_name}"):
                        st.write(f"**Chunks:** {info['count']}")
                        st.write(f"**Hash:** {info['hash'][:12]}...")
                        st.write(f"**Status:** {'Exists' if file_exists else 'Missing'}")
                        
                        # Remove button
                        if st.button(f"Remove", key=f"remove_{info['hash']}"):
                            try:
                                results = st.session_state.vector_store.get(
                                    where={"source": info['source']}
                                )
                                if results.get('ids'):
                                    st.session_state.vector_store.delete(ids=results['ids'])
                                    st.success(f"Removed {file_name}")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.info("📭 No PDFs embedded yet")
                
    except Exception as e:
        st.error(f"Display error: {str(e)}")



def main():
    st.set_page_config(
        page_title="Atlas",
        page_icon="",
        layout="wide"
        )

    # sidebar
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        st.sidebar.warning("No Gemini API key found in .env.")
        api_key = st.sidebar.text_input(
            "Enter your Gemini API Key", type="password"
        )
        if api_key:
            # save the api key this time to the env file.
            with open(env_path, "a") as f:
                f.write(f"\nGEMINI_API_KEY={api_key}\n")
            os.environ["GEMINI_API_KEY"] = api_key
            st.sidebar.success("API key saved to .env")
        else:
            st.stop()

    os.environ["GEMINI_API_KEY"] = api_key

    from src.config import Config
    from src.pdf_processor import PDF_processor
    from src.vectors import setup_vs, add_documents, query
    from src.chat import get_respo

    st.title("Welcome sir, how may I assist you") 
    st.markdown(f"Process the documents present in {PDF_DIR}")

    # initialize components in session state
    if "config" not in st.session_state:
        st.session_state.config = Config()
    
    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = PDF_processor()
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = setup_vs(api_key)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    # process the uploaded documents.
    with st.sidebar:
    # chat interface.
        st.header("📁 PDF Folder Management")
        st.info(f"**Watching folder:** `{PDF_DIR}`")
        
        # Create PDF directory if it doesn't exist
        pdf_dir_path = Path(PDF_DIR)
        if not pdf_dir_path.exists():
            pdf_dir_path.mkdir(parents=True, exist_ok=True)
            st.warning(f"Created directory: `{PDF_DIR}`")
        
        # Show current folder contents
        pdf_list = list(Path(PDF_DIR).rglob("*.pdf"))
        st.write(f"Found **{len(pdf_list)}** PDF(s) in folder")
        
        # Sync controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sync PDFs", type="primary"):
                with st.spinner("Syncing PDF folder..."):
                    added, removed = sync_pdf_folder_with_vectorstore()
                    if added == 0 and removed == 0:
                        st.info("Already in sync!")
                    else:
                        st.success(f"Sync complete! Added: {added}, Removed: {removed}")
        
        with col2:
            if st.button("Auto Sync"):
                st.session_state.auto_sync = not st.session_state.get('auto_sync', False)
                if st.session_state.auto_sync:
                    st.success("Auto sync enabled")
                else:
                    st.info("Auto sync disabled")

    # Auto sync every 30 seconds if enabled
    if st.session_state.get('auto_sync', False):
        if 'last_sync' not in st.session_state:
            st.session_state.last_sync = 0
        
        current_time = time.time()
        if current_time - st.session_state.last_sync > 30:
            with st.spinner("Auto syncing..."):
                added, removed = sync_pdf_folder_with_vectorstore()
                if added > 0 or removed > 0:
                    st.toast(f"Auto sync: +{added} -{removed}")
            st.session_state.last_sync = current_time

    # Display vector store status
    display_vectorstore_status()

    st.markdown("---")
    st.header("Which query may I assist you regarding this document Sir")

    if not st.session_state.documents_processed:
        st.info("Please upload and process documents to start chatting.")
        return

    # Display chat history 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # display chat message
    if prompt := st.chat_input("What queries do you have regarding this pdf Sir"):
        # add the user mssg.
        st.session_state.messages.append({"role": "user", "content":prompt})

        # display the user query.
        with st.chat_message("user"):
            st.markdown(prompt)

        # st.chat_mssg is used for creating chating bubbles.
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Query the vector store directly
                    results = query(st.session_state.vector_store, prompt)
                
                    # Process results into expected format
                    processed_results = []
                    for doc in results:
                        processed_results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                            })

                    response = get_respo(
                            prompt,
                            processed_results,
                            # arr[start(inc): stop(exc): step]
                            st.session_state.messages[:-1]
                            )

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Sorry sir but there is an error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()

