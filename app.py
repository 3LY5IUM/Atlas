import streamlit as st
import os
import sys
from dotenv import load_dotenv

from pathlib import Path

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

    with st.sidebar:
        pdf_paths = Path(PDF_DIR).rglob("*.pdf")          # recursive
        pdf_list  = sorted(pdf_paths)                     # list[Path]

        st.write(f"Found **{len(pdf_list)}** PDF(s) in _{PDF_DIR}_")
        scan_now = st.button("Process PDFs")

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
    if scan_now and pdf_list:
        with st.spinner("Processing the uploaded documents to extract all relavent data..."):
            try:
                all_elements = []

                for pdf_path in pdf_list:
                    st.info(f"Processing: {pdf_path.name}")
                    
                    # CHANGE: Pass vector_store to process_pdf for duplicate checking
                    elements = st.session_state.pdf_processor.process_pdf(
                        str(pdf_path), 
                        st.session_state.vector_store
                    )
                    
                    # CHANGE: Only add if elements exist (not already processed)
                    if elements:
                        all_elements.extend(elements)
                    else:
                        st.info(f"📋 {pdf_path.name} already processed - skipped")

                # CHANGE: Handle case where all files were already processed
                if all_elements:
                    add_documents(st.session_state.vector_store, all_elements)
                    st.success(f"Successfully analyzed {len(pdf_list)} documents with {len(all_elements)} elements")
                else:
                    st.info("📋 All documents already processed - no new elements added")
                
                # Always mark as processed so chat interface becomes available
                st.session_state.documents_processed = True

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    # chat interface.
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

