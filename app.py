import streamlit as st
import os
import sys
from dotenv import load_dotenv

from pathlib import Path
# add the dir to python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
# Load .env into environment
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)



# Disable Streamlit's file watcher to avoid torch.classes inspection
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
    st.markdown("Upload any PDF document and I will analyze it to answer any query regarding it")


    with st.sidebar:
        st.header("Upload PDF documents")
        uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type="pdf",
                accept_multiple_files=True,
                help="this is where you are suppose to upload pdf files that you want to query."
                )

        process_docs = st.button("Process Document", type="primary")



    # initialize components in session state( they stay in memory accross different times you reload stuff and the if statements ensure that they run only once.
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

    if process_docs and uploaded_files:
        with st.spinner("Processing the uploaded documents to extract all relavent data..."):
            try:
                all_elements = []

                for uploaded_file in uploaded_files:
                    st.info(f"Processing: {uploaded_file.name}")

                    # save uploaded file temporarily
                    safe_name = Path(uploaded_file.name).name
                    temp_path = f"temp_{safe_name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # process pdf
                    elements = st.session_state.pdf_processor.process_pdf(temp_path)
                    all_elements.extend(elements)

                    # clean up temp file.... everything else will close on its own coz of with statement.
                    os.remove(temp_path)


                add_documents(st.session_state.vector_store, all_elements)
                st.session_state.documents_processed = True

                st.success(f"Sucessfully analyzed {len(uploaded_files)} documents with {len(all_elements)} elements")

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
                    # with st.sidebar:
                    #     st.write("**Debug Info:**")
                    #     st.write(f"Messages count: {len(st.session_state.messages)}")
                    #     st.write(f"Documents processed: {st.session_state.documents_processed}")
                    #
                    #     if st.button("Clear Chat History"):
                    #         st.session_state.messages = []
                    #         st.rerun()
                except Exception as e:
                    error_msg = f"Sorry sir but there is an error generating response: {str(e)}"
                    st.error(error_msg)
                    # st.code(textwrap.indent(traceback.format_exc(), "    "))
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})



if __name__ == "__main__":
    main()



