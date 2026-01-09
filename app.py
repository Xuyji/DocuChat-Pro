import streamlit as st

#  Sqlite FIX
try:
    __import__('pysqlite3')
    import sys

    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# config
st.set_page_config(page_title="DocuChat Pro", layout="wide")
st.title(" DocuChat Pro (Gemini 3 & GPT-4)")

# Initialize State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "current_model_type" not in st.session_state:
    st.session_state.current_model_type = None

# sidebar
with st.sidebar:
    st.header("️ Power Settings")
    model_choice = st.radio("Choose your LLM:", ["Gemini 3 Pro", "GPT-4o"])

    api_key = ""
    if "Gemini" in model_choice:
        api_key = st.text_input("Google API Key", type="password")
        model_type = "gemini"
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
        model_type = "openai"

    st.divider()
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        st.success(f"File loaded: {uploaded_file.name}")

# reset logic
if st.session_state.current_model_type != model_type:
    st.session_state.vectorstore = None
    st.session_state.current_model_type = model_type
    st.session_state.messages = []

if uploaded_file and (st.session_state.current_file != uploaded_file.name):
    st.session_state.vectorstore = None
    st.session_state.current_file = uploaded_file.name
    st.session_state.messages = []

# Process speed
if uploaded_file and api_key and st.session_state.vectorstore is None:
    with st.spinner(f"⚡ Processing {uploaded_file.name} at max speed..."):
        try:
            # Save Temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Load & Split
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                           chunk_overlap=300)  # Larger chunks for Pro models
            splits = text_splitter.split_documents(docs)

            # Embed
            if model_type == "gemini":
                # Using the newest embedding model
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            else:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

            # Store
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.success(" Ready")

        except Exception as e:
            st.error(f"Processing Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

#  Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a complex question..."):
    if not st.session_state.vectorstore:
        st.warning("Please upload a PDF and enter an API Key first.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Select Top-Tier Models
                if model_type == "gemini":

                    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=api_key)
                else:
                    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

                # Retrieval (Fetch more context: k=5)
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                )

                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")