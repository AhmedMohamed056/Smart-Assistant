import streamlit as st
import os
from dotenv import load_dotenv

# =========================
# Modern LangChain Imports
# =========================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 

# =========================
# Load environment variables
# =========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# =========================
# Streamlit Config
# =========================
st.set_page_config(
    page_title="Smart Assistant - RAG",
    layout="wide"
)

st.title("🤖 Smart Assistant – Customer Support RAG")
st.write("Upload policy documents and ask contextual questions based ONLY on them.")

# =========================
# Temp directory for PDFs
# =========================
PDF_DIR = "temp_pdfs"
DB_DIR = "temp_chroma_db"

os.makedirs(PDF_DIR, exist_ok=True)

# =========================
# Initialize Session States
# =========================
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# Sidebar – Upload PDFs
# =========================
st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF policy documents",
    type=["pdf"],
    accept_multiple_files=True
)

def process_and_initialize(files):
    documents = []
    
    for file in files:
        file_path = os.path.join(PDF_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question. "
        "Return ONLY the formulated standalone question and absolutely NOTHING else. No intro, no outro."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are a professional customer support assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not found in the context, say: "
        "'I’m sorry, this information is not available in the provided documents.'\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

if uploaded_files and st.sidebar.button("Process Documents"):
    with st.spinner("Processing documents and building Vector Database..."):
        st.session_state.qa_chain = process_and_initialize(uploaded_files)
        st.session_state.messages = [] 
        st.sidebar.success("Documents processed successfully!")

# =========================
# 💬 Chat Interface 
# =========================
st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

welcome_placeholder = st.empty()

if len(st.session_state.messages) == 0:
    with welcome_placeholder.container():
        st.markdown("### 👋 Welcome to the Smart Assistant!")
        st.info("I am ready to answer your questions based on the uploaded documents. Please upload your PDFs from the sidebar to get started.")

if user_query := st.chat_input("Ask a question about the uploaded documents..."):
    
    welcome_placeholder.empty()

    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload and process documents first from the sidebar.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)

        chat_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke({
                    "input": user_query,
                    "chat_history": chat_history
                })
                answer = response["answer"]
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        st.rerun()