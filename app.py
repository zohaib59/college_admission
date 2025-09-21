import os
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# === Load environment variables ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Setup logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Guardrails ===
def is_safe_input(user_input):
    unsafe_keywords = [
        "kill", "bomb", "murder", "fight", "attack", "abuse", "rape", "weapon",
        "how to make", "harm", "violence", "terror", "explosive", "porn", "sex"
    ]
    return not any(keyword in user_input.lower() for keyword in unsafe_keywords)

# === PDF Processing ===
def process_pdf(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        return chunks
    except Exception as e:
        logger.error(f"PDF loading failed: {e}")
        return []

# === Embedding Setup ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Streamlit UI ===
st.set_page_config(page_title="College Info Chatbot", layout="centered")
st.title("🎓 College Info Assistant (Powered by GPT-3.5)")
st.markdown("Ask about **admissions, faculty, fees, alumni, working hours**, and more.")

# === Session State Initialization ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_active" not in st.session_state:
    st.session_state.session_active = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# === Sidebar Controls ===
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("📄 Upload College Prospectus (PDF)", type=["pdf"])
    if uploaded_file:
        with open("temp_prospectus.pdf", "wb") as f:
            f.write(uploaded_file.read())
        chunks = process_pdf("temp_prospectus.pdf")
        st.session_state.vectorstore = FAISS.from_documents(chunks, embedding_model)
        st.success("Prospectus uploaded and processed!")

    if not st.session_state.session_active:
        if st.button("🟢 Start Conversation"):
            st.session_state.session_active = True
            st.success("Conversation started.")
    else:
        if st.button("🔴 End Conversation"):
            st.session_state.session_active = False
            st.success("Conversation ended. API usage paused.")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    st.markdown("---")
    st.markdown("💡 Use 'Start Conversation' to begin and 'End Conversation' to pause API usage.")

# === Main Chat Logic ===
if st.session_state.session_active and st.session_state.vectorstore:
    user_input = st.text_input("📩 Enter your question:", "")
    if user_input:
        if not is_safe_input(user_input):
            st.error("🚫 Inappropriate query detected.")
        else:
            try:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key, streaming=True)
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

                with st.spinner("Thinking..."):
                    response = qa_chain.run(user_input)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                response = f"⚠️ Internal error: {str(e)}"

            st.markdown("### 📘 Response")
            st.write(response)

            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("Assistant", response))

elif st.session_state.session_active and not st.session_state.vectorstore:
    st.warning("📄 Please upload a college prospectus to begin.")

# === Display Chat History ===
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🧠 Chat History")
    for role, message in st.session_state.chat_history:
        st.markdown(f"**{role}:** {message}")
