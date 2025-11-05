import os
import streamlit as st
from dotenv import load_dotenv
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM

# Environment Setup 
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Universal RAG Chatbot")
st.caption("Upload PDFs or text files → Build a knowledge base → Chat naturally with AI powered by Mistral 7B Instruct v0.2")

# Custom Hugging Face LLM Wrapper
class HFInferenceLLM(LLM):
    """LangChain-compatible wrapper for Mistral 7B using conversational endpoint."""
    model_id: str
    token: str
    client: Any = None

    def __init__(self, model_id: str, token: str):
        super().__init__(model_id=model_id, token=token)
        object.__setattr__(
            self,
            "client",
            InferenceClient(
                token=token,
                base_url="https://router.huggingface.co/hf-inference"
            )
        )

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        """Use chat_completion for models that support conversational tasks."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(
                model=self.model_id,  
                messages=messages,
                temperature=0.5,
                max_tokens=512
            )

            if hasattr(response, "generated_text"):
                return response.generated_text
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "")
                if "generated_text" in response:
                    return response["generated_text"]
            if isinstance(response, (list, tuple)) and response:
                first = response[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"]
                return str(first)
            return str(response)
        except Exception as e:
            raise RuntimeError(f"Chat completion failed: {e}")

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

# File Upload Section 
uploaded_files = st.file_uploader(
    "📂 Upload one or more files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("temp_docs", exist_ok=True)
    docs = []

    for file in uploaded_files:
        path = os.path.join("temp_docs", file.name)
        with open(path, "wb") as f:
            f.write(file.read())

        loader = PyPDFLoader(path) if file.name.lower().endswith(".pdf") else TextLoader(path)
        docs.extend(loader.load())

    # Text Splitting 
    st.info("🔄 Splitting documents and generating embeddings...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Embeddings + Vector Store 
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedder)
    st.success("✅ Knowledge base built successfully!")

    # Initialize Mistral 7B LLM 
    llm = HFInferenceLLM("mistralai/Mistral-7B-Instruct-v0.2", HF_TOKEN)

    # Chat Section
    st.subheader("💬 Chat with your documents")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Optional: Clear chat history
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Render previous messages
    for role, content in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    # Chat input
    if query := st.chat_input("Ask anything about your uploaded documents..."):
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("🔎 Searching your knowledge base..."):
            results = db.similarity_search(query, k=3)
            doc_context = "\n\n".join([doc.page_content for doc in results])

        conversation_context = "\n".join(
            [f"{'User' if r == 'user' else 'Assistant'}: {m}" for r, m in st.session_state.chat_history]
        )

        prompt = f"""
You are a helpful AI assistant that answers based on user-uploaded documents.

Conversation so far:
{conversation_context}

Relevant document context:
{doc_context}

Now answer the user's latest question in a natural, conversational tone.
"""

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer = llm.invoke(prompt)
                st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))

else:
    st.info("⬆️ Please upload one or more files to start chatting.")
