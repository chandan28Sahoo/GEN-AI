"""
Streamlit UI entry point.

Run with:
    streamlit run app.py
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.tools.youtube_loader import fetch_transcript
from src.utils.vector_store import build_vector_store
from src.chains.rag_chain import build_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎬", layout="centered")
st.title("🎬 YouTube Chatbot")
st.caption("Chat with any YouTube video using AI")

# ── Session state ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Sidebar — video input ────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔗 Video URL")
    url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Load Video", use_container_width=True) and url:
        with st.spinner("Fetching transcript and building index..."):
            transcript = fetch_transcript(url)
            vectorstore = build_vector_store(transcript)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            st.session_state.chain = build_rag_chain(retriever)
            st.session_state.chat_history = []
        st.success("✅ Video loaded! Start chatting.")

# ── Chat messages ─────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask something about the video..."):
    if st.session_state.chain is None:
        st.warning("Please load a YouTube video first.")
    else:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke(
                {"question": question, "chat_history": st.session_state.chat_history}
            )
            answer = result["answer"]
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.extend(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )
