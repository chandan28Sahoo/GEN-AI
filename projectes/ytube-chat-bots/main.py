"""
CLI entry point — chat with a YouTube video from the terminal.

Usage:
    python main.py --url "https://www.youtube.com/watch?v=J5_-l7WIO_w"
"""
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from src.tools.youtube_loader import fetch_transcript, extract_video_id
from src.utils.vector_store import get_or_build_vector_store
from src.chains.rag_chain import build_rag_chain
from src.utils.vector_store import video_id_exists


def main():
    parser = argparse.ArgumentParser(description="YouTube Chatbot CLI")
    parser.add_argument("--url", required=False, help="YouTube video URL")
    args = parser.parse_args()
    url: str = args.url or os.getenv("URL") or ""
    if not url:
        parser.error("Provide --url or set the URL environment variable.")

    video_id = extract_video_id(url)
    print(f"\n🎬 Video ID: {video_id}")

    # Only fetch transcript if the video hasn't been indexed yet
    
    if video_id_exists(video_id):
        transcript = ""  # not needed — vector store already built
    else:
        print(f"📥 Fetching transcript for: {url}")
        transcript = fetch_transcript(url)

    vectorstore = get_or_build_vector_store(video_id, transcript)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ Vector store ready.")

    chain = build_rag_chain(retriever)
    chat_history = []

    print("\n✅ Ready! Type your question (or 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        result = chain.invoke({"question": question, "chat_history": chat_history})
        answer = result["answer"]
        print(f"\nBot: {answer}\n")
        chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage
    main()
