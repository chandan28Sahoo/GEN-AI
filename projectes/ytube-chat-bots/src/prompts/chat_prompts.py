from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# RAG prompt — used when answering questions from retrieved transcript chunks
# ---------------------------------------------------------------------------
RAG_SYSTEM = """You are a helpful assistant that answers questions about a YouTube video.
Use ONLY the following transcript context to answer.
If the answer is not in the context, say "I don't have enough information from this video to answer that."

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Condense question prompt — rewrites a follow-up question to be standalone
# ---------------------------------------------------------------------------
CONDENSE_SYSTEM = """Given the conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question. 
Output ONLY the rephrased question, nothing else."""

condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)
