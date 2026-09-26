from langchain_core.outputs import llm_result
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


from src.prompts.chat_prompts import rag_prompt, condense_prompt


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    """Build a conversational RAG chain with question condensing.

    Args:
        retriever: A LangChain retriever backed by the video's vector store.

    Returns:
        A runnable chain that accepts {'question', 'chat_history'} and returns a string.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Step 1: Condense follow-up questions into standalone ones
    condense_chain = condense_prompt | llm | StrOutputParser()

    def condense_question(inputs: dict) -> str:
        if inputs.get("chat_history"):
            return condense_chain.invoke(inputs)
        return inputs["question"]

    # Step 2: Retrieve → format → generate
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(condense_question)
        )
        | RunnablePassthrough.assign(
            context=lambda x: _format_docs(retriever.invoke(x["standalone_question"]))
        )
        | RunnablePassthrough.assign(
            answer=rag_prompt | llm | StrOutputParser()
        )
    )
    # rag_chain = (
    #     # Stage 1: {question, chat_history} -> add standalone_question
    #     {
    #         "question": lambda x: x["question"],
    #         "chat_history": lambda x: x["chat_history"],
    #         "standalone_question": RunnableLambda(condense_question),
    #     }
    #     # Stage 2: -> add context
    #     | {
    #         "question": lambda x: x["question"],
    #         "chat_history": lambda x: x["chat_history"],
    #         "standalone_question": lambda x: x["standalone_question"],
    #         "context": lambda x: _format_docs(retriever.invoke(x["standalone_question"])),
    #     }
    #     # Stage 3: -> add answer
    #     | {
    #         "question": lambda x: x["question"],
    #         "chat_history": lambda x: x["chat_history"],
    #         "standalone_question": lambda x: x["standalone_question"],
    #         "context": lambda x: x["context"],
    #         "answer": rag_prompt | llm | StrOutputParser(),
    #     }
    # )

    return rag_chain
