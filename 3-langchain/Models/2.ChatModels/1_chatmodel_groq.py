from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import os
import dotenv

dotenv.load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY"), 
    max_completion_tokens=1000, 
    temperature=0.7
)

messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="Explain Python decorators with an example."),
]

response = llm.invoke(messages)

print(response.content)