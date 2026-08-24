from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv

dotenv.load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_AI_API_KEY")
)

messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="Explain Python decorators with an example."),
]

response = llm.invoke(messages)

print(response.content)

