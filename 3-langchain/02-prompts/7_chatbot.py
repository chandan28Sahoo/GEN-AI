from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
)

chat_model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage(content='You are a helpful assistant')
]

print(f"Model: {llm.repo_id}")

while True:
    input_text = input("You: ")
    chat_history.append(HumanMessage(content=input_text))
    if input_text.lower() == 'exit':
        print("Thank you for using the chatbot. Goodbye!")
        break
    result = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")
    print(" ")

