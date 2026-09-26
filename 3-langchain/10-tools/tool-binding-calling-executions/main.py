from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """multiply two numbers"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """add two numbers"""
    return a + b

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # or your target Gemini model
    google_api_key=os.getenv("GOOGLE_AI_API_KEY"),
)

model_with_tool = llm.bind_tools([multiply, add])

query = HumanMessage('can you multiply 3 with 1000')
messages: list[BaseMessage] = [query]

# 1. Model decides to call the tool
result = model_with_tool.invoke(messages)
messages.append(result)

# 2. Extract tool call and execute
tool_call = result.tool_calls[0]
selected_tool = {"multiply": multiply, "add": add}[tool_call["name"]]
output = selected_tool.invoke(tool_call["args"])

# 3. Create an explicit ToolMessage referencing the ID
messages.append(ToolMessage(
    content=str(output),
    tool_call_id=tool_call["id"]
))

# 4. Invoke model again to get final answer
final_response = model_with_tool.invoke(messages)
print(final_response.content)