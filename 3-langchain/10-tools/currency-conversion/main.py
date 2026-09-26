
import requests, os
from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # or your target Gemini model
    google_api_key=os.getenv("GOOGLE_AI_API_KEY"),
)

#tool create
@tool
def get_currency_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Fetch the conversion rate between a base currency and a target currency.
    Uses frankfurter.app — free, no API key required.
    Example: base_currency='USD', target_currency='INR'
    """
    url = f"https://api.frankfurter.app/latest?from={base_currency}&to={target_currency}"
    data = requests.get(url).json()
    return data["rates"][target_currency]

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate


print(get_currency_conversion_factor.invoke({'base_currency': 'USD', 'target_currency': 'INR'}))
print(convert.invoke({'base_currency_value': 100, 'conversion_rate': get_currency_conversion_factor.invoke({'base_currency': 'USD', 'target_currency': 'INR'})}))

#tool buinding
llm_with_tools = llm.bind_tools([get_currency_conversion_factor, convert])

messages: list[BaseMessage] = [HumanMessage("what is the conversion of USD to INR and based on that can you convert 10 usd to inr")]

ai_message: AIMessage = llm_with_tools.invoke(messages)
print("Step 1 tool calls:", ai_message.tool_calls)
# → [{'name': 'get_currency_conversion_factor', ...}]  (only 1, by design)
# The LLM can't call `convert` yet — it doesn't know the rate until step 1 finishes.

# --- Agentic loop: keep executing tools until the LLM stops calling them ---
tool_registry = {
    "get_currency_conversion_factor": get_currency_conversion_factor,
    "convert": convert,
}

# Track the injected value so we can supply it to `convert`
conversion_rate: Optional[float] = None

while ai_message.tool_calls:
    messages.append(ai_message)   # add AIMessage to history

    for tc in ai_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        if tool_name == "convert":
            # InjectedToolArg: the LLM never supplies conversion_rate.
            # We inject it from the result of the previous tool call.
            tool_args = {**tool_args, "conversion_rate": conversion_rate}

        result = tool_registry[tool_name].invoke(tool_args)
        print(f"  Tool '{tool_name}' → {result}")

        # Remember the rate so we can inject it into `convert`
        if tool_name == "get_currency_conversion_factor":
            conversion_rate = result

        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    ai_message = llm_with_tools.invoke(messages)
    print("Next tool calls:", ai_message.tool_calls)

print("\nFinal answer:", ai_message.content)