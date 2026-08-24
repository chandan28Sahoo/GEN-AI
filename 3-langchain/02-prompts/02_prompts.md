# 02 - Prompt Engineering, Templates & Chatbots

---

## 🌟 1. What are Prompt Templates & Messages in LangChain?

A **Prompt Template** is a structured, reproducible recipe for generating dynamic prompts for AI models. Instead of hardcoding raw strings with Python f-strings, prompt templates allow you to define parameterized variables, validate inputs, serialize prompts to disk, and inject dynamic conversation history.

### The Message Roles
In modern chat applications, prompts are composed of message objects:
- **`SystemMessage`**: Sets the behavior, tone, rules, and personality of the AI.
- **`HumanMessage`**: Represents user instructions or questions.
- **`AIMessage`**: Represents previous responses from the assistant (used for conversation history).
- **`MessagesPlaceholder`**: A dynamic placeholder slot where past conversation turns are inserted at runtime.

```
┌────────────────────────────────────────────────────────┐
│                   ChatPromptTemplate                   │
├────────────────────────────────────────────────────────┤
│  System: "You are a customer support agent."           │
│  MessagesPlaceholder(variable_name="chat_history")     │
│  Human: "{query}"                                      │
└───────────────────────────┬────────────────────────────┘
                            │ (Injected at runtime)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   Rendered Messages                    │
├────────────────────────────────────────────────────────┤
│  [SystemMessage("You are a customer support agent.")]  │
│  [HumanMessage("My order is #4321")]                   │  ◄── From chat_history
│  [AIMessage("I see your order for shoes.")]            │  ◄── From chat_history
│  [HumanMessage("Where is my refund?")]                 │  ◄── From {query}
└────────────────────────────────────────────────────────┘
```

---

## ❓ 2. Why Do We Need Them & When Are They Used?

### The Problems They Solve:
1. **Preventing Prompt Injection & Formatting Bugs**: Parameterized templates validate that all required variables exist and escape malicious formatting characters safely.
2. **Dynamic Chat History**: In multi-turn chatbots, the conversation grows on every turn. `MessagesPlaceholder` lets you pass an arbitrary list of past messages without manually rewriting string templates.
3. **Reusability & Version Control**: Templates can be saved to `.json` or `.yaml` files, shared across teams, and versioned in Git independently of Python logic.
4. **Streamlit / UI Integration**: Seamlessly connect dropdowns, sliders, and text inputs to parameterized prompt templates.

---

## 🚀 3. How Do We Use Them?

### 📂 Files in this Module

| File | Topic | What it Demonstrates |
| :--- | :--- | :--- |
| [1_prompt_ui.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/02-prompts/1_prompt_ui.py) | **Streamlit Prompt UI** | Interactive Web UI with dropdowns feeding into a parameterized `PromptTemplate`. |
| [2_prompt_generator.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/02-prompts/2_prompt_generator.py) | **Prompt Serialization** | Creating a validated `PromptTemplate` and saving it to `template.json`. |
| [3_prompt_ui_loaded.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/02-prompts/3_prompt_ui_loaded.py) | **Loading Prompts from File** | Loading a serialized `template.json` with `load_prompt()` and running it in Streamlit. |
| [4_chat_prompt_template.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/02-prompts/4_chat_prompt_template.py) | **ChatPromptTemplate** | Multi-role template structure using `('system', '...')` and `('human', '...')` tuples. |
| [5_messages.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/02-prompts/5_messages.py) | **Message Objects** | Working with explicit `SystemMessage`, `HumanMessage`, and `AIMessage` objects. |
| [6_message_placeholder.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/02-prompts/6_message_placeholder.py) | **MessagesPlaceholder** | Dynamically slotting past conversation history into a template at runtime. |
| [7_chatbot.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/02-prompts/7_chatbot.py) | **Terminal Chatbot** | Interactive multi-turn CLI chatbot that maintains conversation context. |

---

### Step-by-Step Code Walkthroughs

#### 1. `PromptTemplate` (Single String Templates)
```python
from langchain_core.prompts import PromptTemplate

# Define template with placeholders
template = PromptTemplate(
    template="Explain the research paper titled '{paper_name}' in a {style} style with {length} length.",
    input_variables=["paper_name", "style", "length"],
    validate_template=True
)

# Format/Invoke template
prompt_value = template.invoke({
    "paper_name": "Attention Is All You Need",
    "style": "Beginner-Friendly",
    "length": "Short (2 paragraphs)"
})

print(prompt_value.to_string())
```

---

#### 2. `ChatPromptTemplate` (Role-Based Templates)
```python
from langchain_core.prompts import ChatPromptTemplate

# Define template using role tuples
chat_template = ChatPromptTemplate([
    ('system', 'You are an expert {domain} mentor with 10+ years of experience.'),
    ('human', 'Can you explain {topic} in simple terms?')
])

prompt = chat_template.invoke({
    'domain': 'Artificial Intelligence',
    'topic': 'Transformers'
})

print(prompt.to_messages())
# Output: [SystemMessage(content='You are an expert Artificial Intelligence mentor...'), HumanMessage(content='Can you explain Transformers...')]
```

---

#### 3. `MessagesPlaceholder` & Building a Memory Chatbot
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Define template with a placeholder for dynamic history
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent for Acme Store.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{user_query}')
])

# 2. Maintain conversational state across turns
history = []
model = ChatGroq(model="llama-3.3-70b-versatile")
chain = chat_template | model

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Invoke chain with current history and latest query
    response = chain.invoke({
        "chat_history": history,
        "user_query": user_input
    })
    
    # Append turns to history for the next round
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))
    
    print(f"AI: {response.content}\n")
```

---

## 🚀 Running the Interactive Streamlit Apps

```bash
# 1. Run the interactive research paper summarizer UI
streamlit run 3-langchain/02-prompts/1_prompt_ui.py

# 2. Run the serialized template UI (loads from template.json)
streamlit run 3-langchain/02-prompts/3_prompt_ui_loaded.py
```

---

## ⚙️ Summary Comparison

| Concept | Best For | Typical Format |
| :--- | :--- | :--- |
| `PromptTemplate` | Text completion, simple one-shot prompts | `"Explain {concept} in {language}."` |
| `ChatPromptTemplate` | Conversational models with system/human roles | `[('system', '...'), ('human', '...')]` |
| `MessagesPlaceholder` | Chatbots, agents with dynamic memory | `MessagesPlaceholder(variable_name='history')` |
| `load_prompt()` | Decoupling prompt design from Python code | `load_prompt("path/to/template.json")` |
