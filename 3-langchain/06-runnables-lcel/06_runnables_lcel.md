# 06 - LangChain Expression Language (LCEL) & Core Runnables

---

## 🌟 1. What is LCEL and the Runnable Protocol?

**LangChain Expression Language (LCEL)** is the declarative framework at the core of modern LangChain.

In LCEL, every building block (prompts, models, parsers, custom functions, retrievers) implements the standard **Runnable Interface**. This interface guarantees that every component supports:
- **Synchronous Execution**: `.invoke(input)`
- **Asynchronous Execution**: `.ainvoke(input)`
- **Streaming Output Tokens**: `.stream(input)`
- **Batch Processing**: `.batch([input1, input2])`

```
   ┌────────────────────────────────────────────────────────┐
   │                    Runnable Protocol                   │
   │                                                        │
   │   .invoke()   ──► Synchronous single call              │
   │   .ainvoke()  ──► Async non-blocking call              │
   │   .stream()   ──► Token-by-token streaming             │
   │   .batch()    ──► Concurrent batch execution           │
   └────────────────────────────────────────────────────────┘
```

---

## ❓ 2. Why Do We Need LCEL Core Runnables?

### The Problems They Solve:
1. **Piping Python Functions**: Native Python operators like `|` only work on objects designed for it. `RunnableLambda` allows you to inject any regular Python function directly into an LCEL chain.
2. **Passing Through Inputs**: Often downstream prompts need both the output of an intermediate step *and* the original input query. `RunnablePassthrough` solves this by forwarding inputs without mutation.
3. **Concurrent Execution**: `RunnableParallel` executes tasks concurrently in a background thread pool without writing manual `asyncio` or `threading` boilerplate.

---

## 🚀 3. How Do We Use the 5 Core LCEL Runnables?

### 📂 Files in this Module

| File | Core Runnable | Mental Model | Purpose |
| :--- | :--- | :--- | :--- |
| [1_runnable_sequence.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/06-runnables-lcel/1_runnable_sequence.py) | `RunnableSequence` | **Assembly Line** | Passes output of Step $A$ as input to Step $B$. Same as `A \| B`. |
| [2_runnable_parallel.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/06-runnables-lcel/2_runnable_parallel.py) | `RunnableParallel` | **Fork in Road** | Executes multiple runnables concurrently on the same input dictionary. |
| [3_runnable_passthrough.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/06-runnables-lcel/3_runnable_passthrough.py) | `RunnablePassthrough` | **Mirror / Pass-through** | Forwards original input unchanged alongside new computed fields. |
| [4_runnable_lambda.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/06-runnables-lcel/4_runnable_lambda.py) | `RunnableLambda` | **Custom Plugin** | Wraps any Python function (e.g. word count, regex) to run within LCEL. |
| [5_runnable_branch.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/06-runnables-lcel/5_runnable_branch.py) | `RunnableBranch` | **Switch / Router** | Conditional branching logic based on boolean predicates. |

---

### Step-by-Step Code Examples

#### 1. `RunnableSequence` (`1_runnable_sequence.py`)
```python
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

prompt1 = PromptTemplate.from_template("Write a one-liner joke about {topic}")
prompt2 = PromptTemplate.from_template("Explain why this joke is funny: {text}")

# Explicit construction (equivalent to prompt1 | model | parser | prompt2 | model | parser):
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({"topic": "Quantum Physics"})
print(result)
```

---

#### 2. `RunnableParallel` (`2_runnable_parallel.py`)
```python
from langchain_core.runnables import RunnableParallel

tweet_prompt = PromptTemplate.from_template("Write a punchy tweet about {topic}")
linkedin_prompt = PromptTemplate.from_template("Write a professional LinkedIn post about {topic}")

social_media_chain = RunnableParallel({
    "tweet": tweet_prompt | model | parser,
    "linkedin": linkedin_prompt | model | parser
})

# Both prompts run concurrently:
output = social_media_chain.invoke({"topic": "Generative AI in 2026"})
print("--- TWEET ---")
print(output["tweet"])
print("\n--- LINKEDIN ---")
print(output["linkedin"])
```

---

#### 3. `RunnablePassthrough` (`3_runnable_passthrough.py`)
```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

joke_chain = PromptTemplate.from_template("Tell a joke about {topic}") | model | parser
explain_prompt = PromptTemplate.from_template("Explain why this joke is funny: {joke}")

# Keep the original joke AND compute the explanation:
chain = joke_chain | RunnableParallel({
    "joke": RunnablePassthrough(),  # Forwards the joke string unchanged
    "explanation": explain_prompt | model | parser
})

result = chain.invoke({"topic": "Software Engineers"})
print(f"Joke: {result['joke']}")
print(f"Explanation: {result['explanation']}")
```

---

#### 4. `RunnableLambda` (`4_runnable_lambda.py`)
```python
from langchain_core.runnables import RunnableLambda

def calculate_word_count(text: str) -> int:
    return len(text.split())

def to_uppercase(text: str) -> str:
    return text.upper()

chain = (
    PromptTemplate.from_template("Write a short poem about {topic}")
    | model
    | parser
    | RunnableParallel({
        "poem": RunnablePassthrough(),
        "word_count": RunnableLambda(calculate_word_count),
        "screaming_poem": RunnableLambda(to_uppercase)
    })
)

res = chain.invoke({"topic": "Sunrise"})
print("Poem:", res["poem"])
print("Total Words:", res["word_count"])
```

---

#### 5. `RunnableBranch` (`5_runnable_branch.py`)
```python
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

generate_report = PromptTemplate.from_template("Write a report about {topic}") | model | parser
summarize = PromptTemplate.from_template("Summarize this long report:\n{text}") | model | parser

# If the generated report has > 100 words, summarize it; otherwise pass through as is:
chain = generate_report | RunnableBranch(
    (lambda text: len(text.split()) > 100, summarize),
    RunnablePassthrough()
)

print(chain.invoke({"topic": "The history of the Internet"}))
```

---

## ⚡ LCEL Cheatsheet Summary

```python
# Pipe Operator:
chain = component_a | component_b

# Stream output tokens in real time:
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Batch process multiple inputs concurrently:
results = chain.batch([{"topic": "Physics"}, {"topic": "Math"}, {"topic": "Biology"}])
```
