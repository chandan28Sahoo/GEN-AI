# 05 - Chains: Simple, Sequential, Parallel & Conditional

---

## 🌟 1. What is a Chain in LangChain?

A **Chain** is a sequence of computational steps piped together to accomplish a multi-step task.

In modern LangChain, chains are built using the pipe operator (`|`), part of the **LangChain Expression Language (LCEL)**. A chain can link together:
- Prompt Templates
- Models
- Output Parsers
- Custom Python Functions
- Other Sub-Chains

```
   ┌────────────────────┐     ┌───────────────┐     ┌──────────────────┐
   │   PromptTemplate   │ ──► │  Chat Model   │ ──► │  StrOutputParser │
   └────────────────────┘     └───────────────┘     └──────────────────┘
```

---

## ❓ 2. Why Do We Need Chains & When Are They Used?

### The Problems They Solve:
1. **Multi-Stage Tasks**: Real applications rarely consist of a single LLM call. For example, you may want to generate a 10-page report (Step 1), summarize it into 5 bullet points (Step 2), and translate it into Spanish (Step 3).
2. **Parallel Performance**: If you need to generate blog post ideas, a quiz, and a tweet concurrently from the same article, running them in parallel reduces latency by 70%+.
3. **Dynamic Routing**: Route customer feedback dynamically to different sub-chains depending on whether it is a compliment, a technical bug, or a billing complaint.

---

## 🚀 3. How Do We Use Chains? (4 Core Architectural Patterns)

### 📂 Files in this Module

| File | Architecture Pattern | Key LCEL Components Used |
| :--- | :--- | :--- |
| [1_simple_chain.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/05-chains/1_simple_chain.py) | **Simple Linear Chain** | `PromptTemplate \| Model \| StrOutputParser` |
| [2_sequential_chain.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/05-chains/2_sequential_chain.py) | **Sequential Multi-step Chain** | Multi-stage pipeline: Step 1 output $\rightarrow$ Step 2 input |
| [3_parallel_chain.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/05-chains/3_parallel_chain.py) | **Parallel Execution Chain** | `RunnableParallel` concurrent execution + Merge step |
| [4_conditional_chain.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/05-chains/4_conditional_chain.py) | **Conditional Branching Chain** | Sentiment Classifier + `RunnableBranch` dynamic routing |

---

### Pattern 1: Simple Linear Chain (`1_simple_chain.py`)
```
Input ({topic}) ──► [ PromptTemplate ] ──► [ ChatModel ] ──► [ StrOutputParser ] ──► Final String
```

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = PromptTemplate.from_template("Generate 5 interesting facts about {topic}")
model = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"))
parser = StrOutputParser()

# Connect using the pipe operator:
chain = template | model | parser

result = chain.invoke({"topic": "Black Holes"})
print(result)
```

---

### Pattern 2: Sequential Multi-Step Chain (`2_sequential_chain.py`)
```
Input ({topic})
      │
      ▼
[ Prompt 1 ] ──► [ Model ] ──► [ Detailed Report ]
                                       │
      ┌────────────────────────────────┘
      ▼
[ Prompt 2 ] ──► [ Model ] ──► [ StrOutputParser ] ──► 5-Point Summary
```

```python
prompt1 = PromptTemplate.from_template("Generate a detailed report on {topic}")
prompt2 = PromptTemplate.from_template("Generate a 5-point summary from the following text:\n{text}")

# The output of step 1 flows into prompt2's 'text' variable:
chain = prompt1 | model | prompt2 | model | parser

summary = chain.invoke({"topic": "Renewable Energy"})
print(summary)
```

---

### Pattern 3: Parallel Execution Chain (`3_parallel_chain.py`)
```
                     ┌──► [ Notes Prompt ] ──► [ Model 1 ] ──► [ Notes ] ──┐
                     │                                                     │
Input Text ({text}) ─┤                                                     ├──► [ Merge Prompt ] ──► [ Model ] ──► Final Document
                     │                                                     │
                     └──► [ Quiz Prompt ]  ──► [ Model 2 ] ──► [ Quiz ]  ──┘
```

```python
from langchain_core.runnables import RunnableParallel

notes_prompt = PromptTemplate.from_template("Generate short study notes from this text:\n{text}")
quiz_prompt = PromptTemplate.from_template("Generate a 5-question multiple choice quiz from this text:\n{text}")
merge_prompt = PromptTemplate.from_template("Merge the study notes and quiz into a study guide:\nNotes: {notes}\n\nQuiz: {quiz}")

# 1. Define parallel branch:
parallel_branch = RunnableParallel({
    "notes": notes_prompt | model | parser,
    "quiz": quiz_prompt | model | parser
})

# 2. Connect parallel branch to merge step:
full_chain = parallel_branch | merge_prompt | model | parser

result = full_chain.invoke({"text": "Support Vector Machines (SVMs) are supervised learning algorithms..."})
print(result)
```

---

### Pattern 4: Conditional Routing Chain (`4_conditional_chain.py`)
```
                                               ┌──► (If Positive) ──► [ Thank You Generator ]
                                               │
Input Feedback ──► [ Sentiment Classifier ] ───┼──► (If Negative) ──► [ Apology & Support Generator ]
                                               │
                                               └──► (Fallback)    ──► [ Default Handler ]
```

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

# Step 1: Classifier Model
class FeedbackSentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment classification")

classifier = model.with_structured_output(FeedbackSentiment)

# Step 2: Branch Handlers
positive_prompt = PromptTemplate.from_template("Write a delighted thank you message for this review: {feedback}")
negative_prompt = PromptTemplate.from_template("Write a sincere apology and resolution for this complaint: {feedback}")

# Step 3: Conditional Branching
branch = RunnableBranch(
    (lambda x: x.sentiment == "positive", positive_prompt | model | parser),
    (lambda x: x.sentiment == "negative", negative_prompt | model | parser),
    RunnableLambda(lambda x: "Could not classify sentiment.")
)

chain = classifier | branch
response = chain.invoke("I received my order on time and the product quality is outstanding!")
print(response)
```

---

## 🛠️ Graph Visualization

You can visualize the architecture graph of any chain directly in your console:

```python
chain.get_graph().print_ascii()
```
