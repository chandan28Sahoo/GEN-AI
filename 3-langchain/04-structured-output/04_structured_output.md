# 04 - Structured Output & Schema Enforcement

---

## 🌟 1. What is Native Structured Output in LangChain?

**Structured Output** is a first-class feature in LangChain that binds a target schema (Pydantic model, Python `TypedDict`, or JSON Schema) directly to a Chat Model using **`.with_structured_output(schema)`**.

Instead of writing custom formatting instructions into your prompt and hoping the LLM follows the rules, modern chat models use **under-the-hood Tool/Function Calling** to guarantee that responses strictly match the requested JSON schema.

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                     Target Schema                      │
                                  │  class Review(BaseModel):                              │
                                  │      sentiment: Literal["pos", "neg"]                  │
                                  │      rating: int = Field(ge=1, le=5)                   │
                                  │      summary: str                                      │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                                            model.with_structured_output(Review)
                                                              │
   Unstructured Product Review  ──────────────────────────────┴──────────────────────────────►  Review(sentiment="pos", rating=5, summary="...")
```

---

## ❓ 2. Why Do We Need It vs. Traditional Output Parsers?

### The Problems with Prompt-Based Output Parsing:
1. **Formatting Failures**: Small or fast models frequently forget formatting rules, wrap JSON in markdown fences (` ```json `), or include introductory chatter (`"Sure, here is your JSON:"`).
2. **Brittle Regex**: Parsing markdown text with regex frequently breaks in edge cases.

### Why `.with_structured_output()` is Better:
1. **Guaranteed Schema Adherence**: The LLM's decoding process is constrained directly by the model provider's API (e.g. OpenAI Structured Outputs, Gemini Function Calling).
2. **Clean Code**: You do not need to inject `parser.get_format_instructions()` into your prompt manually.
3. **Direct Python Objects**: Calling `structured_model.invoke("text")` returns an instantiated Pydantic object or typed dictionary immediately.

---

## 🚀 3. How Do We Use It?

### 📂 Files in this Module

| File | Topic | Description |
| :--- | :--- | :--- |
| [1_pydantic_demo.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/04-structured-output/1_pydantic_demo.py) | **Pydantic Fundamentals** | Python Pydantic intro: `BaseModel`, `Field`, validation constraints, and `.model_dump_json()`. |
| [2_typeddict_demo.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/04-structured-output/2_typeddict_demo.py) | **TypedDict Fundamentals** | Python `TypedDict` syntax for typed dictionary schemas. |
| [3_with_structured_output_typeddict.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/04-structured-output/3_with_structured_output_typeddict.py) | **TypedDict Extraction** | Extracting structured product reviews using `TypedDict` and `Annotated` descriptions. |
| [4_with_structured_output_pydantic_hf.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/04-structured-output/4_with_structured_output_pydantic_hf.py) | **Pydantic Model Extraction** | Complete extraction pipeline using strict Pydantic models. |
| [5_with_structured_output_json_hf.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/04-structured-output/5_with_structured_output_json_hf.py) | **Raw JSON Schema** | Structured extraction using raw JSON Schema dictionaries. |
| [json_schema.json](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/04-structured-output/json_schema.json) | **JSON Schema Reference** | Raw JSON schema file defining review structure. |

---

### Step-by-Step Code Walkthroughs

#### Method 1: Using Pydantic (`1_pydantic_demo.py` & `4_with_structured_output_pydantic_hf.py`)
```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_openai import ChatOpenAI

# 1. Define the desired output data schema
class ReviewAnalysis(BaseModel):
    key_themes: list[str] = Field(description="Key themes discussed in the review")
    summary: str = Field(description="A concise 2-sentence summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment")
    pros: Optional[list[str]] = Field(default=None, description="List of pros mentioned")
    cons: Optional[list[str]] = Field(default=None, description="List of cons mentioned")
    reviewer_name: Optional[str] = Field(default=None, description="Name of the reviewer if found")

# 2. Bind schema to the chat model
model = ChatOpenAI(model="gpt-4o")
structured_llm = model.with_structured_output(ReviewAnalysis)

# 3. Invoke with unstructured text
review_text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse!
The Snapdragon 8 Gen 3 makes everything lightning fast and the 200MP camera is stunning.
However, it is heavy in the hand and $1300 is very expensive.
Review by Chandan Sahoo
"""

result = structured_llm.invoke(review_text)

# Result is directly a ReviewAnalysis object!
print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Pros:", result.pros)
print("Cons:", result.cons)
print("Reviewer:", result.reviewer_name)
```

---

#### Method 2: Using `TypedDict` (`3_with_structured_output_typeddict.py`)
If you prefer standard Python dictionaries without Pydantic dependencies:

```python
from typing import TypedDict, Annotated, Literal, Optional
from langchain_openai import ChatOpenAI

class ReviewOutput(TypedDict):
    key_themes: Annotated[list[str], "List of key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg", "neutral"], "Sentiment of the review"]
    pros: Annotated[Optional[list[str]], "List of all pros"]
    cons: Annotated[Optional[list[str]], "List of all cons"]
    name: Annotated[Optional[str], "Name of the reviewer"]

model = ChatOpenAI(model="gpt-4o")
structured_llm = model.with_structured_output(ReviewOutput)

result = structured_llm.invoke(review_text)

# Result is a Python dictionary:
print("Summary:", result["summary"])
print("Sentiment:", result["sentiment"])
```

---

## ⚖️ Pydantic vs TypedDict vs JSON Schema

| Feature | Pydantic (`BaseModel`) | TypedDict | Raw JSON Schema |
| :--- | :--- | :--- | :--- |
| **Validation Engine** | Rust-powered Pydantic v2 runtime | Static type hints only | JSON Schema validator |
| **Data Constraints** | `gt=0`, `min_length=1`, Regex, Custom validators | Basic types only | Standard JSON schema properties |
| **Return Type** | Python Object (`res.field`) | Python Dictionary (`res["field"]`) | Python Dictionary (`res["field"]`) |
| **Best For** | **Python Web APIs (FastAPI), strict production data** | Lightweight pipelines | Polyglot / Multi-language systems |
