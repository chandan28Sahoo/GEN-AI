# 02 - Text Splitters & Chunking Strategies

---

## 🌟 1. What is a Text Splitter in LangChain?

A **Text Splitter** is a component that breaks down large documents (such as 100-page PDFs, books, or source code files) into smaller, semantically meaningful passages called **chunks**.

```
    ┌──────────────────────────────────────────────────────────┐
    │                 Large 100-Page Document                  │
    └─────────────────────────────┬────────────────────────────┘
                                  │  (Text Splitter)
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  [ Chunk 1: ~500 chars ]   [ Chunk 2: ~500 chars ]   [ Chunk 3: ~500 chars ]
```

---

## ❓ 2. Why Do We Need Text Splitters in RAG?

### The Problems They Solve:
1. **LLM Context Window Limits**: Embedding a 50-page book as a single vector creates an "average" vector that loses all specific details. Splitting text ensures each vector represents a focused topic.
2. **Retrieval Precision**: When a user asks *"What is the return policy duration?"*, we want to retrieve the specific 2-sentence paragraph containing the answer, not an entire 40-page user manual.
3. **Preserving Boundary Context with `chunk_overlap`**: If a sentence happens to get cut in half across a chunk boundary, critical meaning is lost. Chunk overlap ensures adjacent chunks share overlapping text so context is never severed.

```
Chunk 1:  [ The return policy allows refunds within 30 days of purchase. ]
                                            └─── Overlap ───┐
Chunk 2:                                  [ 30 days of purchase. Items must be in original box. ]
```

---

## 🚀 3. How Do We Use Text Splitters?

### 📂 Files in this Submodule

| File | Splitter Class | Strategy | Best For |
| :--- | :--- | :--- | :--- |
| [1_length_based.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/02-text-splitters/1_length_based.py) | `CharacterTextSplitter` | Splits naively on a single character separator | Simple plain text |
| [2_text_structure_based.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/02-text-splitters/2_text_structure_based.py) | `RecursiveCharacterTextSplitter` | Recursively tries paragraph (`\n\n`), sentence (`\n`), space (` `), and characters | **Recommended for all general text** |
| [3_markdown_splitting.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/02-text-splitters/3_markdown_splitting.py) | `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)` | Splits along Markdown headers (`#`, `##`), lists, and tables | Markdown documentation |
| [4_python_code_splitting.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/02-text-splitters/4_python_code_splitting.py) | `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` | Splits along Python `class` and `def` boundaries | Source code repositories |
| [5_semantic_meaning_based.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/02-text-splitters/5_semantic_meaning_based.py) | `SemanticChunker` (Experimental) | Splits when embedding distance between consecutive sentences exceeds threshold | Dynamic topic shift splitting |

---

### Step-by-Step Code Walkthroughs

#### 1. `RecursiveCharacterTextSplitter` (The Production Standard)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars,
humanity continues to push the boundaries of what is possible.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements
in technology here on Earth. Satellite communications, GPS, and medical imaging trace their roots back to space innovations.
"""

# Initialize recursive splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # Target maximum character length per chunk
    chunk_overlap=30,    # Characters shared between adjacent chunks
    separators=["\n\n", "\n", " ", ""]  # Priority order of split points
)

chunks = text_splitter.split_text(text)

print(f"Total Chunks Created: {len(chunks)}")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---\n{chunk}")
```

---

#### 2. Code-Aware Splitting (`Language.PYTHON` & `Language.MARKDOWN`)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_details(self):
        return self.name

    def is_passing(self):
        return True
"""

# Keeps functions and classes intact rather than cutting in the middle of a line:
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=150,
    chunk_overlap=0
)

code_chunks = code_splitter.split_text(code)
print(f"Created {len(code_chunks)} code chunks.")
```

---

#### 3. Semantic Chunking (`SemanticChunker`)
Instead of arbitrary character length limits, `SemanticChunker` compares the cosine distance of sentence embeddings and only splits when a topic shift occurs:

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",  # 'percentile', 'standard_deviation', or 'interquartile'
    breakpoint_threshold_amount=3
)

sample_text = """
Farmers work hard in the fields during harvest season. The crops need constant sunlight and irrigation.
The Indian Premier League (IPL) is the largest cricket tournament globally with millions of viewers.
"""

docs = splitter.create_documents([sample_text])
print(f"Semantic Chunks Created: {len(docs)}")  # Automatically splits into Farming chunk and Cricket chunk!
```

---

## ⚙️ Hyperparameter Guide

| Parameter | Recommended Value | What it Controls |
| :--- | :--- | :--- |
| `chunk_size` | 500 – 1500 chars (~100 – 300 tokens) | Size of each chunk. Smaller = more focused vectors; Larger = broader context. |
| `chunk_overlap` | 10% – 20% of `chunk_size` (e.g. 50–100 chars) | Overlap between adjacent chunks to maintain sentence context across boundaries. |
| `separators` | `["\n\n", "\n", " ", ""]` | Hierarchy of splitting delimiters. |
