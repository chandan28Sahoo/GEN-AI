# 03 - Embedding Models & Vector Similarity

---

## 🌟 1. What is an Embedding Model?

An **Embedding Model** is a specialized neural network trained to translate human language (words, sentences, or entire documents) into a **dense vector of floating-point numbers** (e.g. 384, 1536, or 3072 dimensions).

### The Intuition: Language as Geometry
In vector space, words and sentences with **similar meanings** are positioned close together, while unrelated sentences are far apart.

```
"Virat Kohli plays cricket for India"  ──►  [ 0.12, -0.45,  0.89, ... ]  ──┐
                                                                           ├─► High Cosine Similarity (~0.92)
"Sachin Tendulkar is an Indian batsman" ──►  [ 0.14, -0.42,  0.87, ... ]  ──┘

"How to bake a chocolate cake at home" ──►  [-0.88,  0.11, -0.34, ... ]  ──── Low Cosine Similarity (~0.10)
```

---

## ❓ 2. Why Do We Need It & When Is It Used?

### The Problem It Solves
Traditional keyword search (like `grep` or `SQL LIKE "%keyword%"`) fails when users use synonyms or phrasing that doesn't match the exact words in the database:
- **Keyword Search**: Searching for *"automobile repair"* will **fail** to find documents containing *"car maintenance"*.
- **Semantic Vector Search**: Both phrases produce nearly identical embedding vectors, allowing the search engine to retrieve the document instantly.

### Core Applications
1. **Semantic Search & RAG**: Finding relevant passages to feed into LLM prompts.
2. **Recommendation Systems**: Finding similar articles, products, or videos.
3. **Clustering & Classification**: Grouping customer support tickets by topic.
4. **Duplicate Detection**: Identifying rephrased questions in forums.

---

## 🚀 3. How Do We Use It in LangChain?

### 📂 Files in this Submodule

| File | Description | Technique Used |
| :--- | :--- | :--- |
| [1_embedding_openai_query.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/03-embedding-models/1_embedding_openai_query.py) | Embed a single query string | `embedding.embed_query(text)` |
| [2_embedding_openai_docs.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/03-embedding-models/2_embedding_openai_docs.py) | Embed a batch of documents | `embedding.embed_documents(doc_list)` |
| [3_document_similarity_openai.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/03-embedding-models/3_document_similarity_openai.py) | Semantic search with OpenAI | OpenAI Embeddings + Scikit-Learn `cosine_similarity` |
| [4_embedding_hf.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/03-embedding-models/4_embedding_hf.py) | Free local embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| [5_document_similarity_hf.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/03-embedding-models/5_document_similarity_hf.py) | Full semantic search with local HF model | Free local embeddings + NumPy `argmax` |

---

### Step-by-Step Walkthrough: Building a Semantic Search Engine

```python
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the local embedding model (no API key required!)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Define your knowledge base documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Step 3: Define a user query (notice the query does not contain the word 'cricketer')
query = "tell me about bumrah"

# Step 4: Convert documents and query into numerical vector arrays
doc_embeddings = embeddings.embed_documents(documents)  # 2D list of shape (5, 384)
query_embedding = embeddings.embed_query(query)        # 1D list of shape (384,)

# Step 5: Compute Cosine Similarity between query vector and all doc vectors
# Cosine similarity ranges from -1.0 (opposite) to 1.0 (identical)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Step 6: Find the index with the highest similarity score
best_index = np.argmax(scores)
highest_score = scores[best_index]

print("Query:", query)
print(f"Most Relevant Document (Index {best_index}):", documents[best_index])
print(f"Similarity Score: {highest_score:.4f}")
```

---

## 🔑 `embed_query()` vs `embed_documents()`

| Method | Input | Output | When to Use |
| :--- | :--- | :--- | :--- |
| **`embed_query(text)`** | `str` (single string) | `list[float]` (1D vector) | Converting the search query entered by the user. |
| **`embed_documents(list_of_texts)`** | `list[str]` (batch) | `list[list[float]]` (2D matrix) | Converting documents or document chunks during database indexing. |

---

## ⚙️ Popular Embedding Models Comparison

| Model | Provider | Dimensions | Pricing | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `all-MiniLM-L6-v2` | Hugging Face (Local) | 384 | **Free** (Local CPU/GPU) | Prototyping, lightweight apps, privacy-critical data |
| `BAAI/bge-small-en-v1.5` | Hugging Face (Local) | 384 | **Free** (Local) | Top MTEB benchmark performance |
| `text-embedding-3-small` | OpenAI | 1536 | $0.02 / 1M tokens | Production cloud applications |
| `text-embedding-3-large` | OpenAI | 3072 | $0.13 / 1M tokens | Complex legal, medical, or multi-lingual texts |
