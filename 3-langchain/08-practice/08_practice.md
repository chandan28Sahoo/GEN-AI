# 08 - Hands-On Practice & Real-World Projects

---

## 🌟 1. What is this Module?

This module provides practical, hands-on exercises and real-world project blueprints to apply everything you learned across Modules 01 through 07.

---

## 📂 Practice Files

- [1_prompt_practice.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/08-practice/1_prompt_practice.py): Interactive playground for testing custom prompt templates and LCEL pipes.

---

## 🎯 4 Step-by-Step Project Challenges (Beginner to Intermediate)

### 🟢 Challenge 1: Multi-Lingual Social Media Post Generator (Beginner)
- **Goal**: Take a technical article topic and concurrently generate a Twitter/X thread, a LinkedIn professional post, and a Reddit summary in parallel.
- **Components to use**:
  - `ChatOpenAI` / `ChatGroq`
  - `PromptTemplate`
  - `RunnableParallel`
  - `StrOutputParser`

---

### 🟡 Challenge 2: Structured Candidate Resume Extractor (Intermediate)
- **Goal**: Load a candidate resume PDF and extract clean, validated JSON with strict data types.
- **Components to use**:
  - `PyPDFLoader` from `07-rag-components/01-document-loaders`
  - `BaseModel` & `Field` with Pydantic
  - `model.with_structured_output(CandidateProfile)`

```python
class CandidateProfile(BaseModel):
    full_name: str
    email: str
    years_of_experience: int
    technical_skills: list[str]
    highest_degree: str
```

---

### 🟠 Challenge 3: End-to-End Local Document Q&A Assistant (Mini RAG)
- **Goal**: Ingest the included [dl-curriculum.pdf](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/09-resources/dl-curriculum.pdf), index it in ChromaDB, and build an interactive Q&A assistant in LCEL.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pathlib import Path

# 1. Ingest PDF Document
pdf_path = Path(__file__).parent.parent / "09-resources" / "dl-curriculum.pdf"
loader = PyPDFLoader(str(pdf_path))
raw_docs = loader.load()

# 2. Split into Overlapping Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)

# 3. Create Local Vector Store with Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. Define Prompt Template
template = ChatPromptTemplate.from_template("""
You are a helpful AI curriculum advisor.
Answer the student's question based strictly on the following course context:

Context:
{context}

Question: {question}

Answer:
""")

# 5. Build LCEL RAG Chain
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

model = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | template
    | model
    | StrOutputParser()
)

# 6. Ask Questions!
response = rag_chain.invoke("What neural network architectures are covered in this curriculum?")
print("AI Response:\n", response)
```

---

### 🔴 Challenge 4: Intelligent Customer Feedback Router & Responder
- **Goal**: Classify incoming customer messages into *Billing*, *Technical Support*, or *Feature Request*, then route to specialized sub-chains.
- **Components to use**:
  - `model.with_structured_output()` for classification
  - `RunnableBranch` for dynamic routing
  - `StrOutputParser` for response generation
