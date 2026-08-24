# 01 - Document Loaders

---

## 🌟 1. What is a Document Loader in LangChain?

A **Document Loader** is a component that extracts text and metadata from diverse external file formats and sources (PDFs, CSVs, plain text files, websites, databases) and converts them into standardized LangChain **`Document`** objects.

### The Structure of a LangChain `Document`
Every document loader outputs one or more `Document` objects:
```python
Document(
    page_content="The text content extracted from the source...",
    metadata={
        "source": "data/sample.pdf",
        "page": 1,
        "author": "Alice"
    }
)
```

```
   Raw External Files (PDF, CSV, TXT, MD)
                   │
                   ▼
        [ Document Loader ]
                   │
                   ▼
    Standardized list[Document]
```

---

## ❓ 2. Why Do We Need Document Loaders?

### The Problems They Solve:
1. **Diverse Formats**: Extracting text from a raw PDF requires complex byte stream parsing and font decoding; a CSV requires row-by-row column mapping; a Directory requires recursion and filtering.
2. **Metadata Preservation**: In RAG systems, knowing *where* an answer came from (the exact page number, source URL, or filename) is critical for citation and hallucination verification.
3. **Memory Efficiency with `lazy_load()`**: For large archives with gigabytes of data, loading everything into RAM crashes your server. LangChain loaders provide lazy generators that yield documents on demand.

---

## 🚀 3. How Do We Use Document Loaders?

### 📂 Files in this Submodule

| File | Loader Class | Target File Format | Included Sample File |
| :--- | :--- | :--- | :--- |
| [1_text_loader.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/1_text_loader.py) | `TextLoader` | Plain text files (`.txt`, `.md`) | [cricket.txt](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/cricket.txt) |
| [2_pdf_loader.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/2_pdf_loader.py) | `PyPDFLoader` | PDF files (page-by-page) | [dl-curriculum.pdf](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/dl-curriculum.pdf) |
| [3_csv_loader.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/3_csv_loader.py) | `CSVLoader` | Tabular CSV files (row-by-row) | [Social_Network_Ads.csv](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/Social_Network_Ads.csv) |
| [4_directory_loader.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/4_directory_loader.py) | `DirectoryLoader` | Bulk directory scanning with glob filters | PDF books in `09-resources` |

---

### Step-by-Step Code Walkthroughs

#### 1. `TextLoader` (`1_text_loader.py`)
```python
from langchain_community.document_loaders import TextLoader
from pathlib import Path

# Always use dynamic paths for reliability
file_path = Path(__file__).parent / 'cricket.txt'

loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
docs = loader.load()

print(f"Loaded {len(docs)} document.")
print(f"Content Preview:\n{docs[0].page_content[:200]}")
print(f"Metadata:\n{docs[0].metadata}")
```

---

#### 2. `PyPDFLoader` (`2_pdf_loader.py`)
Each page of the PDF becomes a distinct `Document` with page number metadata:

```python
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

pdf_path = Path(__file__).parent / 'dl-curriculum.pdf'
loader = PyPDFLoader(str(pdf_path))

# load() returns a list of Document objects (1 per page)
pages = loader.load()

print(f"Total Pages in PDF: {len(pages)}")
print(f"Page 1 Metadata: {pages[0].metadata}")  # {'source': '...', 'page': 0}
print(f"Page 1 Content:\n{pages[0].page_content[:200]}")
```

---

#### 3. `CSVLoader` (`3_csv_loader.py`)
Each row of the CSV becomes a distinct `Document` with column names as keys:

```python
from langchain_community.document_loaders import CSVLoader
from pathlib import Path

csv_path = Path(__file__).parent / 'Social_Network_Ads.csv'
loader = CSVLoader(file_path=str(csv_path))

rows = loader.load()

print(f"Total Rows Loaded: {len(rows)}")
print("First Row Document:\n", rows[0].page_content)
# Output format:
# User ID: 15624510
# Gender: Male
# Age: 19
# EstimatedSalary: 19000
# Purchased: 0
```

---

#### 4. `DirectoryLoader` with Lazy Loading (`4_directory_loader.py`)
```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pathlib import Path

resources_dir = Path(__file__).resolve().parent.parent.parent / '09-resources'

# Recursively find all PDF files in directory and load them using PyPDFLoader
loader = DirectoryLoader(
    path=str(resources_dir),
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# lazy_load() streams documents one by one using a Python generator (saves RAM)
for doc in loader.lazy_load():
    print(f"Loaded page from: {doc.metadata['source']} (Page: {doc.metadata.get('page', 0)})")
```

---

## 💡 Best Practices

> [!TIP]
> 1. **Specify UTF-8**: Always pass `encoding='utf-8'` in `TextLoader` to prevent cross-platform encoding errors between Windows, macOS, and Linux.
> 2. **Use `lazy_load()` for Large Datasets**: For folders with hundreds of large PDFs, calling `.load()` can trigger Out-Of-Memory (OOM) crashes. `lazy_load()` yields documents one at a time.
