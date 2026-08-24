from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pathlib import Path

# Directory path pointing to 09-resources where sample PDF books are kept
resources_dir = Path(__file__).resolve().parent.parent.parent / '09-resources'

loader = DirectoryLoader(
    path=str(resources_dir),
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# load vs lazy_load
# docs = loader.load()
docs = loader.lazy_load()

print("Loading documents lazily from directory:")
for document in docs:
    print("Metadata:", document.metadata)
    break  # print first document metadata as demo