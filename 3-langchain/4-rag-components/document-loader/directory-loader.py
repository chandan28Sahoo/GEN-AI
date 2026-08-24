from langchain_community.document_loaders import  DirectoryLoader, PyPDFLoader
from pathlib import Path



# dl_curriculum_pdf = Path(__file__).parent / 'dl-curriculum.pdf'

books_dir = Path(__file__).parent / 'books'

loader = DirectoryLoader(
    path=books_dir,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

#load vs lezy_load
# docs = loader.load()
docs = loader.lazy_load()

# print(len(docs))
for document in docs:
    print(document.metadata)