from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

dl_curriculum_pdf = Path(__file__).resolve().parent.parent.parent / '09-resources' / 'dl-curriculum.pdf'
loader = PyPDFLoader(str(dl_curriculum_pdf))

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(f"Total chunks created: {len(result)}")
print("First Chunk:\n", result[0])