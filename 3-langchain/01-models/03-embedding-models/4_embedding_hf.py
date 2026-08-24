from langchain_huggingface import HuggingFaceEmbeddings
import dotenv

dotenv.load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Delhi is the capital of India"
    # "Kolkata is the capital of West Bengal",
    # "Paris is the capital of France"
]

vectors = embeddings.embed_documents(documents)

print(vectors)