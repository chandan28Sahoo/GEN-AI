import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

load_dotenv()


# 1. Create LangChain documents for IPL players
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# 2. Initialize Chroma Vector Store
persist_dir = str(Path(__file__).parent / 'my_chroma_db')
vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_AI_API_KEY")
    ),
    persist_directory=persist_dir,
    collection_name='ipl_players'
)

# 3. Add documents to vector store
print("--- Adding Documents ---")
doc_ids = vector_store.add_documents(docs)
print(f"Added {len(doc_ids)} documents with IDs: {doc_ids}")

# 4. View documents
print("\n--- View Documents in Collection ---")
collection_data = vector_store.get(include=['documents', 'metadatas'])
print(f"Stored Document Count: {len(collection_data['documents'])}")

# 5. Semantic Similarity Search
print("\n--- Similarity Search: 'Who among these are a bowler?' ---")
search_results = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)
for i, res in enumerate(search_results, 1):
    print(f"Result {i}: {res.page_content[:100]}... | Meta: {res.metadata}")

# 6. Similarity Search with Scores
print("\n--- Similarity Search with Score ---")
results_with_scores = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
for res, score in results_with_scores:
    print(f"Score: {score:.4f} | Document: {res.page_content[:80]}...")

# 7. Metadata Filtering
print("\n--- Metadata Filtering (team == Chennai Super Kings) ---")
filtered_results = vector_store.similarity_search(
    query='captain leader',
    filter={"team": "Chennai Super Kings"}
)
for res in filtered_results:
    print(f"CSK Result: {res.page_content[:80]}...")

# 8. Update a document
if doc_ids:
    target_id = doc_ids[0]
    updated_doc1 = Document(
        page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history.",
        metadata={"team": "Royal Challengers Bangalore", "status": "legend"}
    )
    print(f"\n--- Updating Document (ID: {target_id}) ---")
    vector_store.update_document(document_id=target_id, document=updated_doc1)

# 9. Delete a document
if len(doc_ids) > 1:
    delete_id = doc_ids[-1]
    print(f"\n--- Deleting Document (ID: {delete_id}) ---")
    vector_store.delete(ids=[delete_id])
    print("Document deleted successfully.")
