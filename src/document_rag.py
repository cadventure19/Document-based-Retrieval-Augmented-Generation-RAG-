import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# --- Load environment variables ---
load_dotenv()

# --- Define directories ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# --- Local embeddings ---
class LocalEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embeddings = LocalEmbeddings()

# --- Load or create vector store ---
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("Vector store created and persisted.")
else:
    print("Vector store already exists.")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# --- Retrieve relevant docs ---
retriever = db.as_retriever(search_type="similarity_score_threshold",
                            search_kwargs={"k": 3, "score_threshold": 0.3})
query = "Who is Odysseus' wife?"
relevant_docs = retriever.get_relevant_documents(query)

# --- Prepare context from relevant docs ---
context_text = "\n".join([doc.page_content for doc in relevant_docs])

# --- Call OpenAI API directly using retrieved context ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",  # make sure you have quota
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer the following question based on the context below:\n{context_text}\nQuestion: {query}"}
    ],
    temperature=0
)

print("\n--- Answer ---")
print(response.choices[0].message.content)