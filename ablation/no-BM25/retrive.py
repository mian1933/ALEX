import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
import chromadb

CHROMA_DB_PATH = "[PATH_TO_YOUR_CHROMA_DB_FOLDER]"
COLLECTION_NAME = "[YOUR_COLLECTION_NAME]"
MODEL_NAME = "[PATH_TO_YOUR_SENTENCE_TRANSFORMER_MODEL]"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ChromaVectorSearchEngine:
    def __init__(self, collection, model):
        self.collection = collection
        self.model = model

    def search(self, query_str: str, top_k: int = 1) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            query_str, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )

        if not results['ids'][0]:
            return []

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]
            })
        return formatted_results

class SystemInitializer:
    def __init__(self):
        print("=" * 20 + " Retrieval System Initialization Start " + "=" * 20)
        self.collection = self._connect_to_chromadb()
        self.engine = self._initialize_engine()
        print("=" * 20 + " ✅ Retrieval System Initialization Complete " + "=" * 20 + "\n")

    def _connect_to_chromadb(self):
        print("--- Connecting to ChromaDB... ---")
        if not os.path.exists(CHROMA_DB_PATH):
            raise FileNotFoundError(f"Error: Database path '{CHROMA_DB_PATH}' does not exist. Please run the database creation script first.")

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            print(f"✅ Successfully connected to collection '{COLLECTION_NAME}', containing {collection.count()} documents.\n")
            return collection
        except Exception as e:
            raise ValueError(f"Error: Could not get collection '{COLLECTION_NAME}'. Error: {e}")

    def _initialize_engine(self) -> ChromaVectorSearchEngine:
        print("--- Loading query model... ---")
        query_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        engine = ChromaVectorSearchEngine(
            self.collection,
            query_model
        )
        print("✅ Query model loaded successfully.\n")
        return engine

try:
    print("Initializing retrieval system module...")
    _system_instance = SystemInitializer()
except Exception as e:
    print(f"Fatal Error: Retrieval system initialization failed! Error: {e}")
    _system_instance = None

def query(query_text: str, top_k: int = 1, **kwargs) -> List[Dict[str, Any]]:
    if _system_instance is None:
        print("Error: Retrieval system was not initialized successfully, cannot perform query.")
        return []

    if not query_text or not isinstance(query_text, str):
        print("Error: Invalid query text.")
        return []

    return _system_instance.engine.search(query_str=query_text, top_k=top_k)