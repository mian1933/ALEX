import os
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
import chromadb
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import json

CHROMA_DB_PATH = "[PATH_TO_YOUR_CHROMA_DB_FOLDER]"
COLLECTION_NAME = "[YOUR_COLLECTION_NAME]"
MODEL_NAME = "[PATH_TO_YOUR_SENTENCE_TRANSFORMER_MODEL]"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_text(text: Any) -> List[str]:
    if not isinstance(text, str): return []
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()


def load_all_chunks_from_db(collection: chromadb.Collection) -> (Dict[str, str], Dict[str, List[str]]):
    print("--- Loading all text chunks from the database to support BM25...")
    all_data = collection.get(include=["documents"])
    all_ids = all_data['ids']
    all_documents = all_data['documents']
    if not all_documents:
        raise RuntimeError("Failed to load any document chunks from the database.")

    id_to_document = {id_str: doc for id_str, doc in zip(all_ids, all_documents)}

    print(f"Tokenizing {len(all_documents)} text chunks...")
    id_to_tokenized = {id_str: preprocess_text(doc) for id_str, doc in tqdm(id_to_document.items())}

    print("✅ All text chunks loaded and tokenized.")
    return id_to_document, id_to_tokenized


class ChromaHybridSearchEngine:
    def __init__(self, id_to_document, id_to_tokenized, collection, model):
        self.id_to_document = id_to_document
        self.id_to_tokenized = id_to_tokenized
        self.collection = collection
        self.model = model

    def search(self, query_str: str, top_k: int = 1, recall_k: int = 100, alpha: float = 0.7) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            query_str, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=recall_k,
            include=["metadatas", "distances"]
        )

        if not results['ids'][0]:
            return []

        candidate_ids = results['ids'][0]
        vector_scores = [1 - dist for dist in results['distances'][0]]
        candidate_metadatas = results['metadatas'][0]

        candidate_tokenized_docs = [self.id_to_tokenized[id_str] for id_str in candidate_ids]
        bm25_reranker = BM25Okapi(candidate_tokenized_docs)
        query_tokens = preprocess_text(query_str)
        bm25_scores = bm25_reranker.get_scores(query_tokens)

        fused_results = []
        min_vec_score, max_vec_score = (min(vector_scores), max(vector_scores)) if len(vector_scores) > 1 else (0, 1)
        min_bm25_score, max_bm25_score = (min(bm25_scores), max(bm25_scores)) if len(bm25_scores) > 1 else (0, 1)

        for i, doc_id in enumerate(candidate_ids):
            norm_v = (vector_scores[i] - min_vec_score) / (max_vec_score - min_vec_score + 1e-9)
            norm_b = (bm25_scores[i] - min_bm25_score) / (max_bm25_score - min_bm25_score + 1e-9)
            hybrid_score = alpha * norm_v + (1 - alpha) * norm_b

            fused_results.append({
                'content': self.id_to_document[doc_id],
                'metadata': candidate_metadatas[i],
                'score': float(hybrid_score)
            })

        fused_results.sort(key=lambda x: x['score'], reverse=True)
        return fused_results[:top_k]


class SystemInitializer:
    def __init__(self):
        print("=" * 20 + " Retrieval System Initialization Start " + "=" * 20)
        self.collection = self._connect_to_chromadb()
        self.id_to_document, self.id_to_tokenized = load_all_chunks_from_db(self.collection)
        self.engine = self._initialize_engine()
        print("=" * 20 + " ✅ Retrieval System Initialization Complete " + "=" * 20 + "\n")

    def _connect_to_chromadb(self):
        print("--- Connecting to ChromaDB... ---")
        if not os.path.exists(CHROMA_DB_PATH):
            raise FileNotFoundError(
                f"Error: Database path '{CHROMA_DB_PATH}' does not exist. Please run the database creation script first.")

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            print(
                f"✅ Successfully connected to collection '{COLLECTION_NAME}', containing {collection.count()} documents.\n")
            return collection
        except Exception as e:
            raise ValueError(f"Error: Could not get collection '{COLLECTION_NAME}'. Error: {e}")

    def _initialize_engine(self) -> ChromaHybridSearchEngine:
        print("--- Loading query model... ---")
        query_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        engine = ChromaHybridSearchEngine(
            self.id_to_document,
            self.id_to_tokenized,
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


def query(query_text: str, top_k: int = 1, recall_k: int = 100, alpha: float = 0.7) -> List[Dict[str, Any]]:
    if _system_instance is None:
        print("Error: Retrieval system was not initialized successfully, cannot perform query.")
        return []

    if not query_text or not isinstance(query_text, str):
        print("Error: Invalid query text.")
        return []

    return _system_instance.engine.search(query_str=query_text, top_k=top_k, recall_k=recall_k, alpha=alpha)