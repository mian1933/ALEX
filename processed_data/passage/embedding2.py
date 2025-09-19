import os
import json
from typing import List, Dict, Any, Iterator
from sentence_transformers import SentenceTransformer
import re
import time
import torch
import chromadb
from tqdm import tqdm

CHROMA_DB_PATH = "[PATH_TO_YOUR_CHROMA_DB_FOLDER]"
COLLECTION_NAME = "[YOUR_COLLECTION_NAME]"
JSON_FILE_PATH = "[PATH_TO_YOUR_JSONL_FILE]"
TXT_FOLDER_PATH = "[PATH_TO_YOUR_TXT_FOLDER]"
MODEL_NAME = "[PATH_TO_YOUR_SENTENCE_TRANSFORMER_MODEL]"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKEN_CHUNK_SIZE = 1000
TOKEN_OVERLAP = 100
DOC_CHUNK_SIZE = 128
EMBEDDING_BATCH_SIZE = 128


def split_text_by_tokens(text: str, tokenizer, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        if chunk_tokens:
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
    return chunks


def stream_data(json_path: str, txt_folder: str, chunk_size: int) -> Iterator[List[Dict[str, Any]]]:
    chunk = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        raw_text = data.get('text')
                        text = str(raw_text or '').strip()
                        if text:
                            metadata = data.get('metadata', {})
                            if not isinstance(metadata, dict):
                                metadata = {'original_metadata': metadata}
                            chunk.append({'content': text, 'metadata': metadata})
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        print(f"Warning: JSONL file '{json_path}' not found.")

    if os.path.isdir(txt_folder):
        txt_files = [fn for fn in os.listdir(txt_folder) if fn.endswith('.txt')]
        for fn in txt_files:
            file_path = os.path.join(txt_folder, fn)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    chunk.append({'content': content, 'metadata': {'source': fn}})
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
    if chunk:
        yield chunk


def main():
    print("=" * 20 + " STAGE 1: Initializing ChromaDB " + "=" * 20)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    print(f"✅ Database and collection '{COLLECTION_NAME}' initialized successfully.\n")

    print("=" * 20 + " STAGE 2: Populating Database (Chunking & Streaming Mode) " + "=" * 20)

    if collection.count() > 0:
        print(f"✅ Collection already contains {collection.count()} chunks, skipping population step.")
        return

    print(f"Collection is empty, starting initial data chunking, embedding, and ingestion...")
    print(f"Text chunking config: Chunk Size = {TOKEN_CHUNK_SIZE} tokens, Overlap = {TOKEN_OVERLAP} tokens.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("✅ Model loaded successfully.")

    data_generator = stream_data(JSON_FILE_PATH, TXT_FOLDER_PATH, DOC_CHUNK_SIZE)

    total_chunks_processed = 0
    for doc_chunk_id, doc_chunk in enumerate(data_generator):
        if not doc_chunk:
            continue

        start_time = time.time()
        print(f"--- Processing document batch #{doc_chunk_id + 1} (containing {len(doc_chunk)} documents) ---")

        batch_texts = []
        batch_metadatas = []
        batch_ids = []

        for doc in doc_chunk:
            text_chunks = split_text_by_tokens(
                doc['content'],
                model.tokenizer,
                TOKEN_CHUNK_SIZE,
                TOKEN_OVERLAP
            )

            for i, text_chunk in enumerate(text_chunks):
                new_id = str(total_chunks_processed + len(batch_ids))
                new_metadata = doc['metadata'].copy()
                new_metadata['chunk_id'] = i

                if 'path' in new_metadata:
                    new_metadata['source'] = new_metadata['path']

                if 'source' not in new_metadata:
                    new_metadata['source'] = 'unknown_source'

                batch_texts.append(text_chunk)
                batch_metadatas.append(new_metadata)
                batch_ids.append(new_id)

        if not batch_texts:
            print("No valid text chunks produced from this batch, skipping.")
            continue

        print(f"Generated {len(batch_texts)} text chunks from {len(doc_chunk)} documents, preparing for embedding...")

        embeddings_batch = model.encode(
            batch_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=DEVICE
        ).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings_batch,
            documents=batch_texts,
            metadatas=batch_metadatas
        )

        total_chunks_processed += len(batch_texts)
        end_time = time.time()

        print(f"✅ Document batch #{doc_chunk_id + 1} processed in: {end_time - start_time:.2f} seconds.")
        print(f"Total text chunks processed so far: {total_chunks_processed}\n")

    print("=" * 60)
    print("🎉 All data has been successfully chunked, embedded, and indexed!")
    print(f"Database located at: '{CHROMA_DB_PATH}', Collection name: '{COLLECTION_NAME}'")
    print(f"A total of {total_chunks_processed} text chunks were processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()