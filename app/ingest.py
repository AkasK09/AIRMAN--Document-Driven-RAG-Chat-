import os
import sys
import pickle
import uuid
import argparse
import re
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
from rank_bm25 import BM25Okapi

from app.config import settings
from app.utils import setup_logger, clean_text

logger = setup_logger(__name__)

# Cache embedding model to avoid reloading on every request
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text page by page from PDF using PyMuPDF.
    Uses regex to extract ATA codes and sections.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    logger.info(f"Extracting text from PDF: {pdf_path}")
    pages_data = []
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open/parse PDF {pdf_path}: {str(e)}")
        raise ValueError(f"Corrupted or invalid PDF file: {str(e)}")

    if len(doc) == 0:
        raise ValueError(f"The PDF file is empty: {pdf_path}")

    current_ata = None
    current_section = None
    
    ata_pattern = re.compile(r'(ATA\s*\d{2}(?:-\d{2})?(?:-\d{2})?)', re.IGNORECASE)
    chapter_pattern = re.compile(r'(CHAPTER\s*\d+|SECTION\s*\d+)', re.IGNORECASE)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1  # 1-indexed
        
        raw_text = page.get_text("text")
        cleaned_text = clean_text(raw_text)
        
        # Search for ATA or Chapter/Section
        ata_match = ata_pattern.search(cleaned_text)
        if ata_match:
            current_ata = ata_match.group(1).upper()
            
        chap_match = chapter_pattern.search(cleaned_text)
        if chap_match:
            current_section = chap_match.group(1).title()
        
        if cleaned_text:
            pages_data.append({
                "page_number": page_num,
                "text": cleaned_text,
                "ata_chapter": current_ata,
                "section": current_section
            })
            
    doc.close()
    logger.info(f"Extracted {len(pages_data)} pages containing text from {pdf_path}")
    return pages_data

def build_hierarchical_chunks(pages_data: List[Dict[str, Any]], document_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Creates Parent-Child chunks using tiktoken.
    Returns: (list of child chunks, dict of parent_id -> parent_text)
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    all_child_chunks = []
    parent_mapping = {}
    
    current_parent_tokens = []
    current_parent_id = f"{document_name}_parent_{uuid.uuid4().hex[:6]}"
    
    current_parent_metadata = {
        "document_name": document_name,
        "page_number": None,
        "ata_chapter": None,
        "section": None
    }
    
    def process_parent(parent_id, tokens, meta):
        text = tokenizer.decode(tokens)
        parent_mapping[parent_id] = text
        
        # split into children
        child_size = settings.CHILD_CHUNK_SIZE
        overlap = 50
        start = 0
        idx = 0
        while start < len(tokens):
            end = min(start + child_size, len(tokens))
            child_tokens = tokens[start:end]
            child_text = tokenizer.decode(child_tokens).strip()
            
            if child_text:
                chunk_id = f"{parent_id}_c{idx}"
                all_child_chunks.append({
                    "chunk_id": chunk_id,
                    "parent_chunk_id": parent_id,
                    "document_name": meta["document_name"],
                    "page_number": meta["page_number"],
                    "ata_chapter": meta["ata_chapter"],
                    "section": meta["section"],
                    "chunk_text": child_text
                })
                idx += 1
            if end == len(tokens):
                break
            # advance by chunk_size - overlap, ensuring step > 0
            step = child_size - overlap
            if step <= 0: step = child_size
            start += step
            
    for page in pages_data:
        text = page["text"]
        tokens = tokenizer.encode(text)
        
        # update metadata state
        if current_parent_metadata["page_number"] is None:
            current_parent_metadata["page_number"] = page["page_number"]
        if page["ata_chapter"]:
            current_parent_metadata["ata_chapter"] = page["ata_chapter"]
        if page["section"]:
            current_parent_metadata["section"] = page["section"]
            
        start = 0
        while start < len(tokens):
            # how many tokens we can still fit in the current parent
            space_left = settings.PARENT_CHUNK_SIZE - len(current_parent_tokens)
            chunk_tokens = tokens[start:start+space_left]
            current_parent_tokens.extend(chunk_tokens)
            
            start += len(chunk_tokens)
            
            if len(current_parent_tokens) >= settings.PARENT_CHUNK_SIZE:
                process_parent(current_parent_id, current_parent_tokens, current_parent_metadata)
                current_parent_id = f"{document_name}_parent_{uuid.uuid4().hex[:6]}"
                current_parent_tokens = []
                current_parent_metadata["page_number"] = page["page_number"]
                
    if current_parent_tokens:
        process_parent(current_parent_id, current_parent_tokens, current_parent_metadata)
        
    logger.info(f"Generated {len(all_child_chunks)} child chunks and {len(parent_mapping)} parent chunks for document: {document_name}")
    return all_child_chunks, parent_mapping

def load_or_create_vector_store() -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, str], Optional[BM25Okapi]]:
    """
    Load vector store, metadata, parent mapping, and BM25 index.
    """
    index_path = os.path.join(settings.FAISS_PATH, "faiss_index.bin")
    metadata_path = os.path.join(settings.FAISS_PATH, "metadata.pkl")
    parents_path = os.path.join(settings.FAISS_PATH, "parents.pkl")
    bm25_path = settings.BM25_INDEX_PATH
    
    os.makedirs(settings.FAISS_PATH, exist_ok=True)
    
    index = None
    metadata = []
    parent_mapping = {}
    bm25 = None
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        logger.info(f"Loading existing vector store from {settings.FAISS_PATH}")
        try:
            index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            if os.path.exists(parents_path):
                with open(parents_path, "rb") as f:
                    parent_mapping = pickle.load(f)
                    
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    bm25 = pickle.load(f)
                    
            return index, metadata, parent_mapping, bm25
        except Exception as e:
            logger.error(f"Failed to load existing vector store: {str(e)}. Re-creating database.")
            
    # Create new
    model = get_embedding_model()
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)
    logger.info(f"Created new FAISS index (Inner Product, dimension {dimension})")
    
    return index, [], {}, None

def save_vector_store(index: faiss.Index, metadata: List[Dict[str, Any]], parent_mapping: Dict[str, str], bm25: BM25Okapi) -> None:
    """
    Save the FAISS index, metadata, parents, and BM25.
    """
    os.makedirs(settings.FAISS_PATH, exist_ok=True)
    index_path = os.path.join(settings.FAISS_PATH, "faiss_index.bin")
    metadata_path = os.path.join(settings.FAISS_PATH, "metadata.pkl")
    parents_path = os.path.join(settings.FAISS_PATH, "parents.pkl")
    bm25_path = settings.BM25_INDEX_PATH
    
    logger.info(f"Saving vector store with {index.ntotal} vectors to {settings.FAISS_PATH}")
    faiss.write_index(index, index_path)
    
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
        
    with open(parents_path, "wb") as f:
        pickle.dump(parent_mapping, f)
        
    if bm25 is not None:
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

def ingest_pdf(pdf_path: str) -> int:
    """
    Core function to ingest a PDF. Extracts, chunks, embeds, and stores in FAISS + BM25.
    """
    document_name = os.path.basename(pdf_path)
    
    # 1. Extract text page-by-page
    pages_data = extract_text_from_pdf(pdf_path)
    
    # 2. Chunk text
    chunks, new_parents = build_hierarchical_chunks(pages_data, document_name)
    if not chunks:
        logger.warning(f"No chunks created for PDF: {pdf_path}")
        return 0
        
    # 3. Generate embeddings
    model = get_embedding_model()
    texts = [c["chunk_text"] for c in chunks]
    
    logger.info(f"Generating embeddings for {len(texts)} child chunks")
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # Convert to float32
    embeddings = np.array(embeddings).astype("float32")
    
    # Normalize for cosine similarity search
    faiss.normalize_L2(embeddings)
    
    # 4. Load existing index & metadata, append new vectors
    index, metadata, parent_mapping, bm25 = load_or_create_vector_store()
    
    # Add to FAISS index
    index.add(embeddings)
    
    # Append metadata and parents
    metadata.extend(chunks)
    parent_mapping.update(new_parents)
    
    # Rebuild BM25 for all chunks (BM25Okapi doesn't support incremental update easily)
    logger.info("Rebuilding BM25 index for all chunks...")
    corpus_texts = [c["chunk_text"] for c in metadata]
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save back to disk
    save_vector_store(index, metadata, parent_mapping, bm25)
    
    logger.info(f"Ingestion successful. Added {len(chunks)} chunks from {document_name}.")
    return len(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF document(s) into the FAISS vector database")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf-path", type=str, help="Absolute path to a single aviation PDF document")
    group.add_argument("--dir", type=str, help="Path to a directory containing aviation PDF documents")
    args = parser.parse_args()
    
    try:
        if args.pdf_path:
            num_chunks = ingest_pdf(args.pdf_path)
            print(f"Success: Ingested {num_chunks} chunks from {os.path.basename(args.pdf_path)}.")
        else:
            if not os.path.isdir(args.dir):
                print(f"Error: Directory not found: {args.dir}", file=sys.stderr)
                sys.exit(1)
            
            pdf_files = [
                os.path.join(args.dir, f) for f in os.listdir(args.dir)
                if f.lower().endswith(".pdf")
            ]
            
            if not pdf_files:
                print(f"No PDF files found in directory: {args.dir}")
                sys.exit(0)
                
            print(f"Found {len(pdf_files)} PDF files to process in {args.dir}.")
            total_chunks = 0
            for pdf_file in pdf_files:
                print(f"Ingesting: {os.path.basename(pdf_file)}...")
                try:
                    num_chunks = ingest_pdf(pdf_file)
                    total_chunks += num_chunks
                    print(f"  Processed {num_chunks} chunks.")
                except Exception as file_error:
                    print(f"  Error ingesting {os.path.basename(pdf_file)}: {str(file_error)}", file=sys.stderr)
                    
            print(f"Success: Ingested a total of {total_chunks} chunks from {len(pdf_files)} PDF documents.")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
