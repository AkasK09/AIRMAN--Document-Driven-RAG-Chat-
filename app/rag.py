import os
import json
import pickle
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
import requests
from openai import OpenAI
from sentence_transformers import CrossEncoder

from app.config import settings
from app.config import settings
from app.utils import setup_logger, format_subject_name
from app.models import Citation, AskResponse
from app.ingest import get_embedding_model

logger = setup_logger(__name__)

REFUSAL_RESPONSE = "The information is not available in the provided manuals."

# Cache reranker
_reranker_model = None

def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
        _reranker_model = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker_model

def load_stores():
    from app.ingest import load_or_create_vector_store
    index, metadata, parent_mapping, bm25 = load_or_create_vector_store()
    if index is None or index.ntotal == 0 or bm25 is None:
        raise FileNotFoundError(
            "Vector store files not found. Please ingest documents using the ingest endpoint or script first."
        )
    return index, metadata, parent_mapping, bm25

def hybrid_search(
    query: str, 
    index: faiss.Index, 
    metadata: List[Dict[str, Any]], 
    bm25, 
    ata_filter: str = None, 
    subject_filter: str = None,
    chapter_filter: str = None,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Combines FAISS Semantic Search and BM25 Keyword Search using Reciprocal Rank Fusion (RRF).
    """
    # Semantic Search
    model = get_embedding_model()
    query_vector = model.encode([query], show_progress_bar=False)
    query_vector = np.array(query_vector).astype("float32")
    faiss.normalize_L2(query_vector)
    
    # Retrieve more chunks to allow filtering and RRF
    search_k = max(top_k * 3, 100)
    distances, indices = index.search(query_vector, search_k)
    
    # BM25 Keyword Search
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # RRF (Reciprocal Rank Fusion)
    # 1. Rank Semantic
    semantic_ranks = {}
    rank = 1
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(metadata): 
            continue
        chunk_meta = metadata[idx]
        if ata_filter and chunk_meta.get("ata_chapter") != ata_filter:
            continue
        if subject_filter and format_subject_name(chunk_meta.get("document_name")) != subject_filter:
            continue
        if chapter_filter and chunk_meta.get("section") != chapter_filter:
            continue
        semantic_ranks[idx] = rank
        rank += 1
        
    # 2. Rank BM25
    bm25_ranks = {}
    # Filter bm25 candidates by ATA, Subject, Chapter if needed
    valid_indices = []
    for i in range(len(metadata)):
        m = metadata[i]
        if ata_filter and m.get("ata_chapter") != ata_filter:
            continue
        if subject_filter and format_subject_name(m.get("document_name")) != subject_filter:
            continue
        if chapter_filter and m.get("section") != chapter_filter:
            continue
        valid_indices.append(i)
    
    # Get top sorted indices for BM25
    sorted_bm25 = sorted(valid_indices, key=lambda i: bm25_scores[i], reverse=True)[:search_k]
    for rank, idx in enumerate(sorted_bm25, 1):
        bm25_ranks[idx] = rank
        
    # Combine via RRF: score = 1 / (k + rank_semantic) + 1 / (k + rank_bm25)
    rrf_k = 60
    combined_scores = {}
    all_candidate_indices = set(semantic_ranks.keys()).union(set(bm25_ranks.keys()))
    
    for idx in all_candidate_indices:
        s_rank = semantic_ranks.get(idx, 1000)
        b_rank = bm25_ranks.get(idx, 1000)
        combined_scores[idx] = (1.0 / (rrf_k + s_rank)) + (1.0 / (rrf_k + b_rank))
        
    # Sort and take top_k (Top 20 candidate chunks)
    top_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        chunk = metadata[idx].copy()
        chunk["hybrid_score"] = combined_scores[idx]
        
        # safely assign semantic score if present
        if idx in indices[0]:
            chunk["semantic_score"] = float(distances[0][list(indices[0]).index(idx)])
        else:
            chunk["semantic_score"] = 0.0
            
        chunk["bm25_score"] = float(bm25_scores[idx])
        results.append(chunk)
        
    return results

def rerank_chunks(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Reranks the top candidate chunks using a CrossEncoder.
    """
    if not candidates:
        return []
        
    reranker = get_reranker_model()
    pairs = [[query, chunk["chunk_text"]] for chunk in candidates]
    
    scores = reranker.predict(pairs)
    
    for idx, chunk in enumerate(candidates):
        chunk["reranker_score"] = float(scores[idx])
        
    # Sort by reranker score
    candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
    return candidates[:top_k]


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Generate text using Gemini, OpenAI compatible API, or Ollama.
    Attempts to return a JSON string.
    """
    if settings.GEMINI_API_KEY:
        logger.info(f"Using Gemini API with model: {settings.MODEL_NAME}")
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            model_name = settings.MODEL_NAME
            if "gemini" not in model_name.lower():
                model_name = "gemini-2.5-flash"
                
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            content = response.text
            if content:
                return content.strip()
            raise RuntimeError("Gemini API call returned empty content.")
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise RuntimeError(f"Gemini API call failed: {str(e)}")

    elif settings.OPENAI_API_KEY or settings.OPENAI_API_BASE:
        logger.info(f"Using OpenAI compatible API with model: {settings.MODEL_NAME}")
        try:
            client = OpenAI(
                api_key=settings.OPENAI_API_KEY or "dummy_key",
                base_url=settings.OPENAI_API_BASE
            )
            response = client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            logger.error(f"OpenAI compatible API call failed: {str(e)}")
            
    # Default: Ollama
    logger.info(f"Using Ollama API at {settings.OLLAMA_URL} with model: {settings.MODEL_NAME}")
    payload = {
        "model": settings.MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "options": {"temperature": 0.0},
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(f"{settings.OLLAMA_URL}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        content = res_json.get("message", {}).get("content", "")
        return content.strip()
    except Exception as e:
        logger.error(f"Ollama API call failed: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")

def route_query(query: str) -> Dict[str, Any]:
    """
    Agentic Routing: Uses the LLM to classify the query intent to bypass vector search if unneeded.
    """
    system_prompt = (
        "You are an intelligent query router for an Aviation Document Assistant.\n"
        "Analyze the user's input and classify it as either 'chitchat' (greetings, casual conversation, identity questions) "
        "or 'domain' (questions about aviation, meteorology, navigation, maintenance, or any technical topic).\n\n"
        "If it is 'chitchat', provide a friendly, helpful 'response' directly.\n"
        "If it is 'domain', leave 'response' empty.\n\n"
        "Return ONLY a valid JSON object strictly matching this schema:\n"
        "{\n"
        "  \"intent\": \"chitchat\" or \"domain\",\n"
        "  \"response\": \"Your friendly response if chitchat, else empty string\"\n"
        "}"
    )
    try:
        raw_res = call_llm(system_prompt, query)
        parsed = json.loads(raw_res)
        return parsed
    except Exception as e:
        logger.warning(f"Routing failed, defaulting to domain: {str(e)}")
        return {"intent": "domain", "response": ""}

def generate_rag_response(
    question: str, 
    ata_filter: str = None, 
    subject_filter: str = None,
    chapter_filter: str = None,
    debug: bool = False
) -> AskResponse:
    """
    RAG retrieval and response generation workflow with strict hallucination control, Hybrid Search, and Routing.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # 0. Agentic Query Routing
    route_info = route_query(question)
    if route_info.get("intent") == "chitchat" and route_info.get("response"):
        logger.info("Query routed as chitchat. Bypassing vector search.")
        return AskResponse(
            answer=route_info.get("response"),
            confidence="High",
            citations=[],
            retrieved_chunks=[] if debug else None
        )

    # 1. Load Stores
    index, metadata, parent_mapping, bm25 = load_stores()
    
    # 2. Hybrid Retrieval (Top 20 candidate chunks)
    top_20_chunks = hybrid_search(
        question, 
        index, 
        metadata, 
        bm25, 
        ata_filter=ata_filter, 
        subject_filter=subject_filter,
        chapter_filter=chapter_filter,
        top_k=20
    )
    
    if not top_20_chunks:
        logger.info("No documents matched search criteria.")
        return AskResponse(answer=REFUSAL_RESPONSE, citations=[])
        
    # 3. Rerank (Top 3-5 chunks)
    top_5_chunks = rerank_chunks(question, top_20_chunks, top_k=5)

    # 4. Fetch Parent Context for final LLM injection
    context_text = ""
    for idx, chunk in enumerate(top_5_chunks):
        parent_id = chunk.get("parent_chunk_id")
        parent_text = parent_mapping.get(parent_id, chunk["chunk_text"])
        
        doc_name = chunk.get("document_name", "N/A")
        page_num = chunk.get("page_number", "N/A")
        ata = chunk.get("ata_chapter", "N/A")
        sec = chunk.get("section", "N/A")
        
        context_text += f"[Context Index: {idx}]\nDocument: {doc_name}\nATA Chapter: {ata}\nSection: {sec}\nPage: {page_num}\nContent: {parent_text}\n\n"

    # 5. Construct Prompts
    system_prompt = (
        "You are an Aeronautics QA Assistant.\n\n"
        "Answer ONLY from the provided manual context.\n\n"
        "Do not use outside knowledge.\n\n"
        "If the answer is not available in the retrieved context, respond:\n"
        f"'{REFUSAL_RESPONSE}'\n\n"
        "Always cite the source document, section, ATA chapter, and page number when available.\n\n"
        "You MUST output your response as a JSON object with this EXACT schema:\n"
        "{\n"
        "  \"answer\": \"<grounded response>\",\n"
        "  \"confidence\": \"High / Medium / Low\",\n"
        "  \"used_context_indices\": [integer list of context block indices used, e.g. [0, 2]]\n"
        "}\n"
        "Ensure the JSON output is valid."
    )
    
    user_prompt = (
        f"Context:\n{context_text}\n"
        f"Question:\n{question}\n"
        f"Answer:"
    )

    # 6. Invoke LLM
    try:
        llm_response_raw = call_llm(system_prompt, user_prompt)
        logger.debug(f"Raw LLM Response: {llm_response_raw}")
        
        try:
            parsed = json.loads(llm_response_raw)
            answer = parsed.get("answer", "").strip()
            confidence = parsed.get("confidence", "Unknown")
            used_indices = parsed.get("used_context_indices", [])
        except Exception as json_err:
            logger.warning(f"Failed to parse JSON: {str(json_err)}. String fallback.")
            if REFUSAL_RESPONSE.lower() in llm_response_raw.lower() or "not available" in llm_response_raw.lower():
                answer = REFUSAL_RESPONSE
                confidence = "Low"
                used_indices = []
            else:
                answer = llm_response_raw
                confidence = "Unknown"
                used_indices = [0]
                
    except Exception as llm_err:
        logger.error(f"LLM Generation Error: {str(llm_err)}")
        raise llm_err

    # Check for refusal
    if not answer or answer == REFUSAL_RESPONSE or REFUSAL_RESPONSE.lower() in answer.lower():
        logger.info("Response is a refusal.")
        return AskResponse(
            answer=REFUSAL_RESPONSE,
            confidence="Low",
            citations=[],
            retrieved_chunks=top_5_chunks if debug else None
        )

    # 7. Generate Citations
    citations = []
    seen_citations = set()
    
    for idx in used_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= len(top_5_chunks):
            continue
            
        chunk = top_5_chunks[idx]
        doc_name = chunk.get("document_name", "Unknown")
        page_num = chunk.get("page_number")
        ata = chunk.get("ata_chapter")
        sec = chunk.get("section")
        chunk_id = chunk.get("chunk_id")
        parent_id = chunk.get("parent_chunk_id")
        
        citation_key = f"{doc_name}_{page_num}_{ata}_{sec}_{parent_id}"
        if citation_key not in seen_citations:
            citations.append(Citation(
                document=doc_name,
                page=int(page_num) if page_num is not None else None,
                ata_chapter=ata,
                section=sec,
                chunk_id=chunk_id,
                parent_chunk_id=parent_id,
                snippet=chunk.get("chunk_text", "")[:150] + "..."
            ))
            seen_citations.add(citation_key)
            
    # Fallback if no citations populated
    if not citations and top_5_chunks:
        chunk = top_5_chunks[0]
        citations.append(Citation(
            document=chunk.get("document_name", "Unknown"),
            page=int(chunk.get("page_number")) if chunk.get("page_number") is not None else None,
            ata_chapter=chunk.get("ata_chapter"),
            section=chunk.get("section"),
            chunk_id=chunk.get("chunk_id"),
            parent_chunk_id=chunk.get("parent_chunk_id"),
            snippet=chunk.get("chunk_text", "")[:150] + "..."
        ))

    return AskResponse(
        answer=answer,
        confidence=confidence,
        citations=citations,
        retrieved_chunks=top_5_chunks if debug else None
    )
