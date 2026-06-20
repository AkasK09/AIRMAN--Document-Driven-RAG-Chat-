import json
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from app.rag import hybrid_search, rerank_chunks, generate_rag_response, REFUSAL_RESPONSE

@patch("app.rag.get_query_embedding_api")
def test_hybrid_search(mock_get_embedding):
    # Mock embedding API call
    mock_get_embedding.return_value = np.zeros((1024,))
    
    # Mock Index
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[0.8, 0.5, 0.1]]), np.array([[0, 1, 2]]))
    
    # Mock BM25
    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = [10.0, 5.0, 20.0]  # scores for index 0, 1, 2
    
    # Mock metadata
    metadata = [
        {"chunk_id": "c1", "document_name": "doc1.pdf", "ata_chapter": "ATA 10", "chunk_text": "vector match"},
        {"chunk_id": "c2", "document_name": "doc2.pdf", "ata_chapter": "ATA 20", "chunk_text": "weak vector match"},
        {"chunk_id": "c3", "document_name": "doc3.pdf", "ata_chapter": "ATA 10", "chunk_text": "bm25 match"}
    ]
    
    # Test without ATA filter
    results = hybrid_search("test query", mock_index, metadata, mock_bm25, ata_filter=None, top_k=2)
    assert len(results) == 2
    
    # Check RRF logic implicitly by ensuring c3 (high bm25) and c1 (high vector) are present
    chunk_ids = [r["chunk_id"] for r in results]
    assert "c3" in chunk_ids
    assert "c1" in chunk_ids
    
    # Test with ATA filter
    results_filtered = hybrid_search("test query", mock_index, metadata, mock_bm25, ata_filter="ATA 20", top_k=2)
    assert len(results_filtered) == 1
    assert results_filtered[0]["chunk_id"] == "c2"

    # Test with Subject Filter
    results_subject = hybrid_search("test query", mock_index, metadata, mock_bm25, subject_filter="Doc1", top_k=2)
    assert len(results_subject) == 1
    assert results_subject[0]["chunk_id"] == "c1"

    # Test with Chapter Filter
    # Add section to metadata for chapter filtering
    metadata[2]["section"] = "Some Chapter"
    results_chapter = hybrid_search("test query", mock_index, metadata, mock_bm25, chapter_filter="Some Chapter", top_k=2)
    assert len(results_chapter) == 1
    assert results_chapter[0]["chunk_id"] == "c3"

def test_rerank_chunks():
    candidates = [
        {"chunk_id": "c1", "hybrid_score": 0.5, "chunk_text": "text1"},
        {"chunk_id": "c2", "hybrid_score": 0.3, "chunk_text": "text2"},
        {"chunk_id": "c3", "hybrid_score": 0.1, "chunk_text": "text3"}
    ]
    
    reranked = rerank_chunks("query", candidates, top_k=2)
    
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c1"
    assert reranked[0]["reranker_score"] == 0.5
    assert reranked[1]["chunk_id"] == "c2"
    assert reranked[1]["reranker_score"] == 0.3

@patch("app.rag.load_stores")
@patch("app.rag.hybrid_search")
@patch("app.rag.rerank_chunks")
@patch("app.rag.call_llm")
def test_generate_rag_response_success(mock_call, mock_rerank, mock_hybrid, mock_load):
    mock_load.return_value = (MagicMock(), [], {}, MagicMock())
    
    mock_hybrid.return_value = [{"chunk_id": "c1", "parent_chunk_id": "p1", "chunk_text": "mock text"}]
    mock_rerank.return_value = [{"chunk_id": "c1", "document_name": "doc.pdf", "page_number": 1, "parent_chunk_id": "p1", "chunk_text": "mock text"}]
    
    mock_call.return_value = json.dumps({
        "answer": "This is a grounded answer.",
        "confidence": "High",
        "used_context_indices": [0]
    })
    
    res = generate_rag_response("Query", debug=True)
    
    assert res.answer == "This is a grounded answer."
    assert res.confidence == "High"
    assert len(res.citations) == 1
    assert res.citations[0].document == "doc.pdf"
    assert res.citations[0].page == 1
    assert len(res.retrieved_chunks) == 1

@patch("app.rag.load_stores")
@patch("app.rag.hybrid_search")
@patch("app.rag.rerank_chunks")
@patch("app.rag.call_llm")
def test_generate_rag_response_llm_refusal(mock_call, mock_rerank, mock_hybrid, mock_load):
    mock_load.return_value = (MagicMock(), [], {}, MagicMock())
    mock_hybrid.return_value = [{"chunk_id": "c1", "parent_chunk_id": "p1", "chunk_text": "mock text"}]
    mock_rerank.return_value = [{"chunk_id": "c1", "document_name": "doc.pdf", "page_number": 1, "parent_chunk_id": "p1", "chunk_text": "mock text"}]
    
    mock_call.return_value = json.dumps({
        "answer": REFUSAL_RESPONSE,
        "confidence": "Low",
        "used_context_indices": []
    })
    
    res = generate_rag_response("Random query")
    
    assert res.answer == REFUSAL_RESPONSE
    assert res.confidence == "Low"
    assert len(res.citations) == 0
