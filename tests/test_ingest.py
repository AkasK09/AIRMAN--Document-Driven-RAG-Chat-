import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from app.ingest import extract_text_from_pdf, build_hierarchical_chunks, ingest_pdf

@patch("app.ingest.fitz.open")
@patch("app.ingest.os.path.exists", return_value=True)
def test_extract_text_from_pdf_with_metadata(mock_exists, mock_fitz_open):
    mock_doc = MagicMock()
    mock_page_1 = MagicMock()
    mock_page_1.get_text.return_value = "CHAPTER 5\nATA 21-00\nAir Conditioning System"
    mock_page_2 = MagicMock()
    mock_page_2.get_text.return_value = "Page 2 Content without ATA"
    
    mock_doc.__len__.return_value = 2
    mock_doc.__getitem__.side_effect = [mock_page_1, mock_page_2]
    mock_fitz_open.return_value = mock_doc
    
    pages = extract_text_from_pdf("dummy.pdf")
    
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[0]["ata_chapter"] == "ATA 21-00"
    assert pages[0]["section"] == "Chapter 5"
    
    assert pages[1]["page_number"] == 2
    assert pages[1]["ata_chapter"] is None
    assert pages[1]["section"] is None

@patch("app.ingest.settings")
def test_build_hierarchical_chunks(mock_settings):
    mock_settings.PARENT_CHUNK_SIZE = 50
    mock_settings.CHILD_CHUNK_SIZE = 10
    
    pages_data = [
        {"page_number": 1, "text": "This is a short sentence. " * 10, "ata_chapter": "ATA 10", "section": "Chapter 1"},
        {"page_number": 2, "text": "Another page.", "ata_chapter": "ATA 10", "section": "Chapter 1"}
    ]
    
    child_chunks, parent_mapping = build_hierarchical_chunks(pages_data, "test.pdf")
    
    assert len(child_chunks) > 0
    assert len(parent_mapping) > 0
    
    assert child_chunks[0]["document_name"] == "test.pdf"
    assert child_chunks[0]["page_number"] == 1
    assert child_chunks[0]["ata_chapter"] == "ATA 10"
    assert "parent_chunk_id" in child_chunks[0]
    
    parent_id = child_chunks[0]["parent_chunk_id"]
    assert parent_id in parent_mapping
    assert len(parent_mapping[parent_id]) > len(child_chunks[0]["chunk_text"])

@patch("app.ingest.get_embedding_model")
@patch("app.ingest.load_or_create_vector_store")
@patch("app.ingest.save_vector_store")
@patch("app.ingest.extract_text_from_pdf")
@patch("app.ingest.build_hierarchical_chunks")
def test_ingest_pdf(mock_build, mock_extract, mock_save, mock_load, mock_get_model):
    mock_extract.return_value = []
    
    mock_build.return_value = (
        [{"chunk_text": "child 1", "chunk_id": "c1"}],
        {"p1": "parent text 1"}
    )
    
    mock_index = MagicMock()
    mock_index.ntotal = 0
    mock_load.return_value = (mock_index, [], {}, None)
    
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 1024))
    mock_get_model.return_value = mock_model
    
    num_chunks = ingest_pdf("test.pdf")
    
    assert num_chunks == 1
    mock_index.add.assert_called_once()
    mock_save.assert_called_once()
    # Check that save_vector_store was called with an instantiated BM25 object
    args, kwargs = mock_save.call_args
    assert args[3] is not None  # bm25 shouldn't be None
