from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.api import app
from app.models import AskResponse, Citation


client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "running" in response.json()["message"].lower()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("app.ingest.ingest_pdf")
@patch("os.path.exists")
def test_ingest_endpoint_success(mock_exists, mock_ingest):
    mock_exists.return_value = True
    mock_ingest.return_value = 15
    
    response = client.post("/ingest", json={"pdf_path": "C:/docs/aviation_manual.pdf"})
    
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "chunks": 15
    }

def test_ingest_endpoint_missing_path():
    response = client.post("/ingest", json={"pdf_path": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_ingest_endpoint_wrong_filetype():
    response = client.post("/ingest", json={"pdf_path": "C:/docs/aviation_manual.txt"})
    assert response.status_code == 400
    assert "not a pdf" in response.json()["detail"].lower()

@patch("os.path.exists")
def test_ingest_endpoint_file_not_found(mock_exists):
    mock_exists.return_value = False
    response = client.post("/ingest", json={"pdf_path": "C:/docs/nonexistent.pdf"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

@patch("app.api.generate_rag_response")
def test_ask_endpoint_success(mock_generate):
    # Mock AskResponse payload
    mock_response = AskResponse(
        answer="The maximum takeoff weight is 79,010 kg.",
        confidence="High",
        citations=[Citation(document="manual.pdf", page=12, ata_chapter="ATA 10")],
        retrieved_chunks=[{"chunk_id": "c1", "hybrid_score": 0.8}]
    )
    mock_generate.return_value = mock_response

    response = client.post("/ask", json={"question": "What is maximum takeoff weight?", "ata_filter": "ATA 10", "debug": True})
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["answer"] == "The maximum takeoff weight is 79,010 kg."
    assert res_data["confidence"] == "High"
    assert len(res_data["citations"]) == 1
    assert res_data["citations"][0]["document"] == "manual.pdf"
    assert res_data["citations"][0]["ata_chapter"] == "ATA 10"
    assert len(res_data["retrieved_chunks"]) == 1

def test_ask_endpoint_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

@patch("app.rag.load_stores")
def test_filters_endpoint_success(mock_load):
    # Mock load_stores returning faiss, metadata, parents, bm25
    mock_load.return_value = (None, [
        {"document_name": "11-radio-navigation-2014.pdf", "section": "ADF"},
        {"document_name": "11-radio-navigation-2014.pdf", "section": "VOR"},
        {"document_name": "manual.pdf", "ata_chapter": "ATA 10"}
    ], {}, None)
    
    response = client.get("/filters")
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["has_ata"] is True
    assert len(res_data["subjects"]) == 2
    
    radio_nav = next(s for s in res_data["subjects"] if s["id"] == "11-radio-navigation-2014.pdf")
    assert radio_nav["name"] == "Radio Navigation"
    assert set(radio_nav["chapters"]) == {"ADF", "VOR"}

@patch("app.rag.load_stores")
def test_filters_endpoint_file_not_found(mock_load):
    # Mock load_stores raising FileNotFoundError
    mock_load.side_effect = FileNotFoundError("missing store")
    
    response = client.get("/filters")
    
    assert response.status_code == 200
    assert response.json() == {"has_ata": False, "subjects": []}
