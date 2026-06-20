from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class IngestRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute path to the aviation PDF document on local filesystem")

class IngestResponse(BaseModel):
    status: str = Field(..., description="Status of the ingestion process (e.g., success)")
    chunks: int = Field(..., description="Number of chunks created and stored in the FAISS index")

class AskRequest(BaseModel):
    question: str = Field(..., description="Question to ask the RAG assistant")
    ata_filter: Optional[str] = Field(default=None, description="Optional ATA chapter to filter search results")
    subject_filter: Optional[str] = Field(default=None, description="Optional subject to filter search results")
    chapter_filter: Optional[str] = Field(default=None, description="Optional chapter to filter search results")
    debug: bool = Field(default=False, description="Whether to include retrieved chunks and metadata in the response")

class Citation(BaseModel):
    document: str = Field(..., description="Document name from which context was retrieved")
    page: Optional[int] = Field(default=None, description="Page number of the document (1-indexed)")
    ata_chapter: Optional[str] = Field(default=None, description="ATA chapter reference")
    section: Optional[str] = Field(default=None, description="Document section or chapter")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID used as fallback when page number is missing")
    parent_chunk_id: Optional[str] = Field(default=None, description="Parent chunk ID")
    snippet: Optional[str] = Field(default=None, description="Snippet of text from the chunk used as fallback")

class AskResponse(BaseModel):
    answer: str = Field(..., description="LLM generated answer based ONLY on the retrieved context")
    confidence: Optional[str] = Field(default="Unknown", description="Confidence level of the answer (High/Medium/Low)")
    citations: List[Citation] = Field(..., description="List of source citations used to construct the answer")
    retrieved_chunks: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed info about retrieved chunks, hidden if debug is False")
