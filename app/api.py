import os
import time
import re
from typing import List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils import setup_logger, format_subject_name
from app.models import IngestRequest, IngestResponse, AskRequest, AskResponse
from app.ingest import ingest_pdf
from app.rag import generate_rag_response

logger = setup_logger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Aviation Document AI Assistant",
    description="A production-ready RAG assistant to answer questions ONLY from uploaded aviation PDFs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware to log API requests and their durations
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    logger.info(f"Incoming Request: {method} {path}")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Completed Request: {method} {path} - Status: {response.status_code} - Duration: {process_time:.2f}ms")
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"Failed Request: {method} {path} - Duration: {process_time:.2f}ms - Error: {str(e)}")
        # Re-raise to let the global exception handler deal with it
        raise e

# Global Exception Handlers for custom responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail} (Status Code: {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the administrator."}
    )

@app.get("/health", tags=["Monitoring"], response_model=dict)
async def health_check():
    """
    Health check endpoint to verify system status.
    """
    logger.info("Health check endpoint hit")
    return {"status": "healthy"}

@app.post("/ingest", tags=["Ingestion"], response_model=IngestResponse)
async def ingest_document(payload: IngestRequest):
    """
    Load a PDF document, split it into chunks, embed the chunks, and save to FAISS index.
    """
    pdf_path = payload.pdf_path.strip()
    
    if not pdf_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF path cannot be empty."
        )

    if not pdf_path.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file provided is not a PDF."
        )
        
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found at the specified path: {pdf_path}"
        )

    logger.info(f"API request to ingest PDF: {pdf_path}")
    
    try:
        num_chunks = ingest_pdf(pdf_path)
        return IngestResponse(
            status="success",
            chunks=num_chunks
        )
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf)
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as e:
        logger.error(f"Error during ingestion API call: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/ask", tags=["RAG Query"], response_model=AskResponse)
async def ask_question(payload: AskRequest):
    """
    Query the aviation assistant. Returns answer and citations.
    """
    question = payload.question.strip()
    
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )
        
    logger.info(f"API request with question: '{question}' (debug={payload.debug})")
    
    try:
        response = generate_rag_response(
            question,
            ata_filter=payload.ata_filter,
            subject_filter=payload.subject_filter,
            chapter_filter=payload.chapter_filter,
            debug=payload.debug
        )
        return response
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Vector index is not initialized. Please ingest a document first. Details: {str(fnf)}"
        )
    except Exception as e:
        logger.error(f"Error during ask API call: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )

@app.get("/documents", tags=["Metadata"], response_model=List[str])
async def get_documents():
    """
    Get the list of unique document names currently indexed in the vector store.
    """
    logger.info("API request to list indexed documents")
    try:
        from app.rag import load_stores
        _, metadata, _, _ = load_stores()
        unique_docs = sorted(list(set(chunk["document_name"] for chunk in metadata)))
        return unique_docs
    except FileNotFoundError:
        # Return empty list if vector store files don't exist yet
        return []
    except Exception as e:
        logger.error(f"Error retrieving indexed documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve indexed documents: {str(e)}"
        )

@app.get("/filters", tags=["Metadata"], response_model=dict)
async def get_filters():
    """
    Get the dynamic hierarchy of subjects and chapters, and detect if ATA applies.
    """
    logger.info("API request to fetch metadata filters")
    try:
        from app.rag import load_stores
        _, metadata, _, _ = load_stores()
        
        ata_count = 0
        subjects_dict = {}
        
        for chunk in metadata:
            if chunk.get("ata_chapter"):
                ata_count += 1
                
            doc_id = chunk.get("document_name")
            if not doc_id:
                continue
                
            subject_name = format_subject_name(doc_id)
            section = chunk.get("section")
            
            if doc_id not in subjects_dict:
                subjects_dict[doc_id] = {
                    "id": doc_id,
                    "name": subject_name,
                    "chapters": set()
                }
                
            if section:
                subjects_dict[doc_id]["chapters"].add(section)
                
        # Robust ATA detection: Require at least 200 chunks or 5% of corpus to have ATA tags
        # to avoid false positives from stray text in training books.
        has_ata = ata_count > 200 or (len(metadata) > 0 and ata_count / len(metadata) > 0.05)
                
        # Format response
        subjects_list = []
        for doc_id, data in subjects_dict.items():
            subjects_list.append({
                "id": data["id"],
                "name": data["name"],
                "chapters": sorted(list(data["chapters"]))
            })
            
        subjects_list = sorted(subjects_list, key=lambda x: x["name"])
        
        return {
            "has_ata": has_ata,
            "subjects": subjects_list
        }
        
    except FileNotFoundError:
        return {"has_ata": False, "subjects": []}
    except Exception as e:
        logger.error(f"Error retrieving filters: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve filters: {str(e)}"
        )
