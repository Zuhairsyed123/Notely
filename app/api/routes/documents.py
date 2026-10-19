"""Document management routes: upload, paste, retrieval, chunk inspection, and deletion."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.core.exceptions import (
    CorruptDocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidChunkConfigError,
    ScannedPDFError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.schemas import (
    ChunkConfigRequest,
    ChunkItemResponse,
    DocumentChunksResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    PasteNotesRequest,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.services.chunker import DocumentChunker
from app.services.document_loader import DocumentLoader
from app.services.document_store import get_document_store
from app.services.retriever import SemanticRetriever
from app.services.vector_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentSummary, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=settings.notely_default_chunk_size),
    chunk_overlap: int = Form(default=settings.notely_default_chunk_overlap),
) -> DocumentSummary:
    """Upload and process a PDF or TXT document, extract pages, chunk, and index."""
    doc_store = get_document_store()
    vector_store = get_vector_store()

    try:
        content = await file.read()
        existing_hashes = doc_store.get_existing_hashes()

        # Load and parse document
        document = DocumentLoader.load_from_bytes(
            filename=file.filename or "uploaded_file",
            content=content,
            existing_hashes=existing_hashes,
        )

        # Chunk document
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(document)

        # Index chunks in vector store
        indexed_count = vector_store.add_chunks(chunks)

        # Update document state
        document.chunk_count = indexed_count
        document.status = "indexed"
        doc_store.save_document(document)

        return DocumentSummary(
            id=document.id,
            name=document.name,
            file_type=document.file_type,
            file_size_bytes=document.file_size_bytes,
            page_count=document.page_count,
            chunk_count=document.chunk_count,
            status=document.status,
            created_at=document.created_at,
        )

    except ScannedPDFError as e:
        logger.warning("Scanned PDF error: %s", str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except (EmptyDocumentError, UnsupportedFileTypeError, InvalidChunkConfigError) as e:
        logger.warning("Validation error on upload: %s", str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except (FileTooLargeError, CorruptDocumentError) as e:
        logger.warning("Document processing error: %s", str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error processing document upload: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the document: {e}",
        )


@router.post("/paste", response_model=DocumentSummary, status_code=status.HTTP_201_CREATED)
def paste_notes(
    request: PasteNotesRequest,
    chunk_size: int = settings.notely_default_chunk_size,
    chunk_overlap: int = settings.notely_default_chunk_overlap,
) -> DocumentSummary:
    """Ingest pasted raw text notes, chunk, and index them into the vector store."""
    doc_store = get_document_store()
    vector_store = get_vector_store()

    try:
        document = DocumentLoader.load_from_pasted_text(
            title=request.title,
            text=request.text,
            existing_hashes=doc_store.get_existing_hashes(),
        )

        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(document)

        indexed_count = vector_store.add_chunks(chunks)
        document.chunk_count = indexed_count
        document.status = "indexed"
        doc_store.save_document(document)

        return DocumentSummary(
            id=document.id,
            name=document.name,
            file_type=document.file_type,
            file_size_bytes=document.file_size_bytes,
            page_count=document.page_count,
            chunk_count=document.chunk_count,
            status=document.status,
            created_at=document.created_at,
        )

    except (EmptyDocumentError, InvalidChunkConfigError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error processing pasted notes: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process pasted notes: {e}",
        )


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """List all stored documents and their current indexing status."""
    doc_store = get_document_store()
    docs = doc_store.list_documents()
    summaries = [
        DocumentSummary(
            id=d.id,
            name=d.name,
            file_type=d.file_type,
            file_size_bytes=d.file_size_bytes,
            page_count=d.page_count,
            chunk_count=d.chunk_count,
            status=d.status,
            created_at=d.created_at,
            error_message=d.error_message,
        )
        for d in docs
    ]
    return DocumentListResponse(total_documents=len(summaries), documents=summaries)


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document_detail(document_id: str) -> DocumentDetailResponse:
    """Retrieve full details of a specific document including page previews."""
    doc_store = get_document_store()
    try:
        doc = doc_store.get_document(document_id)
        summary = DocumentSummary(
            id=doc.id,
            name=doc.name,
            file_type=doc.file_type,
            file_size_bytes=doc.file_size_bytes,
            page_count=doc.page_count,
            chunk_count=doc.chunk_count,
            status=doc.status,
            created_at=doc.created_at,
            error_message=doc.error_message,
        )
        pages_data = [
            {
                "page_number": p.page_number,
                "char_count": p.char_count,
                "preview": p.cleaned_text[:300] + ("..." if len(p.cleaned_text) > 300 else ""),
            }
            for p in doc.pages
        ]
        return DocumentDetailResponse(
            document=summary,
            full_text_preview=doc.get_full_text()[:1000],
            pages=pages_data,
        )
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document, its metadata, and all associated vector chunks from ChromaDB."""
    doc_store = get_document_store()
    vector_store = get_vector_store()
    try:
        doc_store.delete_document(document_id, vector_store=vector_store)
        return {"status": "success", "message": f"Document '{document_id}' successfully deleted."}
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/{document_id}/chunks", response_model=DocumentChunksResponse)
def inspect_or_rechunk_document(
    document_id: str,
    config: ChunkConfigRequest,
    reindex: bool = False,
) -> DocumentChunksResponse:
    """Chunk Explorer endpoint: inspect or simulate chunking with custom settings, or re-index."""
    doc_store = get_document_store()
    vector_store = get_vector_store()

    try:
        doc = doc_store.get_document(document_id)
        chunker = DocumentChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = chunker.chunk_document(doc)

        if reindex:
            # Delete old chunks and re-embed new chunks
            vector_store.delete_document_chunks(document_id)
            vector_store.add_chunks(chunks)
            doc.chunk_count = len(chunks)
            doc_store.save_document(doc)

        chunk_items = [
            ChunkItemResponse(
                chunk_id=c.id,
                document_id=c.document_id,
                document_name=c.document_name,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                text=c.text,
                char_count=c.char_count,
                start_char=c.start_char,
                end_char=c.end_char,
            )
            for c in chunks
        ]

        return DocumentChunksResponse(
            document_id=doc.id,
            document_name=doc.name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            total_chunks=len(chunk_items),
            chunks=chunk_items,
        )
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except InvalidChunkConfigError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# Dedicated Search endpoint
search_router = APIRouter(tags=["Search"])


@search_router.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Perform cosine-similarity semantic search across indexed chunks."""
    retriever = SemanticRetriever()
    results = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    items = [
        SearchResultItem(
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("document_id", ""),
            document_name=r.get("document_name", ""),
            page_number=r.get("page_number", 1),
            chunk_index=r.get("chunk_index", 0),
            text=r.get("text", ""),
            score=r.get("score", 0.0),
        )
        for r in results
    ]
    return SearchResponse(
        query=request.query,
        total_results=len(items),
        results=items,
    )
