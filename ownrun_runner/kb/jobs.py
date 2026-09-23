from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from ownrun_runner.client import ServerClient
from ownrun_runner.config import Config
from ownrun_runner.kb.store import (
    SUPPORTED_FORMATS,
    KbVectorStore,
    chunk_sections,
    embed_texts,
    parse_document,
)

log = logging.getLogger(__name__)


def _empty_extract_message(fmt: str) -> str:
    if fmt == "md":
        return "markdown file has no extractable text"
    if fmt == "txt":
        return "text file is empty"
    return "no extractable text (OCR not supported in v1)"


def _store(config: Config) -> KbVectorStore:
    return KbVectorStore(Path(config.kb_dir), config.kb_embed_model)


def run_kb_ingest(
    config: Config,
    request: dict[str, Any],
    *,
    job_id: int,
    client: ServerClient,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    doc_id = int(request.get("document_id") or 0)
    kb_id = int(request.get("kb_id") or 0)
    fmt = str(request.get("format") or "").lower()
    embed_model = str(request.get("embedding_model") or config.kb_embed_model).strip()
    if not doc_id or not kb_id or fmt not in SUPPORTED_FORMATS:
        raise RuntimeError("Invalid kb.ingest request payload")

    def cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    client.kb_document_status(doc_id, {"status": "running", "error_message": None})

    raw = client.kb_download_document(doc_id)
    with tempfile.TemporaryDirectory(prefix="ownrun-kb-") as tmp:
        path = Path(tmp) / f"doc.{fmt}"
        path.write_bytes(raw)
        if cancelled():
            raise RuntimeError("cancelled")

        sections = parse_document(path, fmt)
        if not sections:
            raise RuntimeError(_empty_extract_message(fmt))

        chunks = chunk_sections(
            sections,
            chunk_size=config.kb_chunk_size,
            overlap=config.kb_chunk_overlap,
            max_chunks=config.kb_max_chunks_per_doc,
        )
        if not chunks:
            raise RuntimeError(_empty_extract_message(fmt))

        page_count = max((s.page or 0 for s in sections), default=0) or None
        chapter_count = len({s.chapter for s in sections if s.chapter}) or None

        def prog(done: int, total: int) -> None:
            if on_progress:
                on_progress(
                    {
                        "document_id": doc_id,
                        "chunks_embedded": done,
                        "chunks_total": total,
                        "status": "running",
                    }
                )

        if cancelled():
            raise RuntimeError("cancelled")

        try:
            embeddings = embed_texts(
                config.ollama_base_url,
                embed_model,
                [c.text for c in chunks],
                on_progress=prog,
            )
        except Exception as e:
            raise RuntimeError(
                f"Embedding failed ({embed_model}): {e}. "
                f"Install with: ollama pull {embed_model}"
            ) from e

        # Resolve kb slug / project / filename from agent-visible metadata isn't available
        # for ingest jobs; pull from status update response after we set counts, or use
        # a lightweight path: store with placeholders then... Better: include in download
        # headers? For now fetch via a status GET isn't there. Use request + defaults.
        filename = str(request.get("filename") or f"document.{fmt}")
        kb_slug = str(request.get("kb_slug") or f"kb-{kb_id}")
        project = request.get("project")
        project_s = str(project) if project else None

        # Prefer values returned from content endpoint? Client may parse Content-Disposition.
        # Re-read after status — include meta on status response by fetching document via
        # a new client helper that uses the purge/status path. Simpler: pass from server
        # in job payload. Update server enqueueIngest to include filename/slug/project.

        store = _store(config)
        try:
            store.upsert_document(
                document_id=doc_id,
                kb_id=kb_id,
                kb_slug=kb_slug,
                project=project_s,
                filename=filename,
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=embed_model,
            )
        finally:
            store.close()

    client.kb_document_status(
        doc_id,
        {
            "status": "ready",
            "error_message": None,
            "chunk_count": len(chunks),
            "page_count": page_count,
            "chapter_count": chapter_count,
            "embedding_model": embed_model,
        },
    )

    duration_ms = int((time.monotonic() - started) * 1000)
    return {
        "text": f"Ingested document {doc_id}: {len(chunks)} chunks",
        "document_id": doc_id,
        "kb_id": kb_id,
        "chunk_count": len(chunks),
        "page_count": page_count,
        "chapter_count": chapter_count,
        "embedding_model": embed_model,
        "duration_ms": duration_ms,
    }


def run_kb_delete(
    config: Config,
    request: dict[str, Any],
    *,
    job_id: int,
    client: ServerClient,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    store = _store(config)
    removed = 0
    try:
        if request.get("cascade") and request.get("kb_id"):
            kb_id = int(request["kb_id"])
            doc_ids = request.get("document_ids") or []
            if isinstance(doc_ids, list) and doc_ids:
                for raw_id in doc_ids:
                    doc_id = int(raw_id)
                    removed += store.delete_document(doc_id)
                    client.kb_purge_document(doc_id)
            else:
                removed += store.delete_kb(kb_id)
            client.kb_purge_base(kb_id)
        else:
            doc_id = int(request.get("document_id") or 0)
            if not doc_id:
                raise RuntimeError("Invalid kb.delete request payload")
            removed += store.delete_document(doc_id)
            client.kb_purge_document(doc_id)
    finally:
        store.close()

    return {
        "text": f"Deleted vectors ({removed} chunks)",
        "removed_chunks": removed,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
