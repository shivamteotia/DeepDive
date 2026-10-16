"""Ingest one PDF into a project: Docling -> HybridChunker -> OpenAI embeddings -> Qdrant.

Usage: uv run python ingest.py <project_id> <pdf>
       uv run python ingest.py --migrate-to <new_collection>   # add keyword vectors to an existing collection
       uv run python ingest.py --reindex-to <new_collection>   # re-chunk every ready document (smart chunking)
       uv run python ingest.py --sync-counts                   # after switching QDRANT_COLLECTION: refresh Postgres counts
"""

import argparse
import hashlib
import re
import uuid
import zlib
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import NamedTuple

from qdrant_client import QdrantClient, models

import chunking

EMBED_BATCH = 100
UPSERT_BATCH = 64
MIGRATE_BATCH = UPSERT_BATCH  # ~1 MB per Cloud upsert at 1536 dims; 256 risked the 5 s client timeout
SPARSE_VECTOR = "bm25"
BM25_K1 = 1.2
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./,][a-z0-9]+)*")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def point_id(project_id: str, sha: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"maxrim:{project_id}:{sha}:{chunk_index}"))


def doc_filter(project_id: str, sha: str) -> models.Filter:
    return models.Filter(must=[
        models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id)),
        models.FieldCondition(key="document_sha256", match=models.MatchValue(value=sha)),
    ])


def already_indexed(qdrant: QdrantClient, collection: str, project_id: str, sha: str) -> bool:
    """True only if every chunk of this document is present (a half-finished upsert doesn't count)."""
    if not qdrant.collection_exists(collection):
        return False
    points, _ = qdrant.scroll(collection, scroll_filter=doc_filter(project_id, sha), limit=1)
    if not points:
        return False
    expected = points[0].payload["chunk_count"]
    current = models.Filter(must=[*doc_filter(project_id, sha).must, models.FieldCondition(
        key="chunker_version", match=models.MatchValue(value=chunking.CHUNKER_VERSION))])
    return qdrant.count(collection, count_filter=current, exact=True).count == expected


class ParsedDoc(NamedTuple):
    pieces: list[chunking.Piece]
    title: str
    page_count: int


def _plain_encoding(embed_model: str):
    import tiktoken

    try:
        base = tiktoken.encoding_for_model(embed_model)
    except KeyError:
        base = tiktoken.get_encoding("cl100k_base")
    # Same encoding minus special tokens, so PDF text containing "<|endoftext|>" can't crash token counting.
    return tiktoken.Encoding(name=f"{base.name}_plain", pat_str=base._pat_str, mergeable_ranks=base._mergeable_ranks,
                             special_tokens={})


def token_counter(embed_model: str) -> chunking.CountTokens:
    enc = _plain_encoding(embed_model)
    return lambda s: len(enc.encode(s))


def _pdf_meta_title(path: Path) -> str:
    try:
        import pypdfium2 as pdfium

        return (pdfium.PdfDocument(str(path)).get_metadata_dict().get("Title") or "").strip()
    except Exception:  # noqa: BLE001 — metadata is a best-effort title fallback
        return ""


def piece_from_chunk(c, title: str = "") -> chunking.Piece:
    """A Docling chunk as a Piece: its own text (headings kept apart), pages and whether it holds a table.
    Docling keeps the document title at level 0 of every heading path; it is dropped (the label carries it)."""
    items = c.meta.doc_items
    headings = list(c.meta.headings or [])
    if title and headings and headings[0].strip() == title.strip():
        headings = headings[1:]
    return chunking.Piece(text=c.text, headings=headings,
                          page_numbers=sorted({p.page_no for it in items for p in it.prov}),
                          content_type="table" if any(it.label.value == "table" for it in items) else "text")


def parse_pdf(path: Path, filename: str, embed_model: str) -> ParsedDoc:
    """Docling parse + HybridChunker pieces (text without headings), the title and the page count."""
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    doc = DocumentConverter().convert(path).document
    chunker = HybridChunker(tokenizer=OpenAITokenizer(tokenizer=_plain_encoding(embed_model),
                                                      max_tokens=chunking.CHUNK_MAX_TOKENS))
    docling_title = next((t.text for t in doc.texts if t.label.value == "title"), "")
    pieces = [piece_from_chunk(c, docling_title) for c in chunker.chunk(doc)]
    return ParsedDoc(pieces, chunking.resolve_title(docling_title, _pdf_meta_title(path), filename), doc.num_pages())


def llm_parser(openai_client, model: str) -> chunking.Parse:
    from ask import parse_llm  # lazy: ask imports ingest

    return partial(parse_llm, openai_client, model)


def embed_text(chunk: dict) -> str:
    """What gets embedded and keyword-indexed: the evidence text plus the search-only context."""
    return chunk["text"] + ("\n" + chunk["context"] if chunk["context"] else "")


def embed(openai_client, model: str, texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = openai_client.embeddings.create(model=model, input=texts[i:i + EMBED_BATCH])
        vectors.extend(d.embedding for d in resp.data)
    return vectors


def sparse_vector(text: str, *, query: bool = False) -> models.SparseVector:
    """Keyword vector for the IDF-weighted sparse index: BM25 term-frequency saturation with b=0 (chunks are
    near-uniform in length); Qdrant applies IDF. Query terms weigh 1.0. Colliding crc32 indices are summed."""
    weights: dict[int, float] = {}
    for token, tf in Counter(TOKEN_RE.findall(text.lower())).items():
        i = zlib.crc32(token.encode())
        weights[i] = weights.get(i, 0.0) + (1.0 if query else tf * (BM25_K1 + 1) / (tf + BM25_K1))
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


MIGRATE_HINT = "run `uv run python ingest.py --migrate-to <new_collection>` and set QDRANT_COLLECTION to it"


def has_sparse(qdrant: QdrantClient, collection: str) -> bool:
    return SPARSE_VECTOR in (qdrant.get_collection(collection).config.params.sparse_vectors or {})


def ensure_collection(qdrant: QdrantClient, collection: str, dim: int) -> None:
    if not qdrant.collection_exists(collection):
        qdrant.create_collection(
            collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    elif not has_sparse(qdrant, collection):
        raise ValueError(f"collection {collection!r} has no {SPARSE_VECTOR!r} keyword vector; {MIGRATE_HINT}")
    for field in ("project_id", "document_sha256"):
        qdrant.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD)
    # Qdrant Cloud may refuse filters on unindexed fields; already_indexed filters on the chunker version.
    qdrant.create_payload_index(collection, "chunker_version", models.PayloadSchemaType.INTEGER)


def ingest(project_id: str, path: Path, *, qdrant: QdrantClient, openai_client, collection: str, embed_model: str,
           source_filename: str | None = None, context_model: str = "", contextual: bool = True) -> int:
    """Returns number of points written; 0 means the document was already fully indexed."""
    if not project_id.strip():
        raise ValueError("project_id is required")
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError(f"not a PDF file: {path}")

    if qdrant.collection_exists(collection) and not has_sparse(qdrant, collection):  # before Docling and paid embeddings
        raise ValueError(f"collection {collection!r} has no {SPARSE_VECTOR!r} keyword vector; {MIGRATE_HINT}")
    sha = sha256_file(path)
    if already_indexed(qdrant, collection, project_id, sha):
        return 0

    name = source_filename or path.name
    prepared = prepare_document(path, source_filename=name, openai_client=openai_client, embed_model=embed_model,
                                context_model=context_model, contextual=contextual)
    return write_document(qdrant, collection, project_id=project_id, sha=sha, source_filename=name,
                          embed_model=embed_model, prepared=prepared)


class Prepared(NamedTuple):
    doc_type: str
    chunks: list[dict]
    vectors: list[list[float]]


def prepare_document(path: Path, *, source_filename: str, openai_client, embed_model: str, context_model: str = "",
                     contextual: bool = True) -> Prepared:
    parsed = parse_pdf(path, source_filename, embed_model)
    parse = llm_parser(openai_client, context_model) if context_model else None
    doc_type, chunks = chunking.build_chunks(parsed.pieces, page_count=parsed.page_count, title=parsed.title,
                                             count_tokens=token_counter(embed_model), parse=parse, contextual=contextual)
    if not chunks:
        raise ValueError(f"no text extracted from {path.name}")
    return Prepared(doc_type, chunks, embed(openai_client, embed_model, [embed_text(c) for c in chunks]))


def write_document(qdrant: QdrantClient, collection: str, *, project_id: str, sha: str, source_filename: str,
                   embed_model: str, prepared: Prepared) -> int:
    ensure_collection(qdrant, collection, len(prepared.vectors[0]))
    indexed_at = datetime.now(UTC).isoformat()
    n = len(prepared.chunks)
    points = [
        models.PointStruct(
            id=point_id(project_id, sha, i),
            vector={"": vec, SPARSE_VECTOR: sparse_vector(embed_text(c))},
            payload={**c, "project_id": project_id, "document_sha256": sha, "source_filename": source_filename,
                     "chunk_index": i, "chunk_count": n, "embedding_model": embed_model, "indexed_at": indexed_at},
        )
        for i, (c, vec) in enumerate(zip(prepared.chunks, prepared.vectors, strict=True))
    ]
    # Drop any leftovers (half-finished run, or an older version that produced more chunks).
    qdrant.delete(collection, points_selector=models.FilterSelector(filter=doc_filter(project_id, sha)), wait=True)
    for i in range(0, len(points), UPSERT_BATCH):
        qdrant.upsert(collection, points[i:i + UPSERT_BATCH], wait=True)
    return n


def migrate(qdrant: QdrantClient, src: str, dst: str) -> int:
    """Copy every point of src into dst with the same id, payload and dense vector, adding the bm25 keyword vector.
    No Docling, no OpenAI; src is only read. Safe to re-run (same ids overwrite)."""
    if src == dst:
        raise ValueError("source and destination collections must differ")
    ensure_collection(qdrant, dst, qdrant.get_collection(src).config.params.vectors.size)
    copied, offset = 0, None
    while True:
        points, offset = qdrant.scroll(src, limit=MIGRATE_BATCH, offset=offset, with_payload=True, with_vectors=True)
        if points:
            qdrant.upsert(dst, [
                models.PointStruct(id=p.id, payload=p.payload, vector={
                    "": p.vector[""] if isinstance(p.vector, dict) else p.vector,
                    SPARSE_VECTOR: sparse_vector(p.payload.get("text", "")),
                })
                for p in points
            ], wait=True)
            copied += len(points)
        if offset is None:
            break
    src_n, dst_n = qdrant.count(src, exact=True).count, qdrant.count(dst, exact=True).count
    if dst_n < src_n:
        raise RuntimeError(f"migration incomplete: {dst} has {dst_n} points, {src} has {src_n}")
    return copied


def prepare_in_child(path: str, filename: str, embed_model: str, context_model: str, contextual: bool) -> Prepared:
    """prepare_document inside a jobs.IngestRunner child with its own OpenAI client, so a native Docling crash
    (Li2024 did) fails one document instead of the whole re-index."""
    from openai import OpenAI

    from config import Settings

    return prepare_document(Path(path), source_filename=filename, openai_client=OpenAI(api_key=Settings().openai_api_key),
                            embed_model=embed_model, context_model=context_model, contextual=contextual)


def reindex(*, qdrant: QdrantClient, openai_client, sessions, src: str, dst: str, upload_dir: Path, embed_model: str,
            context_model: str = "", contextual: bool = True, prepare=None) -> dict:
    """Re-chunk every ready document from its stored PDF into dst. Each file is parsed, contextualised and embedded
    once and written to every project that has it. src (the live collection) and Postgres are never touched (the API
    keeps serving src until QDRANT_COLLECTION is switched; then run --sync-counts). Documents already complete in dst
    are skipped, so a re-run after the switch only does new uploads. prepare(path, filename) -> Prepared defaults to
    an in-process prepare_document."""
    from sqlalchemy import select

    import db

    if src == dst:
        raise ValueError("destination collection must differ from the live collection")
    if prepare is None:
        def prepare(path: Path, filename: str) -> Prepared:
            return prepare_document(path, source_filename=filename, openai_client=openai_client, embed_model=embed_model,
                                    context_model=context_model, contextual=contextual)
    with sessions() as s:
        rows = [(str(d.project_id), d.sha256, d.filename)
                for d in s.scalars(select(db.Document).where(db.Document.status == "ready").order_by(db.Document.created_at))]
    cache: dict[str, Prepared | Exception] = {}
    out = {"documents": 0, "points": 0, "already": 0, "skipped": [], "failed": []}
    for project_id, sha, filename in rows:
        path = upload_dir / project_id / f"{sha}.pdf"
        if not path.is_file():
            out["skipped"].append(filename)
            continue
        if already_indexed(qdrant, dst, project_id, sha):
            out["already"] += 1
            continue
        if sha not in cache:
            try:
                cache[sha] = prepare(path, filename)
            except Exception as exc:  # noqa: BLE001 — one bad PDF must not stop the rest
                cache[sha] = exc
        prepared = cache[sha]
        if isinstance(prepared, Exception):
            out["failed"].append(f"{filename}: {type(prepared).__name__}: {prepared}")
            continue
        out["points"] += write_document(qdrant, dst, project_id=project_id, sha=sha, source_filename=filename,
                                        embed_model=embed_model, prepared=prepared)
        out["documents"] += 1
    return out


def sync_counts(qdrant: QdrantClient, sessions, collection: str) -> int:
    """After switching QDRANT_COLLECTION: set each ready document's chunk_count and doc_type from that collection."""
    from sqlalchemy import select, update

    import db

    with sessions() as s:
        rows = [(d.id, str(d.project_id), d.sha256) for d in s.scalars(select(db.Document).where(db.Document.status == "ready"))]
    updated = 0
    for doc_id, project_id, sha in rows:
        f = doc_filter(project_id, sha)
        first, _ = qdrant.scroll(collection, scroll_filter=f, limit=1, with_payload=["doc_type"])
        if not first:
            continue
        n = qdrant.count(collection, count_filter=f, exact=True).count
        with sessions() as s:
            s.execute(update(db.Document).where(db.Document.id == doc_id).values(chunk_count=n,
                                                                              doc_type=first[0].payload.get("doc_type")))
            s.commit()
        updated += 1
    return updated


def main() -> None:
    from openai import OpenAI

    from config import Settings

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_id", nargs="?")
    ap.add_argument("pdf", type=Path, nargs="?")
    ap.add_argument("--migrate-to", metavar="COLLECTION",
                    help="copy QDRANT_COLLECTION into COLLECTION with keyword vectors added, then exit")
    ap.add_argument("--reindex-to", metavar="COLLECTION", help="re-chunk every ready document into COLLECTION, then exit")
    ap.add_argument("--sync-counts", action="store_true", help="update Postgres chunk counts/types from QDRANT_COLLECTION")
    args = ap.parse_args()

    s = Settings()
    qdrant = QdrantClient(url=s.qdrant_cluster_endpoint, api_key=s.qdrant_api_key)
    if args.migrate_to:
        n = migrate(qdrant, s.qdrant_collection, args.migrate_to)
        print(f"copied {n} points {s.qdrant_collection} -> {args.migrate_to}; set QDRANT_COLLECTION={args.migrate_to} in .env")
        return
    if args.sync_counts:
        import db

        print(f"updated {sync_counts(qdrant, db.make_sessions(s.database_url), s.qdrant_collection)} documents from QDRANT_COLLECTION")
        return
    if args.reindex_to:
        import db
        import jobs

        ctx_model = s.openai_context_model or s.openai_fast_model
        runner = jobs.IngestRunner(fn=prepare_in_child)
        try:
            out = reindex(qdrant=qdrant, openai_client=None, sessions=db.make_sessions(s.database_url),
                          src=s.qdrant_collection, dst=args.reindex_to, upload_dir=Path(__file__).parent / "data" / "uploads",
                          embed_model=s.openai_embed_model, prepare=lambda path, fn: runner.run(
                              str(path), fn, s.openai_embed_model, ctx_model, s.contextual_chunks))
        finally:
            runner.shutdown()
        print(f"re-indexed {out['documents']} documents ({out['points']} points, {out['already']} already done) "
              f"into {args.reindex_to}; after switching QDRANT_COLLECTION run: uv run python ingest.py --sync-counts")
        for fn in out["skipped"]:
            print(f"  skipped (no stored PDF): {fn}")
        for line in out["failed"]:
            print(f"  failed: {line}")
        return
    if args.project_id is None or args.pdf is None:
        ap.error("project_id and pdf are required unless --migrate-to, --reindex-to or --sync-counts is given")
    n = ingest(
        args.project_id, args.pdf,
        qdrant=qdrant,
        openai_client=OpenAI(api_key=s.openai_api_key),
        collection=s.qdrant_collection,
        embed_model=s.openai_embed_model,
        context_model=s.openai_context_model or s.openai_fast_model,
        contextual=s.contextual_chunks,
    )
    print(f"{args.pdf.name}: " + (f"indexed {n} chunks" if n else "already indexed, skipped"))


if __name__ == "__main__":
    main()
