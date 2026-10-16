"""Ingest one PDF into a project: Docling -> HybridChunker -> OpenAI embeddings -> Qdrant.

Usage: uv run python ingest.py <project_id> <pdf>
       uv run python ingest.py --migrate-to <new_collection>   # add keyword vectors to an existing collection
"""

import argparse
import hashlib
import re
import uuid
import zlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

CHUNK_MAX_TOKENS = 512
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
    return qdrant.count(collection, count_filter=doc_filter(project_id, sha), exact=True).count == expected


def chunk_pdf(path: Path, embed_model: str) -> list[dict]:
    import tiktoken
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    try:
        base = tiktoken.encoding_for_model(embed_model)
    except KeyError:
        base = tiktoken.get_encoding("cl100k_base")
    # Same encoding minus special tokens, so PDF text containing "<|endoftext|>" can't crash token counting.
    enc = tiktoken.Encoding(name=f"{base.name}_plain", pat_str=base._pat_str, mergeable_ranks=base._mergeable_ranks, special_tokens={})
    doc = DocumentConverter().convert(path).document
    chunker = HybridChunker(tokenizer=OpenAITokenizer(tokenizer=enc, max_tokens=CHUNK_MAX_TOKENS))

    # contextualize = headings + text: what gets embedded and shown as evidence
    return [chunk_record(chunker.contextualize(chunk), chunk.meta) for chunk in chunker.chunk(doc)]


def chunk_record(text: str, meta) -> dict:
    items = meta.doc_items
    return {
        "text": text,
        "page_numbers": sorted({p.page_no for it in items for p in it.prov}),
        "section_headings": meta.headings or [],
        "content_type": "table" if any(it.label.value == "table" for it in items) else "text",
    }


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


def ingest(project_id: str, path: Path, *, qdrant: QdrantClient, openai_client, collection: str, embed_model: str,
           source_filename: str | None = None) -> int:
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

    chunks = chunk_pdf(path, embed_model)
    if not chunks:
        raise ValueError(f"no text extracted from {path.name}")
    vectors = embed(openai_client, embed_model, [c["text"] for c in chunks])
    ensure_collection(qdrant, collection, len(vectors[0]))

    indexed_at = datetime.now(UTC).isoformat()
    points = [
        models.PointStruct(
            id=point_id(project_id, sha, i),
            vector={"": vec, SPARSE_VECTOR: sparse_vector(c["text"])},
            payload={
                **c,
                "project_id": project_id,
                "document_sha256": sha,
                "source_filename": source_filename or path.name,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "embedding_model": embed_model,
                "indexed_at": indexed_at,
            },
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors, strict=True))
    ]
    # Drop any leftovers (half-finished run, or an older version that produced more chunks).
    qdrant.delete(collection, points_selector=models.FilterSelector(filter=doc_filter(project_id, sha)), wait=True)
    for i in range(0, len(points), UPSERT_BATCH):
        qdrant.upsert(collection, points[i:i + UPSERT_BATCH], wait=True)
    return len(points)


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


def main() -> None:
    from openai import OpenAI

    from config import Settings

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_id", nargs="?")
    ap.add_argument("pdf", type=Path, nargs="?")
    ap.add_argument("--migrate-to", metavar="COLLECTION",
                    help="copy QDRANT_COLLECTION into COLLECTION with keyword vectors added, then exit")
    args = ap.parse_args()

    s = Settings()
    qdrant = QdrantClient(url=s.qdrant_cluster_endpoint, api_key=s.qdrant_api_key)
    if args.migrate_to:
        n = migrate(qdrant, s.qdrant_collection, args.migrate_to)
        print(f"copied {n} points {s.qdrant_collection} -> {args.migrate_to}; set QDRANT_COLLECTION={args.migrate_to} in .env")
        return
    if args.project_id is None or args.pdf is None:
        ap.error("project_id and pdf are required unless --migrate-to is given")
    n = ingest(
        args.project_id, args.pdf,
        qdrant=qdrant,
        openai_client=OpenAI(api_key=s.openai_api_key),
        collection=s.qdrant_collection,
        embed_model=s.openai_embed_model,
    )
    print(f"{args.pdf.name}: " + (f"indexed {n} chunks" if n else "already indexed, skipped"))


if __name__ == "__main__":
    main()
