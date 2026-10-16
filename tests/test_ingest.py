import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

import chunking
import db
import ingest

COLL = "unit"
FAKE_PIECES = [chunking.Piece(text=f"chunk {i} " + "word " * 60, headings=[h], page_numbers=[i + 1])
               for i, h in enumerate(["Methods", "Results", "Discussion"])]


def fake_parse_pdf(pieces):
    return lambda path, filename, model: ingest.ParsedDoc(pieces=pieces, title="Fake paper", page_count=5)


class FakeOpenAI:
    def __init__(self):
        self.calls = 0
        self.embedded = []
        self.embeddings = self
        self.responses = SimpleNamespace(parse=self._parse)

    def create(self, model, input):
        self.calls += 1
        self.embedded += input
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t)), 1.0, 0.5]) for t in input])

    def _parse(self, *, model, instructions, input, text_format):
        if text_format is chunking.ChunkContext:
            return SimpleNamespace(output_parsed=chunking.ChunkContext(context="CTXWORD situates this chunk"))
        return SimpleNamespace(output_parsed=chunking.DocTypeGuess(doc_type="research_article", reason="r"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "parse_pdf", fake_parse_pdf(FAKE_PIECES))
    monkeypatch.setattr(ingest, "token_counter", lambda model: lambda s: len(s.split()))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    q = QdrantClient(":memory:")
    oa = FakeOpenAI()
    run = lambda project, path=pdf, **kw: ingest.ingest(project, path, qdrant=q, openai_client=oa, collection=COLL,
                                                         embed_model="m", **kw)
    return SimpleNamespace(q=q, oa=oa, pdf=pdf, run=run)


def count(q, project=None):
    f = None
    if project:
        f = ingest.models.Filter(must=[ingest.models.FieldCondition(key="project_id", match=ingest.models.MatchValue(value=project))])
    return q.count(COLL, count_filter=f, exact=True).count


def test_point_id_is_deterministic():
    assert ingest.point_id("a", "sha", 0) == ingest.point_id("a", "sha", 0)
    assert ingest.point_id("a", "sha", 0) != ingest.point_id("b", "sha", 0)
    assert ingest.point_id("a", "sha", 0) != ingest.point_id("a", "sha", 1)


def test_reingest_same_file_adds_nothing(env):
    assert env.run("projA") == 3
    assert env.run("projA") == 0
    assert count(env.q) == 3
    assert env.oa.calls == 1  # second run skipped before embedding


def test_partial_upsert_is_repaired(env):
    env.run("projA")
    env.q.delete(COLL, points_selector=[ingest.point_id("projA", ingest.sha256_file(env.pdf), 2)])
    assert env.run("projA") == 3
    assert count(env.q) == 3


def test_same_file_in_other_project_is_isolated(env):
    env.run("projA")
    env.run("projB")
    assert count(env.q, "projA") == 3
    assert count(env.q, "projB") == 3


def test_payload_has_provenance(env):
    env.run("projA")
    (p,), _ = env.q.scroll(COLL, limit=1, scroll_filter=ingest.doc_filter("projA", ingest.sha256_file(env.pdf)))
    for key in ("project_id", "document_sha256", "source_filename", "page_numbers", "chunk_index", "chunk_count", "text"):
        assert key in p.payload


def test_rejects_bad_input(env, tmp_path):
    with pytest.raises(ValueError):
        env.run("  ")
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    with pytest.raises(ValueError):
        env.run("projA", txt)


SAMPLES = sorted(Path("data").rglob("*.pdf"), key=lambda p: p.stat().st_size)  # smallest first: cheapest live run


@pytest.mark.live
@pytest.mark.skipif(not SAMPLES, reason="put a PDF under data/")
def test_live_ingest_twice_no_duplicates():
    from openai import OpenAI

    from config import Settings

    s = Settings()
    coll = f"{s.qdrant_collection}_test"
    q = QdrantClient(url=s.qdrant_cluster_endpoint, api_key=s.qdrant_api_key)
    kw = {"qdrant": q, "openai_client": OpenAI(api_key=s.openai_api_key), "collection": coll, "embed_model": s.openai_embed_model}
    try:
        n = ingest.ingest("live-test", SAMPLES[0], **kw)
        assert n > 0
        assert ingest.ingest("live-test", SAMPLES[0], **kw) == 0
        assert q.count(coll, exact=True).count == n
    finally:
        if q.collection_exists(coll):
            q.delete_collection(coll)


def test_stale_extra_chunks_are_removed(env, monkeypatch):
    env.run("projA")
    monkeypatch.setattr(ingest, "parse_pdf", fake_parse_pdf(FAKE_PIECES[:2]))
    sha = ingest.sha256_file(env.pdf)
    env.q.delete(COLL, points_selector=[ingest.point_id("projA", sha, 0)])  # force a re-ingest
    assert env.run("projA") == 2
    assert count(env.q) == 2


def test_piece_from_chunk_maps_docling_meta():
    item = lambda label, *pages: SimpleNamespace(label=SimpleNamespace(value=label), prov=[SimpleNamespace(page_no=p) for p in pages])
    chunk = lambda items, headings: SimpleNamespace(text="t", meta=SimpleNamespace(doc_items=items, headings=headings))
    p = ingest.piece_from_chunk(chunk([item("text", 4), item("table", 5, 4)], None))
    assert p == chunking.Piece(text="t", headings=[], page_numbers=[4, 5], content_type="table")
    p = ingest.piece_from_chunk(chunk([item("text")], ["Methods"]))
    assert p.content_type == "text" and p.page_numbers == [] and p.headings == ["Methods"]


def test_plain_encoding_tolerates_special_tokens():
    import tiktoken
    base = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.Encoding(name="p", pat_str=base._pat_str, mergeable_ranks=base._mergeable_ranks, special_tokens={})
    assert enc.encode("risk <|endoftext|> matrix")


def test_source_filename_overrides_path_name(env):
    env.run("projA")  # default: path name
    sha = ingest.sha256_file(env.pdf)
    ingest.ingest("projB", env.pdf, qdrant=env.q, openai_client=env.oa, collection=COLL, embed_model="m",
                  source_filename="Original Name.pdf")
    (a,), _ = env.q.scroll(COLL, limit=1, scroll_filter=ingest.doc_filter("projA", sha))
    (b,), _ = env.q.scroll(COLL, limit=1, scroll_filter=ingest.doc_filter("projB", sha))
    assert a.payload["source_filename"] == "paper.pdf" and b.payload["source_filename"] == "Original Name.pdf"


def weights(v):
    return dict(zip(v.indices, v.values, strict=True))


def test_sparse_vector_is_deterministic_and_keeps_ratios_and_decimals_whole():
    text = "PaO2/FiO2 was 21.32%, values, values."
    v = ingest.sparse_vector(text)
    assert v == ingest.sparse_vector(text)
    w = weights(v)
    for tok in (b"pao2/fio2", b"21.32", b"was", b"values"):
        assert zlib.crc32(tok) in w
    assert zlib.crc32(b"pao2") not in w and zlib.crc32(b"21") not in w
    assert weights(ingest.sparse_vector("pao2/fio2 21.32", query=True)).keys() <= w.keys()  # question matches chunk


def test_sparse_vector_doc_weights_saturate_and_query_weights_are_one():
    doc = weights(ingest.sparse_vector("missing missing values"))
    assert doc[zlib.crc32(b"missing")] == pytest.approx(2 * 2.2 / 3.2)
    assert doc[zlib.crc32(b"values")] == pytest.approx(1.0)
    assert set(weights(ingest.sparse_vector("missing missing values", query=True)).values()) == {1.0}


def test_sparse_vector_of_text_without_tokens_is_empty():
    for text in ("", "   ", "?!%"):
        v = ingest.sparse_vector(text)
        assert v.indices == [] and v.values == []


def test_sparse_vector_sums_colliding_indices(monkeypatch):
    monkeypatch.setattr(ingest.zlib, "crc32", lambda b: 7)
    v = ingest.sparse_vector("alpha beta", query=True)
    assert v.indices == [7] and v.values == [2.0]


def test_new_collection_has_keyword_vector_and_points_carry_it(env):
    env.run("projA")
    assert ingest.has_sparse(env.q, COLL)
    for p in env.q.scroll(COLL, with_vectors=True, limit=10)[0]:
        assert set(p.vector) == {"", ingest.SPARSE_VECTOR}
        assert weights(p.vector[ingest.SPARSE_VECTOR]) == pytest.approx(weights(ingest.sparse_vector(p.payload["text"])))


def test_ingest_into_collection_without_keyword_vector_fails_before_writing(env):
    env.q.create_collection(COLL, vectors_config=ingest.models.VectorParams(size=3, distance=ingest.models.Distance.COSINE))
    with pytest.raises(ValueError, match="--migrate-to"):
        env.run("projA")
    assert count(env.q) == 0
    assert env.oa.calls == 0  # schema is checked before Docling and the paid embedding call


def test_unmigrated_collection_fails_even_for_an_already_indexed_document(env):
    env.q.create_collection(COLL, vectors_config=ingest.models.VectorParams(size=3, distance=ingest.models.Distance.COSINE))
    sha = ingest.sha256_file(env.pdf)
    env.q.upsert(COLL, [ingest.models.PointStruct(id=ingest.point_id("projA", sha, 0), vector=[1.0, 0.0, 0.0], payload={
        "project_id": "projA", "document_sha256": sha, "chunk_count": 1, "text": "t"})])
    with pytest.raises(ValueError, match="--migrate-to"):  # not a silent "already indexed"
        env.run("projA")


def old_collection(q, name="old", n=300):  # n > MIGRATE_BATCH exercises paging
    q.create_collection(name, vectors_config=ingest.models.VectorParams(size=3, distance=ingest.models.Distance.COSINE))
    q.upsert(name, [
        ingest.models.PointStruct(id=ingest.point_id("projA", "sha", i), vector=[1.0, float(i), 0.0], payload={
            "project_id": "projA", "document_sha256": "sha", "chunk_index": i, "text": f"missing values chunk {i}"})
        for i in range(n)
    ])


def test_migrate_copies_points_and_adds_keyword_vector():
    q = QdrantClient(":memory:")
    old_collection(q)
    assert ingest.migrate(q, "old", "new") == 300
    assert q.count("new", exact=True).count == 300
    src = {p.id: p for p in q.scroll("old", limit=500, with_vectors=True)[0]}
    for p in q.scroll("new", limit=500, with_vectors=True)[0]:
        assert p.payload == src[p.id].payload
        assert p.vector[""] == pytest.approx(src[p.id].vector)
        assert weights(p.vector[ingest.SPARSE_VECTOR]) == pytest.approx(weights(ingest.sparse_vector(p.payload["text"])))
    assert not ingest.has_sparse(q, "old") and q.count("old", exact=True).count == 300  # source untouched


def test_migrate_rerun_adds_nothing_and_tolerates_newer_points():
    q = QdrantClient(":memory:")
    old_collection(q)
    ingest.migrate(q, "old", "new")
    q.upsert("new", [ingest.models.PointStruct(  # a document ingested into the new collection after migrating
        id=ingest.point_id("projA", "sha2", 0), payload={"project_id": "projA", "text": "new doc"},
        vector={"": [0.0, 1.0, 0.0], ingest.SPARSE_VECTOR: ingest.sparse_vector("new doc")})])
    assert ingest.migrate(q, "old", "new") == 300
    assert q.count("new", exact=True).count == 301


def test_migrate_refuses_same_collection():
    with pytest.raises(ValueError, match="differ"):
        ingest.migrate(QdrantClient(":memory:"), "c", "c")


def test_points_carry_type_title_label_and_version(env):
    env.run("projA")
    p = env.q.scroll(COLL, limit=1, with_payload=True)[0][0].payload
    assert p["doc_type"] == "generic" and p["paper_title"] == "Fake paper"  # Methods/Results only: ambiguous, no model
    assert p["chunker_version"] == chunking.CHUNKER_VERSION
    assert p["text"].startswith("Document: Fake paper\nSection: ") and p["context"] == ""


def test_context_feeds_search_but_never_the_evidence_text(env):
    env.run("projA", context_model="ctx-model")
    assert any("CTXWORD" in t for t in env.oa.embedded)
    for p in env.q.scroll(COLL, limit=10, with_payload=True, with_vectors=True)[0]:
        assert "CTXWORD" not in p.payload["text"] and p.payload["context"].startswith("CTXWORD")
        assert zlib.crc32(b"ctxword") in weights(p.vector[ingest.SPARSE_VECTOR])
        assert p.payload["doc_type"] == "research_article"  # ambiguous -> fake LLM tie-break


def test_blank_context_leaves_no_trailing_newline():
    assert ingest.embed_text({"text": "t", "context": ""}) == "t"
    assert ingest.embed_text({"text": "t", "context": "c"}) == "t\nc"


def test_points_from_an_older_chunker_are_rechunked(env):
    assert env.run("projA") == 3
    env.q.set_payload(COLL, {"chunker_version": 1}, points=ingest.doc_filter("projA", ingest.sha256_file(env.pdf)))
    assert not ingest.already_indexed(env.q, COLL, "projA", ingest.sha256_file(env.pdf))
    assert env.run("projA") == 3


def seed(sessions, upload_dir, docs):
    """docs: [(project_name, sha, filename, status, write_pdf)] -> {filename: (project_id, doc_id)}"""
    out, projects = {}, {}
    with sessions() as s:
        u = db.User(email="r@x.io", password_hash="h")
        s.add(u)
        s.flush()
        for pname, sha, fn, status, write in docs:
            if pname not in projects:
                projects[pname] = db.Project(owner_id=u.id, name=pname)
                s.add(projects[pname])
                s.flush()
            pid = projects[pname].id
            d = db.Document(project_id=pid, filename=fn, sha256=sha, status=status)
            s.add(d)
            s.flush()
            if write:
                (upload_dir / str(pid)).mkdir(parents=True, exist_ok=True)
                (upload_dir / str(pid) / f"{sha}.pdf").write_bytes(b"%PDF-1.4 " + sha.encode())
            out[fn] = (str(pid), d.id)
        s.commit()
    return out


def test_reindex_parses_each_file_once_and_updates_postgres(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ingest, "parse_pdf", lambda path, filename, model: calls.append(path) or
                        ingest.ParsedDoc(pieces=FAKE_PIECES, title="Fake paper", page_count=5))
    monkeypatch.setattr(ingest, "token_counter", lambda model: lambda s: len(s.split()))
    sessions = db.make_sessions("sqlite://")
    a, b = "a" * 64, "b" * 64
    ids = seed(sessions, tmp_path, [("P1", a, "a.pdf", "ready", True), ("P2", a, "a-copy.pdf", "ready", True),
                                    ("P1", b, "gone.pdf", "ready", False), ("P1", "c" * 64, "bad.pdf", "failed", True)])
    q = QdrantClient(":memory:")
    out = ingest.reindex(qdrant=q, openai_client=FakeOpenAI(), sessions=sessions, src="unit", dst="new",
                         upload_dir=tmp_path, embed_model="m")
    assert out == {"documents": 2, "points": 6, "skipped": ["gone.pdf"], "failed": []}
    assert len(calls) == 1  # same sha in two projects: parsed once
    for fn in ("a.pdf", "a-copy.pdf"):
        pid, doc_id = ids[fn]
        assert q.count("new", count_filter=ingest.doc_filter(pid, a), exact=True).count == 3
        with sessions() as s:
            d = s.get(db.Document, doc_id)
            assert d.chunk_count == 3 and d.doc_type == "generic"


def test_reindex_reports_a_pdf_the_parser_crashes_on_and_continues(tmp_path, monkeypatch):
    def parse(path, filename, model):
        if filename == "crash.pdf":
            raise RuntimeError("layout model crashed")
        return ingest.ParsedDoc(pieces=FAKE_PIECES, title="T", page_count=5)
    monkeypatch.setattr(ingest, "parse_pdf", parse)
    monkeypatch.setattr(ingest, "token_counter", lambda model: lambda s: len(s.split()))
    sessions = db.make_sessions("sqlite://")
    seed(sessions, tmp_path, [("P", "d" * 64, "crash.pdf", "ready", True), ("P", "e" * 64, "ok.pdf", "ready", True)])
    out = ingest.reindex(qdrant=QdrantClient(":memory:"), openai_client=FakeOpenAI(), sessions=sessions, src="unit",
                         dst="new", upload_dir=tmp_path, embed_model="m")
    assert out["documents"] == 1 and out["failed"] == ["crash.pdf: RuntimeError: layout model crashed"]


def test_reindex_refuses_the_live_collection(tmp_path):
    with pytest.raises(ValueError, match="differ"):
        ingest.reindex(qdrant=QdrantClient(":memory:"), openai_client=FakeOpenAI(), sessions=db.make_sessions("sqlite://"),
                       src="unit", dst="unit", upload_dir=tmp_path, embed_model="m")
