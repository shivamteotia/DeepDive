import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

import ingest

COLL = "unit"
FAKE_CHUNKS = [
    {"text": f"chunk {i}", "page_numbers": [i + 1], "section_headings": ["Results"], "content_type": "text"}
    for i in range(3)
]


class FakeOpenAI:
    def __init__(self):
        self.calls = 0
        self.embeddings = self

    def create(self, model, input):
        self.calls += 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t)), 1.0, 0.5]) for t in input])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "chunk_pdf", lambda path, model: FAKE_CHUNKS)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    q = QdrantClient(":memory:")
    oa = FakeOpenAI()
    run = lambda project, path=pdf: ingest.ingest(project, path, qdrant=q, openai_client=oa, collection=COLL, embed_model="m")
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
    monkeypatch.setattr(ingest, "chunk_pdf", lambda path, model: FAKE_CHUNKS[:2])
    sha = ingest.sha256_file(env.pdf)
    env.q.delete(COLL, points_selector=[ingest.point_id("projA", sha, 0)])  # force a re-ingest
    assert env.run("projA") == 2
    assert count(env.q) == 2


def test_chunk_record_maps_docling_meta():
    item = lambda label, *pages: SimpleNamespace(label=SimpleNamespace(value=label), prov=[SimpleNamespace(page_no=p) for p in pages])
    rec = ingest.chunk_record("t", SimpleNamespace(doc_items=[item("text", 4), item("table", 5, 4)], headings=None))
    assert rec == {"text": "t", "page_numbers": [4, 5], "section_headings": [], "content_type": "table"}
    rec = ingest.chunk_record("t", SimpleNamespace(doc_items=[item("text")], headings=["Methods"]))
    assert rec["content_type"] == "text" and rec["page_numbers"] == [] and rec["section_headings"] == ["Methods"]


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
