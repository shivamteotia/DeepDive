"""A paper's screening work (decisions, blind labels, extracted values, PDF link) survives regrouping and merges."""
import uuid

from sqlalchemy import delete, select

import db
from tests.test_api_extraction import add, extract, with_extraction
from tests.test_api_references import upload
from tests.test_api_screening import all_papers, by_pmid, draft, screen, setup


def _work(api, pid, paper_id):
    with api.maxrim.sessions() as s:
        doc = db.Document(project_id=uuid.UUID(pid), filename="x.pdf", sha256=uuid.uuid4().hex * 2, status="ready")
        s.add(doc)
        s.flush()
        s.get(db.Paper, uuid.UUID(paper_id)).document_id = doc.id
        s.commit()
        return doc.id


def _state(api, paper_id):
    with api.maxrim.sessions() as s:
        p = s.get(db.Paper, uuid.UUID(paper_id))
        values = list(s.scalars(select(db.ExtractValue).where(db.ExtractValue.paper_id == p.id))) if p else []
        return p, [v.value for v in values]


def test_a_later_import_keeps_the_work_on_a_paper_that_gains_a_record(api, signup):
    h, pid, llm = setup(api, signup, n=6)
    draft(api, h, pid)
    screen(api, h, pid)
    with_extraction(api, llm)
    p0 = by_pmid(all_papers(api, h, pid))["1000"]
    api.patch(f"/api/v1/projects/{pid}/papers/{p0['id']}/decision", headers=h,
              json={"decision": "exclude", "reason": "wrong population", "blind_label": "exclude"})
    extract(api, h, pid, add(api, h, pid).json()["id"], tab="all")
    doc_id = _work(api, pid, p0["id"])
    upload(api, h, pid, "second.nbib", "PMID- 1000\nDP  - 2019\nTI  - Study 0 of PCOS\nAB  - x\nFAU - Author0, A\n")
    a0 = by_pmid(all_papers(api, h, pid))["1000"]
    assert a0["id"] == p0["id"] and len(a0["sources"]) >= 1
    assert (a0["decision"], a0["decided_by"], a0["human_reason"]) == ("exclude", "human", "wrong population")
    paper, values = _state(api, a0["id"])
    assert paper.document_id == doc_id and values


def test_merging_a_duplicate_keeps_the_removed_papers_pdf_decision_and_values(api, signup):
    h, pid, llm = setup(api, signup, n=6)
    draft(api, h, pid)
    screen(api, h, pid)
    with_extraction(api, llm)
    papers = by_pmid(all_papers(api, h, pid))
    a, b = papers["1001"], papers["1000"]  # b: included, gets the PDF, the human decision and the values
    extract(api, h, pid, add(api, h, pid).json()["id"], tab="all")
    api.patch(f"/api/v1/projects/{pid}/papers/{b['id']}/decision", headers=h,
              json={"decision": "include", "reason": "checked full text"})
    doc_id = _work(api, pid, b["id"])
    with api.maxrim.sessions() as s:
        s.execute(delete(db.ExtractValue).where(db.ExtractValue.paper_id == uuid.UUID(a["id"])))
        d = db.PossibleDuplicate(project_id=uuid.UUID(pid), paper_a=uuid.UUID(a["id"]), paper_b=uuid.UUID(b["id"]),
                                 similarity=0.95)
        s.add(d)
        s.commit()
        dup = str(d.id)
    r = api.post(f"/api/v1/projects/{pid}/possible-duplicates/{dup}", headers=h, json={"action": "merge"})
    assert r.status_code == 200, r.text
    merged = next(p for p in all_papers(api, h, pid)["items"] if p["id"] == a["id"])
    assert (merged["decision"], merged["human_reason"]) == ("include", "checked full text")
    paper, values = _state(api, a["id"])
    assert paper.document_id == doc_id and values


def test_an_earlier_paper_id_goes_to_the_group_holding_most_of_its_records():
    import funnel
    p, q = uuid.uuid4(), uuid.uuid4()
    assert funnel._reuse_ids([[0], [1, 2]], [p, p, p], {p}) == [None, p]  # split: not the first group, the majority
    assert funnel._reuse_ids([[0, 1, 2]], [p, q, q], {p, q}) == [q]  # merge: q's id, p is carried
    assert funnel._reuse_ids([[0]], [p], set()) == [None]  # pinned papers are never reused


def test_a_dismissed_pair_stays_dismissed_after_a_merge(api, signup):
    h, pid, _ = setup(api, signup, n=6)
    papers = by_pmid(all_papers(api, h, pid))
    a, b, x = (uuid.UUID(papers[k]["id"]) for k in ("1001", "1000", "1002"))
    with api.maxrim.sessions() as s:
        s.add(db.PossibleDuplicate(project_id=uuid.UUID(pid), paper_a=b, paper_b=x, similarity=0.9, resolved=True))
        d = db.PossibleDuplicate(project_id=uuid.UUID(pid), paper_a=a, paper_b=b, similarity=0.95)
        s.add(d)
        s.commit()
        dup = str(d.id)
    api.post(f"/api/v1/projects/{pid}/possible-duplicates/{dup}", headers=h, json={"action": "merge"})
    with api.maxrim.sessions() as s:
        pairs = {frozenset((r.paper_a, r.paper_b)) for r in s.scalars(select(db.PossibleDuplicate).where(
            db.PossibleDuplicate.resolved.is_(True)))}
    assert frozenset((a, x)) in pairs
