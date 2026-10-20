import json
from types import SimpleNamespace

import extraction as ex
import jobs
from tests.test_api_screening import all_papers, by_pmid, draft, screen, setup

DESIGN = {"name": "Study design", "instruction": "The study design", "kind": "category",
          "options": ["retrospective cohort", "prospective cohort", "randomized controlled trial (RCT)"]}


def with_extraction(api, llm):
    """The screening fake also answers extraction calls: from passages when given, else from the abstract."""
    calls = []

    def parse(model, instructions, input, text_format):
        if text_format is not ex.Answer:
            return llm.parse(model, instructions, input, text_format)
        data = json.loads(input)
        calls.append(data)
        text = " ".join(data["passages"]) or data["abstract"]
        if "prospective cohort" in text:
            return SimpleNamespace(output_parsed=ex.Answer(value="prospective cohort", quote="prospective cohort", reason="r"))
        if "retrospective" in text:
            return SimpleNamespace(output_parsed=ex.Answer(value="retrospective cohort", quote="retrospective chart review",
                                                           reason="r"))
        return SimpleNamespace(output_parsed=ex.Answer(value="not stated", quote="", reason="not stated"))

    api.maxrim.openai.responses = SimpleNamespace(parse=parse)
    return calls


def screened(api, signup):
    h, pid, llm = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)  # papers 0, 3, ..., 27 included
    return h, pid, llm


def add(api, h, pid, body=DESIGN):
    return api.post(f"/api/v1/projects/{pid}/columns", headers=h, json=body)


def extract(api, h, pid, column_id, tab="included", confirm=True):
    a = api.post(f"/api/v1/projects/{pid}/actions", headers=h,
                 json={"tool": "extract_column", "column_id": column_id, "tab": tab})
    assert a.status_code == 201, a.text
    if confirm:
        api.post(f"/api/v1/actions/{a.json()['id']}/confirm", headers=h)
    return api.get(f"/api/v1/actions/{a.json()['id']}", headers=h).json()


def test_columns_are_created_listed_and_validated(api, signup):
    h, pid, _ = screened(api, signup)
    r = add(api, h, pid)
    assert r.status_code == 201 and r.json()["name"] == "Study design"
    assert add(api, h, pid, {**DESIGN, "options": ["only one"]}).status_code == 422
    assert [c["name"] for c in api.get(f"/api/v1/projects/{pid}/columns", headers=h).json()] == ["Study design"]


def test_extraction_needs_confirm_and_fills_the_included_papers(api, signup):
    h, pid, llm = screened(api, signup)
    calls = with_extraction(api, llm)
    col = add(api, h, pid).json()
    pending = extract(api, h, pid, col["id"], confirm=False)
    assert pending["status"] == "pending" and pending["estimate"]["calls"] == 10 and calls == []
    done = extract(api, h, pid, col["id"])
    assert done["status"] == "done" and done["result"]["filled"] == 10 and len(calls) == 10
    got = api.get(f"/api/v1/columns/{col['id']}", headers=h).json()
    assert got["counts"] == {"retrospective cohort": 10} and all(v["source"] == "abstract" for v in got["values"])
    paper = by_pmid(all_papers(api, h, pid))["1000"]
    assert paper["columns"]["Study design"]["value"] == "retrospective cohort"


def test_full_text_is_read_when_the_pdf_is_ingested_and_rerun_fills_only_what_is_missing(api, signup, monkeypatch):
    import db
    h, pid, llm = screened(api, signup)
    calls = with_extraction(api, llm)
    col = add(api, h, pid).json()
    extract(api, h, pid, col["id"])
    target = by_pmid(all_papers(api, h, pid))["1003"]
    with api.maxrim.sessions() as s:  # its PDF is now ingested
        import uuid
        doc = db.Document(project_id=uuid.UUID(pid), filename="x.pdf", sha256="ab" * 32, status="ready")
        s.add(doc)
        s.flush()
        s.get(db.Paper, uuid.UUID(target["id"])).document_id = doc.id
        s.commit()
    seen = []

    def passages(state, project_id, sha, query):
        seen.append((str(project_id), sha))
        return ["Methods: in this prospective cohort study we enrolled 80 women."]

    monkeypatch.setattr(jobs, "passages_for", passages)
    calls.clear()
    again = extract(api, h, pid, col["id"])
    assert again["estimate"]["calls"] == 1 and len(calls) == 1 and seen == [(pid, "ab" * 32)]
    value = by_pmid(all_papers(api, h, pid))["1003"]["columns"]["Study design"]
    assert (value["value"], value["source"]) == ("prospective cohort", "fulltext")


def test_other_projects_cannot_see_or_fill_a_column(api, signup):
    h, pid, _ = screened(api, signup)
    col = add(api, h, pid).json()
    other = signup("b@example.com")
    theirs = api.post("/api/v1/projects", headers=other, json={"name": "theirs"}).json()["id"]
    assert api.get(f"/api/v1/columns/{col['id']}", headers=other).status_code == 404
    r = api.post(f"/api/v1/projects/{theirs}/actions", headers=other,
                 json={"tool": "extract_column", "column_id": col["id"], "tab": "all"})
    assert r.status_code == 404


def test_blind_listing_hides_extracted_values(api, signup):
    h, pid, llm = screened(api, signup)
    with_extraction(api, llm)
    extract(api, h, pid, add(api, h, pid).json()["id"])
    assert not any("columns" in p for p in all_papers(api, h, pid, blind=1)["items"])


def test_failed_calls_are_retried_and_counted(api, signup):
    h, pid, llm = screened(api, signup)
    calls = with_extraction(api, llm)
    real = api.maxrim.openai.responses.parse
    state = {"down": True}

    def flaky(model, instructions, input, text_format):
        if text_format is ex.Answer and state["down"]:
            raise RuntimeError("rate limited")
        return real(model, instructions, input, text_format)

    api.maxrim.openai.responses = SimpleNamespace(parse=flaky)
    col = add(api, h, pid).json()
    first = extract(api, h, pid, col["id"])
    assert first["result"]["errors"] == 10 and first["result"]["filled"] == 0
    state["down"] = False
    second = extract(api, h, pid, col["id"])
    assert second["estimate"]["calls"] == 10 and second["result"]["filled"] == 10 and len(calls) == 10


def test_extraction_refuses_the_duplicates_tab_and_duplicate_names(api, signup):
    h, pid, _ = screened(api, signup)
    col = add(api, h, pid).json()
    r = api.post(f"/api/v1/projects/{pid}/actions", headers=h,
                 json={"tool": "extract_column", "column_id": col["id"], "tab": "duplicates"})
    assert r.status_code == 422
    assert add(api, h, pid, {**DESIGN, "name": "study DESIGN "}).status_code == 409


def test_a_run_never_grows_past_the_confirmed_papers(api, signup):
    h, pid, llm = setup(api, signup)
    calls = with_extraction(api, llm)
    draft(api, h, pid)
    col = add(api, h, pid).json()
    pending = extract(api, h, pid, col["id"], confirm=False)  # 0 included so far
    assert pending["estimate"]["calls"] == 0
    screen(api, h, pid)  # now 10 included
    api.post(f"/api/v1/actions/{pending['id']}/confirm", headers=h)
    assert calls == []  # it only does what was confirmed


def test_a_pdf_without_passages_is_not_billed_again(api, signup, monkeypatch):
    import uuid

    import db
    h, pid, llm = screened(api, signup)
    calls = with_extraction(api, llm)
    target = by_pmid(all_papers(api, h, pid))["1003"]
    with api.maxrim.sessions() as s:
        doc = db.Document(project_id=uuid.UUID(pid), filename="x.pdf", sha256="cd" * 32, status="ready")
        s.add(doc)
        s.flush()
        s.get(db.Paper, uuid.UUID(target["id"])).document_id = doc.id
        s.commit()
    monkeypatch.setattr(jobs, "passages_for", lambda *a: [])  # its chunks are not in this collection
    col = add(api, h, pid).json()
    extract(api, h, pid, col["id"])
    calls.clear()
    again = extract(api, h, pid, col["id"])
    assert again["estimate"]["calls"] == 0 and calls == []


def test_a_reread_with_the_same_answer_is_not_billed_a_third_time(api, signup, monkeypatch):
    import uuid

    import db
    h, pid, llm = screened(api, signup)
    calls = with_extraction(api, llm)
    col = add(api, h, pid).json()
    extract(api, h, pid, col["id"])  # from the abstract
    target = by_pmid(all_papers(api, h, pid))["1003"]
    with api.maxrim.sessions() as s:  # the PDF becomes ready later, but its chunks are not in this collection
        doc = db.Document(project_id=uuid.UUID(pid), filename="x.pdf", sha256="ef" * 32, status="ready")
        s.add(doc)
        s.flush()
        s.get(db.Paper, uuid.UUID(target["id"])).document_id = doc.id
        s.commit()
    monkeypatch.setattr(jobs, "passages_for", lambda *a: [])
    assert extract(api, h, pid, col["id"])["estimate"]["calls"] == 1  # one upgrade attempt, same answer
    calls.clear()
    assert extract(api, h, pid, col["id"])["estimate"]["calls"] == 0 and calls == []


def test_a_full_text_not_stated_keeps_the_value_found_in_the_abstract(api, signup, monkeypatch):
    import uuid

    import db
    h, pid, llm = screened(api, signup)
    with_extraction(api, llm)
    col = add(api, h, pid).json()
    extract(api, h, pid, col["id"])  # retrospective cohort, from the abstract
    target = by_pmid(all_papers(api, h, pid))["1003"]
    with api.maxrim.sessions() as s:
        doc = db.Document(project_id=uuid.UUID(pid), filename="x.pdf", sha256="12" * 32, status="ready")
        s.add(doc)
        s.flush()
        s.get(db.Paper, uuid.UUID(target["id"])).document_id = doc.id
        s.commit()
    monkeypatch.setattr(jobs, "passages_for", lambda *a: ["Methods were as described before."])
    real = api.maxrim.openai.responses.parse

    def silent(model, instructions, input, text_format):  # the full-text read misses what the abstract says
        if text_format is ex.Answer:
            return SimpleNamespace(output_parsed=ex.Answer(value="not stated", quote="", reason="not stated"))
        return real(model, instructions, input, text_format)

    api.maxrim.openai.responses = SimpleNamespace(parse=silent)
    run = extract(api, h, pid, col["id"])
    assert run["estimate"]["calls"] == 1 and run["result"]["fulltext"] == 0 and run["result"]["not_stated"] == 0
    value = by_pmid(all_papers(api, h, pid))["1003"]["columns"]["Study design"]
    assert value["value"] == "retrospective cohort"
    assert extract(api, h, pid, col["id"])["estimate"]["calls"] == 0  # and it is not billed again
