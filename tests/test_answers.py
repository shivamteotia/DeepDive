"""Saving a research answer as CSV or Word, and charting numbers from its findings (never invented ones)."""
import csv
import io
import json
import uuid

import docx

import answers
import db
from tests.test_agent import Script, call, outputs, say
from tests.test_api_research import project
from tests.test_api_screening import setup

RESULT = {
    "question": "Literature summary on: ML for ICU mortality", "status": "needs_review", "gaps": "No pediatric data.",
    "claims": [
        {"text": "Sun 2026 reported XGBoost AUC 0.899 internally.", "evidence_ids": ["E1"], "verdict": "SUPPORTED"},
        {"text": "Cheng 2026 found LightGBM AUROC 0.841.", "evidence_ids": ["E2", "E3"], "verdict": "THINLY_SUPPORTED"}],
    "evidence": [
        {"eid": "E1", "source_filename": "Sun 2026, JMIR (PMID 42766803)", "page_numbers": [], "text": "...",
         "source_type": "pubmed", "url": "https://pubmed.ncbi.nlm.nih.gov/42766803/", "title": "XGBoost for SIC"},
        {"eid": "E2", "source_filename": "JPan2023.pdf", "page_numbers": [4, 5], "text": "..."},
        {"eid": "E3", "source_filename": "WHO", "page_numbers": [], "text": "...", "source_type": "web",
         "url": "https://who.int/x", "title": "WHO page"}]}


_CLOCK = [0]


def run_in(api, pid, cid=None, result=RESULT, status="completed", question="ML for ICU mortality"):
    from datetime import timedelta
    _CLOCK[0] += 1  # distinct times: the Windows clock can repeat within a test
    when = db.utcnow() + timedelta(seconds=_CLOCK[0])
    with api.maxrim.sessions() as s:
        r = db.ResearchRun(project_id=uuid.UUID(pid), question=question, max_iterations=1, status=status, result=result,
                           conversation_id=uuid.UUID(cid) if cid else None, pubmed=True, created_at=when,
                           finished_at=when if status == "completed" else None)
        s.add(r)
        s.commit()
        return str(r.id)


def test_an_answer_exports_as_csv_with_one_row_per_finding(api, signup):
    h = signup()
    pid = project(api, h, "A")
    rid = run_in(api, pid)
    r = api.get(f"/api/v1/research-runs/{rid}/export", headers=h, params={"format": "csv"})
    assert r.status_code == 200 and r.headers["content-disposition"].endswith('.csv"')
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert [x["Finding"] for x in rows] == [c["text"] for c in RESULT["claims"]]
    assert rows[0]["Verdict"] == "supported" and rows[1]["Verdict"] == "thinly supported"
    assert rows[0]["Sources"] == "[1] Sun 2026, JMIR (PMID 42766803)"
    assert rows[1]["Sources"] == "[2] JPan2023.pdf p. 4, 5; [3] WHO page"
    assert rows[0]["Links"] == "https://pubmed.ncbi.nlm.nih.gov/42766803/" and rows[1]["Links"] == "https://who.int/x"


def test_an_answer_exports_as_a_word_report(api, signup):
    h = signup()
    pid = project(api, h, "A")
    rid = run_in(api, pid)
    r = api.get(f"/api/v1/research-runs/{rid}/export", headers=h, params={"format": "docx"})
    assert r.status_code == 200 and r.headers["content-disposition"].endswith('.docx"')
    text = "\n".join(p.text for p in docx.Document(io.BytesIO(r.content)).paragraphs)
    assert "ML for ICU mortality" in text and "Literature summary on" not in text  # the user's question
    assert "Sun 2026 reported XGBoost AUC 0.899 internally. [1]" in text and "(thinly supported)" in text
    assert "No pediatric data." in text and "References" in text and "PMID 42766803" in text


def test_answer_export_is_owner_only_and_needs_a_finished_answer(api, signup):
    h = signup()
    pid = project(api, h, "A")
    rid = run_in(api, pid)
    other = signup("b@example.com")
    assert api.get(f"/api/v1/research-runs/{rid}/export", headers=other, params={"format": "csv"}).status_code == 404
    assert api.get(f"/api/v1/research-runs/{rid}/export", headers=h, params={"format": "pdf"}).status_code == 422
    queued = run_in(api, pid, result=None, status="queued")
    assert api.get(f"/api/v1/research-runs/{queued}/export", headers=h, params={"format": "csv"}).status_code == 409


def test_a_chart_keeps_only_values_its_finding_states():
    def parse(instructions, input, fmt):
        data = json.loads(input)
        assert data["request"] == "AUROC by study" and len(data["findings"]) == 2
        return fmt(title="AUROC by study", y_label="AUROC", points=[
            answers.ChartPoint(label="Sun 2026", value="0.899", finding=1),
            answers.ChartPoint(label="Cheng 2026", value="0.841", finding=2),
            answers.ChartPoint(label="Made up 2026", value="0.95", finding=2),   # not in finding 2
            answers.ChartPoint(label="Out of range", value="0.7", finding=9)])
    chart = answers.make_chart(parse, RESULT, "AUROC by study")
    assert chart["points"] == [{"label": "Sun 2026", "value": 0.899, "finding": 1},
                               {"label": "Cheng 2026", "value": 0.841, "finding": 2}]
    assert chart["dropped"] == 2 and chart["title"] == "AUROC by study" and chart["y_label"] == "AUROC"
    assert "error" in answers.make_chart(lambda *a: None, RESULT, "x")


def test_the_assistant_saves_and_charts_the_chats_latest_answer(api, signup, monkeypatch):
    h, pid, _ = setup(api, signup)
    cid = api.post(f"/api/v1/projects/{pid}/conversations", headers=h, json={}).json()["id"]
    run_in(api, pid, cid=cid, result={**RESULT, "claims": RESULT["claims"][:1]})
    newest = run_in(api, pid, cid=cid)
    monkeypatch.setattr(answers, "make_chart", lambda parse, result, request: {"title": request, "points": [], "dropped": 0})
    script = Script([call("export_answer", {"format": "docx"}), call("chart_answer", {"request": "AUROC by study"}, "c2")],
                    say("Done."))
    api.maxrim.openai.responses.create = script.create
    api.post(f"/api/v1/conversations/{cid}/agent", headers=h, json={"message": "save it and chart it"})
    saved, chart = outputs(script.requests[1])
    assert saved["answer_export"] == {"run_id": newest, "format": "docx", "findings": 2, "question": "ML for ICU mortality"}
    assert chart["chart"]["title"] == "AUROC by study" and chart["chart"]["run_id"] == newest


def test_numbers_match_ranges_leading_dots_and_thousands_only():
    assert answers._numbers("AUROC 0.80-0.85") == {0.8, 0.85}
    assert answers._numbers("COVID-19 cohort") == {19.0}
    assert answers._numbers("AUC .899 and 1,250 patients") == {0.899, 1250.0}
    assert answers._numbers("AUCs 1,2 and 3") == {1.0, 2.0, 3.0}
    assert answers._numbers("89.9% accuracy") == {89.9}


def test_csv_cells_cannot_run_as_formulas_and_docx_survives_control_characters():
    res = {**RESULT, "claims": [{"text": "=HYPERLINK(\"http://x\") claims\x0c more", "evidence_ids": ["E1"],
                                 "verdict": "SUPPORTED"}], "gaps": "a\x0bb"}
    rows = list(csv.reader(io.StringIO(answers.to_csv(res).decode("utf-8-sig"))))
    assert rows[1][1].startswith("'=")
    text = "\n".join(p.text for p in docx.Document(io.BytesIO(answers.to_docx("Q\x00", res))).paragraphs)
    assert "claims more" in text.replace("  ", " ") or "claims more" in text


def test_sources_are_numbered_like_the_ui_one_number_per_page_or_url():
    res = {**RESULT, "claims": [{"text": "a", "evidence_ids": ["E1", "E4"], "verdict": "SUPPORTED"},
                                {"text": "b", "evidence_ids": ["E2"], "verdict": "SUPPORTED"}],
           "evidence": RESULT["evidence"] + [{**RESULT["evidence"][0], "eid": "E4", "text": "another chunk"}]}
    _, data = answers.rows(res)
    assert data[0][3] == "[1] Sun 2026, JMIR (PMID 42766803)" and data[1][3] == "[2] JPan2023.pdf p. 4, 5"
    text = "\n".join(p.text for p in docx.Document(io.BytesIO(answers.to_docx("Q", res))).paragraphs)
    assert text.count("PMID 42766803") == 1  # one reference per source, not per chunk


def test_points_past_the_limit_count_as_dropped():
    many = {**RESULT, "claims": [{"text": " ".join(str(i) for i in range(1, 60)), "evidence_ids": [], "verdict": "SUPPORTED"}]}

    def parse(instructions, input, fmt):
        return fmt(title="t", y_label="y", points=[answers.ChartPoint(label=str(i), value=str(i), finding=1)
                                                   for i in range(1, 50)])
    chart = answers.make_chart(parse, many, "x")
    assert len(chart["points"]) == answers.MAX_POINTS and chart["dropped"] == 49 - answers.MAX_POINTS


def test_another_chats_answer_is_never_the_latest(api, signup):
    h, pid, _ = setup(api, signup)
    mine = api.post(f"/api/v1/projects/{pid}/conversations", headers=h, json={}).json()["id"]
    theirs = api.post(f"/api/v1/projects/{pid}/conversations", headers=h, json={}).json()["id"]
    run_in(api, pid, cid=theirs)
    script = Script([call("export_answer", {"format": "csv"})], say("x"))
    api.maxrim.openai.responses.create = script.create
    api.post(f"/api/v1/conversations/{mine}/agent", headers=h, json={"message": "save it"})
    assert "error" in outputs(script.requests[1])[0]
