import uuid

import db
import evaluate_screening as es
import screening as sc
from tests.test_screening import DESIGN, PAPER, YEAR_2015, FakeParse, v


def gold(*rows):
    return {"criteria": [c.model_dump() for c in (YEAR_2015, DESIGN)],
            "papers": [{**PAPER, "id": str(i), "title": f"Paper {i}", "abstract": ab, "year": year, "label": label}
                       for i, (ab, year, label) in enumerate(rows)]}


def test_metrics_count_recall_on_includes_precision_and_review_pile():
    g = gold(("A retrospective chart review of women.", 2019, "include"),   # AI include  -> true include
             ("A retrospective chart review of women.", 2010, "include"),   # AI exclude (year) -> a missed include
             ("A randomized controlled trial of women.", 2019, "exclude"),  # AI exclude -> correct
             ("FAIL", 2019, "include"),                                     # model fails -> review, still counted kept
             ("A retrospective chart review of men.", 2019, "exclude"))     # AI include -> a wrong include

    def answer(payload: str):
        import json
        data = json.loads(payload)
        if data["abstract"] == "FAIL":
            raise RuntimeError("down")
        retro = "retrospective" in data["abstract"]
        return sc.AiScreen(verdicts=[v("c4", "yes" if retro else "no",
                                       "retrospective chart review" if retro else "randomized controlled trial")])

    out = es.evaluate(g, FakeParse({sc.AiScreen: answer}))
    m = out["metrics"]
    assert (m["includes"], m["includes_kept"], m["recall_includes"]) == (3, 2, round(2 / 3, 3))
    assert m["ai_include"] == 2 and m["precision_include"] == 0.5
    assert m["review_share"] == 0.2 and m["n"] == 5
    assert [p["id"] for p in out["missed_includes"]] == ["1"]
    assert out["missed_includes"][0]["verdicts"][0]["criterion_id"] == "c1"


def test_export_writes_blind_labels_with_paper_fields_and_criteria():
    sessions = db.make_sessions("sqlite://")
    with sessions() as s:
        user = db.User(email="x@example.test", password_hash="x")
        s.add(user)
        s.flush()
        project = db.Project(owner_id=user.id, name="p")
        s.add(project)
        s.flush()
        papers = [db.Paper(project_id=project.id, title=f"T{i}", abstract="a", year=2019, pub_types=["Journal Article"])
                  for i in range(3)]
        s.add_all(papers)
        s.add(db.CriteriaSet(project_id=project.id, version=1, criteria=[YEAR_2015.model_dump()]))
        s.flush()
        s.add_all([db.ScreeningDecision(project_id=project.id, paper_id=papers[0].id, criteria_version=1, blind_label="include"),
                   db.ScreeningDecision(project_id=project.id, paper_id=papers[1].id, criteria_version=1, blind_label="exclude"),
                   db.ScreeningDecision(project_id=project.id, paper_id=papers[2].id, criteria_version=1, human_decision="include")])
        s.commit()
        pid = project.id
    out = es.export(sessions, pid)
    assert out["criteria"][0]["id"] == "c1" and out["criteria_version"] == 1
    assert sorted((p["title"], p["label"]) for p in out["papers"]) == [("T0", "include"), ("T1", "exclude")]
    assert {"abstract", "year", "pub_types", "language"} <= set(out["papers"][0])
    assert es.export(sessions, uuid.uuid4())["papers"] == []


def _project(s):
    user = db.User(email=f"{uuid.uuid4().hex}@example.test", password_hash="x")
    s.add(user)
    s.flush()
    project = db.Project(owner_id=user.id, name="p")
    s.add(project)
    s.flush()
    return project


def test_export_uses_each_papers_newest_row_and_skips_superseded_preprints():
    sessions = db.make_sessions("sqlite://")
    with sessions() as s:
        project = _project(s)
        cleared, relabelled, article, preprint = (db.Paper(project_id=project.id, title=t) for t in ("cleared", "re", "art", "pre"))
        s.add_all([cleared, relabelled, article, preprint])
        s.flush()
        preprint.superseded_by = article.id
        s.add_all([db.CriteriaSet(project_id=project.id, version=v, criteria=[YEAR_2015.model_dump()]) for v in (1, 2)])
        s.add_all([
            db.ScreeningDecision(project_id=project.id, paper_id=cleared.id, criteria_version=1, blind_label="include"),
            db.ScreeningDecision(project_id=project.id, paper_id=cleared.id, criteria_version=2, blind_label=None),
            db.ScreeningDecision(project_id=project.id, paper_id=relabelled.id, criteria_version=1, blind_label="include"),
            db.ScreeningDecision(project_id=project.id, paper_id=relabelled.id, criteria_version=2, blind_label="exclude"),
            db.ScreeningDecision(project_id=project.id, paper_id=article.id, criteria_version=2, blind_label="include"),
            db.ScreeningDecision(project_id=project.id, paper_id=preprint.id, criteria_version=1, blind_label="include")])
        s.commit()
        pid = project.id
    out = es.export(sessions, pid)
    assert sorted((p["title"], p["label"]) for p in out["papers"]) == [("art", "include"), ("re", "exclude")]


def test_export_cli_refuses_empty_gold_sets_and_overwrites(tmp_path, monkeypatch):
    import pytest
    sessions = db.make_sessions("sqlite://")
    with sessions() as s:
        pid = _project(s).id
        s.commit()
    monkeypatch.setattr(es, "_sessions", lambda: sessions)
    out = tmp_path / "gold.json"
    with pytest.raises(SystemExit, match="no screening criteria"):
        es.main(["export", "--project", str(pid), "--out", str(out)])
    with sessions() as s:
        s.add(db.CriteriaSet(project_id=pid, version=1, criteria=[YEAR_2015.model_dump()]))
        s.commit()
    with pytest.raises(SystemExit, match="no blind labels"):
        es.main(["export", "--project", str(pid), "--out", str(out)])
    with sessions() as s:
        paper = db.Paper(project_id=pid, title="t")
        s.add(paper)
        s.flush()
        s.add(db.ScreeningDecision(project_id=pid, paper_id=paper.id, criteria_version=1, blind_label="include"))
        s.commit()
    assert es.main(["export", "--project", str(pid), "--out", str(out)]) == 0 and out.exists()
    with pytest.raises(SystemExit, match="already exists"):
        es.main(["export", "--project", str(pid), "--out", str(out)])
    assert es.main(["export", "--project", str(pid), "--out", str(out), "--force"]) == 0
