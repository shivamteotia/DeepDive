"""A fixed, stratified labelling sample for the screening recall eval, labelled blind."""
import uuid

from sqlalchemy import select

import evaluate_screening as es
from tests.test_api_screening import draft, screen, setup


def screened(api, signup):
    h, pid, _ = setup(api, signup)  # 30 papers
    draft(api, h, pid)
    screen(api, h, pid)  # 10 include; the rest exclude or review
    return h, pid


def counts(api, h, pid):
    return api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"limit": 1}).json()["counts"]


def test_the_sample_is_stratified_by_the_ai_pile_and_fixed(api, signup):
    h, pid = screened(api, signup)
    piles = counts(api, h, pid)
    r = api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 12})
    assert r.status_code == 201, r.text
    import db
    with api.maxrim.sessions() as s:
        rows = list(s.scalars(select(db.LabelSample)))
    by = {}
    for x in rows:
        by.setdefault(x.stratum, []).append(x)
    assert len(by["include"]) == 4 and len(rows) == 12 == r.json()["size"]  # a third of 12
    assert {k: v[0].stratum_size for k, v in by.items()} == {"include": piles["included"], "review": piles["review"],
                                                             "exclude": piles["excluded"]}
    assert api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 12}).status_code == 409


def test_the_sample_is_listed_blind_and_shuffled(api, signup):
    h, pid = screened(api, signup)
    api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 12})
    out = api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"tab": "sample", "blind": 1, "limit": 50}).json()
    assert out["total"] == 12 and len(out["items"]) == 12 and out["counts"]["sample"] == 12
    assert not any(k in p for p in out["items"] for k in ("decision", "ai_decision", "verdicts", "columns"))
    again = api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"tab": "sample", "blind": 1, "limit": 50}).json()
    assert [p["id"] for p in out["items"]] == [p["id"] for p in again["items"]]  # a fixed order across sittings
    titles = [p["title"] for p in out["items"]]
    assert titles != sorted(titles, key=lambda t: int(t.split()[1]))  # not in import order


def test_small_piles_are_topped_up_and_screening_is_required(api, signup):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    assert api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 12}).status_code == 409  # not screened
    screen(api, h, pid)
    out = api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 1000}).json()
    assert out["size"] == 30  # larger than the project: every screened paper


def test_export_carries_stratum_weights_and_eval_weights_recall(api, signup):
    h, pid = screened(api, signup)
    api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 12})
    items = api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"tab": "sample", "blind": 1, "limit": 50}).json()["items"]
    for p in items:
        api.patch(f"/api/v1/projects/{pid}/papers/{p['id']}/decision", headers=h, json={"blind_label": "include"})
    gold = es.export(api.maxrim.sessions, uuid.UUID(pid))
    assert len(gold["papers"]) == 12 and all(p["weight"] >= 1 for p in gold["papers"])
    by = {}
    for p in gold["papers"]:
        by.setdefault(p["stratum"], set()).add(p["weight"])
    assert all(len(w) == 1 for w in by.values())  # one weight per stratum: its total / its sample size


def test_weighted_recall_counts_each_paper_for_its_stratum():
    def paper(i, label, weight):
        return {"id": str(i), "label": label, "weight": weight}
    papers = [paper(0, "include", 1.0), paper(1, "include", 10.0)]
    m = es.weighted_recall(papers, ["include", "exclude"])
    assert m == round(1.0 / 11.0, 3)  # the missed include stands for 10 papers in the excluded pile
    assert es.weighted_recall([paper(0, "exclude", 1.0)], ["exclude"]) is None


def _sample_ids(api, h, pid):
    return [p["id"] for p in api.get(f"/api/v1/projects/{pid}/papers", headers=h,
                                     params={"tab": "sample", "blind": 1, "limit": 50}).json()["items"]]


def _label(api, h, pid, paper_id, label="include"):
    api.patch(f"/api/v1/projects/{pid}/papers/{paper_id}/decision", headers=h, json={"blind_label": label})


def test_weights_use_labelled_sampled_papers_only(api, signup):
    import db
    h, pid = screened(api, signup)
    r = api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 12})
    assert set(r.json()) == {"size", "criteria_version"}  # pile sizes would hint at the AI's answers
    ids = _sample_ids(api, h, pid)
    for i in ids[:6]:  # half the sample labelled so far
        _label(api, h, pid, i)
    outside = next(p["id"] for p in api.get(f"/api/v1/projects/{pid}/papers", headers=h,
                                            params={"limit": 50}).json()["items"] if p["id"] not in ids)
    _label(api, h, pid, outside)
    gold = es.export(api.maxrim.sessions, uuid.UUID(pid))
    by_id = {p["id"]: p for p in gold["papers"]}
    assert by_id[outside]["weight"] is None and by_id[outside]["stratum"] is None  # not in the estimate
    with api.maxrim.sessions() as s:
        rows = {str(x.paper_id): x for x in s.scalars(select(db.LabelSample))}
    for i in ids[:6]:
        x = rows[i]
        labelled = sum(1 for j in ids[:6] if rows[j].stratum == x.stratum)
        assert by_id[i]["weight"] == round(x.stratum_size / labelled, 4)
    assert gold["sample"] == {"size": 12, "labelled": 6, "stale": False}


def test_weighted_recall_skips_unweighted_papers():
    papers = [{"id": "0", "label": "include", "weight": 2.0}, {"id": "1", "label": "include", "weight": None}]
    assert es.weighted_recall(papers, ["include", "exclude"]) == 1.0


def test_a_merged_sampled_paper_keeps_its_place_in_the_sample(api, signup):
    import db
    import funnel
    h, pid = screened(api, signup)
    api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 12})
    sampled = uuid.UUID(_sample_ids(api, h, pid)[0])
    with api.maxrim.sessions() as s:
        other = s.scalar(select(db.Paper.id).where(db.Paper.project_id == uuid.UUID(pid), db.Paper.id.not_in(
            select(db.LabelSample.paper_id))))
        funnel.carry(s, sampled, other)
        funnel._drop_papers(s, {sampled})
        s.commit()
        assert s.scalar(select(db.LabelSample.paper_id).where(db.LabelSample.paper_id == other)) == other
    assert len(_sample_ids(api, h, pid)) == 12


def test_the_sample_count_matches_its_list_and_staleness_is_reported(api, signup):
    import funnel
    h, pid = screened(api, signup)
    api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 12})
    gone = uuid.UUID(_sample_ids(api, h, pid)[0])
    with api.maxrim.sessions() as s:  # SQLite: no FK cascade, the sample row stays behind
        funnel._drop_papers(s, {gone})
        s.commit()
    listing = api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"tab": "sample", "blind": 1, "limit": 50}).json()
    assert listing["counts"]["sample"] == listing["total"] == len(listing["items"]) == 11
    from tests.test_api_screening import ADULTS, DESIGN, YEAR
    api.put(f"/api/v1/projects/{pid}/criteria", headers=h, json={"criteria": [YEAR, DESIGN, ADULTS]})
    assert es.export(api.maxrim.sessions, uuid.UUID(pid))["sample"]["stale"] is True
