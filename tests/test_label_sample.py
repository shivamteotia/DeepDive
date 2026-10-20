"""A fixed, stratified labelling sample for the screening recall eval, labelled blind."""
import uuid

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
    strata = r.json()["strata"]
    assert strata["include"] == {"sampled": 4, "total": piles["included"]}  # a third of 12
    assert sum(x["sampled"] for x in strata.values()) == 12
    assert {k: x["total"] for k, x in strata.items()} == {"include": piles["included"], "review": piles["review"],
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
    strata = api.post(f"/api/v1/projects/{pid}/label-sample", headers=h, json={"size": 1000}).json()["strata"]
    assert all(x["sampled"] == x["total"] for x in strata.values())  # larger than the project: everything


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
