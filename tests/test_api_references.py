NBIB = """PMID- 111\r
DP  - 2024 Feb 6\r
TI  - Time-restricted eating in women with PCOS.\r
LID - 10.1/shared [doi]\r
AB  - Abstract from PubMed.\r
FAU - Smith, Jane\r
LA  - eng\r
PT  - Journal Article\r
PT  - Randomized Controlled Trial\r
\r
PMID- 222\r
DP  - 2019\r
TI  - Metabolic profile of women with PCOS.\r
FAU - Lee, Ann\r
PT  - Journal Article\r
"""

BIB = """@article {PMID:111,\r
\tTitle = {Time-restricted eating in women with PCOS},\r
\tAuthor = {Smith, Jane},\r
\tDOI = {10.1/shared},\r
\tYear = {2024},\r
\tJournal = {Some journal},\r
}\r
@article {PMID:333,\r
\tTitle = {Ovarian morphology in adolescents},\r
\tAuthor = {Kim, Bo},\r
\tYear = {2021},\r
}\r
@article {broken,\r
\tTitle = {no end\r
"""


def project(api, h, name="P"):
    return api.post("/api/v1/projects", headers=h, json={"name": name}).json()["id"]


def upload(api, h, pid, name, text):
    return api.post(f"/api/v1/projects/{pid}/reference-files", headers=h, files={"file": (name, text.encode(), "text/plain")})


def papers(api, h, pid, tab="all"):
    return api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"tab": tab}).json()


def test_upload_two_exports_dedups_across_files(api, signup):
    h = signup()
    pid = project(api, h)
    a = upload(api, h, pid, "pubmed.nbib", NBIB)
    assert a.status_code == 201 and a.json()["records"] == 2 and a.json()["papers"] == 2
    b = upload(api, h, pid, "europepmc.bib", BIB).json()
    assert b["records"] == 2 and b["papers"] == 3 and b["merged"] == 1 and len(b["errors"]) == 1
    listing = papers(api, h, pid)
    assert listing["total"] == 3 and listing["counts"]["all"] == 3
    shared = next(p for p in listing["items"] if p["pmid"] == "111")
    assert sorted(shared["sources"]) == ["europepmc", "pubmed"] and shared["dedup_reason"] == "doi"
    assert shared["abstract"] == "Abstract from PubMed." and "Randomized Controlled Trial" in shared["pub_types"]


def test_same_file_twice_adds_nothing(api, signup):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "pubmed.nbib", NBIB)
    again = upload(api, h, pid, "renamed.nbib", NBIB)
    assert again.status_code == 200 and again.json()["already_imported"] is True
    assert papers(api, h, pid)["total"] == 2


def test_papers_are_project_scoped(api, signup):
    h, other = signup(), signup("b@example.com")
    pid = project(api, h)
    upload(api, h, pid, "pubmed.nbib", NBIB)
    assert api.get(f"/api/v1/projects/{pid}/papers", headers=other).status_code == 404
    assert upload(api, other, pid, "x.nbib", NBIB).status_code == 404
    mine = project(api, other, "mine")
    assert papers(api, other, mine)["total"] == 0


def test_unreadable_file_is_rejected_with_a_reason(api, signup):
    h = signup()
    r = upload(api, h, project(api, h), "notes.txt", "hello")
    assert r.status_code == 422 and "format" in r.json()["detail"]


def test_split_undoes_a_merge_and_survives_the_next_import(api, signup):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "pubmed.nbib", NBIB)
    upload(api, h, pid, "europepmc.bib", BIB)
    shared = next(p for p in papers(api, h, pid)["items"] if p["pmid"] == "111")
    r = api.post(f"/api/v1/projects/{pid}/papers/{shared['id']}/split", headers=h)
    assert r.status_code == 200 and len(r.json()["papers"]) == 2
    upload(api, h, pid, "more.bib", "@article {PMID:444,\n Title = {Something else},\n Author = {Ng, Al},\n Year = {2020},\n}\n")
    items = papers(api, h, pid)["items"]
    assert len(items) == 5 and sum(p["pmid"] == "111" for p in items) == 2
    assert {p["dedup_reason"] for p in items if p["pmid"] == "111"} == {"manual"}


def test_possible_duplicate_merge_and_dismiss(api, signup):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "a.bib", "@article {k1,\n Title = {Effect of TRE in women with PCOS},\n Author = {Smith, J},\n Year = {2020},\n}\n"
                                  "@article {k2,\n Title = {Effect of TRE in women with PCOS: a pilot},\n Author = {Jones, K},\n Year = {2020},\n}\n"
                                  "@article {k3,\n Title = {Effect of TRE in women with PCOS: an update},\n Author = {Wu, L},\n Year = {2021},\n}\n")
    dups = papers(api, h, pid, "duplicates")
    assert dups["counts"]["duplicates"] >= 2 and dups["items"][0]["possible_duplicate"]["similarity"] >= 0.85
    first, second = dups["items"][0]["possible_duplicate"]["id"], dups["items"][-1]["possible_duplicate"]["id"]
    assert api.post(f"/api/v1/projects/{pid}/possible-duplicates/{first}", headers=h, json={"action": "dismiss"}).status_code == 200
    m = api.post(f"/api/v1/projects/{pid}/possible-duplicates/{second}", headers=h, json={"action": "merge"})
    assert m.status_code == 200 and m.json()["paper"]["dedup_reason"] == "manual"
    assert papers(api, h, pid)["total"] == 2


def test_other_users_paper_cannot_be_split(api, signup):
    h, other = signup(), signup("b@example.com")
    pid = project(api, h)
    upload(api, h, pid, "pubmed.nbib", NBIB)
    paper = papers(api, h, pid)["items"][0]
    assert api.post(f"/api/v1/projects/{pid}/papers/{paper['id']}/split", headers=other).status_code == 404


THREE = ("@article {k1,\n Title = {Effect of TRE in women with PCOS},\n Author = {Smith, J},\n Year = {2020},\n DOI = {10.1/k1},\n}\n"
         "@article {k2,\n Title = {Effect of TRE in women with PCOS: a pilot},\n Author = {Jones, K},\n Year = {2020},\n}\n"
         "@article {k3,\n Title = {Effect of TRE in women with PCOS: an update},\n Author = {Wu, L},\n Year = {2021},\n}\n")


def dup_between(api, h, pid, title_a, title_b):
    for d in papers(api, h, pid, "duplicates")["items"]:
        x, y = d["title"], d["possible_duplicate"]["other"]["title"]
        if (x == title_a and title_b in y) or (y == title_a and title_b in x):  # title_a exact: it prefixes the others
            return d["possible_duplicate"]
    return None


def test_pending_duplicates_of_a_merged_paper_survive_the_next_import(api, signup):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "a.bib", THREE)
    d = dup_between(api, h, pid, "Effect of TRE in women with PCOS", "a pilot")
    api.post(f"/api/v1/projects/{pid}/possible-duplicates/{d['id']}", headers=h, json={"action": "merge"})
    assert dup_between(api, h, pid, "Effect of TRE in women with PCOS", "an update") is not None
    upload(api, h, pid, "b.bib", "@article {z,\n Title = {Unrelated},\n Author = {Ng, A},\n Year = {2019},\n}\n")
    assert dup_between(api, h, pid, "Effect of TRE in women with PCOS", "an update") is not None


def test_new_record_with_the_doi_of_a_merged_paper_joins_it(api, signup):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "a.bib", THREE)
    d = dup_between(api, h, pid, "Effect of TRE in women with PCOS", "a pilot")
    api.post(f"/api/v1/projects/{pid}/possible-duplicates/{d['id']}", headers=h, json={"action": "merge"})
    before = papers(api, h, pid)["total"]
    upload(api, h, pid, "c.nbib", "PMID- 9\r\nDP  - 2020\r\nTI  - Effect of TRE in women with PCOS.\r\nLID - 10.1/k1 [doi]\r\nFAU - Smith, J\r\n")
    after = papers(api, h, pid)
    merged = next(p for p in after["items"] if p["doi"] == "10.1/k1")
    assert sum(p["doi"] == "10.1/k1" for p in after["items"]) == 1, "the new copy must join the merged paper"
    assert after["total"] == before and set(merged["sources"]) == {"bibtex", "pubmed"} and merged["pmid"] == "9"


def test_records_matching_split_papers_never_undo_the_split(api, signup):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "pubmed.nbib", NBIB)
    upload(api, h, pid, "europepmc.bib", BIB)
    shared = next(p for p in papers(api, h, pid)["items"] if p["pmid"] == "111")
    api.post(f"/api/v1/projects/{pid}/papers/{shared['id']}/split", headers=h)
    upload(api, h, pid, "third.bib", "@article {PMID:111,\n Title = {Time-restricted eating in women with PCOS},\n"
                                     " Author = {Smith, Jane},\n DOI = {10.1/shared},\n Year = {2024},\n Note = {third copy},\n}\n")
    items = papers(api, h, pid)["items"]
    assert sum(p["pmid"] == "111" for p in items) == 3  # the split stays; the new copy is its own paper ...
    assert papers(api, h, pid, "duplicates")["counts"]["duplicates"] >= 2  # ... flagged against both halves


def test_long_free_text_fields_are_clipped_to_their_columns():
    import funnel
    from references import Record
    r = Record(source="bibtex", source_key="k", title="t", abstract="", authors=["A" * 300], first_author="a" * 300,
               year=2020, journal="", doi="10.1/" + "x" * 300, pmid="", pmcid="", language="English; French; German",
               pub_types=[], keywords=[], raw={})
    f = funnel._paper_fields([r], "single")
    assert len(f["first_author"]) <= 200 and len(f["language"]) <= 20 and len(f["doi"]) <= 255


def test_keep_the_journal_article_supersedes_the_preprint(api, signup):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "a.bib",
           "@article {PMID:1,\n Title = {Diabetes screening among women with PCOS},\n Author = {Vollmer, A},\n Year = {2024},\n"
           " DOI = {10.1186/a},\n Journal = {BMC endocrine disorders},\n}\n"
           "@misc {PPR:PPR9,\n Title = {Diabetes screening among women with PCOS.},\n Author = {Vollmer, A},\n Year = {2024},\n"
           " DOI = {10.21203/rs.3.rs-1/v1},\n Publisher = {Research Square},\n}\n")
    d = papers(api, h, pid, "duplicates")["items"][0]["possible_duplicate"]
    r = api.post(f"/api/v1/projects/{pid}/possible-duplicates/{d['id']}", headers=h, json={"action": "keep_article"})
    assert r.status_code == 200
    items = {p["doi"]: p for p in papers(api, h, pid)["items"]}
    assert items["10.21203/rs.3.rs-1/v1"]["superseded_by"] == items["10.1186/a"]["id"]
    assert items["10.1186/a"]["superseded_by"] is None and papers(api, h, pid, "duplicates")["counts"]["duplicates"] == 0


def test_keep_article_refuses_when_neither_is_a_preprint(api, signup):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "a.bib", THREE)
    d = papers(api, h, pid, "duplicates")["items"][0]["possible_duplicate"]
    r = api.post(f"/api/v1/projects/{pid}/possible-duplicates/{d['id']}", headers=h, json={"action": "keep_article"})
    assert r.status_code == 409 and "preprint" in r.json()["detail"]
