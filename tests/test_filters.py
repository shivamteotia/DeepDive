"""Filtering and counting papers for the assistant: by year, database, decision, journal or an extracted value."""
import json
import uuid

from sqlalchemy import select

import agent
from tests.test_agent import Script, call, chat, say
from tests.test_api_extraction import add, extract, screened, with_extraction

NO_FILTER = {"year_from": None, "year_to": None, "database": None, "column": None, "value": None}


def tool(api, pid, name, args):
    with api.maxrim.sessions() as s:
        ctx = agent.ToolContext(s=s, project_id=uuid.UUID(pid), parse=None, ask_research=None)
        return agent.call_tool(name, json.dumps(args), ctx)


def with_design(api, signup):
    h, pid, llm = screened(api, signup)  # 30 papers (one from 2010, the rest 2019), 10 included
    with_extraction(api, llm)
    extract(api, h, pid, add(api, h, pid).json()["id"], tab="all")
    return h, pid


def test_counts_by_year_database_decision_and_extracted_value(api, signup):
    _, pid = with_design(api, signup)
    by = lambda what, **kw: tool(api, pid, "count_papers", {"tab": "all", "by": what, **NO_FILTER, **kw})
    assert by("year") == {"tab": "all", "by": "year", "total": 30, "counts": {"2019": 29, "2010": 1}}
    assert by("database")["counts"] == {"pubmed": 30}
    assert by("decision")["counts"]["include"] == 10
    design = by("column", column="Study design")["counts"]
    assert design["retrospective cohort"] == 10 and sum(design.values()) == 30
    assert by("year", year_from=2015)["total"] == 29


def test_get_papers_filters_by_year_database_and_value(api, signup):
    _, pid = with_design(api, signup)
    papers = lambda **kw: tool(api, pid, "get_papers", {"tab": "all", "limit": 20, **NO_FILTER, **kw})
    old = papers(year_to=2015)
    assert old["total"] == 1 and old["items"][0]["year"] == 2010
    assert "error" in papers(database="SCOPUS") and papers(database="PubMed")["total"] == 30  # none from Scopus
    retro = papers(column="study design", value="Retrospective Cohort")
    assert retro["total"] == 10 and all(p["columns"]["Study design"] == "retrospective cohort" for p in retro["items"])
    assert tool(api, pid, "get_papers", {"tab": "all", "limit": 5})["total"] == 30  # filters are optional


def test_bad_filters_are_errors_the_model_can_read(api, signup):
    _, pid = with_design(api, signup)
    out = tool(api, pid, "count_papers", {"tab": "all", "by": "column", **NO_FILTER})
    assert "column" in out["error"]
    out = tool(api, pid, "get_papers", {"tab": "all", "limit": 5, **NO_FILTER, "column": "Country"})
    assert "Study design" in out["error"]  # names the columns that exist
    assert "error" in tool(api, pid, "count_papers", {"tab": "duplicates", "by": "year", **NO_FILTER})


def test_the_assistant_counts_from_the_database(api, signup):
    h, pid = with_design(api, signup)
    script = Script([call("count_papers", {"tab": "included", "by": "column", **NO_FILTER, "column": "Study design"})],
                    say("All 10 included papers are retrospective cohorts."))
    out = chat(api, h, pid, script, "how many included papers per study design?")[1].json()
    result = next(m for m in out["messages"] if m["role"] == "tool")["content"]["result"]
    assert result["counts"] == {"retrospective cohort": 10}


def test_counts_include_superseded_papers_like_the_tabs(api, signup):
    import db
    h, pid = with_design(api, signup)
    with api.maxrim.sessions() as s:
        papers = list(s.scalars(select(db.Paper).where(db.Paper.project_id == uuid.UUID(pid)).limit(2)))
        papers[0].superseded_by = papers[1].id
        s.commit()
    listed = api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"limit": 1}).json()["total"]
    assert tool(api, pid, "count_papers", {"tab": "all", "by": "year", **NO_FILTER})["total"] == listed == 30
    assert tool(api, pid, "get_papers", {"tab": "all", "limit": 5, **NO_FILTER, "year_from": 1900})["total"] == 30


def test_database_names_are_mapped_and_unknown_ones_are_errors(api, signup):
    _, pid = with_design(api, signup)
    papers = lambda **kw: tool(api, pid, "get_papers", {"tab": "all", "limit": 5, **NO_FILTER, **kw})
    assert papers(database="MEDLINE")["total"] == 30  # alias of pubmed
    for name in ("Web of Science", "Google Scholar"):
        out = papers(database=name)
        assert "error" in out and "pubmed" in out["error"]  # names the databases this project has


def test_truncated_counts_say_so_and_add_up(api, signup, monkeypatch):
    import funnel
    _, pid = with_design(api, signup)
    monkeypatch.setattr(funnel, "MAX_COUNTS", 1)
    out = tool(api, pid, "count_papers", {"tab": "all", "by": "year", **NO_FILTER})
    assert out["counts"] == {"2019": 29} and out["other"] == 1 and out["distinct"] == 2
    assert sum(out["counts"].values()) + out["other"] == out["total"]
