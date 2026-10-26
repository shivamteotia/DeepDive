import json

from qdrant_client import QdrantClient, models

import ingest
import papers

# The model sees the papers as P1, P2, ... in this order.
CARDS = {"sha-pan": papers.Card("JPan2023.pdf", "Development and validation of an in-hospital mortality model for "
                                "severe community-acquired pneumonia (SCAP) in the ICU | Jingjing Pan"),
         "sha-mar": papers.Card("BJMarafino2018.pdf", "Natural language processing of clinical notes for ICU "
                                "mortality | Marafino")}


def parse_returning(resolution):
    seen = []

    def parse(instructions, input, fmt):
        seen.append(input)
        return resolution
    return parse, seen


def res(*mentions):
    """res(("the SCAP study", "P1"), ("the stroke study", "")): one (words, paper id) per named study."""
    return papers.Resolution(rationale="r", named=[w for w, _ in mentions],
                             matches=[papers.Mention(words=w, paper=p) for w, p in mentions if p])


def test_a_question_naming_no_paper_searches_everything():
    parse, _ = parse_returning(res())
    assert papers.resolve(parse, "What AUROC values were reported?", CARDS) == papers.Route(None, [])


def test_a_named_paper_restricts_the_search_to_it():
    parse, seen = parse_returning(res(("the SCAP study", "P1")))
    assert papers.resolve(parse, "How many patients were in the SCAP study?", CARDS) == papers.Route(["sha-pan"], [])
    body = json.loads(seen[0])  # the model sees the question and each paper's id, file and card
    assert body["question"] == "How many patients were in the SCAP study?"
    assert body["papers"][0] == {"id": "P1", "file": "JPan2023.pdf", "card": CARDS["sha-pan"].text}


def test_papers_with_the_same_file_name_stay_apart():
    # Uploads keep their own names; two different PDFs called download.pdf must both reach the model.
    same = {"sha-1": papers.Card("download.pdf", "SCAP mortality model"),
            "sha-2": papers.Card("download.pdf", "NLP of ICU notes")}
    parse, seen = parse_returning(res(("the NLP study", "P2")))
    assert papers.resolve(parse, "the NLP study?", same) == papers.Route(["sha-2"], [])
    assert [p["id"] for p in json.loads(seen[0])["papers"]] == ["P1", "P2"]


def test_a_paper_the_project_does_not_have_is_reported_missing():
    parse, _ = parse_returning(res(("the acute ischemic stroke study", "")))
    assert papers.resolve(parse, "What AUC did the stroke study reach?", CARDS) == \
        papers.Route([], ["the acute ischemic stroke study"])


def test_no_refusal_when_a_card_carries_the_missing_names_words():
    # Measured: the model refused "the ICU epilepsy model" in a project holding the epilepsy paper (c03). A refusal
    # answers nothing, so it needs a check the model can't talk past: no distinctive word of the name on any card.
    parse, _ = parse_returning(res(("the pneumonia mortality model", "")))
    assert papers.resolve(parse, "q", CARDS) == papers.Route(None, [])  # "pneumonia" is on JPan's card
    parse, _ = parse_returning(res(("the Criticality Index models", "")))
    assert papers.resolve(parse, "q", CARDS) == papers.Route([], ["the Criticality Index models"])


def test_the_word_check_matches_other_forms_of_a_word():
    # "deep-learning" vs "Deep learning", "transformer" vs "Transformers": a refusal takes one reply, so the word
    # check must not miss a paper the project holds over a plural or a hyphen.
    cards = {"s": papers.Card("x.pdf", "Deep learning with Transformers for ICU notes")}
    for name in ("the deep-learning study", "the transformer study"):
        assert papers._on_a_card([name], cards), name
    assert not papers._on_a_card(["the acute ischemic stroke study"], cards)


def test_a_backed_refusal_needs_one_reply():
    calls = []
    parse = lambda instructions, input, fmt: calls.append(1) or res(("the acute ischemic stroke study", ""))
    assert papers.resolve(parse, "q", CARDS).shas == [] and len(calls) == 1


def test_a_comparison_searches_each_named_paper():
    parse, _ = parse_returning(res(("SCAP", "P1"), ("NLP", "P2")))
    route = papers.resolve(parse, "Compare the SCAP and NLP cohorts", CARDS)
    assert sorted(route.shas) == ["sha-mar", "sha-pan"] and route.missing == []


def test_a_route_needs_two_agreeing_replies():
    # Measured: one reply in ~70 refused or misrouted an answerable question, never the same one twice running.
    replies = [res(("SCAP", "P1")), res()]  # route, then "names nothing": disagree -> search everything
    parse = lambda instructions, input, fmt: replies.pop(0)
    assert papers.resolve(parse, "q", CARDS) == papers.Route(None, [])
    replies = [res(("SCAP", "P1")), res(("the SCAP study", "P1"))]  # same papers: agree
    assert papers.resolve(parse, "q", CARDS) == papers.Route(["sha-pan"], [])
    calls = []
    parse = lambda instructions, input, fmt: calls.append(1) or res()
    assert papers.resolve(parse, "q", CARDS) == papers.Route(None, []) and len(calls) == 1  # no route: one call


def test_which_studies_questions_search_everything():
    # Measured: "Which studies name Beth Israel Deaconess as a data source?" was routed to one paper by both replies
    # (x07 needs two). A question asking which papers do something names none; it must not be narrowed.
    calls = []
    parse = lambda instructions, input, fmt: calls.append(1) or res(("Beth Israel", "P2"))
    for q in ("Which studies name Beth Israel Deaconess as a data source?", "which of the papers used TF-IDF?",
              "What studies compared XGBoost with logistic regression?"):
        assert papers.resolve(parse, q, CARDS) == papers.Route(None, [])
    assert calls == []


def test_a_mixed_reply_searches_everything():
    # Measured on golden v2: "some named studies matched, some not" was usually a missed match (x03 lost a paper).
    parse, _ = parse_returning(res(("SCAP", "P1"), ("the NLP pipeline study", "")))
    assert papers.resolve(parse, "Compare SCAP and the NLP pipeline study", CARDS) == papers.Route(None, [])


def test_resolver_fails_open():
    for bad in (None, res(("X", "P9")), res(("SCAP", "P1"), ("X", "JPan2023.pdf")),
                res(("  ", ""))):  # no reply; an invented id (alone or beside a real one); a blank mention
        parse, _ = parse_returning(bad)
        assert papers.resolve(parse, "q", CARDS) == papers.Route(None, [])
    parse, _ = parse_returning(res(("SCAP", "P1")))
    assert papers.resolve(parse, "q", {}) == papers.Route(None, [])  # no cards: nothing to resolve against


def test_cards_come_from_each_papers_front_matter_and_abstract():
    q = QdrantClient(":memory:")
    q.create_collection("c", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    rows = [(0, "Paper: Dynamic Mortality Risk Predictions for Children\nSection: Front matter\nTrujillo Rivera",
             "Front matter of the study by Eduardo Trujillo Rivera et al., published in 2022 in JAMA Netw Open."),
            (1, "Paper: Dynamic Mortality Risk Predictions for Children\nSection: Abstract\nObjective: predict", ""),
            (2, "Paper: Dynamic Mortality Risk Predictions for Children\nSection: Abstract\nthe Criticality Index", ""),
            (papers.CARD_CHUNKS, "Paper: Dynamic\nSection: Methods\nMethods text that must not be in the card", "")]
    q.upsert("c", [models.PointStruct(id=ingest.point_id("demo", "sha-ea", i), vector=[1, 0], payload={
        "project_id": "demo", "document_sha256": "sha-ea", "source_filename": "EA2022.pdf", "chunk_index": i,
        "text": text, "context": ctx}) for i, text, ctx in rows])
    out = papers.cards(q, "c", "demo", ["sha-ea", "sha-missing"])
    assert set(out) == {"sha-ea"} and out["sha-ea"].file == "EA2022.pdf"
    card = out["sha-ea"].text
    assert card.startswith("Dynamic Mortality Risk Predictions for Children | ")
    assert "published in 2022" in card  # the front matter's summary: first author, year, journal
    assert "Criticality Index" in card  # an alias deep in the abstract still reaches the resolver
    assert "Methods text" not in card and card.count("Section:") == 0
    assert papers.cards(q, "c", "other-project", ["sha-ea"]) == {}  # point ids are project-scoped


def test_search_named_scopes_and_splits_the_search():
    # One function for the API run and evaluate.py, so retrieval is measured the way the app searches.
    calls = []
    search = lambda q, k, sha256_in: calls.append((k, sha256_in)) or [{"sha": s} for s in sha256_in]
    papers.search_named(search, "q", None, ["a", "b", "c"], 6)  # names nothing: every ready paper, full k
    papers.search_named(search, "q", ["b"], ["a", "b", "c"], 6)  # one paper: only it, full k
    hits = papers.search_named(search, "q", ["a", "c"], ["a", "b", "c"], 6)  # a comparison: each paper, k split
    assert calls == [(6, ["a", "b", "c"]), (6, ["b"]), (3, ["a"]), (3, ["c"])]
    assert hits == [{"sha": "a"}, {"sha": "c"}]
    assert papers.search_named(search, "q", [], ["a"], 6) == []  # declined: nothing is searched
    calls.clear()
    papers.search_named(search, "q", ["a", "not-ready"], ["a", "b"], 6)  # a paper outside ready is dropped
    papers.search_named(search, "q", ["not-ready"], ["a", "b"], 6)  # ...and if none is left, every ready paper
    assert calls == [(6, ["a"]), (6, ["a", "b"])]


def test_search_named_searches_each_paper_for_a_which_studies_question():
    # x03: "which studies used TF-IDF?" names no paper; one pooled search let two papers crowd out the third.
    calls = []
    search = lambda q, k, sha256_in: calls.append((k, sha256_in)) or []
    papers.search_named(search, "Which studies used TF-IDF?", None, ["a", "b", "c"], 6)
    papers.search_named(search, "Which studies used TF-IDF?", None, ["a"], 6)  # one paper: nothing to split
    papers.search_named(search, "Which studies used TF-IDF?", ["b"], ["a", "b", "c"], 6)  # a route still wins
    assert calls == [(3, ["a"]), (3, ["b"]), (3, ["c"]), (6, ["a"]), (6, ["b"])]


def test_resolver_schema_has_no_defaults():
    from openai.lib._pydantic import to_strict_json_schema
    assert '"default"' not in json.dumps(to_strict_json_schema(papers.Resolution))  # OpenAI strict mode forbids them
