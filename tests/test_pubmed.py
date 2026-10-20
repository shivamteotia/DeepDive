import httpx

import pubmed

EFETCH = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle><MedlineCitation><PMID>40000001</PMID><Article>
  <Journal><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue><Title>Critical Care</Title></Journal>
  <ArticleTitle>Machine learning for early ICU mortality prediction</ArticleTitle>
  <Abstract><AbstractText Label="BACKGROUND">Early warning matters.</AbstractText>
   <AbstractText Label="RESULTS">The model reached an AUROC of 0.89.</AbstractText></Abstract>
  <AuthorList><Author><LastName>Smith</LastName><Initials>J</Initials></Author>
   <Author><LastName>Lee</LastName><Initials>K</Initials></Author></AuthorList>
  <ELocationID EIdType="doi">10.1/cc.1</ELocationID></Article></MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID>40000002</PMID><Article>
  <Journal><JournalIssue><PubDate><MedlineDate>2024 Jan-Feb</MedlineDate></PubDate></JournalIssue><Title>J X</Title></Journal>
  <ArticleTitle>A letter without an abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def transport(seen, search=None, fetch=EFETCH):
    def h(request):
        seen.append(request)
        if "esearch" in request.url.path:
            return httpx.Response(200, json=search or {"esearchresult": {"idlist": ["40000001", "40000002"]}})
        return httpx.Response(200, text=fetch)
    return httpx.MockTransport(h)


def test_latest_papers_become_cited_evidence():
    seen = []
    hits = pubmed.search("ML for early ICU mortality", api_key="ncbi-key", project_id="p1", transport=transport(seen))
    q = dict(seen[0].url.params)
    assert q["sort"] == "pub_date" and q["retmax"] == str(pubmed.MAX_RESULTS) and q["api_key"] == "ncbi-key"
    assert q["term"] == "ML for early ICU mortality" and dict(seen[1].url.params)["id"] == "40000001,40000002"
    assert len(hits) == 1  # the letter has no abstract: nothing to cite
    h = hits[0]
    assert h["source_type"] == "pubmed" and h["project_id"] == "p1" and h["pmid"] == "40000001"
    assert h["url"] == "https://pubmed.ncbi.nlm.nih.gov/40000001/" and h["year"] == 2025 and h["doi"] == "10.1/cc.1"
    assert h["source_filename"] == "Smith 2025, Critical Care (PMID 40000001)"
    assert "BACKGROUND: Early warning matters." in h["text"] and "AUROC of 0.89" in h["text"]


def test_no_results_are_empty_but_failures_raise():
    import pytest
    assert pubmed.search("q", api_key="", project_id="p", transport=transport([], search={"esearchresult": {"idlist": []}})) == []
    for bad in (transport([], fetch="<not xml"), httpx.MockTransport(lambda r: httpx.Response(429)),
                httpx.MockTransport(lambda r: httpx.Response(200, json=["not", "a", "dict"]))):
        with pytest.raises(pubmed.PubMedError):
            pubmed.search("q", api_key="secret-key", project_id="p", transport=bad)


def test_a_consortium_first_author_is_named():
    xml = EFETCH.replace("<Author><LastName>Smith</LastName><Initials>J</Initials></Author>",
                         "<Author><CollectiveName>ICU Study Group</CollectiveName></Author>")
    hit = pubmed.search("q", api_key="", project_id="p", transport=transport([], fetch=xml))[0]
    assert hit["source_filename"].startswith("ICU Study Group 2025")


def test_a_question_becomes_a_keyword_query():
    calls = []

    def parse(instructions, input, fmt):
        calls.append(input)
        return fmt(query="deep learning sepsis prediction ICU")
    q = "What do recent studies report about the AUROC of deep learning models for sepsis prediction in the ICU?"
    assert pubmed.keyword_query(parse, q) == "deep learning sepsis prediction ICU" and q in calls[0]
    assert pubmed.keyword_query(lambda *a: None, q) == q  # no answer: the question itself
    assert pubmed.keyword_query(lambda *a: (_ for _ in ()).throw(RuntimeError("down")), q) == q
    assert pubmed.keyword_query(lambda i, inp, fmt: fmt(query="  "), q) == q


def test_no_key_still_works_and_sends_no_key():
    seen = []
    pubmed.search("q", api_key="", project_id="p", transport=transport(seen))
    assert "api_key" not in dict(seen[0].url.params)


def test_the_summary_framing_never_reaches_pubmed():
    seen = []
    pubmed.search(pubmed.review_question("ML for early ICU mortality"), api_key="", project_id="p", transport=transport(seen))
    assert dict(seen[0].url.params)["term"] == "ML for early ICU mortality"


def test_an_esearch_error_inside_a_200_is_a_failure():
    import pytest
    with pytest.raises(pubmed.PubMedError):
        pubmed.search("q", api_key="", project_id="p",
                      transport=transport([], search={"esearchresult": {"ERROR": "Empty term and query_key"}}))
