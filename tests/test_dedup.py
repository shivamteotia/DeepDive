import dedup
from references import Record


def rec(title="A study of PCOS", year=2020, first_author="smith", doi="", pmid="", pmcid="", abstract="", source="pubmed"):
    return Record(source=source, source_key=pmid or doi or title, title=title, abstract=abstract,
                  authors=[first_author.title()], first_author=first_author, year=year, journal="", doi=doi, pmid=pmid,
                  pmcid=pmcid, language="eng", pub_types=[], keywords=[], raw={})


def sizes(groups):
    return sorted(len(g.members) for g in groups)


def test_same_doi_or_pmid_merges_transitively():
    g, _ = dedup.group([rec(doi="10.1/a"), rec(doi="10.1/a", pmid="9"), rec(pmid="9", title="other"), rec(doi="10.1/b", title="x")])
    assert sizes(g) == [1, 3]
    assert {x.reason for x in g} == {"doi", "single"}


def test_pmcid_merges():
    g, _ = dedup.group([rec(pmcid="PMC1", title="t1"), rec(pmcid="PMC1", title="t2")])
    assert sizes(g) == [2] and g[0].reason == "pmid"


def test_title_year_author_merges_records_without_ids():
    g, _ = dedup.group([rec(title="Effect of TRE in PCOS.", year=2024), rec(title="effect of tre in pcos", year=2025)])
    assert sizes(g) == [2] and g[0].reason == "title_year_author"


def test_title_match_needs_same_author_and_close_year():
    g, _ = dedup.group([rec(title="Effect of TRE in PCOS", year=2024), rec(title="Effect of TRE in PCOS", year=2020)])
    assert sizes(g) == [1, 1]
    g, _ = dedup.group([rec(title="Effect of TRE in PCOS"), rec(title="Effect of TRE in PCOS", first_author="jones")])
    assert sizes(g) == [1, 1]


def test_different_dois_never_merge_but_are_flagged():
    g, near = dedup.group([rec(title="Effect of TRE in PCOS", doi="10.1/a"), rec(title="Effect of TRE in PCOS", doi="10.1/erratum")])
    assert sizes(g) == [1, 1] and len(near) == 1 and near[0].similarity == 1.0


def test_different_dois_block_a_transitive_merge():
    # a and c share a PMID with b, but a and c carry different DOIs: a and c must not end up in one paper
    g, _ = dedup.group([rec(doi="10.1/a", pmid="1"), rec(pmid="1", title="b"), rec(doi="10.1/c", pmid="1", title="c")])
    assert all(len({r for r in x.members} & {0, 2}) < 2 for x in g)


def test_similar_titles_are_listed_not_merged():
    g, near = dedup.group([rec(title="Effect of TRE in women with PCOS", first_author="a"),
                           rec(title="Effect of TRE in women with PCOS: a pilot", first_author="b")])
    assert sizes(g) == [1, 1] and len(near) == 1 and near[0].similarity >= dedup.TITLE_SIMILAR


def test_canonical_record_prefers_abstract_then_pubmed():
    recs = [rec(source="europepmc"), rec(source="europepmc", abstract="long abstract"), rec(source="pubmed", abstract="short")]
    assert dedup.pick_canonical(recs) == 2
    assert dedup.pick_canonical([rec(source="europepmc", abstract="x"), rec(source="europepmc", abstract="longer one")]) == 1


def test_different_pmids_never_merge():
    g, _ = dedup.group([rec(pmid="1", doi="10.1/a"), rec(pmid="2", doi="10.1/a", title="b")])
    assert sizes(g) == [1, 1]


def test_preprint_and_article_with_the_same_title_stay_apart_but_are_listed():
    article = rec(title="Diabetes screening among women with PCOS", year=2024)
    preprint = rec(title="Diabetes screening among women with PCOS.", year=2024, source="europepmc").model_copy(
        update={"journal": "Research Square", "pub_types": ["misc"]})
    g, near = dedup.group([article, preprint])
    assert sizes(g) == [1, 1] and len(near) == 1
    assert dedup.is_preprint(rec().model_copy(update={"pub_types": ["Journal Article", "Preprint"]}))


def test_title_step_needs_an_author_and_a_year():
    g, _ = dedup.group([rec(title="Correction", first_author="", year=2020), rec(title="Correction", first_author="", year=2020)])
    assert sizes(g) == [1, 1]
    g, _ = dedup.group([rec(title="Editorial", year=None), rec(title="Editorial", year=None)])
    assert sizes(g) == [1, 1]


def test_year_window_does_not_chain():
    g, _ = dedup.group([rec(title="Effect of TRE in PCOS", year=y) for y in (2018, 2019, 2020)])
    assert sizes(g) == [1, 2]
