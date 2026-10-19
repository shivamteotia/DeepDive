
import references

NBIB = """PMID- 38176412\r
OWN - NLM\r
DP  - 2024 Feb 6\r
TI  - Time-restricted eating: Watching the clock to treat obesity.\r
LID - S1550-4131(23)00454-0 [pii]\r
LID - 10.1016/j.cmet.2023.12.004 [doi]\r
AB  - Time-restricted eating (TRE) has become a popular strategy to treat obesity. TRE \r
      involves confining the eating window to 4-10 h per day.\r
FAU - Ezpeleta, Mark\r
AU  - Ezpeleta M\r
FAU - Cienfuegos, Sofia\r
AU  - Cienfuegos S\r
LA  - eng\r
PT  - Journal Article\r
PT  - Review\r
JT  - Cell metabolism\r
MH  - Female\r
OT  - fasting\r
\r
PMID- 40000001\r
DP  - 2019\r
TI  - Metabolic profile of women with polycystic ovary syndrome.\r
AID - 10.1000/XYZ.1 [doi]\r
PMC - PMC1234567\r
FAU - Smith, Jane\r
LA  - eng\r
PT  - Journal Article\r
"""

BIB = """@article {PMID:42371138,\r
\tTitle = {Effect of two types of time-restricted eating on weight in women with {PCOS}: a randomized controlled trial},\r
\tAuthor = {Aminian, Maryam and Hajshafiha, Masoumeh},\r
\tDOI = {10.1007/s00394-026-04034-3},\r
\tYear = {2026},\r
\tJournal = {European journal of nutrition},\r
\tAbstract = {<h4>Purpose</h4>Polycystic ovary syndrome (PCOS) is a prevalent disorder.<h4>Methods</h4>In this trial, 75 women.},\r
\tURL = {https://doi.org/10.1007/s00394-026-04034-3},\r
}\r
\r
@misc {PPR:PPR1024731,\r
\tTitle = {Beyond Hormones: time-restricted eating for PCOS},\r
\tAuthor = {von Damm, Isabela and Son, Federico},\r
\tDOI = {https://doi.org/10.1101/2025.05.21.25327203},\r
\tPublisher = {medRxiv},\r
\tYear = {2025},\r
}\r
"""


def test_nbib_joins_continuation_lines_and_reads_ids():
    recs, errs = references.parse_file("x.nbib", NBIB.encode())
    r = recs[0]
    assert errs == [] and len(recs) == 2
    assert r.pmid == "38176412" and r.doi == "10.1016/j.cmet.2023.12.004" and r.source == "pubmed"
    assert r.abstract == ("Time-restricted eating (TRE) has become a popular strategy to treat obesity. TRE "
                          "involves confining the eating window to 4-10 h per day.")
    assert r.first_author == "ezpeleta" and r.authors == ["Ezpeleta, Mark", "Cienfuegos, Sofia"]
    assert r.year == 2024 and r.pub_types == ["Journal Article", "Review"] and r.language == "eng"
    assert r.journal == "Cell metabolism" and r.keywords == ["Female", "fasting"]


def test_nbib_doi_from_aid_and_pmcid():
    r = references.parse_file("x.nbib", NBIB.encode())[0][1]
    assert r.doi == "10.1000/xyz.1" and r.pmcid == "PMC1234567" and r.abstract == "" and r.year == 2019


def test_europepmc_bib_reads_key_strips_html_and_nested_braces():
    recs, errs = references.parse_file("e.bib", BIB.encode())
    r, pre = recs
    assert errs == [] and r.pmid == "42371138" and r.source == "europepmc" and r.source_key == "PMID:42371138"
    assert r.title == "Effect of two types of time-restricted eating on weight in women with PCOS: a randomized controlled trial"
    assert "<h4>" not in r.abstract and r.abstract.startswith("Purpose Polycystic ovary syndrome")
    assert r.doi == "10.1007/s00394-026-04034-3" and r.first_author == "aminian" and r.year == 2026
    assert pre.pmid == "" and pre.doi == "10.1101/2025.05.21.25327203" and pre.journal == "medRxiv"
    assert pre.first_author == "von damm"


def test_malformed_entries_are_counted_not_lost():
    recs, errs = references.parse_file("e.bib", (BIB + "\n@article {broken,\n Title = {no end\n").encode())
    assert len(recs) == 2 and len(errs) == 1 and "broken" in errs[0]
    recs, errs = references.parse_file("x.nbib", (NBIB + "\nPMID- \nTI  - no id\n").encode())
    assert len(recs) == 2 and len(errs) == 1


def test_format_is_sniffed_when_the_extension_lies():
    assert references.parse_file("export.txt", NBIB.encode())[0][0].source == "pubmed"
    assert references.parse_file("export.txt", BIB.encode())[0][0].source == "europepmc"
    recs, errs = references.parse_file("notes.txt", b"hello")
    assert recs == [] and errs and "format" in errs[0]


def test_doi_and_title_normalisation():
    assert references.norm_doi("https://doi.org/10.1/ABC") == "10.1/abc" and references.norm_doi("doi:10.1/x ") == "10.1/x"
    assert references.norm_title("<i>Effect</i> of TRE: a  Trial.") == "effect of tre a trial"


# --- code-checker findings ---

def test_unclosed_quote_reports_an_error_instead_of_hanging():
    recs, errs = references.parse_bibtex('@article{k1, title = "abc}\n@article{k2, title={ok}, doi={10.1/k2}}')
    assert errs and [r.source_key for r in recs][-1] == "k2"


def test_unreadable_trailing_fields_are_reported():
    recs, errs = references.parse_bibtex('@article{k3, title = "a" # "b", doi = {10.1/z}}')
    assert len(recs) == 1 and recs[0].title == "a" and errs and "k3" in errs[0]


def test_entries_without_a_key_are_reported():
    recs, errs = references.parse_bibtex('@article{,\n title={x}}\n@article{k5\n title={y}}\n@article{k6, title={z}}')
    assert [r.source_key for r in recs] == ["k6"] and len(errs) == 2


def test_surname_from_medline_initials_form():
    assert references._surname("Smith J") == "smith" and references._surname("Tussing-Humphreys LA") == "tussing-humphreys"
    assert references._surname("Jane Smith") == "smith" and references._surname("von Damm, Isabela") == "von damm"


def test_escaped_braces_do_not_break_an_entry():
    recs, errs = references.parse_bibtex(r'@article{k8, title = {50\% of \{x\} women}, doi={10.1/k8}}')
    assert errs == [] and recs[0].title == "50% of x women" and recs[0].doi == "10.1/k8"


def test_more_doi_resolver_forms():
    assert references.norm_doi("http://www.doi.org/10.1/A") == "10.1/a" == references.norm_doi("doi.org/10.1/A")


def test_pubmed_keeps_every_language():
    r = references.parse_file("x.nbib", b"PMID- 5\nTI  - t\nLA  - fre\nLA  - eng\n")[0][0]
    assert r.language == "fre|eng"
