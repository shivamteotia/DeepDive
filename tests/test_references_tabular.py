"""CSV / Excel / RIS exports (Scopus, Web of Science, Embase, Cochrane, generic)."""
import io
from pathlib import Path

import pytest

import dedup
import references as refs

SCOPUS_CSV = (
    '﻿"Authors","Author full names","Title","Year","Source title","DOI","Abstract","Author Keywords",'
    '"Language of Original Document","Document Type","Source","EID"\n'
    '"Wu Y.; Huang B.","Wu, Yafei (58857058100); Huang, Bingsheng (60004343500)","Bmal1 deficiency in a PCOS-like rat model",'
    '"2025","Reproductive BioMedicine Online","10.1016/j.rbmo.2025.104972","Rats were studied.","PCOS; autophagy",'
    '"English","Article","Scopus","2-s2.0-105011037895"\n'
    '"Amin S.N.","Amin, Shaimaa Nasr (56438842000)","Insight Towards Induction of PCOS Phenotypes","2022",'
    '"Jordan Journal of Biological Sciences","https://doi.org/10.54319/JJBS/150510","","","English","Review","Scopus",'
    '"2-s2.0-85146710433"\n'
    '"No title row","","","2020","","","","","","","Scopus","2-s2.0-1"\n')

WOS_TXT = ("PT\tAU\tAF\tTI\tSO\tLA\tDT\tDE\tAB\tDI\tPY\tUT\tPM\n"
           "J\tLee, A; Kim, B\tLee, Ann; Kim, Bo\tMetformin in women with PCOS\tFERTILITY AND STERILITY\tEnglish\t"
           "Article\tPCOS; metformin\tA retrospective cohort of 300 women.\t10.1016/j.fertnstert.2020.01.001\t2020\t"
           "WOS:000512345600001\t32001234\n")

RIS = """TY  - JOUR
DB  - Embase
AU  - Smith, Jane
AU  - Doe, John
TI  - Time-restricted eating in women with PCOS
T2  - Some Journal
PY  - 2024
AB  - A randomized trial of 60 women.
DO  - https://doi.org/10.1/SHARED
LA  - English
KW  - PCOS
KW  - diet
AN  - L2031234567
ER  -

TY  - CONF
T1  - Ovarian morphology in adolescents
A1  - Kim, Bo
Y1  - 2021/03/04/
JF  - Proceedings of something
N2  - Ultrasound of 50 girls.
ID  - CN-01234567
ER  -

TY  - JOUR
AU  - Nobody, A
ER  -
"""


def test_scopus_csv_maps_columns_and_reports_rows_without_a_title():
    recs, errs = refs.parse_file("scopus.csv", SCOPUS_CSV.encode())
    assert len(recs) == 2 and len(errs) == 1 and "title" in errs[0]
    a, b = recs
    assert (a.source, a.source_key, a.year, a.doi) == ("scopus", "2-s2.0-105011037895", 2025, "10.1016/j.rbmo.2025.104972")
    assert a.authors == ["Wu, Yafei", "Huang, Bingsheng"] and a.first_author == "wu"  # full names, Scopus ids dropped
    assert a.abstract == "Rats were studied." and a.keywords == ["PCOS", "autophagy"] and a.language == "English"
    assert a.pub_types == ["Article"] and b.pub_types == ["Review"] and b.doi == "10.54319/jjbs/150510"


def test_web_of_science_tab_delimited_export():
    [r], errs = refs.parse_file("savedrecs.txt", WOS_TXT.encode())
    assert errs == [] and (r.source, r.source_key, r.pmid, r.year) == ("wos", "WOS:000512345600001", "32001234", 2020)
    assert r.authors == ["Lee, Ann", "Kim, Bo"] and r.abstract.startswith("A retrospective") and r.journal == "FERTILITY AND STERILITY"


def test_ris_records_with_tag_variants_and_a_bad_record():
    recs, errs = refs.parse_file("embase.ris", RIS.encode())
    assert len(recs) == 2 and len(errs) == 1
    a, b = recs
    assert (a.source, a.source_key, a.doi, a.year) == ("embase", "L2031234567", "10.1/shared", 2024)
    assert a.authors == ["Smith, Jane", "Doe, John"] and a.keywords == ["PCOS", "diet"] and a.journal == "Some Journal"
    assert (b.title, b.year, b.abstract, b.journal, b.source) == ("Ovarian morphology in adolescents", 2021,
                                                                  "Ultrasound of 50 girls.", "Proceedings of something", "ris")
    assert b.source_key == "CN-01234567" and b.pub_types == ["conf"]  # a RIS record type, not publication-type evidence


def test_excel_export_reads_the_first_sheet():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    for row in [l.split("\t") for l in WOS_TXT.strip().split("\n")]:
        wb.active.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    [r], errs = refs.parse_file("export.xlsx", buf.getvalue())
    assert errs == [] and r.title == "Metformin in women with PCOS" and r.year == 2020


def test_csv_with_semicolons_and_unknown_files():
    semi = "Title;Year;DOI;Abstract\nA study of things;2019;10.5/x;Text.\n"
    [r], _ = refs.parse_file("generic.csv", semi.encode())
    assert (r.source, r.title, r.year, r.doi) == ("csv", "A study of things", 2019, "10.5/x")
    assert r.source_key  # generic rows get a stable key
    recs, errs = refs.parse_file("notes.csv", b"name,colour\nann,blue\n")
    assert recs == [] and "title" in errs[0].lower()


def test_tabular_and_ris_records_deduplicate_with_pubmed_by_doi():
    from tests.test_api_references import NBIB
    recs = refs.parse_file("p.nbib", NBIB.encode())[0] + refs.parse_file("e.ris", RIS.encode())[0]
    groups, _ = dedup.group(recs)
    merged = [g for g in groups if len(g.members) > 1]
    assert len(merged) == 1 and merged[0].reason == "doi"


REAL = Path(__file__).parent.parent / "bibliography" / "scopus_export_Jul 16-2026_String_1.xlsx"


@pytest.mark.skipif(not REAL.exists(), reason="the real Scopus export is not in bibliography/")
def test_real_scopus_excel_export():
    pytest.importorskip("openpyxl")
    recs, errs = refs.parse_file(REAL.name, REAL.read_bytes())
    assert len(recs) == 142 and errs == []
    assert all(r.source == "scopus" and r.title and r.source_key.startswith("2-s2.0-") for r in recs)
    assert sum(bool(r.doi) for r in recs) > 120 and all(r.year for r in recs)


# --- code-checker findings ---

def test_csv_doubled_quotes_inside_fields_keep_the_row_aligned():
    text = '"Authors","Title","Year","EID"\n"Wu Y.","The ""lean"" PCOS, revisited","2025","2-s2.0-1"\n'
    [r], _ = refs.parse_file("scopus.csv", text.encode())
    assert (r.title, r.year, r.source_key, r.source) == ('The "lean" PCOS, revisited', 2025, "2-s2.0-1", "scopus")


def test_huge_or_malformed_csv_is_reported_not_raised():
    big = "Title,Abstract\nA,\"" + "x" * 300_000 + "\"\n"
    recs, _ = refs.parse_file("big.csv", big.encode())
    assert len(recs) == 1 and len(recs[0].abstract) == 300_000
    recs, errs = refs.parse_file("bad.csv", b"Title,Year\n\"unclosed,2020\n" + b"\x00" * 3)  # returns, never raises
    assert isinstance(recs, list) and isinstance(errs, list)


def test_ris_record_types_are_not_publication_type_evidence():
    ris = "TY  - JOUR\nTI  - A conference abstract\nM3  - Conference Abstract\nER  -\nTY  - JOUR\nTI  - Plain\nER  -\n"
    a, b = refs.parse_file("e.ris", ris.encode())[0]
    assert a.pub_types == ["Conference Abstract"] and b.pub_types == ["jour"]  # lower-case = not a PubMed-style type


def test_excel_with_a_far_away_cell_stays_small(monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.append(["Title", "Year"])
    wb.active.append(["A study", "2020"])
    wb.active.cell(row=200_000, column=5000, value="x")  # formatting debris far away
    buf = io.BytesIO()
    wb.save(buf)
    recs, _ = refs.parse_file("far.xlsx", buf.getvalue())
    assert [r.title for r in recs] == ["A study"]


def test_abbreviated_scopus_authors_give_the_surname():
    text = '"Authors","Title","Year"\n"Wu Y.; Huang B.","T one","2025"\n"Wu Y., Huang B.","T two","2025"\n'
    a, b = refs.parse_file("s.csv", text.encode())[0]
    assert a.first_author == "wu" and b.first_author == "wu"


def test_excel_csv_in_cp1252_and_utf16_tab_files():
    [r], _ = refs.parse_file("excel.csv", "Authors,Title,Year\nMüller A.,Café study,2020\n".encode("cp1252"))
    assert (r.title, r.first_author) == ("Café study", "müller")
    [w], _ = refs.parse_file("wos.txt", ("﻿" + WOS_TXT).encode("utf-16-le"))
    assert w.title == "Metformin in women with PCOS"


def test_nul_bytes_and_padded_ris_values_are_cleaned():
    [r], _ = refs.parse_file("n.csv", b"Title,Journal/Book\nA\x00b,Fertil Steril\n")
    assert r.title == "Ab" and r.journal == "Fertil Steril"
    ris = "TY  - JOUR\nTI  - T\nT2  -\n  Journal Name\nDB  - Ovid MEDLINE(R)\nAN  - 32001234\nER  -\n"
    [x], _ = refs.parse_file("m.ris", ris.encode())
    assert x.journal == "Journal Name" and x.pmid == "32001234" and x.source == "pubmed"
