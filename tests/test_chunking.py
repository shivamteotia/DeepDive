import json

import pytest

import chunking
from chunking import Piece


def words(s: str) -> int:
    return len(s.split())


def P(text, *headings, pages=(1,), table=False):
    return Piece(text=text, headings=list(headings), page_numbers=list(pages), content_type="table" if table else "text")


LONG = "word " * 60

ARTICLE = [
    P("Development and validation of a SCAP mortality model", pages=(1,)),
    P("Jane Doe, MD, Some Hospital", "Jane Doe, MD,", pages=(1,)),
    P("Background " + LONG, "Abstract", pages=(1,)),
    P("pneumonia; nomogram", "Keywords", pages=(2,)),
    P("We studied " + LONG, "Methods", pages=(3,)),
    P("AUC was 0.893 " + LONG, "Results", pages=(4,)),
    P("short tail", "Results", pages=(4,)),
    P("Supported by grant 123.", "Funding", pages=(9,)),
    P("The authors declare no competing interests.", "Competing interests", pages=(9,)),
    P("1. Smith J. Some paper. doi 10.1000/xyz", "References", pages=(10,)),
    P("Figure 1 shows the nomogram.", "Figure 1.", pages=(11,)),
]

REPORT = [
    P("National Guideline on Pneumonia", pages=(1,)),
    P("1 Introduction 2 Diagnosis 3 Treatment", "Table of contents", pages=(2,)),
    P("This guideline " + LONG, "Foreword", pages=(3,)),
    P(LONG, "Executive summary", pages=(4,)),
    P(LONG, "1 Introduction", pages=(5,)),
    P(LONG, "2 Diagnosis", pages=(8,)),
    P(LONG, "2.3 Severity scores", pages=(9,)),
    P("We thank the panel.", "Acknowledgements", pages=(40,)),
]


class FakeParse:
    """Stands in for ask.parse_llm: records calls; returns a value per output format, or raises."""

    def __init__(self, by_fmt):
        self.by_fmt, self.calls = by_fmt, []

    def __call__(self, instructions, input, fmt):
        self.calls.append((instructions, input, fmt))
        out = self.by_fmt.get(fmt)
        if isinstance(out, Exception):
            raise out
        return out(input) if callable(out) else out


def test_clear_article_is_classified_without_an_llm_call():
    parse = FakeParse({})
    doc_type, reasons = chunking.classify(ARTICLE, 11, words, parse)
    assert doc_type == "research_article" and parse.calls == []
    assert "abstract heading" in reasons and "references" in reasons


def test_clear_report_is_classified_without_an_llm_call():
    parse = FakeParse({})
    assert chunking.classify(REPORT, 45, words, parse)[0] == "report_guideline" and parse.calls == []


def test_no_signals_is_generic_without_an_llm_call():
    parse = FakeParse({})
    assert chunking.classify([P(LONG, "Overview"), P(LONG, "Details")], 3, words, parse) == ("generic", [])
    assert parse.calls == []


AMBIGUOUS = [P(LONG, "Abstract", pages=(1,)), P(LONG, "Executive summary"), P(LONG, "Contents"), P(LONG, "References")]


def test_ambiguous_document_asks_the_llm_with_quoted_text():
    guess = chunking.DocTypeGuess(doc_type="report_guideline", reason="numbered chapters")
    parse = FakeParse({chunking.DocTypeGuess: guess})
    doc_type, reasons = chunking.classify(AMBIGUOUS, 5, words, parse)
    assert doc_type == "report_guideline" and reasons[-1] == "llm: numbered chapters"
    (instructions, input, _fmt), = parse.calls
    assert "ignore" in instructions.lower() and "word" in json.loads(input)["document_start"]


@pytest.mark.parametrize("out", [None, RuntimeError("api down")])
def test_failed_llm_tie_break_falls_back_to_generic(out):
    assert chunking.classify(AMBIGUOUS, 5, words, FakeParse({chunking.DocTypeGuess: out}))[0] == "generic"


def test_ambiguous_without_an_llm_is_generic():
    assert chunking.classify(AMBIGUOUS, 5, words, None)[0] == "generic"


def test_take_tokens_stops_at_budget():
    assert chunking.take_tokens(["a b", "c d", "e f"], 4, words) == "a b\n\nc d"
