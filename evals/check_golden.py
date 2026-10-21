"""Check the golden set against the papers themselves (free, local): every case with a `quote` must find it verbatim on
its `page` of its first source PDF, and at least one of its `facts` must appear in its source PDFs. Whitespace, case,
dashes and PDF hyphenation artifacts are normalised; nothing else. Run after any edit to evals/golden.json.

    uv run python -m evals.check_golden          # exit 1 if a case fails
"""

import json
import sys
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS = HERE.parent / "DATA" / "Papers"
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", " ": " ", " ": " ", " ": " "})


def norm(text: str) -> str:
    return " ".join(text.replace("￾", "").replace("­", "").translate(_DASHES).split()).casefold()


@cache
def pages(filename: str) -> dict[int, str]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(PAPERS / filename))
    try:
        return {i + 1: norm(doc[i].get_textpage().get_text_range()) for i in range(len(doc))}
    finally:
        doc.close()


def problems(case: dict) -> list[str]:
    """Why a case fails the check ([] when it passes, or when it has nothing to check)."""
    out = []
    sources = case.get("source") or []
    missing = [s for s in sources if not (PAPERS / s).exists()]
    if missing:
        return [f"missing PDF(s): {', '.join(missing)}"]
    if case.get("quote"):
        if not sources:
            return ["a quote needs a source"]
        text = pages(sources[0]).get(case.get("page"), "")
        if norm(case["quote"]) not in text:
            found = [p for p, t in pages(sources[0]).items() if norm(case["quote"]) in t]
            out.append(f"quote not on page {case.get('page')} of {sources[0]}" + (f" (found on {found})" if found else ""))
    if case.get("facts") and sources:
        everything = " ".join(t for s in sources for t in pages(s).values())
        if not any(norm(f) in everything for f in case["facts"]):
            out.append("no fact appears in the source PDFs")
    return out


def main() -> int:
    cases = json.loads((HERE / "golden.json").read_text(encoding="utf-8"))["cases"]
    failed = 0
    for c in cases:
        for p in problems(c):
            failed += 1
            print(f"  FAIL {c['id']}: {p}")
    checked = sum(1 for c in cases if c.get("quote") or (c.get("facts") and c.get("source")))
    print(f"checked {checked} of {len(cases)} cases against the PDFs: {failed} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
