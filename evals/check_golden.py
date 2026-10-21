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
_DASHES = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2009": " ", "\u202f": " ", "\u00a0": " "})


def norm(text: str) -> str:
    return " ".join(text.replace("\ufffe", "").replace("\u00ad", "").translate(_DASHES).split()).casefold()


def variants(text: str) -> tuple[str, str]:
    """pdfium marks a line-break hyphen with U+FFFE: a soft hyphen ("cur-rently") or a real one ("single-center")."""
    return norm(text), norm(text.replace("\ufffe", "-"))


@cache
def pages(filename: str) -> dict[int, str]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(PAPERS / filename))
    try:
        return {i + 1: " ".join(variants(doc[i].get_textpage().get_text_range())) for i in range(len(doc))}
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
    if sources and (case.get("facts") or case.get("facts_all")):
        everything = " ".join(t for s in sources for t in pages(s).values())
        if case.get("facts") and not any(norm(f) in everything for f in case["facts"]):
            out.append("no fact appears in the source PDFs")
        out += [f"fact {f!r} not in the source PDFs" for f in case.get("facts_all", []) if norm(f) not in everything]
    return out


def main() -> int:
    cases = json.loads((HERE / "golden.json").read_text(encoding="utf-8"))["cases"]
    failed = 0
    for c in cases:
        for p in problems(c):
            failed += 1
            print(f"  FAIL {c['id']}: {p}")
    checked = sum(1 for c in cases if c.get("quote") or ((c.get("facts") or c.get("facts_all")) and c.get("source")))
    print(f"checked {checked} of {len(cases)} cases against the PDFs: {failed} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
