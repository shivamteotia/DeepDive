"""Exports: a table (headers + rows) as CSV or Excel. Cells come from imported files and model output, so a value
that a spreadsheet would run as a formula (= + - @, tab, CR) is written as text with a leading apostrophe."""

import csv
import io

FORMULA_START = ("=", "+", "-", "@", "\t", "\r")  # CSV: spreadsheets evaluate all of these on open


def _cell(v, starts=FORMULA_START):
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return v
    v = str(v)
    return "'" + v if v.startswith(starts) else v


def to_csv(headers: list[str], rows: list[list]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([_cell(h) for h in headers])
    w.writerows([_cell(v) for v in row] for row in rows)
    return ("\ufeff" + buf.getvalue()).encode("utf-8")  # the BOM makes Excel read UTF-8


def to_xlsx(headers: list[str], rows: list[list], sheet: str) -> bytes:
    from openpyxl import Workbook  # lazy: only exports need it (installed with docling)
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    def cell(v):
        v = _cell(v, ("=",))  # xlsx: only "=" text becomes a formula; "-0.5" stays as written
        return ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v  # control chars break the file

    wb = Workbook(write_only=True)
    ws = wb.create_sheet(sheet[:31] or "Papers")
    ws.append([cell(h) for h in headers])
    for row in rows:
        ws.append([cell(v) for v in row])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
