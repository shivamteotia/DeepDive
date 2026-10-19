"""Group reference records from several exports into papers: same DOI, then same PMID/PMCID, then same normalised
title + first author + year (±1). Two records with different DOIs or different PMIDs never end up in one paper, and the
title step never joins a preprint with a journal article (same title, separate versions). Similar titles that are
not merged are returned as possible duplicates for the user to decide."""

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from references import Record, norm_title

TITLE_SIMILAR = 0.85  # possible-duplicate listing only (never merges); "X" vs "X: a pilot" scores 0.89
YEAR_SLACK = 1
BUCKET_TOKENS = 3
PREPRINT_SERVERS = ("medrxiv", "biorxiv", "research square", "ssrn", "preprints", "arxiv", "authorea", "osf")

Reason = Literal["doi", "pmid", "title_year_author", "single"]
_PRIORITY = ("doi", "pmid", "title_year_author")


@dataclass
class Group:
    members: list[int]  # indices into the input records
    reason: Reason


@dataclass
class PossibleDup:
    a: int  # record indices, a < b, in different groups
    b: int
    similarity: float


class _Groups:
    """Union-find that refuses a union which would put two different DOIs, or two different PMIDs, in one group."""

    def __init__(self, records: list[Record]):
        self.parent = list(range(len(records)))
        self.dois = [{r.doi} if r.doi else set() for r in records]
        self.pmids = [{r.pmid} if r.pmid else set() for r in records]
        self.reasons: list[set[str]] = [set() for _ in records]

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int, reason: str) -> None:
        a, b = self.find(i), self.find(j)
        if a == b or len(self.dois[a] | self.dois[b]) > 1 or len(self.pmids[a] | self.pmids[b]) > 1:
            return
        self.parent[b] = a
        self.dois[a] |= self.dois[b]
        self.pmids[a] |= self.pmids[b]
        self.reasons[a] |= self.reasons[b] | {reason}


def is_preprint(r: Record) -> bool:
    types = {t.lower() for t in r.pub_types}
    journal = r.journal.lower()
    return "preprint" in types or "misc" in types or any(s in journal for s in PREPRINT_SERVERS)


def _union_by(g: _Groups, keys: list[str], reason: str) -> None:
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k:
            if k in first:
                g.union(first[k], i, reason)
            else:
                first[k] = i


def group(records: list[Record]) -> tuple[list[Group], list[PossibleDup]]:
    g = _Groups(records)
    _union_by(g, [r.doi for r in records], "doi")
    _union_by(g, [r.pmid and f"pmid:{r.pmid}" for r in records], "pmid")
    _union_by(g, [r.pmcid and f"pmcid:{r.pmcid.upper()}" for r in records], "pmid")

    titles = [norm_title(r.title) for r in records]
    by_title: dict[tuple[str, str, bool], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        # No author or no year: "Correction" / "Editorial" items would merge on title alone, so they don't.
        if titles[i] and r.first_author and r.year is not None:
            by_title[(titles[i], r.first_author, is_preprint(r))].append(i)
    for idx in by_title.values():
        idx.sort(key=lambda i: records[i].year)
        anchor = idx[0]
        for y in idx[1:]:
            if records[y].year - records[anchor].year <= YEAR_SLACK:  # measured from the run's first year: no chaining
                g.union(anchor, y, "title_year_author")
            else:
                anchor = y

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        members[g.find(i)].append(i)
    groups = [Group(m, next((r for r in _PRIORITY if r in g.reasons[root]), "single") if len(m) > 1 else "single")
              for root, m in members.items()]

    # ponytail: pairwise within buckets of the first title words; fine for a few thousand records, use a token index
    # (e.g. MinHash) if one bucket grows to tens of thousands.
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(titles):
        if t:
            buckets[" ".join(t.split()[:BUCKET_TOKENS])].append(i)
    near: dict[tuple[int, int], PossibleDup] = {}
    for idx in buckets.values():
        for n, a in enumerate(idx):
            for b in idx[n + 1:]:
                ra, rb = g.find(a), g.find(b)
                if ra == rb or (min(ra, rb), max(ra, rb)) in near:
                    continue
                sim = SequenceMatcher(None, titles[a], titles[b]).ratio()
                if sim >= TITLE_SIMILAR:
                    near[(min(ra, rb), max(ra, rb))] = PossibleDup(a, b, round(sim, 3))
    return groups, list(near.values())


def pick_canonical(records: list[Record]) -> int:
    """Index of the record whose fields represent the paper: has an abstract, then PubMed, then the longest abstract,
    then has a DOI / PMID, then the title text — never input order, which follows random ids."""
    def key(i: int) -> tuple:
        r = records[i]
        return (bool(r.abstract), r.source == "pubmed", len(r.abstract), bool(r.doi), bool(r.pmid), r.title, r.source_key)

    return max(range(len(records)), key=key)
