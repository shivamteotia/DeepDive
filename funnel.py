"""Reference funnel service (DB work for the API): store reference imports, regroup references into papers, list papers,
split and merge. Every function takes the server-checked project id; nothing here trusts a client-supplied project."""

import hashlib
import uuid
from types import SimpleNamespace

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session

import db
import dedup
import references

TABS = ("all", "duplicates", "fulltext")
CLIP = {"doi": 255, "pmid": 20, "pmcid": 20, "first_author": 200, "language": 20}  # String(n) columns on Paper


class Conflict(ValueError):
    """The request is valid but the data says no (e.g. keep_article on a pair with no preprint)."""


def _lock(s: Session, project_id: uuid.UUID) -> None:
    """Serialise imports/splits/merges of one project (Postgres row lock; SQLite has one writer anyway)."""
    s.execute(select(db.Project.id).where(db.Project.id == project_id).with_for_update())


def _records(refs: list[db.Reference]) -> list[references.Record]:
    return [references.Record(**r.fields, raw={}) for r in refs]


def _paper_fields(recs: list[references.Record], reason: str) -> dict:
    """Canonical fields: the best record first, gaps filled from the others; publication types from all of them."""
    best = recs[dedup.pick_canonical(recs)]
    ordered = [best] + [r for r in recs if r is not best]

    def first(attr: str):
        return next((getattr(r, attr) for r in ordered if getattr(r, attr)), getattr(best, attr))

    pub_types = list(dict.fromkeys(t for r in sorted(ordered, key=lambda r: r.source != "pubmed") for t in r.pub_types))
    out = {"title": best.title, "abstract": best.abstract, "year": first("year"), "doi": first("doi"),
           "pmid": first("pmid"), "pmcid": first("pmcid"), "journal": first("journal"), "first_author": best.first_author,
           "authors": best.authors, "language": first("language"), "pub_types": pub_types,
           "sources": sorted({r.source for r in recs}), "dedup_reason": reason}
    return out | {k: (out[k] or "")[:n] for k, n in CLIP.items()}


def paper_out(p: db.Paper) -> dict:
    return {"id": str(p.id), "title": p.title, "abstract": p.abstract, "year": p.year, "doi": p.doi, "pmid": p.pmid,
            "pmcid": p.pmcid, "journal": p.journal, "first_author": p.first_author, "authors": p.authors,
            "language": p.language, "pub_types": p.pub_types, "sources": p.sources, "dedup_reason": p.dedup_reason,
            "document_id": str(p.document_id) if p.document_id else None, "fulltext_status": p.fulltext_status,
            "fulltext_url": p.fulltext_url, "fulltext_license": p.fulltext_license,
            "superseded_by": str(p.superseded_by) if p.superseded_by else None}


def _drop_papers(s: Session, ids: set[uuid.UUID]) -> None:
    if ids:
        s.execute(update(db.Paper).where(db.Paper.superseded_by.in_(ids)).values(superseded_by=None))
        s.execute(delete(db.PossibleDuplicate).where(or_(db.PossibleDuplicate.paper_a.in_(ids),
                                                         db.PossibleDuplicate.paper_b.in_(ids))))
        s.execute(delete(db.Paper).where(db.Paper.id.in_(ids)))


def _save_group(s: Session, project_id: uuid.UUID, refs: list[db.Reference], recs: list[references.Record],
                idx: list[int], reason: str, by_members: dict) -> uuid.UUID:
    """The paper for these free references: the existing row if it has exactly these members, else a new one."""
    fields = _paper_fields([recs[i] for i in idx], reason)
    pid = by_members.get(frozenset(refs[i].id for i in idx))
    if pid is None:
        paper = db.Paper(project_id=project_id, **fields)
        s.add(paper)
        s.flush()
        return paper.id
    paper = s.get(db.Paper, pid)  # same members as before: keep the row (and its decisions), refresh its fields
    for k, v in fields.items():
        setattr(paper, k, v)
    return pid


def regroup(s: Session, project_id: uuid.UUID) -> None:
    """Rebuild papers from all the project's references. Pinned references (user splits/merges) never move: a dedup
    group touching one pinned paper adds its free references to that paper; a group touching several (the user split
    them) keeps its free references as their own paper, flagged as a possible duplicate of each. A paper whose member
    set is unchanged keeps its id (and so its screening decisions).
    ponytail: regroups the whole project per import; make it incremental if projects reach tens of thousands."""
    refs = list(s.scalars(select(db.Reference).where(db.Reference.project_id == project_id).order_by(db.Reference.id)))
    recs = _records(refs)
    groups, near = dedup.group(recs)
    old: dict[uuid.UUID, set[uuid.UUID]] = {}
    for r in refs:
        if r.paper_id is not None and not r.pinned:
            old.setdefault(r.paper_id, set()).add(r.id)
    pinned_papers = {r.paper_id for r in refs if r.pinned and r.paper_id is not None}
    by_members = {frozenset(m): pid for pid, m in old.items() if pid not in pinned_papers}
    kept: set[uuid.UUID] = set(pinned_papers)
    paper_of: list[uuid.UUID | None] = [r.paper_id if r.pinned else None for r in refs]
    flagged: list[tuple[uuid.UUID, uuid.UUID, float]] = []
    touched: set[uuid.UUID] = set()
    for g in groups:
        anchors = {refs[i].paper_id for i in g.members if refs[i].pinned}
        free = [i for i in g.members if not refs[i].pinned]
        if not free:
            continue
        if len(anchors) == 1:
            pid = next(iter(anchors))
            touched.add(pid)
        else:
            pid = _save_group(s, project_id, refs, recs, free, g.reason if len(free) > 1 else "single", by_members)
            flagged += [(pid, a, 1.0) for a in anchors]
        kept.add(pid)
        for i in free:
            refs[i].paper_id = pid
            paper_of[i] = pid
    s.flush()
    for pid in touched:  # a pinned paper that gained references: refresh its fields from all of them
        members = [r for r in refs if r.paper_id == pid]
        for k, v in _paper_fields(_records(members), "manual").items():
            setattr(s.get(db.Paper, pid), k, v)
    _drop_papers(s, set(old) - kept)

    dismissed = {frozenset((d.paper_a, d.paper_b)) for d in s.scalars(select(db.PossibleDuplicate).where(
        db.PossibleDuplicate.project_id == project_id, db.PossibleDuplicate.resolved))}
    s.execute(delete(db.PossibleDuplicate).where(db.PossibleDuplicate.project_id == project_id,
                                                 db.PossibleDuplicate.resolved.is_(False)))
    pairs = {frozenset((paper_of[d.a], paper_of[d.b])): d.similarity for d in near}
    pairs |= {frozenset((a, b)): sim for a, b, sim in flagged}
    for pair, sim in pairs.items():
        if len(pair) == 2 and None not in pair and pair not in dismissed:
            a, b = sorted(pair, key=str)
            s.add(db.PossibleDuplicate(project_id=project_id, paper_a=a, paper_b=b, similarity=sim))
    s.flush()


def _counts(s: Session, project_id: uuid.UUID) -> dict:
    papers = s.scalar(select(func.count()).select_from(db.Paper).where(db.Paper.project_id == project_id))
    refs = s.scalar(select(func.count()).select_from(db.Reference).where(db.Reference.project_id == project_id))
    dups = s.scalar(select(func.count()).select_from(db.PossibleDuplicate).where(
        db.PossibleDuplicate.project_id == project_id, db.PossibleDuplicate.resolved.is_(False)))
    return {"papers": papers, "references": refs, "possible_duplicates": dups}


def import_summary(s: Session, imp: db.ReferenceImport, already: bool) -> dict:
    c = _counts(s, imp.project_id)
    return {"import": {"id": str(imp.id), "filename": imp.filename, "records": imp.record_count},
            "records": imp.record_count, "errors": imp.errors, "already_imported": already, "papers": c["papers"],
            "merged": c["references"] - c["papers"], "possible_duplicates": c["possible_duplicates"]}


def import_file(s: Session, project_id: uuid.UUID, filename: str, data: bytes) -> tuple[dict, bool]:
    """(summary, created). Raises ValueError when nothing in the file can be read."""
    sha = hashlib.sha256(data).hexdigest()
    _lock(s, project_id)
    existing = s.scalar(select(db.ReferenceImport).where(db.ReferenceImport.project_id == project_id,
                                                         db.ReferenceImport.sha256 == sha))
    if existing is not None:
        return import_summary(s, existing, True), False
    recs, errs = references.parse_file(filename, data)
    if not recs:
        raise ValueError(errs[0] if errs else f"{filename}: no records found")
    imp = db.ReferenceImport(project_id=project_id, filename=filename[:255], sha256=sha, record_count=len(recs),
                             errors=errs)
    s.add(imp)
    s.flush()
    for r in recs:
        s.add(db.Reference(project_id=project_id, import_id=imp.id, source=r.source, source_key=r.source_key[:255],
                           fields=r.model_dump(exclude={"raw"}), raw=r.raw))
    s.flush()
    regroup(s, project_id)
    s.commit()
    return import_summary(s, imp, False), True


def list_papers(s: Session, project_id: uuid.UUID, tab: str, offset: int, limit: int) -> dict:
    counts = _counts(s, project_id)
    tab_counts = {"all": counts["papers"], "duplicates": counts["possible_duplicates"],
                  "fulltext": s.scalar(select(func.count()).select_from(db.Paper).where(
                      db.Paper.project_id == project_id, db.Paper.fulltext_status != "not_tried"))}
    if tab == "duplicates":
        rows = list(s.scalars(select(db.PossibleDuplicate).where(
            db.PossibleDuplicate.project_id == project_id, db.PossibleDuplicate.resolved.is_(False))
            .order_by(db.PossibleDuplicate.similarity.desc(), db.PossibleDuplicate.id).offset(offset).limit(limit)))
        items = [{**paper_out(s.get(db.Paper, d.paper_a)), "possible_duplicate": {
            "id": str(d.id), "other": paper_out(s.get(db.Paper, d.paper_b)), "similarity": d.similarity}} for d in rows]
        return {"total": tab_counts["duplicates"], "counts": tab_counts, "items": items}
    q = select(db.Paper).where(db.Paper.project_id == project_id)
    if tab == "fulltext":
        q = q.where(db.Paper.fulltext_status != "not_tried")
    rows = s.scalars(q.order_by(db.Paper.created_at, db.Paper.id).offset(offset).limit(limit))
    return {"total": tab_counts[tab], "counts": tab_counts, "items": [paper_out(p) for p in rows]}


def _paper_in(s: Session, project_id: uuid.UUID, paper_id: uuid.UUID) -> db.Paper | None:
    p = s.get(db.Paper, paper_id)
    return p if p is not None and p.project_id == project_id else None


def _refs_of(s: Session, paper_id: uuid.UUID) -> list[db.Reference]:
    return list(s.scalars(select(db.Reference).where(db.Reference.paper_id == paper_id)))


def split(s: Session, project_id: uuid.UUID, paper_id: uuid.UUID) -> list[dict] | None:
    """One paper per reference, all pinned. None if the paper is not in this project; [] if nothing to split."""
    _lock(s, project_id)
    paper = _paper_in(s, project_id, paper_id)
    if paper is None:
        return None
    refs = _refs_of(s, paper.id)
    if len(refs) < 2:
        return []
    out = []
    for r in refs:
        new = db.Paper(project_id=project_id, **_paper_fields(_records([r]), "manual"))
        s.add(new)
        s.flush()
        r.paper_id, r.pinned = new.id, True
        out.append(new)
    _drop_papers(s, {paper.id})
    s.commit()
    return [paper_out(p) for p in out]


def resolve_duplicate(s: Session, project_id: uuid.UUID, dup_id: uuid.UUID, action: str) -> dict | None:
    """merge: paper_b's references join paper_a (pinned, reason manual); dismiss: hide the pair for good;
    keep_article: the preprint of the pair is marked superseded by the journal article (kept, not screened)."""
    _lock(s, project_id)
    d = s.get(db.PossibleDuplicate, dup_id)
    if d is None or d.project_id != project_id or d.resolved:
        return None
    if action == "dismiss":
        d.resolved = True
        s.commit()
        return {"resolved": True}
    if action == "keep_article":
        pair = [s.get(db.Paper, d.paper_a), s.get(db.Paper, d.paper_b)]
        pre = [p for p in pair if dedup.is_preprint(SimpleNamespace(pub_types=p.pub_types, journal=p.journal))]
        if len(pre) != 1:
            raise Conflict("Can't tell which one is the preprint: neither or both look like preprints. "
                           "Merge or dismiss this pair instead.")
        article = pair[1] if pre[0] is pair[0] else pair[0]
        pre[0].superseded_by, d.resolved = article.id, True
        s.commit()
        return {"paper": paper_out(article), "superseded": paper_out(pre[0])}
    a = s.get(db.Paper, d.paper_a)
    refs = _refs_of(s, d.paper_a) + _refs_of(s, d.paper_b)
    for r in refs:
        r.paper_id, r.pinned = a.id, True
    for k, v in _paper_fields(_records(refs), "manual").items():
        setattr(a, k, v)
    _drop_papers(s, {d.paper_b})
    s.commit()
    return {"paper": paper_out(a)}
