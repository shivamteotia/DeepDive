"""Reference funnel service (DB work for the API): store reference imports, regroup references into papers, list papers,
split and merge. Every function takes the server-checked project id; nothing here trusts a client-supplied project."""

import hashlib
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

import db
import dedup
import extraction
import fulltext
import references
import screening as sc

TABS = ("all", "included", "review", "excluded", "duplicates", "fulltext")
_TAB_DECISION = {"included": "include", "review": "review", "excluded": "exclude"}
SCREEN_BATCH = 50  # papers written per commit during screen-all
AI_FIELDS = ("verdicts", "ai_decision", "decision", "decided_by", "disagrees", "human_reason")
FULLTEXT_FIELDS = ("fulltext_status", "fulltext_url", "fulltext_license", "document_id")  # only included papers get one
CLIP = {"doi": 255, "pmid": 20, "pmcid": 20, "first_author": 200, "language": 100}  # String(n) columns on Paper


class Conflict(ValueError):
    """The request is valid but the data says no (e.g. keep_article on a pair with no preprint)."""


def lock(s: Session, project_id: uuid.UUID) -> None:
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
    lock(s, project_id)
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


def _final():
    return func.coalesce(db.ScreeningDecision.human_decision, db.ScreeningDecision.ai_decision)


def _decision_join(q, version: int | None):
    """Join each paper's real (non-test) decision under this criteria version."""
    return q.join(db.ScreeningDecision, (db.ScreeningDecision.paper_id == db.Paper.id)
                  & (db.ScreeningDecision.criteria_version == version) & db.ScreeningDecision.test.is_(False))


def decision_fields(d: db.ScreeningDecision | None, blind: bool = False) -> dict:
    out = {"verdicts": d.verdicts if d else [], "ai_decision": d.ai_decision if d else None,
           "decision": (d.human_decision or d.ai_decision) if d else None, "decided_by": d.decided_by if d else None,
           "disagrees": bool(d and d.disagrees), "human_reason": d.human_reason if d else "",
           "blind_label": d.blind_label if d else None}
    return {k: v for k, v in out.items() if k not in AI_FIELDS} if blind else out


def list_papers(s: Session, project_id: uuid.UUID, tab: str, offset: int, limit: int, blind: bool = False) -> dict:
    """blind: the AI's verdicts and decisions (and the decision tab counts that would reveal them) are left out."""
    counts = _counts(s, project_id)
    version = latest_version(s, project_id)
    tab_counts = {"all": counts["papers"], "duplicates": counts["possible_duplicates"],
                  "fulltext": s.scalar(select(func.count()).select_from(db.Paper).where(
                      db.Paper.project_id == project_id, db.Paper.fulltext_status != "not_tried"))}
    if blind:
        tab_counts.pop("fulltext")
    else:
        by = dict(s.execute(_decision_join(select(_final(), func.count()).select_from(db.Paper), version)
                            .where(db.Paper.project_id == project_id).group_by(_final())).all())
        tab_counts |= {t: by.get(d, 0) for t, d in _TAB_DECISION.items()}
    if tab == "duplicates":
        rows = list(s.scalars(select(db.PossibleDuplicate).where(
            db.PossibleDuplicate.project_id == project_id, db.PossibleDuplicate.resolved.is_(False))
            .order_by(db.PossibleDuplicate.similarity.desc(), db.PossibleDuplicate.id).offset(offset).limit(limit)))
        items = [{**paper_out(s.get(db.Paper, d.paper_a)), "possible_duplicate": {
            "id": str(d.id), "other": paper_out(s.get(db.Paper, d.paper_b)), "similarity": d.similarity}} for d in rows]
        return {"total": tab_counts["duplicates"], "counts": tab_counts, "criteria_version": version,
                "references": counts["references"], "items": items}
    q = select(db.Paper).where(db.Paper.project_id == project_id)
    if tab == "fulltext":
        q = q.where(db.Paper.fulltext_status != "not_tried")
    if tab in _TAB_DECISION:
        q = _decision_join(q, version).where(_final() == _TAB_DECISION[tab])
    rows = list(s.scalars(q.order_by(db.Paper.created_at, db.Paper.id).offset(offset).limit(limit)))
    dec = _decisions(s, version, [p.id for p in rows])
    cols = {} if blind else paper_columns(s, project_id, [p.id for p in rows])  # extracted values hint at decisions
    return {"total": tab_counts[tab], "counts": tab_counts, "criteria_version": version, "references": counts["references"],
            "items": [_blinded(paper_out(p), blind) | decision_fields(dec.get(p.id), blind)
                      | ({} if blind else {"columns": cols.get(p.id, {})}) for p in rows]}


def _blinded(paper: dict, blind: bool) -> dict:
    """Blind mode: full-text fields go too (a fetched PDF means the paper was included)."""
    return {k: v for k, v in paper.items() if k not in FULLTEXT_FIELDS} if blind else paper


def _paper_in(s: Session, project_id: uuid.UUID, paper_id: uuid.UUID) -> db.Paper | None:
    p = s.get(db.Paper, paper_id)
    return p if p is not None and p.project_id == project_id else None


def _refs_of(s: Session, paper_id: uuid.UUID) -> list[db.Reference]:
    return list(s.scalars(select(db.Reference).where(db.Reference.paper_id == paper_id)))


def split(s: Session, project_id: uuid.UUID, paper_id: uuid.UUID) -> list[dict] | None:
    """One paper per reference, all pinned. None if the paper is not in this project; [] if nothing to split."""
    lock(s, project_id)
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
    lock(s, project_id)
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


# --- screening: criteria versions, test run, decisions, pending actions ---

def latest_version(s: Session, project_id: uuid.UUID) -> int | None:
    return s.scalar(select(func.max(db.CriteriaSet.version)).where(db.CriteriaSet.project_id == project_id))


def criteria_set(s: Session, project_id: uuid.UUID, version: int | None = None) -> db.CriteriaSet | None:
    version = latest_version(s, project_id) if version is None else version
    return s.scalar(select(db.CriteriaSet).where(db.CriteriaSet.project_id == project_id, db.CriteriaSet.version == version))


def criteria_out(cs: db.CriteriaSet) -> dict:
    return {"version": cs.version, "sentence": cs.sentence, "criteria": cs.criteria, "created_at": cs.created_at}


def save_criteria(s: Session, project_id: uuid.UUID, criteria: list[sc.Criterion], sentence: str | None = None) -> dict:
    """A new criteria version; sentence None keeps the previous version's sentence."""
    lock(s, project_id)
    prev = criteria_set(s, project_id)
    cs = db.CriteriaSet(project_id=project_id, version=(prev.version if prev else 0) + 1,
                        sentence=sentence if sentence is not None else (prev.sentence if prev else ""),
                        criteria=[c.model_dump() for c in criteria])
    s.add(cs)
    s.commit()
    return criteria_out(cs)


def _criteria(cs: db.CriteriaSet) -> list[sc.Criterion]:
    return [sc.Criterion(**c) for c in cs.criteria]


def _decisions(s: Session, version: int | None, paper_ids: list[uuid.UUID], test: bool = False) -> dict:
    if version is None or not paper_ids:
        return {}
    rows = s.scalars(select(db.ScreeningDecision).where(db.ScreeningDecision.criteria_version == version,
                                                        db.ScreeningDecision.test.is_(test),
                                                        db.ScreeningDecision.paper_id.in_(paper_ids)))
    return {d.paper_id: d for d in rows}


def _screenable(s: Session, project_id: uuid.UUID) -> list[db.Paper]:
    """Papers to screen: all but preprints the user replaced by their journal article."""
    return list(s.scalars(select(db.Paper).where(db.Paper.project_id == project_id, db.Paper.superseded_by.is_(None))
                          .order_by(db.Paper.created_at, db.Paper.id)))


def _test_items(s: Session, rows: list[db.ScreeningDecision]) -> list[dict]:
    return [{"paper": paper_out(s.get(db.Paper, d.paper_id)), "verdicts": d.verdicts, "ai_decision": d.ai_decision,
             "feedback": d.feedback} for d in rows]


def run_test(s: Session, project_id: uuid.UUID, parse: sc.Parse) -> dict | None:
    """Screen a TEST_SAMPLE of papers under the latest criteria (test rows, not decisions). Already tested under this
    version: the stored rows, at no cost. None if there are no criteria yet."""
    cs = criteria_set(s, project_id)
    if cs is None:
        return None
    criteria, papers = _criteria(cs), [paper_out(p) for p in _screenable(s, project_id)]
    out = {"version": cs.version, "estimate": sc.estimate(papers, criteria), "seconds": None}
    stored = list(s.scalars(select(db.ScreeningDecision).where(
        db.ScreeningDecision.project_id == project_id, db.ScreeningDecision.criteria_version == cs.version,
        db.ScreeningDecision.test.is_(True)).order_by(db.ScreeningDecision.id)))
    if stored:
        order = {p["id"]: n for n, p in enumerate(sc.test_sample(papers, criteria, seed=cs.version))}
        return out | {"items": _test_items(s, sorted(stored, key=lambda d: order.get(str(d.paper_id), len(order))))}
    sample = sc.test_sample(papers, criteria, seed=cs.version)
    start = time.monotonic()
    verdicts = sc.screen_many(parse, sample, criteria)
    rows = [db.ScreeningDecision(project_id=project_id, paper_id=uuid.UUID(p["id"]), criteria_version=cs.version,
                                 verdicts=[v.model_dump() for v in vs], ai_decision=sc.decide(criteria, vs, bool(p["abstract"])),
                                 test=True) for p, vs in zip(sample, verdicts, strict=True)]
    s.add_all(rows)
    s.commit()  # ponytail: two concurrent test runs of one version collide on the unique key (500); the UI runs one
    return out | {"items": _test_items(s, rows), "seconds": round(time.monotonic() - start, 1)}


def set_feedback(s: Session, project_id: uuid.UUID, paper_id: uuid.UUID, feedback: str) -> dict | None:
    """Thumbs up/down on a test row of the latest criteria version. None if that paper was not in the test."""
    d = _decisions(s, latest_version(s, project_id), [paper_id], test=True).get(paper_id)
    if d is None or d.project_id != project_id:
        return None
    d.feedback = feedback
    s.commit()
    return {"paper_id": str(paper_id), "criteria_version": d.criteria_version, "feedback": feedback}


def _history(s: Session, version: int, paper_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[db.ScreeningDecision]]:
    """Each paper's real decision rows up to this criteria version, oldest first."""
    out: dict[uuid.UUID, list[db.ScreeningDecision]] = {}
    if paper_ids:
        for d in s.scalars(select(db.ScreeningDecision).where(
                db.ScreeningDecision.paper_id.in_(paper_ids), db.ScreeningDecision.test.is_(False),
                db.ScreeningDecision.criteria_version <= version).order_by(db.ScreeningDecision.criteria_version)):
            out.setdefault(d.paper_id, []).append(d)
    return out


def _carry(rows: list[db.ScreeningDecision]) -> dict:
    """Human work from a paper's newest row (any version): it carries to the next version's row."""
    d = rows[-1] if rows else None
    return {"human_decision": d.human_decision if d else None, "human_reason": d.human_reason if d else "",
            "blind_label": d.blind_label if d else None}


def _current_row(s: Session, project_id: uuid.UUID, paper_id: uuid.UUID, version: int,
                 rows: list[db.ScreeningDecision]) -> db.ScreeningDecision:
    """The paper's row under this version, created (with the human work carried from its newest row) if missing."""
    if rows and rows[-1].criteria_version == version:
        return rows[-1]
    h = _carry(rows)
    d = db.ScreeningDecision(project_id=project_id, paper_id=paper_id, criteria_version=version, verdicts=[],
                             ai_decision=None, human_decision=h["human_decision"], human_reason=h["human_reason"] or "",
                             decided_by="human" if h["human_decision"] else "ai", disagrees=False,
                             blind_label=h["blind_label"], test=False)
    s.add(d)
    return d


def set_decision(s: Session, project_id: uuid.UUID, paper_id: uuid.UUID, changes: dict) -> dict | None:
    """changes: {"decision": include|exclude|None, "reason"} (None clears the human decision) and/or {"blind_label"}.
    Applies to the latest criteria version. None if the paper is not in this project; Conflict without criteria.
    A blind_label-only change answers without the AI's fields (the labeller must not see them)."""
    if _paper_in(s, project_id, paper_id) is None:
        return None
    version = latest_version(s, project_id)
    if version is None:
        raise Conflict("No screening criteria yet")
    d = _current_row(s, project_id, paper_id, version, _history(s, version, [paper_id]).get(paper_id, []))
    if "decision" in changes:
        d.human_decision = changes["decision"]
        d.human_reason = (changes.get("reason") or "") if d.human_decision else ""
        d.decided_by = "human" if d.human_decision else "ai"
        d.disagrees = bool(d.human_decision and d.ai_decision and d.human_decision != d.ai_decision)
    if "blind_label" in changes:
        d.blind_label = changes["blind_label"]
    s.commit()
    return {"paper_id": str(paper_id), "criteria_version": version} | decision_fields(d, blind="decision" not in changes)


def _same(a: sc.Criterion, b: dict) -> bool:
    """Same question to the model. An extracted field's exclude_values is the user's filter, not part of the question:
    changing only that re-applies the stored values."""
    return all(getattr(a, k) == b.get(k) for k in ("id", "kind", "check", "definition", "field", "value", "options"))


def _screen_plan(s: Session, project_id: uuid.UUID, version: int) -> tuple[list[sc.Criterion], list[tuple]]:
    """(criteria, todo). todo: (paper dict, copied verdicts, criteria to ask) for each paper without an AI decision
    under this version. AI yes/no verdicts from the paper's newest screened earlier version are copied when that
    criterion is unchanged since; metadata checks are free and always re-run (the paper's fields may have changed)."""
    criteria = _criteria(criteria_set(s, project_id, version))
    by_id = {c.id: c for c in criteria}
    papers = _screenable(s, project_id)
    history = _history(s, version, [p.id for p in papers])
    unchanged: dict[int, set[str]] = {}

    def same_since(v: int) -> set[str]:
        if v not in unchanged:
            old = {c["id"]: c for c in criteria_set(s, project_id, v).criteria}
            unchanged[v] = {c.id for c in criteria if c.check in ("ai", "extract") and c.id in old and _same(c, old[c.id])}
        return unchanged[v]

    todo = []
    for p in papers:
        rows = history.get(p.id, [])
        if rows and rows[-1].criteria_version == version and rows[-1].ai_decision is not None:
            continue
        src = next((d for d in reversed(rows) if d.criteria_version < version and d.ai_decision is not None), None)
        keep = same_since(src.criteria_version) if src else set()
        copied = {}
        for v in src.verdicts if src else []:
            c = by_id.get(v["criterion_id"])
            if c is None or c.id not in keep:
                continue
            if c.check == "extract" and v.get("value"):  # the model answered (even "not stated"): re-apply the filter
                again = sc.reapply_extract(c, sc.Verdict(**v))
                if again is not None:
                    copied[c.id] = again
            elif c.check == "ai" and v["verdict"] != "unclear":  # unclear is asked again: it may have been a failure
                copied[c.id] = sc.Verdict(**v)
        excluded = any(sc.excludes(by_id[k], v) for k, v in copied.items())  # still excluded: nothing to pay for
        todo.append((paper_out(p), copied, [] if excluded else [c for c in criteria if c.id not in copied]))
    return criteria, todo


def screen_estimate(s: Session, project_id: uuid.UUID, version: int) -> dict:
    _, todo = _screen_plan(s, project_id, version)
    def free_exclusion(p: dict, ask: list[sc.Criterion]) -> bool:  # metadata excludes it before any model call
        return any(c.check == "metadata" and (v := sc._meta_or_error(c, p)) is not None and sc.excludes(c, v) for c in ask)

    parts = [sc.estimate([p], [] if free_exclusion(p, ask) else ask) for p, _, ask in todo]
    return {"papers": len(todo), **{k: sum(e[k] for e in parts) for k in ("calls", "input_tokens", "output_tokens")}}


def _write_batch(sessions: sessionmaker, project_id: uuid.UUID, version: int, criteria: list[sc.Criterion],
                 batch: list[tuple], asked: list[list[sc.Verdict]]) -> Counter:
    """Write AI results onto each paper's row under this version, read fresh now: a decision the user made while we
    screened is kept, not overwritten or collided with."""
    counts: Counter = Counter()
    with sessions() as s:
        pids = [uuid.UUID(t[0]["id"]) for t in batch]
        alive = set(s.scalars(select(db.Paper.id).where(db.Paper.id.in_(pids))))
        history = _history(s, version, pids)
        for (paper, copied, ask), new, pid in zip(batch, asked, pids, strict=True):
            if pid not in alive:  # regrouped away by an import while we screened
                continue
            got = copied | {v.criterion_id: v for v in new}
            verdicts = [got[c.id] for c in criteria if c.id in got]
            d = _current_row(s, project_id, pid, version, history.get(pid, []))
            d.verdicts, d.ai_decision = [v.model_dump() for v in verdicts], sc.decide(criteria, verdicts, bool(paper["abstract"]))
            d.disagrees = bool(d.human_decision and d.human_decision != d.ai_decision)
            counts[d.ai_decision] += 1
        s.commit()
    return counts


def screen_all(sessions: sessionmaker, project_id: uuid.UUID, version: int, parse: sc.Parse) -> dict:
    """Screen every paper without an AI decision under this criteria version, committing every SCREEN_BATCH papers
    (a re-run continues where a crash stopped)."""
    with sessions() as s:
        criteria, todo = _screen_plan(s, project_id, version)
    counts: Counter = Counter()
    calls, lock = [0], threading.Lock()

    def counted(*args):  # the real number of model calls (metadata exclusions make none)
        with lock:
            calls[0] += 1
        return parse(*args)

    with ThreadPoolExecutor(max_workers=sc.SCREEN_WORKERS) as pool:
        for i in range(0, len(todo), SCREEN_BATCH):
            batch = todo[i:i + SCREEN_BATCH]
            asked = list(pool.map(lambda t: sc.screen_paper(counted, t[0], t[2]) if t[2] else [], batch))
            try:
                counts += _write_batch(sessions, project_id, version, criteria, batch, asked)
            except IntegrityError:  # a user decision or an import landed between our read and commit: re-read once
                counts += _write_batch(sessions, project_id, version, criteria, batch, asked)
    return {"screened": counts["include"] + counts["exclude"] + counts["review"], "include": counts["include"],
            "exclude": counts["exclude"], "review": counts["review"], "calls": calls[0]}


def action_out(a: db.PendingAction) -> dict:
    return {"id": str(a.id), "project_id": str(a.project_id), "tool": a.tool, "args": a.args,
            "criteria_version": a.criteria_version, "estimate": a.estimate, "status": a.status, "result": a.result,
            "error": a.error, "created_at": a.created_at}


class Invalid(ValueError):
    """The request is malformed for this data (e.g. extracting from the duplicates tab)."""


EXTRACT_TABS = ("all", "included", "review", "excluded", "fulltext")


class NotFound(LookupError):
    """A referenced row (e.g. an extraction column) is not in this project."""


def propose(s: Session, project_id: uuid.UUID, tool: str, args: dict | None = None) -> dict:
    """A pending action with its estimate; nothing runs until the owner confirms it. Conflict without criteria (an
    extraction on the "all" tab needs none); NotFound for a column of another project."""
    args = dict(args or {})
    version = latest_version(s, project_id)
    if tool == "extract_column":
        col = project_column(s, project_id, args.get("column_id"))
        if col is None:
            raise NotFound("Column not found")
        tab = args.get("tab") or "included"
        if tab not in EXTRACT_TABS:
            raise Invalid(f"Extract from one of: {', '.join(EXTRACT_TABS)}")
        if version is None and tab in _TAB_DECISION:
            raise Conflict("No screening decisions yet: extract from the 'all' tab, or screen first")
        targets = extract_targets(s, project_id, col.id, tab)
        args = {"column_id": str(col.id), "tab": tab, "paper_ids": [p["id"] for p, _ in targets]}  # never more on confirm
        estimate = {"papers": len(targets), "column": col.name, "tab": args["tab"],
                    **extraction.estimate([p for p, _ in targets], sum(bool(sha) for _, sha in targets))}
    else:
        if version is None:
            raise Conflict("No screening criteria yet")
        estimate = (screen_estimate(s, project_id, version) if tool == "screen_all"
                    else {"papers": len(included_without_pdf(s, project_id))})
    a = db.PendingAction(project_id=project_id, tool=tool, args=args, criteria_version=version, status="pending",
                         estimate=estimate)
    s.add(a)
    s.commit()
    return action_out(a)


# --- full text ---

def included_without_pdf(s: Session, project_id: uuid.UUID) -> list[db.Paper]:
    """Papers whose final decision (latest criteria) is include and that have no usable PDF yet (none, or one whose
    ingest failed: fetching again re-queues it)."""
    q = _decision_join(select(db.Paper), latest_version(s, project_id)).outerjoin(
        db.Document, db.Document.id == db.Paper.document_id).where(
        db.Paper.project_id == project_id, db.Paper.superseded_by.is_(None), _final() == "include",
        or_(db.Paper.document_id.is_(None), db.Document.status == "failed"))
    return list(s.scalars(q.order_by(db.Paper.created_at, db.Paper.id)))


def linked_elsewhere(s: Session, doc_id: uuid.UUID, paper_id: uuid.UUID) -> bool:
    """The document already is another paper's PDF (one PDF never stands for two papers)."""
    return s.scalar(select(db.Paper.id).where(db.Paper.document_id == doc_id, db.Paper.id != paper_id).limit(1)) is not None


def link_document(paper: db.Paper, doc_id: uuid.UUID, status: str, url: str | None = None,
                  license: str | None = None) -> None:
    paper.document_id, paper.fulltext_status, paper.fulltext_url, paper.fulltext_license = doc_id, status, url, license


def match_document(s: Session, project_id: uuid.UUID, doc_id: uuid.UUID, first_page: str) -> uuid.UUID | None:
    """Link an uploaded PDF to the included paper its first page names, if exactly one confidently matches and the
    document is not linked yet. Candidates: included papers still without a PDF (tens, not the whole export)."""
    if not first_page or s.scalar(select(db.Paper.id).where(db.Paper.document_id == doc_id).limit(1)) is not None:
        return None
    papers = [p for p in included_without_pdf(s, project_id) if p.document_id is None]
    hit = fulltext.match_upload(first_page, [{"id": str(p.id), "title": p.title, "doi": p.doi} for p in papers])
    if hit is None:
        return None
    paper = s.get(db.Paper, uuid.UUID(hit), with_for_update=True, populate_existing=True)  # fresh after the lock
    if paper is None or paper.document_id is not None:  # linked meanwhile (fetch or explicit upload)
        s.rollback()
        return None
    link_document(paper, doc_id, "uploaded")
    s.commit()
    return paper.id


def extracted_counts(s: Session, project_id: uuid.UUID) -> dict[str, dict[str, int]]:
    """For each extracted field of the latest criteria: how many screened papers have each value ("18 retrospective
    cohort, 2 prospective cohort"); "no answer" = a failed call or an unverified value."""
    cs = criteria_set(s, project_id)
    fields = {c["id"]: f'{c["name"]} ({c["id"]})' for c in (cs.criteria if cs else []) if c.get("check") == "extract"}
    if not fields:
        return {}
    out = {name: Counter() for name in fields.values()}
    rows = s.scalars(select(db.ScreeningDecision.verdicts).where(
        db.ScreeningDecision.project_id == project_id, db.ScreeningDecision.criteria_version == cs.version,
        db.ScreeningDecision.test.is_(False)))
    for verdicts in rows:
        for v in verdicts or []:
            if v["criterion_id"] in fields:
                out[fields[v["criterion_id"]]][v.get("value") or "no answer"] += 1
    return {name: dict(c.most_common()) for name, c in out.items()}


# --- data extraction columns ---

def column_out(c: db.ExtractColumn) -> dict:
    return {"id": str(c.id), "name": c.name, "instruction": c.instruction, "kind": c.kind, "options": c.options,
            "created_at": c.created_at}


def add_column(s: Session, project_id: uuid.UUID, col: extraction.Column) -> dict:
    """Conflict when the project already has a column of that name (values are shown by name)."""
    taken = {n.casefold() for n in s.scalars(select(db.ExtractColumn.name).where(db.ExtractColumn.project_id == project_id))}
    if " ".join(col.name.split()).casefold() in taken:
        raise Conflict(f"There is already a column named {col.name.strip()!r}")
    c = db.ExtractColumn(project_id=project_id, name=col.name.strip()[:200], instruction=col.instruction.strip(),
                         kind=col.kind, options=[o.strip() for o in col.options] if col.options else None)
    s.add(c)
    s.commit()
    return column_out(c)


def project_column(s: Session, project_id: uuid.UUID, column_id: uuid.UUID | str | None) -> db.ExtractColumn | None:
    try:
        c = s.get(db.ExtractColumn, uuid.UUID(str(column_id))) if column_id else None
    except ValueError:
        return None
    return c if c is not None and c.project_id == project_id else None


def list_columns(s: Session, project_id: uuid.UUID) -> list[dict]:
    rows = s.scalars(select(db.ExtractColumn).where(db.ExtractColumn.project_id == project_id)
                     .order_by(db.ExtractColumn.created_at))
    filled = dict(s.execute(select(db.ExtractValue.column_id, func.count()).where(
        db.ExtractValue.project_id == project_id).group_by(db.ExtractValue.column_id)).all())
    return [column_out(c) | {"filled": filled.get(c.id, 0)} for c in rows]


def column_values(s: Session, c: db.ExtractColumn) -> dict:
    rows = s.execute(select(db.ExtractValue, db.Paper.title).join(db.Paper, db.Paper.id == db.ExtractValue.paper_id)
                     .where(db.ExtractValue.column_id == c.id).order_by(db.Paper.created_at, db.Paper.id)).all()
    values = [{"paper_id": str(v.paper_id), "title": title, "value": v.value, "quote": v.quote, "reason": v.reason,
               "source": v.source} for v, title in rows]
    return column_out(c) | {"values": values, "counts": dict(Counter(v["value"] for v in values).most_common(30))}


def _papers_in_tab(s: Session, project_id: uuid.UUID, tab: str) -> list[db.Paper]:
    q = select(db.Paper).where(db.Paper.project_id == project_id, db.Paper.superseded_by.is_(None))
    if tab in _TAB_DECISION:
        q = _decision_join(q, latest_version(s, project_id)).where(_final() == _TAB_DECISION[tab])
    elif tab == "fulltext":
        q = q.where(db.Paper.fulltext_status != "not_tried")
    return list(s.scalars(q.order_by(db.Paper.created_at, db.Paper.id)))


def _naive(t):
    return t.replace(tzinfo=None) if t is not None and t.tzinfo else t


def extract_targets(s: Session, project_id: uuid.UUID, column_id: uuid.UUID, tab: str,
                    only: set[str] | None = None) -> list[tuple[dict, str | None]]:
    """(paper, sha of its ingested PDF or None) for papers in the tab that still need a value: none yet, or one read
    from the abstract before the paper's PDF finished ingesting. `only`: the paper ids a confirmed action covers."""
    papers = _papers_in_tab(s, project_id, tab)
    have = {v.paper_id: v for v in s.scalars(select(db.ExtractValue).where(db.ExtractValue.column_id == column_id))}
    ready = {d.id: d for d in s.scalars(select(db.Document).where(db.Document.project_id == project_id,
                                                                  db.Document.status == "ready"))}
    out = []
    for p in papers:
        if only is not None and str(p.id) not in only:
            continue
        doc = ready.get(p.document_id) if p.document_id else None
        v = have.get(p.id)
        upgrade = v is not None and v.source == "abstract" and doc is not None and \
            _naive(doc.updated_at) > _naive(v.updated_at)  # a PDF that yielded nothing is not billed again
        if v is None or upgrade:
            out.append((paper_out(p), doc.sha256 if doc else None))
    return out


def save_value(s: Session, project_id: uuid.UUID, column_id: uuid.UUID, paper_id: uuid.UUID,
               x: extraction.Extracted) -> None:
    v = s.scalar(select(db.ExtractValue).where(db.ExtractValue.column_id == column_id, db.ExtractValue.paper_id == paper_id))
    if v is None:
        v = db.ExtractValue(project_id=project_id, column_id=column_id, paper_id=paper_id)
        s.add(v)
    v.value, v.quote, v.reason, v.source = x.value, x.quote, x.reason, x.source
    v.updated_at = db.utcnow()  # even when nothing changed: the upgrade rule compares it with the document


def paper_columns(s: Session, project_id: uuid.UUID, paper_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict]:
    """{paper id: {column name: {value, source, quote}}} for a page of papers."""
    if not paper_ids:
        return {}
    rows = s.execute(select(db.ExtractValue, db.ExtractColumn.name).join(
        db.ExtractColumn, db.ExtractColumn.id == db.ExtractValue.column_id).where(
        db.ExtractValue.project_id == project_id, db.ExtractValue.paper_id.in_(paper_ids))).all()
    out: dict[uuid.UUID, dict] = {}
    for v, name in rows:
        out.setdefault(v.paper_id, {})[name] = {"value": v.value, "source": v.source, "quote": v.quote}
    return out
