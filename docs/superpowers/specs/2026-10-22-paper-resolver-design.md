# Paper resolver (roadmap P1, §4.1) — design

## Why
Measured on golden v2 (2026-10-21/22): wrong-paper answers survive the verifier because it only sees passages.
i01/i04/i05 ask about a paper that is not in the project and get another paper's numbers, with the asked-about
name attached ("the NLP study included 455 patients" from the pneumonia paper). x03/x04 (comparisons) miss one
paper's evidence. The verifier cannot know which paper "the NLP study" is; the fix is to resolve the named paper
before retrieval.

## What
Before `research.run`, for project-document runs (not PubMed, not web runs):
1. **Cards** — one line per ready paper: file name, the front-matter chunk (title, authors) and the abstract
   chunk's search context (what the study is). Built at query time with one Qdrant `retrieve` of the
   deterministic point ids for chunk 0 and 1; no re-ingest, no new table. Paper text is untrusted data.
2. **Resolve** — one fast-model call (Pydantic-validated): does the question name or describe specific
   paper(s)? which listed files match? which named papers match none?
3. **Route**
   - names nothing → unchanged (search all ready papers).
   - matches → search only the matched papers; with 2+ matches, search each paper separately so every one
     contributes evidence (comparison coverage).
   - names only papers the project does not have → no search; the answer is a gap: "This project has no
     paper matching: X." (needs_review, no claims).
   - some matched, some not → search the matched ones; the gap notes the missing ones.
4. **Trace** — a `resolve` step in the run log (matched files, unmatched names).

Fail open: no cards, a malformed resolver reply, or an error → today's behaviour.

## As built (measured on the 68 golden questions, each check run twice)
- Cards: chunks 0-3 (an alias like "Criticality Index" first appears in chunk 2), 600 chars each, plus the
  front matter's search context (first author, year, journal). Papers are shown as P1, P2... (file names can repeat).
- Model: the main model. The fast model refused or misrouted 12 of 49 answerable questions.
- A narrowed search needs two agreeing replies. A mixed reply (some named studies matched, some not) searches
  everything: it was usually a missed match.
- A refusal needs no distinctive word of the missing name on any card (prompt rules alone let the model refuse
  "the ICU epilepsy model" beside the epilepsy paper), and every ready paper must have a card.
- "Which studies / what papers ..." questions are never narrowed (x07 was routed to one of its two papers).
- Result: 0 answerable questions refused or misrouted; 1 of 3 isolation questions refused (i01); the other two
  search everything as before, and the verifier is the remaining check there.

## Non-goals (for now)
DOI/alias tables, a stored registry, section filters, native BM25, rerankers (roadmap P1b/P2).

## Risk and how it is measured
A false "no match" refuses an answerable question. Measured by the e2e eval: answerable det.completed,
det.fact and det.evidence_hit must not drop; det.abstain (isolation) and det.attribution should rise.
