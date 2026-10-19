# Section outline (heading hierarchy + canonical sections) — design

Date: 2026-10-19. Status: user asked to "just start" (no Q&A); calls below marked as mine.

## Why

Docling gives every PDF heading the same level, so each chunk carries only its nearest heading. Stored paths today
(collection 2026-10-19):

- JPan2023: `Study design`, `Patients`, `Data collection`, `Statistical analysis`, `Basic demographic characteristics`,
  `Construction of the predictive model` all sit beside `Discussion`; the paper has no Methods/Results heading, so
  nothing says they are Methods or Results.
- EA2022: `Figure 1.` … `Figure 4.` are treated as section headings.
- A 37-heading government report is one flat list.

User's intent: sort the headings, build parent/child relations between headings and subheadings, and give every
chunk that structure, so retrieval (and the agent reading evidence) knows which part of a paper a chunk is from —
"Methods > Statistical analysis", not "Statistical analysis".

## What gets built

One new pure module `sections.py` (injected `parse`, like `chunking.py`) and a hook in `chunking.build_chunks`:

1. **Collect** the document's heading occurrences in reading order: consecutive pieces with the same last heading are
   one occurrence; each has its text, first page and the first ~25 words under it.
2. **Outline (LLM, one call per document, `OPENAI_CONTEXT_MODEL` / fast model):** for each occurrence the model
   returns `is_heading` (false for captions, running headers, author names Docling mislabelled), `level` (1-4),
   `parent` (index of an earlier occurrence or null) and `section`, one of
   `front_matter, abstract, introduction, methods, results, discussion, conclusion, declarations, references,
   appendix, other`. Headings are JSON-quoted data in the prompt.
3. **Validate and repair (deterministic, never re-asks):** missing / out-of-range entries become level-1 `other`;
   section numbering ("2", "2.1", "2.1.3", "IV.") wins over the model for level and parent; a parent must be an
   earlier real heading with a lower level, otherwise the nearest such earlier heading is used, otherwise it is a
   root; level = parent level + 1 (max 4); a child with section `other` inherits its parent's section.
4. **Paths:** each real heading's path is its ancestors + itself. A root whose own text does not name its section
   (e.g. "Study design" → methods) gets the canonical section name prepended as a virtual parent
   ("Methods > Study design"); sections `other`, `front_matter` get no virtual parent.
5. **Misplaced content:** a non-heading occurrence's pieces take the path and section of the previous real heading,
   and its former heading text is prepended to the piece text (a caption "Figure 1." stays in the evidence).
6. **Chunks** carry `section_headings` = full path and a new payload field `section` (keyword index created with the
   collection). The label line becomes `Section: Methods > Statistical analysis`, so the path is embedded and
   keyword-indexed; the agent sees it in every evidence passage. `CHUNKER_VERSION` 2 → 3.

Order in `build_chunks`: classify (unchanged, raw pieces) → outline → regroup → label → context.

## Fallbacks (my calls)

- No model, a failed call, or more than `MAX_OUTLINE_HEADINGS = 300` occurrences: numbering-only repair; section from
  heading-name regexes (abstract, introduction/background, methods/materials, results, discussion, conclusion(s),
  references) else `other`. A document with no numbering and no model keeps today's flat paths.
- Outline failure never fails ingestion.

## Non-goals (for now)

- Section-filtered or section-boosted search (`ask.search(sections=...)`) and planner routing to sections: next step,
  once the eval shows the outline helps. The `section` payload index makes it cheap to add.
- Re-chunking boundaries by section (HybridChunker boundaries stay); tables/figure extraction changes.

## Constraints

- Hard rules in CLAUDE.md: PDF text is data (JSON-quoted), Pydantic-validated LLM output, model names from env,
  no new dependencies, idempotent point IDs unchanged.
- Existing collections must be re-chunked: `ingest.py --reindex-to <new>`, then `--sync-counts`.

## Done when

- Unit tests pass (`uv run pytest -m "not live"`), code-checker clean.
- Re-chunked collection: `evaluate.py` recall@k / MRR not below run 8 (0.95 / 0.85); e2e `evals.run` + `evals.compare`
  vs run 8 not FAIL. (Paid: confirm with the user before running.)
