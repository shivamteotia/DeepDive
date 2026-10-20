// Screening funnel UI: funnel counts, criteria cards, 20-paper test, table + review queue, blind labelling, and the
// screening assistant chat with Confirm cards for paid steps. Uses app.js globals ($, h, api, state, toast).
// All server text (titles, abstracts, quotes, agent replies) is rendered with textContent via h().
"use strict";

const SC_PAGE = 100;
const SC_TABS = [["all", "All"], ["included", "Included"], ["review", "Needs review"], ["excluded", "Excluded"],
  ["duplicates", "Possible duplicates"], ["fulltext", "Full text"]];
const ICON = { yes: "✅", no: "❌", unclear: "❔" };

const DECISION_OF_TAB = { included: "include", review: "review", excluded: "exclude" };

const sc = {
  tab: "all", blind: false, criteria: null, listing: null, items: [], test: null,
  gen: 0,            // bumped whenever what the list shows changes (project, tab, blind); stale responses are dropped
  moved: 0,          // loaded papers a decision moved out of the current tab (the server's offsets shift by this many)
  queue: null,       // {i, busy} while the review queue is open; it walks sc.items
  testing: false,
  messages: [],      // agent chat of the current conversation
  sending: false,
};

// A verdict "yes" means the criterion's condition holds; whether that is good depends on the criterion's kind.
function verdictIcon(c, v) {
  if (!v) return "–";
  if (c.check === "extract") return `${ICON[v.verdict] || "❔"} ${v.value || "no answer"}`;  // the extracted value itself
  return ICON[v.verdict] || "❔";
}

function checkLabel(c) {
  if (c.check === "metadata") return `metadata · ${c.field} ${c.value}`;
  if (c.check === "extract") return `AI extracts · dropping ${(c.exclude_values || []).join(", ") || "nothing"}`;
  return "AI reads abstract";
}

// "Study design: 9 cross-sectional, 4 retrospective cohort, …" over a list of papers' verdicts.
function valueCounts(c, verdictLists) {
  const n = new Map();
  for (const vs of verdictLists) {
    const v = (vs || []).find((x) => x.criterion_id === c.id);
    if (v) n.set(v.value || "no answer", (n.get(v.value || "no answer") || 0) + 1);
  }
  return [...n].sort((a, b) => b[1] - a[1]).map(([k, x]) => `${x} ${k}`).join(", ");
}

function decisionLabel(p) {
  if (!p.decision) return "Not screened";
  const word = { include: "Included", exclude: "Excluded", review: "Needs review" }[p.decision];
  return p.decided_by === "human" ? `${word} · you` : `${word} · AI`;
}

// ---------- loading ----------
// Background refreshes (a run finished, a chat turn, an upload) must not reset the list under an open review queue:
// then only the counts and criteria are refreshed.
async function scRefresh() {
  if (!sc.queue) return scLoad();
  if ($("#screening").hidden) return;
  const projectId = state.projectId;
  const gen = sc.gen;
  const [criteria, listing] = await Promise.all([
    api("GET", `/api/v1/projects/${projectId}/criteria`).catch(() => null),
    scFetch(0),
  ]);
  if (gen !== sc.gen || projectId !== state.projectId) return;
  sc.criteria = criteria;
  sc.listing = { ...sc.listing, counts: listing.counts, references: listing.references };
  scRenderFunnel();
  scRenderTabs();
}

async function scLoad() {
  if ($("#screening").hidden) return;
  const gen = ++sc.gen;
  const projectId = state.projectId;
  const [criteria, listing] = await Promise.all([
    api("GET", `/api/v1/projects/${projectId}/criteria`).catch(() => null),
    scFetch(0),
  ]);
  if (gen !== sc.gen || projectId !== state.projectId) return;  // a newer load (tab, blind, project) owns the view
  sc.criteria = criteria;
  sc.listing = listing;
  sc.items = listing.items;
  sc.moved = 0;
  scRender();
}

// Blind labelling shows only the labelling sample or all papers, never a tab that reveals the AI's decisions.
const scTab = () => (sc.blind ? sc.blindTab || "sample" : sc.tab);

async function scMakeSample() {
  try {
    const out = await api("POST", `/api/v1/projects/${state.projectId}/label-sample`, { size: 150 });
    toast(`Labelling sample ready: ${out.size} papers, shuffled. Label them blind with I / E.`);
    sc.blindTab = "sample";
    await scLoad();
  } catch (ex) { toast(ex.message); }
}

function scFetch(offset) {
  const tab = scTab();
  const q = new URLSearchParams({ tab, offset, limit: SC_PAGE });
  if (sc.blind) q.set("blind", "1");
  return api("GET", `/api/v1/projects/${state.projectId}/papers?${q}`);
}

async function scMore() {
  const gen = sc.gen;
  const more = await scFetch(sc.items.length - sc.moved);  // decided papers left this tab: the next page starts earlier
  if (gen !== sc.gen) return;
  const seen = new Set(sc.items.map((p) => p.id));
  sc.items = sc.items.concat(more.items.filter((p) => !seen.has(p.id)));
  scRenderBody();
}

// ---------- rendering ----------
function scRender() {
  scRenderFunnel();
  scRenderCriteria();
  scRenderTest();
  scRenderActions();
  scRenderTabs();
  scRenderBody();
}

function scRenderFunnel() {
  const c = sc.listing?.counts || {};
  const steps = [[sc.listing?.references ?? 0, "records"], [c.all ?? 0, "papers"]];
  if (!sc.blind && c.included != null) steps.push([c.included, "included"]);
  if (!sc.blind && c.fulltext != null) steps.push([c.fulltext, "full text"]);
  $("#funnel").replaceChildren(...steps.flatMap(([n, label], i) => [
    i ? h("span", { class: "arrow" }, "→") : null,
    h("span", { class: "step" }, h("strong", {}, Number(n).toLocaleString()), ` ${label}`)]).filter(Boolean));
}

function scRenderCriteria() {
  const box = $("#criteria");
  if (!sc.criteria) {
    box.replaceChildren(h("h3", {}, "Criteria"),
      h("p", { class: "muted small" }, "No criteria yet. Tell the screening assistant what you are looking for, e.g. “adults with PCOS, retrospective studies, since 2015, no reviews”."));
    return;
  }
  const cards = sc.criteria.criteria.map((c) => h("div", { class: `crit ${c.kind}` },
    h("div", { class: "crit-head" },
      h("span", { class: "crit-kind" }, c.kind === "include_if" ? "Include if" : "Exclude if"),
      h("strong", {}, c.name),
      h("span", { class: "badge neutral" }, checkLabel(c)),
      h("button", { class: "del", type: "button", title: "Delete criterion", "aria-label": `Delete ${c.name}`,
        onclick: () => scDeleteCriterion(c.id) }, "✕")),
    h("div", { class: "muted small" }, c.definition,
      c.check === "extract" ? ` Options: ${(c.options || []).join(", ")}.` : "")));
  box.replaceChildren(h("div", { class: "row between" }, h("h3", {}, `Criteria · version ${sc.criteria.version}`),
    sc.blind ? null : h("button", { class: "btn small", type: "button", disabled: sc.testing, onclick: () => scTest() },
      sc.testing ? "Testing…" : "Test on 20 papers")), ...cards);
}

async function scDeleteCriterion(id) {
  const left = sc.criteria.criteria.filter((c) => c.id !== id);
  if (!left.length) { toast("Keep at least one criterion."); return; }
  try {
    sc.criteria = await api("PUT", `/api/v1/projects/${state.projectId}/criteria`, { criteria: left });
    sc.test = null;
    scRender();
  } catch (ex) { toast(ex.message); }
}

async function scTest() {
  if (sc.testing || sc.blind) return;  // one paid test at a time; never while labelling blind
  sc.testing = true;
  scRenderCriteria();
  const projectId = state.projectId;
  const box = $("#test-run");
  box.hidden = false;
  box.replaceChildren(h("div", { class: "progress" }, h("span", { class: "spinner" }), "Screening 20 papers…"));
  try {
    const test = await api("POST", `/api/v1/projects/${projectId}/screening/test`);
    if (projectId === state.projectId) sc.test = { ...test, projectId };
  } catch (ex) { toast(ex.message); } finally {
    sc.testing = false;
    if (projectId === state.projectId) { scRenderCriteria(); scRenderTest(); }
  }
}

function scRenderTest() {
  const box = $("#test-run");
  const t = sc.test;
  if (!t || sc.blind || t.projectId !== state.projectId || !sc.criteria || t.version !== sc.criteria.version) {
    box.hidden = !sc.testing || sc.blind;
    return;
  }
  box.hidden = false;
  const est = t.estimate;
  const rows = t.items.map((it) => {
    const fb = (value) => h("button", {
      class: `btn small${it.feedback === value ? " primary" : ""}`, type: "button",
      "aria-label": value === "up" ? "Agree" : "Disagree",
      onclick: async () => {
        try {
          await api("PATCH", `/api/v1/projects/${state.projectId}/screening/test/${it.paper.id}`, { feedback: value });
          it.feedback = value;
          scRenderTest();
        } catch (ex) { toast(ex.message); }
      },
    }, value === "up" ? "👍" : "👎");
    return h("tr", {},
      h("td", {}, h("div", { class: "t-title" }, it.paper.title), h("div", { class: "muted small" }, it.paper.year ?? "")),
      ...sc.criteria.criteria.map((c) => verdictCell(c, it.verdicts.find((v) => v.criterion_id === c.id))),
      h("td", {}, { include: "Include", exclude: "Exclude", review: "Needs review" }[it.ai_decision] || ""),
      h("td", { class: "nowrap" }, fb("up"), " ", fb("down")));
  });
  box.replaceChildren(
    h("div", { class: "row between" }, h("h3", {}, "Test on 20 papers"),
      h("button", { class: "btn link small", type: "button", onclick: () => { sc.test = null; scRenderTest(); } }, "Hide")),
    h("p", { class: "muted small" },
      `${t.seconds != null ? `Took ${t.seconds} s. ` : ""}Screening all papers would take about ${est.calls.toLocaleString()} model calls `
      + `(~${Math.round((est.input_tokens + est.output_tokens) / 1000).toLocaleString()}k tokens). `
      + "Thumbs down where the AI got it wrong, then refine the criteria in chat."),
    ...sc.criteria.criteria.filter((c) => c.check === "extract").map((c) => h("p", { class: "small" },
      h("strong", {}, `${c.name}: `), valueCounts(c, t.items.map((it) => it.verdicts)))),
    h("div", { class: "table-wrap" }, h("table", { class: "sc-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Paper"), ...sc.criteria.criteria.map((c) => h("th", {}, c.name)),
        h("th", {}, "AI decision"), h("th", {}, "Right?"))),
      h("tbody", {}, rows))));
}

function scRenderActions() {
  const bar = $("#sc-buttons");
  if (!sc.criteria) { bar.replaceChildren(); return; }
  bar.replaceChildren(...[
    h("button", { class: "btn", type: "button", onclick: () => scPropose("screen_all") }, "Screen all papers…"),
    (sc.listing?.counts?.included ?? 0) > 0
      ? h("button", { class: "btn", type: "button", onclick: () => scPropose("fetch_full_text") }, "Fetch free full text…") : null,
  ].filter(Boolean));
}

async function scPropose(tool) {
  try {
    const action = await api("POST", `/api/v1/projects/${state.projectId}/actions`, { tool });
    const card = actionCard(action);
    $("#sc-actions").append(card);
    card.scrollIntoView({ block: "nearest" });
  } catch (ex) { toast(ex.message); }
}

function scRenderTabs() {
  const counts = sc.listing?.counts || {};
  const tabs = sc.blind ? [["sample", "Labelling sample"], ["all", "All"]] : SC_TABS;
  $("#sc-tabs").replaceChildren(...tabs.map(([key, label]) => h("button", {
    class: `tab${scTab() === key ? " active" : ""}`, type: "button", role: "tab",
    onclick: () => { if (sc.blind) sc.blindTab = key; else sc.tab = key; sc.queue = null; scLoad().catch((ex) => toast(ex.message)); },
  }, label, counts[key] != null ? h("span", { class: "count" }, Number(counts[key]).toLocaleString()) : null)),
  sc.blind && !counts.sample ? h("button", { class: "btn small primary", type: "button", onclick: () => scMakeSample(),
    title: "150 papers drawn from the AI's include, review and exclude piles, shuffled, for the screening recall eval" },
  "Create labelling sample (150)") : null);
}

function verdictCell(c, v) {
  const td = h("td", { class: "verdict" });
  if (!v) { td.append("–"); return td; }
  const btn = h("button", { class: "v-btn", type: "button", title: `${c.name}: ${v.verdict}`,
    onclick: (e) => { e.stopPropagation(); showPopover(btn, c, v); } }, verdictIcon(c, v));
  td.append(btn);
  return td;
}

function showPopover(anchor, c, v) {
  document.querySelector(".popover")?.remove();
  const pop = h("div", { class: "popover", role: "dialog" },
    h("strong", {}, `${c.name}: ${c.check === "extract" ? `${v.value || "no answer"} (${v.verdict})` : v.verdict}`),
    v.quote ? h("blockquote", {}, `“${v.quote}”`) : null,
    h("div", { class: "muted small" }, v.reason));
  document.body.append(pop);
  const r = anchor.getBoundingClientRect();
  pop.style.top = `${window.scrollY + r.bottom + 6}px`;
  pop.style.left = `${Math.max(8, Math.min(window.scrollX + r.left - 120, window.innerWidth - pop.offsetWidth - 8))}px`;
  setTimeout(() => document.addEventListener("click", () => pop.remove(), { once: true }));
}

function scRenderBody() {
  const body = $("#sc-body");
  if (sc.queue) { body.replaceChildren(renderQueue()); return; }
  const total = sc.listing?.total ?? 0;
  if (!sc.items.length) {
    body.replaceChildren(h("p", { class: "muted empty-note" },
      sc.listing?.counts?.all ? "Nothing in this tab." : "Import a reference export (PubMed .nbib, BibTeX .bib, RIS, CSV or Excel) to start."));
    return;
  }
  const tab = scTab();
  const head = h("div", { class: "row between" },
    h("span", { class: "muted small" }, `${sc.items.length.toLocaleString()} of ${total.toLocaleString()}`),
    tab !== "duplicates" ? h("button", { class: "btn primary small", type: "button", onclick: openQueue },
      sc.blind ? `▶ Label ${total.toLocaleString()} blind` : `▶ Review the ${total.toLocaleString()}`) : null);
  const table = tab === "duplicates" ? duplicatesTable() : papersTable(tab);
  const more = sc.items.length < total
    ? h("button", { class: "btn small", type: "button", onclick: () => scMore().catch((ex) => toast(ex.message)) }, "Load more") : null;
  body.replaceChildren(...[head, h("div", { class: "table-wrap" }, table), more].filter(Boolean));
}

function columnCell(name, x) {
  const td = h("td", { class: "verdict" });
  if (!x) { td.append("–"); return td; }
  const btn = h("button", { class: "v-btn", type: "button", title: `${name}: from the ${x.source}`,
    onclick: (e) => {
      e.stopPropagation();
      showPopover(btn, { name }, { verdict: x.source === "fulltext" ? "from the full text" : "from the abstract",
        quote: x.quote, reason: x.value });
    } }, x.value);
  td.append(btn);
  return td;
}

function papersTable(tab) {
  const crit = sc.blind || !sc.criteria ? [] : sc.criteria.criteria;
  const cols = sc.blind ? [] : [...new Set(sc.items.flatMap((p) => Object.keys(p.columns || {})))];
  const rows = sc.items.map((p, i) => h("tr", { class: "clickable", onclick: () => openQueue(i) },
    h("td", {}, h("div", { class: "t-title" }, p.title || "(no title)"),
      h("div", { class: "muted small" }, [p.year, p.journal, (p.sources || []).join(" + ")].filter(Boolean).join(" · "),
        p.abstract ? "" : " · no abstract")),
    ...crit.map((c) => verdictCell(c, (p.verdicts || []).find((v) => v.criterion_id === c.id))),
    ...cols.map((name) => columnCell(name, (p.columns || {})[name])),
    sc.blind ? h("td", {}, p.blind_label ? `Labelled: ${p.blind_label}` : "–")
      : h("td", { class: "nowrap" }, decisionLabel(p), p.disagrees ? h("span", { class: "badge review", title: "Your decision differs from the AI's" }, "≠ AI") : null),
    tab === "fulltext" ? fulltextCell(p) : null));
  return h("table", { class: "sc-table" },
    h("thead", {}, h("tr", {}, h("th", {}, "Paper"), ...crit.map((c) => h("th", { title: c.definition }, c.name)),
      ...cols.map((name) => h("th", { class: "col-extract", title: "Extracted column" }, name)),
      h("th", {}, sc.blind ? "Your label" : "Decision"), tab === "fulltext" ? h("th", {}, "Full text") : null)),
    h("tbody", {}, rows));
}

function fulltextCell(p) {
  const status = { open_access_found: "Free PDF", uploaded: "Uploaded", not_found: "Upload needed", not_tried: "–" }[p.fulltext_status]
    || p.fulltext_status;
  const td = h("td", { onclick: (e) => e.stopPropagation() }, status,
    p.fulltext_license ? h("span", { class: "muted small" }, ` (${p.fulltext_license})`) : null);
  if (!p.document_id) {
    const input = h("input", { type: "file", accept: "application/pdf,.pdf", hidden: true });
    input.addEventListener("change", async () => {
      const form = new FormData();
      form.append("file", input.files[0]);
      try {
        await api("POST", `/api/v1/projects/${state.projectId}/papers/${p.id}/pdf`, form, true);
        toast("PDF uploaded — indexing…");
        await Promise.all([scRefresh(), loadDocuments()]);
      } catch (ex) { toast(ex.message); }
    });
    td.append(" ", h("label", { class: "btn small" }, "Upload PDF", input));
  }
  return td;
}

function duplicatesTable() {
  const resolve = (p, action) => async (e) => {
    e.stopPropagation();
    try {
      await api("POST", `/api/v1/projects/${state.projectId}/possible-duplicates/${p.possible_duplicate.id}`, { action });
      await scLoad();
    } catch (ex) { toast(ex.message); }
  };
  const paper = (x) => h("div", {}, h("div", { class: "t-title" }, x.title),
    h("div", { class: "muted small" }, [x.year, x.journal, x.doi, (x.sources || []).join(" + ")].filter(Boolean).join(" · ")));
  return h("table", { class: "sc-table" },
    h("thead", {}, h("tr", {}, h("th", {}, "Paper"), h("th", {}, "Possible duplicate"), h("th", {}, "Similarity"), h("th", {}, ""))),
    h("tbody", {}, sc.items.map((p) => h("tr", {},
      h("td", {}, paper(p)), h("td", {}, paper(p.possible_duplicate.other)),
      h("td", {}, `${Math.round(p.possible_duplicate.similarity * 100)}%`),
      h("td", { class: "nowrap" },
        h("button", { class: "btn small", type: "button", onclick: resolve(p, "merge") }, "Merge"), " ",
        h("button", { class: "btn small", type: "button", onclick: resolve(p, "keep_article"),
          title: "One is a preprint of the other: keep the journal article" }, "Keep article"), " ",
        h("button", { class: "btn small", type: "button", onclick: resolve(p, "dismiss") }, "Not duplicates"))))));
}

// ---------- review queue ----------
function openQueue(i = 0) {
  sc.queue = { i: typeof i === "number" ? i : 0 };  // walks sc.items, loading further pages as it reaches the end
  scRenderBody();
}

// Title/abstract text with each verdict quote marked; built from text nodes only.
function highlighted(text, quotes) {
  const lower = text.toLowerCase();
  const spans = [];
  for (const q of quotes) {
    const needle = q.trim().toLowerCase();
    if (needle.length < 3) continue;
    let at = lower.indexOf(needle);
    while (at !== -1) { spans.push([at, at + needle.length]); at = lower.indexOf(needle, at + needle.length); }
  }
  spans.sort((a, b) => a[0] - b[0]);
  const out = [];
  let pos = 0;
  for (const [a, b] of spans) {
    if (a < pos) continue;
    out.push(text.slice(pos, a), h("mark", {}, text.slice(a, b)));
    pos = b;
  }
  out.push(text.slice(pos));
  return out;
}

function renderQueue() {
  const { i } = sc.queue;
  const items = sc.items;
  const total = sc.listing?.total ?? items.length;
  const p = items[i];
  const back = h("button", { class: "btn link small", type: "button", onclick: closeQueue }, "← back to table");
  if (!p) return h("div", { class: "card queue" }, h("p", {}, "That’s all in this list."), back);
  const crit = sc.blind || !sc.criteria ? [] : sc.criteria.criteria;
  const quotes = sc.blind ? [] : (p.verdicts || []).map((v) => v.quote || "").filter((q) => !q.includes(": "));  // not metadata
  const reason = h("input", { class: "reason", placeholder: "Reason (optional)", maxlength: "2000", "aria-label": "Reason" });
  const decide = async (value) => {
    const q = sc.queue;
    if (!q || q.busy) return;  // a double press must not skip the next paper
    q.busy = true;
    try {
      if (value) {
        const body = sc.blind ? { blind_label: value } : { decision: value, reason: reason.value.trim() };
        const out = await api("PATCH", `/api/v1/projects/${state.projectId}/papers/${p.id}/decision`, body);
        Object.assign(p, out);
        const tabDecision = sc.blind ? null : DECISION_OF_TAB[sc.tab];
        if (tabDecision && (p.decision || null) !== tabDecision) sc.moved += 1;
      }
      if (sc.queue !== q) return;  // closed (Esc, back, tab, project) while saving
      q.i += 1;
      if (q.i >= sc.items.length && sc.items.length - sc.moved < total) await scMore();
    } catch (ex) {
      toast(ex.message);
      if (sc.queue === q) scRenderBody();  // show the paper the index now points at, never a stale one
      return;
    } finally { q.busy = false; }
    if (sc.queue === q) scRenderBody();
  };
  const label = sc.blind ? "Blind labelling" : SC_TABS.find(([k]) => k === sc.tab)?.[1];
  return h("div", { class: "card queue", tabindex: "-1" },
    h("div", { class: "row between" }, h("span", { class: "muted small" }, `${label} · ${i + 1} of ${total.toLocaleString()}`), back),
    h("h3", {}, ...highlighted(p.title || "(no title)", quotes)),
    h("div", { class: "muted small" }, [p.year, p.journal, (p.sources || []).join(" + ")].filter(Boolean).join(" · ")),
    h("p", { class: "abstract" }, ...(p.abstract ? highlighted(p.abstract, quotes) : [h("em", {}, "No abstract")])),
    crit.length ? h("ul", { class: "verdicts" }, crit.map((c) => {
      const v = (p.verdicts || []).find((x) => x.criterion_id === c.id);
      return h("li", {}, `${verdictIcon(c, v)} · `, h("strong", {}, c.name),
        v?.quote ? ` — “${v.quote}”` : "", v ? h("span", { class: "muted small" }, ` ${v.reason}`) : " not checked");
    })) : null,
    !sc.blind && p.decision ? h("p", { class: "small" }, `Current: ${decisionLabel(p)}`) : null,
    sc.blind && p.blind_label ? h("p", { class: "small" }, `Your label: ${p.blind_label}`) : null,
    sc.blind ? null : reason,
    h("div", { class: "row" },
      h("button", { class: "btn primary", type: "button", onclick: () => decide("include") }, "Include (I)"),
      h("button", { class: "btn", type: "button", onclick: () => decide("exclude") }, "Exclude (E)"),
      h("button", { class: "btn", type: "button", onclick: () => decide(null) }, "Skip (S)")));
}

function closeQueue() {
  sc.queue = null;
  scLoad().catch((ex) => toast(ex.message));
}

function queueKeys(e) {
  if (!sc.queue || $("#screening").hidden || e.repeat || e.target.closest?.("input, textarea, select")
    || e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  const btns = [...document.querySelectorAll(".queue .row:last-child .btn")];
  const idx = { i: 0, e: 1, s: 2 }[key];
  if (idx != null && btns[idx]) { e.preventDefault(); btns[idx].click(); }
  if (key === "escape") closeQueue();
}

// ---------- pending actions (Confirm cards) ----------
function actionCard(action) {
  const card = h("div", { class: "card action-card" });
  let inFlight = false;
  const fill = () => {
    const a = action;
    const what = a.tool === "screen_all" ? "Screen all papers"
      : a.tool === "extract_column" ? `Extract “${a.estimate?.column || "column"}” for the ${a.estimate?.tab || "included"} papers`
        : "Fetch free full text for included papers";
    const est = a.estimate || {};
    const detail = a.tool === "extract_column"
      ? `${(est.papers ?? 0).toLocaleString()} papers · ${(est.calls ?? 0).toLocaleString()} model calls · ${(est.with_fulltext ?? 0).toLocaleString()} read from the full text`
      : a.tool === "screen_all"
      ? `${(est.papers ?? 0).toLocaleString()} papers · ${(est.calls ?? 0).toLocaleString()} model calls · ~${Math.round(((est.input_tokens ?? 0) + (est.output_tokens ?? 0)) / 1000).toLocaleString()}k tokens`
      : `${(est.papers ?? 0).toLocaleString()} papers · Europe PMC and Unpaywall (free, legal copies only)`;
    const parts = [h("strong", {}, what), h("div", { class: "muted small" },
      a.criteria_version != null ? `${detail} · criteria version ${a.criteria_version}` : detail)];
    if (a.status === "pending") {
      parts.push(h("div", { class: "row" },
        h("button", { class: "btn primary small", type: "button", disabled: inFlight, onclick: () => act("confirm") }, "Confirm"),
        h("button", { class: "btn small", type: "button", disabled: inFlight, onclick: () => act("cancel") }, "Cancel")));
    } else if (a.status === "confirmed" || a.status === "running") {
      parts.push(h("div", { class: "progress" }, h("span", { class: "spinner" }), "Running…"));
    } else {
      const r = a.result || {};
      const summary = a.status === "done"
        ? (a.tool === "extract_column" ? `Done: ${r.filled ?? 0} papers filled (${r.fulltext ?? 0} from full text, ${r.not_stated ?? 0} not stated).`
          : a.tool === "screen_all" ? `Done: ${r.include ?? 0} included, ${r.review ?? 0} need review, ${r.exclude ?? 0} excluded.`
          : `Done: ${r.found ?? 0} free PDFs found, ${r.not_found ?? 0} need an upload.`)
        : a.status === "failed" ? `Failed: ${a.error || "unknown error"}` : `${a.status[0].toUpperCase()}${a.status.slice(1)}.`;
      parts.push(h("div", { class: `small ${a.status === "failed" ? "error" : ""}` }, summary));
    }
    card.replaceChildren(...parts);
  };
  const act = async (verb) => {
    if (inFlight) return;
    inFlight = true;
    fill();
    try {
      action = await api("POST", `/api/v1/actions/${action.id}/${verb}`);
    } catch (ex) {
      toast(ex.message);
      action = await api("GET", `/api/v1/actions/${action.id}`).catch(() => action);
    }
    inFlight = false;
    fill();
    if (action.status === "confirmed" || action.status === "running") await pollAction();
  };
  const pollAction = async () => {
    let wait = 1500;
    while (action.status === "confirmed" || action.status === "running") {
      await new Promise((r) => setTimeout(r, wait));
      if (!card.isConnected || !state.token) return;
      try { action = await api("GET", `/api/v1/actions/${action.id}`); wait = 1500; } catch { wait = Math.min(wait * 2, 30000); }
      fill();
    }
    await Promise.all([scRefresh(), loadDocuments()]).catch(() => {});
  };
  fill();
  if (action.status === "confirmed" || action.status === "running") pollAction();
  else if (action.status === "pending") {  // a chat card holds the proposal-time snapshot: show the real status
    api("GET", `/api/v1/actions/${action.id}`).then((a) => {
      action = a;
      fill();
      if (a.status === "confirmed" || a.status === "running") pollAction();
    }).catch(() => {});
  }
  return card;
}

// ---------- screening assistant (agent chat) ----------
async function scLoadAgent() {
  const id = state.conversationId;
  if (!id) { sc.messages = []; renderAgent(); return; }
  try {
    const convo = await api("GET", `/api/v1/conversations/${id}`);
    if (id !== state.conversationId) return;
    sc.messages = convo.messages || [];
  } catch { sc.messages = []; }
  renderAgent();
}

const TOOL_LABEL = {
  import_status: "Checked the imports", deduplicate_status: "Checked deduplication", get_funnel_status: "Checked the funnel",
  draft_criteria: "Drafted criteria", update_criteria: "Saved criteria", test_screen: "Tested on 20 papers",
  propose_action: "Proposed a step", get_papers: "Read papers", ask_research_question: "Asked your papers",
  add_column: "Added a column", get_column: "Read a column", export_table: "Built a table",
  count_papers: "Counted papers", web_search: "Searched the web", search_pubmed: "Asked PubMed",
  export_answer: "Saved the answer", chart_answer: "Built a chart",
};

async function downloadFrom(path, fallback) {
  const res = await fetch(path, { headers: { Authorization: `Bearer ${state.token}` } });
  if (!res.ok) {
    const d = (await res.json().catch(() => ({}))).detail;
    throw new Error(typeof d === "string" ? d : `Download failed (${res.status})`);
  }
  const name = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") || "")?.[1] || fallback;
  saveBlob(await res.blob(), name);
}

function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, text) {  // attributes via setAttribute and text via textContent: no HTML injection
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  if (text != null) el.textContent = text;
  return el;
}

function barChart(chart) {
  const pts = chart.points, W = 640, rowH = 26, left = 210, top = 34, H = top + pts.length * rowH + 30;
  const max = Math.max(...pts.map((p) => p.value), 0) || 1;
  const svg = svgEl("svg", { xmlns: SVG_NS, viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": chart.title,
    "font-family": "system-ui, sans-serif", "font-size": 12 });
  svg.append(svgEl("rect", { x: 0, y: 0, width: W, height: H, fill: "#ffffff" }));
  svg.append(svgEl("text", { x: 8, y: 20, "font-size": 14, "font-weight": 600, fill: "#111" }, chart.title));
  pts.forEach((p, i) => {
    const y = top + i * rowH, w = ((W - left - 70) * p.value) / max;
    svg.append(svgEl("text", { x: left - 8, y: y + 15, "text-anchor": "end", fill: "#333" }, p.label));
    svg.append(svgEl("rect", { x: left, y: y + 3, width: Math.max(w, 1), height: rowH - 8, fill: "#3b6fd8", rx: 2 }));
    svg.append(svgEl("text", { x: left + w + 6, y: y + 15, fill: "#111" }, `${p.value} [${p.finding}]`));
  });
  svg.append(svgEl("text", { x: left, y: H - 8, fill: "#555" }, `${chart.y_label} · [n] = finding number in the answer`));
  return svg;
}

function chartCard(chart) {
  const svg = barChart(chart);
  const csv = ["label,value,finding", ...chart.points.map((p) => `"${p.label.replace(/"/g, '""')}",${p.value},${p.finding}`)].join("\n");
  return h("div", { class: "card chart" }, svg,
    chart.dropped ? h("p", { class: "muted small" }, `${chart.dropped} value(s) left out: their finding does not state them.`) : null,
    h("div", { class: "row" },
      h("button", { class: "btn small", type: "button",
        onclick: () => saveBlob(new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" }), "maxrim-chart.svg") }, "Download SVG"),
      h("button", { class: "btn small", type: "button",
        onclick: () => saveBlob(new Blob(["\ufeff" + csv], { type: "text/csv" }), "maxrim-chart-data.csv") }, "Download data (CSV)")));
}

async function scDownload(tab, format) {
  const res = await fetch(`/api/v1/projects/${state.projectId}/export?tab=${encodeURIComponent(tab)}&format=${format}`,
    { headers: { Authorization: `Bearer ${state.token}` } });
  if (!res.ok) {
    const d = (await res.json().catch(() => ({}))).detail;
    throw new Error(typeof d === "string" ? d : `Export failed (${res.status})`);
  }
  const name = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") || "")?.[1] || `papers.${format}`;
  const url = URL.createObjectURL(await res.blob());
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderAgent() {
  const log = $("#agent-log");
  if (!sc.messages.length) {
    log.replaceChildren(h("p", { class: "muted small" },
      "Describe what you are screening for. I can draft criteria, test them on 20 papers and propose screening the rest — paid steps run only when you press Confirm."));
    return;
  }
  log.replaceChildren(...sc.messages.map((m) => {
    if (m.role === "user") return h("div", { class: "msg user" }, m.content.text);
    if (m.role === "assistant") return h("div", { class: "msg bot" }, m.content.text);
    const c = m.content;
    if (c.action) return actionCard(c.action);
    const ans = c.result && c.result.answer_export;
    if (ans) return h("div", { class: "tool-line" }, `✓ Answer: ${ans.findings} findings `,
      h("button", { class: "btn small", type: "button",
        onclick: () => downloadFrom(`/api/v1/research-runs/${ans.run_id}/export?format=${ans.format}`, `answer.${ans.format}`)
          .catch((err) => toast(err.message)) }, `Download ${ans.format === "docx" ? "Word" : "CSV"}`));
    if (c.result && c.result.chart) return chartCard(c.result.chart);
    const ex = c.result && c.result.export;
    if (ex) return h("div", { class: "tool-line" }, `✓ Table: ${ex.papers} ${ex.tab} papers, ${ex.columns.length} columns `,
      h("button", { class: "btn small", type: "button",
        onclick: () => sc.blind ? toast("Turn off blind labelling to export: the table shows the AI's decisions.")
          : scDownload(ex.tab, ex.format).catch((err) => toast(err.message)) },
        `Download ${ex.format === "xlsx" ? "Excel" : "CSV"}`));
    const failed = c.result && c.result.error;
    return h("div", { class: `tool-line${failed ? " error" : ""}` },
      `${failed ? "⚠" : "✓"} ${TOOL_LABEL[c.tool] || c.tool}${failed ? `: ${typeof c.result.error === "string"
        ? c.result.error : JSON.stringify(c.result.error)}` : ""}`);
  }));
  log.scrollTop = log.scrollHeight;
}

async function sendAgent(e) {
  e.preventDefault();
  const text = $("#agent-input").value.trim();
  if (!text || sc.sending) return;
  sc.sending = true;
  $("#agent-send").disabled = true;
  const projectId = state.projectId;
  sc.messages.push({ role: "user", content: { text } });
  sc.messages.push({ role: "assistant", content: { text: "…" } });
  renderAgent();
  $("#agent-input").value = "";
  let cid = state.conversationId;
  const here = () => projectId === state.projectId && cid === state.conversationId;
  try {
    if (!cid) {
      cid = (await api("POST", `/api/v1/projects/${projectId}/conversations`, { title: text.slice(0, 80) })).id;
      if (projectId !== state.projectId || state.conversationId) throw new Error("You switched chats; your message was not sent.");
      state.conversationId = cid;
      loadConversations().catch(() => {});
    }
    const out = await api("POST", `/api/v1/conversations/${cid}/agent`, { message: text });
    if (here()) sc.messages.splice(-2, 2, ...out.messages);
    else toast("The assistant answered in the chat you left.");
  } catch (ex) {
    if (here() || !cid) sc.messages.splice(-1, 1, { role: "assistant", content: { text: `Error: ${ex.message}` } });
    else toast(ex.message);
  } finally {
    sc.sending = false;
    $("#agent-send").disabled = false;
  }
  if (here() || !cid) renderAgent();
  if (projectId === state.projectId) scRefresh().catch(() => {});
}

// ---------- reference import ----------
async function importReferences(files) {
  for (const f of files) {
    const form = new FormData();
    form.append("file", f);
    try {
      const r = await api("POST", `/api/v1/projects/${state.projectId}/reference-files`, form, true);
      toast(r.already_imported ? `${f.name}: already imported`
        : `${f.name}: ${r.records} records → ${r.papers} papers (${r.merged} merged, ${r.possible_duplicates} possible duplicates)`
          + (r.errors.length ? `; ${r.errors.length} entries could not be read` : ""));
    } catch (ex) { toast(`${f.name}: ${ex.message}`); }
  }
  $("#ref-upload").value = "";
  await scRefresh();
}

// ---------- view switching & hooks from app.js ----------
function showView(name) {
  const screening = name === "screening";
  $("#screening").hidden = !screening;
  $("#composer").hidden = screening;
  $("#hero").hidden = screening || state.turns.length > 0;
  $("#thread").hidden = screening || state.turns.length === 0;
  $("#view-chat").classList.toggle("active", !screening);
  $("#view-screening").classList.toggle("active", screening);
  store.set("maxrim.view", name);
  if (screening && state.projects.length) { scLoad().catch((ex) => toast(ex.message)); scLoadAgent(); }
}

window.screening = {
  onProject() {
    sc.queue = null; sc.test = null; sc.items = []; sc.gen += 1;
    $("#sc-actions").replaceChildren(h("span", { id: "sc-buttons", class: "sc-actions" }));
    scLoad().catch(() => {});
    scLoadAgent();
  },
  onConversation() { scLoadAgent(); },
  onThreadRender() { if (!$("#screening").hidden) { $("#hero").hidden = true; $("#thread").hidden = true; } },
};

$("#view-chat").addEventListener("click", () => showView("chat"));
$("#view-screening").addEventListener("click", () => showView("screening"));
$("#ref-upload").addEventListener("change", (e) => importReferences([...e.target.files]));
$("#agent-form").addEventListener("submit", sendAgent);
$("#agent-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendAgent(e); }
});
$("#blind-toggle").addEventListener("change", (e) => {
  sc.blind = e.target.checked;
  sc.queue = null;
  scLoad().catch((ex) => toast(ex.message));
});
$("#sc-export").addEventListener("change", (e) => {
  const format = e.target.value;
  e.target.value = "";
  if (!format) return;
  if (sc.blind) { toast("Turn off blind labelling to export: the table shows the AI's decisions."); return; }
  scDownload(sc.tab, format).catch((ex) => toast(ex.message));
});
document.addEventListener("keydown", queueKeys);
if (store.get("maxrim.view") === "screening") showView("screening");
