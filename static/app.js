// MaxRIM UI: plain JS, no build step. All server text (PDF/web snippets, answers) is rendered with textContent.
"use strict";

const $ = (sel) => document.querySelector(sel);
const POLL_MS = 2000;
const EXAMPLES = [
  "What were the main limitations reported?",
  "Which variables predicted in-hospital mortality?",
  "How well were the models calibrated?",
  "Compare the modelling approaches of the studies",
];

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* private mode */ } },
};

const state = {
  token: store.get("maxrim.token"),
  email: store.get("maxrim.email"),
  registering: false,
  webAvailable: false,
  projects: [],
  projectId: store.get("maxrim.project"),
  documents: [],
  conversations: [],
  conversationId: null,
  turns: [],        // [{run, evidence}]
  busy: false,
  view: 0,          // bumped on every chat/project switch or logout; stale async work checks it and stops
};

const newView = () => ++state.view;

// ---------- tiny DOM helper ----------
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 4000);
}

// ---------- API ----------
async function api(method, path, body, isForm = false) {
  const headers = state.token ? { Authorization: `Bearer ${state.token}` } : {};
  if (body && !isForm) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method, headers, body: isForm ? body : body ? JSON.stringify(body) : undefined });
  if (res.status === 401 && state.token) { logout(); throw new Error("Session expired — please sign in again."); }
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `Request failed (${res.status})`);
  }
  return data;
}

// ---------- auth ----------
function showAuth() {
  $("#app").hidden = true;
  $("#auth").hidden = false;
  $("#auth-email").focus();
}

function setAuthMode(registering) {
  state.registering = registering;
  $("#auth-submit").textContent = registering ? "Create account" : "Sign in";
  $("#auth-toggle").textContent = registering ? "Have an account? Sign in" : "New here? Create an account";
  $("#auth-password").autocomplete = registering ? "new-password" : "current-password";
  $("#auth-error").hidden = true;
}

async function submitAuth(e) {
  e.preventDefault();
  const email = $("#auth-email").value.trim();
  const password = $("#auth-password").value;
  const err = $("#auth-error");
  err.hidden = true;
  try {
    if (state.registering) await api("POST", "/auth/register", { email, password });
    const { access_token } = await api("POST", "/auth/login", { email, password });
    state.token = access_token;
    state.email = email;
    store.set("maxrim.token", access_token);
    store.set("maxrim.email", email);
    await boot();
  } catch (ex) {
    err.textContent = ex.message;
    err.hidden = false;
  }
}

function logout() {
  newView();
  state.token = null;
  store.set("maxrim.token", null);
  showAuth();
}

// ---------- projects & documents ----------
async function loadProjects() {
  state.projects = await api("GET", "/api/v1/projects");
  if (!state.projects.length) {
    state.projects = [await api("POST", "/api/v1/projects", { name: "My research" })];
  }
  if (!state.projects.some((p) => p.id === state.projectId)) state.projectId = state.projects[0].id;
  store.set("maxrim.project", state.projectId);
  const sel = $("#project-select");
  sel.replaceChildren(...state.projects.map((p) => h("option", { value: p.id, selected: p.id === state.projectId }, p.name)));
}

async function newProject() {
  const name = await promptInline("Project name");
  if (!name) return;
  const p = await api("POST", "/api/v1/projects", { name });
  state.projectId = p.id;
  await loadProjects();
  await openProject();
}

// The viewer has no prompt(); a small inline input in the sidebar instead.
function promptInline(label) {
  return new Promise((resolve) => {
    const input = h("input", { placeholder: label, "aria-label": label, maxlength: "200" });
    const row = h("div", { class: "row" }, input,
      h("button", { class: "btn small primary", type: "button", onclick: () => done(input.value.trim()) }, "Add"),
      h("button", { class: "btn small", type: "button", onclick: () => done("") }, "Cancel"));
    input.style.cssText = "flex:1;min-width:0;border:1px solid var(--line);border-radius:8px;padding:6px 8px;background:var(--bg)";
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") done(input.value.trim()); if (e.key === "Escape") done(""); });
    const anchor = $("#project-select").closest(".side-block");
    anchor.append(row);
    input.focus();
    function done(v) { row.remove(); resolve(v); }
  });
}

async function loadDocuments() {
  const projectId = state.projectId;
  const docs = await api("GET", `/api/v1/projects/${projectId}/documents`);
  if (projectId !== state.projectId) return;  // the user switched projects meanwhile; that project's load wins
  state.documents = docs;
  renderDocuments();
  clearTimeout(loadDocuments.timer);
  if (state.documents.some((d) => d.status === "queued" || d.status === "processing")) {
    loadDocuments.timer = setTimeout(() => loadDocuments().catch(() => {}), 4000);
  }
}

function hasReadyPapers() {
  return state.documents.some((d) => d.status === "ready");
}

function renderDocuments() {
  const papers = hasReadyPapers();
  $("#no-papers-hint").hidden = papers;
  $("#no-papers-hint").textContent = state.webAvailable ? "No papers yet — turn on Web search to start, or upload a PDF."
    : "No papers yet — upload a PDF to start.";
  $("#question").placeholder = papers ? "Ask a question about your papers…" : "Ask a question…";
  const list = $("#document-list");
  if (!state.documents.length) {
    list.replaceChildren(h("li", { class: "empty-note" }, "No papers yet — upload PDFs to start."));
    return;
  }
  list.replaceChildren(...state.documents.map((d) => h("li", { title: d.error || d.status },
    h("span", { class: `dot ${d.status}` }),
    h("span", { class: "label" }, d.filename),
    h("span", { class: "muted small" }, d.status === "ready" ? `${d.chunk_count ?? ""}` : d.status))));
}

async function upload(files) {
  for (const f of files) {
    const form = new FormData();
    form.append("file", f);
    try {
      await api("POST", `/api/v1/projects/${state.projectId}/documents`, form, true);
      toast(`Uploaded ${f.name} — indexing…`);
    } catch (ex) {
      toast(`${f.name}: ${ex.message}`);
    }
  }
  $("#upload").value = "";
  await loadDocuments();
}

// ---------- conversations ----------
async function loadConversations() {
  state.conversations = await api("GET", `/api/v1/projects/${state.projectId}/conversations`);
  renderConversations();
}

function renderConversations() {
  const list = $("#conversation-list");
  if (!state.conversations.length) {
    list.replaceChildren(h("li", { class: "empty-note" }, "Your chats will appear here."));
    return;
  }
  list.replaceChildren(...state.conversations.map((c) => h("li", {
    class: c.id === state.conversationId ? "active" : "", tabindex: "0", role: "button",
    onclick: () => openConversation(c.id),
    onkeydown: (e) => { if (e.key === "Enter") openConversation(c.id); },
  },
  h("span", { class: "label" }, c.title),
  h("button", { class: "del", title: "Delete chat", "aria-label": `Delete ${c.title}`,
    onclick: (e) => { e.stopPropagation(); deleteConversation(c.id); } }, "✕"))));
}

async function openConversation(id) {
  const view = newView();
  state.conversationId = id;
  renderConversations();
  closeSidebar();
  let convo;
  try {
    convo = await api("GET", `/api/v1/conversations/${id}`);
  } catch (ex) {
    toast(ex.message);
    return;
  }
  if (view !== state.view) return;  // the user opened another chat meanwhile
  state.turns = convo.runs.map((run) => ({ run, evidence: null }));
  renderThread();
  for (const t of state.turns) {
    if (t.run.status === "queued" || t.run.status === "running") poll(t).catch(() => {});
    else if (t.run.status === "completed") loadEvidence(t).catch(() => {});
  }
}

async function deleteConversation(id) {
  await api("DELETE", `/api/v1/conversations/${id}`);
  if (id === state.conversationId) newChat();
  await loadConversations();
}

function newChat() {
  newView();
  state.conversationId = null;
  state.turns = [];
  renderConversations();
  renderThread();
  closeSidebar();
  $("#question").focus();
}

// ---------- asking ----------
async function ask(question) {
  if (state.busy || !question) return;
  const useWeb = $("#web-toggle").checked && state.webAvailable;
  if (!hasReadyPapers() && !useWeb) {  // web-only chat needs no papers; papers uploaded later join later turns
    toast(state.webAvailable ? "No papers yet — turn on Web search or upload a PDF."
      : "Upload a PDF and wait until it is indexed before asking.");
    return;
  }
  setBusy(true);
  const view = state.view;
  const projectId = state.projectId;
  try {
    let conversationId = state.conversationId;
    if (!conversationId) conversationId = (await api("POST", `/api/v1/projects/${projectId}/conversations`, {})).id;
    const run = await api("POST", `/api/v1/projects/${projectId}/research-runs`, {
      question, conversation_id: conversationId, web_search: useWeb,
    });
    if (view !== state.view) {  // user switched chat/project while we were posting: don't graft it onto that view
      toast("Your question was sent to the previous chat.");
      loadConversations().catch(() => {});
      return;
    }
    state.conversationId = conversationId;
    const turn = { run, evidence: null };
    state.turns.push(turn);
    $("#question").value = "";
    autosize();
    renderThread();
    loadConversations().catch(() => {});
    await poll(turn);
  } catch (ex) {
    toast(ex.message);
  } finally {
    setBusy(false);
  }
}

async function poll(turn) {
  const view = state.view;
  let wait = POLL_MS;
  while (turn.run.status === "queued" || turn.run.status === "running") {
    await new Promise((r) => setTimeout(r, wait));
    if (view !== state.view || !state.turns.includes(turn)) return;  // user switched chats or logged out
    try {
      turn.run = await api("GET", `/api/v1/research-runs/${turn.run.id}`);
      wait = POLL_MS;
    } catch {
      wait = Math.min(wait * 2, 30000);  // network blip: back off and keep trying
      continue;
    }
    renderTurn(turn);
  }
  if (turn.run.status === "completed" && view === state.view) await loadEvidence(turn);
}

async function loadEvidence(turn) {
  turn.evidence = await api("GET", `/api/v1/research-runs/${turn.run.id}/evidence`);
  renderTurn(turn);
}

function setBusy(b) {
  state.busy = b;
  $("#send").disabled = b;
}

// ---------- rendering ----------
function renderThread() {
  const has = state.turns.length > 0;
  $("#hero").hidden = has;
  $("#thread").hidden = !has;
  $("#thread").replaceChildren(...state.turns.map((t) => {
    t.el = h("article", { class: "turn" });
    fillTurn(t);
    return t.el;
  }));
  scrollToEnd();
}

function renderTurn(turn) {
  if (!turn.el) return;
  fillTurn(turn);
}

function scrollToEnd() {
  const th = $("#thread");
  th.scrollTop = th.scrollHeight;
}

function fillTurn(turn) {
  const { run } = turn;
  const parts = [h("h2", { class: "q" }, run.question)];
  if (run.standalone_question) parts.push(h("p", { class: "rewritten" }, `Searched as: “${run.standalone_question}”`));

  if (run.status === "queued" || run.status === "running") {
    parts.push(h("div", { class: "card" },
      h("div", { class: "progress" }, h("span", { class: "spinner" }),
        run.status === "queued" ? "Queued…" : `${!run.web_search ? "Reading your papers" : hasReadyPapers() ? "Reading your papers and the web" : "Searching the web"}, then verifying each claim…`),
      h("div", { class: "skeleton", style: "width:92%" }), h("div", { class: "skeleton", style: "width:76%" }),
      h("div", { class: "skeleton", style: "width:84%" })));
  } else if (run.status === "failed") {
    parts.push(h("div", { class: "card" }, h("span", { class: "badge bad" }, "Failed"), " ",
      h("span", { class: "muted" }, run.error || "The research run failed.")));
  } else if (run.result) {
    parts.push(...answerCards(turn));
  }
  turn.el.replaceChildren(...parts);
}

function sourceKey(e) {
  return e.url || `${e.source_filename}|${(e.page_numbers || []).join(",")}`;
}

function answerCards(turn) {
  const { run, evidence } = turn;
  const res = run.result;
  const byEid = new Map((evidence || []).map((e) => [e.eid, e]));
  // Number sources in order of first citation, merging passages from the same page/URL.
  const order = [];
  const numOf = new Map();
  for (const c of res.claims) {
    for (const eid of c.evidence_ids) {
      const e = byEid.get(eid);
      const key = e ? sourceKey(e) : eid;
      if (!numOf.has(key)) { numOf.set(key, order.length + 1); order.push({ key, e: e || null, eids: [] }); }
      order[numOf.get(key) - 1].eids.push(eid);
    }
  }
  const cid = (n) => `src-${run.id}-${n}`;

  const nOk = res.claims.filter((c) => c.verdict === "SUPPORTED").length;
  const nThin = res.claims.length - nOk;
  const verified = res.status === "verified";
  const meta = h("div", { class: "meta" },
    h("span", { class: `badge ${verified ? "ok" : "review"}` }, verified ? "✓ Verified" : "Needs review"),
    run.web_search ? h("span", { class: "badge web" }, evidence?.length && !evidence.some((e) => e.source_type !== "web") ? "Web only" : "Web + papers")
      : h("span", { class: "badge neutral" }, "Papers only"),
    h("span", { class: "muted small" }, `${res.claims.length} finding${res.claims.length === 1 ? "" : "s"} · ${order.length} source${order.length === 1 ? "" : "s"}`));

  const summary = h("div", { class: "card" }, meta);
  if (res.claims.length) {
    const total = res.claims.length;
    summary.append(
      h("div", { class: "meter", role: "img", "aria-label": `${nOk} supported, ${nThin} thinly supported` },
        h("span", { class: "m-ok", style: `width:${(100 * nOk) / total}%` }),
        h("span", { class: "m-thin", style: `width:${(100 * nThin) / total}%` })),
      h("div", { class: "meter-legend" }, h("span", {}, `● ${nOk} supported by the cited text`),
        nThin ? h("span", {}, `● ${nThin} thinly supported`) : null),
      h("h3", { style: "margin-top:16px" }, "Key findings"),
      h("ol", { class: "findings" }, res.claims.map((c, i) => h("li", {},
        h("span", { class: "n" }, `${i + 1}.`),
        h("div", {}, c.text,
          c.verdict === "THINLY_SUPPORTED" ? h("span", { class: "thin-note" }, "(thinly supported)") : null,
          [...new Set(c.evidence_ids.map((eid) => numOf.get(byEid.get(eid) ? sourceKey(byEid.get(eid)) : eid)))].map((n) => {
            const isWeb = order[n - 1]?.e?.source_type === "web";
            return h("button", { class: `cite${isWeb ? " web" : ""}`, type: "button", title: "Show source",
              onclick: () => flashSource(cid(n)) }, n);
          }))))));
  } else {
    summary.append(h("p", { class: "muted", style: "margin:0" }, "No claim in the sources answers this question."));
  }
  if (res.gaps) summary.append(h("div", { class: "gaps" }, h("strong", {}, "What the sources don’t say"), res.gaps));

  const cards = [summary];
  if (order.length) {
    cards.push(h("div", { class: "card" }, h("h3", {}, "Sources"),
      evidence == null ? h("div", { class: "progress" }, h("span", { class: "spinner" }), "Loading sources…")
        : h("div", { class: "sources" }, order.map((s, i) => sourceCard(s, i + 1, cid(i + 1))))));
  }
  return cards;
}

function safeUrl(u) {
  try { const x = new URL(u); return x.protocol === "http:" || x.protocol === "https:" ? x.href : null; } catch { return null; }
}

function sourceCard(s, n, id) {
  const e = s.e;
  if (!e) return h("div", { class: "source", id }, h("div", { class: "s-head" }, h("span", { class: "s-num" }, `[${n}]`),
    h("span", { class: "s-title" }, "Source unavailable")));
  const isWeb = e.source_type === "web";
  const url = isWeb ? safeUrl(e.url) : null;
  let host = "";
  if (url) { try { host = new URL(url).hostname.replace(/^www\./, ""); } catch { /* ignore */ } }
  const pages = (e.page_numbers || []).length ? `p. ${e.page_numbers.join(", ")}` : "";
  const section = (e.section_headings || []).slice(-1)[0];
  const card = h("div", { class: `source${isWeb ? " web" : ""}`, id },
    h("div", { class: "s-head" }, h("span", { class: "s-num" }, `[${n}]`),
      url ? h("a", { class: "s-title", href: url, target: "_blank", rel: "noopener noreferrer" }, e.title || e.source_filename)
        : h("span", { class: "s-title" }, e.source_filename)),
    h("div", { class: "s-meta" }, isWeb ? `🌐 ${host || "web"}` : `📄 PDF${pages ? ` · ${pages}` : ""}${section ? ` · ${section}` : ""}`),
    h("div", { class: "s-text" }, e.text));
  if ((e.text || "").length > 260) {
    const more = h("button", { class: "btn link more", type: "button" }, "Show more");
    more.addEventListener("click", () => {
      card.classList.toggle("open");
      more.textContent = card.classList.contains("open") ? "Show less" : "Show more";
    });
    card.append(more);
  }
  return card;
}

function flashSource(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1600);
}

// ---------- layout helpers ----------
function autosize() {
  const ta = $("#question");
  ta.style.height = "auto";
  ta.style.height = `${Math.min(ta.scrollHeight, 180)}px`;
}
function openSidebar() { $("#sidebar").classList.add("open"); }
function closeSidebar() { $("#sidebar").classList.remove("open"); }

async function openProject() {
  newView();
  state.conversationId = null;
  state.turns = [];
  state.documents = [];  // never judge the new project by the old one's papers
  renderThread();
  renderDocuments();
  await Promise.all([loadDocuments(), loadConversations()]);
}

async function boot() {
  $("#auth").hidden = true;
  $("#app").hidden = false;
  $("#user-email").textContent = state.email || "";
  try {
    const cfg = await api("GET", "/api/v1/config");
    state.webAvailable = cfg.web_search;
    const wrap = $("#web-toggle-wrap");
    wrap.classList.toggle("disabled", !cfg.web_search);
    $("#web-toggle").disabled = !cfg.web_search;
    if (!cfg.web_search) wrap.title = "Web search is not enabled on this server";
    await loadProjects();
    await openProject();
  } catch (ex) {
    toast(ex.message);
  }
}

// ---------- wiring ----------
function wire() {
  $("#auth-form").addEventListener("submit", submitAuth);
  $("#auth-toggle").addEventListener("click", () => setAuthMode(!state.registering));
  $("#logout").addEventListener("click", logout);
  $("#new-chat").addEventListener("click", newChat);
  $("#new-project").addEventListener("click", () => newProject().catch((e) => toast(e.message)));
  $("#project-select").addEventListener("change", (e) => {
    state.projectId = e.target.value;
    store.set("maxrim.project", state.projectId);
    openProject().catch((ex) => toast(ex.message));
  });
  $("#upload").addEventListener("change", (e) => upload([...e.target.files]));
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask($("#question").value.trim()); });
  $("#question").addEventListener("input", autosize);
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("#question").value.trim()); }
  });
  $("#open-side").addEventListener("click", openSidebar);
  $("#close-side").addEventListener("click", closeSidebar);
  $("#examples").replaceChildren(...EXAMPLES.map((q) => h("button", { class: "chip-btn", type: "button",
    onclick: () => { $("#question").value = q; autosize(); $("#question").focus(); } }, q)));
}

wire();
setAuthMode(false);
if (state.token) boot(); else showAuth();
