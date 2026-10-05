/* Plain-text rendering only. Dataset coordinates are Unicode code points. */
"use strict";
const $ = id => document.getElementById(id);
const state = {datasets: [], dataset: null, rows: [], total: 0, offset: 0, limit: 50,
  record: null, span: null, mode: new URL(location.href).searchParams.get("mode") === "compare" ? "compare" : "document",
  listRequest: null, recordRequest: null};
const filterIds = ["model", "collection", "cohort", "kind", "split"];
const number = n => Number(n).toLocaleString();
const percent = n => `${Math.round(n * 100)}%`;
const friendly = s => s ? s.replaceAll("_", " ") : "Unspecified";
const element = (tag, cls, text) => {const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = text; return e;};
const codepoints = text => Array.from(text);
let searchTimer, toastTimer;

function apiURL(path, params = {}) {
  const url = new URL(path, location.origin);
  if (state.dataset) url.searchParams.set("dataset", state.dataset.id);
  Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  return url;
}
async function fetchJSON(path, params = {}, signal) {
  const response = await fetch(apiURL(path, params), {signal});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}
function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $("toast").hidden = true, 3200);
}
async function copy(text, label) {
  try {await navigator.clipboard.writeText(text); toast(label);}
  catch {toast("Clipboard unavailable. You can select and copy the text directly.");}
}
function updateURL() {
  const url = new URL(location.href);
  url.searchParams.set("dataset", state.dataset.id);
  if (state.record) url.searchParams.set("id", state.record.id); else url.searchParams.delete("id");
  if (state.mode === "compare") url.searchParams.set("mode", "compare"); else url.searchParams.delete("mode");
  history.replaceState(null, "", url);
}
function filterParams(offset = state.offset) {
  const params = {offset, limit: state.limit, q: $("search").value};
  filterIds.forEach(id => params[id] = $(id).value);
  return params;
}
function resetFilters(kind = "mixed") {
  clearTimeout(searchTimer); $("search").value = "";
  filterIds.forEach(id => $(id).value = id === "kind" ? kind : "");
  state.offset = 0;
}
function populateSelect(id, options, label) {
  const select = $(id); select.replaceChildren(new Option(label, ""));
  options.forEach(o => select.add(new Option(`${o.label} (${number(o.count)})`, o.value)));
}
async function selectDataset(id, initialID) {
  state.listRequest?.abort(); state.recordRequest?.abort();
  state.dataset = state.datasets.find(d => d.id === id) || state.datasets[0];
  $("dataset").value = state.dataset.id;
  populateSelect("model", state.dataset.models, "All models");
  populateSelect("collection", state.dataset.collections, "All collections");
  populateSelect("cohort", state.dataset.cohorts, "All batches");
  populateSelect("split", state.dataset.splits, "All splits");
  $("dataset-stats").replaceChildren();
  [[state.dataset.mixed, "mixed documents"], [state.dataset.controls, "human controls"], [state.dataset.models.length, "AI models"]].forEach(([n, label]) => {
    const stat = element("div", "stat"); stat.append(element("strong", "", number(n)), element("span", "", label)); $("dataset-stats").append(stat);
  });
  state.record = null; state.span = null; resetFilters();
  $("document").hidden = true; $("empty").hidden = false; $("reader-loading").hidden = true;
  renderInspector();
  if (initialID) {$("kind").value = ""; $("search").value = initialID;}
  await loadList({initialID});
}
async function loadList({initialID} = {}) {
  state.recordRequest?.abort();
  state.listRequest?.abort(); const controller = new AbortController(); state.listRequest = controller;
  $("result-count").textContent = "Finding documents…";
  $("record-list").setAttribute("aria-busy", "true");
  try {
    const data = await fetchJSON("/api/records", filterParams(), controller.signal);
    if (controller !== state.listRequest) return;
    state.rows = data.rows; state.total = data.total;
    renderList();
    if (initialID) await loadRecord(initialID);
    else if (!state.rows.some(r => r.id === state.record?.id)) {
      if (state.rows.length) await loadRecord(state.rows[0].id);
      else clearDocument("No documents match.", "Try a different model, title, or collection — or reset the filters.");
    }
  } catch (e) {
    if (e.name !== "AbortError") {$("result-count").textContent = "Could not load results"; $("record-list").replaceChildren(element("p", "list-message error", e.message)); toast(e.message);}
  } finally {if (controller === state.listRequest) $("record-list").removeAttribute("aria-busy");}
}
function renderList() {
  $("result-count").textContent = `${number(state.total)} document${state.total === 1 ? "" : "s"}`;
  $("record-list").replaceChildren();
  if (!state.rows.length) $("record-list").append(element("p", "list-message", "No matches. Try broadening the filters."));
  state.rows.forEach(row => {
    const card = element("button", `record-card${row.id === state.record?.id ? " selected" : ""}`);
    card.dataset.id = row.id; card.title = row.id; card.setAttribute("aria-pressed", String(row.id === state.record?.id));
    card.append(element("div", "record-title", row.title), element("div", "record-author", row.source_author || "Unspecified author"));
    const tags = element("div", "record-tags");
    tags.append(element("span", "tag", row.kind === "human_control" ? "Human control" : row.model_labels.join(" · ")), element("span", "", friendly(row.split)), element("span", "ai-percent", `${percent(row.ai_fraction)} AI`));
    card.append(tags); card.addEventListener("click", () => loadRecord(row.id)); $("record-list").append(card);
  });
  $("page-info").textContent = state.total ? `${number(state.offset + 1)}–${number(Math.min(state.offset + state.limit, state.total))} of ${number(state.total)}` : "0 results";
  $("previous-page").disabled = state.offset === 0;
  $("next-page").disabled = state.offset + state.limit >= state.total;
  $("random").disabled = state.total === 0;
}
function clearDocument(title, message) {
  state.recordRequest?.abort(); state.record = null; state.span = null;
  $("document").hidden = true; $("empty").hidden = false; $("reader-loading").hidden = true;
  $("empty").querySelector("h1").textContent = title; $("empty").querySelector("p").textContent = message;
  renderInspector(); updateURL();
}
async function loadRecord(id) {
  state.recordRequest?.abort(); const controller = new AbortController(); state.recordRequest = controller;
  $("reader-loading").hidden = false;
  try {
    const record = await fetchJSON("/api/record", {id}, controller.signal);
    if (controller !== state.recordRequest) return false;
    state.record = record; state.span = null;
    document.body.classList.remove("inspecting");
    renderDocument(); renderList(); renderInspector(); updateURL();
    $("document").hidden = false; $("empty").hidden = true;
    document.querySelector(".reader").scrollTo({top: 0});
    return true;
  } catch (e) {if (e.name !== "AbortError") toast(e.message); return false;}
  finally {if (controller === state.recordRequest) $("reader-loading").hidden = true;}
}
function renderDocument() {
  const r = state.record;
  $("document-title").textContent = r.title;
  $("document-author").textContent = r.source_author ? `Source by ${r.source_author}` : "Source author unspecified";
  const collection = state.dataset.collections.find(c => c.value === r.source_dataset)?.label || r.source_dataset;
  const cohort = state.dataset.cohorts.find(c => c.value === r.cohort)?.label || friendly(r.cohort);
  $("document-eyebrow").textContent = `${collection} / ${cohort} / ${r.split}`;
  $("document-id").textContent = r.id;
  $("output-heading").textContent = r.kind === "human_control" ? "Human control" : "Mixed document";
  $("original-caption").textContent = r.kind === "human_control" ? "Retained in full" : "Before replacement";
  $("output-caption").textContent = r.kind === "human_control" ? "Unchanged source text" : "After replacement";
  const meta = $("document-meta"); meta.replaceChildren();
  meta.append(element("span", `badge ${r.kind === "human_control" ? "human" : "ai"}`, r.kind === "human_control" ? "Human control" : r.model_labels.join(" · ")));
  meta.append(element("span", "badge", `${number(r.characters)} characters`), element("span", "badge", `${percent(r.ai_fraction)} AI`), element("span", "badge", `${r.ai_spans} replacement${r.ai_spans === 1 ? "" : "s"}`));
  if (r.kind === "human_control") {
    const matched = [...new Set(r.paired_records.filter(p => p.kind === "mixed").flatMap(p => p.models))];
    if (matched.length) meta.append(element("span", "badge", `Matched to ${matched.join(" · ")}`));
  }
  const sourceLink = $("source-link"); sourceLink.hidden = true; sourceLink.removeAttribute("href");
  try {const url = new URL(r.source_reference); if (["https:", "http:"].includes(url.protocol)) {sourceLink.href = url.href; sourceLink.hidden = false;}} catch { /* Non-web source reference is visible in provenance. */ }
  $("export").href = apiURL("/api/export", {id: r.id}).href;
  $("pairs").replaceChildren();
  r.paired_records.forEach(pair => {
    const button = element("button", "text-button", pair.kind === "human_control" ? "Open human control ↗" : `Open ${pair.models.join(" / ")} version ↗`);
    button.addEventListener("click", async () => {resetFilters(""); $("search").value = pair.id; await loadList({initialID: pair.id});});
    $("pairs").append(button);
  });
  $("next-ai").disabled = !r.ai_spans;
  renderPassages();
}
function passageNode(text, span, cls, label, original = false) {
  const node = element(state.mode === "compare" ? "div" : "span", cls);
  node.dataset.span = span.index; node.tabIndex = 0; node.setAttribute("role", "button");
  const author = span.label === "ai" && !original ? (span.requested_model || span.reported_model || span.backend || "AI") : state.record.source_author || "Human";
  node.setAttribute("aria-label", `${original ? "Original " : ""}${span.label === "ai" && !original ? "AI" : "Human"} passage ${span.index + 1}, ${author}`);
  if (label) node.append(element("span", "comparison-label", label));
  const content = element("span", "passage-text", text); node.append(content);
  node.addEventListener("click", () => selectSpan(span.index));
  node.addEventListener("keydown", e => {if (e.key === "Enter" || e.key === " ") {e.preventDefault(); selectSpan(span.index);}});
  return node;
}
function renderPassages() {
  $("mode-document").setAttribute("aria-pressed", String(state.mode === "document"));
  $("mode-compare").setAttribute("aria-pressed", String(state.mode === "compare"));
  $("compare-heading").hidden = state.mode !== "compare";
  const container = $("passages"); container.replaceChildren();
  container.className = `passages${state.mode === "compare" ? " compare" : ""}${$("color-spans").checked ? " colored" : ""}`;
  if (!state.record) return;
  const output = codepoints(state.record.text), original = codepoints(state.record.source_text);
  let replacement = 0;
  state.record.spans.forEach(span => {
    if (span.label === "ai") replacement++;
    const outputText = output.slice(span.start, span.end).join("");
  if (state.mode === "document") container.append(passageNode(outputText, span, `passage ${span.label}`));
    else {
      const row = element("div", "comparison-row");
      row.append(passageNode(original.slice(span.source_start, span.source_end).join(""), span, `comparison-cell human${span.label === "ai" ? " replaced-original" : ""}`, span.label === "ai" ? `ORIGINAL · Passage replaced by AI (${replacement})` : "ORIGINAL · Retained human text", true));
      const outputLabel = state.record.kind === "human_control" ? "CONTROL" : "MIXED";
      row.append(passageNode(outputText, span, `comparison-cell ${span.label}`, span.label === "ai" ? `${outputLabel} · AI replacement ${replacement}` : `${outputLabel} · Retained human text`));
      container.append(row);
    }
  });
  highlightSpan();
}
function highlightSpan() {
  document.querySelectorAll("[data-span]").forEach(node => {
    const selected = Number(node.dataset.span) === state.span;
    node.classList.toggle("selected", selected); node.setAttribute("aria-pressed", String(selected));
  });
}
function selectSpan(index) {state.span = index; document.body.classList.add("inspecting"); highlightSpan(); renderInspector();}
function detail(label, value, cls = "") {
  const group = element("div", "detail-group");
  group.append(element("div", "detail-label", label), element("div", `detail-value ${cls}`, value ?? "Not recorded"));
  return group;
}
function promptDetails(label, text) {
  const details = element("details", ""); details.append(element("summary", "", label), element("pre", "", text || "Not recorded")); return details;
}
function renderInspector() {
  const panel = $("inspection"); panel.replaceChildren();
  const r = state.record, span = r?.spans[state.span];
  if (!span) {
    document.body.classList.remove("inspecting");
    const empty = element("div", "inspection-empty"); empty.append(element("span", "inspection-symbol", "↖"), element("h2", "", "Follow the provenance."), element("p", "", "Select any passage to inspect its author and exact span. AI passages include the brief and prompts used to generate them.")); panel.append(empty); return;
  }
  const ai = span.label === "ai", replacement = r.replacements.find(v => v.id === span.replacement_id);
  const count = r.spans.slice(0, span.index + 1).filter(s => s.label === "ai").length;
  panel.append(element("span", `badge ${ai ? "ai" : "human"}`, ai ? "AI generated" : "Human · retained"));
  panel.append(element("h2", "inspection-title", ai ? `Replacement ${count}` : "Source passage"), element("p", "inspection-subtitle", ai ? "Inserted from a condensed brief." : "Copied unchanged from the original document."));
  if (ai) {
    panel.append(detail("Requested model", span.requested_model, "mono"), detail("Reported model", span.reported_model || "Not reported", "mono"));
    panel.append(element("p", "inspection-note", span.model_identity_status === "requested_only" ? "Model identity is requested only; the provider did not report the served model." : span.model_identity_status ? `Identity status: ${friendly(span.model_identity_status)}.` : "Model identity status was not recorded."));
    if (span.backend) panel.append(detail("Backend", span.backend));
  } else panel.append(detail("Source author", span.author_id || r.source_author || "Unspecified"));
  const coords = element("div", "detail-group coordinates");
  [["Document span", span.start, span.end], ["Source span", span.source_start, span.source_end]].forEach(([label, a, b]) => {const c = element("div", ""); c.append(element("div", "detail-label", label), element("div", "detail-value mono", `[${number(a)}, ${number(b)})`)); coords.append(c);});
  panel.append(coords, element("p", "inspection-note", "Positions count Unicode characters. The ending position is exclusive."));
  if (ai) {
    panel.append(detail("Condensed brief", replacement?.brief, "brief"));
    if (replacement?.generated_at) panel.append(detail("Generated", replacement.generated_at));
    panel.append(promptDetails("Generation prompt", replacement?.prompt), promptDetails("Brief prompt", replacement?.brief_prompt));
    if (replacement?.quality) panel.append(promptDetails("Recorded generation checks", JSON.stringify(replacement.quality, null, 2)));
  }
  panel.append(detail("Source ID", r.source_id, "mono"), detail("Source reference", r.source_reference, "mono"), detail("Source license", r.source_license));
  panel.append(detail("Source verification", r.human_verified === true ? "Marked verified in source metadata" : r.human_verified === false ? "Human-origin candidate; not marked verified" : "Not recorded"));
}
async function lookup() {
  const id = $("lookup-id").value.trim(); $("lookup-error").textContent = "";
  try {await fetchJSON("/api/record", {id}); resetFilters(""); $("search").value = id; await loadList({initialID: id}); $("lookup-dialog").close();}
  catch (e) {$("lookup-error").textContent = e.message;}
}
$("dataset").addEventListener("change", () => selectDataset($("dataset").value));
filterIds.forEach(id => $(id).addEventListener("change", () => {state.offset = 0; loadList();}));
$("search").addEventListener("input", () => {clearTimeout(searchTimer); searchTimer = setTimeout(() => {state.offset = 0; loadList();}, 220);});
$("reset").addEventListener("click", () => {resetFilters(); loadList();});
$("previous-page").addEventListener("click", () => {state.offset = Math.max(0, state.offset - state.limit); loadList();});
$("next-page").addEventListener("click", () => {state.offset += state.limit; loadList();});
$("random").addEventListener("click", async () => {
  if (!state.total) return;
  try {const offset = Math.floor(Math.random() * state.total); const data = await fetchJSON("/api/records", {...filterParams(offset), limit: 1}); if (!data.rows.length) return; state.offset = Math.floor(offset / state.limit) * state.limit; await loadList({initialID: data.rows[0].id});}
  catch (e) {toast(e.message);}
});
[["mode-document", "document"], ["mode-compare", "compare"]].forEach(([id, mode]) => $(id).addEventListener("click", () => {state.mode = mode; renderPassages(); updateURL();}));
$("color-spans").addEventListener("change", renderPassages);
$("next-ai").addEventListener("click", () => {
  const ai = state.record?.spans.filter(s => s.label === "ai") || [];
  const next = ai.find(s => state.span === null || s.index > state.span) || ai[0];
  if (!next) return; selectSpan(next.index);
  document.querySelector(`.passages [data-span="${next.index}"]`)?.scrollIntoView({behavior: "smooth", block: "center"});
});
$("copy-id").addEventListener("click", () => copy(state.record.id, "Document ID copied"));
$("copy-link").addEventListener("click", () => copy(location.href, "Document link copied"));
$("lookup").addEventListener("click", () => {$("lookup-error").textContent = ""; $("lookup-dialog").showModal(); $("lookup-id").focus();});
$("close-dialog").addEventListener("click", () => $("lookup-dialog").close());
$("close-inspector").addEventListener("click", () => {document.body.classList.remove("inspecting"); state.span = null; highlightSpan(); renderInspector();});
$("lookup-form").addEventListener("submit", e => {e.preventDefault(); lookup();});
document.addEventListener("keydown", e => {if (e.key === "/" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) && !$("lookup-dialog").open) {e.preventDefault(); $("lookup").click();}});
(async () => {
  try {
    const data = await fetchJSON("/api/datasets"); state.datasets = data.datasets;
    state.datasets.forEach(d => $("dataset").add(new Option(d.name, d.id)));
    const params = new URL(location.href).searchParams;
    await selectDataset(params.get("dataset"), params.get("id"));
  } catch (e) {$("result-count").textContent = "Dataset unavailable"; $("empty").querySelector("p").textContent = e.message; toast(e.message);}
})();
