"use strict";

const PAGE_SIZE = 10;
const $ = (id) => document.getElementById(id);
const form = $("search-form");
const input = $("q");
const list = $("suggestions");
const statusEl = $("status");
const hitsEl = $("hits");

let page = 0;
let activeIndex = -1;
let suggestTimer = null;

// ---------- search ----------

function params(extra = {}) {
  const p = new URLSearchParams({ q: input.value.trim(), ...extra });
  if ($("start").value) p.set("start", $("start").value);
  if ($("end").value) p.set("end", $("end").value);
  return p;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

// Only link out to http(s) URLs; anything else (javascript:, data:, garbage) renders as plain text.
function safeUrl(raw) {
  try {
    const url = new URL(raw);
    return url.protocol === "https:" || url.protocol === "http:" ? url : null;
  } catch {
    return null;
  }
}

// Build highlighted text with DOM nodes (never innerHTML) so headlines cannot inject markup.
function highlight(text, terms) {
  const frag = document.createDocumentFragment();
  if (!terms.length) {
    frag.append(text);
    return frag;
  }
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`\\b(${escaped.join("|")})\\w*`, "gi");
  let last = 0;
  for (const m of text.matchAll(re)) {
    frag.append(text.slice(last, m.index));
    const mark = document.createElement("mark");
    mark.textContent = m[0];
    frag.append(mark);
    last = m.index + m[0].length;
  }
  frag.append(text.slice(last));
  return frag;
}

async function runSearch(newPage = 0) {
  if (!input.value.trim()) return;
  page = newPage;
  closeSuggestions();
  statusEl.textContent = "Searching…";
  try {
    const data = await getJSON(`/api/search?${params({ k: PAGE_SIZE, offset: page * PAGE_SIZE })}`);
    renderHits(data);
    if (page === 0) renderTimeline(await getJSON(`/api/timeline?${params()}`));
  } catch (err) {
    statusEl.textContent = `Search failed: ${err.message}`;
  }
}

function renderHits(data) {
  hitsEl.replaceChildren();
  hitsEl.start = page * PAGE_SIZE + 1;
  const shown = data.hits.length;
  statusEl.textContent = data.total
    ? `${data.total.toLocaleString()} headlines · showing ${page * PAGE_SIZE + 1}–${page * PAGE_SIZE + shown} · ${data.took_ms} ms`
    : `No headlines match “${data.query}”.`;

  for (const hit of data.hits) {
    const li = document.createElement("li");
    const title = document.createElement("div");
    title.className = "title";
    const link = safeUrl(hit.url);
    if (link) {
      const a = document.createElement("a");
      a.href = link.href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.append(highlight(hit.title, data.terms));
      title.append(a);
    } else {
      title.append(highlight(hit.title, data.terms));
    }
    const meta = document.createElement("div");
    meta.className = "meta";
    const time = document.createElement("time");
    time.dateTime = hit.published;
    time.textContent = new Date(hit.published).toUTCString().replace(" GMT", " UTC");
    meta.append(time);
    if (link) meta.append(` · ${link.hostname.replace(/^www\./, "")}`);
    meta.append(` · score ${hit.score.toFixed(2)}`);
    li.append(title, meta);
    hitsEl.append(li);
  }
  $("pager").hidden = data.total <= PAGE_SIZE;
  $("prev").disabled = page === 0;
  $("next").disabled = (page + 1) * PAGE_SIZE >= data.total;
}

function renderTimeline(data) {
  const svg = $("timeline");
  const months = data.buckets;
  const unit = data.granularity;
  $("timeline-heading").textContent = `Matches per ${unit}`;
  $("timeline-section").hidden = months.length < 2;
  svg.querySelectorAll("rect").forEach((r) => r.remove());
  if (months.length < 2) return;

  const max = Math.max(...months.map((m) => m.count));
  const w = 1000;
  const h = 80;
  const bw = w / months.length;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  months.forEach((m, i) => {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    const bh = Math.max(1, (m.count / max) * (h - 4));
    rect.setAttribute("x", i * bw + 1);
    rect.setAttribute("y", h - bh);
    rect.setAttribute("width", Math.max(1, bw - 2));
    rect.setAttribute("height", bh);
    const tip = document.createElementNS("http://www.w3.org/2000/svg", "title");
    tip.textContent = `${m.period}: ${m.count}`;
    rect.append(tip);
    svg.append(rect);
  });
  const peak = months.reduce((a, b) => (b.count > a.count ? b : a));
  const caption = `${months[0].period} to ${months.at(-1).period}; peak ${peak.count.toLocaleString()} on ${peak.period}.`;
  $("timeline-caption").textContent = caption;
  $("timeline-title").textContent = `Headline counts per ${unit}, ${caption}`;
}

// ---------- autocomplete (WAI-ARIA combobox pattern) ----------

function currentWord() {
  const words = input.value.split(/\s+/);
  return words.at(-1).toLowerCase();
}

function closeSuggestions() {
  list.hidden = true;
  list.replaceChildren();
  input.setAttribute("aria-expanded", "false");
  input.removeAttribute("aria-activedescendant");
  activeIndex = -1;
}

function choose(term) {
  const words = input.value.split(/\s+/);
  words[words.length - 1] = term;
  input.value = `${words.join(" ")} `;
  closeSuggestions();
  input.focus();
}

function setActive(i) {
  const items = [...list.children];
  if (!items.length) return;
  activeIndex = (i + items.length) % items.length;
  items.forEach((li, j) => li.setAttribute("aria-selected", String(j === activeIndex)));
  input.setAttribute("aria-activedescendant", items[activeIndex].id);
}

async function updateSuggestions() {
  const prefix = currentWord();
  if (prefix.length < 2) return closeSuggestions();
  const { suggestions } = await getJSON(`/api/suggest?${new URLSearchParams({ prefix })}`);
  list.replaceChildren();
  suggestions.forEach((term, i) => {
    const li = document.createElement("li");
    li.id = `suggestion-${i}`;
    li.role = "option";
    li.textContent = term;
    li.setAttribute("aria-selected", "false");
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      choose(term);
    });
    list.append(li);
  });
  list.hidden = suggestions.length === 0;
  input.setAttribute("aria-expanded", String(!list.hidden));
  activeIndex = -1;
}

input.addEventListener("input", () => {
  clearTimeout(suggestTimer);
  suggestTimer = setTimeout(() => updateSuggestions().catch(closeSuggestions), 120);
});

input.addEventListener("keydown", (e) => {
  if (list.hidden) return;
  if (e.key === "ArrowDown") { e.preventDefault(); setActive(activeIndex + 1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); setActive(activeIndex - 1); }
  else if (e.key === "Escape") { closeSuggestions(); }
  else if (e.key === "Enter" && activeIndex >= 0) { e.preventDefault(); choose(list.children[activeIndex].textContent); }
});

input.addEventListener("blur", () => setTimeout(closeSuggestions, 100));

form.addEventListener("submit", (e) => {
  e.preventDefault();
  runSearch(0);
});
$("prev").addEventListener("click", () => { runSearch(page - 1); $("results").focus(); });
$("next").addEventListener("click", () => { runSearch(page + 1); $("results").focus(); });

getJSON("/api/stats")
  .then((s) => {
    const from = new Date(s.min_ts * 1000).toISOString().slice(0, 10);
    const to = new Date(s.max_ts * 1000).toISOString().slice(0, 10);
    $("corpus-stats").textContent =
      `${s.num_docs.toLocaleString()} unique oil-market headlines from GDELT, ${from} to ${to}.`;
  })
  .catch(() => {});
