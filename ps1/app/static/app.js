// Tomato Leaf Check client: upload the photo, show Screen 1 at once, swap in the checked Screen 2 when ready.
// All text goes through textContent, never innerHTML, so nothing from the server can inject markup.
"use strict";

const POLL_MS = 1500;
const POLL_LIMIT = 80;                     // about 2 minutes, enough for Llama on a busy Pi

const $ = (id) => document.getElementById(id);
const form = $("check-form");
const photo = $("photo");
const checkBtn = $("check-btn");
const statusEl = $("status");

const ABSTAIN_TITLES = {
  poor_quality: "Please take the photo again",
  not_a_plant: "This does not look like a leaf",
  wrong_crop: "This does not look like a tomato leaf",
  low_confidence: "No clear match",
  unsupported_label: "A problem this assistant does not cover",
};

let previewUrl = null;
let pollTimer = null;
let currentJob = null;                     // only this job's explanation may be drawn (stale polls are ignored)

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function show(el, visible) { el.hidden = !visible; }

function fillList(listEl, items) {
  listEl.replaceChildren(...items.map((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));
}

photo.addEventListener("change", () => {
  const file = photo.files && photo.files[0];
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : null;
  $("preview-img").src = previewUrl || "";
  show($("preview"), Boolean(file));
  checkBtn.disabled = !file;
  setStatus("");
});

function renderScreen1(data) {
  const f = data.finding;
  const s1 = data.screen1;
  $("result-kicker").textContent = f.abstained ? "No diagnosis" : "Likely problem";
  $("result-title").textContent = f.abstained ? (ABSTAIN_TITLES[f.abstain_reason] || "No diagnosis") : data.title;
  $("explanation").textContent = s1.explanation;

  const note = $("explanation-source");
  note.classList.toggle("pending", data.screen2_status === "pending");
  note.textContent = data.screen2_status === "pending" ? "Preparing a fuller explanation on this device…" : "";

  fillList($("actions"), s1.actions.map((a) => a.text));
  show($("actions-block"), s1.actions.length > 0);

  $("chemical").textContent = data.chemical_control_line || "";
  show($("chemical"), Boolean(data.chemical_control_line));

  $("escalate-reason").textContent = s1.escalate_reason ? `Why: ${s1.escalate_reason}.` : "";
  $("contact").textContent = s1.contact_source || "";
  show($("escalate"), s1.escalate);

  fillList($("sources"), s1.sources.map((s) => `${s.title} (read ${s.accessed}) ${s.url}`));
  show($("sources-block"), s1.sources.length > 0);

  const total = data.timings_ms.screen1_total;
  $("meta").textContent = `Checked on this device in ${(total / 1000).toFixed(2)} s · model ${f.model_version}` +
    ` · ${f.threshold_version}`;
  show($("result"), true);
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderScreen2(status, screen2) {
  const note = $("explanation-source");
  note.classList.remove("pending");
  if (status === "done" && screen2) {
    $("explanation").textContent = screen2.explanation;
    note.textContent = screen2.explanation_source === "llama"
      ? "Explained by the on-device Llama model and checked before showing."
      : "The fuller explanation did not pass the safety check, so the reviewed text is shown.";
  } else {
    note.textContent = "";
  }
}

async function readJson(res) {
  // A proxy or server crash can answer with HTML; never show a parser error to the farmer.
  try {
    return await res.json();
  } catch (err) {
    return { error: `the device answered with an error (${res.status})` };
  }
}

function pollExplanation(job, tries = 0) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/explain/${encodeURIComponent(job)}`);
      const data = await readJson(res);
      if (job !== currentJob) return;                       // the farmer has moved on to another photo
      if (data.status === "pending" && tries < POLL_LIMIT) return pollExplanation(job, tries + 1);
      renderScreen2(data.status, data.screen2);
    } catch (err) {
      if (job === currentJob) renderScreen2("error", null);
    }
  }, POLL_MS);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = photo.files && photo.files[0];
  if (!file) return;
  clearTimeout(pollTimer);
  currentJob = null;
  checkBtn.disabled = true;
  setStatus("Checking the leaf…");
  const body = new FormData();
  body.append("photo", file, file.name || "leaf.jpg");
  body.append("lang", $("lang").value);
  try {
    const res = await fetch("/api/diagnose", { method: "POST", body });
    const data = await readJson(res);
    if (!res.ok) throw new Error(data.error || `error ${res.status}`);
    setStatus("");
    currentJob = data.job;
    renderScreen1(data);
    if (data.screen2_status === "pending") pollExplanation(data.job);
  } catch (err) {
    setStatus(`Could not check the photo: ${err.message}`, true);
  } finally {
    checkBtn.disabled = !(photo.files && photo.files[0]);
  }
});

$("again").addEventListener("click", () => {
  clearTimeout(pollTimer);
  currentJob = null;
  form.reset();
  photo.value = "";
  photo.dispatchEvent(new Event("change"));
  show($("result"), false);
  window.scrollTo({ top: 0, behavior: "smooth" });
});
