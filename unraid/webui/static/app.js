"use strict";
/* TextyMcSpeechy web UI.
   No framework: the whole app is four views and one list, and a build step
   would cost more than it saves. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = {
  project: null,        // full project object
  view: "projects",
  speaker: null,        // active speaker filter, null = all
  onlyIncluded: false,
  cursor: 0,            // index into the visible clip list
  audio: new Audio(),
  currentJob: null,
  pollTimer: null,
};

/* ---------- plumbing ---------- */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

let toastTimer;
function toast(msg, bad = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("bad", bad);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, bad ? 6000 : 2600);
}

function show(view) {
  state.view = view;
  $$(".view").forEach(v => v.classList.toggle("active", v.id === `view-${view}`));
  $$(".step").forEach(b => b.setAttribute("aria-current", String(b.dataset.view === view)));
  if (view === "review") renderClips();
  if (view === "train") loadCheckpoints();
}

function gateSteps() {
  const p = state.project;
  $$(".step").forEach(b => {
    const v = b.dataset.view;
    b.disabled = !p && v !== "projects";
    if (p && (v === "review" || v === "train")) b.disabled = !(p.clips || []).length;
  });
}

/* ---------- health ---------- */

async function loadHealth() {
  try {
    const h = await api("/api/health");
    const bits = [];
    bits.push(h.gpu
      ? `<span class="${h.gpu_supported ? "ok" : "bad"}">${h.gpu}${h.gpu_supported ? "" : " — unsupported by this PyTorch"}</span>`
      : `<span class="bad">no GPU — check --runtime=nvidia</span>`);
    bits.push(`${h.free_gb} GB free`);
    if (!h.has_checkpoints) {
      // Do not name a language here: en-us is the wrong advice for a British,
      // Australian or Irish voice, and this banner is the first thing read.
      bits.push(`<span class="bad">no pretrained checkpoints — run <code>tms checkpoints &lt;lang&gt;</code> on the console (<code>en-gb</code> or <code>en-us</code>)</span>`);
    }
    $("#health").innerHTML = bits.join(" · ");
  } catch { $("#health").textContent = ""; }
}

/* ---------- projects ---------- */

async function loadProjects() {
  const list = await api("/api/projects");
  $("#project-list").innerHTML = list.length
    ? list.map(p => `
      <div class="card">
        <b>${esc(p.name)}</b>
        <span class="meta">${p.status} · ${p.included}/${p.clips} clips</span>
        <div>
          <button class="primary" data-open="${esc(p.name)}">Open</button>
          <button class="link" data-del="${esc(p.name)}">delete</button>
        </div>
      </div>`).join("")
    : `<p class="hint">No projects yet — create one below.</p>`;
}

async function openProject(name) {
  state.project = await api(`/api/projects/${encodeURIComponent(name)}`);
  state.speaker = null;
  state.cursor = 0;
  gateSteps();
  renderStats();
  $("#batch").value = state.project.batch_size ?? 8;
  $("#scratch").checked = !!state.project.from_scratch;
  $("#restart").checked = !!state.project.restart;
  const hasClips = (state.project.clips || []).length > 0;
  $("#upload-info").hidden = !state.project.source;
  if (state.project.source) {
    $("#upload-info").textContent = `Uploaded: ${state.project.source}`;
    $("#start-import").disabled = false;
  }
  show(hasClips ? "review" : "import");
  pollJobs();
}

/* ---------- import ---------- */

function wireUpload() {
  const drop = $("#drop"), file = $("#file");
  $("#browse").onclick = () => file.click();
  file.onchange = () => file.files[0] && upload(file.files[0]);
  ["dragenter", "dragover"].forEach(e =>
    drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(e =>
    drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", ev => {
    const f = ev.dataTransfer.files[0];
    if (f) upload(f);
  });
}

async function upload(f) {
  if (!state.project) return toast("Open a project first", true);
  const info = $("#upload-info");
  info.hidden = false;
  info.textContent = `Uploading ${f.name}…`;
  const body = new FormData();
  body.append("file", f);
  try {
    const r = await api(`/api/projects/${state.project.name}/upload`, { method: "POST", body });
    info.textContent = `Uploaded ${r.file} (${r.size_mb} MB). Ready to split and transcribe.`;
    $("#start-import").disabled = false;
  } catch (e) {
    info.textContent = `Upload failed: ${e.message}`;
    toast(e.message, true);
  }
}

async function startImport() {
  $("#start-import").disabled = true;
  try {
    const job = await api(`/api/projects/${state.project.name}/import`, { method: "POST" });
    trackJob(job.id, "#import-progress", async () => {
      state.project = await api(`/api/projects/${state.project.name}`);
      gateSteps();
      renderStats();
      show("review");
      toast(`${state.project.clips.length} clips ready to review`);
    });
  } catch (e) {
    toast(e.message, true);
    $("#start-import").disabled = false;
  }
}

/* ---------- job polling ---------- */

function trackJob(jobId, wrapSel, onDone) {
  const wrap = $(wrapSel);
  wrap.hidden = false;
  const fill = $(".fill", wrap), text = $(".progress-text", wrap), log = $(".log", wrap);
  clearInterval(state.pollTimer);
  state.currentJob = jobId;
  state.pollTimer = setInterval(async () => {
    let job;
    try { job = await api(`/api/jobs/${jobId}`); }
    catch { clearInterval(state.pollTimer); return; }

    if (fill) fill.style.width = `${(job.progress * 100).toFixed(1)}%`;
    const epoch = job.epoch != null ? ` · epoch ${job.epoch}` : "";
    text.textContent = `${job.status}${epoch} — ${job.message || ""}`;
    log.textContent = (job.log || []).join("\n");
    log.scrollTop = log.scrollHeight;

    if (job.status !== "running") {
      clearInterval(state.pollTimer);
      if (job.status === "error") toast(job.error || "job failed", true);
      else if (job.status === "done" && onDone) onDone();
      if (job.kind === "train") { $("#stop-train").hidden = true; $("#start-train").hidden = false; }
    }
  }, 1200);
}

async function pollJobs() {
  // Reconnecting to a run already in progress matters: training takes hours and
  // the user will close the tab.
  try {
    const jobs = await api("/api/jobs");
    if (jobs.import) { show("import"); trackJob(jobs.import.id, "#import-progress", () => openProject(state.project.name)); }
    else if (jobs.train) {
      show("train");
      $("#stop-train").hidden = false; $("#start-train").hidden = true;
      trackJob(jobs.train.id, "#train-progress", loadCheckpoints);
    }
  } catch { /* nothing running */ }
}

/* ---------- review ---------- */

const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function visibleClips() {
  return (state.project?.clips || []).filter(c =>
    (state.speaker === null || c.speaker === state.speaker) &&
    (!state.onlyIncluded || c.include));
}

function renderStats() {
  const s = state.project?.stats;
  if (!s) return;
  $("#stats").innerHTML =
    `<b>${s.included_minutes} min</b> across <b>${s.included}</b> of ${s.total} clips` +
    (s.low_confidence ? ` · <span class="low">${s.low_confidence} low confidence</span>` : "");

  const warn = $("#warnings");
  const list = state.project.warnings || [];
  warn.innerHTML = list.map(w => `<div class="notice warn">${esc(w)}</div>`).join("") +
    (s.included_minutes < 10
      ? `<div class="notice warn">Only ${s.included_minutes} minutes included. Under
         about 10 minutes the voice will be rough — include more clips if you can.</div>` : "");

  const speakers = state.project.speakers || 1;
  $("#speaker-filter").innerHTML =
    [`<button class="chip" data-spk="all" aria-pressed="${state.speaker === null}">All speakers</button>`]
      .concat([...Array(speakers).keys()].map(i => {
        const n = state.project.clips.filter(c => c.speaker === i).length;
        return `<button class="chip" data-spk="${i}" aria-pressed="${state.speaker === i}">Speaker ${i + 1} (${n})</button>`;
      }))
      .concat(speakers > 1 ? [`<button class="chip" data-only="${state.speaker ?? 0}">Keep only this speaker</button>`] : [])
      .join("");
}

function renderClips() {
  const clips = visibleClips();
  state.cursor = Math.min(state.cursor, Math.max(clips.length - 1, 0));
  $("#clips").innerHTML = clips.map((c, i) => `
    <div class="clip${i === state.cursor ? " cursor" : ""}" data-id="${c.id}" data-included="${!!c.include}">
      <button class="play" data-play="${c.id}" title="Play">▶</button>
      <span class="spk">${(state.project.speakers || 1) > 1 ? `spk ${c.speaker + 1}` : ""}</span>
      <input class="text${c.edited ? " edited" : ""}" data-text="${c.id}" value="${esc(c.text)}">
      <span class="dur${c.confidence < 0.65 ? " low" : ""}">${(c.end - c.start).toFixed(1)}s</span>
    </div>`).join("");
  renderStats();
}

function playClip(id) {
  state.audio.pause();
  state.audio.src = `/api/projects/${state.project.name}/clips/${id}/audio`;
  state.audio.play().catch(() => toast("could not play clip", true));
}

async function patchClip(id, body) {
  const clip = state.project.clips.find(c => c.id === id);
  Object.assign(clip, body);
  if (body.text !== undefined) clip.edited = true;
  const r = await api(`/api/projects/${state.project.name}/clips/${id}`,
    { method: "PATCH", body: JSON.stringify(body) });
  state.project.stats = r.stats;
  renderStats();
}

function moveCursor(delta) {
  const clips = visibleClips();
  if (!clips.length) return;
  state.cursor = (state.cursor + delta + clips.length) % clips.length;
  renderClips();
  const el = $(`.clip.cursor`);
  el?.scrollIntoView({ block: "nearest" });
}

function wireReview() {
  $("#clips").addEventListener("click", e => {
    const play = e.target.closest("[data-play]");
    if (play) { playClip(play.dataset.play); return; }
    const row = e.target.closest(".clip");
    if (row) {
      state.cursor = visibleClips().findIndex(c => c.id === row.dataset.id);
      $$(".clip").forEach(r => r.classList.toggle("cursor", r === row));
    }
  });

  $("#clips").addEventListener("change", e => {
    const field = e.target.closest("[data-text]");
    if (field) patchClip(field.dataset.text, { text: field.value }).catch(x => toast(x.message, true));
  });

  $("#speaker-filter").addEventListener("click", async e => {
    const only = e.target.closest("[data-only]");
    if (only) {
      await api(`/api/projects/${state.project.name}/clips/bulk`, {
        method: "POST",
        body: JSON.stringify({ action: "include_only_speaker", speaker: Number(only.dataset.only) }),
      });
      state.project = await api(`/api/projects/${state.project.name}`);
      renderClips();
      toast("Kept only that speaker");
      return;
    }
    const chip = e.target.closest("[data-spk]");
    if (!chip) return;
    state.speaker = chip.dataset.spk === "all" ? null : Number(chip.dataset.spk);
    state.cursor = 0;
    renderClips();
  });

  $$("[data-bulk]").forEach(b => b.onclick = async () => {
    await api(`/api/projects/${state.project.name}/clips/bulk`,
      { method: "POST", body: JSON.stringify({ action: b.dataset.bulk }) });
    state.project = await api(`/api/projects/${state.project.name}`);
    renderClips();
  });

  $("#only-included").onchange = e => {
    state.onlyIncluded = e.target.checked;
    state.cursor = 0;
    renderClips();
  };

  // Keyboard-first: this loop runs hundreds of times, so it has to be fast.
  document.addEventListener("keydown", e => {
    if (state.view !== "review") return;
    const editing = e.target.matches("input.text");
    if (editing) {
      if (e.key === "Escape" || e.key === "Enter") e.target.blur();
      return;
    }
    if (e.target.matches("input,select,textarea")) return;
    const clips = visibleClips();
    const cur = clips[state.cursor];
    if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); moveCursor(1); }
    else if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); moveCursor(-1); }
    else if (e.key === " ") { e.preventDefault(); if (cur) playClip(cur.id); }
    else if (e.key.toLowerCase() === "e") {
      e.preventDefault();
      if (cur) patchClip(cur.id, { include: !cur.include }).then(renderClips);
    } else if (e.key === "Enter") {
      e.preventDefault();
      $(`.clip.cursor input.text`)?.focus();
    }
  });
}

/* ---------- training ---------- */

async function loadCheckpoints() {
  if (!state.project) return;
  let list = [];
  try { list = await api(`/api/projects/${state.project.name}/checkpoints`); } catch { /* none yet */ }
  const resumable = list.filter(c => /val_(mel|mos)=/.test(c.name));
  const note = $("#resume-note");
  if (note) {
    note.innerHTML = resumable.length && !$("#restart").checked
      ? `Will <b>continue</b> from epoch ${resumable[0].epoch}.`
      : (resumable.length
          ? `"Start over" is ticked — the epoch counter restarts from the pretrained checkpoint.`
          : `No checkpoint from this voice yet — training will start from the pretrained one.`);
  }
  $("#checkpoints").innerHTML = list.length
    ? list.slice(0, 24).map(c => `
      <div class="card">
        <b>epoch ${c.epoch}</b>
        <span class="meta">${esc(c.name)} · ${c.size_mb} MB</span>
        <button class="primary" data-sample="${esc(c.path)}">Speak this</button>
        <audio controls hidden></audio>
      </div>`).join("")
    : `<p class="hint">No checkpoints yet. They appear a few epochs into training.</p>`;
}

function wireTrain() {
  $("#start-train").onclick = async () => {
    try {
      await api(`/api/projects/${state.project.name}/settings`, {
        method: "PATCH",
        body: JSON.stringify({
          batch_size: Number($("#batch").value),
          from_scratch: $("#scratch").checked,
          restart: $("#restart").checked,
        }),
      });
      const job = await api(`/api/projects/${state.project.name}/train`, { method: "POST" });
      $("#start-train").hidden = true;
      $("#stop-train").hidden = false;
      $("#stop-train").dataset.job = job.id;
      trackJob(job.id, "#train-progress", loadCheckpoints);
      toast("Training started — you can close this tab, it keeps going");
    } catch (e) { toast(e.message, true); }
  };

  $("#stop-train").onclick = async () => {
    const id = $("#stop-train").dataset.job;
    if (id) await api(`/api/jobs/${id}/cancel`, { method: "POST" }).catch(() => {});
    toast("Stopping after the current step…");
  };

  $("#refresh-ckpts").onclick = loadCheckpoints;
  $("#restart").onchange = loadCheckpoints;

  $("#checkpoints").addEventListener("click", async e => {
    const btn = e.target.closest("[data-sample]");
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = "Rendering…";
    try {
      const r = await api(`/api/projects/${state.project.name}/sample`, {
        method: "POST",
        body: JSON.stringify({ checkpoint: btn.dataset.sample, text: $("#sample-text").value }),
      });
      const audio = $("audio", btn.closest(".card"));
      audio.src = r.url;
      audio.hidden = false;
      audio.play().catch(() => {});
    } catch (x) { toast(x.message, true); }
    btn.disabled = false;
    btn.textContent = "Speak this";
  });
}

/* ---------- boot ---------- */

function wireProjects() {
  $("#project-list").addEventListener("click", async e => {
    const open = e.target.closest("[data-open]");
    if (open) return openProject(open.dataset.open);
    const del = e.target.closest("[data-del]");
    if (del && confirm(`Delete project "${del.dataset.del}" and its clips?`)) {
      await api(`/api/projects/${del.dataset.del}`, { method: "DELETE" });
      if (state.project?.name === del.dataset.del) state.project = null;
      gateSteps();
      loadProjects();
    }
  });

  $("#new-project").onsubmit = async e => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target));
    try {
      await api("/api/projects", { method: "POST", body: JSON.stringify(data) });
      e.target.reset();
      await loadProjects();
      openProject(data.name);
    } catch (x) { toast(x.message, true); }
  };
}

$$(".step").forEach(b => b.onclick = () => show(b.dataset.view));
$("#start-import").onclick = startImport;
$("#cancel-import").onclick = async () => {
  if (!state.currentJob) return;
  await api(`/api/jobs/${state.currentJob}/cancel`, { method: "POST" }).catch(() => {});
  toast("Cancelling after the current step…");
};
$("#to-train").onclick = () => show("train");

wireProjects();
wireUpload();
wireReview();
wireTrain();
gateSteps();
loadHealth();
loadProjects();
show("projects");
