"use strict";
const $ = id => document.getElementById(id);
const PANELS = ["idle", "working", "ready", "failed"];
const MAX_BYTES = 15 * 1024 * 1024;
const TYPES = ["image/jpeg", "image/png", "image/webp"];
const HEADLINES = {
  idle: ["Turn your handwriting into a", "font"],
  working: ["Making your", "font"],
  ready: ["Your font is", "ready"],
  failed: ["Let's try", "again"],
};
const LEDES = {
  idle: "Photograph a page where you wrote each letter on its own. Your font is ready in under a minute.",
  working: "This usually takes under a minute.",
  ready: "Type below to try it, then download.",
  failed: "",
};
let run = 0, activeFont = null, thumbUrl = null;

function show(state) {
  document.querySelector("main").dataset.state = state;
  for (const id of PANELS) $(id).hidden = id !== state;
  [$("headline-lead").textContent, $("headline-accent").textContent] = HEADLINES[state];
  $("lede").textContent = LEDES[state];
}

function fail(message) {
  $("error").textContent = message;
  show("failed");
}

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "Something went wrong. Please try again.");
  }
  return response.json();
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function markStep(stage) {
  for (const item of document.querySelectorAll(".steps li")) {
    item.classList.toggle("active", item.dataset.step === stage);
    item.classList.toggle("done", stage === "generate" && item.dataset.step === "analyze");
  }
}

// Follow a job until it stops changing. A newer run cancels this one.
async function follow(id, token) {
  for (;;) {
    const job = await api(`/api/fonts/${id}`);
    if (token !== run) return null;
    if (!["queued", "processing"].includes(job.status)) return job;
    markStep(job.stage);
    await sleep(700);
  }
}

async function showFont(job) {
  const face = new FontFace(job.family, `url("${job.artifacts["font.woff2"]}")`);
  await face.load();
  if (activeFont) document.fonts.delete(activeFont);
  activeFont = face;
  document.fonts.add(face);
  $("try").style.fontFamily = `"${job.family}", cursive`;
  $("ttf").href = job.artifacts["font.ttf"];
  $("woff2").href = job.artifacts["font.woff2"];
  const letters = [...job.supported_characters].filter(c => c.trim() && c !== " ").length;
  $("meta").textContent = job.mapping_errors?.length
    ? `${letters} characters. A few marks could not be matched to a letter, so some may be missing. A sharper photo helps.`
    : `${letters} characters`;
  show("ready");
}

async function finish(job) {
  if (job.status === "ready") return showFont(job);
  if (job.status === "failed") return fail(job.error || "The font could not be built.");
  fail("We could not find any letters in that photo. Try a sharper, brighter shot with space between the letters.");
}

async function start(file) {
  const token = ++run;
  if (!TYPES.includes(file.type)) return fail("Choose a JPEG, PNG or WebP photo.");
  if (file.size > MAX_BYTES) return fail("That photo is over 15 MB. Choose a smaller one.");
  if (thumbUrl) URL.revokeObjectURL(thumbUrl);
  thumbUrl = URL.createObjectURL(file);
  $("thumb").src = thumbUrl;
  markStep("analyze");
  show("working");
  try {
    const form = new FormData();
    form.append("image", file);
    form.append("name", "My Handwriting");
    form.append("auto_generate", "true");
    form.append("allow_partial", "true");
    const job = await api("/api/fonts", {method: "POST", body: form});
    history.replaceState(null, "", "#" + job.id);
    const done = await follow(job.id, token);
    if (done) await finish(done);
  } catch (error) {
    if (token === run) fail(error.message);
  }
}

async function resume(id) {
  const token = ++run;
  markStep("analyze");
  show("working");
  try {
    const job = await follow(id, token);
    if (!job) return;
    $("thumb").src = job.image_url;
    await finish(job);
  } catch {
    if (token === run) reset();
  }
}

function reset() {
  run += 1;
  history.replaceState(null, "", location.pathname);
  $("file").value = "";
  show("idle");
}

$("file").addEventListener("change", () => { if ($("file").files[0]) start($("file").files[0]); });
for (const name of ["dragenter", "dragover"]) $("drop").addEventListener(name, event => {
  event.preventDefault();
  $("drop").classList.add("over");
});
for (const name of ["dragleave", "drop"]) $("drop").addEventListener(name, () => $("drop").classList.remove("over"));
$("drop").addEventListener("drop", event => {
  event.preventDefault();
  if (event.dataTransfer.files[0]) start(event.dataTransfer.files[0]);
});
$("again").addEventListener("click", reset);
$("retry").addEventListener("click", reset);

show("idle");
api("/api/layouts").then(layout => { $("ai-note").hidden = !layout.ai_detection; }).catch(() => {});
if (/^#[0-9a-f]{32}$/.test(location.hash)) resume(location.hash.slice(1));
