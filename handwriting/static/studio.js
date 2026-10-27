"use strict";
const $ = id => document.getElementById(id);
let job = null, boxes = [], selected = 0, source = null, mask = null, drawing = false;
let dragStart = null, dragEnd = null, pollTimer = null, activeFont = null;
const canvas = $("crop-canvas"), context = canvas.getContext("2d");

function notify(message, error = false) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
  $("notice").className = error ? "alert alert-error" : "alert";
}

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  }
  return response.json();
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load the handwriting image."));
    image.src = url;
  });
}

function draw() {
  if (!source) return;
  const visible = $("show-mask").checked && mask ? mask : source;
  context.drawImage(visible, 0, 0, canvas.width, canvas.height);
  boxes.forEach((box, index) => {
    const x = box.x * canvas.width, y = box.y * canvas.height;
    context.strokeStyle = index === selected ? "#dc652b" : "#285b4f";
    context.lineWidth = index === selected ? 3 : 1.5;
    context.strokeRect(x, y, box.width * canvas.width, box.height * canvas.height);
    context.fillStyle = context.strokeStyle;
    context.fillRect(x, Math.max(0, y - 20), 22, 20);
    context.fillStyle = "white";
    context.font = "15px system-ui";
    context.fillText(box.character, x + 4, Math.max(15, y - 5));
  });
  if (dragStart && dragEnd) {
    context.strokeStyle = "#dc652b";
    context.strokeRect(dragStart.x * canvas.width, dragStart.y * canvas.height,
      (dragEnd.x - dragStart.x) * canvas.width, (dragEnd.y - dragStart.y) * canvas.height);
  }
}

function syncEditor() {
  selected = Math.max(0, Math.min(selected, boxes.length - 1));
  $("glyph-choice").replaceChildren(...boxes.map((box, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `${index + 1}. ${box.character}`;
    return option;
  }));
  $("glyph-choice").value = selected;
  const box = boxes[selected];
  if (box) {
    $("character").value = box.character;
    for (const key of ["x", "y", "width", "height"]) $("crop-" + key).value = (box[key] * 100).toFixed(2);
    $("target-height").value = box.target_height ?? "";
    $("baseline").value = box.baseline ?? "";
  }
  $("mapping-json").value = JSON.stringify(boxes, null, 2);
  $("glyph-count").textContent = `${boxes.length} crops · ${new Set(boxes.map(b => b.character)).size} unique characters`;
  draw();
}

async function poll(id) {
  clearTimeout(pollTimer);
  try {
    job = await api(`/api/fonts/${id}`);
    $("ink").value = job.ink_mode;
    $("upload-button").disabled = ["queued", "processing"].includes(job.status);
    $("generate-button").disabled = !$("confirm").checked || ["queued", "processing"].includes(job.status);
    if (["queued", "processing"].includes(job.status)) {
      notify(job.stage === "analyze" ? "Finding ink and proposing letter crops…" : "Tracing your letters and building the font…");
      pollTimer = setTimeout(() => poll(id), 700);
      return;
    }
    notify(job.status === "failed" ? job.error : job.status === "ready" ? "Your font is ready. Type a note or download it below." : "Check the proposed crops, then confirm the mapping.", job.status === "failed");
    boxes = structuredClone(job.glyphs || []);
    source = await loadImage(job.image_url);
    mask = await loadImage(job.mask_url).catch(() => null);
    canvas.width = Math.min(source.width, 1000);
    canvas.height = Math.round(canvas.width * source.height / source.width);
    $("review").hidden = false;
    $("mapping-errors").textContent = (job.mapping_errors || []).join("\n");
    $("mapping-errors").hidden = !job.mapping_errors?.length;
    syncEditor();
    if (job.status === "ready") await showFont();
  } catch (error) { notify(error.message, true); }
}

async function showFont() {
  const face = new FontFace(job.family, `url("${job.artifacts["font.woff2"]}")`);
  await face.load();
  if (activeFont) document.fonts.delete(activeFont);
  activeFont = face;
  document.fonts.add(face);
  $("preview-text").disabled = false;
  $("preview-text").style.fontFamily = `"${job.family}", cursive`;
  $("preview-description").textContent = "This preview uses your generated web font with its own spacing and kerning.";
  $("downloads").hidden = false;
  for (const [id, filename] of [["ttf-download", "font.ttf"], ["woff-download", "font.woff2"], ["css-download", "font.css"], ["png-download", "sample.png"]]) $(id).href = job.artifacts[filename];
  $("font-details").textContent = `${new Set(boxes.map(b => b.character)).size} characters · ${job.kerning_pairs} spacing pairs · version ${job.revision}`;
  updatePreview();
}

function updatePreview() {
  $("preview-text").style.fontSize = $("font-size").value + "px";
  $("size-value").textContent = $("font-size").value + " px";
  const supported = new Set(job?.supported_characters || "");
  const missing = [...new Set([...$("preview-text").value].filter(c => c !== "\n" && !supported.has(c)))];
  $("missing").textContent = job?.status === "ready" && missing.length ? `Not in this font: ${missing.join(" ")}. The browser uses a fallback for these characters.` : "";
}

$("upload-form").addEventListener("submit", async event => {
  event.preventDefault();
  try {
    const file = $("file").files[0];
    if (!file || file.size > 15 * 1024 * 1024) throw new Error("Choose an image of 15 MiB or less.");
    clearTimeout(pollTimer);
    const form = new FormData();
    form.append("image", file);
    form.append("name", $("name").value);
    form.append("rows", JSON.stringify($("rows").value.split("\n").filter(Boolean)));
    form.append("ink_mode", $("ink").value);
    form.append("detector", $("detector").value);
    $("upload-button").disabled = true;
    $("review").hidden = true;
    $("downloads").hidden = true;
    $("preview-text").disabled = true;
    $("preview-text").style.fontFamily = "";
    $("confirm").checked = false;
    notify("Uploading your page…");
    job = await api("/api/fonts", {method: "POST", body: form});
    history.replaceState(null, "", "#" + job.id);
    await poll(job.id);
  } catch (error) { $("upload-button").disabled = false; notify(error.message, true); }
});

$("generate-button").addEventListener("click", async () => {
  if (!job || !$("confirm").checked) return;
  try {
    $("generate-button").disabled = true;
    const options = body => ({method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    await api(`/api/fonts/${job.id}/glyphs`, options({glyphs: boxes, ink_mode: $("ink").value}));
    $("downloads").hidden = true;
    $("preview-text").disabled = true;
    await api(`/api/fonts/${job.id}/generate`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({confirm_mapping: true})});
    await poll(job.id);
  } catch (error) { $("generate-button").disabled = false; notify(error.message, true); }
});

$("glyph-choice").addEventListener("change", () => { selected = Number($("glyph-choice").value); syncEditor(); });
$("show-mask").addEventListener("change", draw);
$("confirm").addEventListener("change", () => { $("generate-button").disabled = !$("confirm").checked; });
$("font-size").addEventListener("input", updatePreview);
$("preview-text").addEventListener("input", updatePreview);
for (const key of ["x", "y", "width", "height"]) $("crop-" + key).addEventListener("change", () => {
  if (boxes[selected]) { boxes[selected][key] = Number($("crop-" + key).value) / 100; syncEditor(); }
});
$("character").addEventListener("change", () => {
  if (boxes[selected]) { boxes[selected].character = $("character").value; syncEditor(); }
});
for (const [id, key] of [["target-height", "target_height"], ["baseline", "baseline"]]) $(id).addEventListener("change", () => {
  if (boxes[selected]) { boxes[selected][key] = $(id).value === "" ? null : Number($(id).value); syncEditor(); }
});
$("draw-crop").addEventListener("click", () => { drawing = true; notify("Drag a rectangle around all strokes of the selected letter, including dots."); });
$("add-glyph").addEventListener("click", () => {
  boxes.push({character: "?", x: .1, y: .1, width: .08, height: .05});
  selected = boxes.length - 1;
  syncEditor();
});
$("remove-glyph").addEventListener("click", () => { boxes.splice(selected, 1); syncEditor(); });
$("apply-json").addEventListener("click", () => {
  try {
    const parsed = JSON.parse($("mapping-json").value);
    if (!Array.isArray(parsed) || !parsed.length || parsed.length > 160) throw new Error("Provide an array of 1 to 160 crop boxes.");
    boxes = parsed; syncEditor(); notify("Mapping updated. Check the crop overlay before generating.");
  } catch (error) { notify(error.message, true); }
});
function position(event) {
  const rect = canvas.getBoundingClientRect();
  return {x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))};
}
canvas.addEventListener("pointerdown", event => {
  const point = position(event);
  if (drawing && boxes[selected]) { dragStart = point; canvas.setPointerCapture(event.pointerId); }
  else {
    const index = boxes.findIndex(b => point.x >= b.x && point.x <= b.x + b.width && point.y >= b.y && point.y <= b.y + b.height);
    if (index >= 0) { selected = index; syncEditor(); }
  }
});
canvas.addEventListener("pointermove", event => { if (dragStart) { dragEnd = position(event); draw(); } });
canvas.addEventListener("pointerup", event => {
  if (!dragStart) return;
  const end = position(event);
  const width = Math.abs(end.x - dragStart.x), height = Math.abs(end.y - dragStart.y);
  if (width > .001 && height > .001) Object.assign(boxes[selected], {x: Math.min(end.x, dragStart.x), y: Math.min(end.y, dragStart.y), width, height});
  dragStart = dragEnd = null; drawing = false; syncEditor(); notify("Crop updated.");
});
canvas.addEventListener("pointercancel", () => { dragStart = dragEnd = null; drawing = false; draw(); });
api("/api/layouts").then(layout => {
  $("rows").value = layout.default_rows.join("\n");
  $("ai-note").hidden = !layout.ai_detection;
  if (/^#[0-9a-f]{32}$/.test(location.hash)) poll(location.hash.slice(1));
}).catch(error => notify(error.message, true));
window.addEventListener("hashchange", () => {
  if (/^#[0-9a-f]{32}$/.test(location.hash)) poll(location.hash.slice(1));
});
