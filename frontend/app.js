const $ = (id) => document.getElementById(id);
const audioInput = $("audioInput");
const dropzone = $("dropzone");
const selectedFile = $("selectedFile");
const analyzeButton = $("analyzeButton");
const apiUrlInput = $("apiUrl");
let currentFile = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStream = null;
let recording = false;
let previewUrl = null;

function apiBase() {
  return apiUrlInput.value.trim().replace(/\/+$/, "");
}
function setConnection(state, message) {
  const el = document.querySelector(".status");
  el.classList.remove("online", "offline");
  if (state) el.classList.add(state);
  $("connectionText").textContent = message;
}
function prettyBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
function setFile(file) {
  currentFile = file || null;
  $("errorBox").classList.add("hidden");
  if (!currentFile) {
    selectedFile.classList.add("hidden");
    $("audioPreview").classList.add("hidden");
    $("audioPreview").removeAttribute("src");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    analyzeButton.disabled = true;
    return;
  }
  $("fileName").textContent = currentFile.name;
  $("fileMeta").textContent = `${prettyBytes(currentFile.size)} · Ready to analyze`;
  selectedFile.classList.remove("hidden");
  analyzeButton.disabled = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(currentFile);
  $("audioPreview").src = previewUrl;
  $("audioPreview").classList.remove("hidden");
}
audioInput.addEventListener("change", () => setFile(audioInput.files[0]));
$("removeFile").addEventListener("click", () => {
  audioInput.value = "";
  setFile(null);
});
["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, e => {
  e.preventDefault(); dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, e => {
  e.preventDefault(); dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) setFile(file);
});

$("recordButton").addEventListener("click", async () => {
  if (recording && mediaRecorder) {
    mediaRecorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showError("This browser does not support microphone recording. Try a recent version of Chrome or Edge, or upload a WAV file.");
    return;
  }
  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    const options = {};
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) options.mimeType = "audio/webm;codecs=opus";
    mediaRecorder = new MediaRecorder(recordingStream, options);
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const type = mediaRecorder.mimeType || "audio/webm";
      const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
      const blob = new Blob(recordedChunks, { type });
      setFile(new File([blob], `recording-${Date.now()}.${ext}`, { type }));
      recordingStream?.getTracks().forEach(track => track.stop());
      recordingStream = null;
      recording = false;
      $("recordButton").classList.remove("recording");
      $("recordButtonText").textContent = "Record again";
      $("recordHint").textContent = "Recording ready. Analyze it when you're ready.";
    };
    mediaRecorder.start();
    recording = true;
    $("recordButton").classList.add("recording");
    $("recordButtonText").textContent = "Stop recording";
    $("recordHint").textContent = "Recording… speak naturally, then stop.";
    $("errorBox").classList.add("hidden");
  } catch (err) {
    showError(`Could not access the microphone: ${err.message || err}. Check browser permissions.`);
  }
});

function showError(message) {
  $("errorBox").textContent = message;
  $("errorBox").classList.remove("hidden");
}
function findEmotion(data) {
  return data.emotion ?? data.predicted_emotion ?? data.prediction ?? data.label ?? data.class ?? data.result?.emotion ?? data.result?.label ?? null;
}
function findScores(data) {
  const raw = data.scores ?? data.probabilities ?? data.emotions ?? data.confidence_scores ?? data.result?.scores ?? data.result?.probabilities ?? null;
  if (Array.isArray(raw)) {
    return raw.map(item => ({
      label: item.label ?? item.emotion ?? item.class ?? item.name ?? "unknown",
      value: Number(item.score ?? item.probability ?? item.value ?? 0)
    }));
  }
  if (raw && typeof raw === "object") {
    return Object.entries(raw).map(([label, value]) => ({
      label, value: Number(typeof value === "object" ? (value.score ?? value.probability ?? value.value ?? 0) : value)
    }));
  }
  return [];
}
function normalizeScore(value) {
  // Treat values in [0,1] as fractions and larger values as percentages.
  return value >= 0 && value <= 1 ? value * 100 : value;
}
function emotionSymbol(label) {
  const key = String(label).toLowerCase();
  if (key.includes("happy") || key.includes("joy")) return "☀";
  if (key.includes("sad")) return "☂";
  if (key.includes("angry")) return "ϟ";
  if (key.includes("fear")) return "◈";
  if (key.includes("surpris")) return "✦";
  if (key.includes("disgust")) return "≈";
  if (key.includes("calm") || key.includes("neutral")) return "◉";
  return "♫";
}
function renderResults(data) {
  const emotion = findEmotion(data);
  const scores = findScores(data).sort((a, b) => b.value - a.value);
  if (!emotion && scores.length === 0) {
    throw new Error("The API responded, but the UI could not find an emotion or score field in its JSON. Check the response shown below and adjust the response mapping in app.js.");
  }
  const label = emotion ?? scores[0].label;
  $("predictedEmotion").textContent = String(label).replaceAll("_", " ");
  $("emotionEmoji").textContent = emotionSymbol(label);
  const modelVersion = data.model_version ?? data.modelVersion ?? null;
  $("predictionSubtext").textContent = modelVersion ? `Model version ${modelVersion}` : "Prediction returned by your local model";
  $("scoreList").innerHTML = "";
  if (scores.length) {
    const max = Math.max(...scores.map(s => normalizeScore(s.value)), 1);
    scores.forEach(item => {
      const pct = Math.max(0, Math.min(100, normalizeScore(item.value)));
      const row = document.createElement("div");
      row.className = "score-row";
      const name = document.createElement("span");
      name.className = "score-name";
      name.textContent = String(item.label).replaceAll("_", " ");
      const track = document.createElement("div");
      track.className = "score-track";
      const fill = document.createElement("div");
      fill.className = "score-fill";
      fill.style.width = `${pct}%`;
      track.appendChild(fill);
      const value = document.createElement("span");
      value.className = "score-value";
      value.textContent = `${pct.toFixed(1)}%`;
      row.append(name, track, value);
      $("scoreList").appendChild(row);
    });
  } else {
    const row = document.createElement("p");
    row.className = "muted";
    row.textContent = "Your API returned a prediction without per-emotion scores.";
    $("scoreList").appendChild(row);
  }
  $("emptyState").classList.add("hidden");
  $("results").classList.remove("hidden");
  $("resultBadge").textContent = "COMPLETE";
  $("resultBadge").classList.add("complete");
}
$("analyzeButton").addEventListener("click", async () => {
  if (!currentFile) return;
  $("errorBox").classList.add("hidden");
  $("analyzeButton").disabled = true;
  $("analyzeButtonText").textContent = "Analyzing…";
  $("resultBadge").textContent = "RUNNING";
  $("resultBadge").classList.remove("complete");
  try {
    const form = new FormData();
    form.append("file", currentFile, currentFile.name);
    // This matches the common FastAPI UploadFile field name "file".
    // If your API uses another field name, change "file" below to match api.py.
    const response = await fetch(`${apiBase()}/predict`, { method: "POST", body: form });
    const bodyText = await response.text();
    let body;
    try { body = JSON.parse(bodyText); } catch { body = { detail: bodyText }; }
    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}: ${body.detail ?? body.message ?? bodyText}. Confirm the endpoint path and upload field in src/api.py.`);
    }
    renderResults(body);
  } catch (err) {
    $("resultBadge").textContent = "ERROR";
    showError(`${err.message || err}\n\nCheck that your FastAPI server is running, the API URL is correct, and CORS allows this UI origin. Open ${apiBase()}/docs to inspect your API routes.`);
  } finally {
    analyzeButton.disabled = !currentFile;
    $("analyzeButtonText").textContent = "Analyze emotion";
  }
});
$("checkApi").addEventListener("click", async () => {
  try {
    const response = await fetch(`${apiBase()}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json().catch(() => ({}));
    setConnection("online", `API connected${data.status ? ` · ${data.status}` : ""}`);
  } catch (err) {
    setConnection("offline", "API unavailable — check URL/server");
    showError(`Could not reach ${apiBase()}/health. If your API uses a different health endpoint, open ${apiBase()}/docs to check its routes. Details: ${err.message || err}`);
  }
});
