const API_BASE_KEY = "aerialvision-api-base";
const initialOrigin = window.location.origin.startsWith("http") ? window.location.origin : "http://localhost:8000";

let API_BASE = initialOrigin;
let selectedFile = null;
let ws = null;
let stream = null;
let latestResult = null;

const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const detectBtn = document.getElementById("detectBtn");
const trackBtn = document.getElementById("trackBtn");
const downloadReportBtn = document.getElementById("downloadReportBtn");
const statusText = document.getElementById("statusText");
const apiBaseInput = document.getElementById("apiBaseInput");
const applyApiBaseBtn = document.getElementById("applyApiBaseBtn");
const confSlider = document.getElementById("confSlider");
const iouSlider = document.getElementById("iouSlider");
const confValue = document.getElementById("confValue");
const iouValue = document.getElementById("iouValue");
const fileMeta = document.getElementById("fileMeta");

const streamVideo = document.getElementById("videoStream");
const startStreamBtn = document.getElementById("startStreamBtn");
const stopStreamBtn = document.getElementById("stopStreamBtn");
const streamStats = document.getElementById("streamStats");

function setStatus(message, isError = false) {
    statusText.textContent = message;
    statusText.style.color = isError ? "#ff9aae" : "#9db2d4";
}

function loadApiBase() {
    const stored = localStorage.getItem(API_BASE_KEY);
    API_BASE = stored || initialOrigin;
    apiBaseInput.value = API_BASE;
}

function applyApiBase() {
    API_BASE = apiBaseInput.value.trim().replace(/\/$/, "") || "http://localhost:8000";
    apiBaseInput.value = API_BASE;
    localStorage.setItem(API_BASE_KEY, API_BASE);
    setStatus(`API base set to ${API_BASE}`);
    fetchHistory();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getThresholds() {
    return {
        conf: Number(confSlider.value),
        iou: Number(iouSlider.value),
    };
}

function updateThresholdLabels() {
    confValue.textContent = Number(confSlider.value).toFixed(2);
    iouValue.textContent = Number(iouSlider.value).toFixed(2);
}

function toWsUrl(httpBase) {
    if (httpBase.startsWith("https://")) return httpBase.replace("https://", "wss://");
    if (httpBase.startsWith("http://")) return httpBase.replace("http://", "ws://");
    return httpBase;
}

function setKpis(totalVehicles, density, congestion) {
    document.getElementById("kpiVehicles").textContent = String(totalVehicles || 0);
    document.getElementById("kpiDensity").textContent = Number(density || 0).toFixed(2);
    document.getElementById("kpiCongestion").textContent = String(congestion || "low").toUpperCase();
}

function renderHotspot(hotspot) {
    const values = hotspot || {};
    document.getElementById("hsTopLeft").textContent = values.top_left || 0;
    document.getElementById("hsTopRight").textContent = values.top_right || 0;
    document.getElementById("hsBottomLeft").textContent = values.bottom_left || 0;
    document.getElementById("hsBottomRight").textContent = values.bottom_right || 0;
}

function renderDetections(detections) {
    const container = document.getElementById("detectionsList");
    if (!detections || detections.length === 0) {
        container.innerHTML = `<div class="detection-item"><span>No detections</span><span>-</span></div>`;
        return;
    }

    container.innerHTML = detections.slice(0, 12).map((det) => `
        <div class="detection-item">
            <span>${det.class_name || det.class || "object"}</span>
            <span>${(Number(det.confidence || 0) * 100).toFixed(1)}%</span>
        </div>
    `).join("");
}

function renderResults(data) {
    latestResult = data;
    const detections = data.detections || [];
    const byClass = data.by_class || {};
    const insights = data.insights || {};

    document.getElementById("totalVehicles").textContent = data.total_vehicles || data.count || 0;
    document.getElementById("carCount").textContent = (byClass.car || 0) + (byClass.van || 0);
    document.getElementById("truckCount").textContent = (byClass.truck || 0) + (byClass.bus || 0);
    document.getElementById("inferenceTime").textContent = `${Math.round(Number(data.inference_time || 0) * 1000)} ms`;
    document.getElementById("avgConf").textContent = `${Number(insights.avg_confidence_pct || 0).toFixed(1)}%`;
    document.getElementById("heavyRatio").textContent = `${Number(insights.heavy_vehicle_ratio_pct || 0).toFixed(1)}%`;

    setKpis(data.total_vehicles || 0, insights.density_per_mpx || 0, insights.congestion_level || "low");
    renderHotspot(insights.hotspot_quadrants);
    renderDetections(detections);

    if (data.annotated_image_url) {
        document.getElementById("resultImg").src = data.annotated_image_url;
    }

    downloadReportBtn.disabled = false;
}

async function runDetect(endpoint) {
    if (!selectedFile) return;
    const { conf, iou } = getThresholds();
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("conf_threshold", String(conf));
    formData.append("iou_threshold", String(iou));

    setStatus(`Running ${endpoint === "detect" ? "image detection" : "video tracking"}...`);
    detectBtn.disabled = true;
    trackBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/api/v1/${endpoint}`, {
            method: "POST",
            body: formData,
        });
        if (!response.ok) {
            const err = await response.text();
            throw new Error(`Request failed: ${response.status} ${err}`);
        }

        const data = await response.json();
        renderResults(data);
        await fetchHistory();
        setStatus(`Inference complete (${endpoint}).`);
    } catch (error) {
        setStatus(error.message, true);
    } finally {
        detectBtn.disabled = false;
        trackBtn.disabled = false;
    }
}

async function fetchHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/v1/history?limit=20`);
        if (!response.ok) return;
        const payload = await response.json();
        drawHistory(payload.events || []);
    } catch (_error) {
        // History is best-effort visualization.
    }
}

function drawHistory(events) {
    const canvas = document.getElementById("historyCanvas");
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#08101f";
    ctx.fillRect(0, 0, width, height);

    if (!events.length) {
        ctx.fillStyle = "#9db2d4";
        ctx.font = "16px Segoe UI";
        ctx.fillText("No inference history yet", 20, 40);
        return;
    }

    const values = events.map((e) => Number(e.total_vehicles || 0));
    const maxValue = Math.max(...values, 1);
    const padding = 24;
    const graphW = width - padding * 2;
    const graphH = height - padding * 2;
    const stepX = graphW / Math.max(values.length - 1, 1);

    ctx.strokeStyle = "#264468";
    ctx.lineWidth = 1;
    for (let i = 0; i < 4; i += 1) {
        const y = padding + (graphH / 3) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
    }

    ctx.strokeStyle = "#35e5ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((value, index) => {
        const x = padding + index * stepX;
        const y = padding + graphH - (value / maxValue) * graphH;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "#66ffb8";
    values.forEach((value, index) => {
        const x = padding + index * stepX;
        const y = padding + graphH - (value / maxValue) * graphH;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    });
}

function downloadReport() {
    if (!latestResult) return;
    const payload = {
        generated_at: new Date().toISOString(),
        api_base: API_BASE,
        thresholds: getThresholds(),
        result: latestResult,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aerialvision-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

async function startStream() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamVideo.srcObject = stream;

        const wsBase = toWsUrl(API_BASE);
        ws = new WebSocket(`${wsBase}/ws/stream`);

        ws.onopen = () => {
            streamStats.hidden = false;
            startStreamBtn.disabled = true;
            stopStreamBtn.disabled = false;
            setStatus("Live stream connected.");
            sendFrameLoop();
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            document.getElementById("fps").textContent = Number(data.fps || 0).toFixed(1);
            document.getElementById("streamVehicles").textContent = data.total_vehicles || 0;
            document.getElementById("streamCongestion").textContent = String(data.insights?.congestion_level || "low").toUpperCase();
            setKpis(data.total_vehicles || 0, data.insights?.density_per_mpx || 0, data.insights?.congestion_level || "low");
            renderHotspot(data.insights?.hotspot_quadrants || {});
        };

        ws.onerror = () => {
            setStatus("WebSocket connection error.", true);
            stopStream();
        };
    } catch (error) {
        setStatus(`Camera error: ${error.message}`, true);
    }
}

function stopStream() {
    if (ws) {
        ws.close();
        ws = null;
    }
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }
    streamVideo.srcObject = null;
    startStreamBtn.disabled = false;
    stopStreamBtn.disabled = true;
    streamStats.hidden = true;
}

function sendFrameLoop() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !streamVideo.videoWidth) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            setTimeout(sendFrameLoop, 120);
        }
        return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = streamVideo.videoWidth;
    canvas.height = streamVideo.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(streamVideo, 0, 0);
    const frame = canvas.toDataURL("image/jpeg", 0.76).split(",")[1];
    const { conf, iou } = getThresholds();

    ws.send(JSON.stringify({
        frame,
        conf_threshold: conf,
        iou_threshold: iou,
    }));

    setTimeout(sendFrameLoop, 120);
}

function handleFileSelect(file) {
    selectedFile = file;
    fileMeta.textContent = `${file.name} (${formatFileSize(file.size)})`;
    detectBtn.disabled = false;
    trackBtn.disabled = false;
}

function bindEvents() {
    uploadArea.addEventListener("click", () => fileInput.click());
    uploadArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadArea.classList.add("dragover");
    });
    uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
    uploadArea.addEventListener("drop", (event) => {
        event.preventDefault();
        uploadArea.classList.remove("dragover");
        if (event.dataTransfer.files.length) {
            handleFileSelect(event.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener("change", (event) => {
        if (event.target.files.length) {
            handleFileSelect(event.target.files[0]);
        }
    });

    confSlider.addEventListener("input", updateThresholdLabels);
    iouSlider.addEventListener("input", updateThresholdLabels);

    detectBtn.addEventListener("click", () => runDetect("detect"));
    trackBtn.addEventListener("click", () => runDetect("track"));
    downloadReportBtn.addEventListener("click", downloadReport);
    startStreamBtn.addEventListener("click", startStream);
    stopStreamBtn.addEventListener("click", stopStream);

    applyApiBaseBtn.addEventListener("click", applyApiBase);
}

function init() {
    loadApiBase();
    updateThresholdLabels();
    bindEvents();
    fetchHistory();
}

init();
