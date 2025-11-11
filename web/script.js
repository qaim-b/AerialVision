// AerialVision Dashboard JavaScript

const API_BASE = 'http://localhost:8000';
let selectedFile = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const detectBtn = document.getElementById('detectBtn');
const resultsSection = document.getElementById('results');

// Upload Area Handling
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        selectedFile = files[0];
        handleFileSelect(selectedFile);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        handleFileSelect(selectedFile);
    }
});

function handleFileSelect(file) {
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    placeholder.innerHTML = `
        <p>✓ ${file.name}</p>
        <span>${formatFileSize(file.size)} • Click to change</span>
    `;
    detectBtn.disabled = false;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Detection
detectBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    detectBtn.disabled = true;
    detectBtn.textContent = 'Processing...';
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('conf_threshold', '0.5');
        
        const response = await fetch(`${API_BASE}/api/v1/detect`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Detection failed');
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        detectBtn.disabled = false;
        detectBtn.textContent = 'Detect Vehicles';
    }
});

function displayResults(data) {
    // Update metrics
    document.getElementById('totalVehicles').textContent = data.total_vehicles || data.count || 0;
    document.getElementById('inferenceTime').textContent = 
        (data.inference_time * 1000).toFixed(0) + 'ms';
    
    // Count by class
    const classCounts = {};
    data.detections.forEach(det => {
        const className = det.class_name || det.class;
        classCounts[className] = (classCounts[className] || 0) + 1;
    });
    
    document.getElementById('carCount').textContent = 
        (classCounts['car'] || 0) + (classCounts['van'] || 0);
    document.getElementById('truckCount').textContent = 
        (classCounts['truck'] || 0) + (classCounts['bus'] || 0);
    
    // Display annotated image
    if (data.annotated_image_url) {
        document.getElementById('resultImg').src = data.annotated_image_url;
    }
    
    // Display detection list
    const detectionsList = document.getElementById('detectionsList');
    detectionsList.innerHTML = data.detections.slice(0, 10).map(det => `
        <div class="detection-item">
            <span class="detection-class">${det.class_name || det.class}</span>
            <span class="detection-conf">${(det.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
    
    if (data.detections.length > 10) {
        detectionsList.innerHTML += `
            <div class="detection-item">
                <span class="detection-class">+ ${data.detections.length - 10} more...</span>
            </div>
        `;
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// WebSocket Streaming
const startStreamBtn = document.getElementById('startStreamBtn');
const stopStreamBtn = document.getElementById('stopStreamBtn');
const streamStats = document.getElementById('streamStats');
let ws = null;
let stream = null;

startStreamBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const videoStream = document.getElementById('videoStream');
        videoStream.srcObject = stream;
        
        // Connect WebSocket
        ws = new WebSocket(`ws://localhost:8000/ws/stream`);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            startStreamBtn.disabled = true;
            stopStreamBtn.disabled = false;
            streamStats.style.display = 'flex';
            
            // Start sending frames
            sendFrame();
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            document.getElementById('streamVehicles').textContent = data.total_vehicles || 0;
            document.getElementById('fps').textContent = data.fps || 0;
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            stopStream();
        };
        
    } catch (error) {
        alert('Error accessing camera: ' + error.message);
    }
});

stopStreamBtn.addEventListener('click', stopStream);

function stopStream() {
    if (ws) {
        ws.close();
        ws = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    startStreamBtn.disabled = false;
    stopStreamBtn.disabled = true;
    streamStats.style.display = 'none';
}

async function sendFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    
    const video = document.getElementById('videoStream');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const base64Image = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    
    ws.send(JSON.stringify({
        frame: base64Image,
        conf_threshold: 0.5
    }));
    
    // Send next frame
    setTimeout(sendFrame, 100); // 10 FPS
}

// Smooth Scroll for Navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        const targetElement = document.querySelector(targetId);
        
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
            
            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        }
    });
});
