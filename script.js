let session = null;
let isRunning = false;
let confidenceThreshold = 0.45;
let videoElement = document.getElementById('video-source');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');

let classMap = { 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck' };

let counts = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
let countedIds = new Set();
let tracks = [];
let nextTrackId = 1;

let lowDensityThreshold = 5;
let highDensityThreshold = 15;

let lineConfig = {
    positionRatio: 0.45
};

let isDraggingLine = false;
let chartInstance = null;
let lastTime = performance.now();
let frameCount = 0;
let currentFps = 0;

setInterval(() => {
    const now = new Date();
    const clockEl = document.getElementById('clock');
    if (clockEl) clockEl.innerText = now.toTimeString().split(' ')[0];
}, 1000);

canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    const dirEl = document.getElementById('counting-direction');
    const directionSelect = dirEl ? dirEl.value : 'horizontal';
    const isHoriz = directionSelect.includes('vertical');

    if (isHoriz) {
        const lineY = canvas.height * lineConfig.positionRatio;
        if (Math.abs(mouseY - lineY) < 25) isDraggingLine = true;
    } else {
        const lineX = canvas.width * lineConfig.positionRatio;
        if (Math.abs(mouseX - lineX) < 25) isDraggingLine = true;
    }
});

window.addEventListener('mousemove', (e) => {
    if (!isDraggingLine) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    const dirEl = document.getElementById('counting-direction');
    const directionSelect = dirEl ? dirEl.value : 'horizontal';
    const isHoriz = directionSelect.includes('vertical');

    if (isHoriz) {
        lineConfig.positionRatio = Math.max(0.05, Math.min(0.95, mouseY / canvas.height));
    } else {
        lineConfig.positionRatio = Math.max(0.05, Math.min(0.95, mouseX / canvas.width));
    }
});

window.addEventListener('mouseup', () => {
    isDraggingLine = false;
});

function resetLinePosition() {
    lineConfig.positionRatio = 0.45;
}

const uploadInput = document.getElementById('upload-video');
if (uploadInput) {
    uploadInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            resetSystemDataOnly();
            const fileNameEl = document.getElementById('file-name');
            if (fileNameEl) fileNameEl.innerText = file.name;
            videoElement.src = URL.createObjectURL(file);
            videoElement.load();
            videoElement.onloadedmetadata = function() {
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                drawDetectionsAndLine();
                if (session) {
                    const startBtn = document.getElementById('btn-start');
                    if (startBtn) startBtn.disabled = false;
                    setStatus('active', 'AI READY');
                }
            };
        }
    });
}

function initChart() {
    const chartCanvas = document.getElementById('trafficChart');
    if (!chartCanvas) return;
    const ctxChart = chartCanvas.getContext('2d');
    chartInstance = new Chart(ctxChart, {
        type: 'bar',
        data: {
            labels: ['Car', 'Motorcycle', 'Bus', 'Truck'],
            datasets: [{
                label: 'Số lượng phương tiện',
                data: [0, 0, 0, 0],
                backgroundColor: ['#2563eb', '#16a34a', '#d97706', '#dc2626'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#f8fafc', font: { size: 11 } } },
                x: { grid: { display: false }, ticks: { color: '#f8fafc', font: { size: 11 } } }
            },
            plugins: { legend: { display: false } }
        }
    });
}
initChart();

function updateConfidence(val) {
    confidenceThreshold = parseFloat(val);
    const confValEl = document.getElementById('conf-val');
    if (confValEl) confValEl.innerText = val;
}

async function loadModel() {
    try {
        setStatus('waiting', 'LOADING MODEL...');
        const modelFileName = 'yolov10n.onnx'; 
        const modelPaths = [`./model/${modelFileName}`, `model/${modelFileName}`, `./${modelFileName}`];

        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        for (let path of modelPaths) {
            try {
                session = await ort.InferenceSession.create(path, { executionProviders: ['wasm'] });
                if (session) break;
            } catch (innerErr) {}
        }

        if (!session) throw new Error("Không tìm thấy model.");
        setStatus('active', 'AI READY');
        if (videoElement.src) {
            const startBtn = document.getElementById('btn-start');
            if (startBtn) startBtn.disabled = false;
        }
    } catch (e) {
        setStatus('error', 'AI ERROR');
    }
}
loadModel();

function setStatus(statusClass, text) {
    const badge = document.getElementById('system-status');
    if (badge) {
        badge.className = `status-badge ${statusClass}`;
        badge.innerText = text;
    }
}

function startAI() {
    if (!videoElement.src || !session) return;
    isRunning = true;
    videoElement.play();
    
    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');
    const capBtn = document.getElementById('btn-capture');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    if (capBtn) capBtn.disabled = false;

    setStatus('active', 'AI RUNNING');
    requestAnimationFrame(processFrame);
}

function stopAI() {
    isRunning = false;
    videoElement.pause();

    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');
    const capBtn = document.getElementById('btn-capture');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (capBtn) capBtn.disabled = true;

    setStatus('stopped', 'AI STOPPED');
}

function resetSystemDataOnly() {
    counts = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    countedIds.clear();
    tracks = [];
    nextTrackId = 1;
    updateUIStats();
}

function resetSystem() {
    stopAI();
    resetSystemDataOnly();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const fileNameEl = document.getElementById('file-name');
    if (fileNameEl) fileNameEl.innerText = "Chưa chọn file";
    videoElement.src = "";
    const startBtn = document.getElementById('btn-start');
    if (startBtn) startBtn.disabled = true;
}

async function processFrame() {
    if (!isRunning) return;
    if (videoElement.paused || videoElement.ended) {
        stopAI();
        return;
    }

    const now = performance.now();
    frameCount++;
    if (now - lastTime >= 1000) {
        currentFps = (frameCount * 1000) / (now - lastTime);
        const fpsEl = document.getElementById('fps-display');
        if (fpsEl) fpsEl.innerText = currentFps.toFixed(1);
        frameCount = 0;
        lastTime = now;
    }

    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    try {
        const { tensor, ratio, dw, dh } = preprocessWithLetterbox(canvas);
        const inputName = session.inputNames[0];
        const results = await session.run({ [inputName]: tensor });
        const output = results[session.outputNames[0]];

        const detections = parseYolov10Output(output, canvas.width, canvas.height, ratio, dw, dh);
        updateTrackingAndCounting(detections);
    } catch (err) {
        console.error("Inference error:", err);
    }

    drawDetectionsAndLine();
    updateUIStats();

    requestAnimationFrame(processFrame);
}

function preprocessWithLetterbox(sourceCanvas) {
    const targetSize = 480;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetSize;
    tempCanvas.height = targetSize;
    const tempCtx = tempCanvas.getContext('2d');

    const sw = sourceCanvas.width;
    const sh = sourceCanvas.height;
    const ratio = Math.min(targetSize / sw, targetSize / sh);
    const nw = sw * ratio;
    const nh = sh * ratio;
    const dw = (targetSize - nw) / 2;
    const dh = (targetSize - nh) / 2;

    tempCtx.fillStyle = '#727272';
    tempCtx.fillRect(0, 0, targetSize, targetSize);
    tempCtx.drawImage(sourceCanvas, dw, dh, nw, nh);

    const imgData = tempCtx.getImageData(0, 0, targetSize, targetSize);
    const { data } = imgData;
    const float32Data = new Float32Array(3 * targetSize * targetSize);

    for (let i = 0; i < targetSize * targetSize; i++) {
        float32Data[i] = data[i * 4] / 255.0;                     
        float32Data[targetSize * targetSize + i] = data[i * 4 + 1] / 255.0;      
        float32Data[2 * targetSize * targetSize + i] = data[i * 4 + 2] / 255.0;  
    }
    return {
        tensor: new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]),
        ratio, dw, dh
    };
}

function parseYolov10Output(output, origWidth, origHeight, ratio, dw, dh) {
    const dets = [];
    const data = output.data;
    const dims = output.dims;

    const parseBox = (x1, y1, x2, y2, conf, clsId) => {
        let rx1 = (x1 - dw) / ratio;
        let ry1 = (y1 - dh) / ratio;
        let rx2 = (x2 - dw) / ratio;
        let ry2 = (y2 - dh) / ratio;

        rx1 = Math.max(0, Math.min(origWidth, rx1));
        ry1 = Math.max(0, Math.min(origHeight, ry1));
        rx2 = Math.max(0, Math.min(origWidth, rx2));
        ry2 = Math.max(0, Math.min(origHeight, ry2));

        if (conf >= confidenceThreshold && classMap[clsId]) {
            dets.push({
                bbox: [rx1, ry1, rx2 - rx1, ry2 - ry1],
                className: classMap[clsId],
                confidence: conf
            });
        }
    };

    if (dims && dims.length === 3) {
        if (dims[2] === 6) {
            let numRows = dims[1];
            for (let i = 0; i < numRows; i++) {
                let offset = i * 6;
                parseBox(data[offset], data[offset + 1], data[offset + 2], data[offset + 3], data[offset + 4], Math.round(data[offset + 5]));
            }
        } else if (dims[1] === 6) {
            let numRows = dims[2];
            for (let i = 0; i < numRows; i++) {
                parseBox(data[0 * numRows + i], data[1 * numRows + i], data[2 * numRows + i], data[3 * numRows + i], data[4 * numRows + i], Math.round(data[5 * numRows + i]));
            }
        }
    }
    return dets;
}

function updateTrackingAndCounting(detections) {
    let currentTracks = [];
    
    tracks.forEach(track => {
        track.age++;
    });

    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const cx = x + w / 2;
        const cy = y + h / 2;

        let matchedTrack = null;
        let minDst = 160; 

        tracks.forEach(track => {
            if (track.className === det.className) {
                const tcx = track.bbox[0] + track.bbox[2] / 2;
                const tcy = track.bbox[1] + track.bbox[3] / 2;
                const dst = Math.hypot(cx - tcx, cy - tcy);
                if (dst < minDst) {
                    minDst = dst;
                    matchedTrack = track;
                }
            }
        });

        if (matchedTrack) {
            const index = tracks.indexOf(matchedTrack);
            if (index > -1) {
                tracks.splice(index, 1);
            }

            const prevCx = matchedTrack.bbox[0] + matchedTrack.bbox[2] / 2;
            const prevCy = matchedTrack.bbox[1] + matchedTrack.bbox[3] / 2;

            matchedTrack.bbox = [x, y, w, h];
            matchedTrack.confidence = det.confidence;
            matchedTrack.age = 0; 

            if (!countedIds.has(matchedTrack.id)) {
                const dirEl = document.getElementById('counting-direction');
                const direction = dirEl ? dirEl.value : 'horizontal';
                const lineVal = lineConfig.positionRatio * (direction.includes('vertical') ? canvas.height : canvas.width);

                let hasCrossed = false;
                if (direction.includes('vertical')) {
                    if ((prevCy < lineVal && cy >= lineVal) || (prevCy > lineVal && cy <= lineVal)) {
                        hasCrossed = true;
                    }
                } else {
                    if ((prevCx < lineVal && cx >= lineVal) || (prevCx > lineVal && cx <= lineVal)) {
                        hasCrossed = true;
                    }
                }

                if (hasCrossed) {
                    countedIds.add(matchedTrack.id);
                    if (counts[det.className] !== undefined) {
                        counts[det.className]++;
                    } else {
                        counts.car++;
                    }
                    counts.total++;
                }
            }
            currentTracks.push(matchedTrack);
        } else {
            currentTracks.push({
                id: nextTrackId++,
                bbox: [x, y, w, h],
                className: det.className,
                confidence: det.confidence,
                age: 0
            });
        }
    });

    tracks.forEach(track => {
        if (track.age < 35) {
            currentTracks.push(track);
        }
    });

    tracks = currentTracks;
}

function drawDetectionsAndLine() {
    const dirEl = document.getElementById('counting-direction');
    const direction = dirEl ? dirEl.value : 'horizontal';
    const isVert = direction.includes('vertical');
    const lineCoord = lineConfig.positionRatio * (isVert ? canvas.height : canvas.width);

    ctx.strokeStyle = isDraggingLine ? '#38bdf8' : '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    if (isVert) {
        ctx.moveTo(0, lineCoord);
        ctx.lineTo(canvas.width, lineCoord);
    } else {
        ctx.moveTo(lineCoord, 0);
        ctx.lineTo(lineCoord, canvas.height);
    }
    ctx.stroke();

    ctx.fillStyle = isDraggingLine ? '#38bdf8' : '#ef4444';
    ctx.font = 'bold 13px Segoe UI';
    if (isVert) {
        ctx.fillText(`VẠCH ĐẾM (${direction.toUpperCase()})`, 15, lineCoord - 8);
    } else {
        ctx.fillText(`VẠCH ĐẾM`, lineCoord + 8, 20);
    }

    tracks.forEach(track => {
        const [x, y, w, h] = track.bbox;
        const color = getCategoryColor(track.className);

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = color;
        ctx.fillRect(x, y - 22, 120, 22);

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px Segoe UI';
        ctx.fillText(`${track.className.toUpperCase()} #${track.id} ${(track.confidence * 100).toFixed(0)}%`, x + 4, y - 6);
    });
}

function getCategoryColor(className) {
    switch (className) {
        case 'car': return '#2563eb';
        case 'motorcycle': return '#16a34a';
        case 'bus': return '#d97706';
        case 'truck': return '#dc2626';
        default: return '#38bdf8';
    }
}

function updateUIStats() {
    const setSafeText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.innerText = val;
    };

    setSafeText('count-car', counts.car);
    setSafeText('count-moto', counts.motorcycle);
    setSafeText('count-bus', counts.bus);
    setSafeText('count-truck', counts.truck);
    setSafeText('count-total', counts.total);

    // Hỗ trợ đồng thời các tên ID thẻ thống kê khác nhau trong HTML của bạn
    setSafeText('car-count', counts.car);
    setSafeText('motorcycle-count', counts.motorcycle);
    setSafeText('bus-count', counts.bus);
    setSafeText('truck-count', counts.truck);

    const activeVehicles = tracks.length;
    let density = 'LOW';
    let densityClass = 'low';

    if (activeVehicles >= highDensityThreshold) {
        density = 'HIGH';
        densityClass = 'high';
        setCongestion(true);
    } else if (activeVehicles >= lowDensityThreshold) {
        density = 'MEDIUM';
        densityClass = 'medium';
        setCongestion(false);
    } else {
        setCongestion(false);
    }

    const densityBadge = document.getElementById('density-status');
    if (densityBadge) {
        densityBadge.className = `density-badge ${densityClass}`;
        densityBadge.innerText = density;
    }

    if (chartInstance) {
        chartInstance.data.datasets[0].data = [counts.car, counts.motorcycle, counts.bus, counts.truck];
        chartInstance.update();
    }
}

function setCongestion(isCongested) {
    const banner = document.getElementById('congestion-banner');
    if (!banner) return;
    if (isCongested) {
        banner.className = 'congestion-banner warning';
        banner.innerText = '⚠️ TRAFFIC CONGESTION WARNING';
    } else {
        banner.className = 'congestion-banner normal';
        banner.innerText = '✓ TRAFFIC NORMAL';
    }
}

function captureFrame() {
    const link = document.createElement('a');
    link.download = `ai-traffic-capture-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
}