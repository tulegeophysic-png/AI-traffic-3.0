let session = null;
let isRunning = false;
let confidenceThreshold = 0.50;
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

let chartInstance = null;
let lastTime = performance.now();
let frameCount = 0;
let currentFps = 0;

setInterval(() => {
    const now = new Date();
    document.getElementById('clock').innerText = now.toTimeString().split(' ')[0];
}, 1000);

document.getElementById('upload-video').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('file-name').innerText = file.name;
        videoElement.src = URL.createObjectURL(file);
        videoElement.load();
        videoElement.onloadedmetadata = function() {
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            if (session) {
                document.getElementById('btn-start').disabled = false;
                setStatus('active', 'AI READY');
            }
        };
    }
});

function initChart() {
    const ctxChart = document.getElementById('trafficChart').getContext('2d');
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
    document.getElementById('conf-val').innerText = val;
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
            document.getElementById('btn-start').disabled = false;
        }
    } catch (e) {
        setStatus('error', 'AI ERROR');
        alert("Không thể tải file yolov10n.onnx. Hãy kiểm tra lại đường dẫn model!");
    }
}
loadModel();

function setStatus(statusClass, text) {
    const badge = document.getElementById('system-status');
    badge.className = `status-badge ${statusClass}`;
    badge.innerText = text;
}

function startAI() {
    if (!videoElement.src || !session) return;
    isRunning = true;
    videoElement.play();
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('btn-capture').disabled = false;
    setStatus('active', 'AI RUNNING');
    requestAnimationFrame(processFrame);
}

function stopAI() {
    isRunning = false;
    videoElement.pause();
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('btn-capture').disabled = true;
    setStatus('stopped', 'AI STOPPED');
}

function resetSystem() {
    stopAI();
    counts = { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
    countedIds.clear();
    tracks = [];
    nextTrackId = 1;
    updateUIStats();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('file-name').innerText = "Chưa chọn file";
    videoElement.src = "";
    document.getElementById('btn-start').disabled = true;
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
        document.getElementById('fps-display').innerText = currentFps.toFixed(1);
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
        updateTrackingAndCounting(detections, canvas.height);
        drawDetections(detections);
        updateUIStats();
    } catch (err) {
        console.error("Inference error:", err);
    }

    requestAnimationFrame(processFrame);
}

function preprocessWithLetterbox(sourceCanvas) {
    const targetSize = 640;
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

    if (dims.length === 3) {
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

function updateTrackingAndCounting(detections, frameHeight) {
    // Đưa vạch đếm lên vị trí 45% chiều cao khung hình để xe kịp được tracking từ xa trước khi qua vạch
    const countingLineY = frameHeight * 0.45;
    let currentTracks = [];
    
    tracks.forEach(track => {
        track.age++;
    });

    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const cx = x + w / 2;
        const cy = y + h / 2;

        let matchedTrack = null;
        let minDst = 120; 

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

            const prevY = matchedTrack.bbox[1] + matchedTrack.bbox[3] / 2;
            matchedTrack.bbox = [x, y, w, h];
            matchedTrack.confidence = det.confidence;
            matchedTrack.age = 0; 

            if (!countedIds.has(matchedTrack.id)) {
                if ((prevY < countingLineY && cy >= countingLineY) || (prevY > countingLineY && cy <= countingLineY)) {
                    countedIds.add(matchedTrack.id);
                    counts[det.className]++;
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
        if (track.age < 15) {
            currentTracks.push(track);
        }
    });

    tracks = currentTracks;
}

function drawDetections(detections) {
    // Đồng bộ vị trí vạch đếm ở 45% chiều cao khung hình
    const countingLineY = canvas.height * 0.45;

    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, countingLineY);
    ctx.lineTo(canvas.width, countingLineY);
    ctx.stroke();

    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 14px Segoe UI';
    ctx.fillText('COUNTING LINE', 15, countingLineY - 8);

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
    document.getElementById('count-car').innerText = counts.car;
    document.getElementById('count-moto').innerText = counts.motorcycle;
    document.getElementById('count-bus').innerText = counts.bus;
    document.getElementById('count-truck').innerText = counts.truck;
    document.getElementById('count-total').innerText = counts.total;

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
    densityBadge.className = `density-badge ${densityClass}`;
    densityBadge.innerText = density;

    if (chartInstance) {
        chartInstance.data.datasets[0].data = [counts.car, counts.motorcycle, counts.bus, counts.truck];
        chartInstance.update();
    }
}

function setCongestion(isCongested) {
    const banner = document.getElementById('congestion-banner');
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