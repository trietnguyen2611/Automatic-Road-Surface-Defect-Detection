/**
 * Flexi-YOLO & YOLOv8n Baseline - Road Surface Defect Detection Client Script
 * Reference: "Flexi-YOLO: A lightweight method for road crack detection in complex environments"
 * PLOS ONE (2025) - https://doi.org/10.1371/journal.pone.0325993
 */

document.addEventListener('DOMContentLoaded', () => {
    // =========================================================
    // 1. STATE MANAGEMENT
    // =========================================================
    const state = {
        activeModel: 'flexi_yolo', // 'flexi_yolo' | 'baseline'
        activeModelName: 'Flexi-YOLO',
        conf: 0.25,
        iou: 0.45,
        classes: {
            'Pothole': true,
            'Crack': true,
            'Manhole': true
        },
        showLabels: true,
        showConf: true,
        isStreaming: false,
        ws: null,
        mediaStream: null,
        lastDetectionResult: null,
        sampleOffset: 0,
        sampleLimit: 24,
        sampleTotal: 0,
        isLoadingSamples: false,
        fpsCounter: 0,
        currentFps: 0
    };

    // =========================================================
    // 2. DOM ELEMENTS RESOLVER
    // =========================================================
    const getEl = (id) => document.getElementById(id);
    const getAll = (sel) => document.querySelectorAll(sel);

    const elements = {
        // Navigation Tabs
        tabBtns: getAll('.category-tab, .tab-btn'),
        tabPanes: getAll('.tab-pane, .tab-content'),

        // Model Selector Buttons
        btnSelectFlexi: getEl('btnSelectFlexi'),
        btnSelectBaseline: getEl('btnSelectBaseline'),
        modelSegmentBtns: getAll('.model-segment-btn'),
        systemStatusBadge: getEl('systemStatusBadge'),
        systemStatusText: getEl('systemStatusText'),
        hudActiveModelName: getEl('hudActiveModelName'),
        hudGflopsBadge: getEl('hudGflopsBadge'),
        webcamModelTag: getEl('webcamModelTag'),
        hudEngineTag: getEl('hudEngineTag'),
        resultModelTag: getEl('resultModelTag'),
        popoverModelBadge: getEl('popoverModelBadge'),

        // Paper Benchmark Modal
        btnOpenPaperModal: getEl('btnOpenPaperModal'),
        modalPaperBenchmark: getEl('modalPaperBenchmark'),
        btnClosePaperModal: getEl('btnClosePaperModal'),
        btnModalCloseAction: getEl('btnModalCloseAction'),
        btnModalSwitchToFlexi: getEl('btnModalSwitchToFlexi'),

        // Settings Popover & Controls
        btnSettingsToggle: getEl('btnSettingsToggle'),
        settingsMenu: getEl('settingsMenu'),
        btnCloseSettings: getEl('btnCloseSettings'),
        sliderConf: getEl('sliderConf'),
        sliderIou: getEl('sliderIou'),
        valConf: getEl('valConf'),
        valIou: getEl('valIou'),
        chkPothole: getEl('chkPothole'),
        chkCrack: getEl('chkCrack'),
        chkManhole: getEl('chkManhole'),
        btnToggleAllClasses: getEl('btnToggleAllClasses'),
        settingsActiveIndicator: getEl('settingsActiveIndicator'),
        chkShowLabels: getEl('chkShowLabels'),
        chkShowConf: getEl('chkShowConf'),
        btnReloadModel: getEl('btnReloadModel'),

        // HUD Stats
        hudTotalDefects: getEl('hudTotalDefects'),
        hudPotholeCount: getEl('hudPotholeCount'),
        hudCrackCount: getEl('hudCrackCount'),
        hudManholeCount: getEl('hudManholeCount'),
        hudFps: getEl('hudFps'),
        hudLatency: getEl('hudLatency'),

        // Webcam Stream
        video: getEl('webcamVideo'),
        canvas: getEl('overlayCanvas') || getEl('streamCanvas'),
        btnStartWebcam: getEl('btnStartWebcam'),
        btnStopWebcam: getEl('btnStopWebcam'),
        webcamPlaceholder: getEl('webcamPlaceholder'),
        hudResTag: getEl('hudResTag'),
        streamDot: getEl('streamDot'),

        // Upload Tab
        dropzoneArea: getEl('dropzoneArea'),
        fileInput: getEl('fileInput'),
        btnBrowseFile: getEl('btnBrowseFile'),
        uploadResultCard: getEl('uploadResultCard'),
        resultImage: getEl('resultImage'),
        resultFilename: getEl('resultFilename'),
        defectTableBody: getEl('defectTableBody'),
        noDefectMsg: getEl('noDefectMsg'),
        btnClearUpload: getEl('btnClearUpload'),

        // Samples Gallery Tab
        samplesGrid: getEl('samplesGrid'),
        galleryCountBadge: getEl('galleryCountBadge'),
        btnLoadMoreSamples: getEl('btnLoadMoreSamples'),
        remainingCountBadge: getEl('remainingCountBadge'),
        galleryLoadingSpinner: getEl('galleryLoadingSpinner'),
        galleryEndMsg: getEl('galleryEndMsg')
    };

    const ctx = elements.canvas ? elements.canvas.getContext('2d') : null;
    const offscreenCanvas = document.createElement('canvas');
    const offscreenCtx = offscreenCanvas.getContext('2d');

    // =========================================================
    // 3. TOAST NOTIFICATION UTILITY
    // =========================================================
    function showToast(message, type = 'info') {
        const existingToast = document.querySelector('.app-toast');
        if (existingToast) existingToast.remove();

        const toast = document.createElement('div');
        toast.className = `app-toast toast-${type}`;
        toast.innerHTML = `<span>${message}</span>`;
        document.body.appendChild(toast);

        setTimeout(() => toast.classList.add('toast-show'), 10);
        setTimeout(() => {
            toast.classList.remove('toast-show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // =========================================================
    // 4. MODEL SELECTION & SWITCHING
    // =========================================================
    async function switchModel(mode) {
        if (state.activeModel === mode) return;

        try {
            showToast(`🔄 Đang chuyển đổi sang mô hình: ${mode === 'flexi_yolo' ? 'Flexi-YOLO (Paper)' : 'YOLOv8n (Baseline)'}...`);
            const res = await fetch('/api/model/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });

            if (!res.ok) throw new Error('Lỗi chuyển đổi mô hình từ server');
            const data = await res.json();
            const info = data.info;

            state.activeModel = mode;
            state.activeModelName = info.name;

            // Cập nhật trạng thái các nút chọn mô hình
            elements.modelSegmentBtns.forEach(btn => {
                const isTarget = btn.getAttribute('data-model') === mode;
                btn.classList.toggle('active', isTarget);
            });

            // Cập nhật text & badge trên giao diện
            if (elements.systemStatusText) elements.systemStatusText.textContent = info.name;
            if (elements.hudActiveModelName) elements.hudActiveModelName.textContent = info.name;
            if (elements.hudGflopsBadge) elements.hudGflopsBadge.textContent = `${info.gflops} GFLOPS`;
            if (elements.webcamModelTag) elements.webcamModelTag.textContent = `Mô hình: ${info.name}`;
            if (elements.hudEngineTag) elements.hudEngineTag.textContent = info.name;
            if (elements.resultModelTag) elements.resultModelTag.textContent = info.name;
            if (elements.popoverModelBadge) elements.popoverModelBadge.textContent = `${mode === 'flexi_yolo' ? 'Flexi-YOLO' : 'YOLOv8n'} Config`;

            showToast(`✅ Đã kích hoạt thành công: ${info.name}!`, 'success');
        } catch (err) {
            console.error('Lỗi chuyển đổi mô hình:', err);
            showToast(`⚠️ Không thể chuyển đổi mô hình: ${err.message}`, 'error');
        }
    }

    // Bind sự kiện chọn mô hình
    if (elements.btnSelectFlexi) {
        elements.btnSelectFlexi.addEventListener('click', () => switchModel('flexi_yolo'));
    }
    if (elements.btnSelectBaseline) {
        elements.btnSelectBaseline.addEventListener('click', () => switchModel('baseline'));
    }

    // =========================================================
    // 5. PAPER BENCHMARK MODAL CONTROLS
    // =========================================================
    function openPaperModal() {
        if (elements.modalPaperBenchmark) {
            elements.modalPaperBenchmark.classList.add('is-open');
            elements.modalPaperBenchmark.setAttribute('aria-hidden', 'false');
        }
    }

    function closePaperModal() {
        if (elements.modalPaperBenchmark) {
            elements.modalPaperBenchmark.classList.remove('is-open');
            elements.modalPaperBenchmark.setAttribute('aria-hidden', 'true');
        }
    }

    if (elements.btnOpenPaperModal) {
        elements.btnOpenPaperModal.addEventListener('click', openPaperModal);
    }
    if (elements.systemStatusBadge) {
        elements.systemStatusBadge.addEventListener('click', openPaperModal);
    }
    if (elements.btnClosePaperModal) {
        elements.btnClosePaperModal.addEventListener('click', closePaperModal);
    }
    if (elements.btnModalCloseAction) {
        elements.btnModalCloseAction.addEventListener('click', closePaperModal);
    }
    if (elements.modalPaperBenchmark) {
        elements.modalPaperBenchmark.addEventListener('click', (e) => {
            if (e.target === elements.modalPaperBenchmark) closePaperModal();
        });
    }
    if (elements.btnModalSwitchToFlexi) {
        elements.btnModalSwitchToFlexi.addEventListener('click', () => {
            switchModel('flexi_yolo');
            closePaperModal();
        });
    }

    // Phím tắt ESC đóng Modal / Popover
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closePaperModal();
            closeSettingsPopover();
        }
    });

    // =========================================================
    // 6. NAVIGATION TABS
    // =========================================================
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            elements.tabBtns.forEach(b => b.classList.remove('active'));
            elements.tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPane = document.getElementById(targetTab);
            if (targetPane) targetPane.classList.add('active');

            if (targetTab === 'tab-samples' && state.sampleTotal === 0) {
                loadSamples(true);
            }
        });
    });

    // =========================================================
    // 7. SETTINGS POPOVER
    // =========================================================
    function toggleSettingsPopover() {
        if (!elements.settingsMenu) return;
        const isVisible = elements.settingsMenu.classList.contains('is-visible');
        if (isVisible) closeSettingsPopover();
        else openSettingsPopover();
    }

    function openSettingsPopover() {
        if (!elements.settingsMenu) return;
        elements.settingsMenu.classList.add('is-visible');
        if (elements.btnSettingsToggle) elements.btnSettingsToggle.setAttribute('aria-expanded', 'true');
    }

    function closeSettingsPopover() {
        if (!elements.settingsMenu) return;
        elements.settingsMenu.classList.remove('is-visible');
        if (elements.btnSettingsToggle) elements.btnSettingsToggle.setAttribute('aria-expanded', 'false');
    }

    if (elements.btnSettingsToggle) {
        elements.btnSettingsToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleSettingsPopover();
        });
    }
    if (elements.btnCloseSettings) {
        elements.btnCloseSettings.addEventListener('click', closeSettingsPopover);
    }
    document.addEventListener('click', (e) => {
        if (elements.settingsMenu && !elements.settingsMenu.contains(e.target) && e.target !== elements.btnSettingsToggle) {
            closeSettingsPopover();
        }
    });

    // Sliders
    if (elements.sliderConf && elements.valConf) {
        elements.sliderConf.addEventListener('input', (e) => {
            state.conf = parseFloat(e.target.value);
            elements.valConf.textContent = `${Math.round(state.conf * 100)}%`;
        });
    }
    if (elements.sliderIou && elements.valIou) {
        elements.sliderIou.addEventListener('input', (e) => {
            state.iou = parseFloat(e.target.value);
            elements.valIou.textContent = `${Math.round(state.iou * 100)}%`;
        });
    }

    // Class filters
    const classCheckboxes = [
        { el: elements.chkPothole, key: 'Pothole' },
        { el: elements.chkCrack, key: 'Crack' },
        { el: elements.chkManhole, key: 'Manhole' }
    ];

    function updateClassIndicator() {
        const count = Object.values(state.classes).filter(Boolean).length;
        if (elements.settingsActiveIndicator) {
            elements.settingsActiveIndicator.textContent = `${count}/3 lớp`;
        }
    }

    classCheckboxes.forEach(({ el, key }) => {
        if (!el) return;
        el.addEventListener('change', () => {
            state.classes[key] = el.checked;
            const tile = el.closest('.defect-tile');
            if (tile) tile.classList.toggle('is-active', el.checked);
            updateClassIndicator();
        });
    });

    if (elements.btnToggleAllClasses) {
        elements.btnToggleAllClasses.addEventListener('click', () => {
            const allActive = Object.values(state.classes).every(Boolean);
            const newState = !allActive;
            classCheckboxes.forEach(({ el, key }) => {
                if (el) {
                    el.checked = newState;
                    state.classes[key] = newState;
                    const tile = el.closest('.defect-tile');
                    if (tile) tile.classList.toggle('is-active', newState);
                }
            });
            elements.btnToggleAllClasses.textContent = newState ? 'Bỏ chọn tất cả' : 'Chọn tất cả';
            updateClassIndicator();
        });
    }

    // Tải lại mô hình
    if (elements.btnReloadModel) {
        elements.btnReloadModel.addEventListener('click', async () => {
            try {
                showToast('🔄 Đang nạp lại trọng số mô hình...');
                const res = await fetch('/api/model/reload', { method: 'POST' });
                const data = await res.json();
                showToast(`✅ Đã tải lại mô hình: ${data.model_name || data.model_file}!`, 'success');
            } catch (err) {
                showToast(`⚠️ Không thể tải lại: ${err.message}`, 'error');
            }
        });
    }

    // =========================================================
    // 8. WEBCAM STREAMING (WEBSOCKET + CANVAS)
    // =========================================================
    if (elements.btnStartWebcam) {
        elements.btnStartWebcam.addEventListener('click', startWebcam);
    }
    if (elements.btnStopWebcam) {
        elements.btnStopWebcam.addEventListener('click', stopWebcam);
    }

    async function startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 360 },
                    facingMode: 'environment'
                }
            });

            state.mediaStream = stream;
            elements.video.srcObject = stream;
            if (elements.webcamPlaceholder) elements.webcamPlaceholder.style.display = 'none';
            if (elements.btnStartWebcam) elements.btnStartWebcam.disabled = true;
            if (elements.btnStopWebcam) elements.btnStopWebcam.disabled = false;
            if (elements.streamDot) elements.streamDot.classList.add('dot-live');
            state.isStreaming = true;

            elements.video.onloadedmetadata = () => {
                if (elements.canvas) {
                    elements.canvas.width = elements.video.videoWidth || 640;
                    elements.canvas.height = elements.video.videoHeight || 360;
                }
                offscreenCanvas.width = 320;
                offscreenCanvas.height = 180;
                connectWebSocket();
            };
        } catch (err) {
            console.error('Lỗi webcam:', err);
            alert(`⚠️ Không thể truy cập Camera: ${err.message}. Vui lòng cho phép quyền truy cập webcam trên trình duyệt.`);
        }
    }

    function stopWebcam() {
        state.isStreaming = false;
        if (state.mediaStream) {
            state.mediaStream.getTracks().forEach(track => track.stop());
            state.mediaStream = null;
        }
        if (state.ws) {
            state.ws.close();
            state.ws = null;
        }
        if (elements.video) elements.video.srcObject = null;
        if (elements.webcamPlaceholder) elements.webcamPlaceholder.style.display = 'flex';
        if (elements.btnStartWebcam) elements.btnStartWebcam.disabled = false;
        if (elements.btnStopWebcam) elements.btnStopWebcam.disabled = true;
        if (elements.streamDot) elements.streamDot.classList.remove('dot-live');

        if (ctx && elements.canvas) {
            ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
        }
        updateHudStats({ total: 0, counts: { Pothole: 0, Crack: 0, Manhole: 0 }, fps: 0, latency_ms: 0 });
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
        state.ws = new WebSocket(wsUrl);

        state.ws.onopen = () => {
            console.log('✅ WebSocket Stream Connected!');
            processFrameLoop();
        };

        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                renderWebcamDetections(data);
            } catch (err) {
                console.error('Frame decode error:', err);
            }
        };

        state.ws.onerror = (err) => console.error('WebSocket Error:', err);
        state.ws.onclose = () => {
            if (state.isStreaming) {
                setTimeout(connectWebSocket, 1500);
            }
        };
    }

    let isProcessingFrame = false;
    function processFrameLoop() {
        if (!state.isStreaming || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;

        if (!isProcessingFrame && elements.video && elements.video.videoWidth > 0) {
            try {
                offscreenCtx.drawImage(elements.video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
                const frameData = offscreenCanvas.toDataURL('image/jpeg', 0.6);

                state.ws.send(JSON.stringify({
                    image: frameData,
                    conf: state.conf,
                    iou: state.iou
                }));
                isProcessingFrame = true;
            } catch (e) {
                console.error('Send frame error:', e);
            }
        }

        requestAnimationFrame(processFrameLoop);
    }

    function renderWebcamDetections(data) {
        isProcessingFrame = false;
        if (!ctx || !elements.canvas) return;

        ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

        const boxes = data.boxes || [];
        const scaleX = elements.canvas.width / (elements.video.videoWidth || 640);
        const scaleY = elements.canvas.height / (elements.video.videoHeight || 360);

        const filteredBoxes = boxes.filter(b => state.classes[b.class_name] !== false);

        filteredBoxes.forEach(item => {
            const [x1, y1, x2, y2] = item.box.map((val, idx) => (idx % 2 === 0 ? val * scaleX : val * scaleY));
            const color = item.color || '#e8a55a';

            // Vẽ viền hộp
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Vẽ nhãn
            const label = `${item.class_name} ${Math.round(item.confidence * 100)}%`;
            ctx.font = '600 12.5px Inter, sans-serif';
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = color;
            ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 8, 22);

            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x1 + 4, Math.max(14, y1 - 6));
        });

        // Cập nhật số liệu HUD
        const counts = { Pothole: 0, Crack: 0, Manhole: 0 };
        filteredBoxes.forEach(b => {
            if (counts[b.class_name] !== undefined) counts[b.class_name]++;
        });

        updateHudStats({
            total: filteredBoxes.length,
            counts: counts,
            fps: data.fps || 0,
            latency_ms: data.latency_ms || 0
        });
    }

    function updateHudStats({ total = 0, counts = {}, fps = 0, latency_ms = 0 }) {
        if (elements.hudTotalDefects) elements.hudTotalDefects.textContent = total;
        if (elements.hudPotholeCount) elements.hudPotholeCount.textContent = counts['Pothole'] || 0;
        if (elements.hudCrackCount) elements.hudCrackCount.textContent = counts['Crack'] || 0;
        if (elements.hudManholeCount) elements.hudManholeCount.textContent = counts['Manhole'] || 0;
        if (elements.hudFps) elements.hudFps.textContent = Math.round(fps);
        if (elements.hudLatency) elements.hudLatency.textContent = `${latency_ms} ms`;
        if (elements.hudResTag) elements.hudResTag.textContent = `FPS: ${Math.round(fps)} · ${latency_ms}ms`;
    }

    // =========================================================
    // 9. IMAGE UPLOAD INSPECTION
    // =========================================================
    if (elements.btnBrowseFile && elements.fileInput) {
        elements.btnBrowseFile.addEventListener('click', () => elements.fileInput.click());
    }

    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                processUploadedFile(e.target.files[0]);
            }
        });
    }

    if (elements.dropzoneArea) {
        ['dragenter', 'dragover'].forEach(eventName => {
            elements.dropzoneArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                elements.dropzoneArea.classList.add('is-dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            elements.dropzoneArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                elements.dropzoneArea.classList.remove('is-dragover');
            });
        });

        elements.dropzoneArea.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                processUploadedFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (elements.btnClearUpload) {
        elements.btnClearUpload.addEventListener('click', () => {
            if (elements.uploadResultCard) elements.uploadResultCard.style.display = 'none';
            if (elements.dropzoneArea) elements.dropzoneArea.style.display = 'block';
            updateHudStats({ total: 0, counts: {} });
        });
    }

    async function processUploadedFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('conf', state.conf);
        formData.append('iou', state.iou);

        showToast(`🔍 Đang phân tích ảnh bằng mô hình: ${state.activeModelName}...`);

        try {
            const res = await fetch('/api/predict/image', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Không thể phân tích hình ảnh');
            const data = await res.json();
            renderUploadResult(data, file.name);
            showToast('✅ Phân tích ảnh hoàn tất!', 'success');
        } catch (err) {
            console.error('Upload inspection error:', err);
            showToast(`⚠️ Lỗi phân tích: ${err.message}`, 'error');
        }
    }

    function renderUploadResult(data, filename) {
        if (elements.dropzoneArea) elements.dropzoneArea.style.display = 'none';
        if (elements.uploadResultCard) elements.uploadResultCard.style.display = 'block';

        if (elements.resultFilename) elements.resultFilename.textContent = filename || data.filename || 'image.jpg';
        if (elements.resultModelTag) elements.resultModelTag.textContent = data.model_name || state.activeModelName;
        if (elements.resultImage) elements.resultImage.src = data.image_base64;

        // Render bảng chi tiết
        if (elements.defectTableBody) {
            elements.defectTableBody.innerHTML = '';
            const detections = data.detections || [];

            if (detections.length === 0) {
                if (elements.noDefectMsg) elements.noDefectMsg.style.display = 'block';
            } else {
                if (elements.noDefectMsg) elements.noDefectMsg.style.display = 'none';
                detections.forEach((det, idx) => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${idx + 1}</td>
                        <td>
                            <span class="badge-pill" style="background:${det.color}20; color:${det.color}; border:1px solid ${det.color}40;">
                                ${det.class_name}
                            </span>
                        </td>
                        <td><strong>${Math.round(det.confidence * 100)}%</strong></td>
                        <td><code>${det.width} × ${det.height} px</code></td>
                    `;
                    elements.defectTableBody.appendChild(tr);
                });
            }
        }

        // Cập nhật số liệu HUD
        updateHudStats({
            total: data.total_defects,
            counts: data.counts,
            fps: data.fps,
            latency_ms: data.latency_ms
        });
    }

    // =========================================================
    // 10. SAMPLES GALLERY (DATASET VIEWER)
    // =========================================================
    async function loadSamples(reset = false) {
        if (state.isLoadingSamples) return;
        state.isLoadingSamples = true;

        if (reset) {
            state.sampleOffset = 0;
            if (elements.samplesGrid) elements.samplesGrid.innerHTML = '';
        }

        if (elements.galleryLoadingSpinner) elements.galleryLoadingSpinner.style.display = 'inline-flex';
        if (elements.btnLoadMoreSamples) elements.btnLoadMoreSamples.style.display = 'none';

        try {
            const res = await fetch(`/api/samples?offset=${state.sampleOffset}&limit=${state.sampleLimit}`);
            const data = await res.json();

            state.sampleTotal = data.total;
            if (elements.galleryCountBadge) {
                elements.galleryCountBadge.textContent = `${data.total} ảnh trong tập Test`;
            }

            const samples = data.samples || [];
            samples.forEach(fname => {
                const card = document.createElement('div');
                card.className = 'gallery-item';
                card.innerHTML = `
                    <div class="sample-img-container">
                        <img src="/api/sample/${fname}" loading="lazy" alt="${fname}">
                        <div class="sample-overlay-hover">
                            <span class="btn-sample-detect">🔍 Nhận diện (${state.activeModel === 'flexi_yolo' ? 'Flexi-YOLO' : 'YOLOv8n'})</span>
                        </div>
                    </div>
                    <span class="sample-name-caption">${fname}</span>
                `;
                card.addEventListener('click', () => runSampleDetection(fname));
                if (elements.samplesGrid) elements.samplesGrid.appendChild(card);
            });

            state.sampleOffset += samples.length;
            const remaining = data.total - state.sampleOffset;

            if (elements.remainingCountBadge) {
                elements.remainingCountBadge.textContent = `(Còn ${Math.max(0, remaining)})`;
            }

            if (data.has_more && elements.btnLoadMoreSamples) {
                elements.btnLoadMoreSamples.style.display = 'inline-flex';
                if (elements.galleryEndMsg) elements.galleryEndMsg.style.display = 'none';
            } else {
                if (elements.galleryEndMsg) elements.galleryEndMsg.style.display = 'block';
            }
        } catch (err) {
            console.error('Lỗi tải ảnh mẫu:', err);
        } finally {
            state.isLoadingSamples = false;
            if (elements.galleryLoadingSpinner) elements.galleryLoadingSpinner.style.display = 'none';
        }
    }

    if (elements.btnLoadMoreSamples) {
        elements.btnLoadMoreSamples.addEventListener('click', () => loadSamples(false));
    }

    async function runSampleDetection(filename) {
        // Chuyển sang Tab 2 Upload để hiển thị kết quả trực quan
        elements.tabBtns.forEach(b => {
            b.classList.toggle('active', b.getAttribute('data-tab') === 'tab-upload');
        });
        elements.tabPanes.forEach(p => {
            p.classList.toggle('active', p.id === 'tab-upload');
        });

        showToast(`🔍 Đang kiểm tra ảnh mẫu [${filename}] bằng ${state.activeModelName}...`);

        try {
            const res = await fetch(`/api/predict/sample/${filename}?conf=${state.conf}&iou=${state.iou}`, {
                method: 'POST'
            });
            if (!res.ok) throw new Error('Không thể nhận diện ảnh mẫu');
            const data = await res.json();
            renderUploadResult(data, filename);
            showToast('✅ Đã phát hiện khuyết tật trên ảnh mẫu!', 'success');
        } catch (err) {
            console.error('Lỗi nhận diện ảnh mẫu:', err);
            showToast(`⚠️ ${err.message}`, 'error');
        }
    }
});
