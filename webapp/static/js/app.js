/**
 * RoadVision AI - Real-time Defect Detection Client Script
 */

document.addEventListener('DOMContentLoaded', () => {
    // State Management
    const state = {
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
        fpsTimer: null,
        fpsCounter: 0,
        currentFps: 0,
        sampleOffset: 0,
        sampleLimit: 24,
        sampleTotal: 0,
        isLoadingSamples: false
    };

    // DOM Elements
    const elements = {
        // Tabs
        tabBtns: document.querySelectorAll('.tab-btn, .category-tab'),
        tabContents: document.querySelectorAll('.tab-content, .tab-pane'),

        // Sidebar & Settings Popover Controls
        sidebar: document.getElementById('settingsSidebar'),
        btnSettingsToggle: document.getElementById('btnSettingsToggle'),
        settingsMenu: document.getElementById('settingsMenu'),
        btnCloseSettings: document.getElementById('btnCloseSettings'),
        btnToggleAllClasses: document.getElementById('btnToggleAllClasses'),
        settingsActiveIndicator: document.getElementById('settingsActiveIndicator'),

        // Sliders & Filters
        sliderConf: document.getElementById('sliderConf'),
        sliderIou: document.getElementById('sliderIou'),
        valConf: document.getElementById('valConf'),
        valIou: document.getElementById('valIou'),
        chkPothole: document.getElementById('chkPothole'),
        chkCrack: document.getElementById('chkCrack'),
        chkManhole: document.getElementById('chkManhole'),
        chkShowLabels: document.getElementById('chkShowLabels'),
        chkShowConf: document.getElementById('chkShowConf'),
        btnReloadModel: document.getElementById('btnReloadModel'),
        systemStatusBadge: document.getElementById('systemStatusBadge'),
        systemStatusText: document.getElementById('systemStatusText'),

        // HUD Stats Counters
        hudTotalDefects: document.getElementById('hudTotalDefects'),
        hudPotholeCount: document.getElementById('hudPotholeCount'),
        hudCrackCount: document.getElementById('hudCrackCount'),
        hudManholeCount: document.getElementById('hudManholeCount'),

        // Webcam Stream
        video: document.getElementById('webcamVideo'),
        canvas: document.getElementById('streamCanvas'),
        placeholder: document.getElementById('streamPlaceholder'),
        btnStartCamera: document.getElementById('btnStartCamera'),
        btnToggleCamera: document.getElementById('btnToggleCamera'),
        btnCameraText: document.getElementById('btnCameraText'),
        btnCameraIcon: document.getElementById('btnCameraIcon'),
        btnSnapshot: document.getElementById('btnSnapshot'),
        metricFps: document.getElementById('metricFps'),
        metricLatency: document.getElementById('metricLatency'),
        cameraDeviceLabel: document.getElementById('cameraDeviceLabel'),

        // Image Inspection
        dropzone: document.getElementById('imageDropzone'),
        fileInput: document.getElementById('fileInput'),
        inspectionResults: document.getElementById('inspectionResults'),
        annotatedImage: document.getElementById('annotatedImage'),
        resultFilename: document.getElementById('resultFilename'),
        resultMeta: document.getElementById('resultMeta'),
        defectTableBody: document.getElementById('defectTableBody'),
        noDefectMsg: document.getElementById('noDefectMsg'),
        btnDownloadAnnotated: document.getElementById('btnDownloadAnnotated'),
        btnExportJson: document.getElementById('btnExportJson'),

        // Samples Gallery
        samplesGrid: document.getElementById('samplesGrid'),
        galleryCountBadge: document.getElementById('galleryCountBadge'),
        galleryScrollContainer: document.getElementById('galleryScrollContainer'),
        btnLoadMoreSamples: document.getElementById('btnLoadMoreSamples'),
        remainingCountBadge: document.getElementById('remainingCountBadge'),
        galleryLoadingSpinner: document.getElementById('galleryLoadingSpinner'),
        galleryEndMsg: document.getElementById('galleryEndMsg'),
        workspaceFrame: document.querySelector('.workspace-frame')
    };

    const ctx = elements.canvas.getContext('2d');
    const offscreenCanvas = document.createElement('canvas');
    const offscreenCtx = offscreenCanvas.getContext('2d');

    // =========================================================
    // 1. NAVIGATION & FLOATING BLUR SETTINGS POPOVER
    // =========================================================
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            elements.tabBtns.forEach(b => b.classList.remove('active'));
            elements.tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(target).classList.add('active');

            if (target === 'tab-samples' && elements.samplesGrid.children.length === 0) {
                loadSamples();
            }
        });
    });

    // Toggle Floating Blur Settings Popover
    function toggleSettingsMenu(forceState = null) {
        if (!elements.settingsMenu) return;
        const isOpen = elements.settingsMenu.classList.contains('open');
        const shouldOpen = forceState !== null ? forceState : !isOpen;

        if (shouldOpen) {
            elements.settingsMenu.classList.add('open');
            elements.btnSettingsToggle?.classList.add('active');
            elements.btnSettingsToggle?.setAttribute('aria-expanded', 'true');
        } else {
            elements.settingsMenu.classList.remove('open');
            elements.btnSettingsToggle?.classList.remove('active');
            elements.btnSettingsToggle?.setAttribute('aria-expanded', 'false');
        }
    }

    if (elements.btnSettingsToggle) {
        elements.btnSettingsToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleSettingsMenu();
        });
    }

    if (elements.btnCloseSettings) {
        elements.btnCloseSettings.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleSettingsMenu(false);
        });
    }

    // Đóng popover khi click ra ngoài
    document.addEventListener('click', (e) => {
        if (!elements.settingsMenu || !elements.settingsMenu.classList.contains('open')) return;
        if (!elements.settingsMenu.contains(e.target) && !elements.btnSettingsToggle?.contains(e.target)) {
            toggleSettingsMenu(false);
        }
    });

    // Đóng khi nhấn phím Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && elements.settingsMenu?.classList.contains('open')) {
            toggleSettingsMenu(false);
        }
    });

    // Slider Controls
    elements.sliderConf.addEventListener('input', (e) => {
        state.conf = parseFloat(e.target.value);
        elements.valConf.textContent = `${Math.round(state.conf * 100)}%`;
    });

    elements.sliderIou.addEventListener('input', (e) => {
        state.iou = parseFloat(e.target.value);
        elements.valIou.textContent = `${Math.round(state.iou * 100)}%`;
    });

    // =========================================================
    // REDESIGNED DEFECT CLASS SELECTOR INTERACTIONS
    // =========================================================
    const classConfigs = [
        { el: elements.chkPothole, name: 'Pothole', tileClass: '.tile-pothole' },
        { el: elements.chkCrack, name: 'Crack', tileClass: '.tile-crack' },
        { el: elements.chkManhole, name: 'Manhole', tileClass: '.tile-manhole' }
    ];

    function updateClassFilterUI() {
        let activeCount = 0;
        classConfigs.forEach(cfg => {
            const isChecked = !!state.classes[cfg.name];
            if (isChecked) activeCount++;

            const tile = document.querySelector(cfg.tileClass);
            if (tile) {
                tile.classList.toggle('is-active', isChecked);
                const icon = tile.querySelector('.tile-status-icon');
                if (icon) icon.textContent = isChecked ? '✓' : '–';
            }
        });

        // Cập nhật số lượng lớp trên nút cài đặt
        if (elements.settingsActiveIndicator) {
            elements.settingsActiveIndicator.textContent = activeCount === 3
                ? '3/3 lớp'
                : (activeCount === 0 ? 'Tắt hết' : `${activeCount}/3 lớp`);
        }

        // Cập nhật nhãn nút Chọn tất cả
        if (elements.btnToggleAllClasses) {
            elements.btnToggleAllClasses.textContent = activeCount === 3 ? 'Bỏ chọn tất cả' : 'Chọn tất cả';
        }

        // Cập nhật lại kết quả kiểm định ảnh (nếu có)
        refreshInspectionTable();
    }

    classConfigs.forEach(({ el, name }) => {
        if (el) {
            el.addEventListener('change', (e) => {
                state.classes[name] = e.target.checked;
                updateClassFilterUI();
            });
        }
    });

    if (elements.btnToggleAllClasses) {
        elements.btnToggleAllClasses.addEventListener('click', (e) => {
            e.stopPropagation();
            const anyActive = Object.values(state.classes).some(v => v);
            const targetState = !anyActive; // Nếu đang bật ít nhất 1 cái thì tắt hết, nếu tắt hết thì bật hết

            elements.chkPothole.checked = targetState;
            elements.chkCrack.checked = targetState;
            elements.chkManhole.checked = targetState;

            state.classes['Pothole'] = targetState;
            state.classes['Crack'] = targetState;
            state.classes['Manhole'] = targetState;

            updateClassFilterUI();
        });
    }

    elements.chkShowLabels.addEventListener('change', (e) => { state.showLabels = e.target.checked; });
    elements.chkShowConf.addEventListener('change', (e) => { state.showConf = e.target.checked; });

    // Reload model
    elements.btnReloadModel.addEventListener('click', async () => {
        try {
            elements.btnReloadModel.disabled = true;
            elements.btnReloadModel.textContent = 'Đang tải lại...';
            const res = await fetch('/api/model/reload', { method: 'POST' });
            const data = await res.json();
            elements.systemStatusText.textContent = `Model: ${data.model_file}`;
            alert(`✅ Đã tải lại mô hình thành công: ${data.model_file}`);
        } catch (err) {
            alert(`⚠️ Lỗi khi tải lại mô hình: ${err.message}`);
        } finally {
            elements.btnReloadModel.disabled = false;
            elements.btnReloadModel.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <polyline points="1 20 1 14 7 14"></polyline>
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                </svg>
                <span>Tải lại mô hình</span>
            `;
        }
    });

    // =========================================================
    // 2. WEBCAM REAL-TIME STREAMING (WEBSOCKET + CANVAS)
    // =========================================================
    elements.btnStartCamera.addEventListener('click', startWebcam);
    elements.btnToggleCamera.addEventListener('click', () => {
        if (state.isStreaming) stopWebcam();
        else startWebcam();
    });

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
            elements.placeholder.style.display = 'none';
            elements.btnToggleCamera.style.display = 'inline-flex';
            elements.btnSnapshot.disabled = false;
            elements.btnCameraIcon.textContent = '⏹';
            elements.btnCameraText.textContent = 'Dừng Camera';
            state.isStreaming = true;

            const videoTrack = stream.getVideoTracks()[0];
            elements.cameraDeviceLabel.textContent = `Camera: ${videoTrack.label || 'Đang phát'}`;

            elements.video.onloadedmetadata = () => {
                elements.canvas.width = elements.video.videoWidth || 640;
                elements.canvas.height = elements.video.videoHeight || 360;
                offscreenCanvas.width = 320;
                offscreenCanvas.height = 180;
                connectWebSocket();
            };
        } catch (err) {
            console.error('Webcam error:', err);
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
        elements.video.srcObject = null;
        elements.placeholder.style.display = 'flex';
        elements.btnToggleCamera.style.display = 'none';
        elements.btnSnapshot.disabled = true;
        ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
        elements.metricFps.textContent = '0';
        elements.metricLatency.textContent = '0';
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
                renderDetections(data);
            } catch (err) {
                console.error('Frame decode error:', err);
            }
        };

        state.ws.onerror = (err) => {
            console.warn('WebSocket error, falling back to HTTP POST loop:', err);
        };

        state.ws.onclose = () => {
            console.log('WebSocket closed.');
        };
    }

    let isWaitingResponse = false;
    let lastFrameTime = performance.now();
    let frameCount = 0;

    function processFrameLoop() {
        if (!state.isStreaming) return;

        // Tính toán client-side FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            state.currentFps = Math.round((frameCount * 1000) / (now - lastFrameTime));
            elements.metricFps.textContent = state.currentFps;
            frameCount = 0;
            lastFrameTime = now;
        }

        if (state.ws && state.ws.readyState === WebSocket.OPEN && !isWaitingResponse) {
            // Vẽ frame video sang offscreen canvas nhỏ (320x180) để tăng tốc độ truyền tải
            offscreenCtx.drawImage(elements.video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
            const frameBase64 = offscreenCanvas.toDataURL('image/jpeg', 0.6);

            isWaitingResponse = true;
            state.ws.send(JSON.stringify({
                image: frameBase64,
                conf: state.conf,
                iou: state.iou
            }));
        }

        requestAnimationFrame(processFrameLoop);
    }

    function renderDetections(data) {
        isWaitingResponse = false;
        if (!state.isStreaming) return;

        const cw = elements.canvas.width;
        const ch = elements.canvas.height;
        const scaleX = cw / offscreenCanvas.width;
        const scaleY = ch / offscreenCanvas.height;

        // 1. Vẽ khung hình video gốc
        ctx.drawImage(elements.video, 0, 0, cw, ch);

        if (data.latency_ms) {
            elements.metricLatency.textContent = data.latency_ms;
        }

        // 2. Cập nhật thống kê HUD theo các lớp được kích hoạt
        if (data.counts) {
            const pCount = state.classes['Pothole'] ? (data.counts['Pothole'] || 0) : 0;
            const cCount = state.classes['Crack'] ? (data.counts['Crack'] || 0) : 0;
            const mCount = state.classes['Manhole'] ? (data.counts['Manhole'] || 0) : 0;
            elements.hudPotholeCount.textContent = pCount;
            elements.hudCrackCount.textContent = cCount;
            elements.hudManholeCount.textContent = mCount;
            elements.hudTotalDefects.textContent = pCount + cCount + mCount;
        }

        // 3. Vẽ các Bounding Boxes
        if (data.boxes && data.boxes.length > 0) {
            data.boxes.forEach(item => {
                if (!state.classes[item.class_name]) return; // Bỏ qua nếu người dùng uncheck

                const [bx1, by1, bx2, by2] = item.box;
                const x1 = bx1 * scaleX;
                const y1 = by1 * scaleY;
                const x2 = bx2 * scaleX;
                const y2 = by2 * scaleY;
                const w = x2 - x1;
                const h = y2 - y1;

                const color = item.color || (item.class_name === 'Pothole' ? '#c64545' : (item.class_name === 'Crack' ? '#e8a55a' : '#5db8a6'));

                // Vẽ Bounding Box
                ctx.strokeStyle = color;
                ctx.lineWidth = 2.5;
                ctx.strokeRect(x1, y1, w, h);

                // Vẽ góc nổi bật
                const cornerLen = Math.min(12, w / 4, h / 4);
                ctx.lineWidth = 4;
                ctx.beginPath();
                // TL
                ctx.moveTo(x1, y1 + cornerLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cornerLen, y1);
                // TR
                ctx.moveTo(x2 - cornerLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cornerLen);
                // BL
                ctx.moveTo(x1, y2 - cornerLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cornerLen, y2);
                // BR
                ctx.moveTo(x2 - cornerLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cornerLen);
                ctx.stroke();

                // Vẽ nhãn chữ
                if (state.showLabels) {
                    const labelText = state.showConf
                        ? `${item.class_name} ${Math.round(item.confidence * 100)}%`
                        : item.class_name;

                    ctx.font = '500 12px Inter, sans-serif';
                    const textWidth = ctx.measureText(labelText).width;
                    const tagHeight = 18;
                    const tagY = Math.max(0, y1 - tagHeight);

                    ctx.fillStyle = color;
                    ctx.fillRect(x1, tagY, textWidth + 8, tagHeight);

                    ctx.fillStyle = '#FFFFFF';
                    ctx.fillText(labelText, x1 + 4, tagY + 13);
                }
            });
        }
    }

    // Chụp ảnh từ webcam
    elements.btnSnapshot.addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = `road-defect-snapshot-${Date.now()}.png`;
        link.href = elements.canvas.toDataURL('image/png');
        link.click();
    });

    // =========================================================
    // 3. IMAGE UPLOAD & INSPECTION
    // =========================================================
    elements.dropzone.addEventListener('click', () => elements.fileInput.click());

    elements.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropzone.classList.add('dragover');
    });

    elements.dropzone.addEventListener('dragleave', () => {
        elements.dropzone.classList.remove('dragover');
    });

    elements.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });

    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0]);
        }
    });

    async function handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn file hình ảnh (JPG, PNG, WEBP).');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('conf', state.conf);
        formData.append('iou', state.iou);

        elements.dropzone.style.opacity = '0.5';
        try {
            const res = await fetch('/api/predict/image', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            const data = await res.json();
            displayImageResult(data);
        } catch (err) {
            console.error('Upload detection error:', err);
            alert(`⚠️ Lỗi khi phát hiện khuyết tật: ${err.message}`);
        } finally {
            elements.dropzone.style.opacity = '1';
        }
    }

    function displayImageResult(data) {
        state.lastDetectionResult = data;
        elements.inspectionResults.style.display = 'flex';
        elements.annotatedImage.src = data.image_base64;
        elements.resultFilename.textContent = data.filename || 'Uploaded Image';
        elements.resultMeta.textContent = `${data.image_size.width}x${data.image_size.height} • ${data.latency_ms}ms • ${data.model_used}`;

        refreshInspectionTable();

        // Tự động cuộn xuống phần kết quả
        elements.inspectionResults.scrollIntoView({ behavior: 'smooth' });
    }

    function refreshInspectionTable() {
        if (!state.lastDetectionResult || !elements.defectTableBody) return;
        const data = state.lastDetectionResult;

        // Cập nhật thống kê HUD dựa trên lớp người dùng đang bật
        const pCount = state.classes['Pothole'] ? (data.counts?.['Pothole'] || 0) : 0;
        const cCount = state.classes['Crack'] ? (data.counts?.['Crack'] || 0) : 0;
        const mCount = state.classes['Manhole'] ? (data.counts?.['Manhole'] || 0) : 0;
        elements.hudPotholeCount.textContent = pCount;
        elements.hudCrackCount.textContent = cCount;
        elements.hudManholeCount.textContent = mCount;
        elements.hudTotalDefects.textContent = pCount + cCount + mCount;

        // Điền dữ liệu vào bảng (chỉ hiện các lớp được bật)
        elements.defectTableBody.innerHTML = '';
        const activeDetections = (data.detections || []).filter(d => state.classes[d.class_name]);

        if (activeDetections.length > 0) {
            elements.noDefectMsg.style.display = 'none';
            activeDetections.forEach(d => {
                const tr = document.createElement('tr');
                const badgeClass = `badge-${d.class_name.toLowerCase()}`;

                tr.innerHTML = `
                    <td><strong>#${d.id}</strong></td>
                    <td><span class="badge-defect ${badgeClass}">${d.class_name}</span></td>
                    <td>
                        <div class="conf-bar">
                            <div class="bar-bg">
                                <div class="bar-fill" style="width:${d.confidence * 100}%; background:${d.color};"></div>
                            </div>
                            <span>${Math.round(d.confidence * 100)}%</span>
                        </div>
                    </td>
                    <td><code>${d.width}×${d.height}</code></td>
                `;
                elements.defectTableBody.appendChild(tr);
            });
        } else {
            elements.noDefectMsg.style.display = 'block';
        }
    }

    // Tải ảnh kết quả
    elements.btnDownloadAnnotated.addEventListener('click', () => {
        if (!state.lastDetectionResult) return;
        const link = document.createElement('a');
        link.download = `detection-${state.lastDetectionResult.filename || 'road'}.jpg`;
        link.href = state.lastDetectionResult.image_base64;
        link.click();
    });

    // Xuất báo cáo JSON
    elements.btnExportJson.addEventListener('click', () => {
        if (!state.lastDetectionResult) return;
        const exportData = { ...state.lastDetectionResult };
        delete exportData.image_base64; // Bỏ chuỗi base64 nặng để file JSON nhẹ nhàng
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.download = `report-${state.lastDetectionResult.filename || 'road'}.json`;
        link.href = url;
        link.click();
        URL.revokeObjectURL(url);
    });

    // =========================================================
    // 4. SAMPLE DATASET GALLERY (LOAD MORE & SCROLL FADE)
    // =========================================================
    async function loadSamples(isAppend = false) {
        if (state.isLoadingSamples) return;
        state.isLoadingSamples = true;

        if (elements.galleryLoadingSpinner) elements.galleryLoadingSpinner.style.display = 'flex';
        if (elements.btnLoadMoreSamples) elements.btnLoadMoreSamples.disabled = true;

        if (!isAppend) {
            state.sampleOffset = 0;
            elements.samplesGrid.innerHTML = '';
        }

        try {
            const res = await fetch(`/api/samples?offset=${state.sampleOffset}&limit=${state.sampleLimit}`);
            const data = await res.json();

            state.sampleTotal = data.total || 0;

            if (data.samples && data.samples.length > 0) {
                data.samples.forEach(filename => {
                    const card = document.createElement('div');
                    card.className = 'sample-card fade-in';
                    card.innerHTML = `
                        <img class="sample-thumbnail" src="/api/sample/${filename}" alt="${filename}" loading="lazy">
                        <div class="sample-info">
                            <span class="sample-title" title="${filename}">${filename}</span>
                            <span class="sample-badge">Chọn</span>
                        </div>
                    `;

                    card.addEventListener('click', () => runSampleDetection(filename));
                    elements.samplesGrid.appendChild(card);
                });

                state.sampleOffset += data.samples.length;
            } else if (!isAppend) {
                elements.samplesGrid.innerHTML = '<p style="color:var(--muted); padding: 24px; text-align: center;">Không tìm thấy ảnh mẫu trong thư mục dữ liệu.</p>';
            }

            // Cập nhật nhãn đếm trên Header
            if (elements.galleryCountBadge) {
                elements.galleryCountBadge.textContent = `${Math.min(state.sampleOffset, state.sampleTotal)} / ${state.sampleTotal.toLocaleString()} ảnh`;
            }

            // Cập nhật trạng thái nút Tải thêm & Thông báo hết ảnh
            const remaining = state.sampleTotal - state.sampleOffset;
            if (data.has_more && remaining > 0) {
                if (elements.btnLoadMoreSamples) {
                    elements.btnLoadMoreSamples.style.display = 'inline-flex';
                    const nextBatch = Math.min(state.sampleLimit, remaining);
                    const loadText = document.getElementById('loadMoreText');
                    if (loadText) loadText.textContent = `Tải thêm ${nextBatch} ảnh mẫu`;
                }
                if (elements.remainingCountBadge) {
                    elements.remainingCountBadge.textContent = `+${remaining.toLocaleString()} còn lại`;
                }
                if (elements.galleryEndMsg) elements.galleryEndMsg.style.display = 'none';
            } else {
                if (elements.btnLoadMoreSamples) elements.btnLoadMoreSamples.style.display = 'none';
                if (elements.galleryEndMsg) elements.galleryEndMsg.style.display = 'block';
            }
        } catch (err) {
            console.error('Failed to load samples:', err);
        } finally {
            state.isLoadingSamples = false;
            if (elements.galleryLoadingSpinner) elements.galleryLoadingSpinner.style.display = 'none';
            if (elements.btnLoadMoreSamples) elements.btnLoadMoreSamples.disabled = false;
        }
    }

    // Sự kiện bấm nút Tải thêm ảnh
    if (elements.btnLoadMoreSamples) {
        elements.btnLoadMoreSamples.addEventListener('click', () => {
            loadSamples(true);
        });
    }

    // Tự động tải thêm khi cuộn gần cuối danh sách trong tab Thư viện mẫu
    if (elements.workspaceFrame) {
        elements.workspaceFrame.addEventListener('scroll', () => {
            const tabSamples = document.getElementById('tab-samples');
            if (!tabSamples || !tabSamples.classList.contains('active')) return;
            if (state.isLoadingSamples || state.sampleOffset >= state.sampleTotal) return;

            const scrollPos = elements.workspaceFrame.scrollTop + elements.workspaceFrame.clientHeight;
            const scrollHeight = elements.workspaceFrame.scrollHeight;

            if (scrollPos >= scrollHeight - 120) {
                loadSamples(true);
            }
        });
    }

    async function runSampleDetection(filename) {
        try {
            // Chuyển sang Tab Upload & Inspection
            document.getElementById('tabBtnUpload').click();

            const res = await fetch(`/api/predict/sample/${filename}?conf=${state.conf}&iou=${state.iou}`, {
                method: 'POST'
            });

            if (!res.ok) throw new Error('Không thể phân tích ảnh mẫu.');
            const data = await res.json();
            displayImageResult(data);
        } catch (err) {
            alert(`⚠️ Lỗi phân tích ảnh mẫu: ${err.message}`);
        }
    }
});
