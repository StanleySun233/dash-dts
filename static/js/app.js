// 全局变量
let currentDialogue = null;
let currentPrediction = null;
let currentPredictionDetails = null;
let currentReassess = null;
let currentReassessDetails = null;
let currentReference = null;

// DOM元素
let dialogueList, dialogueTable, segmentBtn, reassessBtn;
let pkValue, wdValue, f1Value;
let errorModal;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化DOM元素
    dialogueList = document.getElementById('dialogue-list');
    dialogueTable = document.getElementById('dialogue-table');
    segmentBtn = document.getElementById('segment-btn');
    reassessBtn = document.getElementById('reassess-btn');
    pkValue = document.getElementById('pk-value');
    wdValue = document.getElementById('wd-value');
    f1Value = document.getElementById('f1-value');
    
    // 初始化模态框
    const errorModalElement = document.getElementById('errorModal');
    
    if (errorModalElement) {
        errorModal = new bootstrap.Modal(errorModalElement);
    }
    
    loadDialogues();
    setupEventListeners();
});

// 设置事件监听器
function setupEventListeners() {
    segmentBtn.addEventListener('click', handleSegment);
    reassessBtn.addEventListener('click', handleReassess);
}

// 加载对话列表
async function loadDialogues() {
    try {
        const response = await fetch('/api/dialogues');
        const data = await response.json();
        
        if (data.success) {
            renderDialogueList(data.dialogues);
        } else {
            showError('Failed to load dialogue list: ' + data.error);
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

// 渲染对话列表
function renderDialogueList(dialogues) {
    dialogueList.innerHTML = '';
    
    dialogues.forEach(dialogue => {
        const item = document.createElement('div');
        item.className = 'list-group-item dialogue-item';
        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h6 class="mb-1">${dialogue.dial_id}</h6>
                    <small class="text-muted">${dialogue.num_utterances} utterances</small>
                </div>
                <span class="badge bg-secondary">${dialogue.num_utterances}</span>
            </div>
        `;
        
        item.addEventListener('click', () => selectDialogue(dialogue.dial_id, item));
        dialogueList.appendChild(item);
    });
}

// 选择对话
async function selectDialogue(dialId, element) {
    // 更新选中状态
    document.querySelectorAll('.dialogue-item').forEach(item => {
        item.classList.remove('active');
    });
    element.classList.add('active');
    
    try {
        showLoading('Loading dialogue details...');
        
        const response = await fetch(`/api/dialogue/${dialId}`);
        const data = await response.json();
        
        console.log('API Response:', data); // 调试信息
        
        if (data.success) {
            currentDialogue = data;
            currentReference = data.reference;
            currentPrediction = null;
            currentPredictionDetails = null;
            currentReassess = null;
            currentReassessDetails = null;
            
            console.log('Current dialogue loaded:', currentDialogue); // 调试信息
            
            renderDialogueTable();
            resetMetrics();
            reassessBtn.disabled = true;
            
            console.log('About to hide loading modal'); // 调试信息
            hideLoading();
            console.log('Loading modal hidden'); // 调试信息
        } else {
            hideLoading();
            showError('Failed to load dialogue: ' + data.error);
        }
    } catch (error) {
        hideLoading();
        showError('Network error: ' + error.message);
    }
}

// 渲染对话表格
function renderDialogueTable() {
    if (!currentDialogue) return;
    
    dialogueTable.innerHTML = '';
    
    currentDialogue.utterances.forEach((utterance, index) => {
        const row = document.createElement('tr');
        row.className = 'fade-in';
        
        const referenceValue = currentReference[index] || 0;
        const predictionValue = currentPrediction ? currentPrediction[index] : null;
        const predictionDetail = currentPredictionDetails ? currentPredictionDetails[index] : null;
        const reassessValue = currentReassess ? currentReassess[index] : null;
        const reassessDetail = currentReassessDetails ? currentReassessDetails[index] : null;
        
        row.innerHTML = `
            <td>
                <div class="utterance-text">
                    <strong>${index + 1}.</strong> ${utterance}
                </div>
            </td>
            <td>
                <span class="segment-cell segment-${referenceValue}">
                    ${referenceValue}
                </span>
            </td>
            <td>
                <span class="segment-cell ${predictionValue !== null ? `segment-${predictionValue}` : 'segment-empty'}" 
                      ${predictionDetail ? `data-bs-toggle="tooltip" data-bs-placement="top" title="Score: ${predictionDetail.score}\nReason: ${predictionDetail.reason}"` : ''}>
                    ${predictionValue !== null ? predictionValue : '-'}
                </span>
            </td>
            <td>
                <span class="segment-cell ${reassessValue !== null ? `segment-${reassessValue}` : 'segment-empty'}" 
                      ${reassessDetail ? `data-bs-toggle="tooltip" data-bs-placement="top" title="Score: ${reassessDetail.score}\nReason: ${reassessDetail.reason}"` : ''}>
                    ${reassessValue !== null ? reassessValue : '-'}
                </span>
            </td>
            <td>
                ${getStatusIndicator(index)}
            </td>
        `;
        
        dialogueTable.appendChild(row);
    });
    
    // 初始化tooltips
    initializeTooltips();
}

// 获取状态指示器
function getStatusIndicator(index) {
    if (!currentPrediction || !currentReference) {
        return '<span class="status-indicator status-unknown"></span>Unknown';
    }
    
    const reference = currentReference[index];
    const prediction = currentPrediction[index];
    const reassess = currentReassess ? currentReassess[index] : null;
    
    let status = 'Unknown';
    let className = 'status-unknown';
    
    if (reassess !== null) {
        // 使用reassess结果
        if (reference === reassess) {
            status = 'Correct';
            className = 'status-correct';
        } else {
            status = 'Incorrect';
            className = 'status-incorrect';
        }
    } else if (prediction !== null) {
        // 使用预测结果
        if (reference === prediction) {
            status = 'Correct';
            className = 'status-correct';
        } else {
            status = 'Incorrect';
            className = 'status-incorrect';
        }
    }
    
    return `<span class="status-indicator ${className}"></span>${status}`;
}

// 处理分割按钮点击
async function handleSegment() {
    if (!currentDialogue) {
        showError('Please select a dialogue first');
        return;
    }
    
    const enableHandshake = document.getElementById('handshake-check').checked;
    const enableFewShot = document.getElementById('posneg-check').checked;
    const enableSimilarity = document.getElementById('similarity-check').checked;
    
    try {
        showLoading('Performing dialogue segmentation...');
        segmentBtn.disabled = true;
        
        // 并行执行各个API
        let handshakeResults = null;
        let fewShotExamples = null;
        let similarityExamples = null;
        
        // 创建并行任务数组
        const parallelTasks = [];
        
        // 1. Handshake检测
        if (enableHandshake) {
            parallelTasks.push({
                name: 'handshake',
                task: performHandshakeDetection(),
                loadingText: 'Performing Handshake detection...'
            });
        }
        
        // 2. 正负样本生成
        if (enableFewShot) {
            parallelTasks.push({
                name: 'fewshot',
                task: performFewShotGeneration(),
                loadingText: 'Generating positive/negative samples...'
            });
        }
        
        // 3. 相似样本生成
        if (enableSimilarity) {
            parallelTasks.push({
                name: 'similarity',
                task: performSimilarityGeneration(),
                loadingText: 'Generating similar samples...'
            });
        }
        
        // 并行执行所有任务
        if (parallelTasks.length > 0) {
            showLoading('Performing parallel preprocessing...');
            
            const results = await Promise.allSettled(
                parallelTasks.map(async (task) => {
                    try {
                        const result = await task.task;
                        return { name: task.name, result, success: true };
                    } catch (error) {
                        console.warn(`${task.name} failed:`, error);
                        return { name: task.name, result: null, success: false };
                    }
                })
            );
            
            // 处理结果
            results.forEach(({ value }) => {
                if (value.success) {
                    switch (value.name) {
                        case 'handshake':
                            handshakeResults = value.result;
                            break;
                        case 'fewshot':
                            fewShotExamples = value.result;
                            break;
                        case 'similarity':
                            similarityExamples = value.result;
                            break;
                    }
                }
            });
        }
        
        // 4. 执行DTS分割
        showLoading('Performing dialogue topic segmentation...');
        const response = await fetch('/api/segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                dial_id: currentDialogue.dial_id,
                handshake_results: handshakeResults,
                few_shot_examples: fewShotExamples,
                similarity_examples: similarityExamples
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentPrediction = data.prediction;
            currentPredictionDetails = data.prediction_details || [];
            updateMetrics(data.metrics);
            reassessBtn.disabled = false;
            renderDialogueTable();
            
            hideLoading();
        } else {
            hideLoading();
            showError('Segmentation failed: ' + data.error);
        }
    } catch (error) {
        hideLoading();
        showError('Network error: ' + error.message);
    } finally {
        segmentBtn.disabled = false;
    }
}

// 处理重新评估按钮点击
async function handleReassess() {
    if (!currentDialogue || !currentPrediction) {
        showError('Please perform dialogue segmentation first');
        return;
    }
    
    try {
        showLoading('Performing reassessment...');
        reassessBtn.disabled = true;
        
        const response = await fetch('/api/reassess_dialogue', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                dial_id: currentDialogue.dial_id,
                prediction: currentPrediction
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentReassess = data.optimized_prediction;
            currentReassessDetails = data.reassess_details || [];
            updateMetrics(data.metrics);
            renderDialogueTable();
            
            // 显示变化信息
            if (data.changes_made) {
                showInfo(`Reassessment completed, modified ${data.num_changes} prediction points`);
            } else {
                showInfo('Reassessment completed, no consecutive 1s found that need modification');
            }
            
            hideLoading();
        } else {
            hideLoading();
            showError('Reassessment failed: ' + data.error);
        }
    } catch (error) {
        hideLoading();
        showError('Network error: ' + error.message);
    } finally {
        reassessBtn.disabled = false;
    }
}

// 更新指标
function updateMetrics(metrics) {
    pkValue.textContent = metrics.PK ? metrics.PK.toFixed(4) : '--';
    wdValue.textContent = metrics.WD ? metrics.WD.toFixed(4) : '--';
    f1Value.textContent = metrics.F1 ? metrics.F1.toFixed(4) : '--';
}

// 重置指标
function resetMetrics() {
    pkValue.textContent = '--';
    wdValue.textContent = '--';
    f1Value.textContent = '--';
}

// 显示加载状态（使用按钮文字变化）
function showLoading(text) {
    if (segmentBtn) {
        segmentBtn.disabled = true;
        segmentBtn.innerHTML = `<i class="bi bi-hourglass-split"></i> ${text}`;
    }
    console.log('Loading:', text);
}

// 隐藏加载状态
function hideLoading() {
    if (segmentBtn) {
        segmentBtn.disabled = false;
        segmentBtn.innerHTML = `<i class="bi bi-scissors"></i> Segment Dialogue`;
    }
    console.log('Loading completed');
}

// 显示错误信息
function showError(message) {
    document.getElementById('error-message').textContent = message;
    errorModal.show();
}

// 显示信息提示
function showInfo(message) {
    // 创建临时提示
    const alert = document.createElement('div');
    alert.className = 'alert alert-info alert-dismissible fade show position-fixed';
    alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alert);
    
    // 3秒后自动移除
    setTimeout(() => {
        if (alert.parentNode) {
            alert.parentNode.removeChild(alert);
        }
    }, 3000);
}

// 初始化tooltips
function initializeTooltips() {
    // 销毁现有的tooltips
    const existingTooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    existingTooltips.forEach(element => {
        const tooltip = bootstrap.Tooltip.getInstance(element);
        if (tooltip) {
            tooltip.dispose();
        }
    });
    
    // 初始化新的tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            html: true,
            placement: 'top'
        });
    });
}

// 工具函数：防抖
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 工具函数：节流
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Handshake检测 - 异步并行执行，8线程
async function performHandshakeDetection() {
    const utterances = currentDialogue.utterances;
    const results = new Array(utterances.length);
    
    // 创建8个并发批次
    const batchSize = Math.ceil(utterances.length / 8);
    const batches = [];
    
    for (let i = 0; i < utterances.length; i += batchSize) {
        const batch = utterances.slice(i, i + batchSize).map((utterance, batchIndex) => ({
            index: i + batchIndex,
            utterance: utterance,
            context: {
                previous: i + batchIndex > 0 ? [utterances[i + batchIndex - 1]] : [],
                current: utterance,
                next: i + batchIndex < utterances.length - 1 ? [utterances[i + batchIndex + 1]] : []
            }
        }));
        batches.push(batch);
    }
    
    // 并行执行所有批次
    const batchPromises = batches.map(async (batch) => {
        const batchResults = await Promise.all(
            batch.map(async ({ index, context }) => {
                try {
                    const response = await fetch('/api/handshake', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(context)
                    });
                    const data = await response.json();
                    return { index, data };
                } catch (error) {
                    return { index, data: { success: false, error: error.message } };
                }
            })
        );
        return batchResults;
    });
    
    // 等待所有批次完成
    const allBatchResults = await Promise.all(batchPromises);
    
    // 按原始顺序重新排列结果
    allBatchResults.flat().forEach(({ index, data }) => {
        results[index] = data;
    });
    
    return results;
}

// 正负样本生成 - 异步并行执行，8线程
async function performFewShotGeneration() {
    const utterances = currentDialogue.utterances;
    const results = new Array(utterances.length);
    
    // 创建8个并发批次
    const batchSize = Math.ceil(utterances.length / 8);
    const batches = [];
    
    for (let i = 0; i < utterances.length; i += batchSize) {
        const batch = utterances.slice(i, i + batchSize).map((_, batchIndex) => ({
            index: i + batchIndex
        }));
        batches.push(batch);
    }
    
    // 并行执行所有批次
    const batchPromises = batches.map(async (batch) => {
        const batchResults = await Promise.all(
            batch.map(async ({ index }) => {
                try {
                    const response = await fetch('/api/posneg', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ dialogue: utterances })
                    });
                    const data = await response.json();
                    return { index, result: data.result };
                } catch (error) {
                    return { index, result: null };
                }
            })
        );
        return batchResults;
    });
    
    // 等待所有批次完成
    const allBatchResults = await Promise.all(batchPromises);
    
    // 按原始顺序重新排列结果
    allBatchResults.flat().forEach(({ index, result }) => {
        results[index] = result;
    });
    
    return results;
}

// 相似样本生成
async function performSimilarityGeneration() {
    try {
        const response = await fetch('/api/similarity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dialogue: currentDialogue.utterances })
        });
        const data = await response.json();
        return data;
    } catch (error) {
        console.warn('Similarity generation failed:', error);
        return null;
    }
}

// 导出全局函数供调试使用
window.appDebug = {
    currentDialogue,
    currentPrediction,
    currentReassess,
    currentReference,
    loadDialogues,
    selectDialogue,
    hideLoading,
    showLoading
};

// 调试函数
window.debugInfo = function() {
    console.log('Current dialogue:', currentDialogue);
    console.log('Current prediction:', currentPrediction);
    console.log('Current reassess:', currentReassess);
    console.log('Current reference:', currentReference);
};
