"""Preact components as Python strings for HTML embedding."""


def generate_preact_app() -> str:
    """Generate the complete Preact application code."""
    return '''
// Progressive chunk loading: appData is populated by decompression code in html_template
// Chunks load in order: core -> submissions -> grades -> comparisons -> llmCalls -> pdfs

// Reference to window.appData (populated by chunk loader)
let appData = window.appData;

// Update loading progress UI
function updateLoadingUI(loaded, total, chunkName) {
    const statusEl = document.getElementById('loading-status');
    const barEl = document.getElementById('loading-bar');
    if (statusEl) {
        const chunkNames = {
            'core': 'metadata',
            'submissions': 'submissions',
            'grades': 'grades',
            'comparisons': 'comparisons',
            'llmCalls': 'LLM calls',
            'pdfs': 'PDFs'
        };
        statusEl.textContent = `Loading ${chunkNames[chunkName] || chunkName}... (${loaded}/${total})`;
    }
    if (barEl) {
        barEl.style.width = `${(loaded / total) * 100}%`;
    }
}

// Listen for chunk progress
window.addEventListener('appDataProgress', (e) => {
    const { chunk, loaded, total, ready } = e.detail;
    updateLoadingUI(loaded, total, chunk);

    // Initialize app once we have core + submissions + grades
    // This allows showing the grades table before heavy data loads
    console.log('[AppProgress]', chunk, loaded + '/' + total,
        'ready:', JSON.stringify(ready),
        'initialized:', window._appInitialized,
        'submissions:', Object.keys(appData.submissions || {}).length);

    if (ready.core && ready.submissions && ready.grades && !window._appInitialized) {
        console.log('[AppInit] Triggering initializeApp, submissions:', Object.keys(appData.submissions || {}).length);
        window._appInitialized = true;
        initializeApp();
    }
});

// Handle complete event (fallback initialization if needed)
window.addEventListener('appDataComplete', () => {
    // Fallback: ensure app initializes if it hasn't yet
    if (!window._appInitialized && Object.keys(appData.submissions || {}).length > 0) {
        console.log('[AppComplete] Fallback: initializing app now');
        window._appInitialized = true;
        initializeApp();
    }
    // Note: App component listens for appDataComplete to trigger re-render via state
});

// ============ Utility Functions ============

function parseMarkdown(md) {
    if (!md) return '<em class="text-gray">No content available</em>';

    let html = md
        // Escape HTML first
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        // Code blocks (before other processing)
        .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Headers
        .replace(/^### (.*)$/gm, '<h3>$1</h3>')
        .replace(/^## (.*)$/gm, '<h2>$1</h2>')
        .replace(/^# (.*)$/gm, '<h1>$1</h1>')
        // Bold and Italic
        .replace(/\\*\\*\\*(.+?)\\*\\*\\*/g, '<strong><em>$1</em></strong>')
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
        .replace(/___(.+?)___/g, '<strong><em>$1</em></strong>')
        .replace(/__(.+?)__/g, '<strong>$1</strong>')
        .replace(/_(.+?)_/g, '<em>$1</em>')
        // Blockquotes
        .replace(/^> (.*)$/gm, '<blockquote>$1</blockquote>')
        // Unordered lists
        .replace(/^[\\*\\-] (.*)$/gm, '<li>$1</li>')
        // Ordered lists
        .replace(/^\\d+\\. (.*)$/gm, '<li>$1</li>')
        // Line breaks
        .replace(/\\n\\n/g, '</p><p>')
        .replace(/\\n/g, '<br>');

    // Wrap in paragraphs if not already wrapped
    if (!html.startsWith('<')) {
        html = '<p>' + html + '</p>';
    }

    // Clean up consecutive blockquotes
    html = html.replace(/<\\/blockquote>\\s*<blockquote>/g, '<br>');

    // Wrap lists
    html = html.replace(/(<li>.*<\\/li>)/gs, '<ul>$1</ul>');
    html = html.replace(/<\\/ul>\\s*<ul>/g, '');

    return html;
}

function highlightMarkdownSource(md) {
    if (!md) return '<em class="text-gray">No content available</em>';

    let html = md
        // Escape HTML first
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Code blocks (fenced) - must be done first
    html = html.replace(/(```[\\s\\S]*?```)/g, '<span class="md-code-block">$1</span>');

    // Headers (only if not inside code block)
    html = html.replace(/^(#{1,6} .*)$/gm, '<span class="md-header">$1</span>');

    // Horizontal rules
    html = html.replace(/^([-*_]{3,})$/gm, '<span class="md-hr">$1</span>');

    // Bold with ** or __
    html = html.replace(/(\\*\\*[^*]+\\*\\*)/g, '<span class="md-bold">$1</span>');
    html = html.replace(/(__[^_]+__)/g, '<span class="md-bold">$1</span>');

    // Inline code (backticks, not inside code blocks)
    html = html.replace(/(`[^`]+`)/g, '<span class="md-code">$1</span>');

    // Links [text](url)
    html = html.replace(/(\\[[^\\]]+\\]\\([^)]+\\))/g, '<span class="md-link">$1</span>');

    // List items
    html = html.replace(/^([\\*\\-+] )/gm, '<span class="md-list">$1</span>');
    html = html.replace(/^(\\d+\\. )/gm, '<span class="md-list">$1</span>');

    // Blockquotes
    html = html.replace(/^(&gt; .*)/gm, '<span class="md-blockquote">$1</span>');

    return '<div class="markdown-source">' + html + '</div>';
}

function extractLLMOutput(rawJson) {
    if (!rawJson) return null;

    // Try common structures for LLM output
    if (rawJson.output) {
        if (typeof rawJson.output === 'string') return rawJson.output;
        if (rawJson.output.content) return extractContentText(rawJson.output.content);
        if (rawJson.output.text) return rawJson.output.text;
        if (rawJson.output.message) return extractMessageContent(rawJson.output.message);
    }

    if (rawJson.response) {
        if (typeof rawJson.response === 'string') return rawJson.response;
        if (rawJson.response.content) return extractContentText(rawJson.response.content);
        if (rawJson.response.text) return rawJson.response.text;
        if (rawJson.response.message) return extractMessageContent(rawJson.response.message);
    }

    if (rawJson.completion) {
        if (typeof rawJson.completion === 'string') return rawJson.completion;
    }

    if (rawJson.choices && Array.isArray(rawJson.choices) && rawJson.choices.length > 0) {
        const choice = rawJson.choices[0];
        if (choice.message) return extractMessageContent(choice.message);
        if (choice.text) return choice.text;
    }

    if (rawJson.content) {
        return extractContentText(rawJson.content);
    }

    if (rawJson.result) {
        if (typeof rawJson.result === 'string') return rawJson.result;
        if (rawJson.result.content) return extractContentText(rawJson.result.content);
    }

    return null;
}

function extractContentText(content) {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
        return content
            .filter(part => part.type === 'text' || part.text)
            .map(part => part.text || part)
            .join('\\n');
    }
    return null;
}

function extractMessageContent(message) {
    if (typeof message === 'string') return message;
    if (message.content) return extractContentText(message.content);
    if (message.text) return message.text;
    return null;
}

function extractLLMInput(rawJson) {
    if (!rawJson) return null;

    if (rawJson.input) {
        if (typeof rawJson.input === 'string') return rawJson.input;
        if (rawJson.input.messages) return formatMessages(rawJson.input.messages);
        if (rawJson.input.prompt) return rawJson.input.prompt;
    }

    if (rawJson.messages) {
        return formatMessages(rawJson.messages);
    }

    if (rawJson.prompt) {
        return rawJson.prompt;
    }

    if (rawJson.request && rawJson.request.messages) {
        return formatMessages(rawJson.request.messages);
    }

    return null;
}

function formatMessages(messages) {
    if (!Array.isArray(messages)) return null;

    return messages.map(msg => {
        const role = msg.role || 'unknown';
        let content = '';

        if (typeof msg.content === 'string') {
            content = msg.content;
        } else if (Array.isArray(msg.content)) {
            content = msg.content
                .filter(part => part.type === 'text')
                .map(part => part.text)
                .join('\\n');
        }

        return `**${role.charAt(0).toUpperCase() + role.slice(1)}:**\\n\\n${content}`;
    }).join('\\n\\n---\\n\\n');
}

function formatJsonWithHighlighting(obj) {
    const jsonStr = JSON.stringify(obj, null, 2);
    return jsonStr
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
        .replace(/: "([^"]*)"/g, ': <span class="json-string">"$1"</span>')
        .replace(/: (\\d+\\.?\\d*)/g, ': <span class="json-number">$1</span>')
        .replace(/: (true|false)/g, ': <span class="json-boolean">$1</span>')
        .replace(/: (null)/g, ': <span class="json-null">$1</span>');
}

function generateHistogramSVG(distribution, maxGrade, width, height) {
    width = width || 300;
    height = height || 100;

    if (!distribution || distribution.length === 0) {
        return '<em class="text-gray">No distribution data</em>';
    }

    const barWidth = width / distribution.length;
    const maxProb = Math.max(...distribution) || 1;
    const chartHeight = height - 25;

    let bars = '';
    distribution.forEach((prob, i) => {
        const barHeight = (prob / maxProb) * chartHeight;
        const x = i * barWidth;
        const y = chartHeight - barHeight;
        const opacity = prob > 0 ? 0.7 + (prob / maxProb) * 0.3 : 0.1;
        bars += `<rect x="${x}" y="${y}" width="${barWidth - 1}" height="${barHeight}" fill="#3b82f6" opacity="${opacity}"/>`;
        if (distribution.length <= 20) {
            bars += `<text x="${x + barWidth/2}" y="${height - 5}" text-anchor="middle" font-size="9" fill="#6b7280">${i}</text>`;
        }
    });

    return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f9fafb" rx="4"/>
        ${bars}
    </svg>`;
}

function formatDate(isoString) {
    if (!isoString) return '';
    try {
        return new Date(isoString).toLocaleString();
    } catch {
        return isoString;
    }
}

// ============ Shared Components ============

function InspectionScreen({ title, metadata, children }) {
    return html`
        <div class="inspection-screen">
            <div class="inspection-header">
                <h4>${title}</h4>
                ${metadata && html`<div class="inspection-meta">${metadata}</div>`}
            </div>
            <div class="inspection-content">
                ${children}
            </div>
        </div>
    `;
}

function CollapsibleSection({ title, defaultOpen = false, children }) {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return html`
        <div class="collapsible-section ${isOpen ? 'open' : ''}">
            <div class="collapsible-header" onClick=${() => setIsOpen(!isOpen)}>
                <span class="chevron">▶</span>
                ${title}
            </div>
            <div class="collapsible-content">
                ${children}
            </div>
        </div>
    `;
}

function MarkdownContent({ content }) {
    return html`<div class="markdown-content"
        dangerouslySetInnerHTML=${{ __html: parseMarkdown(content) }} />`;
}

function MarkdownSource({ content }) {
    return html`<div
        dangerouslySetInnerHTML=${{ __html: highlightMarkdownSource(content) }} />`;
}

function JsonDisplay({ data }) {
    return html`<pre class="json-display"
        dangerouslySetInnerHTML=${{ __html: formatJsonWithHighlighting(data) }} />`;
}

function Histogram({ distribution, maxGrade, width = 400, height = 120 }) {
    return html`<div class="histogram-container"
        dangerouslySetInnerHTML=${{ __html: generateHistogramSVG(distribution, maxGrade, width, height) }} />`;
}

// ============ LLM Call Components ============

function LLMCallInspection({ call }) {
    const output = useMemo(() => extractLLMOutput(call.raw_json), [call]);
    const input = useMemo(() => extractLLMInput(call.raw_json), [call]);

    return html`
        <${InspectionScreen}
            title="LLM Call"
            metadata=${html`<span class="call-id">${call.call_id}</span>`}
        >
            <div class="llm-output-section">
                <h5>Output</h5>
                <div class="llm-output-content">
                    ${output ? html`<${MarkdownSource} content=${output} />`
                             : html`<em class="text-gray">No output data found</em>`}
                </div>
            </div>
            <${CollapsibleSection} title="Input">
                ${input ? html`<${MarkdownSource} content=${input} />`
                        : html`<em class="text-gray">No input data found</em>`}
            <//>
            <${CollapsibleSection} title="Raw JSON">
                <${JsonDisplay} data=${call.raw_json} />
            <//>
        <//>
    `;
}

function LLMCallsList({ calls, title = "LLM Calls", egfName = null }) {
    // Reusable component for displaying a list of LLM calls with tabs
    // egfName is used to look up raw_json from allLLMCalls when not present in call object
    const [activeCall, setActiveCall] = useState(0);

    // Enrich calls with raw_json from allLLMCalls if not present
    // This de-duplicates LLM call data - raw_json is stored only in allLLMCalls
    const enrichedCalls = useMemo(() => {
        if (!calls || calls.length === 0) return [];
        return calls.map(call => {
            // If raw_json already present (e.g., from comparison modal), use it
            if (call.raw_json) return call;
            // Otherwise look up from allLLMCalls
            if (egfName && appData.allLLMCalls?.[egfName]?.[call.call_id]) {
                return {
                    ...call,
                    raw_json: appData.allLLMCalls[egfName][call.call_id].raw_json
                };
            }
            // Fallback: search all EGFs for this call_id
            for (const eName of (appData.egfNames || [])) {
                const callData = appData.allLLMCalls?.[eName]?.[call.call_id];
                if (callData?.raw_json) {
                    return { ...call, raw_json: callData.raw_json };
                }
            }
            return call;
        });
    }, [calls, egfName]);

    if (!enrichedCalls || enrichedCalls.length === 0) {
        return html`<p class="no-calls-message">No LLM calls recorded</p>`;
    }

    // Generate tab label: use pass_number if available, otherwise index+1
    const getTabLabel = (call, idx) => {
        if (call.pass_number !== undefined && call.pass_number !== null) {
            return call.pass_number + 1;
        }
        return idx + 1;
    };

    return html`
        <div class="llm-calls-section">
            ${title && html`<h4>${title}</h4>`}
            <div class="subtabs">
                ${enrichedCalls.map((call, idx) => html`
                    <button
                        class="subtab-btn ${idx === activeCall ? 'active' : ''}"
                        onClick=${() => setActiveCall(idx)}
                        title=${call.call_id}
                        key=${call.call_id || idx}
                    >${getTabLabel(call, idx)}</button>
                `)}
            </div>
            <${LLMCallInspection} call=${enrichedCalls[activeCall]} />
        </div>
    `;
}

// Legacy alias for backwards compatibility
function LLMCallsSection({ calls, egfName = null }) {
    return html`<${LLMCallsList} calls=${calls} title="LLM Calls" egfName=${egfName} />`;
}

// ============ Comparison Components ============

function SubmissionCard({ submissionId, submissions, isWinner, label }) {
    const sub = submissions[submissionId];
    const name = sub?.student_name || sub?.student_id || submissionId;
    const isExternal = !sub;

    return html`
        <div class="submission-card ${isWinner ? 'winner' : ''} ${isExternal ? 'external' : ''}">
            <span class="submission-label">${label}</span>
            <span class="submission-name">${name}</span>
            ${isWinner && html`<span class="winner-badge">Winner</span>`}
            ${isExternal && html`<span class="external-badge">External</span>`}
        </div>
    `;
}

function ComparisonCard({ comparison, submissions, onExpand }) {
    const subA = submissions[comparison.submission_a];
    const subB = submissions[comparison.submission_b];
    const nameA = subA?.student_name || subA?.student_id || comparison.submission_a;
    const nameB = subB?.student_name || subB?.student_id || comparison.submission_b;

    return html`
        <div class="comparison-card" onClick=${onExpand}>
            <div class="comparison-summary">
                <span class="participant ${comparison.winner === 'A' ? 'winner' : ''}">${nameA}</span>
                <span class="vs">vs</span>
                <span class="participant ${comparison.winner === 'B' ? 'winner' : ''}">${nameB}</span>
            </div>
            <div class="comparison-meta">
                <span class="confidence">${(comparison.confidence * 100).toFixed(0)}%</span>
                ${comparison.is_external && html`<span class="external-badge">External</span>`}
            </div>
        </div>
    `;
}

function ComparisonsTab({ comparisons, submissions, onComparisonClick }) {
    if (!comparisons || comparisons.length === 0) {
        return html`<p class="no-comparisons text-gray">No comparisons involving this submission</p>`;
    }

    return html`
        <div class="comparisons-list">
            <p class="comparisons-count">${comparisons.length} comparison(s)</p>
            ${comparisons.map((comp, idx) => html`
                <${ComparisonCard}
                    comparison=${comp}
                    submissions=${submissions}
                    onExpand=${() => onComparisonClick(comp)}
                    key=${comp.comparison_id}
                />
            `)}
        </div>
    `;
}

// ============ Tab Content Components ============

function EssayTab({ submission }) {
    return html`<${MarkdownContent} content=${submission.essay_markdown} />`;
}

function PDFTab({ submission }) {
    if (!submission.pdf_base64) {
        return html`<div class="pdf-not-available">
            <em class="text-gray">No PDF available for this submission</em>
        </div>`;
    }

    const pdfUrl = 'data:application/pdf;base64,' + submission.pdf_base64;
    return html`
        <div class="pdf-container">
            <iframe src=${pdfUrl} class="pdf-viewer"></iframe>
        </div>
    `;
}

function GroundTruthTab({ submission, maxGrade, noiseAssumption }) {
    return html`
        <div>
            <div class="grade-display">
                <span class="grade-value">${submission.ground_truth_grade ?? 'N/A'}</span>
                <span class="grade-max">/ ${maxGrade}</span>
            </div>
            <div class="histogram-section">
                <h4>Teacher Noise Distribution (${noiseAssumption})</h4>
                <${Histogram}
                    distribution=${submission.gt_distribution}
                    maxGrade=${maxGrade}
                />
            </div>
            ${submission.gt_justification && html`
                <div class="justification-section">
                    <h4>Justification</h4>
                    <${MarkdownContent} content=${submission.gt_justification} />
                </div>
            `}
        </div>
    `;
}

function EGFTab({ gradeDetail, maxGrade, egfLabel, egfName = null }) {
    if (!gradeDetail) {
        return html`<p class="text-gray">Not graded by ${egfLabel}</p>`;
    }

    return html`
        <div>
            <div class="grade-display">
                <span class="grade-value">${gradeDetail.grade}</span>
                <span class="grade-max">/ ${maxGrade}</span>
            </div>
            <div class="histogram-section">
                <h4>Grade Distribution</h4>
                <${Histogram}
                    distribution=${gradeDetail.grade_distribution}
                    maxGrade=${maxGrade}
                />
            </div>
            ${gradeDetail.justification && html`
                <div class="justification-section">
                    <h4>Justification</h4>
                    <${MarkdownContent} content=${gradeDetail.justification} />
                </div>
            `}
            <${LLMCallsSection} calls=${gradeDetail.llm_calls} egfName=${egfName} />
        </div>
    `;
}

// ============ Comparison Modal Component ============

function ComparisonPairModal({ isOpen, onClose, pairData, onSubmissionClick, onBack, canGoBack }) {
    const [activeEGF, setActiveEGF] = useState(0);

    useEffect(() => {
        if (isOpen) {
            setActiveEGF(0);
        }
    }, [isOpen, pairData]);

    if (!isOpen || !pairData) return null;

    const { subA, subB, gtA, gtB, gtWinner, gtGap, comparisons } = pairData;

    const submissionA = appData.submissions[subA];
    const submissionB = appData.submissions[subB];
    const nameA = submissionA?.student_name || submissionA?.student_id || subA;
    const nameB = submissionB?.student_name || submissionB?.student_id || subB;

    // Gather all LLM calls
    const allCalls = useMemo(() => {
        const calls = {};
        for (const egfName of appData.egfNames) {
            const egfCalls = appData.allLLMCalls?.[egfName] || {};
            for (const [callId, call] of Object.entries(egfCalls)) {
                calls[callId] = call;
            }
        }
        return calls;
    }, []);

    const handleOverlayClick = (e) => {
        if (e.target.classList.contains('modal-overlay')) onClose();
    };

    useEffect(() => {
        const handleEsc = (e) => { if (e.key === 'Escape') onClose(); };
        document.addEventListener('keydown', handleEsc);
        return () => document.removeEventListener('keydown', handleEsc);
    }, [onClose]);

    const currentComparison = comparisons[activeEGF];

    return html`
        <div class="modal-overlay active" onClick=${handleOverlayClick}>
            <div class="modal-content" style="max-width: 900px;">
                <div class="modal-header">
                    ${canGoBack && html`<button class="modal-back" onClick=${onBack}>←</button>`}
                    <h3>Comparison Details</h3>
                    <button class="modal-close" onClick=${onClose}>×</button>
                </div>
                <div class="modal-body">
                    <div class="comparison-pair-header">
                        <div class="comparison-essay-card"
                             onClick=${() => onSubmissionClick(subA)}>
                            <span class="essay-label">Essay A</span>
                            <span class="essay-name clickable">${nameA}</span>
                            <span class="essay-grade">GT: ${gtA}</span>
                            ${gtWinner === 'A' && html`<span class="gt-winner-badge">GT Winner</span>`}
                        </div>
                        <div class="vs-divider">
                            <span class="vs-text">vs</span>
                            <span class="gap-text">Gap: ${gtGap}</span>
                        </div>
                        <div class="comparison-essay-card"
                             onClick=${() => onSubmissionClick(subB)}>
                            <span class="essay-label">Essay B</span>
                            <span class="essay-name clickable">${nameB}</span>
                            <span class="essay-grade">GT: ${gtB}</span>
                            ${gtWinner === 'B' && html`<span class="gt-winner-badge">GT Winner</span>`}
                        </div>
                    </div>

                    ${gtWinner === '-' && html`
                        <div class="gt-tie-notice">
                            Ground truth grades are equal - no clear winner
                        </div>
                    `}

                    ${comparisons.length > 1 && html`
                        <div class="egf-tabs">
                            ${comparisons.map((item, idx) => html`
                                <button
                                    class="tab-btn ${idx === activeEGF ? 'active' : ''}"
                                    onClick=${() => setActiveEGF(idx)}
                                    key=${item.egfName}
                                >${appData.egfLabels[item.egfName] || item.egfName}</button>
                            `)}
                        </div>
                    `}

                    ${currentComparison && html`
                        <div class="comparison-result-section">
                            <h4>${comparisons.length === 1 ? (appData.egfLabels[currentComparison.egfName] || currentComparison.egfName) + ' Result' : 'Result'}</h4>

                            <div class="result-summary">
                                <span class="result-label">Winner:</span>
                                <span class="result-value ${currentComparison.comparison.winner === gtWinner ? 'correct' : (gtWinner === '-' ? '' : 'incorrect')}">
                                    ${currentComparison.comparison.winner || '-'}
                                    ${currentComparison.comparison.winner === gtWinner && gtWinner !== '-' && html`<span class="check-mark"> ✓</span>`}
                                    ${currentComparison.comparison.winner && currentComparison.comparison.winner !== gtWinner && gtWinner !== '-' && html`<span class="cross-mark"> ✗</span>`}
                                </span>
                                ${currentComparison.comparison.confidence && html`
                                    <span class="result-confidence">(${(currentComparison.comparison.confidence * 100).toFixed(0)}% confidence)</span>
                                `}
                            </div>

                            ${currentComparison.comparison.justification && html`
                                <${CollapsibleSection} title="Justification" defaultOpen=${true}>
                                    <${MarkdownContent} content=${currentComparison.comparison.justification} />
                                <//>
                            `}

                            ${(() => {
                                const relevantCalls = (currentComparison.comparison.call_ids || [])
                                    .map(id => allCalls[id])
                                    .filter(Boolean);
                                return relevantCalls.length > 0 && html`
                                    <${CollapsibleSection} title="LLM Calls (${relevantCalls.length})" defaultOpen=${false}>
                                        <${LLMCallsList} calls=${relevantCalls} title=${null} />
                                    <//>
                                `;
                            })()}
                        </div>
                    `}
                </div>
            </div>
        </div>
    `;
}

// ============ Submission Modal Component ============

function Modal({ isOpen, onClose, submissionId, onComparisonClick, onBack, canGoBack }) {
    const [activeTab, setActiveTab] = useState('essay');

    useEffect(() => {
        if (isOpen) {
            setActiveTab('essay');
        }
    }, [isOpen, submissionId]);

    if (!isOpen || !submissionId) return null;

    const submission = appData.submissions[submissionId];
    if (!submission) return null;

    const title = submission.student_name || submission.student_id || submissionId;

    // Gather all comparisons for this submission across all EGFs
    const allComparisons = useMemo(() => {
        const result = [];
        const seen = new Set();
        for (const egfName of appData.egfNames) {
            const egfComps = appData.egfComparisons?.[egfName]?.[submissionId] || [];
            for (const comp of egfComps) {
                if (!seen.has(comp.comparison_id)) {
                    seen.add(comp.comparison_id);
                    result.push(comp);
                }
            }
        }
        return result;
    }, [submissionId]);

    // Build tabs
    const tabs = [
        { id: 'essay', label: 'Essay' },
        { id: 'pdf', label: 'PDF' },
        { id: 'gt', label: 'Ground Truth' },
        ...appData.egfNames.map((name, idx) => ({
            id: `egf-${idx}`,
            label: appData.egfLabels[name] || name,
            egfName: name
        })),
    ];

    if (allComparisons.length > 0) {
        tabs.push({ id: 'comparisons', label: `Comparisons (${allComparisons.length})` });
    }

    const handleOverlayClick = (e) => {
        if (e.target.classList.contains('modal-overlay')) onClose();
    };

    useEffect(() => {
        const handleEsc = (e) => { if (e.key === 'Escape') onClose(); };
        document.addEventListener('keydown', handleEsc);
        return () => document.removeEventListener('keydown', handleEsc);
    }, [onClose]);

    return html`
        <div class="modal-overlay active" onClick=${handleOverlayClick}>
            <div class="modal-content">
                <div class="modal-header">
                    ${canGoBack && html`<button class="modal-back" onClick=${onBack}>←</button>`}
                    <h3>${title}</h3>
                    <button class="modal-close" onClick=${onClose}>×</button>
                </div>
                <div class="modal-tabs">
                    ${tabs.map(tab => html`
                        <button
                            class="tab-btn ${activeTab === tab.id ? 'active' : ''}"
                            onClick=${() => setActiveTab(tab.id)}
                            key=${tab.id}
                        >${tab.label}</button>
                    `)}
                </div>
                <div class="modal-body">
                    ${activeTab === 'essay' && html`<${EssayTab} submission=${submission} />`}
                    ${activeTab === 'pdf' && html`<${PDFTab} submission=${submission} />`}
                    ${activeTab === 'gt' && html`
                        <${GroundTruthTab}
                            submission=${submission}
                            maxGrade=${appData.maxGrade}
                            noiseAssumption=${appData.noiseAssumption}
                        />
                    `}
                    ${tabs.filter(t => t.id.startsWith('egf-')).map(tab =>
                        activeTab === tab.id && html`
                            <${EGFTab}
                                gradeDetail=${appData.egfGrades[tab.egfName]?.[submissionId]}
                                maxGrade=${appData.maxGrade}
                                egfLabel=${tab.label}
                                egfName=${tab.egfName}
                                key=${tab.id}
                            />
                        `
                    )}
                    ${activeTab === 'comparisons' && html`
                        <${ComparisonsTab}
                            comparisons=${allComparisons}
                            submissions=${appData.submissions}
                            onComparisonClick=${onComparisonClick}
                        />
                    `}
                </div>
            </div>
        </div>
    `;
}

// ============ Comparisons Table Component ============

function ComparisonWinnerCell({ winner, gtWinner, hasGT }) {
    // winner: "A", "B", "", "TIE", undefined
    // gtWinner: "A", "B", "-" (tie or no GT)
    // hasGT: boolean - whether both submissions have GT grades
    let className = 'grade-cell';

    const normalizedWinner = winner?.toUpperCase() || '';
    const displayWinner = normalizedWinner === 'TIE' ? '-' : (normalizedWinner || '-');

    if (!hasGT && normalizedWinner) {
        // No GT grades available - show gray background
        className += ' grade-neutral';
    } else if (gtWinner && gtWinner !== '-' && normalizedWinner) {
        if (normalizedWinner === gtWinner) {
            className += ' grade-match';  // Correct prediction
        } else if (normalizedWinner === 'TIE') {
            className += ' grade-close';  // Called tie when there was a winner
        } else {
            className += ' grade-far';  // Wrong winner
        }
    }

    return html`<td class=${className}>${displayWinner}</td>`;
}

function ComparisonsTable({ onComparisonClick }) {
    // Build map of all comparison pairs across all EGFs
    const comparisonData = useMemo(() => {
        const pairMap = new Map();  // key: "subA|subB" -> { subA, subB, egfWinners: {}, gtWinner, gtGap, comparisons: [] }

        for (const egfName of appData.egfNames) {
            const egfComps = appData.egfComparisons?.[egfName] || {};
            // egfComps is keyed by submission_id, each value is array of comparisons
            const seen = new Set();

            for (const submissionId of Object.keys(egfComps)) {
                for (const comp of egfComps[submissionId]) {
                    // Skip if already processed this comparison
                    if (seen.has(comp.comparison_id)) continue;
                    seen.add(comp.comparison_id);

                    const subA = comp.submission_a;
                    const subB = comp.submission_b;

                    // Check if both submissions have GT grades
                    const gtA = appData.submissions[subA]?.ground_truth_grade;
                    const gtB = appData.submissions[subB]?.ground_truth_grade;
                    const hasGT = gtA !== null && gtA !== undefined && gtB !== null && gtB !== undefined;

                    // Create canonical key (always same order)
                    const key = subA < subB ? `${subA}|${subB}` : `${subB}|${subA}`;
                    const isFlipped = subA > subB;

                    if (!pairMap.has(key)) {
                        const [canonicalA, canonicalB] = subA < subB ? [subA, subB] : [subB, subA];
                        const [canonicalGtA, canonicalGtB] = subA < subB ? [gtA, gtB] : [gtB, gtA];

                        // Compute GT winner (only if both GT grades exist)
                        let gtWinner = '-';
                        let gtGap = '-';
                        if (hasGT) {
                            if (canonicalGtA > canonicalGtB) gtWinner = 'A';
                            else if (canonicalGtA < canonicalGtB) gtWinner = 'B';
                            gtGap = Math.abs(canonicalGtA - canonicalGtB);
                        }

                        pairMap.set(key, {
                            subA: canonicalA,
                            subB: canonicalB,
                            gtA: canonicalGtA,
                            gtB: canonicalGtB,
                            hasGT,
                            gtWinner,
                            gtGap,
                            egfWinners: {},
                            comparisons: [],  // Store actual comparison objects by EGF
                        });
                    }

                    // Add this EGF's winner (adjusting if flipped)
                    let winner = comp.winner?.toUpperCase() || '';
                    if (isFlipped && (winner === 'A' || winner === 'B')) {
                        winner = winner === 'A' ? 'B' : 'A';
                    }

                    const pair = pairMap.get(key);
                    pair.egfWinners[egfName] = winner;

                    // Store the comparison object (adjusted for canonical order)
                    // Only add if we haven't already added one for this EGF
                    if (!pair.comparisons.some(c => c.egfName === egfName)) {
                        const adjustedComp = isFlipped ? {
                            ...comp,
                            submission_a: subB,
                            submission_b: subA,
                            winner: winner,
                        } : { ...comp, winner: winner };
                        pair.comparisons.push({ egfName, comparison: adjustedComp });
                    }
                }
            }
        }

        // Convert to array and sort: GT comparisons first (by gap descending), then no-GT comparisons
        return Array.from(pairMap.values()).sort((a, b) => {
            // No-GT comparisons go to the end
            if (a.hasGT && !b.hasGT) return -1;
            if (!a.hasGT && b.hasGT) return 1;
            if (!a.hasGT && !b.hasGT) return 0;
            // Both have GT: sort by gap descending
            return b.gtGap - a.gtGap;
        });
    }, []);

    // Virtual scrolling configuration
    const ROW_HEIGHT = 41;
    const CONTAINER_HEIGHT = 400;
    const BUFFER = 5;

    const [scrollTop, setScrollTop] = useState(0);

    // Calculate visible range
    const totalHeight = comparisonData.length * ROW_HEIGHT;
    const visibleStart = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER);
    const visibleEnd = Math.min(
        comparisonData.length,
        Math.ceil((scrollTop + CONTAINER_HEIGHT) / ROW_HEIGHT) + BUFFER
    );
    const visiblePairs = comparisonData.slice(visibleStart, visibleEnd);
    const offsetY = visibleStart * ROW_HEIGHT;

    const handleScroll = useCallback((e) => {
        setScrollTop(e.target.scrollTop);
    }, []);

    if (comparisonData.length === 0) {
        return null;  // Don't show section if no annotated comparisons
    }

    const getDisplayName = (subId) => {
        const sub = appData.submissions[subId];
        return sub?.student_name || sub?.student_id || subId;
    };

    return html`
        <section class="grades-section">
            <h2>Comparisons</h2>
            <p style="color: var(--gray-600); font-size: 0.875rem; margin-bottom: 1rem;">
                Pairwise comparisons. Click a row to view details.
                Colors: <span style="color: var(--success);">correct</span>,
                <span style="color: var(--warning);">tie vs winner</span>,
                <span style="color: var(--danger);">wrong winner</span>,
                <span style="color: var(--gray-500);">no GT (unevaluable)</span>.
                <span style="margin-left: 1rem; color: var(--gray-400);">(${comparisonData.length} pairs)</span>
            </p>
            <div class="table-container virtual-table"
                 style="height: ${Math.min(CONTAINER_HEIGHT, totalHeight + 50)}px; overflow-y: auto;"
                 onScroll=${handleScroll}>
                <div style="height: ${totalHeight}px; position: relative;">
                    <table class="grades-table" style="position: absolute; top: ${offsetY}px; width: 100%;">
                        <thead>
                            <tr>
                                <th>Essay A</th>
                                <th>Essay B</th>
                                <th>GT Winner</th>
                                <th>GT Gap</th>
                                ${appData.egfNames.map(name => html`
                                    <th title=${name} key=${name}>${appData.egfLabels[name] || name}</th>
                                `)}
                            </tr>
                        </thead>
                        <tbody>
                            ${visiblePairs.map((pair, idx) => html`
                                <tr class="grade-row" onClick=${() => onComparisonClick(pair)} key=${visibleStart + idx}>
                                    <td>${getDisplayName(pair.subA)}</td>
                                    <td>${getDisplayName(pair.subB)}</td>
                                    <td class="grade-cell">${pair.gtWinner}</td>
                                    <td class="grade-cell">${pair.gtGap}</td>
                                    ${appData.egfNames.map(egfName => html`
                                        <${ComparisonWinnerCell}
                                            winner=${pair.egfWinners[egfName]}
                                            gtWinner=${pair.gtWinner}
                                            hasGT=${pair.hasGT}
                                            key=${egfName}
                                        />
                                    `)}
                                </tr>
                            `)}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    `;
}

// ============ Grades Table Component ============

function GradeCell({ grade, gtGrade }) {
    let className = 'grade-cell';
    if (gtGrade !== null && gtGrade !== undefined && grade !== null && grade !== undefined) {
        const diff = Math.abs(grade - gtGrade);
        if (diff === 0) className += ' grade-match';
        else if (diff === 1) className += ' grade-close';
        else className += ' grade-far';
    }
    return html`<td class=${className}>${grade ?? '-'}</td>`;
}

function GradesTable({ onRowClick }) {
    const submissionIds = useMemo(() => Object.keys(appData.submissions), []);

    // Virtual scrolling configuration
    const ROW_HEIGHT = 41;  // pixels per row (including border)
    const CONTAINER_HEIGHT = 500;  // visible container height
    const BUFFER = 5;  // extra rows to render above/below viewport

    const [scrollTop, setScrollTop] = useState(0);
    const containerRef = { current: null };  // Manual ref since no useRef

    // Calculate visible range
    const totalHeight = submissionIds.length * ROW_HEIGHT;
    const visibleStart = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER);
    const visibleEnd = Math.min(
        submissionIds.length,
        Math.ceil((scrollTop + CONTAINER_HEIGHT) / ROW_HEIGHT) + BUFFER
    );
    const visibleIds = submissionIds.slice(visibleStart, visibleEnd);
    const offsetY = visibleStart * ROW_HEIGHT;

    const handleScroll = useCallback((e) => {
        setScrollTop(e.target.scrollTop);
    }, []);

    return html`
        <section class="grades-section">
            <h2>Grades by Submission</h2>
            <p style="color: var(--gray-600); font-size: 0.875rem; margin-bottom: 1rem;">
                Click a row to view details. Colors:
                <span style="color: var(--success);">exact match</span>,
                <span style="color: var(--warning);">within 1</span>,
                <span style="color: var(--danger);">2+ difference</span>
                <span style="margin-left: 1rem; color: var(--gray-400);">(${submissionIds.length} submissions)</span>
            </p>
            <div class="table-container virtual-table"
                 style="height: ${CONTAINER_HEIGHT}px; overflow-y: auto;"
                 onScroll=${handleScroll}>
                <div style="height: ${totalHeight}px; position: relative;">
                    <table class="grades-table" style="position: absolute; top: ${offsetY}px; width: 100%;">
                        <thead>
                            <tr>
                                <th>Student</th>
                                <th>GT</th>
                                ${appData.egfNames.map(name => html`
                                    <th title=${name} key=${name}>${appData.egfLabels[name] || name}</th>
                                `)}
                            </tr>
                        </thead>
                        <tbody>
                            ${visibleIds.map(sid => {
                                const sub = appData.submissions[sid];
                                const studentDisplay = sub.student_name || sub.student_id || sid;
                                const gtGrade = sub.ground_truth_grade;

                                return html`
                                    <tr class="grade-row" onClick=${() => onRowClick(sid)} key=${sid}>
                                        <td>${studentDisplay}</td>
                                        <td class="grade-cell">${gtGrade ?? '-'}</td>
                                        ${appData.egfNames.map(egfName => {
                                            const gradeDetail = appData.egfGrades[egfName]?.[sid];
                                            return html`<${GradeCell}
                                                grade=${gradeDetail?.grade}
                                                gtGrade=${gtGrade}
                                                key=${egfName}
                                            />`;
                                        })}
                                    </tr>
                                `;
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    `;
}

// ============ Main App ============

function App() {
    const [submissionModalOpen, setSubmissionModalOpen] = useState(false);
    const [selectedSubmission, setSelectedSubmission] = useState(null);
    const [comparisonModalOpen, setComparisonModalOpen] = useState(false);
    const [selectedComparison, setSelectedComparison] = useState(null);
    const [modalHistory, setModalHistory] = useState([]);
    // Track when all data is loaded to trigger final re-render
    const [allDataLoaded, setAllDataLoaded] = useState(false);

    // Listen for data complete to trigger re-render with full data
    useEffect(() => {
        const handleComplete = () => setAllDataLoaded(true);
        window.addEventListener('appDataComplete', handleComplete);
        return () => window.removeEventListener('appDataComplete', handleComplete);
    }, []);

    const openSubmissionModal = useCallback((submissionId, addToHistory = false) => {
        // If transitioning from another modal, save current state to history
        if (addToHistory && comparisonModalOpen && selectedComparison) {
            setModalHistory(prev => [...prev, { type: 'comparison', data: selectedComparison }]);
        }
        setComparisonModalOpen(false);
        setSelectedSubmission(submissionId);
        setSubmissionModalOpen(true);
        document.body.style.overflow = 'hidden';
    }, [comparisonModalOpen, selectedComparison]);

    const openComparisonModal = useCallback((pairData, addToHistory = false) => {
        // If transitioning from another modal, save current state to history
        if (addToHistory && submissionModalOpen && selectedSubmission) {
            setModalHistory(prev => [...prev, { type: 'submission', data: selectedSubmission }]);
        }
        setSubmissionModalOpen(false);
        setSelectedComparison(pairData);
        setComparisonModalOpen(true);
        document.body.style.overflow = 'hidden';
    }, [submissionModalOpen, selectedSubmission]);

    const closeAllModals = useCallback(() => {
        setSubmissionModalOpen(false);
        setComparisonModalOpen(false);
        setModalHistory([]);
        document.body.style.overflow = '';
    }, []);

    const handleBack = useCallback(() => {
        if (modalHistory.length === 0) return;

        const prev = modalHistory[modalHistory.length - 1];
        setModalHistory(h => h.slice(0, -1));

        if (prev.type === 'submission') {
            setComparisonModalOpen(false);
            setSelectedSubmission(prev.data);
            setSubmissionModalOpen(true);
        } else if (prev.type === 'comparison') {
            setSubmissionModalOpen(false);
            setSelectedComparison(prev.data);
            setComparisonModalOpen(true);
        }
    }, [modalHistory]);

    // Handler for clicking on essay from comparison modal
    const handleComparisonSubmissionClick = useCallback((submissionId) => {
        openSubmissionModal(submissionId, true);
    }, [openSubmissionModal]);

    // Handler for clicking a comparison from submission modal's comparisons tab
    const handleSubmissionComparisonClick = useCallback((comparison) => {
        // Build pairData from the comparison
        const subA = comparison.submission_a;
        const subB = comparison.submission_b;
        const gtA = appData.submissions[subA]?.ground_truth_grade;
        const gtB = appData.submissions[subB]?.ground_truth_grade;

        let gtWinner = '-';
        if (gtA != null && gtB != null) {
            if (gtA > gtB) gtWinner = 'A';
            else if (gtA < gtB) gtWinner = 'B';
        }

        // Find all comparisons for this pair across EGFs (one per EGF)
        const comparisons = [];
        const seenEGFs = new Set();
        for (const egfName of appData.egfNames) {
            if (seenEGFs.has(egfName)) continue;
            const egfComps = appData.egfComparisons?.[egfName] || {};
            let found = false;
            for (const subId of Object.keys(egfComps)) {
                if (found) break;
                for (const comp of egfComps[subId]) {
                    if ((comp.submission_a === subA && comp.submission_b === subB) ||
                        (comp.submission_a === subB && comp.submission_b === subA)) {
                        // Adjust winner if flipped
                        let winner = comp.winner?.toUpperCase() || '';
                        if (comp.submission_a === subB && (winner === 'A' || winner === 'B')) {
                            winner = winner === 'A' ? 'B' : 'A';
                        }
                        comparisons.push({
                            egfName,
                            comparison: {
                                ...comp,
                                submission_a: subA,
                                submission_b: subB,
                                winner
                            }
                        });
                        seenEGFs.add(egfName);
                        found = true;
                        break;  // Found for this EGF
                    }
                }
            }
        }

        const pairData = {
            subA,
            subB,
            gtA: gtA ?? 0,
            gtB: gtB ?? 0,
            gtWinner,
            gtGap: Math.abs((gtA ?? 0) - (gtB ?? 0)),
            comparisons
        };

        openComparisonModal(pairData, true);  // Add to history
    }, [openComparisonModal]);

    // Show message if no submissions loaded yet (shouldn't happen if init timing is correct)
    if (!appData.submissions || Object.keys(appData.submissions).length === 0) {
        return html`<div class="grades-section" style="text-align: center; padding: 2rem;">
            <p style="color: var(--gray-500);">No submissions found in data.</p>
        </div>`;
    }

    const canGoBack = modalHistory.length > 0;

    return html`
        <${GradesTable} onRowClick=${openSubmissionModal} />
        <${ComparisonsTable} onComparisonClick=${openComparisonModal} key=${allDataLoaded ? 'loaded' : 'loading'} />
        <${Modal}
            isOpen=${submissionModalOpen}
            onClose=${closeAllModals}
            submissionId=${selectedSubmission}
            onComparisonClick=${handleSubmissionComparisonClick}
            onBack=${handleBack}
            canGoBack=${canGoBack}
        />
        <${ComparisonPairModal}
            isOpen=${comparisonModalOpen}
            onClose=${closeAllModals}
            pairData=${selectedComparison}
            onSubmissionClick=${handleComparisonSubmissionClick}
            onBack=${handleBack}
            canGoBack=${canGoBack}
        />
    `;
}

// Initialize app - called when appData is ready
function initializeApp() {
    console.log('[initializeApp] Called, checking Preact globals...');
    // Ensure Preact globals are available (set by inline bundle)
    if (typeof render === 'undefined' || typeof html === 'undefined') {
        console.error('[initializeApp] Preact not loaded yet, retrying...');
        setTimeout(initializeApp, 50);
        return;
    }
    console.log('[initializeApp] Preact ready, rendering App...');
    const mountPoint = document.getElementById('grades-app-mount');
    console.log('[initializeApp] Mount point:', mountPoint ? 'found' : 'NOT FOUND');

    // Clear the mount point first to remove loading spinner
    // This ensures clean replacement when Preact renders
    if (mountPoint && !window._mountCleared) {
        mountPoint.innerHTML = '';
        window._mountCleared = true;
    }

    try {
        render(html`<${App} />`, mountPoint);
        console.log('[initializeApp] Render complete');
    } catch (err) {
        console.error('[initializeApp] Render failed:', err);
    }
}
'''
