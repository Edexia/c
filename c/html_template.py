"""Static HTML template structure for analysis reports."""

from pathlib import Path

# Cache for Preact bundle
_PREACT_BUNDLE_CACHE: str | None = None


def _load_preact_bundle() -> str:
    """Load and concatenate Preact UMD bundle from assets."""
    global _PREACT_BUNDLE_CACHE
    if _PREACT_BUNDLE_CACHE is not None:
        return _PREACT_BUNDLE_CACHE

    assets = Path(__file__).parent / "assets"
    scripts = []
    for name in ['preact.min.js', 'preact-hooks.min.js', 'htm.min.js']:
        path = assets / name
        if path.exists():
            scripts.append(path.read_text())

    _PREACT_BUNDLE_CACHE = '\n'.join(scripts)
    return _PREACT_BUNDLE_CACHE


def get_preact_script() -> str:
    """Get the inline Preact script tag with all dependencies."""
    bundle = _load_preact_bundle()
    return f'''<script>
{bundle}
// Setup globals for app
window.h = preact.h;
window.render = preact.render;
window.useState = preactHooks.useState;
window.useEffect = preactHooks.useEffect;
window.useCallback = preactHooks.useCallback;
window.useMemo = preactHooks.useMemo;
window.html = htm.bind(preact.h);
</script>'''

# Decompression code using native DecompressionStream API
DECOMPRESS_JS = '''
<script>
// Gzip decompression using native DecompressionStream API
async function decompressChunk(base64Data) {
    if (!base64Data) return {};
    try {
        // Decode base64 to bytes
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Decompress using native API
        const stream = new Blob([bytes]).stream();
        const decompressed = stream.pipeThrough(new DecompressionStream('gzip'));
        const text = await new Response(decompressed).text();
        return JSON.parse(text);
    } catch (e) {
        console.error('Decompression failed:', e);
        return {};
    }
}

// Progressive chunk loading
window.appData = {
    egfNames: [],
    egfLabels: {},
    submissions: {},
    egfGrades: {},
    egfComparisons: {},
    allLLMCalls: {},
    pdfs: {},
    maxGrade: 40,
    noiseAssumption: 'expected',
    _ready: { core: false, submissions: false, grades: false }
};

window.appDataLoadState = {
    chunksLoaded: 0,
    totalChunks: 6,
    errors: []
};

// Merge chunk into appData
function mergeChunk(chunkName, data) {
    if (!data || Object.keys(data).length === 0) return;

    switch(chunkName) {
        case 'core':
            Object.assign(window.appData, data);
            window.appData._ready.core = true;
            break;
        case 'submissions':
            window.appData.submissions = data;
            window.appData._ready.submissions = true;
            break;
        case 'grades':
            window.appData.egfGrades = data;
            window.appData._ready.grades = true;
            break;
        case 'comparisons':
            window.appData.egfComparisons = data;
            break;
        case 'llmCalls':
            window.appData.allLLMCalls = data;
            break;
    }

    window.appDataLoadState.chunksLoaded++;

    // Dispatch progress event
    window.dispatchEvent(new CustomEvent('appDataProgress', {
        detail: {
            chunk: chunkName,
            loaded: window.appDataLoadState.chunksLoaded,
            total: window.appDataLoadState.totalChunks,
            ready: window.appData._ready
        }
    }));
}

// Load all chunks progressively (PDFs are lazy-loaded separately)
async function loadAllChunks() {
    const chunkOrder = ['core', 'submissions', 'grades', 'comparisons', 'llmCalls'];

    // Count actual chunks that exist and have content
    let actualChunks = 0;
    for (const name of chunkOrder) {
        const el = document.getElementById('chunk-' + name);
        if (el && el.textContent.trim()) actualChunks++;
    }
    window.appDataLoadState.totalChunks = actualChunks;

    for (const chunkName of chunkOrder) {
        const el = document.getElementById('chunk-' + chunkName);
        if (el && el.textContent.trim()) {
            try {
                const data = await decompressChunk(el.textContent.trim());
                // Free memory: clear DOM element after reading
                el.textContent = '';
                el.remove();
                mergeChunk(chunkName, data);
            } catch (e) {
                console.error('Failed to load chunk:', chunkName, e);
                window.appDataLoadState.errors.push({ chunk: chunkName, error: e });
                // Still increment counter on error so loading completes
                window.appDataLoadState.chunksLoaded++;
            }
        }
        // Small yield to allow UI updates between chunks
        await new Promise(r => setTimeout(r, 0));
    }

    // Signal complete
    window.dispatchEvent(new CustomEvent('appDataComplete'));
}

// Lazy PDF loader - decompresses PDFs on-demand when user views them
window._pdfsLoaded = false;
async function getPdf(submissionId) {
    // Return cached if already loaded
    if (window.appData.pdfs[submissionId]) {
        return window.appData.pdfs[submissionId];
    }

    // First PDF request - decompress the entire chunk once
    if (!window._pdfsLoaded) {
        const el = document.getElementById('chunk-pdfs');
        if (el && el.textContent.trim()) {
            const data = await decompressChunk(el.textContent.trim());
            el.textContent = '';
            el.remove();
            window.appData.pdfs = data;
        }
        window._pdfsLoaded = true;
    }

    return window.appData.pdfs[submissionId] || null;
}
window.getPdf = getPdf;

// Start loading when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadAllChunks);
} else {
    loadAllChunks();
}
</script>
'''


def generate_html_shell(
    page_title: str,
    favicon_link: str,
    css: str,
    summary_comment: str,
    static_content: str,
    compressed_chunks: dict[str, str],
    app_js: str,
) -> str:
    """Generate the complete HTML document shell with compressed chunks.

    Args:
        page_title: Title for the browser tab
        favicon_link: HTML for favicon link tag
        css: Complete CSS styles
        summary_comment: HTML comment with summary markdown (for IDE viewing)
        static_content: Static HTML sections (header, charts, etc.)
        compressed_chunks: Dict of chunk_name -> gzip+base64 compressed data
        app_js: Preact application JavaScript code

    Returns:
        Complete HTML document as string

    Note: Data is split into compressed chunks for progressive loading:
    - core: metadata (tiny, loads first)
    - submissions: student data without PDFs
    - grades: grade data
    - comparisons: comparison data
    - llmCalls: LLM call data (heaviest)
    - pdfs: PDF data (also heavy)
    """
    # Build chunk script tags
    chunk_scripts = []
    chunk_order = ['core', 'submissions', 'grades', 'comparisons', 'llmCalls', 'pdfs']
    for name in chunk_order:
        data = compressed_chunks.get(name, '')
        if data:
            chunk_scripts.append(f'<script id="chunk-{name}" type="application/gzip+base64">{data}</script>')

    chunks_html = '\n    '.join(chunk_scripts)
    preact_script = get_preact_script()

    return f'''{summary_comment}<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    {favicon_link}
    <style>
{css}
    </style>
</head>
<body>
    <div class="container">
        {static_content}

        <!-- Preact App Mount Point - shows loading state until data is ready -->
        <div id="grades-app-mount">
            <div class="grades-section" style="text-align: center; padding: 3rem;">
                <div class="loading-spinner"></div>
                <p id="loading-status" style="color: var(--gray-500); margin-top: 1rem;">Loading data...</p>
                <div id="loading-progress" style="width: 200px; height: 4px; background: var(--gray-200); border-radius: 2px; margin: 1rem auto;">
                    <div id="loading-bar" style="width: 0%; height: 100%; background: var(--primary); border-radius: 2px; transition: width 0.3s;"></div>
                </div>
            </div>
        </div>

        <footer>
            Generated by c - EGF Analysis CLI
        </footer>
    </div>

    <!-- Decompression and chunk loading code -->
    {DECOMPRESS_JS}

    <!-- Preact + htm (inlined for offline support and fast TTFP) -->
    {preact_script}

    <!-- Preact App (initializes when core data is ready) -->
    <script>
{app_js}
    </script>

    <!-- Compressed data chunks (gzip + base64) - loaded progressively -->
    {chunks_html}
</body>
</html>'''
