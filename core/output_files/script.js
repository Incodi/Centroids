/*
drawing of thing i (script file) analyze
                                                                           x  x   x  x x                     
                                                              xx xx xxxx x                x  x x             
                                                          xxxx                                   x xx        
                                                        xx                                           xxx     
                  x xxxxxxxxx xxxxxxxx                 xx                                               xxx  
              xxxx                    xxx             x     x         x                                   xx 
           xxx                           xxx          x      xx      xx   xxxxxxx          xxxxxxxx        x 
       xxxx               xxxxx             xxx       x        xx  xx    x      xx        x        x       x 
      xx                  xxxxx               xxx     x         xxx     xx    xx  xx      x        x       x 
    xx                    xxxxx        x x x x        xx        x        xxxxxx     xxxxxxx    xxxxx        x
   xx                             x xx                xxx     xx                          xxxxxx            x
  x                         xxxxx                   xx      xxx                           x                x 
 x                        xx                      xxx       x                            x               xx  
x                      xxx                      xxx                                     x             xxx    
x                     xx                       xxxx xxxxxxxxxx xxxxxx                             xx x       
x                     x                                             xxxxxxx              xxxx xx x           
x                      xx                                                  xxxxxxxxxxxxxx                    
xx                       xxx xx x                                                                            
 xx                              x xxx xxxxxxxxx                                                          
   xx                                           xx                                                         
     xx                                           x                                                         
       xxx                                     xxx                                                         
           xxxxxxx                    xxx xxxxx                                                            
                  x xx x xx x x xxxxxx                                                                       
*/

const CONFIG = {
    highlightSize: 80, normalSize: 15, focusStarSize: 40,
    dimmedOpacity: 0.2, highlightDuration: 2000, textOutlineDelay: 100,
    pointFontSize: 7, colorScaleZoomRatio: 0.25, regressionLinePoints: 100,
    defaultColorScale: 'rdbu_r', defaultFocusText: __FOCUS_TEXT__
};

const metricsData = __METRICS_JSON__;
const chartTitles = __TITLES_JSON__;
const textNames = __TEXTS_JSON__;
const metricDisplayNames = __METRIC_NAMES_JSON__;
const metricColorScales = __METRIC_COLORScales_JSON__;
const metricKeys = __METRIC_KEYS_JSON__;
const hoverPanelPalettes = __HOVER_PANEL_PALETTES_JSON__;
const hoverTextPalettes = __HOVER_TEXT_PALETTES_JSON__;
const semanticEmbeddings = __SEMANTIC_EMBEDDINGS_JSON__;
const burrowsSimilarity = __BURROWS_SIMILARITY_JSON__;
const fightingWordsUnigrams = __FIGHTING_WORDS_UNIGRAMS_JSON__;
const burrowsReduced = __BURROWS_REDUCED_JSON__;
const burrowsFunctionWords = __BURROWS_FUNCTION_WORDS_JSON__;
const burrowsWordFrequencies = __BURROWS_WORD_FREQUENCIES_JSON__;
const CAPITAL_CITIES = __CAPITAL_CITIES_JSON__;

const CLUSTER_TURBO_COLORSCALE = [
    [0.0, '#30123b'], [0.08, '#4145ab'], [0.16, '#4675ed'], [0.24, '#39a2fc'],
    [0.32, '#1bcfd4'], [0.40, '#24eca6'], [0.48, '#61fc6c'], [0.56, '#a4fc3b'],
    [0.64, '#d1e834'], [0.72, '#f3c63a'], [0.80, '#fe9b2d'], [0.88, '#f36315'],
    [0.96, '#d93806'], [1.0, '#7a0403']
];

const FOCUS_OVERLAY_META = 'focusOverlay';
const CORR_FOCUS_OVERLAY_META = 'corrFocusOverlay';
const CONNECTION_META = 'semanticConnection';
const PLOT_CONFIG = { responsive: true };
const GRID_STYLE = { showgrid: true, gridwidth: 1, gridcolor: '#e0e0e0' };
const FOCUS_CLUSTER_COLOR = '#ffeb3b';
const FOCUS_CLUSTER_LINE_COLOR = '#b8860b';
const EVIL_CLUSTER_COLOR = 'rgba(168, 85, 247, 0.30)';
const EVIL_CLUSTER_LINE_COLOR = 'rgba(168, 85, 247, 0.40)';
const MAIN_PLOT_RIGHT_RESERVE_PX = 230;
const HIST_FILL = '#85EDFF80';
const HIST_LINE = '#85EDFF';
const NORMAL_CURVE_COLOR = '#FFBCD9';
const FOCUS_AREA_FILL = 'rgba(255, 212, 0, 0.20)';
const WORLD_CAPITAL_OUTLINE = '#ffd400';

const CLUSTER_BAR_POS_GRADIENT = 'linear-gradient(90deg, #fbd0d0, #E85A93)';
const CLUSTER_BAR_NEG_GRADIENT = 'linear-gradient(90deg, #d0d6fb, #2FA3E6)';
const CLUSTER_BAR_TRACK_COLOR = '#f0f0f0';
const CLUSTER_BAR_TEXT_POS_COLOR = '#E85A93';
const CLUSTER_BAR_TEXT_NEG_COLOR = '#2FA3E6';

const VIEW_ELEMENT_IDS = {
    correlation: ['correlationContainer', 'correlationControls', 'correlationStats'],
    functionWord: ['functionWordContainer', 'functionWordControls', 'functionWordStats'],
    delta: ['deltaContainer', 'deltaControls', 'deltaStats']
};

const DELTA_SORT_KEYS = {
    separators: 'deltaContribution', drivers: 'driverScore', shared: 'sharedScore'
};
const DELTA_SORT_LABELS = {
    separators: 'biggest separators', drivers: 'similarity drivers', shared: 'shared tendency'
};

let mainPlot = null;
let originalPlotX = null;
let originalPlotY = null;
let originalAxes = null;
let mainPlotResizeBound = false;
let mainPlotResizeTimer = null;
let baseHoverTexts = null;

let currentMetric = __ACTIVE_METRIC_JSON__;
let currentColorDataForPlot = [];
let currentColorscale = CLUSTER_TURBO_COLORSCALE;
let currentCmin = 0;
let currentCmax = 1;

let viewMode = 'umap';
let mapMode = 'semantic';
let umapLabelsVisible = true;
let correlationLabelsVisible = true;
let hideGrayPoints = false;

let focusedTextName = null;
let focusedTextIndex = -1;
let focusClusterOnly = false;
let focusClusterMetric = null;
let focusClusterMetricUserSelected = false;
let focusSimilarityMode = 'cosine';
let evilFocusTextIndex = -1;
let evilFocusTextName = null;

const clusterStatsState = {
    view: 'word',
    source: null,
    clusterId: null,
    clusterIdUserSelected: false,
    compareMode: 'corpus',
    compareClusterId: null,
    sortMode: 'share',
    signFilter: 'both',
    inspectWord: null
};

let semanticLineModeEnabled = false;
let semanticLineCount = 10;
let mainConnectionToken = 0;
let correlationConnectionToken = 0;

let correlationXMetric = metricKeys[0];
let correlationYMetric = metricKeys.length > 1 ? metricKeys[1] : metricKeys[0];

const functionWordSelection = new Set(
    burrowsFunctionWords && burrowsFunctionWords.length ? [burrowsFunctionWords[0]] : ['the']
);
let functionWordShowHistogram = true;
let functionWordShowMarkers = true;
let functionWordPlotMode = 'zscore';

const deltaState = {
    textA: null, textB: null, sortMode: 'separators', topN: 25,
    wordStats: null, deltaDistribution: null, totalWordCount: 0, initialized: false
};

const clusterMetricKeys = [];
const textLookup = {};
const metricCatalog = [];

const isNum = v => v !== null && v !== undefined && !isNaN(v);
const asNum = v => isNum(v) ? Number(v) : null;
const mean = arr => {
    if (!arr || arr.length === 0) return null;
    let total = 0;
    for (let i = 0; i < arr.length; i++) total += arr[i];
    return total / arr.length;
};
const variance = arr => {
    if (!arr || arr.length === 0) return null;
    const avg = mean(arr);
    let total = 0;
    for (let i = 0; i < arr.length; i++) total += (arr[i] - avg) * (arr[i] - avg);
    return total / arr.length;
};
const stddev = arr => {
    const v = variance(arr);
    return v === null ? null : Math.sqrt(v);
};
const median = sorted => {
    if (!sorted || sorted.length === 0) return null;
    const n = sorted.length;
    if (n % 2 === 1) return sorted[(n - 1) / 2];
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
};
const ordinal = n => {
    const v = Math.round(n);
    const last2 = v % 100;
    if (last2 >= 11 && last2 <= 13) return v + 'th';
    const last = v % 10;
    if (last === 1) return v + 'st';
    if (last === 2) return v + 'nd';
    if (last === 3) return v + 'rd';
    return v + 'th';
};
const fmt = (v, digits) => {
    if (digits === undefined) digits = 4;
    if (!isNum(v)) return 'N/A';
    const s = Number(v).toFixed(digits).replace(/\.?0+$/, '');
    return s === '-0' ? '0' : s;
};

const fmtDelta = (v, digits = 4) => {
    if (!isNum(v)) return '0';
    const s = fmt(v, digits);
    if (s === '0') return '0';
    return (v > 0 ? '+' : '') + s;
};

const makeDiv = (text, mb) => {
    const d = document.createElement('div');
    d.textContent = text;
    if (mb) d.style.marginBottom = mb;
    return d;
};
const clearBody = panelId => {
    const body = document.querySelector('#' + panelId + ' .panelBody');
    if (!body) return null;
    body.innerHTML = '';
    return body;
};
const fwStats = title => {
    const c = document.getElementById('functionWordStatsContent');
    if (!c) return null;
    c.innerHTML = '';
    if (title) {
        const h = document.createElement('strong');
        h.textContent = title;
        c.appendChild(h);
    }
    return c;
};
const findTrace = (plot, meta) => {
    for (let i = 0; i < plot.data.length; i++) {
        if (plot.data[i].meta === meta) return i;
    }
    return -1;
};
const listen = (id, evt, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(evt, fn);
};
const addOutlines = (selector, cls) => {
    if (!selector) selector = '.scatterlayer .textpoint text';
    if (!cls) cls = 'plotly-stroke-text';
    const els = document.querySelectorAll(selector);
    for (let i = 0; i < els.length; i++) els[i].classList.add(cls);
};
const countLE = (sorted, v) => {
    let lo = 0, hi = sorted.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    return lo;
};
const empPct = (sorted, v) => {
    if (!sorted || !sorted.length) return 0;
    return (countLE(sorted, v) / sorted.length) * 100;
};
const normalPdf = (x, m, s) => {
    if (s <= 0) return 0;
    const z = (x - m) / s;
    return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
};
const normalCdf = z => {
    const sign = z < 0 ? -1 : 1;
    const x = Math.abs(z) / Math.SQRT2;
    const t = 1 / (1 + 0.3275911 * x);
    const poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592;
    const erf = 1 - poly * t * Math.exp(-x * x);
    return 0.5 * (1 + sign * erf);
};
const pearson = (xs, ys) => {
    const px = [], py = [];
    for (let i = 0; i < xs.length; i++) {
        if (isNum(xs[i]) && isNum(ys[i])) { px.push(xs[i]); py.push(ys[i]); }
    }
    if (px.length < 2) return null;
    const mx = mean(px), my = mean(py);
    let num = 0, sxx = 0, syy = 0;
    for (let j = 0; j < px.length; j++) {
        num += (px[j] - mx) * (py[j] - my);
        sxx += (px[j] - mx) * (px[j] - mx);
        syy += (py[j] - my) * (py[j] - my);
    }
    if (sxx === 0 || syy === 0) return null;
    return num / Math.sqrt(sxx * syy);
};
const linearReg = (xs, ys) => {
    const n = xs.length;
    let sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (let i = 0; i < n; i++) {
        sx += xs[i]; sy += ys[i];
        sxy += xs[i] * ys[i]; sx2 += xs[i] * xs[i];
    }
    const denom = n * sx2 - sx * sx;
    const slope = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
    const intercept = n === 0 ? 0 : (sy - slope * sx) / n;
    return { slope, intercept };
};
const polyReg = (xs, ys) => {
    const n = xs.length;
    let sx = 0, sy = 0, sx2 = 0, sx3 = 0, sx4 = 0, sxy = 0, sx2y = 0;
    for (let i = 0; i < n; i++) {
        const x = xs[i], y = ys[i];
        sx += x; sy += y;
        sx2 += x * x; sx3 += x * x * x; sx4 += x * x * x * x;
        sxy += x * y; sx2y += x * x * y;
    }
    const M = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]];
    const V = [sy, sxy, sx2y];
    for (let a = 0; a < 3; a++) {
        let p = a;
        for (let b = a + 1; b < 3; b++) if (Math.abs(M[b][a]) > Math.abs(M[p][a])) p = b;
        const tmp = M[a]; M[a] = M[p]; M[p] = tmp;
        const tv = V[a]; V[a] = V[p]; V[p] = tv;
        for (let c = a + 1; c < 3; c++) {
            const f = M[c][a] / M[a][a];
            for (let d = a; d < 3; d++) M[c][d] -= f * M[a][d];
            V[c] -= f * V[a];
        }
    }
    const co = [0, 0, 0];
    for (let e = 2; e >= 0; e--) {
        co[e] = V[e];
        for (let f2 = e + 1; f2 < 3; f2++) co[e] -= M[e][f2] * co[f2];
        co[e] /= M[e][e];
    }
    return { a: co[2], b: co[1], c: co[0] };
};
const logisticReg = (xs, ys) => {
    const lo = Math.min.apply(null, ys);
    const hi = Math.max.apply(null, ys);
    if (hi === lo) return null;
    const n = xs.length;
    const norm = [];
    for (let i = 0; i < n; i++) {
        let p = (ys[i] - lo) / (hi - lo);
        if (p <= 0) p = 1e-6;
        if (p >= 1) p = 1 - 1e-6;
        norm.push(p);
    }
    let alpha = 0, beta = 0;
    for (let iter = 0; iter < 50; iter++) {
        let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
        for (let j = 0; j < n; j++) {
            const x = xs[j];
            const target = norm[j];
            const eta = alpha + beta * x;
            const mu = 1 / (1 + Math.exp(-eta));
            const w = Math.max(mu * (1 - mu), 1e-9);
            const r = target - mu;
            g0 += r;
            g1 += r * x;
            h00 += w;
            h01 += w * x;
            h11 += w * x * x;
        }
        const det = h00 * h11 - h01 * h01;
        if (Math.abs(det) < 1e-12) break;
        const dAlpha = (g0 * h11 - g1 * h01) / det;
        const dBeta = (g1 * h00 - g0 * h01) / det;
        alpha += dAlpha;
        beta += dBeta;
        if (Math.abs(dAlpha) < 1e-8 && Math.abs(dBeta) < 1e-8) break;
    }
    return { alpha, beta, minY: lo, maxY: hi };
};
const regModel = (xs, ys, type) => {
    if (!xs || xs.length < 2) return null;
    if (type === 'linear') return linearReg(xs, ys);
    if (type === 'polynomial') return polyReg(xs, ys);
    if (type === 'logistic') return logisticReg(xs, ys);
    return null;
};
const predictY = (x, type, model) => {
    if (!model) return null;
    if (type === 'linear') return model.slope * x + model.intercept;
    if (type === 'polynomial') return model.a * x * x + model.b * x + model.c;
    if (type === 'logistic') {
        const n = 1 / (1 + Math.exp(-(model.alpha + model.beta * x)));
        return n * (model.maxY - model.minY) + model.minY;
    }
    return null;
};
const regLine = (xs, ys, type, model) => {
    if (!model || xs.length < 2) return null;
    const lo = Math.min.apply(null, xs), hi = Math.max.apply(null, xs);
    const steps = CONFIG.regressionLinePoints;
    const lx = [], ly = [];
    for (let i = 0; i <= steps; i++) {
        const x = lo + ((hi - lo) * i) / steps;
        lx.push(x);
        ly.push(predictY(x, type, model));
    }
    return { x: lx, y: ly };
};
const residualRank = (texts, xs, ys, type, model) => {
    if (!model) return [];
    const out = [];
    for (let i = 0; i < xs.length; i++) {
        const p = predictY(xs[i], type, model);
        if (p === null || isNaN(p)) continue;
        const r = ys[i] - p;
        out.push({ text: texts[i], residual: r, absResidual: Math.abs(r) });
    }
    out.sort((a, b) => b.absResidual - a.absResidual);
    return out;
};
const sampleCurve = (minX, maxX, steps, fn) => {
    const xs = [], ys = [];
    for (let i = 0; i <= steps; i++) {
        const x = minX + ((maxX - minX) * i) / steps;
        xs.push(x);
        ys.push(fn(x));
    }
    return { x: xs, y: ys };
};
const decodeSvg = url => {
    if (!url) return null;
    const comma = url.indexOf(',');
    if (comma < 0) return null;
    const header = url.slice(0, comma);
    const payload = url.slice(comma + 1);
    if (/;base64/i.test(header)) {
        const bin = atob(payload);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return new TextDecoder('utf-8').decode(bytes);
    }
    return decodeURIComponent(payload);
};
const injectOutlines = svg => {
    return svg.replace(/<text(\b[^>]*)>/g, (full, attrs) => {
        let selfClose = '';
        let a = attrs;
        if (/\s*\/$/.test(a)) { selfClose = '/'; a = a.replace(/\s*\/$/, ''); }
        let existingStyle = '';
        a = a.replace(/\s+style\s*=\s*"([^"]*)"/i, (m, c) => { existingStyle = c; return ''; });
        const strip = s => {
            return s.replace(/(?:^|;)\s*stroke\s*:[^;]*/gi, '')
                    .replace(/(?:^|;)\s*stroke-width\s*:[^;]*/gi, '')
                    .replace(/(?:^|;)\s*stroke-linejoin\s*:[^;]*/gi, '')
                    .replace(/(?:^|;)\s*stroke-linecap\s*:[^;]*/gi, '')
                    .replace(/(?:^|;)\s*paint-order\s*:[^;]*/gi, '')
                    .replace(/^\s*;+/, '').replace(/;+\s*$/, '').trim();
        };
        a = a.replace(/\s+stroke\s*=\s*"[^"]*"/gi, '')
             .replace(/\s+stroke-width\s*=\s*"[^"]*"/gi, '')
             .replace(/\s+stroke-linejoin\s*=\s*"[^"]*"/gi, '')
             .replace(/\s+stroke-linecap\s*=\s*"[^"]*"/gi, '')
             .replace(/\s+paint-order\s*=\s*"[^"]*"/gi, '');
        const clean = strip(existingStyle);
        let fs = null;
        const sm = clean.match(/font-size\s*:\s*(\d+(?:\.\d+)?)/);
        if (sm) fs = parseFloat(sm[1]);
        if (fs === null) {
            const am = a.match(/font-size\s*=\s*"(\d+(?:\.\d+)?)"/);
            if (am) fs = parseFloat(am[1]);
        }
        const sw = fs !== null ? Math.max(2, fs / 4.5).toFixed(1) : '3';
        const st = 'stroke: white; stroke-width: ' + sw + '; paint-order: stroke; stroke-linejoin: round; stroke-linecap: round;';
        const finalStyle = clean ? clean + '; ' + st : st;
        return '<text' + a + ' style="' + finalStyle + '"' +
               ' stroke="white" stroke-width="' + sw + '" stroke-linejoin="round" stroke-linecap="round"' + selfClose + '>';
    });
};
const svgToPng = (svg, width, height) => {
    return new Promise((resolve, reject) => {
        const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
            const c = document.createElement('canvas');
            c.width = width; c.height = height;
            const ctx = c.getContext('2d');
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, width, height);
            ctx.drawImage(img, 0, 0, width, height);
            URL.revokeObjectURL(url);
            resolve(c.toDataURL('image/png'));
        };
        img.onerror = e => { URL.revokeObjectURL(url); reject(e); };
        img.src = url;
    });
};

const isBurrows = key => {
    if (!key) return false;
    const s = (key + ' ' + (metricDisplayNames[key] || '')).toLowerCase();
    return s.indexOf('burrows') !== -1;
}; /* hi :) */
const getAlgo = key => {
    if (!key) return '';
    const s = (key + ' ' + (metricDisplayNames[key] || '')).toLowerCase();
    if (s.indexOf('leiden') !== -1) return 'Leiden';
    if (s.indexOf('louvain') !== -1) return 'Louvain';
    if (s.indexOf('kmeans') !== -1 || s.indexOf('k-means') !== -1) return 'k-means';
    if (s.indexOf('hdbscan') !== -1) return 'HDBSCAN';
    if (s.indexOf('burrows') !== -1) return 'Leiden';
    return '';
};
const isClusterKey = key => {
    if (!key) return false;
    if (key === 'cluster' || key.indexOf('cluster_') === 0) return true;
    const s = (key + ' ' + (metricDisplayNames[key] || '')).toLowerCase();
    if (s.indexOf('burrows') === -1) return false;
    const kw = ['cluster', 'leiden', 'louvain', 'kmeans', 'k-means', 'hdbscan'];
    for (let i = 0; i < kw.length; i++) {
        if (s.indexOf(kw[i]) !== -1) return true;
    }
    return false;
};
const clusterLabel = key => {
    if (!key) return '';
    return metricDisplayNames[key] || key;
};
const clusterValueLabel = (key, id) => {
    const algo = getAlgo(key) || 'Leiden';
    if (id < 0) return 'unclustered';
    return isBurrows(key) ? 'Burrows ' + algo + ' cluster ' + id : algo + ' cluster ' + id;
};
const colorbarTitle = key => {
    if (isClusterKey(key)) return clusterLabel(key);
    return metricDisplayNames[key] || key;
};
const preferredCluster = mode => {
    const b = [], reg = [];
    for (let i = 0; i < clusterMetricKeys.length; i++) {
        if (isBurrows(clusterMetricKeys[i])) b.push(clusterMetricKeys[i]);
        else reg.push(clusterMetricKeys[i]);
    }
    if (mode === 'burrows' && b.length) {
        for (let j = 0; j < b.length; j++) if (b[j] === 'burrows_cluster') return 'burrows_cluster';
        return b[0];
    }
    if (reg.length) {
        for (let k = 0; k < reg.length; k++) if (reg[k] === 'cluster') return 'cluster';
        return reg[0];
    }
    return clusterMetricKeys[0] || null;
};
const metricRange = values => {
    let lo = Infinity, hi = -Infinity, ok = false;
    for (let i = 0; i < values.length; i++) {
        if (!isNum(values[i])) continue;
        if (values[i] < lo) lo = values[i];
        if (values[i] > hi) hi = values[i];
        ok = true;
    }
    return ok ? { min: lo, max: hi } : null;
};
const colorData = key => {
    const raw = metricsData[key];
    if (!raw) return null;
    const b = isBurrows(key), c = isClusterKey(key);
    const valid = [];
    for (let i = 0; i < raw.length; i++) if (isNum(raw[i])) valid.push(raw[i]);
    const out = [];
    if (b) {
        if (!valid.length) { for (let j = 0; j < raw.length; j++) out.push(NaN); return out; }
        if (c) {
            for (let k = 0; k < raw.length; k++) out.push(isNum(raw[k]) ? raw[k] : -1);
            return out;
        }
        let avg = 0;
        for (let m = 0; m < valid.length; m++) avg += valid[m];
        avg /= valid.length;
        for (let n = 0; n < raw.length; n++) out.push(isNum(raw[n]) ? raw[n] : avg);
        return out;
    }
    for (let p = 0; p < raw.length; p++) out.push(isNum(raw[p]) ? raw[p] : NaN);
    return out;
};
const stripBurrowsClusterLine = line => {
    return !/burrow[s]?_cluster/i.test(line);
};
const stripBurrowsClusterFromHoverString = s => {
    if (typeof s !== 'string') return s;
    if (!/burrow[s]?_cluster/i.test(s)) return s;
    const parts = s.split(/<br\s*\/?>|\n/i);
    const kept = [];
    for (let i = 0; i < parts.length; i++) {
        if (stripBurrowsClusterLine(parts[i])) kept.push(parts[i]);
    }
    return kept.join('<br>');
};
const sanitizeMainHoverText = () => {
    if (!mainPlot || !mainPlot.data) return;
    for (let i = 0; i < mainPlot.data.length; i++) {
        const tr = mainPlot.data[i];
        if (Array.isArray(tr.hovertext)) {
            let changed = false;
            const ht = [];
            for (let j = 0; j < tr.hovertext.length; j++) {
                const v = stripBurrowsClusterFromHoverString(tr.hovertext[j]);
                if (v !== tr.hovertext[j]) changed = true;
                ht.push(v);
            }
            if (changed) Plotly.restyle(mainPlot, { hovertext: [ht] }, [i]);
        }
        if (typeof tr.hovertemplate === 'string') {
            const tmpl = stripBurrowsClusterFromHoverString(tr.hovertemplate);
            if (tmpl !== tr.hovertemplate) Plotly.restyle(mainPlot, { hovertemplate: [tmpl] }, [i]);
        }
    }
};
const hoverPointColor = pointIndex => {
    const traces = mainPlot.querySelectorAll('.scatterlayer .trace');
    const points = traces.length ? traces[0].querySelectorAll('.point') : [];
    const point = points[pointIndex];
    if (point) {
        const computed = window.getComputedStyle(point);
        const fill = computed.fill || point.style.fill;
        if (fill && fill !== 'none') return fill;
    }
    return '#555555';
};
const addHoverGradient = (colors, index) => {
    const svg = mainPlot && mainPlot.querySelector('svg');
    if (!svg || !colors || colors.length < 2) return null;
    let defs = svg.querySelector('defs');
    if (!defs) {
        defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        svg.insertBefore(defs, svg.firstChild);
    }
    const id = 'atlas-hover-gradient-' + index;
    const old = document.getElementById(id);
    if (old) old.remove();
    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient.setAttribute('id', id);
    gradient.setAttribute('x1', '0');
    gradient.setAttribute('y1', '0');
    gradient.setAttribute('x2', '0');
    gradient.setAttribute('y2', '1');
    for (let i = 0; i < colors.length; i++) {
        const stop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop.setAttribute('offset', ((i / (colors.length - 1)) * 100) + '%');
        stop.setAttribute('stop-color', colors[i]);
        gradient.appendChild(stop);
    }
    defs.appendChild(gradient);
    return 'url(#' + id + ')';
};
const styleMainHover = pointIndex => {
    const panelColors = hoverPanelPalettes[pointIndex];
    const textColors = hoverTextPalettes[pointIndex];
    if ((!panelColors || panelColors.length < 2) &&
        (!textColors || textColors.length < 2)) return;
    const hoverGroups = mainPlot.querySelectorAll('.hoverlayer .hovertext');
    if (!hoverGroups.length) return;

    const gradient = addHoverGradient(panelColors, pointIndex);
    const panelOutline = hoverPointColor(pointIndex);
    const textOutline = textColors && textColors.length ? textColors[0] : null;
    const textColor = textColors && textColors.length > 1 ? textColors[1] : null;

    for (let i = 0; i < hoverGroups.length; i++) {
        const group = hoverGroups[i];
        const background = group.querySelector('path, rect');
        if (background) {
            if (gradient) {
                background.setAttribute('fill', gradient);
                background.style.setProperty('fill', gradient, 'important');
            }
            background.setAttribute('stroke', panelOutline);
            background.setAttribute('stroke-width', '6');
            background.setAttribute('stroke-linejoin', 'round');
            background.setAttribute('paint-order', 'stroke');
            background.style.setProperty('stroke', panelOutline, 'important');
            background.style.setProperty('stroke-width', '6px', 'important');
            background.style.setProperty('paint-order', 'stroke', 'important');
        }
        const labels = group.querySelectorAll('text, tspan');
        for (let j = 0; j < labels.length; j++) {
            if (textColor) {
                labels[j].setAttribute('fill', textColor);
                labels[j].style.setProperty('fill', textColor, 'important');
                labels[j].style.setProperty('color', textColor, 'important');
            }
            if (textOutline) {
                labels[j].setAttribute('stroke', textOutline);
                labels[j].setAttribute('stroke-width', '2');
                labels[j].setAttribute('paint-order', 'stroke');
                labels[j].setAttribute('stroke-linejoin', 'round');
                labels[j].style.setProperty('stroke', textOutline, 'important');
                labels[j].style.setProperty('stroke-width', '2px', 'important');
                labels[j].style.setProperty('paint-order', 'stroke', 'important');
            }
        }
    }
};
const setupHoverStyling = () => {
    if (!mainPlot || typeof mainPlot.on !== 'function') return;
    if (typeof MutationObserver === 'undefined') return;

    let lastHoverPoint = -1;
    let hoverlayer = null;
    let observer = null;
    const OBS_CONFIG = {
        childList: true, attributes: true, characterData: true, subtree: true
    };

    const applyStyles = () => {
        if (lastHoverPoint < 0 || !hoverlayer) return;
        if (observer) observer.disconnect();
        styleMainHover(lastHoverPoint);
        if (observer && hoverlayer && hoverlayer.isConnected) {
            observer.observe(hoverlayer, OBS_CONFIG);
        }
    };

    const bind = () => {
        const hl = mainPlot.querySelector('.hoverlayer');
        if (!hl) {
            setTimeout(bind, 200);
            return;
        }
        hoverlayer = hl;
        if (!observer) {
            observer = new MutationObserver(() => {
                if (lastHoverPoint < 0) return;
                applyStyles();
            });
        }
        observer.disconnect();
        observer.observe(hoverlayer, OBS_CONFIG);
    };

    bind();

    mainPlot.on('plotly_hover', event => {
        if (!event || !event.points || !event.points.length) return;
        lastHoverPoint = event.points[0].pointNumber;
        const current = mainPlot.querySelector('.hoverlayer');
        if (current && current !== hoverlayer) {
            hoverlayer = current;
            bind();
        }
        requestAnimationFrame(() => {
            requestAnimationFrame(applyStyles);
        });
    });

    mainPlot.on('plotly_unhover', () => {
        lastHoverPoint = -1;
    });
};

const makeDraggable = (panel, handle) => {
    let dragging = false, sx = 0, sy = 0, sl = 0, st = 0;
    const move = e => {
        if (!dragging) return;
        panel.style.left = (sl + e.clientX - sx) + 'px';
        panel.style.top = (st + e.clientY - sy) + 'px';
    };
    const up = () => {
        dragging = false;
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
    };
    handle.addEventListener('mousedown', e => {
        const tag = (e.target.tagName || '').toUpperCase();
        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'BUTTON' || tag === 'TEXTAREA' || tag === 'A') return;
        e.preventDefault();
        e.stopPropagation();
        const rect = panel.getBoundingClientRect();
        panel.style.left = rect.left + 'px';
        panel.style.top = rect.top + 'px';
        panel.style.right = 'auto';
        panel.style.bottom = 'auto';
        sx = e.clientX; sy = e.clientY;
        sl = rect.left; st = rect.top;
        dragging = true;
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
    });
};
const addCollapse = (panel, handle) => {
    const btn = document.createElement('button');
    btn.className = 'panelCollapseBtn';
    btn.textContent = '-';
    btn.title = 'Collapse or expand this panel';
    let savedHeight = '';
    let savedBottom = ''; // i remembered your bottom
    btn.addEventListener('click', () => {
        const body = panel.querySelector('.panelBody');
        const collapsing = body.style.display !== 'none';
        if (collapsing) {
            savedHeight = panel.style.height;
            savedBottom = panel.style.bottom;
            panel.style.height = 'auto';
            panel.style.bottom = 'auto';
            body.style.display = 'none';
        } else {
            body.style.display = '';
            panel.style.height = savedHeight;
            panel.style.bottom = savedBottom;
        }
        btn.textContent = collapsing ? '+' : '-';
    });
    handle.appendChild(btn);
};
const makePanel = (id, pos, border, content, opts) => {
    if (!opts) opts = {};
    const resizable = opts.resizable !== false;
    const panel = document.createElement('div');
    panel.id = id;
    const display = pos.display;
    const style = {
        position: 'fixed', background: 'white', padding: '5px',
        border: border || '0.5px solid black',
        zIndex: String(opts.zIndex || 9999),
        resize: resizable ? 'both' : 'none',
        overflow: resizable ? 'auto' : 'visible',
        minWidth: opts.minWidth || '120px', minHeight: '30px',
        maxHeight: '90vh', boxSizing: 'border-box'
    };
    for (const k in pos) {
        if (k !== 'display') style[k] = pos[k];
    }
    for (const s in style) panel.style[s] = style[s];
    const handle = document.createElement('div');
    handle.className = 'panelDragHandle';
    handle.textContent = opts.dragLabel || '...';
    panel.appendChild(handle);
    const body = document.createElement('div');
    body.className = 'panelBody';
    body.innerHTML = content || '';
    panel.appendChild(body);
    if (display) panel.style.display = display;
    document.body.appendChild(panel);
    makeDraggable(panel, handle);
    if (resizable) addCollapse(panel, handle);
    return panel;
};
const makeFullView = (containerId, plotId) => {
    const c = document.createElement('div');
    c.id = containerId;
    c.className = 'fullView';
    c.innerHTML = '<div id="' + plotId + '" class="fullViewPlot"></div>';
    document.body.appendChild(c);
    return c;
};
const drawPlot = (plotId, traces, layout, fixedHeight) => {
    const el = document.getElementById(plotId);
    el.style.height = fixedHeight ? (fixedHeight + 'px') : '';
    layout.autosize = true;
    return Plotly.newPlot(plotId, traces, layout, PLOT_CONFIG);
};
const emptyPlot = (plotId, title) => {
    return drawPlot(plotId, [], { title: title }, null);
};

const snapshotXY = () => {
    if (originalPlotX && originalPlotY && originalPlotX.length > 0) return true;
    const traces = [];
    if (mainPlot._fullData && mainPlot._fullData[0]) traces.push(mainPlot._fullData[0]);
    if (mainPlot.data && mainPlot.data[0]) traces.push(mainPlot.data[0]);
    for (let i = 0; i < traces.length; i++) {
        const t = traces[i];
        if (!t || !t.x || !t.y) continue;
        if (t.x.length > 0 && t.x.length === t.y.length) {
            originalPlotX = Array.prototype.slice.call(t.x);
            originalPlotY = Array.prototype.slice.call(t.y);
            return true;
        }
    }
    return false;
};
const snapshotAxes = () => {
    const layout = mainPlot.layout || {};
    const read = a => {
        return { autorange: !a || a.autorange !== false, range: a && a.range ? a.range.slice() : null };
    };
    originalAxes = { x: read(layout.xaxis), y: read(layout.yaxis) };
};
const resetAxes = mode => {
    const keep = mode === 'semantic' && originalAxes;
    const update = {};
    const names = ['x', 'y'];
    for (let i = 0; i < names.length; i++) {
        const n = names[i];
        const saved = keep ? originalAxes[n] : null;
        const key = n + 'axis';
        if (saved && !saved.autorange && saved.range) {
            update[key + '.autorange'] = false;
            update[key + '.range'] = saved.range;
        } else {
            update[key + '.autorange'] = true;
        }
    }
    return Plotly.relayout(mainPlot, update);
};
const applyPlotViewportSize = () => {
    if (!mainPlot) return;
    mainPlot.style.width = 'calc(100% - ' + MAIN_PLOT_RIGHT_RESERVE_PX + 'px)';
    mainPlot.style.margin = '0';
    mainPlot.style.boxSizing = 'border-box';
    mainPlot.style.height = '100vh';
    if (typeof CSS !== 'undefined' && CSS.supports && CSS.supports('height', '100dvh')) {
        mainPlot.style.height = '100dvh';
    }
};
const mainPlotDoResize = () => {
    if (!mainPlot || !mainPlot.data) return;
    applyPlotViewportSize();
    let w = mainPlot.clientWidth;
    let h = mainPlot.clientHeight;
    if (!w || !h) {
        w = window.innerWidth - MAIN_PLOT_RIGHT_RESERVE_PX;
        h = window.innerHeight;
    }
    if (!w || !h) return;
    Plotly.relayout(mainPlot, {
        autosize: false,
        width: w,
        height: h
    });
};
const mainPlotScheduleResize = () => {
    if (mainPlotResizeTimer) clearTimeout(mainPlotResizeTimer);
    mainPlotResizeTimer = setTimeout(() => {
        mainPlotResizeTimer = null;
        mainPlotDoResize();
    }, 100);
};
const fitPlot = () => {
    if (!mainPlot) return;
    mainPlotDoResize();

    if (mainPlotResizeBound) return;
    mainPlotResizeBound = true;

    window.addEventListener('resize', mainPlotScheduleResize);
    window.addEventListener('orientationchange', mainPlotScheduleResize);
    window.addEventListener('load', mainPlotScheduleResize);
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', mainPlotScheduleResize);
    }
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(mainPlotScheduleResize);
        ro.observe(document.body);
        if (mainPlot.parentElement && mainPlot.parentElement !== document.body) {
            ro.observe(mainPlot.parentElement);
        }
    }
};
const fillMetricSelect = (sel, value, filter) => {
    if (!sel) return;
    const f = (filter || '').trim().toLowerCase();
    const prev = value || sel.value;
    const matches = [];
    for (let i = 0; i < metricCatalog.length; i++) {
        if (!f || metricCatalog[i].search.indexOf(f) !== -1) matches.push(metricCatalog[i]);
    }
    sel.innerHTML = '';
    if (matches.length === 0) {
        const o = document.createElement('option');
        o.value = ''; o.text = 'No matching metrics'; o.disabled = true; o.selected = true;
        sel.appendChild(o);
        return;
    }
    for (let j = 0; j < matches.length; j++) {
        const opt = document.createElement('option');
        opt.value = matches[j].key;
        opt.text = matches[j].label;
        sel.appendChild(opt);
    }
    let found = false;
    for (let k = 0; k < matches.length; k++) if (matches[k].key === prev) found = true;
    sel.value = found ? prev : matches[0].key;
};
const fillClusterSelect = (sel, value) => {
    if (!sel) return value;
    sel.innerHTML = '';
    for (let i = 0; i < clusterMetricKeys.length; i++) {
        const o = document.createElement('option');
        o.value = clusterMetricKeys[i];
        o.text = clusterLabel(clusterMetricKeys[i]);
        sel.appendChild(o);
    }
    for (let j = 0; j < clusterMetricKeys.length; j++) {
        if (clusterMetricKeys[j] === value) { sel.value = value; return value; }
    }
    if (clusterMetricKeys.length > 0) { sel.value = clusterMetricKeys[0]; return clusterMetricKeys[0]; }
    return value;
};
const applyCluster = mode => {
    if (focusClusterMetricUserSelected) return;
    const pref = preferredCluster(mode);
    if (!pref) return;
    focusClusterMetric = fillClusterSelect(document.getElementById('focusClusterMetricSelect'), pref);
};
const buildSizes = () => {
    const a = [];
    for (let i = 0; i < textNames.length; i++) a.push(CONFIG.normalSize);
    return a;
};
const buildOpacity = () => {
    const cluster = focusClusterOnly && focusClusterMetric ? metricsData[focusClusterMetric] : null;
    let focusVal = null;
    if (cluster && focusedTextIndex >= 0 && isNum(cluster[focusedTextIndex])) {
        focusVal = cluster[focusedTextIndex];
    }
    const a = [];
    for (let i = 0; i < textNames.length; i++) {
        if (i === focusedTextIndex) { a.push(1); continue; }
        if (hideGrayPoints && !isNum(currentColorDataForPlot[i])) { a.push(0); continue; }
        if (focusVal !== null && cluster[i] !== focusVal) { a.push(0); continue; }
        a.push(1);
    }
    return a;
};
const makeOverlay = meta => {
    return {
        x: [], y: [], mode: 'markers+text', type: 'scatter', meta: meta,
        hoverinfo: 'skip', showlegend: false, text: [],
        textposition: 'middle center',
        textfont: { size: 9, color: '#000' },
        marker: { symbol: 'star', size: CONFIG.focusStarSize, color: '#ffd400',
                  line: { color: '#111', width: 2 } }
    };
};
const ensureOverlay = (plot, meta) => {
    if (!plot || !plot.data) return Promise.resolve();
    if (findTrace(plot, meta) >= 0) return Promise.resolve();
    return Plotly.addTraces(plot, makeOverlay(meta));
};
const overlayFront = (plot, meta) => {
    if (!plot || !plot.data) return Promise.resolve();
    const idx = findTrace(plot, meta);
    const last = plot.data.length - 1;
    if (idx >= 0 && idx !== last) return Plotly.moveTraces(plot, [idx], [last]);
    return Promise.resolve();
};
const plotCoord = (plot, ti, axis, pi) => {
    const full = plot._fullData && plot._fullData[ti];
    const data = plot.data && plot.data[ti];
    const vals = (full && full[axis]) || (data && data[axis]);
    return vals ? vals[pi] : undefined;
};
const baseColor = (plot, pi) => {
    const t = plot._fullData && plot._fullData[0];
    if (!t) return null;
    const c = t.marker && t.marker.color;
    if (Array.isArray(c) || (c && ArrayBuffer.isView(c))) return c[pi];
    return typeof c === 'string' ? c : null;
};

/* ==================================================================
 * CAPITAL CITIES
 * ------------------------------------------------------------------
 * Capital markers are rendered by restyling the MAIN trace's markers
 * in place (no extra traces), exactly as in the old atlas_template.html.
 *
 * For the currently-active cluster metric key, we set:
 *   - non-capitals      → circle, black 1px outline (metric-colored fill)
 *   - cluster capitals  → pentagon, black 1px outline (metric-colored fill)
 *   - world capital     → pentagon, gold (#ffd400) 3px outline
 * and prepend a "<i>capital of the corpus/cluster</i>" note to the
 * hovertext of the relevant point.
 * ================================================================== */

const capitalsForMetric = key =>
    (CAPITAL_CITIES && CAPITAL_CITIES[key]) ? CAPITAL_CITIES[key] : null;

const capitalKindFor = (caps, name) => {
    if (!caps) return null;
    if (caps.world === name) return 'world';
    if (caps.byId) {
        for (const cid in caps.byId) {
            if (caps.byId[cid] === name) return 'cluster';
        }
    }
    return null;
};

const withCapitalNote = (hover, kind) => {
    if (!kind) return hover;
    const note = kind === 'world' ? 'capital of the corpus' : 'capital of the cluster';
    const br = hover.indexOf('<br>');
    if (br < 0) return hover;
    return hover.slice(0, br + 4) + '<i>' + note + '</i><br>' + hover.slice(br + 4);
};

const applyCapitals = key => {
    if (!mainPlot || !mainPlot.data || !mainPlot.data[0]) return Promise.resolve();
    const tr = mainPlot.data[0];
    if (baseHoverTexts === null) {
        const ht = tr.hovertext;
        if (Array.isArray(ht)) baseHoverTexts = ht.slice();
        else if (typeof ht === 'string') baseHoverTexts = textNames.map(() => ht);
        else baseHoverTexts = textNames.map(n => '<b>' + n + '</b><br>');
    }
    const caps = capitalsForMetric(key);
    const symbols = [], lineColors = [], lineWidths = [], hovers = [];
    for (let i = 0; i < textNames.length; i++) {
        const kind = capitalKindFor(caps, textNames[i]);
        const isWorld = kind === 'world';
        const isCap = kind !== null;
        symbols.push(isCap ? 'pentagon' : 'circle');
        lineColors.push(isWorld ? WORLD_CAPITAL_OUTLINE : 'black');
        lineWidths.push(isWorld ? 3 : 1);
        hovers.push(withCapitalNote(baseHoverTexts[i] || '', kind));
    }
    return Plotly.restyle(mainPlot, {
        'marker.symbol': [symbols],
        'marker.line.color': [lineColors],
        'marker.line.width': [lineWidths],
        'hovertext': [hovers]
    }, [0]);
};

const updateOverlay = () => {
    const idx = findTrace(mainPlot, FOCUS_OVERLAY_META);
    if (idx < 0) return Promise.resolve();
    if (focusedTextIndex < 0 || !focusedTextName) {
        return Plotly.restyle(mainPlot, { x: [[]], y: [[]], text: [[]] }, [idx]);
    }
    const update = {
        x: [[plotCoord(mainPlot, 0, 'x', focusedTextIndex)]],
        y: [[plotCoord(mainPlot, 0, 'y', focusedTextIndex)]],
        text: [[focusedTextName]]
    };
    const bc = baseColor(mainPlot, focusedTextIndex);
    const fv = currentColorDataForPlot[focusedTextIndex];
    if (typeof bc === 'string') update['marker.color'] = [[bc]];
    else if (!isNum(fv)) update['marker.color'] = [['#808080']];
    else {
        update['marker.color'] = [[fv]];
        update['marker.colorscale'] = [currentColorscale];
        update['marker.cmin'] = [currentCmin];
        update['marker.cmax'] = [currentCmax];
    }
    return Plotly.restyle(mainPlot, update, [idx]);
};
const refreshFocusMarker = () => {
    Plotly.restyle(mainPlot, { 'marker.opacity': [buildOpacity()] }, [0]);
    ensureOverlay(mainPlot, FOCUS_OVERLAY_META)
        .then(updateOverlay)
        .then(refreshConnections)
        .then(() => overlayFront(mainPlot, FOCUS_OVERLAY_META));
};
const syncColors = () => {
    const t = mainPlot.data && mainPlot.data[0];
    const c = t && t.marker && t.marker.color;
    if (c === undefined || c === null) {
        currentColorDataForPlot = [];
        for (let i = 0; i < textNames.length; i++) currentColorDataForPlot.push(null);
    } else if (Array.isArray(c)) {
        currentColorDataForPlot = c.slice();
    } else if (ArrayBuffer.isView(c)) {
        currentColorDataForPlot = Array.from(c);
    } else {
        currentColorDataForPlot = [];
        for (let j = 0; j < textNames.length; j++) currentColorDataForPlot.push(c);
    }
};
const readColorState = () => {
    const m = mainPlot.data[0] && mainPlot.data[0].marker;
    if (!m) return;
    if (m.colorscale !== undefined) currentColorscale = m.colorscale;
    if (isNum(m.cmin)) currentCmin = m.cmin;
    if (isNum(m.cmax)) currentCmax = m.cmax;
};
const connColor = (rank, total) => {
    const ratio = total <= 1 ? 0 : rank / (total - 1);
    const r = Math.round(235 * (1 - ratio) + 70 * ratio);
    const g = Math.round(35 * (1 - ratio) + 55 * ratio);
    const b = Math.round(45 * (1 - ratio) + 150 * ratio);
    return 'rgb(' + r + ', ' + g + ', ' + b + ')';
};
const cosineNeighbors = idx => {
    const fe = semanticEmbeddings[idx];
    if (!fe) return [];
    const out = [];
    for (let i = 0; i < textNames.length; i++) {
        const e = semanticEmbeddings[i];
        if (i === idx || !e) continue;
        const len = Math.min(fe.length, e.length);
        let sim = 0;
        for (let d = 0; d < len; d++) sim += fe[d] * e[d];
        out.push({ text: textNames[i], index: i, similarity: sim });
    }
    out.sort((a, b) => b.similarity - a.similarity);
    return out;
};
const burrowsNeighbors = idx => {
    const row = burrowsSimilarity[idx];
    if (!row) return [];
    const out = [];
    for (let i = 0; i < textNames.length; i++) {
        if (i === idx || !isNum(row[i])) continue;
        out.push({ text: textNames[i], index: i, similarity: row[i] });
    }
    out.sort((a, b) => b.similarity - a.similarity);
    return out;
};
const focusNeighbors = idx => {
    if (idx < 0) return [];
    return focusSimilarityMode === 'burrows' ? burrowsNeighbors(idx) : cosineNeighbors(idx);
};
const connTrace = (from, to, rank, total) => {
    return {
        x: [from.x, to.x], y: [from.y, to.y], mode: 'lines', type: 'scatter',
        meta: CONNECTION_META, hoverinfo: 'skip', showlegend: false,
        line: { color: connColor(rank, total), width: 2 }
    };
};
const removeConns = plot => {
    const idxs = [];
    for (let i = 0; i < plot.data.length; i++) {
        if (plot.data[i].meta === CONNECTION_META) idxs.push(i);
    }
    if (!idxs.length) return Promise.resolve();
    return Plotly.deleteTraces(plot, idxs.reverse());
};
const refreshConnections = () => {
    const token = ++mainConnectionToken;
    return removeConns(mainPlot).then(() => {
        if (token !== mainConnectionToken) return;
        if (!semanticLineModeEnabled || focusedTextIndex < 0 || !focusedTextName) return;
        const fp = {
            x: plotCoord(mainPlot, 0, 'x', focusedTextIndex),
            y: plotCoord(mainPlot, 0, 'y', focusedTextIndex)
        };
        const nb = focusNeighbors(focusedTextIndex).slice(0, semanticLineCount);
        const traces = [];
        for (let i = 0; i < nb.length; i++) {
            const tp = { x: plotCoord(mainPlot, 0, 'x', nb[i].index),
                         y: plotCoord(mainPlot, 0, 'y', nb[i].index) };
            traces.push(connTrace(fp, tp, i, nb.length));
        }
        if (traces.length && token === mainConnectionToken) {
            return Plotly.addTraces(mainPlot, traces);
        }
    }).then(() => {
        return overlayFront(mainPlot, FOCUS_OVERLAY_META);
    });
};
const refreshCorrConnections = () => {
    const plot = document.getElementById('correlationPlot');
    if (!plot || !plot.data || !plot.data[0]) return Promise.resolve();
    const token = ++correlationConnectionToken;
    return removeConns(plot).then(() => {
        if (token !== correlationConnectionToken) return;
        if (!semanticLineModeEnabled || !focusedTextName) return;
        const texts = plot.data[0].text || [];
        let fp = -1;
        for (let i = 0; i < texts.length; i++) if (texts[i] === focusedTextName) { fp = i; break; }
        if (fp < 0) return;
        const entry = textLookup[focusedTextName.toLowerCase()];
        if (!entry) return;
        const from = { x: plot.data[0].x[fp], y: plot.data[0].y[fp] };
        const nb = focusNeighbors(entry.index).slice(0, semanticLineCount);
        const traces = [];
        for (let j = 0; j < nb.length; j++) {
            let target = -1;
            for (let t = 0; t < texts.length; t++) if (texts[t] === nb[j].text) { target = t; break; }
            if (target < 0) continue;
            traces.push(connTrace(from,
                { x: plot.data[0].x[target], y: plot.data[0].y[target] },
                j, nb.length));
        }
        if (traces.length && token === correlationConnectionToken) {
            return Plotly.addTraces(plot, traces);
        }
    }).then(() => {
        return overlayFront(plot, CORR_FOCUS_OVERLAY_META);
    });
};
const refreshCorrFocus = () => {
    const plot = document.getElementById('correlationPlot');
    if (!plot || !plot.data || !plot.data[0]) return;
    const texts = plot.data[0].text || [];
    let pi = -1;
    if (focusedTextName) {
        for (let i = 0; i < texts.length; i++) if (texts[i] === focusedTextName) { pi = i; break; }
    }
    ensureOverlay(plot, CORR_FOCUS_OVERLAY_META).then(() => {
        const idx = findTrace(plot, CORR_FOCUS_OVERLAY_META);
        if (idx < 0) return null;
        if (pi < 0) return Plotly.restyle(plot, { x: [[]], y: [[]], text: [[]] }, [idx]);
        const update = {
            x: [[plot.data[0].x[pi]]], y: [[plot.data[0].y[pi]]],
            text: [[focusedTextName]]
        };
        const bc = baseColor(plot, pi);
        if (typeof bc === 'string') update['marker.color'] = [[bc]];
        return Plotly.restyle(plot, update, [idx]);
    }).then(() => {
        return overlayFront(plot, CORR_FOCUS_OVERLAY_META);
    }).then(refreshCorrConnections);
};
const percentileOf = (key, idx) => {
    const vals = metricsData[key];
    if (!vals || idx < 0 || idx >= vals.length) return null;
    const fv = vals[idx];
    if (!isNum(fv)) return { percentile: null, value: fv };
    const valid = [];
    for (let i = 0; i < vals.length; i++) if (isNum(vals[i])) valid.push(vals[i]);
    if (!valid.length) return { percentile: null, value: fv };
    valid.sort((a, b) => a - b);
    return { percentile: (countLE(valid, fv) / valid.length) * 100, value: fv };
};
const updatePctLabel = () => {
    const lbl = document.getElementById('focusMetricPercentile');
    if (!lbl) return;
    if (!focusedTextName || focusedTextIndex < 0) { lbl.textContent = ''; return; }
    const mlabel = metricDisplayNames[currentMetric] || currentMetric;
    const info = percentileOf(currentMetric, focusedTextIndex);
    if (!info || info.percentile === null) {
        lbl.textContent = mlabel + ': no score';
        return;
    }
    const whole = Math.abs(info.value - Math.round(info.value)) < 1e-9;
    const disp = whole ? String(Math.round(info.value)) : fmt(info.value, 4);
    lbl.textContent = mlabel + ': ' + ordinal(info.percentile) + ' percentile (' + disp + ')';
};
const afterFocusChange = () => {
    refreshFocusMarker();
    refreshCorrFocus();
    updateTextList();
    updatePctLabel();
    const focusSelect = document.getElementById('focusTextSelect');
    if (focusSelect && focusedTextName) {
        const idx = textNames.indexOf(focusedTextName);
        if (idx >= 0 && focusSelect.querySelector('option[value="' + idx + '"]')) {
            focusSelect.value = String(idx);
        }
    }
    if (viewMode === 'functionWord') {
        updateFwPlot();
        if (clusterStatsState.view === 'cluster') {
            if (!clusterStatsState.clusterIdUserSelected) clusterStatsState.clusterId = null;
            renderClusterStats();
        }
    }
};
const focusText = name => {
    const typed = (name || '').trim();
    if (!typed) {
        focusedTextName = null;
        focusedTextIndex = -1;
        afterFocusChange();
        return;
    }
    const match = textLookup[typed.toLowerCase()];
    if (!match) { alert('No text named "' + typed + '" was found.'); return; }
    focusedTextName = match.name;
    focusedTextIndex = match.index;
    afterFocusChange();
    if (viewMode === 'delta' && focusedTextIndex >= 0 && deltaState.textA !== focusedTextIndex) {
        deltaState.textA = focusedTextIndex;
        if (deltaState.textB === null || deltaState.textB === focusedTextIndex) {
            deltaState.textB = defaultDeltaPartner(focusedTextIndex);
        }
        syncDeltaDropdowns();
        updateDeltaPlot();
    }
};
const flashPoint = idx => {
    const sizes = buildSizes();
    const ops = buildOpacity();
    for (let i = 0; i < ops.length; i++) if (ops[i] !== 0) ops[i] = CONFIG.dimmedOpacity;
    sizes[idx] = CONFIG.highlightSize;
    ops[idx] = 1;
    Plotly.restyle(mainPlot, { 'marker.size': [sizes], 'marker.opacity': [ops] }, [0]);
    setTimeout(() => {
        Plotly.restyle(mainPlot, { 'marker.size': [buildSizes()], 'marker.opacity': [buildOpacity()] }, [0])
            .then(refreshFocusMarker);
    }, CONFIG.highlightDuration);
};
const updateTextList = () => {
    const order = document.getElementById('textSortOrder').value;
    const list = document.getElementById('textList');
    const head = document.getElementById('textListHeading');
    list.innerHTML = '';

    if (semanticLineModeEnabled && focusedTextName) {
        const isB = focusSimilarityMode === 'burrows';
        const modeLbl = isB ? 'Stylistically (Burrows)' : 'Semantically';
        const scoreLbl = isB ? 'burrows' : 'cosine';
        head.textContent = 'Most ' + modeLbl + ' Similar to ' + focusedTextName + ':';
        const nb = focusNeighbors(focusedTextIndex);
        for (let i = 0; i < nb.length; i++) {
            const row = makeDiv((i + 1) + '. ' + nb[i].text + ' (' + scoreLbl + ': ' + fmt(nb[i].similarity, 4) + ')');
            row.title = 'Click to puff this text on the map (does not change focus)';
            ((idx) => {
                row.addEventListener('click', () => { flashPoint(idx); });
            })(nb[i].index);
            list.appendChild(row);
        }
        return;
    }

    head.textContent = 'Texts';
    const vals = metricsData[currentMetric];
    const uni = fightingWordsUnigrams[currentMetric];
    const useVals = vals;
    const items = [];
    for (let j = 0; j < textNames.length; j++) {
        if (useVals && isNum(useVals[j])) items.push({ name: textNames[j], index: j, value: useVals[j] });
    }
    if (order === 'alpha-asc') items.sort((a, b) => a.name.localeCompare(b.name));
    else if (order === 'alpha-desc') items.sort((a, b) => b.name.localeCompare(a.name));
    else if (order === 'metric-asc') items.sort((a, b) => (a.value || 0) - (b.value || 0));
    else if (order === 'metric-desc') items.sort((a, b) => (b.value || 0) - (a.value || 0));

    for (let k = 0; k < items.length; k++) {
        const u = uni ? uni[items[k].name] : null;
        const txt = (k + 1) + '. ' + items[k].name + ' (' + fmt(items[k].value, 4) + ')' + (u ? ' ' + u : '');
        const r = makeDiv(txt);
        r.title = 'Click to puff this text on the map (does not change focus)';
        if (items[k].index === focusedTextIndex) r.classList.add('focusedTextItem');
        ((idx) => {
            r.addEventListener('click', () => { flashPoint(idx); });
        })(items[k].index);
        list.appendChild(r);
    }
};

const updateColorInputs = () => {
    const m = mainPlot.data[0] && mainPlot.data[0].marker;
    if (!m) return;
    document.getElementById('colorMin').value = m.cmin;
    document.getElementById('colorMax').value = m.cmax;
};
const applyColorScale = (lo, hi) => {
    Plotly.restyle(mainPlot, { 'marker.cmin': lo, 'marker.cmax': hi }, [0]).then(() => {
        currentCmin = lo;
        currentCmax = hi;
        refreshFocusMarker();
    });
};
const zoomColor = factor => {
    const lo = parseFloat(document.getElementById('colorMin').value);
    const hi = parseFloat(document.getElementById('colorMax').value);
    const range = hi - lo;
    const center = (lo + hi) / 2;
    const nlo = center - range * factor;
    const nhi = center + range * factor;
    document.getElementById('colorMin').value = nlo;
    document.getElementById('colorMax').value = nhi;
    applyColorScale(nlo, nhi);
};

const changeMetric = key => {
    const data = colorData(key);
    if (!data) return;
    const range = metricRange(data);
    if (!range) return;
    const cluster = isClusterKey(key);
    const scale = cluster ? CLUSTER_TURBO_COLORSCALE : (metricColorScales[key] || CONFIG.defaultColorScale);

    currentMetric = key;
    currentColorDataForPlot = data;
    currentColorscale = scale;
    currentCmin = range.min;
    currentCmax = range.max;

    const markerUpdate = {
        'marker.color': [data],
        'marker.colorscale': [scale],
        'marker.autocolorscale': [false],
        'marker.cmin': range.min,
        'marker.cmax': range.max,
        'marker.colorbar.title': colorbarTitle(key),
        'marker.colorbar.tickvals': [null],
        'marker.colorbar.ticktext': [null]
    };
    if (cluster) {
        markerUpdate['marker.colorbar.showticklabels'] = [false];
    } else {
        markerUpdate['marker.colorbar.showticklabels'] = [true];
    }

    Promise.all([
        Plotly.restyle(mainPlot, markerUpdate, [0]),
        Plotly.relayout(mainPlot, { 'title.text': chartTitles[key] || 'Text Semantic Clusters' })
    ]).then(() => {
        updateColorInputs();
        updateTextList();
        updatePctLabel();
        refreshFocusMarker();
        applyCapitals(key);
    });
};


const collectPoints = (xs, ys) => {
    const out = { texts: [], x: [], y: [], colors: [] };
    for (let i = 0; i < xs.length; i++) {
        if (!isNum(xs[i]) || !isNum(ys[i])) continue;
        out.texts.push(textNames[i]);
        out.x.push(xs[i]);
        out.y.push(ys[i]);
        out.colors.push(i);
    }
    return out;
};
const outlierList = (parent, title, items) => {
    const h = makeDiv(title, '4px');
    h.style.marginTop = '10px';
    h.style.fontWeight = '600';
    parent.appendChild(h);
    for (let i = 0; i < items.length; i++) {
        parent.appendChild(makeDiv((i + 1) + '. ' + items[i].text + ' (|res|=' + fmt(items[i].absResidual, 4) + ')', '3px'));
    }
};
const updateCorrStats = (points, corr, type, model) => {
    const body = clearBody('correlationStats');
    if (!body) return;
    const h = document.createElement('strong');
    h.textContent = 'Correlation Statistics';
    body.appendChild(h);
    const wrap = document.createElement('div');
    wrap.style.marginTop = '8px';
    wrap.appendChild(makeDiv('Sample Size: ' + points.x.length, '6px'));
    if (corr !== null) {
        const abs = Math.abs(corr);
        let strength = ' (Weak)';
        if (abs >= 0.7) strength = ' (Strong)';
        else if (abs >= 0.4) strength = ' (Moderate)';
        wrap.appendChild(makeDiv('Pearson r: ' + fmt(corr, 4) + strength, '6px'));
        wrap.appendChild(makeDiv('R\u00B2: ' + fmt(corr * corr, 4), '6px'));
    } else {
        wrap.appendChild(makeDiv('Pearson r: unavailable', '6px'));
    }
    wrap.appendChild(makeDiv('X variance/std: ' + fmt(variance(points.x), 4) + ' / ' + fmt(stddev(points.x), 4), '6px'));
    wrap.appendChild(makeDiv('Y variance/std: ' + fmt(variance(points.y), 4) + ' / ' + fmt(stddev(points.y), 4), '6px'));
    const ranked = residualRank(points.texts, points.x, points.y, type, model);
    if (ranked.length) {
        outlierList(wrap, 'Greatest Outliers', ranked.slice(0, 5));
        outlierList(wrap, 'Least Outliers', ranked.slice(-5).reverse());
    }
    body.appendChild(wrap);
};
const updateCorrPlot = () => {
    const xs = document.getElementById('xMetricSelect');
    const ys = document.getElementById('yMetricSelect');
    const type = document.getElementById('regressionType').value;
    correlationXMetric = xs.value;
    correlationYMetric = ys.value;
    const xd = metricsData[correlationXMetric];
    const yd = metricsData[correlationYMetric];
    if (!xd || !yd) { alert('One of the selected metrics has no data.'); return; }
    const points = collectPoints(xd, yd);
    const corr = pearson(xd, yd);
    const xl = metricDisplayNames[correlationXMetric];
    const yl = metricDisplayNames[correlationYMetric];
    let title = xl + ' vs ' + yl;
    if (corr !== null) title += ' (r = ' + corr.toFixed(3) + ')';

    const scatter = {
        x: points.x, y: points.y, mode: 'markers+text', type: 'scatter',
        name: '', text: points.texts, textposition: 'middle center',
        textfont: { size: CONFIG.pointFontSize, color: 'black' },
        marker: { size: CONFIG.normalSize, color: points.colors,
                  colorscale: 'Viridis', showscale: false,
                  line: { color: 'black', width: 1 } },
        hovertext: points.texts, hoverinfo: 'text+x+y'
    };
    const model = regModel(points.x, points.y, type);
    const line = regLine(points.x, points.y, type, model);
    const traces = [scatter];
    if (line) {
        traces.push({
            x: line.x, y: line.y, mode: 'lines', type: 'scatter',
            name: type + ' fit',
            line: { color: 'rgba(220, 53, 69, 0.2)', width: 3, dash: 'solid' },
            hoverinfo: 'skip', showlegend: false
        });
    }
    const layout = {
        title: { text: title, x: 0.5, xanchor: 'center', font: { size: 14 } },
        xaxis: { title: xl, showgrid: GRID_STYLE.showgrid, gridwidth: GRID_STYLE.gridwidth, gridcolor: GRID_STYLE.gridcolor },
        yaxis: { title: yl, showgrid: GRID_STYLE.showgrid, gridwidth: GRID_STYLE.gridwidth, gridcolor: GRID_STYLE.gridcolor },
        plot_bgcolor: '#f8f9fa', paper_bgcolor: 'white',
        hovermode: 'closest', showlegend: false,
        font: { family: 'Arial, sans-serif' },
        margin: { t: 55, l: 70, r: 70, b: 55 }
    };
    drawPlot('correlationPlot', traces, layout, null).then(() => {
        return Plotly.Plots.resize(document.getElementById('correlationPlot'));
    }).then(() => {
        refreshCorrFocus();
        addOutlines('#correlationPlot .scatterlayer .textpoint text', 'correlationStrokeText');
    });
    updateCorrStats(points, corr, type, model);
};
const populateCorrDropdowns = () => {
    const xs = document.getElementById('xMetricSelect');
    const ys = document.getElementById('yMetricSelect');
    if (!xs || !ys) return;
    fillMetricSelect(xs, correlationXMetric, document.getElementById('xMetricSearchInput').value);
    fillMetricSelect(ys, correlationYMetric, document.getElementById('yMetricSearchInput').value);
    if (xs.value) correlationXMetric = xs.value;
    if (ys.value) correlationYMetric = ys.value;
};


const selectedWords = () => {
    const out = [];
    functionWordSelection.forEach(w => {
        if (Array.isArray(burrowsWordFrequencies[w])) out.push(w);
    });
    out.sort((a, b) => a.localeCompare(b));
    return out;
};


const aggregatedRates = () => {
    const words = selectedWords();
    if (!words.length) return null;
    const rates = [];
    for (let i = 0; i < textNames.length; i++) {
        let sum = 0, bad = false;
        for (let w = 0; w < words.length; w++) {
            const v = burrowsWordFrequencies[words[w]][i];
            if (!isNum(v)) { bad = true; break; }
            sum += v;
        }
        rates.push(bad ? null : sum);
    }
    return { words, rates };
};
const wordLabel = words => {
    if (words.length === 1) return '"' + words[0] + '"';
    if (words.length <= 4) {
        const s = [];
        for (let i = 0; i < words.length; i++) s.push('"' + words[i] + '"');
        return s.join(' + ');
    }
    return words.length + ' function words (' + words.slice(0, 3).join(', ') + ', ...)';
};
const axisTitle = words => {
    if (words.length === 1) return 'Proportion of tokens that are "' + words[0] + '"';
    return 'Combined proportion of tokens that are any of ' + wordLabel(words);
};
const updateWordSummary = () => {
    const s = document.getElementById('functionWordSelectionSummary');
    if (!s) return;
    const w = selectedWords();
    s.textContent = w.length ? 'Selected (' + w.length + '): ' + w.join(', ') : 'No function words selected.';
};
const populateWordControls = () => {
    const sel = document.getElementById('functionWordSelect');
    if (!sel) return;
    const filter = document.getElementById('functionWordSearchInput').value.trim().toLowerCase();
    const words = burrowsFunctionWords || [];
    sel.innerHTML = '';
    for (let i = 0; i < words.length; i++) {
        if (filter && words[i].toLowerCase().indexOf(filter) === -1) continue;
        const o = document.createElement('option');
        o.value = words[i];
        o.text = words[i];
        o.selected = functionWordSelection.has(words[i]);
        sel.appendChild(o);
    }
    if (!sel.options.length) {
        const o2 = document.createElement('option');
        o2.value = ''; o2.text = 'No matching function words'; o2.disabled = true;
        sel.appendChild(o2);
    }
    document.getElementById('functionWordPlotModeSelect').value = functionWordPlotMode;
    updateWordSummary();
};
const focusedCluster = () => {
    if (focusedTextIndex < 0 || !focusClusterMetric) return null;
    const data = metricsData[focusClusterMetric];
    if (!data || focusedTextIndex >= data.length) return null;
    return isNum(data[focusedTextIndex]) ? data[focusedTextIndex] : null;
};
const wordCtx = (words, rates) => {
    const validIdx = [], validVal = [];
    for (let i = 0; i < rates.length; i++) {
        if (isNum(rates[i])) { validIdx.push(i); validVal.push(rates[i]); }
    }
    const m = mean(validVal);
    let s = stddev(validVal);
    if (!isFinite(s) || s <= 0) s = Math.max(1e-9, m * 1e-3);
    const sorted = validVal.slice().sort((a, b) => a - b);
    const dmin = sorted[0], dmax = sorted[sorted.length - 1];
    const margin = Math.max(0.02 * s, 1e-6);

    const ctx = {
        words, wordRates: rates, validIndices: validIdx, validValues: validVal,
        sortedValues: sorted, mean: m, std: s, dataMin: dmin, dataMax: dmax,
        minX: Math.max(0, Math.min(dmin - margin, m - 4 * s)),
        maxX: Math.max(dmax + margin, m + 4 * s),
        wordLabel: wordLabel(words),
        axisTitle: axisTitle(words),
        zOf: r => (r - m) / s,
        clusterRates: null, clusterMean: null, clusterStd: null, clusterPercentileMean: null,
        focusRate: null, focusPercentile: null, focusZ: null, focusNormalPercentile: null,
        evilClusterRates: null, evilClusterMean: null, evilClusterStd: null, evilClusterPercentileMean: null,
        evilFocusRate: null, evilFocusPercentile: null, evilFocusZ: null
    };

    const cdata = focusClusterMetric ? metricsData[focusClusterMetric] : null;
    const fcv = focusedCluster();
    ctx.isInFocusedCluster = i => fcv !== null && cdata && cdata[i] === fcv;
    if (fcv !== null) {
        const cr = [];
        for (let j = 0; j < validIdx.length; j++) {
            if (ctx.isInFocusedCluster(validIdx[j])) cr.push(rates[validIdx[j]]);
        }
        if (cr.length) {
            ctx.clusterRates = cr;
            ctx.clusterMean = mean(cr);
            ctx.clusterStd = stddev(cr);
            ctx.clusterPercentileMean = empPct(sorted, ctx.clusterMean);
        }
    }
    if (focusedTextIndex >= 0 && isNum(rates[focusedTextIndex])) {
        ctx.focusRate = rates[focusedTextIndex];
        ctx.focusPercentile = empPct(sorted, ctx.focusRate);
        ctx.focusZ = ctx.zOf(ctx.focusRate);
        ctx.focusNormalPercentile = normalCdf(ctx.focusZ) * 100;
    }

    ctx.evilClusterId = (clusterStatsState.compareMode === 'cluster') ? clusterStatsState.compareClusterId : null;
    if (ctx.evilClusterId !== null && asNum(ctx.evilClusterId) === asNum(clusterStatsState.clusterId)) {
        ctx.evilClusterId = null;
    }

    const evilCdata = clusterStatsState.source ? metricsData[clusterStatsState.source] : null;
    if (ctx.evilClusterId !== null && evilCdata) {
        ctx.isInEvilCluster = i => asNum(evilCdata[i]) === asNum(ctx.evilClusterId);
        const ecr = [];
        for (let j2 = 0; j2 < validIdx.length; j2++) {
            if (ctx.isInEvilCluster(validIdx[j2])) ecr.push(rates[validIdx[j2]]);
        }
        if (ecr.length) {
            ctx.evilClusterRates = ecr;
            ctx.evilClusterMean = mean(ecr);
            ctx.evilClusterStd = stddev(ecr);
            ctx.evilClusterPercentileMean = empPct(sorted, ctx.evilClusterMean);
        }
    } else {
        ctx.isInEvilCluster = () => false;
    }

    if (evilFocusTextIndex >= 0 && isNum(rates[evilFocusTextIndex])) {
        ctx.evilFocusRate = rates[evilFocusTextIndex];
        ctx.evilFocusPercentile = empPct(sorted, ctx.evilFocusRate);
        ctx.evilFocusZ = ctx.zOf(ctx.evilFocusRate);
    }

    return ctx;
};
const groupIdx = ctx => {
    const g = { focus: [], cluster: [], evil: [], other: [] };
    for (let i = 0; i < ctx.validIndices.length; i++) {
        const idx = ctx.validIndices[i];
        const inCluster = ctx.isInFocusedCluster(idx);
        const inEvil = ctx.isInEvilCluster && ctx.isInEvilCluster(idx);
        if (idx === focusedTextIndex) g.focus.push(idx);
        else if (inCluster && inEvil) { g.cluster.push(idx); g.evil.push(idx); }
        else if (inCluster) g.cluster.push(idx);
        else if (inEvil) g.evil.push(idx);
        else g.other.push(idx);
    }
    return g;
};
const markerTrace = (idxs, ptFn, name, mk, hover) => {
    const pts = [];
    for (let i = 0; i < idxs.length; i++) pts.push(ptFn(idxs[i]));
    const xs = [], ys = [], labels = [];
    for (let j = 0; j < pts.length; j++) {
        xs.push(pts[j].x); ys.push(pts[j].y); labels.push(pts[j].label);
    }
    return {
        x: xs, y: ys, text: labels, mode: 'markers', type: 'scatter',
        name: name, marker: mk, hoverinfo: 'text+x', hovertemplate: hover
    };
};
const groupTraces = (groups, ptFn, hover) => {
    const traces = [];
    if (groups.other.length && !hideGrayPoints) {
        traces.push(markerTrace(groups.other, ptFn, 'Other texts', {
            size: 5, color: 'rgba(120, 120, 120, 0.5)',
            line: { color: 'rgba(80, 80, 80, 0.8)', width: 0.5 }
        }, hover));
    }
    if (groups.cluster.length) {
        traces.push(markerTrace(groups.cluster, ptFn, 'Focused cluster', {
            size: 9, color: FOCUS_CLUSTER_COLOR,
            line: { color: FOCUS_CLUSTER_LINE_COLOR, width: 1.5 }
        }, hover));
    }
    if (groups.evil.length) {
        traces.push(markerTrace(groups.evil, ptFn, 'Evil focus cluster', {
            size: 9, color: EVIL_CLUSTER_COLOR,
            line: { color: EVIL_CLUSTER_LINE_COLOR, width: 1.5 }
        }, hover));
    }
    return traces;
};
const focusStar = (x, y, text, hover) => {
    return {
        x: [x], y: [y], mode: 'markers+text', type: 'scatter',
        name: 'Focus text', text: [text], textposition: 'top center',
        textfont: { size: 10, color: '#000' },
        marker: { symbol: 'star', size: CONFIG.focusStarSize, color: '#ffd400',
                  line: { color: '#111', width: 2 } },
        hoverinfo: hover ? 'text+x' : 'text',
        hovertemplate: hover || '%{text}<extra></extra>'
    };
};
const clusterDiamond = (x, y, text) => {
    return {
        x: [x], y: [y], mode: 'markers+text', type: 'scatter',
        name: 'Cluster mean', showlegend: false, text: [text],
        textposition: 'bottom center',
        textfont: { size: 10, color: '#7a0403' },
        marker: { symbol: 'diamond', size: 14, color: '#f36315',
                  line: { color: '#7a0403', width: 2 } },
        hoverinfo: 'text', hovertemplate: '%{text}<extra></extra>'
    };
};
const clusterLine = (x, topY, ctx) => {
    return {
        x: [x, x], y: [0, topY], mode: 'lines', type: 'scatter',
        name: ctx.clusterRates ? 'Focused cluster mean (n=' + ctx.clusterRates.length + ')' : 'Focused cluster mean',
        line: { color: '#7a0403', width: 2, dash: 'dash' }, hoverinfo: 'skip'
    };
};
const evilFocusStar = (x, y, text, hover) => {
    return {
        x: [x], y: [y], mode: 'markers+text', type: 'scatter',
        name: 'Evil focus text', text: [text], textposition: 'bottom center',
        textfont: { size: 10, color: 'rgba(107, 33, 168, 0.85)' },
        marker: { symbol: 'star', size: CONFIG.focusStarSize, color: EVIL_CLUSTER_LINE_COLOR, opacity: 0.8,
                  line: { color: '#6b21a8', width: 2 } },
        hoverinfo: hover ? 'text+x' : 'text',
        hovertemplate: hover || '%{text}<extra></extra>'
    };
};
const evilClusterDiamond = (x, y, text) => {
    return {
        x: [x], y: [y], mode: 'markers+text', type: 'scatter',
        name: 'Evil cluster mean :(', showlegend: false, text: [text], textposition: 'bottom center',
        textfont: { size: 10, color: 'rgba(107, 33, 168, 0.85)' },
        marker: { symbol: 'diamond', size: 14, color: EVIL_CLUSTER_LINE_COLOR, opacity: 0.85,
                  line: { color: '#6b21a8', width: 2 } },
        hoverinfo: 'text', hovertemplate: '%{text}<extra></extra>'
    };
};
const evilClusterLine = (x, topY, ctx) => {
    return {
        x: [x, x], y: [0, topY], mode: 'lines', type: 'scatter',
        name: ctx.evilClusterRates ? 'Evil cluster mean (n=' + ctx.evilClusterRates.length + ')' : 'Evil cluster mean',
        line: { color: EVIL_CLUSTER_LINE_COLOR, width: 2, dash: 'dot' }, hoverinfo: 'skip'
    };
};
const histTrace = (values, s, minX, maxX, kind) => {
    const count = values.length;
    let bw = s > 0 && count > 1 ? 3.49 * s * Math.pow(count, -1 / 3) : 0;
    if (!isFinite(bw) || bw <= 0) bw = Math.max((maxX - minX) / 20, 1e-9);
    const bc = Math.max(10, Math.min(60, Math.ceil((maxX - minX) / bw)));
    const ew = (maxX - minX) / bc;
    const bins = [];
    for (let i = 0; i < bc; i++) bins.push(0);
    const centers = [];
    for (let j = 0; j < bc; j++) centers.push(minX + ew * (j + 0.5));
    for (let k = 0; k < values.length; k++) {
        let idx = Math.floor((values[k] - minX) / ew);
        if (idx < 0) idx = 0;
        if (idx > bc - 1) idx = bc - 1;
        bins[idx]++;
    }
    const ys = [];
    for (let m = 0; m < bc; m++) ys.push(bins[m] / (count * ew));
    const format = kind === 'z' ? '%{x:.2f}' : '%{x:.4f}';
    return {
        x: centers, y: ys, type: 'bar',
        name: kind === 'z' ? 'Observed z-scores' : 'Observed rate',
        marker: { color: HIST_FILL, line: { color: HIST_LINE, width: 1 } },
        hoverinfo: 'x+y',
        hovertemplate: kind + ': ' + format + '<br>density: %{y:.4f}<extra></extra>'
    };
};
const wordLayout = (title, xaxis, yaxis, bottomMargin) => {
    const xa = {};
    xa.showgrid = GRID_STYLE.showgrid; xa.gridwidth = GRID_STYLE.gridwidth;
    xa.gridcolor = GRID_STYLE.gridcolor; xa.zeroline = false;
    for (const k in xaxis) xa[k] = xaxis[k];
    const ya = {};
    ya.showgrid = GRID_STYLE.showgrid; ya.gridwidth = GRID_STYLE.gridwidth;
    ya.gridcolor = GRID_STYLE.gridcolor; ya.zeroline = false;
    for (const k2 in yaxis) ya[k2] = yaxis[k2];
    return {
        title: { text: title, x: 0.5, xanchor: 'center', font: { size: 14 } },
        xaxis: xa, yaxis: ya,
        plot_bgcolor: '#f8f9fa', paper_bgcolor: 'white',
        hovermode: 'closest', showlegend: true,
        legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.0, yanchor: 'bottom',
                  bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1 },
        barmode: 'overlay',
        font: { family: 'Arial, sans-serif' },
        margin: { t: 70, l: 70, r: 70, b: bottomMargin }
    };
};
const wordTitle = (prefix, ctx) => {
    return prefix + ' ' + ctx.wordLabel + ' rate';
};
const focusAnnotation = ctx => {
    const base = { xref: 'paper', yref: 'paper', x: 0.01, y: 0.99,
                 xanchor: 'left', yanchor: 'top', showarrow: false };
    if (ctx.focusZ === null) {
        base.font = { size: 11, color: '#777' };
        base.text = 'Focus a text to see its percentile';
        return base;
    }
    base.align = 'left';
    base.bgcolor = 'rgba(255, 244, 179, 0.95)';
    base.bordercolor = '#111';
    base.borderwidth = 1;
    base.borderpad = 6;
    base.font = { size: 13, color: '#000' };
    base.text = '<b>' + focusedTextName + '</b><br>' +
                ordinal(ctx.focusPercentile) + ' percentile (actual)<br>' +
                ordinal(ctx.focusNormalPercentile) + ' percentile (normal curve)<br>' +
                'z = ' + ctx.focusZ.toFixed(2);
    return base;
};
const zView = ctx => {
    const zmin = Math.min(-4, ctx.zOf(ctx.dataMin) - 0.25);
    const zmax = Math.max(4, ctx.zOf(ctx.dataMax) + 0.25);
    const stdNorm = z => normalPdf(z, 0, 1);
    const curve = sampleCurve(zmin, zmax, 360, stdNorm);
    const traces = [];
    if (functionWordShowHistogram) {
        const zvals = [];
        for (let i = 0; i < ctx.validValues.length; i++) zvals.push(ctx.zOf(ctx.validValues[i]));
        traces.push(histTrace(zvals, 1, zmin, zmax, 'z'));
    }
    if (ctx.focusZ !== null) {
        const sx = [], sy = [];
        for (let j = 0; j < curve.x.length; j++) {
            if (curve.x[j] <= ctx.focusZ) { sx.push(curve.x[j]); sy.push(curve.y[j]); }
        }
        const cz = Math.max(zmin, Math.min(zmax, ctx.focusZ));
        sx.push(cz);
        sy.push(stdNorm(cz));
        traces.push({
            x: sx, y: sy, mode: 'lines', type: 'scatter',
            fill: 'tozeroy', fillcolor: FOCUS_AREA_FILL,
            line: { width: 0 },
            name: 'Area below focus (' + ctx.focusNormalPercentile.toFixed(1) + '%)',
            hoverinfo: 'skip'
        });
    }
    traces.push({
        x: curve.x, y: curve.y, mode: 'lines', type: 'scatter',
        name: 'Standard normal', line: { color: NORMAL_CURVE_COLOR, width: 3 }, hoverinfo: 'skip'
    });
    if (ctx.clusterMean !== null) {
        const cz2 = ctx.zOf(ctx.clusterMean);
        const cy = stdNorm(cz2);
        traces.push(clusterLine(cz2, cy, ctx));
        traces.push(clusterDiamond(cz2, cy,
            'cluster mean<br>z=' + cz2.toFixed(2) + '<br>' + ctx.clusterPercentileMean.toFixed(1) + '%'));
    }
    if (ctx.evilClusterMean !== null) {
        const ezc2 = ctx.zOf(ctx.evilClusterMean);
        const ecy = stdNorm(ezc2);
        traces.push(evilClusterLine(ezc2, ecy, ctx));
        traces.push(evilClusterDiamond(ezc2, ecy,
            'evil cluster mean :(<br>z=' + ezc2.toFixed(2) + '<br>' + ctx.evilClusterPercentileMean.toFixed(1) + '%'));
    }
    if (functionWordShowMarkers) {
        const groups = groupIdx(ctx);
        const ptFn = i => {
            const z = ctx.zOf(ctx.wordRates[i]);
            const pct = empPct(ctx.sortedValues, ctx.wordRates[i]);
            return { x: z, y: stdNorm(z), label: textNames[i] + ' (' + pct.toFixed(1) + '%)' };
        };
        const grp = groupTraces(groups, ptFn, '%{text}<br>z: %{x:.2f}<extra></extra>');
        for (let k = 0; k < grp.length; k++) traces.push(grp[k]);
    }
    if (ctx.focusZ !== null) {
        traces.push(focusStar(ctx.focusZ, stdNorm(ctx.focusZ),
            focusedTextName + '<br>z=' + ctx.focusZ.toFixed(2) + '<br>' + ctx.focusPercentile.toFixed(1) + '%'));
    }
    if (ctx.evilFocusZ !== null) {
        traces.push(evilFocusStar(ctx.evilFocusZ, stdNorm(ctx.evilFocusZ),
            'Evil focus: ' + evilFocusTextName + '<br>z=' + ctx.evilFocusZ.toFixed(2) + '<br>' + ctx.evilFocusPercentile.toFixed(1) + '%'));
    }
    const low = Math.ceil(zmin), high = Math.floor(zmax);
    const step = Math.max(1, Math.ceil((high - low) / 12));
    const tvals = [], tlabels = [];
    for (let t = low; t <= high; t += step) {
        tvals.push(t);
        tlabels.push(t + '\u03C3<br>' + (normalCdf(t) * 100).toFixed(1) + '%');
    }
    const layout = wordLayout(wordTitle('Z-score distribution of', ctx),
        { title: 'Z-score (std devs from mean) of: ' + ctx.axisTitle,
          zeroline: true, zerolinecolor: '#999',
          range: [zmin, zmax], tickmode: 'array', tickvals: tvals, ticktext: tlabels },
        { title: 'Density (standard normal)' }, 90);
    layout.annotations = [focusAnnotation(ctx)];
    return { traces, layout };
};
const pctView = ctx => {
    const count = ctx.sortedValues.length;
    const cdfX = [ctx.minX], cdfY = [0];
    for (let i = 0; i < count; i++) {
        cdfX.push(ctx.sortedValues[i]);
        cdfY.push(((i + 1) / count) * 100);
    }
    const traces = [
        { x: cdfX, y: cdfY, mode: 'lines', type: 'scatter', name: 'Empirical CDF',
          line: { shape: 'hv', color: '#d93806', width: 3 }, hoverinfo: 'skip' },
        { x: [ctx.minX, ctx.maxX], y: [50, 50], mode: 'lines', type: 'scatter',
          name: 'Median (50th)', line: { color: 'rgba(120,120,120,0.65)', width: 1, dash: 'dot' },
          hoverinfo: 'skip' }
    ];
    const pctOf = r => empPct(ctx.sortedValues, r);
    if (functionWordShowMarkers && !hideGrayPoints) {
        const others = [];
        for (let j = 0; j < ctx.validIndices.length; j++) {
            const idx = ctx.validIndices[j];
            if (idx !== focusedTextIndex && !ctx.isInFocusedCluster(idx) &&
                !(ctx.isInEvilCluster && ctx.isInEvilCluster(idx))) others.push(ctx.wordRates[idx]);
        }
        if (others.length) {
            const ys = [];
            for (let k = 0; k < others.length; k++) ys.push(pctOf(others[k]));
            traces.push({
                x: others, y: ys, mode: 'markers', type: 'scatter', name: 'Other texts',
                marker: { size: 5, color: 'rgba(120,120,120,0.45)',
                          line: { color: 'rgba(80,80,80,0.75)', width: 0.5 } },
                hoverinfo: 'skip'
            });
        }
    }
    if (ctx.clusterRates && ctx.clusterRates.length > 0) {
        const cys = [];
        for (let m = 0; m < ctx.clusterRates.length; m++) cys.push(pctOf(ctx.clusterRates[m]));
        traces.push({
            x: ctx.clusterRates, y: cys, mode: 'markers', type: 'scatter',
            name: 'Focused cluster (n=' + ctx.clusterRates.length + ')',
            marker: { size: 9, color: FOCUS_CLUSTER_COLOR,
                      line: { color: FOCUS_CLUSTER_LINE_COLOR, width: 1.5 } },
            hoverinfo: 'skip'
        });
    }
    if (ctx.evilClusterRates && ctx.evilClusterRates.length > 0) {
        const ecys = [];
        for (let m2 = 0; m2 < ctx.evilClusterRates.length; m2++) ecys.push(pctOf(ctx.evilClusterRates[m2]));
        traces.push({
            x: ctx.evilClusterRates, y: ecys, mode: 'markers', type: 'scatter',
            name: 'Evil cluster (n=' + ctx.evilClusterRates.length + ')',
            marker: { size: 9, color: EVIL_CLUSTER_COLOR,
                      line: { color: EVIL_CLUSTER_LINE_COLOR, width: 1.5 } },
            hoverinfo: 'skip'
        });
    }
    if (ctx.clusterMean !== null) {
        traces.push(clusterDiamond(ctx.clusterMean, ctx.clusterPercentileMean,
            'cluster mean<br>' + ctx.clusterPercentileMean.toFixed(1) + '%'));
    }
    if (ctx.evilClusterMean !== null) {
        traces.push(evilClusterDiamond(ctx.evilClusterMean, ctx.evilClusterPercentileMean,
            'evil cluster mean<br>' + ctx.evilClusterPercentileMean.toFixed(1) + '%'));
    }
    if (ctx.focusRate !== null) {
        traces.push(focusStar(ctx.focusRate, ctx.focusPercentile,
            focusedTextName + '<br>' + ctx.focusPercentile.toFixed(1) + '%'));
        traces.push({
            x: [ctx.focusRate, ctx.focusRate], y: [0, ctx.focusPercentile],
            mode: 'lines', type: 'scatter', name: 'Focus guide',
            line: { color: 'rgba(255, 212, 0, 0.75)', width: 2, dash: 'dash' },
            hoverinfo: 'skip', showlegend: false
        });
    }
    if (ctx.evilFocusRate !== null) {
        traces.push(evilFocusStar(ctx.evilFocusRate, ctx.evilFocusPercentile,
            'Evil focus: ' + evilFocusTextName + '<br>' + ctx.evilFocusPercentile.toFixed(1) + '%'));
    }
    const layout = wordLayout(wordTitle('Percentile of', ctx),
        { title: ctx.axisTitle, range: [ctx.minX, ctx.maxX] },
        { title: 'Percentile', range: [0, 100],
          tickvals: [0, 10, 25, 50, 75, 90, 100], ticksuffix: '%' }, 80);
    return { traces, layout };
};
const densityView = ctx => {
    const at = x => normalPdf(x, ctx.mean, ctx.std);
    const curve = sampleCurve(ctx.minX, ctx.maxX, 240, at);
    const traces = [];
    if (functionWordShowHistogram) traces.push(histTrace(ctx.validValues, ctx.std, ctx.minX, ctx.maxX, 'rate'));
    traces.push({
        x: curve.x, y: curve.y, mode: 'lines', type: 'scatter',
        name: 'Normal fit', line: { color: NORMAL_CURVE_COLOR, width: 3 }, hoverinfo: 'skip'
    });
    if (ctx.clusterMean !== null) traces.push(clusterLine(ctx.clusterMean, at(ctx.clusterMean), ctx));
    if (ctx.evilClusterMean !== null) traces.push(evilClusterLine(ctx.evilClusterMean, at(ctx.evilClusterMean), ctx));
    if (functionWordShowMarkers) {
        const groups = groupIdx(ctx);
        const ptFn = i => {
            return { x: ctx.wordRates[i], y: at(ctx.wordRates[i]), label: textNames[i] };
        };
        const hover = '%{text}<br>rate: %{x:.4f}<extra></extra>';
        const grp = groupTraces(groups, ptFn, hover);
        for (let i = 0; i < grp.length; i++) traces.push(grp[i]);
        if (groups.focus.length) traces.push(focusStar(ctx.focusRate, at(ctx.focusRate), focusedTextName, hover));
        if (ctx.evilFocusRate !== null) {
            traces.push(evilFocusStar(ctx.evilFocusRate, at(ctx.evilFocusRate), 'Evil focus: ' + evilFocusTextName, hover));
        }
    }
    const layout = wordLayout(wordTitle('Distribution of', ctx),
        { title: ctx.axisTitle, range: [ctx.minX, ctx.maxX] },
        { title: 'Density' }, 80);
    return { traces, layout };
};
const updateWordStats = ctx => {
    const body = fwStats(ctx.wordLabel + ' statistics');
    if (!body) return;
    const wrap = document.createElement('div');
    wrap.style.marginTop = '8px';
    const add = (l, v) => { wrap.appendChild(makeDiv(l + ': ' + v, '4px')); };
    if (ctx.words.length > 1) add('Aggregated words', ctx.words.length + ' (rates summed)');
    add('Sample size', ctx.validValues.length);
    add('Mean rate', ctx.mean.toFixed(6));
    add('Std dev', ctx.std.toFixed(6));
    const med = median(ctx.sortedValues);
    add('Median rate', med === null ? 'N/A' : med.toFixed(6));
    add('Min / Max', ctx.dataMin.toFixed(6) + ' / ' + ctx.dataMax.toFixed(6));
    if (ctx.focusRate !== null) {
        add('Focus text rate', ctx.focusRate.toFixed(6) + ' (z=' + ctx.focusZ.toFixed(3) + ')');
        add('Focus percentile (actual)', ctx.focusPercentile.toFixed(2) + '% (' + ordinal(ctx.focusPercentile) + ')');
        add('Focus percentile (normal)', ctx.focusNormalPercentile.toFixed(2) + '% (' + ordinal(ctx.focusNormalPercentile) + ')');
    }
    const fcv = focusedCluster();
    if (fcv !== null && focusClusterMetric) {
        add('Focus cluster', clusterValueLabel(focusClusterMetric, fcv));
    }
    if (ctx.clusterMean !== null) {
        add('Focused cluster mean', ctx.clusterMean.toFixed(6) + (ctx.clusterRates ? ' (n=' + ctx.clusterRates.length + ')' : ''));
        add('Cluster mean z', ctx.zOf(ctx.clusterMean).toFixed(3));
        if (ctx.clusterStd !== null) add('Focused cluster \u03C3', ctx.clusterStd.toFixed(6));
        if (ctx.clusterPercentileMean !== null) {
            add('Cluster mean percentile', ctx.clusterPercentileMean.toFixed(2) + '% (' + ordinal(ctx.clusterPercentileMean) + ')');
        }
    }
    if (ctx.evilClusterMean !== null) {
        add('Evil cluster mean', ctx.evilClusterMean.toFixed(6) + (ctx.evilClusterRates ? ' (n=' + ctx.evilClusterRates.length + ')' : ''));
        add('Evil cluster mean z', ctx.zOf(ctx.evilClusterMean).toFixed(3));
        if (ctx.evilClusterStd !== null) add('Evil cluster \u03C3', ctx.evilClusterStd.toFixed(6));
        if (ctx.evilClusterPercentileMean !== null) {
            add('Evil cluster mean percentile', ctx.evilClusterPercentileMean.toFixed(2) + '% (' + ordinal(ctx.evilClusterPercentileMean) + ')');
        }
    }
    if (ctx.evilFocusRate !== null) {
        add('Evil focus text rate', ctx.evilFocusRate.toFixed(6) + ' (z=' + ctx.evilFocusZ.toFixed(3) + ')');
        add('Evil focus percentile (actual)', ctx.evilFocusPercentile.toFixed(2) + '% (' + ordinal(ctx.evilFocusPercentile) + ')');
    }
    body.appendChild(wrap);
};
const updateFwPlot = () => {
    const agg = aggregatedRates();
    if (!agg) {
        emptyPlot('functionWordPlot', 'Select at least one function word');
        if (clusterStatsState.view !== 'cluster') {
            const c1 = fwStats();
            if (c1) c1.appendChild(makeDiv('No function words selected.'));
        }
        return;
    }
    let valid = 0;
    for (let i = 0; i < agg.rates.length; i++) if (isNum(agg.rates[i])) valid++;
    if (valid < 2) {
        emptyPlot('functionWordPlot', 'Not enough data for ' + wordLabel(agg.words));
        if (clusterStatsState.view !== 'cluster') {
            const c2 = fwStats();
            if (c2) c2.appendChild(makeDiv('Not enough data.'));
        }
        return;
    }
    const ctx = wordCtx(agg.words, agg.rates);
    let view;
    if (functionWordPlotMode === 'zscore') view = zView(ctx);
    else if (functionWordPlotMode === 'cdf') view = pctView(ctx);
    else view = densityView(ctx);
    drawPlot('functionWordPlot', view.traces, view.layout, null)
        .then(() => Plotly.Plots.resize(document.getElementById('functionWordPlot')));
    if (clusterStatsState.view !== 'cluster') updateWordStats(ctx);
};
const computeWordStats = () => {
    if (!Array.isArray(burrowsFunctionWords) || !burrowsWordFrequencies) return null;
    const stats = {}, zscores = {};
    for (let i = 0; i < burrowsFunctionWords.length; i++) {
        const word = burrowsFunctionWords[i];
        const rates = burrowsWordFrequencies[word];
        if (!Array.isArray(rates)) continue;
        const valid = [];
        for (let j = 0; j < rates.length; j++) if (isNum(rates[j])) valid.push(rates[j]);
        if (valid.length < 3) continue;
        const m = mean(valid), s = stddev(valid);
        if (!isFinite(s) || s <= 0) continue;
        stats[word] = true;
        const z = [];
        for (let k = 0; k < rates.length; k++) {
            z.push(isNum(rates[k]) ? (rates[k] - m) / s : null);
        }
        zscores[word] = z;
    }
    return { stats, zscores };
};
const computeDist = () => {
    const ws = deltaState.wordStats;
    if (!ws) return null;
    const zarr = [];
    for (const w in ws.stats) zarr.push(ws.zscores[w]);
    const deltas = [];
    for (let i = 0; i < textNames.length; i++) {
        for (let j = i + 1; j < textNames.length; j++) {
            let sum = 0, cnt = 0;
            for (let z = 0; z < zarr.length; z++) {
                if (zarr[z][i] === null || zarr[z][j] === null) continue;
                sum += Math.abs(zarr[z][i] - zarr[z][j]);
                cnt++;
            }
            if (cnt) deltas.push(sum / cnt);
        }
    }
    deltas.sort((a, b) => a - b);
    return deltas;
};
const distPct = v => {
    const d = deltaState.deltaDistribution;
    if (!d || !d.length) return null;
    return (countLE(d, v) / d.length) * 100;
};
const defaultDeltaPartner = idx => {
    const row = Array.isArray(burrowsSimilarity) ? burrowsSimilarity[idx] : null;
    if (row) {
        let best = -Infinity, bi = -1;
        for (let i = 0; i < row.length; i++) {
            if (i !== idx && isNum(row[i]) && row[i] > best) { best = row[i]; bi = i; }
        }
        if (bi >= 0) return bi;
    }
    return idx === 0 ? Math.min(1, textNames.length - 1) : 0;
};
const deltaPair = (a, b) => {
    const ws = deltaState.wordStats;
    if (!ws) return null;
    const contrib = [];
    let total = 0;
    for (const w in ws.stats) {
        const za = ws.zscores[w][a], zb = ws.zscores[w][b];
        if (za === null || zb === null) continue;
        const absDiff = Math.abs(za - zb);
        const avgZ = (za + zb) / 2;
        const sameDir = za * zb > 0;
        const minAbs = Math.min(Math.abs(za), Math.abs(zb));
        contrib.push({
            word: w, zA: za, zB: zb, absDiff, avgZ,
            rateA: burrowsWordFrequencies[w][a], rateB: burrowsWordFrequencies[w][b],
            driverScore: sameDir ? minAbs * Math.exp(-absDiff) : 0,
            sharedScore: Math.abs(avgZ) * Math.exp(-absDiff),
            deltaContribution: absDiff
        });
        total += absDiff;
    }
    for (let i = 0; i < contrib.length; i++) {
        contrib[i].shareOfDelta = total > 0 ? (contrib[i].deltaContribution / total) * 100 : 0;
    }
    return { contributions: contrib, nWords: contrib.length, totalDelta: total,
             avgDelta: contrib.length > 0 ? total / contrib.length : 0 };
};
const sortContrib = (arr, mode) => {
    const key = DELTA_SORT_KEYS[mode] || 'deltaContribution';
    const copy = arr.slice();
    copy.sort((a, b) => b[key] - a[key]);
    return copy;
};
const deltaPlotHeight = (words, top, bot) => {
    return Math.max(800, words * 24 + top + bot);
};
const fillDeltaSelect = (sel, filter, selIdx) => {
    if (!sel) return null;
    const f = (filter || '').trim().toLowerCase();
    sel.innerHTML = '';
    for (let i = 0; i < textNames.length; i++) {
        if (f && textNames[i].toLowerCase().indexOf(f) === -1) continue;
        const o = document.createElement('option');
        o.value = String(i);
        o.text = textNames[i];
        sel.appendChild(o);
    }
    const has = sel.querySelector('option[value="' + selIdx + '"]');
    if (selIdx !== null && has) sel.value = String(selIdx);
    else if (sel.options.length > 0) sel.value = sel.options[0].value;
    return sel.value === '' ? null : Number(sel.value);
};
const populateDeltaSelects = () => {
    const a = fillDeltaSelect(document.getElementById('deltaTextASelect'),
        document.getElementById('deltaTextASearch').value, deltaState.textA);
    const b = fillDeltaSelect(document.getElementById('deltaTextBSelect'),
        document.getElementById('deltaTextBSearch').value, deltaState.textB);
    if (a !== null) deltaState.textA = a;
    if (b !== null) deltaState.textB = b;
};
const syncDeltaDropdowns = () => {
    const pairs = [['deltaTextASelect', deltaState.textA], ['deltaTextBSelect', deltaState.textB]];
    for (let i = 0; i < pairs.length; i++) {
        const s = document.getElementById(pairs[i][0]);
        if (s && s.querySelector('option[value="' + pairs[i][1] + '"]')) {
            s.value = String(pairs[i][1]);
        }
    }
};
const setDeltaTopN = n => {
    deltaState.topN = n;
    document.getElementById('deltaTopNRange').value = String(n);
    document.getElementById('deltaTopNValue').textContent = n;
};
const initDelta = () => {
    if (deltaState.initialized) return;
    if (!deltaState.wordStats) deltaState.wordStats = computeWordStats();
    if (!deltaState.wordStats) return;
    deltaState.deltaDistribution = computeDist();
    deltaState.totalWordCount = Object.keys(deltaState.wordStats.stats).length;
    document.getElementById('deltaTopNRange').max = String(deltaState.totalWordCount);
    setDeltaTopN(Math.min(deltaState.topN, deltaState.totalWordCount));
    if (deltaState.textA === null) deltaState.textA = focusedTextIndex >= 0 ? focusedTextIndex : 0;
    if (deltaState.textB === null) deltaState.textB = defaultDeltaPartner(deltaState.textA);
    populateDeltaSelects();
    syncDeltaDropdowns();
    deltaState.initialized = true;
};
const recomputeDelta = () => {
    deltaState.wordStats = computeWordStats();
    deltaState.deltaDistribution = computeDist();
    deltaState.totalWordCount = Object.keys(deltaState.wordStats ? deltaState.wordStats.stats : {}).length;
};
const updateDeltaPlot = () => {
    if (!deltaState.wordStats) deltaState.wordStats = computeWordStats();
    if (!deltaState.wordStats) {
        emptyPlot('deltaPlot', 'No Burrows function-word data available');
        updateDeltaStats(null);
        return;
    }
    renderDeltaPair();
};
const drawDelta = (traces, layout, height) => {
    drawPlot('deltaPlot', traces, layout, height).then(() => {
        document.getElementById('deltaContainer').scrollTop = 0;
    });
};
const renderDeltaPair = () => {
    const a = deltaState.textA, b = deltaState.textB;
    if (a === null || b === null || a < 0 || b < 0 || a === b) {
        emptyPlot('deltaPlot', 'Select two different texts to compare');
        updateDeltaStats(null);
        return;
    }
    const pair = deltaPair(a, b);
    if (!pair || !pair.contributions.length) {
        emptyPlot('deltaPlot', 'Insufficient overlapping data for these texts');
        updateDeltaStats(null);
        return;
    }
    const sorted = sortContrib(pair.contributions, deltaState.sortMode);
    const n = Math.min(deltaState.topN, sorted.length);
    const shown = sorted.slice(0, n).reverse();
    let maxDiff = 0.0001;
    for (let i = 0; i < shown.length; i++) if (shown[i].absDiff > maxDiff) maxDiff = shown[i].absDiff;

    const gaps = [];
    for (let j = 0; j < shown.length; j++) {
        const it = shown[j];
        const ratio = it.absDiff / maxDiff;
        const r = Math.round(39 + 208 * ratio);
        const g = Math.round(164 - 94 * ratio);
        const bl = Math.round(217 - 148 * ratio);
        gaps.push({
            x: [Math.min(it.zA, it.zB), Math.max(it.zA, it.zB)],
            y: [it.word, it.word], mode: 'lines', type: 'scatter',
            line: { color: 'rgba(' + r + ', ' + g + ', ' + bl + ', 0.45)', width: 14 },
            hoverinfo: 'skip', showlegend: false
        });
    }

    const hovers = [], words = [], xsA = [], xsB = [];
    for (let k = 0; k < shown.length; k++) {
        const s = shown[k];
        hovers.push('<b>' + s.word + '</b><br>Rate A: ' + (s.rateA * 100).toFixed(3) + '% (z=' + s.zA.toFixed(2) + ')<br>' +
                    'Rate B: ' + (s.rateB * 100).toFixed(3) + '% (z=' + s.zB.toFixed(2) + ')<br>' +
                    '|\u0394z|: ' + s.absDiff.toFixed(3) + ' (' + s.shareOfDelta.toFixed(2) + '% of \u0394)<br>' +
                    'Shared z: ' + s.avgZ.toFixed(2));
        words.push(s.word);
        xsA.push(s.zA);
        xsB.push(s.zB);
    }

    const traces = gaps.slice();
    traces.push({
        x: xsA, y: words, mode: 'markers', type: 'scatter', name: textNames[a],
        text: hovers, hoverinfo: 'text',
        marker: { size: 13, color: '#D81D23', line: { color: '#D81D23', width: 1.5 } }
    });
    traces.push({
        x: xsB, y: words, mode: 'markers', type: 'scatter', name: textNames[b],
        hoverinfo: 'skip',
        marker: { size: 13, color: '#50498A', line: { color: '#50498A', width: 1.5 } }
    });

    const pct = distPct(pair.avgDelta);
    const sub = 'Burrows \u0394 = ' + pair.avgDelta.toFixed(4) +
        (pct !== null ? ' (' + ordinal(pct) + ' percentile among all pairs)' : '') +
        ' | ' + pair.nWords + ' function words | showing top ' + n +
        (n < pair.nWords ? ' of ' + pair.nWords + ' (raise slider or press "Show all")' : '');

    const layout = {
        title: {
            text: '\u0394 Delta Explorer: <b>' + textNames[a] + '</b> and <b>' + textNames[b] + '</b>' +
                  '<br><sub>' + sub + '<br>Sorted by: ' + (DELTA_SORT_LABELS[deltaState.sortMode] || deltaState.sortMode) + '</sub>',
            x: 0.5, xanchor: 'center'
        },
        xaxis: {
            title: 'Standardized rate (z-score across corpus)',
            zeroline: true, zerolinecolor: '#333', zerolinewidth: 2,
            showgrid: true, gridcolor: '#eee'
        },
        yaxis: {
            title: '', automargin: true, tickfont: { size: 11 },
            type: 'category', categoryorder: 'array',
            categoryarray: words, range: [-0.5, words.length - 0.5]
        },
        plot_bgcolor: '#f8f9fa', paper_bgcolor: 'white',
        hovermode: 'closest', showlegend: true,
        legend: { orientation: 'h', yanchor: 'bottom', y: 1.0, xanchor: 'right', x: 1 },
        font: { family: 'Arial, sans-serif' },
        margin: { t: 120, l: 130, r: 60, b: 70 }
    };
    drawDelta(traces, layout, deltaPlotHeight(shown.length, 120, 70));
    updateDeltaStats(pair);
};
const deltaRow = (word, main, sec) => {
    const r = document.createElement('div');
    r.className = 'deltaWordRow';
    const a = document.createElement('span');
    a.textContent = word;
    const b = document.createElement('span');
    b.className = 'val';
    b.innerHTML = main + (sec ? '<br>' + sec : '');
    r.appendChild(a);
    r.appendChild(b);
    return r;
};
const deltaHead = text => {
    const h = document.createElement('strong');
    h.textContent = text;
    h.style.cssText = 'font-size:12px; display:block; margin-bottom:6px;';
    return h;
};
const deltaNote = (html, css) => {
    const n = document.createElement('div');
    n.style.cssText = css;
    n.innerHTML = html;
    return n;
};
const deltaSection = (title, color, count, buildList) => {
    const det = document.createElement('details');
    det.style.marginTop = '6px';
    const sum = document.createElement('summary');
    sum.className = 'deltaSectionHead';
    sum.style.cssText = 'cursor: pointer; user-select: none; color: ' + color + ';';
    sum.textContent = title + ' (' + count + ')';
    det.appendChild(sum);
    const listWrap = document.createElement('div');
    listWrap.style.cssText = 'max-height: 320px; overflow-y: auto; padding-right: 4px; margin-top: 2px;';
    buildList(listWrap);
    det.appendChild(listWrap);
    return det;
};
const updateDeltaStats = pair => {
    const body = clearBody('deltaStats');
    if (!body) return;
    if (!pair) {
        const n = makeDiv('No pair selected.');
        n.style.fontSize = '11px';
        body.appendChild(n);
        return;
    }
    const a = deltaState.textA, b = deltaState.textB;
    const pc = r => (r * 100).toFixed(2);

    body.appendChild(deltaHead('\u0394 Delta Explorer'));
    body.appendChild(deltaNote(
        '<b>Text A:</b> ' + textNames[a] + '<br><b>Text B:</b> ' + textNames[b],
        'font-size:10px; color:#555; margin-bottom:8px;'));

    const pct = distPct(pair.avgDelta);
    body.appendChild(deltaNote(
        '<b>Burrows \u0394 = ' + pair.avgDelta.toFixed(4) + '</b>' +
        (pct !== null ? '<br>' + ordinal(pct) + ' percentile of all ' + deltaState.deltaDistribution.length + ' pairs' +
        (deltaState.deltaDistribution.length > 1000000 ? ' (that\'s a lot of pairs!)' : '') +
                       '<br><span style="color:#555;">(lower = more similar)</span>' : '') +
        '<br>Words compared: ' + pair.nWords,
        'border:1px solid #ddd; padding:6px; border-radius:3px; margin-bottom:8px; font-size:10px;'));

    const seps = sortContrib(pair.contributions, 'separators');
    body.appendChild(deltaSection(
        'Top separators (|\u0394z| high)', '#ff0000', seps.length,
        listWrap => {
            for (let i = 0; i < seps.length; i++) {
                const s = seps[i];
                listWrap.appendChild(deltaRow(
                    (i + 1) + '. ' + s.word,
                    '|\u0394z|=' + s.absDiff.toFixed(2),
                    s.shareOfDelta.toFixed(1) + '% of \u0394<br>A: ' + pc(s.rateA) + '% (z=' + s.zA.toFixed(2) + ')<br>B: ' + pc(s.rateB) + '% (z=' + s.zB.toFixed(2) + ')'
                ));
            }
        }));

    const drv = sortContrib(pair.contributions, 'drivers');
    body.appendChild(deltaSection(
        'Similarity drivers (both distinctive, same direction)', '#008000', drv.length,
        listWrap => {
            listWrap.appendChild(deltaNote(
                'words that bind them together.',
                'font-size:9px; color:#777; margin-bottom:4px;'));
            for (let j = 0; j < drv.length; j++) {
                const d = drv[j];
                listWrap.appendChild(deltaRow(
                    (j + 1) + '. ' + d.word,
                    '',
                    '|\u0394z|=' + d.absDiff.toFixed(2) + ' | shared z=' + d.avgZ.toFixed(2) +
                    '<br>A: ' + pc(d.rateA) + '% | B: ' + pc(d.rateB) + '%'
                ));
            }
        }));

    const shr = sortContrib(pair.contributions, 'shared');
    body.appendChild(deltaSection(
        'Shared tendencies (both unusual, same direction)', '#ebce13', shr.length,
        listWrap => {
            for (let k = 0; k < shr.length; k++) {
                const sh = shr[k];
                listWrap.appendChild(deltaRow(
                    (k + 1) + '. ' + sh.word,
                    '',
                    (sh.avgZ >= 0 ? 'high' : 'low') + ' | avg z=' + sh.avgZ.toFixed(2) +
                    '<br>A: ' + pc(sh.rateA) + '% | B: ' + pc(sh.rateB) + '%'
                ));
            }
        }));

    const exWrap = document.createElement('div');
    exWrap.style.cssText = 'margin-top:8px; font-size:9px; color:#555;';
    const addExt = (label, zk, rk, m) => {
        const h = makeDiv('Most distinctive words in ' + label + ':');
        h.style.cssText = 'font-weight:600; margin:' + m + ';';
        exWrap.appendChild(h);
        let cp = pair.contributions.slice();
        cp.sort((x, y) => Math.abs(y[zk]) - Math.abs(x[zk]));
        cp = cp.slice(0, 4);
        for (let q = 0; q < cp.length; q++) {
            exWrap.appendChild(makeDiv(cp[q].word + ': z=' + cp[q][zk].toFixed(2) + ' (rate ' + pc(cp[q][rk]) + '%)'));
        }
    };
    addExt('A (' + textNames[a] + ')', 'zA', 'rateA', '0 0 3px 0');
    addExt('B (' + textNames[b] + ')', 'zB', 'rateB', '5px 0 3px 0');
    body.appendChild(exWrap);
};
const clusterIds = labels => {
    const out = [], seen = {};
    for (let i = 0; i < labels.length; i++) {
        if (!isNum(labels[i])) continue;
        const v = Number(labels[i]);
        if (!seen[v]) { seen[v] = true; out.push(v); }
    }
    out.sort((a, b) => a - b);
    return out;
};
const clusterCount = (labels, id) => {
    const target = Number(id);
    let n = 0;
    for (let i = 0; i < labels.length; i++) {
        if (isNum(labels[i]) && Number(labels[i]) === target) n++;
    }
    return n;
};
const defaultCluster = (labels, ids) => {
    if (focusedTextIndex >= 0 && labels && isNum(labels[focusedTextIndex])) {
        const fcv = Number(labels[focusedTextIndex]);
        if (ids.indexOf(fcv) !== -1) return fcv;
    }
    return ids[0];
};
const compareCluster = (ids, excludeId) => {
    const exclude = asNum(excludeId);
    for (let i = 0; i < ids.length; i++) if (ids[i] !== exclude) return ids[i];
    return null;
};
const initClusterState = () => {
    if (!deltaState.wordStats) deltaState.wordStats = computeWordStats();
    if (!clusterMetricKeys.length) return;
    if (clusterStatsState.source === null) {
        clusterStatsState.source = focusClusterMetric || clusterMetricKeys[0];
    }
    const labels = metricsData[clusterStatsState.source];
    const ids = clusterIds(labels);
    if (!ids.length) return;
    if (clusterStatsState.clusterId === null || ids.indexOf(asNum(clusterStatsState.clusterId)) === -1) {
        clusterStatsState.clusterId = defaultCluster(labels, ids);
    }
    if (clusterStatsState.compareMode === 'cluster' && clusterStatsState.compareClusterId !== null) {
        const cmp = asNum(clusterStatsState.compareClusterId);
        const missing = ids.indexOf(cmp) === -1;
        const same = cmp === asNum(clusterStatsState.clusterId);
        if (missing || same) clusterStatsState.compareClusterId = null;
    }
};
const clusterDecomp = (source, clusterId, compareClusterId) => {
    const labels = metricsData[source];
    const ws = deltaState.wordStats;
    if (!labels || !ws) return null;
    const cid = asNum(clusterId);
    const cmpId = asNum(compareClusterId);
    const membersA = [], membersB = [];
    for (let i = 0; i < labels.length; i++) {
        if (!isNum(labels[i])) continue;
        const lv = Number(labels[i]);
        if (lv === cid) membersA.push(i);
        else if (cmpId !== null && lv === cmpId) membersB.push(i);
    }
    if (!membersA.length) return null;
    if (cmpId !== null && !membersB.length) return null;

    const meanPerWord = members => {
        const out = {};
        for (const w in ws.stats) {
            const z = ws.zscores[w];
            let sum = 0, sumSq = 0, n = 0;
            for (let m = 0; m < members.length; m++) {
                const v = z[members[m]];
                if (v === null || v === undefined || isNaN(v)) continue;
                sum += v; sumSq += v * v; n++;
            }
            if (!n) continue;
            const mn = sum / n;
            const varV = n > 1 ? Math.max(0, (sumSq - n * mn * mn) / n) : 0;
            out[w] = { mean: mn, sd: Math.sqrt(varV), n };
        }
        return out;
    };

    const meanA = meanPerWord(membersA);
    const meanB = cmpId !== null ? meanPerWord(membersB) : null;

    const contribs = [];
    let total = 0;
    for (const w2 in meanA) {
        const a = meanA[w2];
        let delta, bMean = null, bSd = null, bN = null;
        if (cmpId === null) {
            delta = a.mean;
        } else {
            const b = meanB[w2];
            if (!b) continue;
            delta = a.mean - b.mean;
            bMean = b.mean; bSd = b.sd; bN = b.n;
        }
        contribs.push({
            word: w2, delta, meanA: a.mean, sdA: a.sd, nA: a.n,
            meanB: bMean, sdB: bSd, nB: bN
        });
        total += Math.abs(delta);
    }

    for (let j = 0; j < contribs.length; j++) {
        const c = contribs[j];
        c.share = total > 0 ? Math.abs(c.delta) / total * 100 : 0;
        c.sign = c.delta >= 0 ? 'over' : 'under';
        c.consistency = (c.sdA > 1e-9) ? Math.abs(c.meanA) / c.sdA : (Math.abs(c.meanA) > 1e-12 ? 99 : 0);
    }

    return {
        source: source, clusterId: cid, compareClusterId: cmpId,
        membersA: membersA.length, membersB: membersB.length,
        contributions: contribs, totalDelta: total,
        avgDelta: contribs.length ? total / contribs.length : 0
    };
};
const filterContribs = (contribs, state) => {
    let arr = contribs.slice();
    if (state.signFilter === 'over') {
        arr = arr.filter(c => c.delta >= 0);
    } else if (state.signFilter === 'under') {
        arr = arr.filter(c => c.delta < 0);
    }
    if (state.sortMode === 'share') arr.sort((a, b) => b.share - a.share);
    else if (state.sortMode === 'signed') arr.sort((a, b) => b.delta - a.delta);
    else if (state.sortMode === 'consistency') arr.sort((a, b) => b.consistency - a.consistency);
    return arr;
};
const selectWord = (word, additive) => {
    if (additive) functionWordSelection.add(word);
    else {
        functionWordSelection.clear();
        functionWordSelection.add(word);
    }
    updateWordSummary();
    populateWordControls();
    if (viewMode === 'functionWord') updateFwPlot();
};

const wordRow = (t, maxAbs) => {
    const row = document.createElement('div');
    row.className = 'clusterWordRow';
    row.title = 'Click to select "' + t.word + '" in the plot';
    const word = document.createElement('span');
    word.className = 'cword';
    word.textContent = t.word;
    const bar = document.createElement('span');
    bar.className = 'cbar';
    const fill = document.createElement('span');
    fill.className = 'cfill';
    fill.style.width = Math.min(100, Math.abs(t.delta) / maxAbs * 100) + '%';
    fill.style.background = t.delta >= 0
        ? 'linear-gradient(90deg, #fbd0d0, #E85A93)'
        : 'linear-gradient(90deg, #d0d6fb, #2FA3E6)';
    bar.appendChild(fill);
    const val = document.createElement('span');
    val.className = 'cval';
    val.textContent = t.share.toFixed(1) + '%';
    val.title = 'z = ' + (t.delta >= 0 ? '+' : '') + t.delta.toFixed(3);
    row.appendChild(word); row.appendChild(bar); row.appendChild(val);
    row.addEventListener('click', e => {
        selectWord(t.word, e.ctrlKey || e.metaKey || e.shiftKey);
    });
    return row;
};

const tableRow = (t, i) => {
    const row = document.createElement('div');
    row.className = 'clusterTableRow';
    row.title = 'Click to select "' + t.word + '" in the plot';
    const rank = document.createElement('span');
    rank.className = 'cti';
    rank.textContent = (i + 1) + '.';
    const word = document.createElement('span');
    word.className = 'ctw';
    word.textContent = t.word;
    const z = document.createElement('span');
    z.className = 'ctv';
    z.textContent = fmtDelta(t.delta, 2);
    z.style.color = t.delta >= 0 ? CLUSTER_BAR_TEXT_POS_COLOR : CLUSTER_BAR_TEXT_NEG_COLOR;
    const sh = document.createElement('span');
    sh.className = 'ctv';
    sh.textContent = t.share.toFixed(1) + '%';
    row.appendChild(rank);
    row.appendChild(word);
    row.appendChild(z);
    row.appendChild(sh);
    row.addEventListener('click', e => {
        selectWord(t.word, e.ctrlKey || e.metaKey || e.shiftKey);
    });
    return row;
};
const renderInspector = (decomp, word) => {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'margin:6px 0; padding:5px 6px; border:1px solid #ddd; border-radius:2px; font-size:10px; line-height:1.45;';
    const head = document.createElement('div');
    head.style.cssText = 'font-weight:600; margin-bottom:3px; font-size:11px;';
    wrap.appendChild(head);
    head.textContent = 'Cluster stats for: ' + word;

    if (!word) {
        wrap.appendChild(makeDiv('Pick a function word above to inspect its contribution.'));
        return wrap;
    }
    let entry = null;
    for (let i = 0; i < decomp.contributions.length; i++) {
        if (decomp.contributions[i].word === word) { entry = decomp.contributions[i]; break; }
    }
    if (!entry) {
        wrap.appendChild(makeDiv('"' + word + '" is not present in this comparison.'));
        return wrap;
    }
    const sorted = decomp.contributions.slice().sort((a, b) => b.share - a.share);
    const rank = sorted.indexOf(entry) + 1;

    const signLbl = entry.delta >= 0 ? 'over-used' : 'under-used';
    const rows = [
        `z\u0304 = ${fmtDelta(entry.delta, 3)} (${signLbl})`,
        'share of \u0394 = ' + entry.share.toFixed(2) + '% (rank ' + rank + '/' + decomp.contributions.length + ')',
        'consistency = ' + entry.consistency.toFixed(2) + ' (\u03C3 within cluster = ' + entry.sdA.toFixed(3) + ')',
        'n (cluster A) = ' + entry.nA
    ];
    if (decomp.compareClusterId !== null && entry.meanB !== null) {
        rows.push('A z\u0304 = ' + entry.meanA.toFixed(3) + ' vs B z\u0304 = ' + entry.meanB.toFixed(3) + ' (n=' + entry.nB + ')');
    }
    for (let j = 0; j < rows.length; j++) wrap.appendChild(makeDiv(rows[j], '1px'));
    return wrap;
};
const statRow = (label, control) => {
    const r = document.createElement('div');
    r.style.cssText = 'display:flex; justify-content:space-between; align-items:center; gap:4px; margin:1px 0;';
    const l = document.createElement('span');
    l.textContent = label;
    r.appendChild(l);
    r.appendChild(control);
    return r;
};
const makeSelect = (label, opts, currentValue, onChange) => {
    const sel = document.createElement('select');
    sel.style.cssText = 'font-size:10px; max-width:150px;';
    const cv = currentValue === null || currentValue === undefined ? '' : String(currentValue);
    for (let i = 0; i < opts.length; i++) {
        const o = document.createElement('option');
        o.value = String(opts[i][0]);
        o.text = opts[i][1];
        if (o.value === cv) o.selected = true;
        sel.appendChild(o);
    }
    sel.addEventListener('change', () => { onChange(sel.value); });
    return statRow(label, sel);
};
const searchRow = (labelText, selectedIdx, onChange) => {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'margin:4px 0;';
    const lbl = document.createElement('div');
    lbl.textContent = labelText;
    lbl.style.fontSize = '10px';
    lbl.style.fontWeight = '600';
    wrap.appendChild(lbl);
    const search = document.createElement('input');
    search.type = 'text';
    search.placeholder = 'Search texts...';
    search.className = 'searchableMetricInput';
    search.style.fontSize = '10px';
    search.style.margin = '2px 0';
    wrap.appendChild(search);
    const sel = document.createElement('select');
    sel.style.cssText = 'font-size:10px; width:100%;';
    wrap.appendChild(sel);

    const populate = filter => {
        const f = (filter || '').trim().toLowerCase();
        sel.innerHTML = '';
        const noneOpt = document.createElement('option');
        noneOpt.value = '';
        noneOpt.text = '\u2014 none \u2014';
        sel.appendChild(noneOpt);
        for (let i = 0; i < textNames.length; i++) {
            if (f && textNames[i].toLowerCase().indexOf(f) === -1) continue;
            const o = document.createElement('option');
            o.value = String(i);
            o.text = textNames[i];
            sel.appendChild(o);
        }
        if (selectedIdx !== null && selectedIdx >= 0 &&
            sel.querySelector('option[value="' + selectedIdx + '"]')) {
            sel.value = String(selectedIdx);
        } else {
            sel.value = '';
        }
    };
    populate('');
    search.addEventListener('input', () => { populate(search.value); });
    sel.addEventListener('change', () => {
        const v = sel.value === '' ? null : Number(sel.value);
        onChange(v);
    });
    return wrap;
};
const evilClusterMean = () => {
    if (evilFocusTextIndex < 0) return null;
    const labels = metricsData[clusterStatsState.source];
    if (!labels || !isNum(labels[evilFocusTextIndex])) return null;
    const evilClusterId = Number(labels[evilFocusTextIndex]);
    if (evilClusterId === asNum(clusterStatsState.clusterId)) return null;
    const ws = deltaState.wordStats;
    if (!ws) return null;
    let sum = 0, numWords = 0;
    for (const w in ws.stats) {
        const zs = ws.zscores[w];
        let wordSum = 0, wordN = 0;
        for (let i = 0; i < labels.length; i++) {
            if (!isNum(labels[i]) || Number(labels[i]) !== evilClusterId) continue;
            const v = zs[i];
            if (v !== null && v !== undefined && !isNaN(v)) { wordSum += v; wordN++; }
        }
        if (wordN > 0) { sum += Math.abs(wordSum / wordN); numWords++; }
    }
    if (!numWords) return null;
    return {
        clusterId: evilClusterId,
        n: clusterCount(labels, evilClusterId),
        meanAbsZ: sum / numWords,
        numWords
    };
};
const clusterControls = () => {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'margin:6px 0; font-size:10px; line-height:1.5;';

    const labels = metricsData[clusterStatsState.source];
    const ids = clusterIds(labels);
    const focusedClusterId = (focusedTextIndex >= 0 && labels && isNum(labels[focusedTextIndex]))
        ? Number(labels[focusedTextIndex]) : null;
    const curClusterId = asNum(clusterStatsState.clusterId);

    wrap.appendChild(makeSelect('Cluster source:',
        clusterMetricKeys.map(k => [k, clusterLabel(k)]),
        clusterStatsState.source,
        v => {
            clusterStatsState.source = v;
            /* Sync the focus cluster metric so the yellow dots on the
             * function-word plot follow the newly-selected cluster source. */
            focusClusterMetric = fillClusterSelect(
                document.getElementById('focusClusterMetricSelect'), v
            );
            clusterStatsState.clusterId = null;
            clusterStatsState.clusterIdUserSelected = false;
            clusterStatsState.compareClusterId = null;
            evilFocusTextIndex = -1;
            evilFocusTextName = null;
            renderClusterStats();
            updateFwPlot();
        }));

    wrap.appendChild(makeSelect('Cluster:',
        ids.map(id => {
            const n = clusterCount(labels, id);
            const lbl = id < 0 ? 'unclustered (n=' + n + ')' : 'cluster ' + id + ' (n=' + n + ')';
            return [id, id === focusedClusterId ? lbl + ' \u2190 focus' : lbl];
        }),
        curClusterId,
        v => {
            clusterStatsState.clusterId = Number(v);
            clusterStatsState.clusterIdUserSelected = true;
            evilFocusTextIndex = -1;
            evilFocusTextName = null;
            renderClusterStats();
            updateFwPlot();
        }));

    wrap.appendChild(makeSelect('Compare:',
        [['corpus', 'Corpus mean'], ['cluster', 'Another cluster']],
        clusterStatsState.compareMode,
        v => {
            clusterStatsState.compareMode = v;
            if (v === 'corpus') {
                evilFocusTextIndex = -1;
                evilFocusTextName = null;
                clusterStatsState.compareClusterId = null;
            } else {
                if (clusterStatsState.compareClusterId === null ||
                    asNum(clusterStatsState.compareClusterId) === curClusterId) {
                    clusterStatsState.compareClusterId = compareCluster(ids, curClusterId);
                }
            }
            renderClusterStats();
            updateFwPlot();
        }));

    if (clusterStatsState.compareMode === 'cluster') {
        const evilRow = searchRow('Evil focus (text):', evilFocusTextIndex >= 0 ? evilFocusTextIndex : null, idx => {
            if (idx === null) {
                evilFocusTextIndex = -1;
                evilFocusTextName = null;
            } else {
                evilFocusTextIndex = idx;
                evilFocusTextName = textNames[idx];
                const labels2 = metricsData[clusterStatsState.source];
                if (labels2 && isNum(labels2[idx])) {
                    const evilCluster = Number(labels2[idx]);
                    clusterStatsState.compareClusterId = evilCluster !== curClusterId ? evilCluster : null;
                }
            }
            renderClusterStats();
            updateFwPlot();
        });
        wrap.appendChild(evilRow);

        const vsOpts = [['', '\u2014 none \u2014']];
        ids.filter(id => id !== curClusterId).forEach(id => {
            const n = clusterCount(labels, id);
            vsOpts.push([id, id < 0 ? 'unclustered (n=' + n + ')' : 'cluster ' + id + ' (n=' + n + ')']);
        });
        wrap.appendChild(makeSelect('vs cluster:', vsOpts, asNum(clusterStatsState.compareClusterId),
            v => {
                clusterStatsState.compareClusterId = v === '' ? null : Number(v);
                evilFocusTextIndex = -1;
                evilFocusTextName = null;
                renderClusterStats();
                updateFwPlot();
            }));

        const evilInfo = evilClusterMean();
        if (evilFocusTextIndex >= 0) {
            const box = document.createElement('div');
            box.style.cssText = 'margin:3px 0 4px 0; padding:4px 6px; background:#faf5ff; border:1px solid #d8b4fe; border-radius:3px; font-size:9px; line-height:1.4; color:#6b21a8;';
            box.innerHTML = '<b>Evil focus:</b> ' + evilFocusTextName +
                (evilInfo
                    ? '<br>Cluster ' + evilInfo.clusterId + ' (n=' + evilInfo.n + ')<br>Mean |z\u0304| = ' + evilInfo.meanAbsZ.toFixed(4)
                    : '<br><i>Same cluster as focus; evil cluster suppressed.</i>');
            wrap.appendChild(box);
        }
    }

    wrap.appendChild(makeSelect('Sort:',
        [['share', 'Contribution'], ['signed', 'Signed z'], ['consistency', 'Consistency']],
        clusterStatsState.sortMode,
        v => { clusterStatsState.sortMode = v; renderClusterStats(); }));

    wrap.appendChild(makeSelect('Sign:',
        [['both', 'Both signs'], ['over', 'Over-used'], ['under', 'Under-used']],
        clusterStatsState.signFilter,
        v => { clusterStatsState.signFilter = v; renderClusterStats(); }));

    const wordOpts = [['', '\u2014 pick a word \u2014']];
    if (deltaState.wordStats) {
        const ws = Object.keys(deltaState.wordStats.stats).sort();
        for (let i = 0; i < ws.length; i++) wordOpts.push([ws[i], ws[i]]);
    }
    wrap.appendChild(makeSelect('Check a word:', wordOpts, clusterStatsState.inspectWord || '',
        v => {
            clusterStatsState.inspectWord = v || null;
            renderClusterStats();
        }));

    return wrap;
};
const renderClusterStats = () => {
    const c = document.getElementById('functionWordStatsContent');
    if (!c) return;
    c.innerHTML = '';

    if (!clusterMetricKeys.length) { c.appendChild(makeDiv('No cluster metrics available.')); return; }
    if (!deltaState.wordStats) deltaState.wordStats = computeWordStats();
    if (!deltaState.wordStats) { c.appendChild(makeDiv('No function-word data available.')); return; }

    initClusterState();
    const source = clusterStatsState.source;
    const clusterId = clusterStatsState.clusterId;
    if (source === null || clusterId === null) { c.appendChild(makeDiv('No cluster selected.')); return; }

    const compareClusterId = clusterStatsState.compareMode === 'cluster'
        ? asNum(clusterStatsState.compareClusterId) : null;
    const decomp = clusterDecomp(source, clusterId, compareClusterId);

    let filtered = [];
    if (decomp) {
        filtered = filterContribs(decomp.contributions, clusterStatsState);
        if (!clusterStatsState.inspectWord && filtered.length) {
            clusterStatsState.inspectWord = filtered[0].word;
        }
    }

    const head = document.createElement('strong');
    head.textContent = 'Cluster ' + clusterId + ' \u2014 ' + clusterLabel(source);
    c.appendChild(head);
    c.appendChild(clusterControls());

    if (!decomp) { c.appendChild(makeDiv('No data for this cluster.')); return; }
    let totalValid = 0;
    {
        const seen = {};
        for (const w0 in deltaState.wordStats.stats) {
            const zs0 = deltaState.wordStats.zscores[w0];
            for (let i0 = 0; i0 < zs0.length; i0++) if (isNum(zs0[i0])) seen[i0] = true;
        }
        totalValid = Object.keys(seen).length;
    }
    const clusterIsCorpus = compareClusterId === null && totalValid > 0 && decomp.membersA === totalValid;
    if (clusterIsCorpus) {
        c.appendChild(deltaNote(
            '<b>This cluster contains every text with valid data</b>, so its centroid <i>is</i> the corpus centroid. ' +
            '\u0394 = 0 and every z\u0304 = 0 by definition. Add more texts, or switch to "Compare: Another cluster", to see non-zero contributions.',
            'font-size:9px; color:#7a0403; background:#fff8f8; border:1px solid #ffd4d4; padding:5px 7px; border-radius:3px; margin:6px 0; line-height:1.4; word-wrap:break-word; overflow-wrap:anywhere;'
        ));
    }
    const cmpLbl = compareClusterId === null
        ? 'vs full corpus centroid' : 'vs cluster ' + compareClusterId;
    const summary = document.createElement('div');
    summary.style.cssText = 'border:1px solid #ddd; padding:5px 6px; border-radius:3px; margin:6px 0; font-size:10px; line-height:1.4;';
    summary.innerHTML = '<b>\u0394 = ' + fmt(decomp.avgDelta, 4) + '</b> ' + cmpLbl + '<br>' +
        'Cluster n = ' + decomp.membersA + ' | Words = ' + decomp.contributions.length +
        (decomp.compareClusterId !== null ? '<br>Compare cluster n = ' + decomp.membersB : '');
    c.appendChild(summary);

    const topN = 12;
    const top = filtered.slice(0, topN);
    if (top.length) {
        let maxAbs = 0;
        for (let i = 0; i < top.length; i++) maxAbs = Math.max(maxAbs, Math.abs(top[i].delta));
        if (maxAbs === 0) maxAbs = 1;

        const barHead = makeDiv('Top contributing words', '4px');
        barHead.style.cssText = 'font-weight:600; font-size:11px; margin-top:4px;';
        c.appendChild(barHead);
        for (let j = 0; j < top.length; j++) c.appendChild(wordRow(top[j], maxAbs));

        if (filtered.length > topN) {
            let cum = 0;
            for (let k = 0; k < topN; k++) cum += filtered[k].share;
            const p = makeDiv('Top ' + topN + ' explain ' + cum.toFixed(1) + '% of \u0394', '0');
            p.style.cssText = 'margin-top:3px; font-size:9px;';
            c.appendChild(p);
        }
    } else {
        c.appendChild(makeDiv('No words match the current filter.'));
    }

    c.appendChild(renderInspector(decomp, clusterStatsState.inspectWord));

    if (filtered.length) {
        const det = document.createElement('details');
        det.style.marginTop = '6px';
        const sum = document.createElement('summary');
        sum.className = 'deltaSectionHead';
        sum.style.cssText = 'cursor:pointer; user-select:none; color:#7a0403;';
        sum.textContent = 'All function words (' + filtered.length + ')';
        det.appendChild(sum);
        const listWrap = document.createElement('div');
        listWrap.style.cssText = 'max-height:320px; overflow-y:auto; padding-right:4px; margin-top:2px;';
        for (let t = 0; t < filtered.length; t++) listWrap.appendChild(tableRow(filtered[t], t));
        det.appendChild(listWrap);
        c.appendChild(det);
    }
};
const updateViewVis = () => {
    for (const mode in VIEW_ELEMENT_IDS) {
        const ids = VIEW_ELEMENT_IDS[mode];
        for (let i = 0; i < ids.length; i++) {
            const el = document.getElementById(ids[i]);
            if (el) el.style.display = mode === viewMode ? 'block' : 'none';
        }
    }
};
const updateSwitchLabel = () => {
    const b = document.getElementById('toggleViewBtn');
    if (!b) return;
    if (viewMode !== 'umap') b.textContent = 'Switch to Map View';
    else if (mapMode === 'burrows') b.textContent = 'Switch to Function Word Frequency';
    else b.textContent = 'Switch to Correlation View';
};
const setView = mode => {
    const isMap = mode === 'umap';
    const panels = ['metricPanel', 'colorPanel', 'textPanel'];
    for (let i = 0; i < panels.length; i++) {
        const p = document.getElementById(panels[i]);
        if (!p) continue;
        p.style.display = isMap ? (panels[i] === 'textPanel' ? 'flex' : 'block') : 'none';
    }
    viewMode = mode;
    updateViewVis();
    updateSwitchLabel();
    if (viewMode === 'correlation') updateCorrPlot();
    else if (viewMode === 'functionWord') {
        updateFwPlot();
        if (clusterStatsState.view === 'cluster') renderClusterStats();
    } else if (viewMode === 'delta') { initDelta(); updateDeltaPlot(); }
};
const coordsFor = mode => {
    if (!snapshotXY()) {
        const t = mainPlot && mainPlot.data && mainPlot.data[0];
        if (t && t.x && t.y && t.x.length > 0) {
            originalPlotX = Array.prototype.slice.call(t.x);
            originalPlotY = Array.prototype.slice.call(t.y);
        } else return null;
    }
    if (mode === 'burrows' && hasBurrowsData()) {
        const xs = [], ys = [];
        for (let i = 0; i < burrowsReduced.length; i++) {
            xs.push(burrowsReduced[i][0]);
            ys.push(burrowsReduced[i][1]);
        }
        return { x: xs, y: ys };
    }
    return { x: originalPlotX, y: originalPlotY };
};
const setMapMode = mode => {
    if (mode === mapMode) return;
    if (mode === 'burrows' && !hasBurrowsData()) return;
    const c = coordsFor(mode);
    if (!c || c.x.length === 0 || c.x.length !== c.y.length) return;
    if (viewMode !== 'umap') setView('umap');
    mapMode = mode;
    const isB = mode === 'burrows';
    document.getElementById('burrowsIndicator').style.display = isB ? 'block' : 'none';
    document.getElementById('deltaExplorerBtn').style.display = isB ? 'block' : 'none';
    const eb = document.getElementById('experimentalBtn');
    eb.classList.toggle('active', isB);
    eb.textContent = isB ? 'Return to semantics mode' : 'Experimental';

    Plotly.restyle(mainPlot, { x: [c.x], y: [c.y] }, [0])
        .then(() => resetAxes(mode))
        .then(updateOverlay)
        .then(() => overlayFront(mainPlot, FOCUS_OVERLAY_META))
        .then(refreshConnections)
        .then(() => {
            if (isB) {
                focusSimilarityMode = 'burrows';
                document.getElementById('focusSimilarityModeSelect').value = 'burrows';
            } else {
                focusSimilarityMode = 'cosine';
                document.getElementById('focusSimilarityModeSelect').value = 'cosine';
            }
            applyCluster(mode);
            refreshFocusMarker();
            updateTextList();
            updateSwitchLabel();
            if (viewMode === 'functionWord') updateFwPlot();
        });
};
const hasBurrowsData = () => {
    return Array.isArray(burrowsReduced) && burrowsReduced.length > 0;
};
const exportMap = () => {
    const c = coordsFor(mapMode);
    if (!c || !c.x || !c.y || !c.x.length || !c.y.length) {
        alert('No map data to export. The map may still be initialising.');
        return;
    }
    const xs = Array.prototype.slice.call(c.x);
    const ys = Array.prototype.slice.call(c.y);
    if (xs.length !== ys.length) { alert('Map data is malformed.'); return; }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < xs.length; i++) {
        if (xs[i] < minX) minX = xs[i];
        if (xs[i] > maxX) maxX = xs[i];
        if (ys[i] < minY) minY = ys[i];
        if (ys[i] > maxY) maxY = ys[i];
    }
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    let xr = [minX - spanX * 0.04, maxX + spanX * 0.04];
    let yr = [minY - spanY * 0.04, maxY + spanY * 0.04];
    const da = (xr[1] - xr[0]) / (yr[1] - yr[0]);
    const MAX = 3200, MIN = 700;
    let w, h;
    if (da >= 1) { w = MAX; h = Math.round(MAX / da); }
    else { h = MAX; w = Math.round(MAX * da); }
    w = Math.max(MIN, w); h = Math.max(MIN, h);
    const ca = w / h;
    const cx = (xr[0] + xr[1]) / 2;
    const cy = (yr[0] + yr[1]) / 2;
    let xs2 = xr[1] - xr[0], ys2 = yr[1] - yr[0];
    if (ca > xs2 / ys2) xs2 = ys2 * ca;
    else ys2 = xs2 / ca;
    xr = [cx - xs2 / 2, cx + xs2 / 2];
    yr = [cy - ys2 / 2, cy + ys2 / 2];

    const hidden = document.createElement('div');
    hidden.style.cssText = 'position:fixed; left:-99999px; top:0; width:' + w + 'px; height:' + h + 'px;';
    document.body.appendChild(hidden);

    const colors = [];
    if (currentColorDataForPlot.length === xs.length) {
        for (let k = 0; k < currentColorDataForPlot.length; k++) {
            colors.push(isNum(currentColorDataForPlot[k]) ? currentColorDataForPlot[k] : null);
        }
    } else {
        for (let m = 0; m < xs.length; m++) colors.push(null);
    }

    const trace = {
        x: xs, y: ys, mode: 'markers+text', type: 'scatter',
        text: textNames.slice(0, xs.length),
        textposition: 'middle center',
        textfont: { size: 13, color: 'black' },
        marker: { size: 42, color: colors, colorscale: currentColorscale,
                  cmin: currentCmin, cmax: currentCmax, showscale: false,
                  line: { color: 'black', width: 1 } },
        hoverinfo: 'skip'
    };
    const layout = {
        width: w, height: h,
        xaxis: { range: xr, visible: false, fixedrange: true, showgrid: false, zeroline: false },
        yaxis: { range: yr, visible: false, fixedrange: true, showgrid: false, zeroline: false },
        plot_bgcolor: 'white', paper_bgcolor: 'white',
        showlegend: false, margin: { t: 0, l: 0, r: 0, b: 0 }
    };

    Plotly.newPlot(hidden, [trace], layout, { staticPlot: false })
        .then(() => Plotly.toImage(hidden, { format: 'svg', width: w, height: h, scale: 1 }))
        .then(svgUrl => {
            const svgText = decodeSvg(svgUrl);
            if (!svgText) throw new Error('svg decode failed');
            return svgToPng(injectOutlines(svgText), w, h);
        })
        .then(dataUrl => {
            const link = document.createElement('a');
            link.download = 'map-' + mapMode + '-' + Date.now() + '.png';
            link.href = dataUrl;
            link.click();
        })
        .catch(() => {
            return Plotly.toImage(hidden, { format: 'png', width: w, height: h, scale: 1 })
                .then(u => {
                    const link = document.createElement('a');
                    link.download = 'map-' + mapMode + '-' + Date.now() + '.png';
                    link.href = u;
                    link.click();
                });
        })
        .finally(() => { document.body.removeChild(hidden); });
};

const injectClusterStatsStyles = () => {
    if (document.getElementById('clusterStatsStyles')) return;
    const style = document.createElement('style');
    style.id = 'clusterStatsStyles';
    style.textContent = [
        '#functionWordStats { font-size:10px; line-height:1.4; }',
        '#correlationStats, #deltaStats { font-size:10px; line-height:1.4; }',
        '#functionWordStats strong { display:block; margin-bottom:8px; padding-bottom:4px; font-size:11px; }',
        '#correlationStats strong { display:block; margin-bottom:8px; padding-bottom:4px; font-size:11px; }',
        '#functionWordStatsToolbar { display:flex; justify-content:flex-end; margin-bottom:4px; }',
        '#toggleClusterStatsBtn { font-size:10px; padding:1px 6px; font-weight:bold; cursor:pointer; }',
        '.clusterWordRow { display:flex; align-items:center; gap:6px; padding:1px 0; font-size:10px; cursor:pointer; border-bottom:1px dotted #eee; }',
        '.clusterWordRow:hover { background:#f5f5f5; }',
        '.clusterWordRow .cword { flex:0 0 64px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }',
        '.clusterWordRow .cbar { flex:1 1 auto; height:8px; background:' + CLUSTER_BAR_TRACK_COLOR + '; border-radius:2px; overflow:hidden; position:relative; }',
        '.clusterWordRow .cfill { display:block; height:100%; }',
        '.clusterWordRow .cval { flex:0 0 48px; text-align:right; font-variant-numeric:tabular-nums; color:#333; }',
        '.clusterTableRow { display:flex; align-items:center; gap:6px; padding:1px 2px; font-size:10px; cursor:pointer; border-bottom:1px dotted #eee; }',
        '.clusterTableRow:hover { background:#f5f5f5; }',
        '.clusterTableRow .cti { flex:0 0 24px; color:#888; }',
        '.clusterTableRow .ctw { flex:1 1 auto; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }',
        '.clusterTableRow .ctv { flex:0 0 52px; text-align:right; font-variant-numeric:tabular-nums; color:#333; }',
        '#deltaStats .deltaWordRow { display:flex; justify-content:space-between; gap:6px; padding:1px 0; border-bottom:1px dotted #eee; }',
        '#deltaStats .deltaWordRow span:first-child { font-weight:600; white-space:nowrap; }',
        '#deltaStats .deltaWordRow .val { text-align:right; font-variant-numeric:tabular-nums; color:#333; min-width:0; word-break:break-word; }',
        '#deltaStats .deltaSectionHead, #functionWordStatsContent .deltaSectionHead { margin-top:10px; margin-bottom:4px; padding-bottom:2px; border-bottom:1px solid #ccc; font-weight:700; }',
        '#deltaStats details > summary::-webkit-details-marker { color:#888; }'
    ].join('\n');
    document.head.appendChild(style);
};

const createMainPanels = () => {
    const togglePanel = makePanel('toggleViewPanel',
        { left: '10px', top: '5px' }, '0.5px solid darkblue',
        '<button id="toggleViewBtn" style="font-weight: bold; min-width: 260px; text-align: left;">Switch to Correlation View</button>',
        { resizable: false, zIndex: 10001, dragLabel: 'view', minWidth: '260px' });

    makePanel('metricPanel', { left: '10px', top: '45px' }, '0.5px solid black',
        '<b>Color Metric</b><br/><input id="metricSearchInput" class="searchableMetricInput" type="text" placeholder="Search metrics..."><br/>' +
        '<select id="metricSelect" class="searchableMetricSelect"></select><br/>' +
        '<button id="toggleLabelsBtn" style="margin-top: 5px; font-size: 10px; padding: 2px 6px;">Hide Labels</button><br/>' +
        '<button id="toggleGrayPointsBtn" style="margin-top: 5px; font-size: 10px; padding: 2px 6px;">Hide Gray Points</button>',
        { minWidth: '220px', dragLabel: 'color metric' });

    const focusBtnPanel = makePanel('focusButtonPanel', { top: '5px' }, '0.5px solid black',
        '<button id="openFocusPanelBtn">Focus Text</button>' +
        '<span id="focusMetricPercentile" style="margin-left: 8px; font-size: 10px; font-weight: 500;"></span>',
        { resizable: false, zIndex: 10050, dragLabel: 'focus' });

    const focusPanel = makePanel('focusPanel', { display: 'none', width: '240px' }, '0.5px solid black',
        '<b>Focus Text</b><br/>' +
        '<input id="focusTextSearchInput" class="searchableMetricInput" type="text" placeholder="Search texts...">' +
        '<select id="focusTextSelect" class="searchableMetricSelect"></select>' +
        '<input id="focusTextInput" type="text" size="20" placeholder="Enter text name"><br/>' +
        '<label style="font-size: 10px;">Similarity: <select id="focusSimilarityModeSelect" style="font-size: 10px;">' +
        '<option value="cosine">Cosine</option><option value="burrows">Burrows</option></select></label><br/>' +
        '<label style="font-size: 10px;"><input id="focusClusterOnlyCheckbox" type="checkbox"> Hide texts outside focused text cluster</label><br/>' +
        '<label style="font-size: 10px;">Clustering: <select id="focusClusterMetricSelect" style="font-size: 10px;"></select></label><br/>' +
        '<label style="font-size: 10px;"><input id="semanticLinesCheckbox" type="checkbox"> Show semantic connections</label><br/>' +
        '<label style="font-size: 10px;">Connections: <input id="semanticLinesSlider" type="range" min="1" max="50" value="10"> <span id="semanticLinesValue">10</span></label><br/>' +
        '<button id="applyFocusBtn" style="margin-top: 5px;">Apply</button> ' +
        '<button id="clearFocusBtn" style="margin-top: 5px;">Clear</button>',
        { zIndex: 10050, minWidth: '180px', dragLabel: 'focus text' });

    const tr = togglePanel.getBoundingClientRect();
    const fl = tr.left + tr.width + 8;
    focusBtnPanel.style.left = fl + 'px';
    focusBtnPanel.style.top = tr.top + 'px';
    focusPanel.style.left = fl + 'px';
    focusPanel.style.top = (tr.top + tr.height + 6) + 'px';

    makePanel('colorPanel', { right: '10px', top: '5px' }, '0.5px solid black',
        '<b>Color Scale</b><br/>Min: <input id="colorMin" type="number" step="any" size="8"><br/>' +
        'Max: <input id="colorMax" type="number" step="any" size="8"><br/>' +
        '<button id="applyColorBtn">Apply</button> <button id="resetColorBtn">Reset</button><br/>' +
        '<button id="zoomInColorBtn">Zoom In</button> <button id="zoomOutColorBtn">Zoom Out</button><br/>' +
        '<button id="exportMapBtn" title="Download a large PNG of the current map">export png :)</button>',
        { minWidth: '160px', dragLabel: 'color scale' });

    makePanel('textPanel', { right: '10px', top: '135px', bottom: '10px', width: '180px' }, '0.5px solid black',
        '<div style="flex: 0 0 auto;"><b id="textListHeading">Texts</b><br/>' +
        '<select id="textSortOrder"><option value="alpha-asc">A-Z</option><option value="alpha-desc">Z-A</option>' +
        '<option value="metric-asc">Low to High</option><option value="metric-desc" selected>High to Low</option></select></div>' +
        '<div id="textList"></div>',
        { minWidth: '180px', dragLabel: 'texts' });

    if (hasBurrowsData()) {
        makePanel('experimentalPanel', { left: '10px', bottom: '10px' }, '0.5px solid #7a0403',
            '<button id="experimentalBtn" title="Toggle Burrows mirror map">Experimental</button>' +
            '<div id="burrowsIndicator" style="display:none; margin-top:4px;">Burrows map active</div>' +
            '<button id="deltaExplorerBtn" title="Explore which function words drive text similarity" style="display:none; margin-top:4px;">\u0394 Explore Deltas</button>',
            { resizable: false, zIndex: 10020, dragLabel: 'experimental' });
    }
};
const createSecondaryViews = () => {
    makeFullView('correlationContainer', 'correlationPlot');
    makeFullView('functionWordContainer', 'functionWordPlot');
    makeFullView('deltaContainer', 'deltaPlot');

    makePanel('correlationControls', { left: '10px', top: '45px', display: 'none' }, '0.5px solid black',
        '<b>Correlation Analysis</b><br/><b>X-Axis</b><br/>' +
        '<input id="xMetricSearchInput" class="searchableMetricInput" type="text" placeholder="Search X metric..."><br/>' +
        '<select id="xMetricSelect" class="searchableMetricSelect"></select><br/><b>Y-Axis</b><br/>' +
        '<input id="yMetricSearchInput" class="searchableMetricInput" type="text" placeholder="Search Y metric..."><br/>' +
        '<select id="yMetricSelect" class="searchableMetricSelect"></select><br/><b>Regression</b><br/>' +
        '<select id="regressionType"><option value="linear">Linear</option><option value="polynomial">Polynomial</option><option value="logistic">Logistic</option></select><br/>' +
        '<button id="updateCorrelationBtn" style="margin-top: 5px;">Update Plot</button><br/>' +
        '<button id="toggleCorrelationLabelsBtn" style="margin-top: 5px;">Toggle Labels</button>',
        { minWidth: '220px', dragLabel: 'correlation analysis' });

    makePanel('correlationStats', { right: '10px', top: '45px', display: 'none' }, '0.5px solid black', '',
        { minWidth: '220px', dragLabel: 'correlation stats' });

    makePanel('functionWordControls', { left: '10px', top: '45px', display: 'none', width: '300px' }, '0.5px solid #7a0403',
        '<b>Function Word Frequency</b><br/>' +
        '<input id="functionWordSearchInput" class="searchableMetricInput" type="text" placeholder="Search function words..."><br/>' +
        '<select id="functionWordSelect" class="searchableMetricSelect" multiple size="8"></select><br/>' +
        '<div style="font-size: 9px; color: #555;">Ctrl/Cmd-click or Shift-click to pick several. Rates of all selected words are summed.</div>' +
        '<button id="functionWordSelectShownBtn" style="margin-top: 4px; font-size: 10px;">Select all shown</button> ' +
        '<button id="functionWordClearBtn" style="margin-top: 4px; font-size: 10px;">Clear</button>' +
        '<div id="functionWordSelectionSummary"></div>' +
        '<label style="font-size: 10px;">View: <select id="functionWordPlotModeSelect" style="font-size: 10px;">' +
        '<option value="zscore">Z-score bell curve (percentile)</option><option value="density">Density</option>' +
        '<option value="cdf">Percentile (CDF)</option></select></label><br/>' +
        '<div style="font-size: 9px; color: #7a0403; margin-top: 4px; line-height:1.35;">The focused text\u2019s cluster is highlighted in yellow automatically.</div>' +
        '<label style="font-size: 10px; margin-top:4px; display:block;"><input id="functionWordShowHistogram" type="checkbox" checked> Show histogram (density &amp; z-score views)</label><br/>' +
        '<label style="font-size: 10px;"><input id="functionWordShowMarkers" type="checkbox" checked> Show individual texts</label><br/>' +
        '<button id="updateFunctionWordBtn" style="margin-top: 5px;">Update Curve</button>',
        { minWidth: '300px', dragLabel: 'function word' });

    makePanel('functionWordStats', { right: '10px', top: '45px', display: 'none', width: '320px' }, '0.5px solid #7a0403',
        '<div id="functionWordStatsToolbar">' +
        '<button id="toggleClusterStatsBtn" title="Switch the panel between single word statistics and cluster contribution statistics">Show Cluster Stats</button>' +
        '</div>' +
        '<div id="functionWordStatsContent"></div>',
        { minWidth: '180px', dragLabel: 'word stats' });

    makePanel('deltaControls', { left: '10px', top: '45px', width: '240px', display: 'none' }, '0.5px solid #7a0403',
        '<b>\u0394 Delta Explorer</b>' +
        '<div style="font-size:9px; color:#555; margin:2px 0 6px 0; line-height:1.3;">Which function words make two texts similar, and which split them apart?</div>' +
        '<label style="font-size:10px; font-weight:600;">Text A</label><br/>' +
        '<input id="deltaTextASearch" class="searchableMetricInput" type="text" placeholder="Search texts...">' +
        '<select id="deltaTextASelect" class="searchableMetricSelect"></select>' +
        '<label style="font-size:10px; font-weight:600;">Text B</label><br/>' +
        '<input id="deltaTextBSearch" class="searchableMetricInput" type="text" placeholder="Search texts...">' +
        '<select id="deltaTextBSelect" class="searchableMetricSelect"></select>' +
        '<button id="deltaSwapBtn" style="font-size:10px; margin-right:3px;">Swap A and B</button>' +
        '<button id="deltaUseFocusedBtn" style="font-size:10px;">Use focused as A</button><br/>' +
        '<button id="deltaUseNearestBtn" style="font-size:10px; margin-top:3px;">Pick nearest to A as B</button><br/>' +
        '<hr style="border:none; border-top:1px solid #ddd; margin:5px 0;">' +
        '<label style="font-size:10px; display:block;">Sort by:</label>' +
        '<select id="deltaSortSelect" style="font-size:10px; width:100%;">' +
        '<option value="separators">Biggest separators (|\u0394z| high)</option>' +
        '<option value="drivers">Similarity drivers (both distinctive)</option>' +
        '<option value="shared">Shared tendency (|avg z| high)</option></select><br/>' +
        '<label style="font-size:10px; display:block; margin-top:3px;">Show top <input id="deltaTopNRange" type="range" min="5" max="80" value="25" style="width:80px;"> <span id="deltaTopNValue">25</span> words</label>' +
        '<button id="deltaShowAllBtn" style="font-size:10px; margin-top:3px;">Show all</button><br/>' +
        '<div id="deltaScrollHint">The plot grows taller as more words are shown. Scroll to see every word.</div>' +
        '<button id="deltaUpdateBtn" style="margin-top:5px; font-weight:bold;">Recompute</button>',
        { minWidth: '240px', dragLabel: 'delta explorer' });

    makePanel('deltaStats', { right: '10px', top: '45px', width: '260px', display: 'none' }, '0.5px solid #7a0403', '',
        { minWidth: '260px', dragLabel: 'delta stats' });
};
const wireMetric = () => {
    const sel = document.getElementById('metricSelect');
    const search = document.getElementById('metricSearchInput');
    search.addEventListener('input', () => {
        const prev = sel.value;
        fillMetricSelect(sel, prev, search.value);
        if (sel.value && sel.value !== prev) changeMetric(sel.value);
    });
    sel.addEventListener('change', () => { changeMetric(sel.value); });
    listen('toggleLabelsBtn', 'click', e => {
        umapLabelsVisible = !umapLabelsVisible;
        Plotly.restyle(mainPlot, { mode: umapLabelsVisible ? 'markers+text' : 'markers' }, [0]);
        e.target.textContent = umapLabelsVisible ? 'Hide Labels' : 'Show Labels';
    });
    listen('toggleGrayPointsBtn', 'click', e => {
        hideGrayPoints = !hideGrayPoints;
        e.target.textContent = hideGrayPoints ? 'Show Gray Points' : 'Hide Gray Points';
        refreshFocusMarker();
        if (viewMode === 'functionWord') updateFwPlot();
    });
    listen('textSortOrder', 'change', updateTextList);
};
const wireColorScale = () => {
    listen('applyColorBtn', 'click', () => {
        const lo = parseFloat(document.getElementById('colorMin').value);
        const hi = parseFloat(document.getElementById('colorMax').value);
        if (isNaN(lo) || isNaN(hi) || lo >= hi) { alert('The color scale minimum must be smaller than the maximum.'); return; }
        applyColorScale(lo, hi);
    });
    listen('resetColorBtn', 'click', () => {
        const c = metricRange(currentColorDataForPlot);
        if (!c) return;
        document.getElementById('colorMin').value = c.min;
        document.getElementById('colorMax').value = c.max;
        applyColorScale(c.min, c.max);
    });
    listen('zoomInColorBtn', 'click', () => { zoomColor(CONFIG.colorScaleZoomRatio); });
    listen('zoomOutColorBtn', 'click', () => { zoomColor(1); });
    listen('exportMapBtn', 'click', exportMap);
};
const wireFocus = () => {
    const populateFocusSelect = filter => {
        const sel = document.getElementById('focusTextSelect');
        const f = (filter || '').trim().toLowerCase();
        const prev = focusedTextName ? textNames.indexOf(focusedTextName) : -1;
        sel.innerHTML = '';
        for (let i = 0; i < textNames.length; i++) {
            if (f && textNames[i].toLowerCase().indexOf(f) === -1) continue;
            const o = document.createElement('option');
            o.value = String(i);
            o.text = textNames[i];
            sel.appendChild(o);
        }
        if (prev >= 0 && sel.querySelector('option[value="' + prev + '"]')) {
            sel.value = String(prev);
        } else if (sel.options.length) {
            sel.value = sel.options[0].value;
        }
    };
    populateFocusSelect('');
    listen('focusTextSearchInput', 'input', e => {
        populateFocusSelect(e.target.value);
        const sel = document.getElementById('focusTextSelect');
        if (sel.value !== '') {
            const idx = Number(sel.value);
            if (!isNaN(idx) && textNames[idx]) {
                document.getElementById('focusTextInput').value = textNames[idx];
            }
        }
    });
    listen('focusTextSelect', 'change', e => {
        const idx = Number(e.target.value);
        if (!isNaN(idx) && textNames[idx]) {
            document.getElementById('focusTextInput').value = textNames[idx];
        }
    });
    listen('openFocusPanelBtn', 'click', () => {
        const p = document.getElementById('focusPanel');
        p.style.display = p.style.display === 'none' ? 'block' : 'none';
    });
    listen('applyFocusBtn', 'click', () => { focusText(document.getElementById('focusTextInput').value); });
    listen('clearFocusBtn', 'click', () => {
        document.getElementById('focusTextInput').value = '';
        document.getElementById('focusTextSearchInput').value = '';
        populateFocusSelect('');
        focusText('');
    });
    listen('focusTextInput', 'keydown', e => { if (e.key === 'Enter') focusText(e.target.value); });
    listen('focusSimilarityModeSelect', 'change', e => {
        focusSimilarityMode = e.target.value;
        refreshFocusMarker();
        refreshCorrFocus();
        updateTextList();
    });
    listen('focusClusterOnlyCheckbox', 'change', e => {
        focusClusterOnly = e.target.checked;
        refreshFocusMarker();
    });
    listen('focusClusterMetricSelect', 'change', e => {
        focusClusterMetricUserSelected = true;
        focusClusterMetric = e.target.value;
        refreshFocusMarker();
        if (viewMode === 'functionWord') {
            if (clusterStatsState.view === 'cluster') {
                clusterStatsState.source = focusClusterMetric;
                clusterStatsState.clusterId = null;
                clusterStatsState.clusterIdUserSelected = false;
                clusterStatsState.compareClusterId = null;
                evilFocusTextIndex = -1;
                evilFocusTextName = null;
                renderClusterStats();
                updateFwPlot();
            } else {
                updateFwPlot();
            }
        }
    });
    listen('semanticLinesCheckbox', 'change', e => {
        semanticLineModeEnabled = e.target.checked;
        refreshFocusMarker();
        refreshCorrFocus();
        updateTextList();
    });
    listen('semanticLinesSlider', 'input', e => {
        semanticLineCount = Math.max(1, Math.min(50, Number(e.target.value) || 1));
        document.getElementById('semanticLinesValue').textContent = semanticLineCount;
        if (semanticLineModeEnabled) {
            refreshConnections();
            refreshCorrConnections();
            updateTextList();
        }
    });
};


const wireViews = () => {
    listen('toggleViewBtn', 'click', () => {
        if (viewMode !== 'umap') setView('umap');
        else setView(mapMode === 'burrows' ? 'functionWord' : 'correlation');
    });
    listen('experimentalBtn', 'click', () => { setMapMode(mapMode === 'burrows' ? 'semantic' : 'burrows'); });
    listen('deltaExplorerBtn', 'click', () => { setView('delta'); });
};
const wireCorr = () => {
    populateCorrDropdowns();
    listen('updateCorrelationBtn', 'click', updateCorrPlot);
    listen('regressionType', 'change', updateCorrPlot);
    listen('xMetricSearchInput', 'input', populateCorrDropdowns);
    listen('yMetricSearchInput', 'input', populateCorrDropdowns);
    listen('toggleCorrelationLabelsBtn', 'click', e => {
        correlationLabelsVisible = !correlationLabelsVisible;
        Plotly.restyle(document.getElementById('correlationPlot'),
            { mode: correlationLabelsVisible ? 'markers+text' : 'markers' }, [0]).then(() => {
                if (correlationLabelsVisible) addOutlines('#correlationPlot .scatterlayer .textpoint text', 'correlationStrokeText');
            });
        e.target.textContent = correlationLabelsVisible ? 'Hide Labels' : 'Show Labels';
    });
};
const wireWords = () => {
    if (hasBurrowsData()) populateWordControls();
    const sel = document.getElementById('functionWordSelect');
    listen('functionWordSearchInput', 'input', populateWordControls);
    sel.addEventListener('change', () => {
        for (let i = 0; i < sel.options.length; i++) {
            const o = sel.options[i];
            if (!o.value) continue;
            if (o.selected) functionWordSelection.add(o.value);
            else functionWordSelection.delete(o.value);
        }
        updateWordSummary();
        updateFwPlot();
    });
    listen('functionWordSelectShownBtn', 'click', () => {
        for (let i = 0; i < sel.options.length; i++) {
            const o = sel.options[i];
            if (!o.value) continue;
            o.selected = true;
            functionWordSelection.add(o.value);
        }
        updateWordSummary();
        updateFwPlot();
    });
    listen('functionWordClearBtn', 'click', () => {
        functionWordSelection.clear();
        for (let i = 0; i < sel.options.length; i++) sel.options[i].selected = false;
        updateWordSummary();
        updateFwPlot();
    });
    listen('functionWordPlotModeSelect', 'change', e => {
        functionWordPlotMode = e.target.value;
        updateFwPlot();
    });
    listen('functionWordShowHistogram', 'change', e => {
        functionWordShowHistogram = e.target.checked;
        updateFwPlot();
    });
    listen('functionWordShowMarkers', 'change', e => {
        functionWordShowMarkers = e.target.checked;
        updateFwPlot();
    });
    listen('updateFunctionWordBtn', 'click', updateFwPlot);

    listen('toggleClusterStatsBtn', 'click', () => {
        const btn = document.getElementById('toggleClusterStatsBtn');
        if (!btn) return;
        if (clusterStatsState.view === 'cluster') {
            clusterStatsState.view = 'word';
            btn.textContent = 'Show Cluster Stats';
            updateFwPlot();
        } else {
            clusterStatsState.view = 'cluster';
            clusterStatsState.clusterId = null;
            clusterStatsState.clusterIdUserSelected = false;
            clusterStatsState.compareClusterId = null;
            evilFocusTextIndex = -1;
            evilFocusTextName = null;
            btn.textContent = 'Back to Word Stats';
            renderClusterStats();
        }
    });
};
const wireDeltaPicker = (searchId, selectId, stateKey) => {
    const si = document.getElementById(searchId);
    const sel = document.getElementById(selectId);

    si.addEventListener('input', () => {
        const n = fillDeltaSelect(sel, si.value, deltaState[stateKey]);
        if (n !== null && n !== deltaState[stateKey]) {
            deltaState[stateKey] = n;
            updateDeltaPlot();
        }
    });

    sel.addEventListener('change', () => {
        deltaState[stateKey] = Number(sel.value);
        updateDeltaPlot();
    });
};
const wireDelta = () => {
    wireDeltaPicker('deltaTextASearch', 'deltaTextASelect', 'textA');
    wireDeltaPicker('deltaTextBSearch', 'deltaTextBSelect', 'textB');
    listen('deltaSwapBtn', 'click', () => {
        const t = deltaState.textA;
        deltaState.textA = deltaState.textB;
        deltaState.textB = t;
        syncDeltaDropdowns();
        updateDeltaPlot();
    });
    listen('deltaUseFocusedBtn', 'click', () => {
        if (focusedTextIndex < 0) { alert('No text is focused.'); return; }
        deltaState.textA = focusedTextIndex;
        if (deltaState.textB === focusedTextIndex) deltaState.textB = defaultDeltaPartner(focusedTextIndex);
        syncDeltaDropdowns();
        updateDeltaPlot();
    });
    listen('deltaUseNearestBtn', 'click', () => {
        if (deltaState.textA === null) return;
        deltaState.textB = defaultDeltaPartner(deltaState.textA);
        syncDeltaDropdowns();
        updateDeltaPlot();
    });
    listen('deltaSortSelect', 'change', e => {
        deltaState.sortMode = e.target.value;
        updateDeltaPlot();
    });
    listen('deltaTopNRange', 'input', e => {
        const max = deltaState.totalWordCount || 80;
        setDeltaTopN(Math.max(5, Math.min(max, Number(e.target.value) || 25)));
        updateDeltaPlot();
    });
    listen('deltaShowAllBtn', 'click', () => {
        const total = deltaState.totalWordCount || 80;
        document.getElementById('deltaTopNRange').max = String(total);
        setDeltaTopN(total);
        updateDeltaPlot();
    });
    listen('deltaUpdateBtn', 'click', () => {
        recomputeDelta();
        updateDeltaPlot();
    });
};
const buildClusterMetricKeys = () => {
    for (let i = 0; i < metricKeys.length; i++) {
        if (isClusterKey(metricKeys[i])) clusterMetricKeys.push(metricKeys[i]);
    }
};
const buildTextLookup = () => {
    for (let ti = 0; ti < textNames.length; ti++) {
        textLookup[textNames[ti].toLowerCase()] = { name: textNames[ti], index: ti };
    }
};
const buildMetricCatalog = () => {
    for (let mi = 0; mi < metricKeys.length; mi++) {
        const mk = metricKeys[mi];
        if (mk in metricsData && metricsData[mk] !== null) {
            const lbl = isClusterKey(mk) ? clusterLabel(mk) : (metricDisplayNames[mk] || mk);
            metricCatalog.push({ key: mk, label: lbl, search: (mk + ' ' + lbl).toLowerCase() });
        }
    }
};
const initPage = () => {
    mainPlot = document.querySelector('.plotly-graph-div');
    if (!mainPlot) return;

    injectClusterStatsStyles();

    buildClusterMetricKeys();
    buildTextLookup();
    buildMetricCatalog();

    sanitizeMainHoverText();

    createMainPanels();
    createSecondaryViews();

    if (typeof mainPlot.on === 'function') mainPlot.on('plotly_afterplot', () => { addOutlines(); });
    setupHoverStyling();
    snapshotXY();
    snapshotAxes();
    fitPlot();

    setTimeout(() => { addOutlines(); }, CONFIG.textOutlineDelay);

    const metricSel = document.getElementById('metricSelect');
    fillMetricSelect(metricSel, currentMetric, '');
    if (metricSel.value) currentMetric = metricSel.value;

    focusClusterMetric = fillClusterSelect(document.getElementById('focusClusterMetricSelect'), preferredCluster('semantic'));
    document.getElementById('focusTextInput').value = CONFIG.defaultFocusText || '';

    syncColors();
    readColorState();

    if (isBurrows(currentMetric)) {
        const filled = colorData(currentMetric);
        if (filled) {
            currentColorDataForPlot = filled;
            Plotly.restyle(mainPlot, { 'marker.color': [filled] }, [0]);
        }
    }

    const isCluster = isClusterKey(currentMetric);
    const initCb = {
        'marker.colorbar.title': colorbarTitle(currentMetric),
        'marker.colorbar.tickvals': [null],
        'marker.colorbar.ticktext': [null]
    };
    if (isCluster) {
        initCb['marker.colorbar.showticklabels'] = [false];
    }
    Plotly.restyle(mainPlot, initCb, [0]);

    sanitizeMainHoverText();

    wireMetric();
    wireColorScale();
    wireFocus();
    wireViews();
    wireCorr();
    wireWords();
    wireDelta();

    updateColorInputs();
    if (CONFIG.defaultFocusText) {
        focusText(CONFIG.defaultFocusText);
    } else {
        refreshFocusMarker();
        updateTextList();
        updatePctLabel();
    }

    applyCapitals(currentMetric);
};

document.addEventListener('DOMContentLoaded', initPage);