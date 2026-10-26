var CONFIG = {
    highlightSize: 80, normalSize: 15, focusStarSize: 40,
    dimmedOpacity: 0.2, highlightDuration: 2000, textOutlineDelay: 100,
    pointFontSize: 7, colorScaleZoomRatio: 0.25, regressionLinePoints: 100,
    defaultColorScale: 'rdbu_r', defaultFocusText: __FOCUS_TEXT__
};

var metricsData = __METRICS_JSON__;
var chartTitles = __TITLES_JSON__;
var textNames = __TEXTS_JSON__;
var metricDisplayNames = __METRIC_NAMES_JSON__;
var metricColorScales = __METRIC_COLORScales_JSON__;
var metricKeys = __METRIC_KEYS_JSON__;
var hoverPanelPalettes = __HOVER_PANEL_PALETTES_JSON__;
var hoverTextPalettes = __HOVER_TEXT_PALETTES_JSON__;
var semanticEmbeddings = __SEMANTIC_EMBEDDINGS_JSON__;
var burrowsSimilarity = __BURROWS_SIMILARITY_JSON__;
var fightingWordsUnigrams = __FIGHTING_WORDS_UNIGRAMS_JSON__;
var burrowsReduced = __BURROWS_REDUCED_JSON__;
var burrowsFunctionWords = __BURROWS_FUNCTION_WORDS_JSON__;
var burrowsWordFrequencies = __BURROWS_WORD_FREQUENCIES_JSON__;

var CLUSTER_TURBO_COLORSCALE = [
    [0.0, '#30123b'], [0.08, '#4145ab'], [0.16, '#4675ed'], [0.24, '#39a2fc'],
    [0.32, '#1bcfd4'], [0.40, '#24eca6'], [0.48, '#61fc6c'], [0.56, '#a4fc3b'],
    [0.64, '#d1e834'], [0.72, '#f3c63a'], [0.80, '#fe9b2d'], [0.88, '#f36315'],
    [0.96, '#d93806'], [1.0, '#7a0403']
];

var FOCUS_OVERLAY_META = 'focusOverlay';
var CORR_FOCUS_OVERLAY_META = 'corrFocusOverlay';
var CONNECTION_META = 'semanticConnection';
var PLOT_CONFIG = { responsive: true };
var GRID_STYLE = { showgrid: true, gridwidth: 1, gridcolor: '#e0e0e0' };
var FOCUS_CLUSTER_COLOR = '#ffeb3b';
var FOCUS_CLUSTER_LINE_COLOR = '#b8860b';
var MAIN_PLOT_RIGHT_RESERVE_PX = 230;

var VIEW_ELEMENT_IDS = {
    correlation: ['correlationContainer', 'correlationControls', 'correlationStats'],
    functionWord: ['functionWordContainer', 'functionWordControls', 'functionWordStats'],
    delta: ['deltaContainer', 'deltaControls', 'deltaStats']
};

var DELTA_SORT_KEYS = {
    separators: 'deltaContribution', drivers: 'driverScore', shared: 'sharedScore'
};
var DELTA_SORT_LABELS = {
    separators: 'biggest separators', drivers: 'similarity drivers', shared: 'shared tendency'
};

var mainPlot = null;
var originalPlotX = null;
var originalPlotY = null;
var originalAxes = null;
var mainPlotResizeBound = false;
var mainPlotResizeTimer = null;

var currentMetric = __ACTIVE_METRIC_JSON__;
var currentColorDataForPlot = [];
var currentColorscale = CLUSTER_TURBO_COLORSCALE;
var currentCmin = 0;
var currentCmax = 1;

var viewMode = 'umap';
var mapMode = 'semantic';
var umapLabelsVisible = true;
var correlationLabelsVisible = true;
var hideGrayPoints = false;

var focusedTextName = null;
var focusedTextIndex = -1;
var focusClusterOnly = false;
var focusClusterMetric = null;
var focusClusterMetricUserSelected = false;
var focusSimilarityMode = 'cosine';

var semanticLineModeEnabled = false;
var semanticLineCount = 10;
var mainConnectionToken = 0;
var correlationConnectionToken = 0;

var correlationXMetric = metricKeys[0];
var correlationYMetric = metricKeys.length > 1 ? metricKeys[1] : metricKeys[0];

var functionWordSelection = new Set(
    burrowsFunctionWords && burrowsFunctionWords.length ? [burrowsFunctionWords[0]] : ['the']
);
var functionWordShowHistogram = true;
var functionWordShowMarkers = true;
var functionWordPlotMode = 'zscore';

var deltaState = {
    textA: null, textB: null, sortMode: 'separators', topN: 25,
    wordStats: null, deltaDistribution: null, totalWordCount: 0, initialized: false
};

var clusterMetricKeys = [];
var textLookup = {};
var metricCatalog = [];

function isNum(v) {
    return v !== null && v !== undefined && !isNaN(v);
}

function mean(arr) {
    if (!arr || arr.length === 0) return null;
    var total = 0;
    for (var i = 0; i < arr.length; i++) total += arr[i];
    return total / arr.length;
}

function variance(arr) {
    if (!arr || arr.length === 0) return null;
    var avg = mean(arr);
    var total = 0;
    for (var i = 0; i < arr.length; i++) total += (arr[i] - avg) * (arr[i] - avg);
    return total / arr.length;
}

function stddev(arr) {
    var v = variance(arr);
    return v === null ? null : Math.sqrt(v);
}

function median(sorted) {
    if (!sorted || sorted.length === 0) return null;
    var n = sorted.length;
    if (n % 2 === 1) return sorted[(n - 1) / 2];
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

function ordinal(n) {
    var v = Math.round(n);
    var last2 = v % 100;
    if (last2 >= 11 && last2 <= 13) return v + 'th';
    var last = v % 10;
    if (last === 1) return v + 'st';
    if (last === 2) return v + 'nd';
    if (last === 3) return v + 'rd';
    return v + 'th';
}

function fmt(v, digits) {
    if (digits === undefined) digits = 4;
    if (!isNum(v)) return 'N/A';
    var s = Number(v).toFixed(digits).replace(/\.?0+$/, '');
    return s === '-0' ? '0' : s;
}

function makeDiv(text, mb) {
    var d = document.createElement('div');
    d.textContent = text;
    if (mb) d.style.marginBottom = mb;
    return d;
}

function clearBody(panelId) {
    var body = document.querySelector('#' + panelId + ' .panelBody');
    if (!body) return null;
    body.innerHTML = '';
    return body;
}

function findTrace(plot, meta) {
    for (var i = 0; i < plot.data.length; i++) {
        if (plot.data[i].meta === meta) return i;
    }
    return -1;
}

function listen(id, evt, fn) {
    var el = document.getElementById(id);
    if (el) el.addEventListener(evt, fn);
}

function addOutlines(selector, cls) {
    if (!selector) selector = '.scatterlayer .textpoint text';
    if (!cls) cls = 'plotly-stroke-text';
    var els = document.querySelectorAll(selector);
    for (var i = 0; i < els.length; i++) els[i].classList.add(cls);
}

function countLE(sorted, v) {
    var lo = 0, hi = sorted.length;
    while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (sorted[mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

function empPct(sorted, v) {
    if (!sorted || !sorted.length) return 0;
    return (countLE(sorted, v) / sorted.length) * 100;
}

function normalPdf(x, m, s) {
    if (s <= 0) return 0;
    var z = (x - m) / s;
    return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
}

function normalCdf(z) {
    var sign = z < 0 ? -1 : 1;
    var x = Math.abs(z) / Math.SQRT2;
    var t = 1 / (1 + 0.3275911 * x);
    var poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592;
    var erf = 1 - poly * t * Math.exp(-x * x);
    return 0.5 * (1 + sign * erf);
}

function pearson(xs, ys) {
    var px = [], py = [];
    for (var i = 0; i < xs.length; i++) {
        if (isNum(xs[i]) && isNum(ys[i])) { px.push(xs[i]); py.push(ys[i]); }
    }
    if (px.length < 2) return null;
    var mx = mean(px), my = mean(py);
    var num = 0, sxx = 0, syy = 0;
    for (var j = 0; j < px.length; j++) {
        num += (px[j] - mx) * (py[j] - my);
        sxx += (px[j] - mx) * (px[j] - mx);
        syy += (py[j] - my) * (py[j] - my);
    }
    if (sxx === 0 || syy === 0) return null;
    return num / Math.sqrt(sxx * syy);
}

function linearReg(xs, ys) {
    var n = xs.length, sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (var i = 0; i < n; i++) {
        sx += xs[i]; sy += ys[i];
        sxy += xs[i] * ys[i]; sx2 += xs[i] * xs[i];
    }
    var denom = n * sx2 - sx * sx;
    var slope = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
    var intercept = n === 0 ? 0 : (sy - slope * sx) / n;
    return { slope: slope, intercept: intercept };
}

function polyReg(xs, ys) {
    var n = xs.length;
    var sx = 0, sy = 0, sx2 = 0, sx3 = 0, sx4 = 0, sxy = 0, sx2y = 0;
    for (var i = 0; i < n; i++) {
        var x = xs[i], y = ys[i];
        sx += x; sy += y;
        sx2 += x * x; sx3 += x * x * x; sx4 += x * x * x * x;
        sxy += x * y; sx2y += x * x * y;
    }
    var M = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]];
    var V = [sy, sxy, sx2y];
    for (var a = 0; a < 3; a++) {
        var p = a;
        for (var b = a + 1; b < 3; b++) if (Math.abs(M[b][a]) > Math.abs(M[p][a])) p = b;
        var tmp = M[a]; M[a] = M[p]; M[p] = tmp;
        var tv = V[a]; V[a] = V[p]; V[p] = tv;
        for (var c = a + 1; c < 3; c++) {
            var f = M[c][a] / M[a][a];
            for (var d = a; d < 3; d++) M[c][d] -= f * M[a][d];
            V[c] -= f * V[a];
        }
    }
    var co = [0, 0, 0];
    for (var e = 2; e >= 0; e--) {
        co[e] = V[e];
        for (var f2 = e + 1; f2 < 3; f2++) co[e] -= M[e][f2] * co[f2];
        co[e] /= M[e][e];
    }
    return { a: co[2], b: co[1], c: co[0] };
}

function logisticReg(xs, ys) {
    var lo = Math.min.apply(null, ys);
    var hi = Math.max.apply(null, ys);
    if (hi === lo) return null;
    var n = xs.length;
    var norm = [];
    for (var i = 0; i < n; i++) {
        var p = (ys[i] - lo) / (hi - lo);
        if (p <= 0) p = 1e-6;
        if (p >= 1) p = 1 - 1e-6;
        norm.push(p);
    }
    var alpha = 0, beta = 0;
    for (var iter = 0; iter < 50; iter++) {
        var g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
        for (var j = 0; j < n; j++) {
            var x = xs[j];
            var target = norm[j];
            var eta = alpha + beta * x;
            var mu = 1 / (1 + Math.exp(-eta));
            var w = Math.max(mu * (1 - mu), 1e-9);
            var r = target - mu;
            g0 += r;
            g1 += r * x;
            h00 += w;
            h01 += w * x;
            h11 += w * x * x;
        }
        var det = h00 * h11 - h01 * h01;
        if (Math.abs(det) < 1e-12) break;
        var dAlpha = (g0 * h11 - g1 * h01) / det;
        var dBeta = (g1 * h00 - g0 * h01) / det;
        alpha += dAlpha;
        beta += dBeta;
        if (Math.abs(dAlpha) < 1e-8 && Math.abs(dBeta) < 1e-8) break;
    }
    return { alpha: alpha, beta: beta, minY: lo, maxY: hi };
}

function regModel(xs, ys, type) {
    if (!xs || xs.length < 2) return null;
    if (type === 'linear') return linearReg(xs, ys);
    if (type === 'polynomial') return polyReg(xs, ys);
    if (type === 'logistic') return logisticReg(xs, ys);
    return null;
}

function predictY(x, type, model) {
    if (!model) return null;
    if (type === 'linear') return model.slope * x + model.intercept;
    if (type === 'polynomial') return model.a * x * x + model.b * x + model.c;
    if (type === 'logistic') {
        var n = 1 / (1 + Math.exp(-(model.alpha + model.beta * x)));
        return n * (model.maxY - model.minY) + model.minY;
    }
    return null;
}

function regLine(xs, ys, type, model) {
    if (!model || xs.length < 2) return null;
    var lo = Math.min.apply(null, xs), hi = Math.max.apply(null, xs);
    var steps = CONFIG.regressionLinePoints;
    var lx = [], ly = [];
    for (var i = 0; i <= steps; i++) {
        var x = lo + ((hi - lo) * i) / steps;
        lx.push(x);
        ly.push(predictY(x, type, model));
    }
    return { x: lx, y: ly };
}

function residualRank(texts, xs, ys, type, model) {
    if (!model) return [];
    var out = [];
    for (var i = 0; i < xs.length; i++) {
        var p = predictY(xs[i], type, model);
        if (p === null || isNaN(p)) continue;
        var r = ys[i] - p;
        out.push({ text: texts[i], residual: r, absResidual: Math.abs(r) });
    }
    out.sort(function(a, b) { return b.absResidual - a.absResidual; });
    return out;
}

function sampleCurve(minX, maxX, steps, fn) {
    var xs = [], ys = [];
    for (var i = 0; i <= steps; i++) {
        var x = minX + ((maxX - minX) * i) / steps;
        xs.push(x);
        ys.push(fn(x));
    }
    return { x: xs, y: ys };
}

function decodeSvg(url) {
    if (!url) return null;
    var comma = url.indexOf(',');
    if (comma < 0) return null;
    var header = url.slice(0, comma);
    var payload = url.slice(comma + 1);
    if (/;base64/i.test(header)) {
        try {
            var bin = atob(payload);
            var bytes = new Uint8Array(bin.length);
            for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            return new TextDecoder('utf-8').decode(bytes);
        } catch (e) { return null; }
    }
    try { return decodeURIComponent(payload); } catch (e) { return null; }
}

function injectOutlines(svg) {
    return svg.replace(/<text(\b[^>]*)>/g, function(full, attrs) {
        var selfClose = '';
        var a = attrs;
        if (/\s*\/$/.test(a)) { selfClose = '/'; a = a.replace(/\s*\/$/, ''); }
        var existingStyle = '';
        a = a.replace(/\s+style\s*=\s*"([^"]*)"/i, function(m, c) { existingStyle = c; return ''; });
        var strip = function(s) {
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
        var clean = strip(existingStyle);
        var fs = null;
        var sm = clean.match(/font-size\s*:\s*(\d+(?:\.\d+)?)/);
        if (sm) fs = parseFloat(sm[1]);
        if (fs === null) {
            var am = a.match(/font-size\s*=\s*"(\d+(?:\.\d+)?)"/);
            if (am) fs = parseFloat(am[1]);
        }
        var sw = fs !== null ? Math.max(2, fs / 4.5).toFixed(1) : '3';
        var st = 'stroke: white; stroke-width: ' + sw + '; paint-order: stroke; stroke-linejoin: round; stroke-linecap: round;';
        var finalStyle = clean ? clean + '; ' + st : st;
        return '<text' + a + ' style="' + finalStyle + '"' +
               ' stroke="white" stroke-width="' + sw + '" stroke-linejoin="round" stroke-linecap="round"' + selfClose + '>';
    });
}

function svgToPng(svg, width, height) {
    return new Promise(function(resolve, reject) {
        var blob;
        try { blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }); }
        catch (e) { reject(e); return; }
        var url = URL.createObjectURL(blob);
        var img = new Image();
        img.onload = function() {
            try {
                var c = document.createElement('canvas');
                c.width = width; c.height = height;
                var ctx = c.getContext('2d');
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, width, height);
                ctx.drawImage(img, 0, 0, width, height);
                URL.revokeObjectURL(url);
                resolve(c.toDataURL('image/png'));
            } catch (e) { URL.revokeObjectURL(url); reject(e); }
        };
        img.onerror = function(e) { URL.revokeObjectURL(url); reject(e); };
        img.src = url;
    });
}

function isBurrows(key) {
    if (!key) return false;
    var s = (key + ' ' + (metricDisplayNames[key] || '')).toLowerCase();
    return s.indexOf('burrows') !== -1;
}

function getAlgo(key) {
    if (!key) return '';
    var s = (key + ' ' + (metricDisplayNames[key] || '')).toLowerCase();
    if (s.indexOf('leiden') !== -1) return 'Leiden';
    if (s.indexOf('louvain') !== -1) return 'Louvain';
    if (s.indexOf('kmeans') !== -1 || s.indexOf('k-means') !== -1) return 'k-means';
    if (s.indexOf('hdbscan') !== -1) return 'HDBSCAN';
    if (s.indexOf('burrows') !== -1) return 'Leiden';
    return '';
}

function isClusterKey(key) {
    if (!key) return false;
    if (key === 'cluster' || key.indexOf('cluster_') === 0) return true;
    var s = (key + ' ' + (metricDisplayNames[key] || '')).toLowerCase();
    if (s.indexOf('burrows') === -1) return false;
    var kw = ['cluster', 'leiden', 'louvain', 'kmeans', 'k-means', 'hdbscan'];
    for (var i = 0; i < kw.length; i++) {
        if (s.indexOf(kw[i]) !== -1) return true;
    }
    return false;
}

function clusterLabel(key) {
    if (!key) return '';
    return metricDisplayNames[key] || key;
}

function clusterValueLabel(key, id) {
    var algo = getAlgo(key) || 'Leiden';
    if (id < 0) return 'unclustered';
    return isBurrows(key) ? 'Burrows ' + algo + ' cluster ' + id : algo + ' cluster ' + id;
}

function colorbarTitle(key) {
    if (isClusterKey(key)) return clusterLabel(key);
    return metricDisplayNames[key] || key;
}

function preferredCluster(mode) {
    var b = [], reg = [];
    for (var i = 0; i < clusterMetricKeys.length; i++) {
        if (isBurrows(clusterMetricKeys[i])) b.push(clusterMetricKeys[i]);
        else reg.push(clusterMetricKeys[i]);
    }
    if (mode === 'burrows' && b.length) {
        for (var j = 0; j < b.length; j++) if (b[j] === 'burrows_cluster') return 'burrows_cluster';
        return b[0];
    }
    if (reg.length) {
        for (var k = 0; k < reg.length; k++) if (reg[k] === 'cluster') return 'cluster';
        return reg[0];
    }
    return clusterMetricKeys[0] || null;
}

function metricRange(values) {
    var lo = Infinity, hi = -Infinity, ok = false;
    for (var i = 0; i < values.length; i++) {
        if (!isNum(values[i])) continue;
        if (values[i] < lo) lo = values[i];
        if (values[i] > hi) hi = values[i];
        ok = true;
    }
    return ok ? { min: lo, max: hi } : null;
}

function colorData(key) {
    var raw = metricsData[key];
    if (!raw) return null;
    var b = isBurrows(key), c = isClusterKey(key);
    var valid = [];
    for (var i = 0; i < raw.length; i++) if (isNum(raw[i])) valid.push(raw[i]);
    var out = [];
    if (b) {
        if (!valid.length) { for (var j = 0; j < raw.length; j++) out.push(NaN); return out; }
        if (c) {
            for (var k = 0; k < raw.length; k++) out.push(isNum(raw[k]) ? raw[k] : -1);
            return out;
        }
        var avg = 0;
        for (var m = 0; m < valid.length; m++) avg += valid[m];
        avg /= valid.length;
        for (var n = 0; n < raw.length; n++) out.push(isNum(raw[n]) ? raw[n] : avg);
        return out;
    }
    for (var p = 0; p < raw.length; p++) out.push(isNum(raw[p]) ? raw[p] : NaN);
    return out;
}

function stripBurrowsClusterLine(line) {
    return !/burrow[s]?_cluster/i.test(line);
}

function stripBurrowsClusterFromHoverString(s) {
    if (typeof s !== 'string') return s;
    if (!/burrow[s]?_cluster/i.test(s)) return s;
    var parts = s.split(/<br\s*\/?>|\n/i);
    var kept = [];
    for (var i = 0; i < parts.length; i++) {
        if (stripBurrowsClusterLine(parts[i])) kept.push(parts[i]);
    }
    return kept.join('<br>');
}

function sanitizeMainHoverText() {
    if (!mainPlot || !mainPlot.data) return;
    for (var i = 0; i < mainPlot.data.length; i++) {
        var tr = mainPlot.data[i];
        if (Array.isArray(tr.hovertext)) {
            var changed = false;
            var ht = [];
            for (var j = 0; j < tr.hovertext.length; j++) {
                var v = stripBurrowsClusterFromHoverString(tr.hovertext[j]);
                if (v !== tr.hovertext[j]) changed = true;
                ht.push(v);
            }
            if (changed) Plotly.restyle(mainPlot, { hovertext: [ht] }, [i]);
        }
        if (typeof tr.hovertemplate === 'string') {
            var tmpl = stripBurrowsClusterFromHoverString(tr.hovertemplate);
            if (tmpl !== tr.hovertemplate) Plotly.restyle(mainPlot, { hovertemplate: [tmpl] }, [i]);
        }
    }
}

function hoverPointColor(pointIndex) {
    var traces = mainPlot.querySelectorAll('.scatterlayer .trace');
    var points = traces.length ? traces[0].querySelectorAll('.point') : [];
    var point = points[pointIndex];
    if (point) {
        var computed = window.getComputedStyle(point);
        var fill = computed.fill || point.style.fill;
        if (fill && fill !== 'none') return fill;
    }
    return '#555555';
}

function addHoverGradient(colors, index) {
    var svg = mainPlot && mainPlot.querySelector('svg');
    if (!svg || !colors || colors.length < 2) return null;
    var defs = svg.querySelector('defs');
    if (!defs) {
        defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        svg.insertBefore(defs, svg.firstChild);
    }
    var id = 'atlas-hover-gradient-' + index;
    var old = document.getElementById(id);
    if (old) old.remove();
    var gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient.setAttribute('id', id);
    gradient.setAttribute('x1', '0');
    gradient.setAttribute('y1', '0');
    gradient.setAttribute('x2', '0');
    gradient.setAttribute('y2', '1');
    for (var i = 0; i < colors.length; i++) {
        var stop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop.setAttribute('offset', ((i / (colors.length - 1)) * 100) + '%');
        stop.setAttribute('stop-color', colors[i]);
        gradient.appendChild(stop);
    }
    defs.appendChild(gradient);
    return 'url(#' + id + ')';
}

function styleMainHover(pointIndex) {
    var panelColors = hoverPanelPalettes[pointIndex];
    var textColors = hoverTextPalettes[pointIndex];
    if ((!panelColors || panelColors.length < 2) &&
        (!textColors || textColors.length < 2)) return;
    var hoverGroups = mainPlot.querySelectorAll('.hoverlayer .hovertext');
    if (!hoverGroups.length) return;

    var gradient = addHoverGradient(panelColors, pointIndex);
    var panelOutline = hoverPointColor(pointIndex);
    var textOutline = textColors && textColors.length ? textColors[0] : null;
    var textColor = textColors && textColors.length > 1 ? textColors[1] : null;

    for (var i = 0; i < hoverGroups.length; i++) {
        var group = hoverGroups[i];
        var background = group.querySelector('path, rect');
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
        var labels = group.querySelectorAll('text, tspan');
        for (var j = 0; j < labels.length; j++) {
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
}

function setupHoverStyling() {
    if (!mainPlot || typeof mainPlot.on !== 'function') return;
    if (typeof MutationObserver === 'undefined') return;

    var lastHoverPoint = -1;
    var hoverlayer = null;
    var observer = null;
    var OBS_CONFIG = {
        childList: true, attributes: true, characterData: true, subtree: true
    };

    function applyStyles() {
        if (lastHoverPoint < 0 || !hoverlayer) return;
        if (observer) observer.disconnect();
        try {
            styleMainHover(lastHoverPoint);
        } catch (e) {
            console.error('[hover style]', e);
        } finally {
            if (observer && hoverlayer && hoverlayer.isConnected) {
                try { observer.observe(hoverlayer, OBS_CONFIG); }
                catch (e) {}
            }
        }
    }

    function bind() {
        var hl = mainPlot.querySelector('.hoverlayer');
        if (!hl) {
            setTimeout(bind, 200);
            return;
        }
        hoverlayer = hl;
        if (!observer) {
            observer = new MutationObserver(function() {
                if (lastHoverPoint < 0) return;
                applyStyles();
            });
        }
        observer.disconnect();
        try { observer.observe(hoverlayer, OBS_CONFIG); }
        catch (e) {}
    }

    bind();

    mainPlot.on('plotly_hover', function(event) {
        if (!event || !event.points || !event.points.length) return;
        lastHoverPoint = event.points[0].pointNumber;
        var current = mainPlot.querySelector('.hoverlayer');
        if (current && current !== hoverlayer) {
            hoverlayer = current;
            bind();
        }
        requestAnimationFrame(function() {
            requestAnimationFrame(applyStyles);
        });
    });

    mainPlot.on('plotly_unhover', function() {
        lastHoverPoint = -1;
    });
}

function makeDraggable(panel, handle) {
    var dragging = false, sx = 0, sy = 0, sl = 0, st = 0;
    function move(e) {
        if (!dragging) return;
        panel.style.left = (sl + e.clientX - sx) + 'px';
        panel.style.top = (st + e.clientY - sy) + 'px';
    }
    function up() {
        dragging = false;
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
    }
    handle.addEventListener('mousedown', function(e) {
        var tag = (e.target.tagName || '').toUpperCase();
        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'BUTTON' || tag === 'TEXTAREA' || tag === 'A') return;
        e.preventDefault();
        e.stopPropagation();
        var rect = panel.getBoundingClientRect();
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

function addCollapse(panel, handle) {
    var btn = document.createElement('button');
    btn.className = 'panelCollapseBtn';
    btn.textContent = '-';
    btn.title = 'Collapse or expand this panel';
    var savedHeight = '';
    var savedBottom = '';
    btn.addEventListener('click', function() {
        var body = panel.querySelector('.panelBody');
        var collapsing = body.style.display !== 'none';
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
}

function makePanel(id, pos, border, content, opts) {
    if (!opts) opts = {};
    var resizable = opts.resizable !== false;
    var panel = document.createElement('div');
    panel.id = id;
    var display = pos.display;
    var style = {
        position: 'fixed', background: 'white', padding: '5px',
        border: border || '0.5px solid black',
        zIndex: String(opts.zIndex || 9999),
        resize: resizable ? 'both' : 'none',
        overflow: resizable ? 'auto' : 'visible',
        minWidth: opts.minWidth || '120px', minHeight: '30px',
        maxHeight: '90vh', boxSizing: 'border-box'
    };
    for (var k in pos) {
        if (k !== 'display') style[k] = pos[k];
    }
    for (var s in style) panel.style[s] = style[s];
    var handle = document.createElement('div');
    handle.className = 'panelDragHandle';
    handle.textContent = opts.dragLabel || '...';
    panel.appendChild(handle);
    var body = document.createElement('div');
    body.className = 'panelBody';
    body.innerHTML = content || '';
    panel.appendChild(body);
    if (display) panel.style.display = display;
    document.body.appendChild(panel);
    makeDraggable(panel, handle);
    if (resizable) addCollapse(panel, handle);
    return panel;
};

function makeFullView(containerId, plotId) {
    var c = document.createElement('div');
    c.id = containerId;
    c.className = 'fullView';
    c.innerHTML = '<div id="' + plotId + '" class="fullViewPlot"></div>';
    document.body.appendChild(c);
    return c;
}

function drawPlot(plotId, traces, layout, fixedHeight) {
    var el = document.getElementById(plotId);
    el.style.height = fixedHeight ? (fixedHeight + 'px') : '';
    layout.autosize = true;
    return Plotly.newPlot(plotId, traces, layout, PLOT_CONFIG);
}

function emptyPlot(plotId, title) {
    return drawPlot(plotId, [], { title: title }, null);
}

function snapshotXY() {
    if (originalPlotX && originalPlotY && originalPlotX.length > 0) return true;
    var traces = [];
    if (mainPlot._fullData && mainPlot._fullData[0]) traces.push(mainPlot._fullData[0]);
    if (mainPlot.data && mainPlot.data[0]) traces.push(mainPlot.data[0]);
    for (var i = 0; i < traces.length; i++) {
        var t = traces[i];
        if (!t || !t.x || !t.y) continue;
        if (t.x.length > 0 && t.x.length === t.y.length) {
            originalPlotX = Array.prototype.slice.call(t.x);
            originalPlotY = Array.prototype.slice.call(t.y);
            return true;
        }
    }
    return false;
}

function snapshotAxes() {
    var layout = mainPlot.layout || {};
    function read(a) {
        return { autorange: !a || a.autorange !== false, range: a && a.range ? a.range.slice() : null };
    }
    originalAxes = { x: read(layout.xaxis), y: read(layout.yaxis) };
}

function resetAxes(mode) {
    var keep = mode === 'semantic' && originalAxes;
    var update = {};
    var names = ['x', 'y'];
    for (var i = 0; i < names.length; i++) {
        var n = names[i];
        var saved = keep ? originalAxes[n] : null;
        var key = n + 'axis';
        if (saved && !saved.autorange && saved.range) {
            update[key + '.autorange'] = false;
            update[key + '.range'] = saved.range;
        } else {
            update[key + '.autorange'] = true;
        }
    }
    return Plotly.relayout(mainPlot, update);
}

function applyPlotViewportSize() {
    if (!mainPlot) return;
    mainPlot.style.width = 'calc(100% - ' + MAIN_PLOT_RIGHT_RESERVE_PX + 'px)';
    mainPlot.style.margin = '0';
    mainPlot.style.boxSizing = 'border-box';
    mainPlot.style.height = '100vh';
    try {
        if (typeof CSS !== 'undefined' && CSS.supports && CSS.supports('height', '100dvh')) {
            mainPlot.style.height = '100dvh';
        }
    } catch (e) {}
}

function mainPlotDoResize() {
    if (!mainPlot || !mainPlot.data) return;
    applyPlotViewportSize();
    var w = mainPlot.clientWidth;
    var h = mainPlot.clientHeight;
    if (!w || !h) {
        w = window.innerWidth - MAIN_PLOT_RIGHT_RESERVE_PX;
        h = window.innerHeight;
    }
    if (!w || !h) return;
    try {
        Plotly.relayout(mainPlot, {
            autosize: false,
            width: w,
            height: h
        });
    } catch (e) {
        console.error('[plot resize]', e);
    }
}

function mainPlotScheduleResize() {
    if (mainPlotResizeTimer) clearTimeout(mainPlotResizeTimer);
    mainPlotResizeTimer = setTimeout(function() {
        mainPlotResizeTimer = null;
        mainPlotDoResize();
    }, 100);
}

function fitPlot() {
    if (!mainPlot) return;
    try { mainPlotDoResize(); } catch (e) { console.error(e); }

    if (mainPlotResizeBound) return;
    mainPlotResizeBound = true;

    window.addEventListener('resize', mainPlotScheduleResize);
    window.addEventListener('orientationchange', mainPlotScheduleResize);
    window.addEventListener('load', mainPlotScheduleResize);
    if (window.visualViewport) {
        try { window.visualViewport.addEventListener('resize', mainPlotScheduleResize); } catch (e) {}
    }
    if (typeof ResizeObserver !== 'undefined') {
        try {
            var ro = new ResizeObserver(mainPlotScheduleResize);
            ro.observe(document.body);
            if (mainPlot.parentElement && mainPlot.parentElement !== document.body) {
                ro.observe(mainPlot.parentElement);
            }
        } catch (e) {
            console.error('[ResizeObserver]', e);
        }
    }
}

function fillMetricSelect(sel, value, filter) {
    if (!sel) return;
    var f = (filter || '').trim().toLowerCase();
    var prev = value || sel.value;
    var matches = [];
    for (var i = 0; i < metricCatalog.length; i++) {
        if (!f || metricCatalog[i].search.indexOf(f) !== -1) matches.push(metricCatalog[i]);
    }
    sel.innerHTML = '';
    if (matches.length === 0) {
        var o = document.createElement('option');
        o.value = ''; o.text = 'No matching metrics'; o.disabled = true; o.selected = true;
        sel.appendChild(o);
        return;
    }
    for (var j = 0; j < matches.length; j++) {
        var opt = document.createElement('option');
        opt.value = matches[j].key;
        opt.text = matches[j].label;
        sel.appendChild(opt);
    }
    var found = false;
    for (var k = 0; k < matches.length; k++) if (matches[k].key === prev) found = true;
    sel.value = found ? prev : matches[0].key;
}

function fillClusterSelect(sel, value) {
    if (!sel) return value;
    sel.innerHTML = '';
    for (var i = 0; i < clusterMetricKeys.length; i++) {
        var o = document.createElement('option');
        o.value = clusterMetricKeys[i];
        o.text = clusterLabel(clusterMetricKeys[i]);
        sel.appendChild(o);
    }
    for (var j = 0; j < clusterMetricKeys.length; j++) {
        if (clusterMetricKeys[j] === value) { sel.value = value; return value; }
    }
    if (clusterMetricKeys.length > 0) { sel.value = clusterMetricKeys[0]; return clusterMetricKeys[0]; }
    return value;
}

function applyClusterForMode(mode) {
    if (focusClusterMetricUserSelected) return;
    var pref = preferredCluster(mode);
    if (!pref) return;
    focusClusterMetric = fillClusterSelect(document.getElementById('focusClusterMetricSelect'), pref);
}

function buildSizes() {
    var a = [];
    for (var i = 0; i < textNames.length; i++) a.push(CONFIG.normalSize);
    return a;
}

function buildOpacity() {
    var cluster = focusClusterOnly && focusClusterMetric ? metricsData[focusClusterMetric] : null;
    var focusVal = null;
    if (cluster && focusedTextIndex >= 0 && isNum(cluster[focusedTextIndex])) {
        focusVal = cluster[focusedTextIndex];
    }
    var a = [];
    for (var i = 0; i < textNames.length; i++) {
        if (i === focusedTextIndex) { a.push(1); continue; }
        if (hideGrayPoints && !isNum(currentColorDataForPlot[i])) { a.push(0); continue; }
        if (focusVal !== null && cluster[i] !== focusVal) { a.push(0); continue; }
        a.push(1);
    }
    return a;
}

function makeOverlay(meta) {
    return {
        x: [], y: [], mode: 'markers+text', type: 'scatter', meta: meta,
        hoverinfo: 'skip', showlegend: false, text: [],
        textposition: 'middle center',
        textfont: { size: 9, color: '#000' },
        marker: { symbol: 'star', size: CONFIG.focusStarSize, color: '#ffd400',
                  line: { color: '#111', width: 2 } }
    };
}

function ensureOverlay(plot, meta) {
    if (!plot || !plot.data) return Promise.resolve();
    if (findTrace(plot, meta) >= 0) return Promise.resolve();
    return Plotly.addTraces(plot, makeOverlay(meta));
}

function overlayFront(plot, meta) {
    if (!plot || !plot.data) return Promise.resolve();
    var idx = findTrace(plot, meta);
    var last = plot.data.length - 1;
    if (idx >= 0 && idx !== last) return Plotly.moveTraces(plot, [idx], [last]);
    return Promise.resolve();
}

function plotCoord(plot, ti, axis, pi) {
    var full = plot._fullData && plot._fullData[ti];
    var data = plot.data && plot.data[ti];
    var vals = (full && full[axis]) || (data && data[axis]);
    return vals ? vals[pi] : undefined;
}

function baseColor(plot, pi) {
    var t = plot._fullData && plot._fullData[0];
    if (!t) return null;
    var c = t.marker && t.marker.color;
    if (Array.isArray(c) || (c && ArrayBuffer.isView(c))) return c[pi];
    return typeof c === 'string' ? c : null;
}

function updateOverlay() {
    var idx = findTrace(mainPlot, FOCUS_OVERLAY_META);
    if (idx < 0) return Promise.resolve();
    if (focusedTextIndex < 0 || !focusedTextName) {
        return Plotly.restyle(mainPlot, { x: [[]], y: [[]], text: [[]] }, [idx]);
    }
    var update = {
        x: [[plotCoord(mainPlot, 0, 'x', focusedTextIndex)]],
        y: [[plotCoord(mainPlot, 0, 'y', focusedTextIndex)]],
        text: [[focusedTextName]]
    };
    var bc = baseColor(mainPlot, focusedTextIndex);
    var fv = currentColorDataForPlot[focusedTextIndex];
    if (typeof bc === 'string') update['marker.color'] = [[bc]];
    else if (!isNum(fv)) update['marker.color'] = [['#808080']];
    else {
        update['marker.color'] = [[fv]];
        update['marker.colorscale'] = [currentColorscale];
        update['marker.cmin'] = [currentCmin];
        update['marker.cmax'] = [currentCmax];
    }
    return Plotly.restyle(mainPlot, update, [idx]);
}

function refreshFocusMarker() {
    Plotly.restyle(mainPlot, { 'marker.opacity': [buildOpacity()] }, [0]);
    ensureOverlay(mainPlot, FOCUS_OVERLAY_META)
        .then(updateOverlay)
        .then(function() { return overlayFront(mainPlot, FOCUS_OVERLAY_META); })
        .then(refreshConnections);
}

function syncColors() {
    var t = mainPlot.data && mainPlot.data[0];
    var c = t && t.marker && t.marker.color;
    if (c === undefined || c === null) {
        currentColorDataForPlot = [];
        for (var i = 0; i < textNames.length; i++) currentColorDataForPlot.push(null);
    } else if (Array.isArray(c)) {
        currentColorDataForPlot = c.slice();
    } else if (ArrayBuffer.isView(c)) {
        currentColorDataForPlot = Array.from(c);
    } else {
        currentColorDataForPlot = [];
        for (var j = 0; j < textNames.length; j++) currentColorDataForPlot.push(c);
    }
}

function readColorState() {
    var m = mainPlot.data[0] && mainPlot.data[0].marker;
    if (!m) return;
    if (m.colorscale !== undefined) currentColorscale = m.colorscale;
    if (isNum(m.cmin)) currentCmin = m.cmin;
    if (isNum(m.cmax)) currentCmax = m.cmax;
}

function connColor(rank, total) {
    var ratio = total <= 1 ? 0 : rank / (total - 1);
    var r = Math.round(235 * (1 - ratio) + 70 * ratio);
    var g = Math.round(35 * (1 - ratio) + 55 * ratio);
    var b = Math.round(45 * (1 - ratio) + 150 * ratio);
    return 'rgb(' + r + ', ' + g + ', ' + b + ')';
}

function cosineNeighbors(idx) {
    var fe = semanticEmbeddings[idx];
    if (!fe) return [];
    var out = [];
    for (var i = 0; i < textNames.length; i++) {
        var e = semanticEmbeddings[i];
        if (i === idx || !e) continue;
        var len = Math.min(fe.length, e.length);
        var sim = 0;
        for (var d = 0; d < len; d++) sim += fe[d] * e[d];
        out.push({ text: textNames[i], index: i, similarity: sim });
    }
    out.sort(function(a, b) { return b.similarity - a.similarity; });
    return out;
}

function burrowsNeighbors(idx) {
    var row = burrowsSimilarity[idx];
    if (!row) return [];
    var out = [];
    for (var i = 0; i < textNames.length; i++) {
        if (i === idx || !isNum(row[i])) continue;
        out.push({ text: textNames[i], index: i, similarity: row[i] });
    }
    out.sort(function(a, b) { return b.similarity - a.similarity; });
    return out;
}

function focusNeighbors(idx) {
    if (idx < 0) return [];
    return focusSimilarityMode === 'burrows' ? burrowsNeighbors(idx) : cosineNeighbors(idx);
}

function connTrace(from, to, rank, total) {
    return {
        x: [from.x, to.x], y: [from.y, to.y], mode: 'lines', type: 'scatter',
        meta: CONNECTION_META, hoverinfo: 'skip', showlegend: false,
        line: { color: connColor(rank, total), width: 2 }
    };
}

function removeConns(plot) {
    var idxs = [];
    for (var i = 0; i < plot.data.length; i++) {
        if (plot.data[i].meta === CONNECTION_META) idxs.push(i);
    }
    if (!idxs.length) return Promise.resolve();
    return Plotly.deleteTraces(plot, idxs.reverse());
}

function refreshConnections() {
    var token = ++mainConnectionToken;
    return removeConns(mainPlot).then(function() {
        if (token !== mainConnectionToken) return;
        if (!semanticLineModeEnabled || focusedTextIndex < 0 || !focusedTextName) return;
        var fp = {
            x: plotCoord(mainPlot, 0, 'x', focusedTextIndex),
            y: plotCoord(mainPlot, 0, 'y', focusedTextIndex)
        };
        var nb = focusNeighbors(focusedTextIndex).slice(0, semanticLineCount);
        var traces = [];
        for (var i = 0; i < nb.length; i++) {
            var tp = { x: plotCoord(mainPlot, 0, 'x', nb[i].index),
                       y: plotCoord(mainPlot, 0, 'y', nb[i].index) };
            traces.push(connTrace(fp, tp, i, nb.length));
        }
        if (traces.length && token === mainConnectionToken) {
            return Plotly.addTraces(mainPlot, traces);
        }
    }).then(function() {
        return overlayFront(mainPlot, FOCUS_OVERLAY_META);
    });
}

function refreshCorrConnections() {
    var plot = document.getElementById('correlationPlot');
    if (!plot || !plot.data || !plot.data[0]) return Promise.resolve();
    var token = ++correlationConnectionToken;
    return removeConns(plot).then(function() {
        if (token !== correlationConnectionToken) return;
        if (!semanticLineModeEnabled || !focusedTextName) return;
        var texts = plot.data[0].text || [];
        var fp = -1;
        for (var i = 0; i < texts.length; i++) if (texts[i] === focusedTextName) { fp = i; break; }
        if (fp < 0) return;
        var entry = textLookup[focusedTextName.toLowerCase()];
        if (!entry) return;
        var from = { x: plot.data[0].x[fp], y: plot.data[0].y[fp] };
        var nb = focusNeighbors(entry.index).slice(0, semanticLineCount);
        var traces = [];
        for (var j = 0; j < nb.length; j++) {
            var target = -1;
            for (var t = 0; t < texts.length; t++) if (texts[t] === nb[j].text) { target = t; break; }
            if (target < 0) continue;
            traces.push(connTrace(from,
                { x: plot.data[0].x[target], y: plot.data[0].y[target] },
                j, nb.length));
        }
        if (traces.length && token === correlationConnectionToken) {
            return Plotly.addTraces(plot, traces);
        }
    }).then(function() {
        return overlayFront(plot, CORR_FOCUS_OVERLAY_META);
    });
}

function refreshCorrFocus() {
    var plot = document.getElementById('correlationPlot');
    if (!plot || !plot.data || !plot.data[0]) return;
    var texts = plot.data[0].text || [];
    var pi = -1;
    if (focusedTextName) {
        for (var i = 0; i < texts.length; i++) if (texts[i] === focusedTextName) { pi = i; break; }
    }
    ensureOverlay(plot, CORR_FOCUS_OVERLAY_META).then(function() {
        var idx = findTrace(plot, CORR_FOCUS_OVERLAY_META);
        if (idx < 0) return null;
        if (pi < 0) return Plotly.restyle(plot, { x: [[]], y: [[]], text: [[]] }, [idx]);
        var update = {
            x: [[plot.data[0].x[pi]]], y: [[plot.data[0].y[pi]]],
            text: [[focusedTextName]]
        };
        var bc = baseColor(plot, pi);
        if (typeof bc === 'string') update['marker.color'] = [[bc]];
        return Plotly.restyle(plot, update, [idx]);
    }).then(function() {
        return overlayFront(plot, CORR_FOCUS_OVERLAY_META);
    }).then(refreshCorrConnections).catch(function(e) {
        console.error('corr focus:', e);
    });
}

function percentileOf(key, idx) {
    var vals = metricsData[key];
    if (!vals || idx < 0 || idx >= vals.length) return null;
    var fv = vals[idx];
    if (!isNum(fv)) return { percentile: null, value: fv };
    var valid = [];
    for (var i = 0; i < vals.length; i++) if (isNum(vals[i])) valid.push(vals[i]);
    if (!valid.length) return { percentile: null, value: fv };
    valid.sort(function(a, b) { return a - b; });
    return { percentile: (countLE(valid, fv) / valid.length) * 100, value: fv };
}

function updatePercentileLabel() {
    var lbl = document.getElementById('focusMetricPercentile');
    if (!lbl) return;
    if (!focusedTextName || focusedTextIndex < 0) { lbl.textContent = ''; return; }
    var mlabel = metricDisplayNames[currentMetric] || currentMetric;
    var info = percentileOf(currentMetric, focusedTextIndex);
    if (!info || info.percentile === null) {
        lbl.textContent = mlabel + ': no score';
        return;
    }
    var whole = Math.abs(info.value - Math.round(info.value)) < 1e-9;
    var disp = whole ? String(Math.round(info.value)) : fmt(info.value, 4);
    lbl.textContent = mlabel + ': ' + ordinal(info.percentile) + ' percentile (' + disp + ')';
}

function afterFocusChange() {
    refreshFocusMarker();
    refreshCorrFocus();
    updateTextList();
    updatePercentileLabel();
    if (viewMode === 'functionWord') updateFunctionWordPlot();
}

function focusText(name) {
    var typed = (name || '').trim();
    if (!typed) {
        focusedTextName = null;
        focusedTextIndex = -1;
        afterFocusChange();
        return;
    }
    var match = textLookup[typed.toLowerCase()];
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
}

function flashPoint(idx) {
    var sizes = buildSizes();
    var ops = buildOpacity();
    for (var i = 0; i < ops.length; i++) if (ops[i] !== 0) ops[i] = CONFIG.dimmedOpacity;
    sizes[idx] = CONFIG.highlightSize;
    ops[idx] = 1;
    Plotly.restyle(mainPlot, { 'marker.size': [sizes], 'marker.opacity': [ops] }, [0]);
    setTimeout(function() {
        Plotly.restyle(mainPlot, { 'marker.size': [buildSizes()], 'marker.opacity': [buildOpacity()] }, [0])
            .then(refreshFocusMarker);
    }, CONFIG.highlightDuration);
}

function updateTextList() {
    var order = document.getElementById('textSortOrder').value;
    var list = document.getElementById('textList');
    var head = document.getElementById('textListHeading');
    list.innerHTML = '';

    if (semanticLineModeEnabled && focusedTextName) {
        var isB = focusSimilarityMode === 'burrows';
        var modeLbl = isB ? 'Stylistically (Burrows)' : 'Semantically';
        var scoreLbl = isB ? 'burrows' : 'cosine';
        head.textContent = 'Most ' + modeLbl + ' Similar to ' + focusedTextName + ':';
        var nb = focusNeighbors(focusedTextIndex);
        for (var i = 0; i < nb.length; i++) {
            var row = makeDiv((i + 1) + '. ' + nb[i].text + ' (' + scoreLbl + ': ' + fmt(nb[i].similarity, 4) + ')');
            row.title = 'Click to puff this text on the map (does not change focus)';
            (function(idx) {
                row.addEventListener('click', function() { flashPoint(idx); });
            })(nb[i].index);
            list.appendChild(row);
        }
        return;
    }

    head.textContent = 'Texts';
    var vals = metricsData[currentMetric];
    var uni = fightingWordsUnigrams[currentMetric];
    var useVals = vals;
    var items = [];
    for (var j = 0; j < textNames.length; j++) {
        if (useVals && isNum(useVals[j])) items.push({ name: textNames[j], index: j, value: useVals[j] });
    }
    if (order === 'alpha-asc') items.sort(function(a, b) { return a.name.localeCompare(b.name); });
    else if (order === 'alpha-desc') items.sort(function(a, b) { return b.name.localeCompare(a.name); });
    else if (order === 'metric-asc') items.sort(function(a, b) { return (a.value || 0) - (b.value || 0); });
    else if (order === 'metric-desc') items.sort(function(a, b) { return (b.value || 0) - (a.value || 0); });

    for (var k = 0; k < items.length; k++) {
        var u = uni ? uni[items[k].name] : null;
        var txt = (k + 1) + '. ' + items[k].name + ' (' + fmt(items[k].value, 4) + ')' + (u ? ' ' + u : '');
        var r = makeDiv(txt);
        r.title = 'Click to puff this text on the map (does not change focus)';
        if (items[k].index === focusedTextIndex) r.classList.add('focusedTextItem');
        (function(idx) {
            r.addEventListener('click', function() { flashPoint(idx); });
        })(items[k].index);
        list.appendChild(r);
    }
}

function updateColorInputs() {
    var m = mainPlot.data[0] && mainPlot.data[0].marker;
    if (!m) return;
    document.getElementById('colorMin').value = m.cmin;
    document.getElementById('colorMax').value = m.cmax;
}

function applyColorScale(lo, hi) {
    Plotly.restyle(mainPlot, { 'marker.cmin': lo, 'marker.cmax': hi }, [0]).then(function() {
        currentCmin = lo;
        currentCmax = hi;
        refreshFocusMarker();
    });
}

function zoomColor(factor) {
    var lo = parseFloat(document.getElementById('colorMin').value);
    var hi = parseFloat(document.getElementById('colorMax').value);
    var range = hi - lo;
    var center = (lo + hi) / 2;
    var nlo = center - range * factor;
    var nhi = center + range * factor;
    document.getElementById('colorMin').value = nlo;
    document.getElementById('colorMax').value = nhi;
    applyColorScale(nlo, nhi);
}

function changeMetric(key) {
    var data = colorData(key);
    if (!data) return;
    var range = metricRange(data);
    if (!range) return;
    var cluster = isClusterKey(key);
    var scale = cluster ? CLUSTER_TURBO_COLORSCALE : (metricColorScales[key] || CONFIG.defaultColorScale);

    currentMetric = key;
    currentColorDataForPlot = data;
    currentColorscale = scale;
    currentCmin = range.min;
    currentCmax = range.max;

    var markerUpdate = {
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
    ]).then(function() {
        updateColorInputs();
        updateTextList();
        updatePercentileLabel();
        refreshFocusMarker();
    });
}

function collectPoints(xs, ys) {
    var out = { texts: [], x: [], y: [], colors: [] };
    for (var i = 0; i < xs.length; i++) {
        if (!isNum(xs[i]) || !isNum(ys[i])) continue;
        out.texts.push(textNames[i]);
        out.x.push(xs[i]);
        out.y.push(ys[i]);
        out.colors.push(i);
    }
    return out;
}

function outlierList(parent, title, items) {
    var h = makeDiv(title, '4px');
    h.style.marginTop = '10px';
    h.style.fontWeight = '600';
    parent.appendChild(h);
    for (var i = 0; i < items.length; i++) {
        parent.appendChild(makeDiv((i + 1) + '. ' + items[i].text + ' (|res|=' + fmt(items[i].absResidual, 4) + ')', '3px'));
    }
}

function updateCorrStats(points, corr, type, model) {
    var body = clearBody('correlationStats');
    if (!body) return;
    var h = document.createElement('strong');
    h.textContent = 'Correlation Statistics';
    body.appendChild(h);
    var wrap = document.createElement('div');
    wrap.style.marginTop = '8px';
    wrap.appendChild(makeDiv('Sample Size: ' + points.x.length, '6px'));
    if (corr !== null) {
        var abs = Math.abs(corr);
        var strength = ' (Weak)';
        if (abs >= 0.7) strength = ' (Strong)';
        else if (abs >= 0.4) strength = ' (Moderate)';
        wrap.appendChild(makeDiv('Pearson r: ' + fmt(corr, 4) + strength, '6px'));
        wrap.appendChild(makeDiv('R\u00B2: ' + fmt(corr * corr, 4), '6px'));
    } else {
        wrap.appendChild(makeDiv('Pearson r: unavailable', '6px'));
    }
    wrap.appendChild(makeDiv('X variance/std: ' + fmt(variance(points.x), 4) + ' / ' + fmt(stddev(points.x), 4), '6px'));
    wrap.appendChild(makeDiv('Y variance/std: ' + fmt(variance(points.y), 4) + ' / ' + fmt(stddev(points.y), 4), '6px'));
    var ranked = residualRank(points.texts, points.x, points.y, type, model);
    if (ranked.length) {
        outlierList(wrap, 'Greatest Outliers', ranked.slice(0, 5));
        outlierList(wrap, 'Least Outliers', ranked.slice(-5).reverse());
    }
    body.appendChild(wrap);
}

function updateCorrPlot() {
    try {
        var xs = document.getElementById('xMetricSelect');
        var ys = document.getElementById('yMetricSelect');
        var type = document.getElementById('regressionType').value;
        correlationXMetric = xs.value;
        correlationYMetric = ys.value;
        var xd = metricsData[correlationXMetric];
        var yd = metricsData[correlationYMetric];
        if (!xd || !yd) { alert('One of the selected metrics has no data.'); return; }
        var points = collectPoints(xd, yd);
        var corr = pearson(xd, yd);
        var xl = metricDisplayNames[correlationXMetric];
        var yl = metricDisplayNames[correlationYMetric];
        var title = xl + ' vs ' + yl;
        if (corr !== null) title += ' (r = ' + corr.toFixed(3) + ')';

        var scatter = {
            x: points.x, y: points.y, mode: 'markers+text', type: 'scatter',
            name: '', text: points.texts, textposition: 'middle center',
            textfont: { size: CONFIG.pointFontSize, color: 'black' },
            marker: { size: CONFIG.normalSize, color: points.colors,
                      colorscale: 'Viridis', showscale: false,
                      line: { color: 'black', width: 1 } },
            hovertext: points.texts, hoverinfo: 'text+x+y'
        };
        var model = regModel(points.x, points.y, type);
        var line = regLine(points.x, points.y, type, model);
        var traces = [scatter];
        if (line) {
            traces.push({
                x: line.x, y: line.y, mode: 'lines', type: 'scatter',
                name: type + ' fit',
                line: { color: 'rgba(220, 53, 69, 0.2)', width: 3, dash: 'solid' },
                hoverinfo: 'skip', showlegend: false
            });
        }
        var layout = {
            title: { text: title, x: 0.5, xanchor: 'center', font: { size: 14 } },
            xaxis: { title: xl, showgrid: GRID_STYLE.showgrid, gridwidth: GRID_STYLE.gridwidth, gridcolor: GRID_STYLE.gridcolor },
            yaxis: { title: yl, showgrid: GRID_STYLE.showgrid, gridwidth: GRID_STYLE.gridwidth, gridcolor: GRID_STYLE.gridcolor },
            plot_bgcolor: '#f8f9fa', paper_bgcolor: 'white',
            hovermode: 'closest', showlegend: false,
            font: { family: 'Arial, sans-serif' },
            margin: { t: 55, l: 70, r: 70, b: 55 }
        };
        drawPlot('correlationPlot', traces, layout, null).then(function() {
            return Plotly.Plots.resize(document.getElementById('correlationPlot'));
        }).then(function() {
            refreshCorrFocus();
            addOutlines('#correlationPlot .scatterlayer .textpoint text', 'correlationStrokeText');
        }).catch(function(e) { console.error('[corr]', e); });
        updateCorrStats(points, corr, type, model);
    } catch (e) {
        console.error('[corr] failed', e);
    }
}

function populateCorrDropdowns() {
    var xs = document.getElementById('xMetricSelect');
    var ys = document.getElementById('yMetricSelect');
    if (!xs || !ys) return;
    fillMetricSelect(xs, correlationXMetric, document.getElementById('xMetricSearchInput').value);
    fillMetricSelect(ys, correlationYMetric, document.getElementById('yMetricSearchInput').value);
    if (xs.value) correlationXMetric = xs.value;
    if (ys.value) correlationYMetric = ys.value;
}

function selectedWords() {
    var out = [];
    functionWordSelection.forEach(function(w) {
        if (Array.isArray(burrowsWordFrequencies[w])) out.push(w);
    });
    out.sort(function(a, b) { return a.localeCompare(b); });
    return out;
}

function aggregatedRates() {
    var words = selectedWords();
    if (!words.length) return null;
    var rates = [];
    for (var i = 0; i < textNames.length; i++) {
        var sum = 0, bad = false;
        for (var w = 0; w < words.length; w++) {
            var v = burrowsWordFrequencies[words[w]][i];
            if (!isNum(v)) { bad = true; break; }
            sum += v;
        }
        rates.push(bad ? null : sum);
    }
    return { words: words, rates: rates };
}

function wordLabel(words) {
    if (words.length === 1) return '"' + words[0] + '"';
    if (words.length <= 4) {
        var s = [];
        for (var i = 0; i < words.length; i++) s.push('"' + words[i] + '"');
        return s.join(' + ');
    }
    return words.length + ' function words (' + words.slice(0, 3).join(', ') + ', ...)';
}

function axisTitle(words) {
    if (words.length === 1) return 'Proportion of tokens that are "' + words[0] + '"';
    return 'Combined proportion of tokens that are any of ' + wordLabel(words);
}

function updateWordSummary() {
    var s = document.getElementById('functionWordSelectionSummary');
    if (!s) return;
    var w = selectedWords();
    s.textContent = w.length ? 'Selected (' + w.length + '): ' + w.join(', ') : 'No function words selected.';
}

function populateWordControls() {
    var sel = document.getElementById('functionWordSelect');
    if (!sel) return;
    var filter = document.getElementById('functionWordSearchInput').value.trim().toLowerCase();
    var words = burrowsFunctionWords || [];
    sel.innerHTML = '';
    for (var i = 0; i < words.length; i++) {
        if (filter && words[i].toLowerCase().indexOf(filter) === -1) continue;
        var o = document.createElement('option');
        o.value = words[i];
        o.text = words[i];
        o.selected = functionWordSelection.has(words[i]);
        sel.appendChild(o);
    }
    if (!sel.options.length) {
        var o2 = document.createElement('option');
        o2.value = ''; o2.text = 'No matching function words'; o2.disabled = true;
        sel.appendChild(o2);
    }
    document.getElementById('functionWordPlotModeSelect').value = functionWordPlotMode;
    updateWordSummary();
}

function focusedCluster() {
    if (focusedTextIndex < 0 || !focusClusterMetric) return null;
    var data = metricsData[focusClusterMetric];
    if (!data || focusedTextIndex >= data.length) return null;
    return isNum(data[focusedTextIndex]) ? data[focusedTextIndex] : null;
}

function wordCtx(words, rates) {
    var validIdx = [], validVal = [];
    for (var i = 0; i < rates.length; i++) {
        if (isNum(rates[i])) { validIdx.push(i); validVal.push(rates[i]); }
    }
    var m = mean(validVal);
    var s = stddev(validVal);
    if (!isFinite(s) || s <= 0) s = Math.max(1e-9, m * 1e-3);
    var sorted = validVal.slice().sort(function(a, b) { return a - b; });
    var dmin = sorted[0], dmax = sorted[sorted.length - 1];
    var margin = Math.max(0.02 * s, 1e-6);

    var ctx = {
        words: words, wordRates: rates, validIndices: validIdx, validValues: validVal,
        sortedValues: sorted, mean: m, std: s, dataMin: dmin, dataMax: dmax,
        minX: Math.max(0, Math.min(dmin - margin, m - 4 * s)),
        maxX: Math.max(dmax + margin, m + 4 * s),
        wordLabel: wordLabel(words),
        axisTitle: axisTitle(words),
        zOf: function(r) { return (r - m) / s; },
        clusterRates: null, clusterMean: null, clusterStd: null, clusterPercentileMean: null,
        focusRate: null, focusPercentile: null, focusZ: null, focusNormalPercentile: null
    };

    var cdata = focusClusterMetric ? metricsData[focusClusterMetric] : null;
    var fcv = focusedCluster();
    ctx.isInFocusedCluster = function(i) {
        return fcv !== null && cdata && cdata[i] === fcv;
    };
    if (fcv !== null) {
        var cr = [];
        for (var j = 0; j < validIdx.length; j++) {
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
    return ctx;
}

function groupIdx(ctx) {
    var g = { focus: [], cluster: [], other: [] };
    for (var i = 0; i < ctx.validIndices.length; i++) {
        var idx = ctx.validIndices[i];
        if (idx === focusedTextIndex) g.focus.push(idx);
        else if (ctx.isInFocusedCluster(idx)) g.cluster.push(idx);
        else g.other.push(idx);
    }
    return g;
}

function markerTrace(idxs, ptFn, name, mk, hover) {
    var pts = [];
    for (var i = 0; i < idxs.length; i++) pts.push(ptFn(idxs[i]));
    var xs = [], ys = [], labels = [];
    for (var j = 0; j < pts.length; j++) {
        xs.push(pts[j].x); ys.push(pts[j].y); labels.push(pts[j].label);
    }
    return {
        x: xs, y: ys, text: labels, mode: 'markers', type: 'scatter',
        name: name, marker: mk, hoverinfo: 'text+x', hovertemplate: hover
    };
}

function groupTraces(groups, ptFn, hover) {
    var traces = [];
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
    return traces;
}

function focusStar(x, y, text, hover) {
    return {
        x: [x], y: [y], mode: 'markers+text', type: 'scatter',
        name: 'Focus text', text: [text], textposition: 'top center',
        textfont: { size: 10, color: '#000' },
        marker: { symbol: 'star', size: CONFIG.focusStarSize, color: '#ffd400',
                  line: { color: '#111', width: 2 } },
        hoverinfo: hover ? 'text+x' : 'text',
        hovertemplate: hover || '%{text}<extra></extra>'
    };
}

function clusterDiamond(x, y, text) {
    return {
        x: [x], y: [y], mode: 'markers+text', type: 'scatter',
        name: 'Cluster mean', showlegend: false, text: [text],
        textposition: 'bottom center',
        textfont: { size: 10, color: '#7a0403' },
        marker: { symbol: 'diamond', size: 14, color: '#f36315',
                  line: { color: '#7a0403', width: 2 } },
        hoverinfo: 'text', hovertemplate: '%{text}<extra></extra>'
    };
}

function clusterLine(x, topY, ctx) {
    return {
        x: [x, x], y: [0, topY], mode: 'lines', type: 'scatter',
        name: ctx.clusterRates ? 'Focused cluster mean (n=' + ctx.clusterRates.length + ')' : 'Focused cluster mean',
        line: { color: '#7a0403', width: 2, dash: 'dash' }, hoverinfo: 'skip'
    };
}

function histTrace(values, s, minX, maxX, kind) {
    var count = values.length;
    var bw = s > 0 && count > 1 ? 3.49 * s * Math.pow(count, -1 / 3) : 0;
    if (!isFinite(bw) || bw <= 0) bw = Math.max((maxX - minX) / 20, 1e-9);
    var bc = Math.max(10, Math.min(60, Math.ceil((maxX - minX) / bw)));
    var ew = (maxX - minX) / bc;
    var bins = [];
    for (var i = 0; i < bc; i++) bins.push(0);
    var centers = [];
    for (var j = 0; j < bc; j++) centers.push(minX + ew * (j + 0.5));
    for (var k = 0; k < values.length; k++) {
        var idx = Math.floor((values[k] - minX) / ew);
        if (idx < 0) idx = 0;
        if (idx > bc - 1) idx = bc - 1;
        bins[idx]++;
    }
    var ys = [];
    for (var m = 0; m < bc; m++) ys.push(bins[m] / (count * ew));
    var format = kind === 'z' ? '%{x:.2f}' : '%{x:.4f}';
    return {
        x: centers, y: ys, type: 'bar',
        name: kind === 'z' ? 'Observed z-scores' : 'Observed rate',
        marker: { color: '#EA476750', line: { color: '#EA4767', width: 1 } },
        hoverinfo: 'x+y',
        hovertemplate: kind + ': ' + format + '<br>density: %{y:.4f}<extra></extra>'
    };
}

function wordLayout(title, xaxis, yaxis, bottomMargin) {
    var xa = {};
    xa.showgrid = GRID_STYLE.showgrid; xa.gridwidth = GRID_STYLE.gridwidth;
    xa.gridcolor = GRID_STYLE.gridcolor; xa.zeroline = false;
    for (var k in xaxis) xa[k] = xaxis[k];
    var ya = {};
    ya.showgrid = GRID_STYLE.showgrid; ya.gridwidth = GRID_STYLE.gridwidth;
    ya.gridcolor = GRID_STYLE.gridcolor; ya.zeroline = false;
    for (var k2 in yaxis) ya[k2] = yaxis[k2];
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
}

function wordTitle(prefix, ctx) {
    return prefix + ' ' + ctx.wordLabel + ' rate';
}

function focusAnnotation(ctx) {
    var base = { xref: 'paper', yref: 'paper', x: 0.01, y: 0.99,
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
}

function zView(ctx) {
    var zmin = Math.min(-4, ctx.zOf(ctx.dataMin) - 0.25);
    var zmax = Math.max(4, ctx.zOf(ctx.dataMax) + 0.25);
    function stdNorm(z) { return normalPdf(z, 0, 1); }
    var curve = sampleCurve(zmin, zmax, 360, stdNorm);
    var traces = [];
    if (functionWordShowHistogram) {
        var zvals = [];
        for (var i = 0; i < ctx.validValues.length; i++) zvals.push(ctx.zOf(ctx.validValues[i]));
        traces.push(histTrace(zvals, 1, zmin, zmax, 'z'));
    }
    if (ctx.focusZ !== null) {
        var sx = [], sy = [];
        for (var j = 0; j < curve.x.length; j++) {
            if (curve.x[j] <= ctx.focusZ) { sx.push(curve.x[j]); sy.push(curve.y[j]); }
        }
        var cz = Math.max(zmin, Math.min(zmax, ctx.focusZ));
        sx.push(cz);
        sy.push(stdNorm(cz));
        traces.push({
            x: sx, y: sy, mode: 'lines', type: 'scatter',
            fill: 'tozeroy', fillcolor: 'rgba(255, 212, 0, 0.40)',
            line: { width: 0 },
            name: 'Area below focus (' + ctx.focusNormalPercentile.toFixed(1) + '%)',
            hoverinfo: 'skip'
        });
    }
    traces.push({
        x: curve.x, y: curve.y, mode: 'lines', type: 'scatter',
        name: 'Standard normal', line: { color: '#d93806', width: 3 }, hoverinfo: 'skip'
    });
    if (ctx.clusterMean !== null) {
        var cz2 = ctx.zOf(ctx.clusterMean);
        var cy = stdNorm(cz2);
        traces.push(clusterLine(cz2, cy, ctx));
        traces.push(clusterDiamond(cz2, cy,
            'cluster mean<br>z=' + cz2.toFixed(2) + '<br>' + ctx.clusterPercentileMean.toFixed(1) + '%'));
    }
    if (functionWordShowMarkers) {
        var groups = groupIdx(ctx);
        var ptFn = function(i) {
            var z = ctx.zOf(ctx.wordRates[i]);
            var pct = empPct(ctx.sortedValues, ctx.wordRates[i]);
            return { x: z, y: stdNorm(z), label: textNames[i] + ' (' + pct.toFixed(1) + '%)' };
        };
        var grp = groupTraces(groups, ptFn, '%{text}<br>z: %{x:.2f}<extra></extra>');
        for (var k = 0; k < grp.length; k++) traces.push(grp[k]);
    }
    if (ctx.focusZ !== null) {
        traces.push(focusStar(ctx.focusZ, stdNorm(ctx.focusZ),
            focusedTextName + '<br>z=' + ctx.focusZ.toFixed(2) + '<br>' + ctx.focusPercentile.toFixed(1) + '%'));
    }
    var low = Math.ceil(zmin), high = Math.floor(zmax);
    var step = Math.max(1, Math.ceil((high - low) / 12));
    var tvals = [], tlabels = [];
    for (var t = low; t <= high; t += step) {
        tvals.push(t);
        tlabels.push(t + '\u03C3<br>' + (normalCdf(t) * 100).toFixed(1) + '%');
    }
    var layout = wordLayout(wordTitle('Z-score distribution of', ctx),
        { title: 'Z-score (std devs from mean) of: ' + ctx.axisTitle,
          zeroline: true, zerolinecolor: '#999',
          range: [zmin, zmax], tickmode: 'array', tickvals: tvals, ticktext: tlabels },
        { title: 'Density (standard normal)' }, 90);
    layout.annotations = [focusAnnotation(ctx)];
    return { traces: traces, layout: layout };
}

function pctView(ctx) {
    var count = ctx.sortedValues.length;
    var cdfX = [ctx.minX], cdfY = [0];
    for (var i = 0; i < count; i++) {
        cdfX.push(ctx.sortedValues[i]);
        cdfY.push(((i + 1) / count) * 100);
    }
    var traces = [
        { x: cdfX, y: cdfY, mode: 'lines', type: 'scatter', name: 'Empirical CDF',
          line: { shape: 'hv', color: '#d93806', width: 3 }, hoverinfo: 'skip' },
        { x: [ctx.minX, ctx.maxX], y: [50, 50], mode: 'lines', type: 'scatter',
          name: 'Median (50th)', line: { color: 'rgba(120,120,120,0.65)', width: 1, dash: 'dot' },
          hoverinfo: 'skip' }
    ];
    var pctOf = function(r) { return empPct(ctx.sortedValues, r); };
    if (functionWordShowMarkers && !hideGrayPoints) {
        var others = [];
        for (var j = 0; j < ctx.validIndices.length; j++) {
            var idx = ctx.validIndices[j];
            if (idx !== focusedTextIndex && !ctx.isInFocusedCluster(idx)) others.push(ctx.wordRates[idx]);
        }
        if (others.length) {
            var ys = [];
            for (var k = 0; k < others.length; k++) ys.push(pctOf(others[k]));
            traces.push({
                x: others, y: ys, mode: 'markers', type: 'scatter', name: 'Other texts',
                marker: { size: 5, color: 'rgba(120,120,120,0.45)',
                          line: { color: 'rgba(80,80,80,0.75)', width: 0.5 } },
                hoverinfo: 'skip'
            });
        }
    }
    if (ctx.clusterRates && ctx.clusterRates.length > 0) {
        var cys = [];
        for (var m = 0; m < ctx.clusterRates.length; m++) cys.push(pctOf(ctx.clusterRates[m]));
        traces.push({
            x: ctx.clusterRates, y: cys, mode: 'markers', type: 'scatter',
            name: 'Focused cluster (n=' + ctx.clusterRates.length + ')',
            marker: { size: 9, color: FOCUS_CLUSTER_COLOR,
                      line: { color: FOCUS_CLUSTER_LINE_COLOR, width: 1.5 } },
            hoverinfo: 'skip'
        });
    }
    if (ctx.clusterMean !== null) {
        traces.push(clusterDiamond(ctx.clusterMean, ctx.clusterPercentileMean,
            'cluster mean<br>' + ctx.clusterPercentileMean.toFixed(1) + '%'));
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
    var layout = wordLayout(wordTitle('Percentile of', ctx),
        { title: ctx.axisTitle, range: [ctx.minX, ctx.maxX] },
        { title: 'Percentile', range: [0, 100],
          tickvals: [0, 10, 25, 50, 75, 90, 100], ticksuffix: '%' }, 80);
    return { traces: traces, layout: layout };
}

function densityView(ctx) {
    var at = function(x) { return normalPdf(x, ctx.mean, ctx.std); };
    var curve = sampleCurve(ctx.minX, ctx.maxX, 240, at);
    var traces = [];
    if (functionWordShowHistogram) traces.push(histTrace(ctx.validValues, ctx.std, ctx.minX, ctx.maxX, 'rate'));
    traces.push({
        x: curve.x, y: curve.y, mode: 'lines', type: 'scatter',
        name: 'Normal fit', line: { color: '#d93806', width: 3 }, hoverinfo: 'skip'
    });
    if (ctx.clusterMean !== null) traces.push(clusterLine(ctx.clusterMean, at(ctx.clusterMean), ctx));
    if (functionWordShowMarkers) {
        var groups = groupIdx(ctx);
        var ptFn = function(i) {
            return { x: ctx.wordRates[i], y: at(ctx.wordRates[i]), label: textNames[i] };
        };
        var hover = '%{text}<br>rate: %{x:.4f}<extra></extra>';
        var grp = groupTraces(groups, ptFn, hover);
        for (var i = 0; i < grp.length; i++) traces.push(grp[i]);
        if (groups.focus.length) traces.push(focusStar(ctx.focusRate, at(ctx.focusRate), focusedTextName, hover));
    }
    var layout = wordLayout(wordTitle('Distribution of', ctx),
        { title: ctx.axisTitle, range: [ctx.minX, ctx.maxX] },
        { title: 'Density' }, 80);
    return { traces: traces, layout: layout };
}

function updateWordStats(ctx) {
    var body = clearBody('functionWordStats');
    if (!body) return;
    var h = document.createElement('strong');
    h.textContent = ctx.wordLabel + ' statistics';
    body.appendChild(h);
    var wrap = document.createElement('div');
    wrap.style.marginTop = '8px';
    function add(l, v) { wrap.appendChild(makeDiv(l + ': ' + v, '4px')); }
    if (ctx.words.length > 1) add('Aggregated words', ctx.words.length + ' (rates summed)');
    add('Sample size', ctx.validValues.length);
    add('Mean rate', ctx.mean.toFixed(6));
    add('Std dev', ctx.std.toFixed(6));
    var med = median(ctx.sortedValues);
    add('Median rate', med === null ? 'N/A' : med.toFixed(6));
    add('Min / Max', ctx.dataMin.toFixed(6) + ' / ' + ctx.dataMax.toFixed(6));
    if (ctx.focusRate !== null) {
        add('Focus text rate', ctx.focusRate.toFixed(6) + ' (z=' + ctx.focusZ.toFixed(3) + ')');
        add('Focus percentile (actual)', ctx.focusPercentile.toFixed(2) + '% (' + ordinal(ctx.focusPercentile) + ')');
        add('Focus percentile (normal)', ctx.focusNormalPercentile.toFixed(2) + '% (' + ordinal(ctx.focusNormalPercentile) + ')');
    }
    var fcv = focusedCluster();
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
    body.appendChild(wrap);
}

function updateFunctionWordPlot() {
    try {
        var agg = aggregatedRates();
        if (!agg) {
            emptyPlot('functionWordPlot', 'Select at least one function word');
            var b = clearBody('functionWordStats');
            if (b) b.appendChild(makeDiv('No function words selected.'));
            return;
        }
        var valid = 0;
        for (var i = 0; i < agg.rates.length; i++) if (isNum(agg.rates[i])) valid++;
        if (valid < 2) {
            emptyPlot('functionWordPlot', 'Not enough data for ' + wordLabel(agg.words));
            clearBody('functionWordStats');
            return;
        }
        var ctx = wordCtx(agg.words, agg.rates);
        var view;
        if (functionWordPlotMode === 'zscore') view = zView(ctx);
        else if (functionWordPlotMode === 'cdf') view = pctView(ctx);
        else view = densityView(ctx);
        drawPlot('functionWordPlot', view.traces, view.layout, null)
            .then(function() { return Plotly.Plots.resize(document.getElementById('functionWordPlot')); })
            .catch(function(e) { console.error('[fn]', e); });
        updateWordStats(ctx);
    } catch (e) {
        console.error('[fn]', e);
    }
}

function computeWordStats() {
    if (!Array.isArray(burrowsFunctionWords) || !burrowsWordFrequencies) return null;
    var stats = {}, zscores = {};
    for (var i = 0; i < burrowsFunctionWords.length; i++) {
        var word = burrowsFunctionWords[i];
        var rates = burrowsWordFrequencies[word];
        if (!Array.isArray(rates)) continue;
        var valid = [];
        for (var j = 0; j < rates.length; j++) if (isNum(rates[j])) valid.push(rates[j]);
        if (valid.length < 3) continue;
        var m = mean(valid), s = stddev(valid);
        if (!isFinite(s) || s <= 0) continue;
        stats[word] = true;
        var z = [];
        for (var k = 0; k < rates.length; k++) {
            z.push(isNum(rates[k]) ? (rates[k] - m) / s : null);
        }
        zscores[word] = z;
    }
    return { stats: stats, zscores: zscores };
}

function computeDist() {
    var ws = deltaState.wordStats;
    if (!ws) return null;
    var zarr = [];
    for (var w in ws.stats) zarr.push(ws.zscores[w]);
    var deltas = [];
    for (var i = 0; i < textNames.length; i++) {
        for (var j = i + 1; j < textNames.length; j++) {
            var sum = 0, cnt = 0;
            for (var z = 0; z < zarr.length; z++) {
                if (zarr[z][i] === null || zarr[z][j] === null) continue;
                sum += Math.abs(zarr[z][i] - zarr[z][j]);
                cnt++;
            }
            if (cnt) deltas.push(sum / cnt);
        }
    }
    deltas.sort(function(a, b) { return a - b; });
    return deltas;
}

function distPct(v) {
    var d = deltaState.deltaDistribution;
    if (!d || !d.length) return null;
    return (countLE(d, v) / d.length) * 100;
}

function defaultDeltaPartner(idx) {
    var row = Array.isArray(burrowsSimilarity) ? burrowsSimilarity[idx] : null;
    if (row) {
        var best = -Infinity, bi = -1;
        for (var i = 0; i < row.length; i++) {
            if (i !== idx && isNum(row[i]) && row[i] > best) { best = row[i]; bi = i; }
        }
        if (bi >= 0) return bi;
    }
    return idx === 0 ? Math.min(1, textNames.length - 1) : 0;
}

function deltaPair(a, b) {
    var ws = deltaState.wordStats;
    if (!ws) return null;
    var contrib = [];
    var total = 0;
    for (var w in ws.stats) {
        var za = ws.zscores[w][a], zb = ws.zscores[w][b];
        if (za === null || zb === null) continue;
        var absDiff = Math.abs(za - zb);
        var avgZ = (za + zb) / 2;
        var sameDir = za * zb > 0;
        var minAbs = Math.min(Math.abs(za), Math.abs(zb));
        contrib.push({
            word: w, zA: za, zB: zb, absDiff: absDiff, avgZ: avgZ,
            rateA: burrowsWordFrequencies[w][a], rateB: burrowsWordFrequencies[w][b],
            driverScore: sameDir ? minAbs * Math.exp(-absDiff) : 0,
            sharedScore: Math.abs(avgZ) * Math.exp(-absDiff),
            deltaContribution: absDiff
        });
        total += absDiff;
    }
    for (var i = 0; i < contrib.length; i++) {
        contrib[i].shareOfDelta = total > 0 ? (contrib[i].deltaContribution / total) * 100 : 0;
    }
    return { contributions: contrib, nWords: contrib.length, totalDelta: total,
             avgDelta: contrib.length > 0 ? total / contrib.length : 0 };
}

function sortContrib(arr, mode) {
    var key = DELTA_SORT_KEYS[mode] || 'deltaContribution';
    var copy = arr.slice();
    copy.sort(function(a, b) { return b[key] - a[key]; });
    return copy;
}

function deltaPlotHeight(words, top, bot) {
    return Math.max(800, words * 24 + top + bot);
}

function fillDeltaSelect(sel, filter, selIdx) {
    if (!sel) return null;
    var f = (filter || '').trim().toLowerCase();
    sel.innerHTML = '';
    for (var i = 0; i < textNames.length; i++) {
        if (f && textNames[i].toLowerCase().indexOf(f) === -1) continue;
        var o = document.createElement('option');
        o.value = String(i);
        o.text = textNames[i];
        sel.appendChild(o);
    }
    var has = sel.querySelector('option[value="' + selIdx + '"]');
    if (selIdx !== null && has) sel.value = String(selIdx);
    else if (sel.options.length > 0) sel.value = sel.options[0].value;
    return sel.value === '' ? null : Number(sel.value);
}

function populateDeltaSelects() {
    var a = fillDeltaSelect(document.getElementById('deltaTextASelect'),
        document.getElementById('deltaTextASearch').value, deltaState.textA);
    var b = fillDeltaSelect(document.getElementById('deltaTextBSelect'),
        document.getElementById('deltaTextBSearch').value, deltaState.textB);
    if (a !== null) deltaState.textA = a;
    if (b !== null) deltaState.textB = b;
}

function syncDeltaDropdowns() {
    var pairs = [['deltaTextASelect', deltaState.textA], ['deltaTextBSelect', deltaState.textB]];
    for (var i = 0; i < pairs.length; i++) {
        var s = document.getElementById(pairs[i][0]);
        if (s && s.querySelector('option[value="' + pairs[i][1] + '"]')) {
            s.value = String(pairs[i][1]);
        }
    }
}

function setDeltaTopN(n) {
    deltaState.topN = n;
    document.getElementById('deltaTopNRange').value = String(n);
    document.getElementById('deltaTopNValue').textContent = n;
}

function initDelta() {
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
}

function recomputeDelta() {
    deltaState.wordStats = computeWordStats();
    deltaState.deltaDistribution = computeDist();
    deltaState.totalWordCount = Object.keys(deltaState.wordStats ? deltaState.wordStats.stats : {}).length;
}

function updateDeltaPlot() {
    if (!deltaState.wordStats) deltaState.wordStats = computeWordStats();
    if (!deltaState.wordStats) {
        emptyPlot('deltaPlot', 'No Burrows function-word data available');
        updateDeltaStats(null);
        return;
    }
    renderDeltaPair();
}

function drawDelta(traces, layout, height) {
    drawPlot('deltaPlot', traces, layout, height).then(function() {
        document.getElementById('deltaContainer').scrollTop = 0;
    }).catch(function(e) { console.error('[delta]', e); });
}

function renderDeltaPair() {
    var a = deltaState.textA, b = deltaState.textB;
    if (a === null || b === null || a < 0 || b < 0 || a === b) {
        emptyPlot('deltaPlot', 'Select two different texts to compare');
        updateDeltaStats(null);
        return;
    }
    var pair = deltaPair(a, b);
    if (!pair || !pair.contributions.length) {
        emptyPlot('deltaPlot', 'Insufficient overlapping data for these texts');
        updateDeltaStats(null);
        return;
    }
    var sorted = sortContrib(pair.contributions, deltaState.sortMode);
    var n = Math.min(deltaState.topN, sorted.length);
    var shown = sorted.slice(0, n).reverse();
    var maxDiff = 0.0001;
    for (var i = 0; i < shown.length; i++) if (shown[i].absDiff > maxDiff) maxDiff = shown[i].absDiff;

    var gaps = [];
    for (var j = 0; j < shown.length; j++) {
        var it = shown[j];
        var ratio = it.absDiff / maxDiff;
        var r = Math.round(39 + 208 * ratio);
        var g = Math.round(164 - 94 * ratio);
        var bl = Math.round(217 - 148 * ratio);
        gaps.push({
            x: [Math.min(it.zA, it.zB), Math.max(it.zA, it.zB)],
            y: [it.word, it.word], mode: 'lines', type: 'scatter',
            line: { color: 'rgba(' + r + ', ' + g + ', ' + bl + ', 0.45)', width: 14 },
            hoverinfo: 'skip', showlegend: false
        });
    }

    var hovers = [], words = [], xsA = [], xsB = [];
    for (var k = 0; k < shown.length; k++) {
        var s = shown[k];
        hovers.push('<b>' + s.word + '</b><br>Rate A: ' + (s.rateA * 100).toFixed(3) + '% (z=' + s.zA.toFixed(2) + ')<br>' +
                    'Rate B: ' + (s.rateB * 100).toFixed(3) + '% (z=' + s.zB.toFixed(2) + ')<br>' +
                    '|\u0394z|: ' + s.absDiff.toFixed(3) + ' (' + s.shareOfDelta.toFixed(2) + '% of \u0394)<br>' +
                    'Shared z: ' + s.avgZ.toFixed(2));
        words.push(s.word);
        xsA.push(s.zA);
        xsB.push(s.zB);
    }

    var traces = gaps.slice();
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

    var pct = distPct(pair.avgDelta);
    var sub = 'Burrows \u0394 = ' + pair.avgDelta.toFixed(4) +
        (pct !== null ? ' (' + ordinal(pct) + ' percentile among all pairs)' : '') +
        ' | ' + pair.nWords + ' function words | showing top ' + n +
        (n < pair.nWords ? ' of ' + pair.nWords + ' (raise slider or press "Show all")' : '');

    var layout = {
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
}

function deltaRow(word, main, sec) {
    var r = document.createElement('div');
    r.className = 'deltaWordRow';
    var a = document.createElement('span');
    a.textContent = word;
    var b = document.createElement('span');
    b.className = 'val';
    b.innerHTML = main + (sec ? '<br>' + sec : '');
    r.appendChild(a);
    r.appendChild(b);
    return r;
}

function deltaHead(text) {
    var h = document.createElement('strong');
    h.textContent = text;
    h.style.cssText = 'font-size:12px; display:block; margin-bottom:6px;';
    return h;
}

function deltaNote(html, css) {
    var n = document.createElement('div');
    n.style.cssText = css;
    n.innerHTML = html;
    return n;
}

function deltaSection(title, color, count, buildList) {
    var det = document.createElement('details');
    det.style.marginTop = '6px';
    var sum = document.createElement('summary');
    sum.className = 'deltaSectionHead';
    sum.style.cssText = 'cursor: pointer; user-select: none; color: ' + color + ';';
    sum.textContent = title + ' (' + count + ')';
    det.appendChild(sum);
    var listWrap = document.createElement('div');
    listWrap.style.cssText = 'max-height: 320px; overflow-y: auto; padding-right: 4px; margin-top: 2px;';
    buildList(listWrap);
    det.appendChild(listWrap);
    return det;
}

function updateDeltaStats(pair) {
    var body = clearBody('deltaStats');
    if (!body) return;
    if (!pair) {
        var n = makeDiv('No pair selected.');
        n.style.fontSize = '11px';
        body.appendChild(n);
        return;
    }
    var a = deltaState.textA, b = deltaState.textB;
    function pc(r) { return (r * 100).toFixed(2); }

    body.appendChild(deltaHead('\u0394 Delta Explorer'));
    body.appendChild(deltaNote(
        '<b>Text A:</b> ' + textNames[a] + '<br><b>Text B:</b> ' + textNames[b],
        'font-size:10px; color:#555; margin-bottom:8px;'));

    var pct = distPct(pair.avgDelta);
    body.appendChild(deltaNote(
        '<b>Burrows \u0394 = ' + pair.avgDelta.toFixed(4) + '</b>' +
        (pct !== null ? '<br>' + ordinal(pct) + ' percentile of all ' + deltaState.deltaDistribution.length + ' pairs' +
        (deltaState.deltaDistribution.length > 1000000 ? ' (that\'s a lot of pairs!)' : '') +
                       '<br><span style="color:#555;">(lower = more similar)</span>' : '') +
        '<br>Words compared: ' + pair.nWords,
        'border:1px solid #ddd; padding:6px; border-radius:3px; margin-bottom:8px; font-size:10px;'));

    var seps = sortContrib(pair.contributions, 'separators');
    body.appendChild(deltaSection(
        'Top separators (|\u0394z| high)', '#ff0000', seps.length,
        function(listWrap) {
            for (var i = 0; i < seps.length; i++) {
                var s = seps[i];
                listWrap.appendChild(deltaRow(
                    (i + 1) + '. ' + s.word,
                    '|\u0394z|=' + s.absDiff.toFixed(2),
                    s.shareOfDelta.toFixed(1) + '% of \u0394<br>A: ' + pc(s.rateA) + '% (z=' + s.zA.toFixed(2) + ')<br>B: ' + pc(s.rateB) + '% (z=' + s.zB.toFixed(2) + ')'
                ));
            }
        }));

    var drv = sortContrib(pair.contributions, 'drivers');
    body.appendChild(deltaSection(
        'Similarity drivers (both distinctive, same direction)', '#008000', drv.length,
        function(listWrap) {
            listWrap.appendChild(deltaNote(
                'Words where A & B are both distinctive the same way. words that bind them together.',
                'font-size:9px; color:#777; margin-bottom:4px;'));
            for (var j = 0; j < drv.length; j++) {
                var d = drv[j];
                listWrap.appendChild(deltaRow(
                    (j + 1) + '. ' + d.word,
                    '',
                    '|\u0394z|=' + d.absDiff.toFixed(2) + ' | shared z=' + d.avgZ.toFixed(2) +
                    '<br>A: ' + pc(d.rateA) + '% | B: ' + pc(d.rateB) + '%'
                ));
            }
        }));

    var shr = sortContrib(pair.contributions, 'shared');
    body.appendChild(deltaSection(
        'Shared tendencies (both unusual, same direction)', '#ebce13', shr.length,
        function(listWrap) {
            for (var k = 0; k < shr.length; k++) {
                var sh = shr[k];
                listWrap.appendChild(deltaRow(
                    (k + 1) + '. ' + sh.word,
                    '',
                    (sh.avgZ >= 0 ? 'high' : 'low') + ' | avg z=' + sh.avgZ.toFixed(2) +
                    '<br>A: ' + pc(sh.rateA) + '% | B: ' + pc(sh.rateB) + '%'
                ));
            }
        }));

    var exWrap = document.createElement('div');
    exWrap.style.cssText = 'margin-top:8px; font-size:9px; color:#555;';
    function addExt(label, zk, rk, m) {
        var h = makeDiv('Most distinctive words in ' + label + ':');
        h.style.cssText = 'font-weight:600; margin:' + m + ';';
        exWrap.appendChild(h);
        var cp = pair.contributions.slice();
        cp.sort(function(x, y) { return Math.abs(y[zk]) - Math.abs(x[zk]); });
        cp = cp.slice(0, 4);
        for (var q = 0; q < cp.length; q++) {
            exWrap.appendChild(makeDiv(cp[q].word + ': z=' + cp[q][zk].toFixed(2) + ' (rate ' + pc(cp[q][rk]) + '%)'));
        }
    }
    addExt('A (' + textNames[a] + ')', 'zA', 'rateA', '0 0 3px 0');
    addExt('B (' + textNames[b] + ')', 'zB', 'rateB', '5px 0 3px 0');
    body.appendChild(exWrap);
}

function updateViewVis() {
    for (var mode in VIEW_ELEMENT_IDS) {
        var ids = VIEW_ELEMENT_IDS[mode];
        for (var i = 0; i < ids.length; i++) {
            var el = document.getElementById(ids[i]);
            if (el) el.style.display = mode === viewMode ? 'block' : 'none';
        }
    }
}

function updateSwitchLabel() {
    var b = document.getElementById('toggleViewBtn');
    if (!b) return;
    if (viewMode !== 'umap') b.textContent = 'Switch to Map View';
    else if (mapMode === 'burrows') b.textContent = 'Switch to Function Word Frequency';
    else b.textContent = 'Switch to Correlation View';
}

function setView(mode) {
    try {
        var isMap = mode === 'umap';
        var panels = ['metricPanel', 'colorPanel', 'textPanel'];
        for (var i = 0; i < panels.length; i++) {
            var p = document.getElementById(panels[i]);
            if (!p) continue;
            p.style.display = isMap ? (panels[i] === 'textPanel' ? 'flex' : 'block') : 'none';
        }
        viewMode = mode;
        updateViewVis();
        updateSwitchLabel();
        if (viewMode === 'correlation') updateCorrPlot();
        else if (viewMode === 'functionWord') updateFunctionWordPlot();
        else if (viewMode === 'delta') { initDelta(); updateDeltaPlot(); }
    } catch (e) {
        console.error('setView', e);
    }
}

function coordsFor(mode) {
    if (!snapshotXY()) {
        var t = mainPlot && mainPlot.data && mainPlot.data[0];
        if (t && t.x && t.y && t.x.length > 0) {
            originalPlotX = Array.prototype.slice.call(t.x);
            originalPlotY = Array.prototype.slice.call(t.y);
        } else return null;
    }
    if (mode === 'burrows' && hasBurrowsData()) {
        var xs = [], ys = [];
        for (var i = 0; i < burrowsReduced.length; i++) {
            xs.push(burrowsReduced[i][0]);
            ys.push(burrowsReduced[i][1]);
        }
        return { x: xs, y: ys };
    }
    return { x: originalPlotX, y: originalPlotY };
}

function setMapMode(mode) {
    if (mode === mapMode) return;
    if (mode === 'burrows' && !hasBurrowsData()) return;
    var c = coordsFor(mode);
    if (!c || c.x.length === 0 || c.x.length !== c.y.length) return;
    if (viewMode !== 'umap') setView('umap');
    mapMode = mode;
    var isB = mode === 'burrows';
    document.getElementById('burrowsIndicator').style.display = isB ? 'block' : 'none';
    document.getElementById('deltaExplorerBtn').style.display = isB ? 'block' : 'none';
    var eb = document.getElementById('experimentalBtn');
    eb.classList.toggle('active', isB);
    eb.textContent = isB ? 'Experimental (active)' : 'Experimental';

    Plotly.restyle(mainPlot, { x: [c.x], y: [c.y] }, [0])
        .then(function() { return resetAxes(mode); })
        .then(updateOverlay)
        .then(function() { return overlayFront(mainPlot, FOCUS_OVERLAY_META); })
        .then(refreshConnections)
        .then(function() {
            if (isB) {
                focusSimilarityMode = 'burrows';
                document.getElementById('focusSimilarityModeSelect').value = 'burrows';
            } else {
                focusSimilarityMode = 'cosine';
                document.getElementById('focusSimilarityModeSelect').value = 'cosine';
            }
            applyClusterForMode(mode);
            refreshFocusMarker();
            updateTextList();
            updateSwitchLabel();
            if (viewMode === 'functionWord') updateFunctionWordPlot();
        })
        .catch(function(e) { console.error('mapMode', e); });
}

function hasBurrowsData() {
    return Array.isArray(burrowsReduced) && burrowsReduced.length > 0;
}

function exportMap() {
    var c = coordsFor(mapMode);
    if (!c || !c.x || !c.y || !c.x.length || !c.y.length) {
        alert('No map data to export. The map may still be initialising.');
        return;
    }
    var xs = Array.prototype.slice.call(c.x);
    var ys = Array.prototype.slice.call(c.y);
    if (xs.length !== ys.length) { alert('Map data is malformed.'); return; }

    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (var i = 0; i < xs.length; i++) {
        if (xs[i] < minX) minX = xs[i];
        if (xs[i] > maxX) maxX = xs[i];
        if (ys[i] < minY) minY = ys[i];
        if (ys[i] > maxY) maxY = ys[i];
    }
    var spanX = maxX - minX || 1;
    var spanY = maxY - minY || 1;
    var xr = [minX - spanX * 0.04, maxX + spanX * 0.04];
    var yr = [minY - spanY * 0.04, maxY + spanY * 0.04];
    var da = (xr[1] - xr[0]) / (yr[1] - yr[0]);
    var MAX = 3200, MIN = 700;
    var w, h;
    if (da >= 1) { w = MAX; h = Math.round(MAX / da); }
    else { h = MAX; w = Math.round(MAX * da); }
    w = Math.max(MIN, w); h = Math.max(MIN, h);
    var ca = w / h;
    var cx = (xr[0] + xr[1]) / 2;
    var cy = (yr[0] + yr[1]) / 2;
    var xs2 = xr[1] - xr[0], ys2 = yr[1] - yr[0];
    if (ca > xs2 / ys2) xs2 = ys2 * ca;
    else ys2 = xs2 / ca;
    xr = [cx - xs2 / 2, cx + xs2 / 2];
    yr = [cy - ys2 / 2, cy + ys2 / 2];

    var hidden = document.createElement('div');
    hidden.style.cssText = 'position:fixed; left:-99999px; top:0; width:' + w + 'px; height:' + h + 'px;';
    document.body.appendChild(hidden);

    var colors = [];
    if (currentColorDataForPlot.length === xs.length) {
        for (var k = 0; k < currentColorDataForPlot.length; k++) {
            colors.push(isNum(currentColorDataForPlot[k]) ? currentColorDataForPlot[k] : null);
        }
    } else {
        for (var m = 0; m < xs.length; m++) colors.push(null);
    }

    var trace = {
        x: xs, y: ys, mode: 'markers+text', type: 'scatter',
        text: textNames.slice(0, xs.length),
        textposition: 'middle center',
        textfont: { size: 13, color: 'black' },
        marker: { size: 18, color: colors, colorscale: currentColorscale,
                  cmin: currentCmin, cmax: currentCmax, showscale: false,
                  line: { color: 'black', width: 1 } },
        hoverinfo: 'skip'
    };
    var layout = {
        width: w, height: h,
        xaxis: { range: xr, visible: false, fixedrange: true, showgrid: false, zeroline: false },
        yaxis: { range: yr, visible: false, fixedrange: true, showgrid: false, zeroline: false },
        plot_bgcolor: 'white', paper_bgcolor: 'white',
        showlegend: false, margin: { t: 0, l: 0, r: 0, b: 0 }
    };

    Plotly.newPlot(hidden, [trace], layout, { staticPlot: false })
        .then(function() { return Plotly.toImage(hidden, { format: 'svg', width: w, height: h, scale: 1 }); })
        .then(function(svgUrl) {
            var svgText = decodeSvg(svgUrl);
            if (!svgText) throw new Error('svg decode failed');
            return svgToPng(injectOutlines(svgText), w, h);
        })
        .then(function(dataUrl) {
            var link = document.createElement('a');
            link.download = 'map-' + mapMode + '-' + Date.now() + '.png';
            link.href = dataUrl;
            link.click();
        })
        .catch(function(e) {
            console.warn('SVG path failed, using PNG fallback', e);
            return Plotly.toImage(hidden, { format: 'png', width: w, height: h, scale: 1 })
                .then(function(u) {
                    var link = document.createElement('a');
                    link.download = 'map-' + mapMode + '-' + Date.now() + '.png';
                    link.href = u;
                    link.click();
                });
        })
        .catch(function(e) { console.error('export failed', e); })
        .finally(function() { document.body.removeChild(hidden); });
}

function createMainPanels() {
    var togglePanel = makePanel('toggleViewPanel',
        { left: '10px', top: '5px' }, '0.5px solid darkblue',
        '<button id="toggleViewBtn" style="font-weight: bold; min-width: 260px; text-align: left;">Switch to Correlation View</button>',
        { resizable: false, zIndex: 10001, dragLabel: 'view', minWidth: '260px' });

    makePanel('metricPanel', { left: '10px', top: '45px' }, '0.5px solid black',
        '<b>Color Metric</b><br/><input id="metricSearchInput" class="searchableMetricInput" type="text" placeholder="Search metrics..."><br/>' +
        '<select id="metricSelect" class="searchableMetricSelect"></select><br/>' +
        '<button id="toggleLabelsBtn" style="margin-top: 5px; font-size: 10px; padding: 2px 6px;">Hide Labels</button><br/>' +
        '<button id="toggleGrayPointsBtn" style="margin-top: 5px; font-size: 10px; padding: 2px 6px;">Hide Gray Points</button>',
        { minWidth: '220px', dragLabel: 'color metric' });

    var focusBtnPanel = makePanel('focusButtonPanel', { top: '5px' }, '0.5px solid black',
        '<button id="openFocusPanelBtn">Focus Text</button>' +
        '<span id="focusMetricPercentile" style="margin-left: 8px; font-size: 10px; font-weight: 500;"></span>',
        { resizable: false, zIndex: 10050, dragLabel: 'focus' });

    var focusPanel = makePanel('focusPanel', { display: 'none' }, '0.5px solid black',
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
        { zIndex: 10050, minWidth: '240px', dragLabel: 'focus text' });

    var tr = togglePanel.getBoundingClientRect();
    var fl = tr.left + tr.width + 8;
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
            '<button id="deltaExplorerBtn" title="Explore which function words drive text similarity" style="display:none; margin-top:4px;">THE \u0394 DELTA EXPLORER</button>',
            { resizable: false, zIndex: 10020, dragLabel: 'experimental' });
    }
}

function createSecondaryViews() {
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

    makePanel('functionWordControls', { left: '10px', top: '45px', display: 'none' }, '0.5px solid #7a0403',
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
        { minWidth: '260px', dragLabel: 'function word' });

    makePanel('functionWordStats', { right: '10px', top: '45px', display: 'none' }, '0.5px solid #7a0403', '',
        { minWidth: '260px', dragLabel: 'word stats' });

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
}

function wireMetricAndLabelControls() {
    var sel = document.getElementById('metricSelect');
    var search = document.getElementById('metricSearchInput');
    search.addEventListener('input', function() {
        var prev = sel.value;
        fillMetricSelect(sel, prev, search.value);
        if (sel.value && sel.value !== prev) changeMetric(sel.value);
    });
    sel.addEventListener('change', function() { changeMetric(sel.value); });
    listen('toggleLabelsBtn', 'click', function(e) {
        umapLabelsVisible = !umapLabelsVisible;
        Plotly.restyle(mainPlot, { mode: umapLabelsVisible ? 'markers+text' : 'markers' }, [0]);
        e.target.textContent = umapLabelsVisible ? 'Hide Labels' : 'Show Labels';
    });
    listen('toggleGrayPointsBtn', 'click', function(e) {
        hideGrayPoints = !hideGrayPoints;
        e.target.textContent = hideGrayPoints ? 'Show Gray Points' : 'Hide Gray Points';
        refreshFocusMarker();
        if (viewMode === 'functionWord') updateFunctionWordPlot();
    });
    listen('textSortOrder', 'change', updateTextList);
}

function wireColorScale() {
    listen('applyColorBtn', 'click', function() {
        var lo = parseFloat(document.getElementById('colorMin').value);
        var hi = parseFloat(document.getElementById('colorMax').value);
        if (isNaN(lo) || isNaN(hi) || lo >= hi) { alert('The color scale minimum must be smaller than the maximum.'); return; }
        applyColorScale(lo, hi);
    });
    listen('resetColorBtn', 'click', function() {
        var c = metricRange(currentColorDataForPlot);
        if (!c) return;
        document.getElementById('colorMin').value = c.min;
        document.getElementById('colorMax').value = c.max;
        applyColorScale(c.min, c.max);
    });
    listen('zoomInColorBtn', 'click', function() { zoomColor(CONFIG.colorScaleZoomRatio); });
    listen('zoomOutColorBtn', 'click', function() { zoomColor(1); });
    listen('exportMapBtn', 'click', exportMap);
}

function wireFocus() {
    listen('openFocusPanelBtn', 'click', function() {
        var p = document.getElementById('focusPanel');
        p.style.display = p.style.display === 'none' ? 'block' : 'none';
    });
    listen('applyFocusBtn', 'click', function() { focusText(document.getElementById('focusTextInput').value); });
    listen('clearFocusBtn', 'click', function() {
        document.getElementById('focusTextInput').value = '';
        focusText('');
    });
    listen('focusTextInput', 'keydown', function(e) { if (e.key === 'Enter') focusText(e.target.value); });
    listen('focusSimilarityModeSelect', 'change', function(e) {
        focusSimilarityMode = e.target.value;
        refreshFocusMarker();
        refreshCorrFocus();
        updateTextList();
    });
    listen('focusClusterOnlyCheckbox', 'change', function(e) {
        focusClusterOnly = e.target.checked;
        refreshFocusMarker();
    });
    listen('focusClusterMetricSelect', 'change', function(e) {
        focusClusterMetricUserSelected = true;
        focusClusterMetric = e.target.value;
        refreshFocusMarker();
        if (viewMode === 'functionWord') updateFunctionWordPlot();
    });
    listen('semanticLinesCheckbox', 'change', function(e) {
        semanticLineModeEnabled = e.target.checked;
        refreshFocusMarker();
        refreshCorrFocus();
        updateTextList();
    });
    listen('semanticLinesSlider', 'input', function(e) {
        semanticLineCount = Math.max(1, Math.min(50, Number(e.target.value) || 1));
        document.getElementById('semanticLinesValue').textContent = semanticLineCount;
        if (semanticLineModeEnabled) {
            refreshConnections();
            refreshCorrConnections();
            updateTextList();
        }
    });
}

function wireViews() {
    listen('toggleViewBtn', 'click', function() {
        if (viewMode !== 'umap') setView('umap');
        else setView(mapMode === 'burrows' ? 'functionWord' : 'correlation');
    });
    listen('experimentalBtn', 'click', function() { setMapMode(mapMode === 'burrows' ? 'semantic' : 'burrows'); });
    listen('deltaExplorerBtn', 'click', function() { setView('delta'); });
}

function wireCorr() {
    populateCorrDropdowns();
    listen('updateCorrelationBtn', 'click', updateCorrPlot);
    listen('regressionType', 'change', updateCorrPlot);
    listen('xMetricSearchInput', 'input', populateCorrDropdowns);
    listen('yMetricSearchInput', 'input', populateCorrDropdowns);
    listen('toggleCorrelationLabelsBtn', 'click', function(e) {
        correlationLabelsVisible = !correlationLabelsVisible;
        Plotly.restyle(document.getElementById('correlationPlot'),
            { mode: correlationLabelsVisible ? 'markers+text' : 'markers' }, [0]).then(function() {
                if (correlationLabelsVisible) addOutlines('#correlationPlot .scatterlayer .textpoint text', 'correlationStrokeText');
            });
        e.target.textContent = correlationLabelsVisible ? 'Hide Labels' : 'Show Labels';
    });
}

function wireWords() {
    if (hasBurrowsData()) populateWordControls();
    var sel = document.getElementById('functionWordSelect');
    listen('functionWordSearchInput', 'input', populateWordControls);
    sel.addEventListener('change', function() {
        for (var i = 0; i < sel.options.length; i++) {
            var o = sel.options[i];
            if (!o.value) continue;
            if (o.selected) functionWordSelection.add(o.value);
            else functionWordSelection.delete(o.value);
        }
        updateWordSummary();
        updateFunctionWordPlot();
    });
    listen('functionWordSelectShownBtn', 'click', function() {
        for (var i = 0; i < sel.options.length; i++) {
            var o = sel.options[i];
            if (!o.value) continue;
            o.selected = true;
            functionWordSelection.add(o.value);
        }
        updateWordSummary();
        updateFunctionWordPlot();
    });
    listen('functionWordClearBtn', 'click', function() {
        functionWordSelection.clear();
        for (var i = 0; i < sel.options.length; i++) sel.options[i].selected = false;
        updateWordSummary();
        updateFunctionWordPlot();
    });
    listen('functionWordPlotModeSelect', 'change', function(e) {
        functionWordPlotMode = e.target.value;
        updateFunctionWordPlot();
    });
    listen('functionWordShowHistogram', 'change', function(e) {
        functionWordShowHistogram = e.target.checked;
        updateFunctionWordPlot();
    });
    listen('functionWordShowMarkers', 'change', function(e) {
        functionWordShowMarkers = e.target.checked;
        updateFunctionWordPlot();
    });
    listen('updateFunctionWordBtn', 'click', updateFunctionWordPlot);
}

function wireDeltaPicker(searchId, selectId, stateKey) {
    var si = document.getElementById(searchId);
    var sel = document.getElementById(selectId);
    si.addEventListener('input', function() {
        var n = fillDeltaSelect(sel, si.value, deltaState[stateKey]);
        if (n !== null && n !== deltaState[stateKey]) {
            deltaState[stateKey] = n;
            updateDeltaPlot();
        }
    });
    sel.addEventListener('change', function() {
        deltaState[stateKey] = Number(sel.value);
        updateDeltaPlot();
    });
}

function wireDelta() {
    wireDeltaPicker('deltaTextASearch', 'deltaTextASelect', 'textA');
    wireDeltaPicker('deltaTextBSearch', 'deltaTextBSelect', 'textB');
    listen('deltaSwapBtn', 'click', function() {
        var t = deltaState.textA;
        deltaState.textA = deltaState.textB;
        deltaState.textB = t;
        syncDeltaDropdowns();
        updateDeltaPlot();
    });
    listen('deltaUseFocusedBtn', 'click', function() {
        if (focusedTextIndex < 0) { alert('No text is focused.'); return; }
        deltaState.textA = focusedTextIndex;
        if (deltaState.textB === focusedTextIndex) deltaState.textB = defaultDeltaPartner(focusedTextIndex);
        syncDeltaDropdowns();
        updateDeltaPlot();
    });
    listen('deltaUseNearestBtn', 'click', function() {
        if (deltaState.textA === null) return;
        deltaState.textB = defaultDeltaPartner(deltaState.textA);
        syncDeltaDropdowns();
        updateDeltaPlot();
    });
    listen('deltaSortSelect', 'change', function(e) {
        deltaState.sortMode = e.target.value;
        updateDeltaPlot();
    });
    listen('deltaTopNRange', 'input', function(e) {
        var max = deltaState.totalWordCount || 80;
        setDeltaTopN(Math.max(5, Math.min(max, Number(e.target.value) || 25)));
        updateDeltaPlot();
    });
    listen('deltaShowAllBtn', 'click', function() {
        var total = deltaState.totalWordCount || 80;
        document.getElementById('deltaTopNRange').max = String(total);
        setDeltaTopN(total);
        updateDeltaPlot();
    });
    listen('deltaUpdateBtn', 'click', function() {
        recomputeDelta();
        updateDeltaPlot();
    });
}

function buildClusterMetricKeys() {
    for (var i = 0; i < metricKeys.length; i++) {
        if (isClusterKey(metricKeys[i])) clusterMetricKeys.push(metricKeys[i]);
    }
}

function buildTextLookup() {
    for (var ti = 0; ti < textNames.length; ti++) {
        textLookup[textNames[ti].toLowerCase()] = { name: textNames[ti], index: ti };
    }
}

function buildMetricCatalog() {
    for (var mi = 0; mi < metricKeys.length; mi++) {
        var mk = metricKeys[mi];
        if (mk in metricsData && metricsData[mk] !== null) {
            var lbl = isClusterKey(mk) ? clusterLabel(mk) : (metricDisplayNames[mk] || mk);
            metricCatalog.push({ key: mk, label: lbl, search: (mk + ' ' + lbl).toLowerCase() });
        }
    }
}

function initPage() {
    mainPlot = document.querySelector('.plotly-graph-div');
    if (!mainPlot) return;

    buildClusterMetricKeys();
    buildTextLookup();
    buildMetricCatalog();

    try { sanitizeMainHoverText(); } catch (e) { console.error(e); }

    createMainPanels();
    createSecondaryViews();

    try {
        if (typeof mainPlot.on === 'function') mainPlot.on('plotly_afterplot', function() { addOutlines(); });
        setupHoverStyling();
    } catch (e) { console.error(e); }
    try { snapshotXY(); } catch (e) { console.error(e); }
    try { snapshotAxes(); } catch (e) { console.error(e); }
    try { fitPlot(); } catch (e) { console.error(e); }

    setTimeout(function() { addOutlines(); }, CONFIG.textOutlineDelay);

    var metricSel = document.getElementById('metricSelect');
    fillMetricSelect(metricSel, currentMetric, '');
    if (metricSel.value) currentMetric = metricSel.value;

    focusClusterMetric = fillClusterSelect(document.getElementById('focusClusterMetricSelect'), preferredCluster('semantic'));
    document.getElementById('focusTextInput').value = CONFIG.defaultFocusText || '';

    syncColors();
    readColorState();

    if (isBurrows(currentMetric)) {
        var filled = colorData(currentMetric);
        if (filled) {
            currentColorDataForPlot = filled;
            Plotly.restyle(mainPlot, { 'marker.color': [filled] }, [0]);
        }
    }

    var isCluster = isClusterKey(currentMetric);
    var initCb = {
        'marker.colorbar.title': colorbarTitle(currentMetric),
        'marker.colorbar.tickvals': [null],
        'marker.colorbar.ticktext': [null]
    };
    if (isCluster) {
        initCb['marker.colorbar.showticklabels'] = [false];
    }
    Plotly.restyle(mainPlot, initCb, [0]);

    try { sanitizeMainHoverText(); } catch (e) { console.error(e); }

    wireMetricAndLabelControls();
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
        updatePercentileLabel();
    }
}

document.addEventListener('DOMContentLoaded', initPage);

