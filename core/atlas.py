#NEW ATLAS.py
import argparse
import csv
import gc
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').setLevel(logging.ERROR)
logging.getLogger('torch.utils._pytree').setLevel(logging.ERROR)

import torch
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import umap

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:
    SklearnHDBSCAN = None

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from lexicalrichness import LexicalRichness

try:
    import enchant
    ENCHANT_AVAILABLE = True
    enchant_dict = None
    enchant_word_check_cache = {}
except ImportError:
    enchant = None
    ENCHANT_AVAILABLE = False
    enchant_dict = None
    enchant_word_check_cache = {}

from plotly import graph_objects as go

try:
    from atlas_gameologist import GameologistExporter
    GAMEOLOGIST_AVAILABLE = True
except ImportError:
    GameologistExporter = None
    GAMEOLOGIST_AVAILABLE = False

from helpers.atlas_config import *
from helpers.atlas_config import (
    _format_trimmed_decimal,
    _is_word_count_metric_config,
)
from helpers.atlas_cache import SimpleCache
from metricsmixin.atlas_metrics import TextMetricsMixin
from helpers.atlas_processor import TextProcessor
import helpers.atlas_config as cfg

try:
    import igraph as ig
    import leidenalg as la
    LEIDEN_AVAILABLE = True
except ImportError:
    ig = None
    la = None
    LEIDEN_AVAILABLE = False

_WORD_RUN = re.compile(r"\w+")
_SEQ = r"\w+(?: \w+)*"
_PURE_TERMS_RE = re.compile(rf"^\\b(?:\(\?:)?({_SEQ}(?:\|{_SEQ})*)\)?\\b$")
_pure_terms_cache: Dict[Any, Optional[Dict[int, frozenset]]] = {}
_HEX_COLOR_RE = re.compile(r'^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$')

_HOVER_EXCLUDED = {
    'latinate_word_ratio', 'coordinate_clause_ratio', 'subordinate_clause_ratio',
    'imperative_exclamation_count', 'academic_word_density',
    'article_count_raw', 'conjunction_count_raw', 'vulnerability',
    'family_exclamations', 'pronoun_article_ratio',
    'deictic_spatial_temporal', 'elaboration_explanation_ratio',
    'narration_continuation_ratio',
    'insult', 'bodily_humor', 'dis_ratio',
    'word_burstiness', 'semantic_disparity', 'guppa', 'pronoun_switching',
    'compression', 'moving_lzma_cr', 'normalized_compression_ratio',
    'window_normalized_lzma_cr',
    'vader_positivity', 'vader_volatility',
    'vader_positivity_centroid', 'vader_volatility_centroid',
    'fighting_words_peak_z1', 'fighting_words_floor_z1000',
    'fighting_words_peak_z1_centroid', 'fighting_words_floor_z1000_centroid',
}


def _ambiguous(alts) -> bool:
    alts = list(alts)
    for u in alts:
        for v in alts:
            lu, lv = len(u), len(v)
            for s in range(1, lu):
                if lu - s <= lv and u[s:] == v[:lu - s]:
                    return True
            if u != v and lu < lv and any(v[i:i + lu] == u for i in range(lv - lu + 1)):
                return True
    return False


def _pure_terms(pattern):
    try:
        return _pure_terms_cache[pattern]
    except KeyError:
        pass

    terms = None
    src = getattr(pattern, 'pattern', None)

    if isinstance(src, str) and not (pattern.flags & re.VERBOSE):
        m = _PURE_TERMS_RE.match(src)
        if m:
            has_open_wrapper = src.startswith('\\b(?:')
            has_close_wrapper = src.endswith(')\\b')
            wrapper_state_matches = (has_open_wrapper == has_close_wrapper)

            if wrapper_state_matches:
                lower = bool(pattern.flags & re.IGNORECASE)
                alts = {
                    tuple(w.lower() if lower else w for w in a.split(' '))
                    for a in m.group(1).split('|')
                }
                if not _ambiguous(alts):
                    by_k: Dict[int, set] = {}
                    for a in alts:
                        by_k.setdefault(len(a), set()).add(a)
                    terms = {k: frozenset(v) for k, v in by_k.items()}

    _pure_terms_cache[pattern] = terms
    return terms


def _round_list(arr, decimals: int = 5):
    return np.round(np.asarray(arr, dtype=np.float64), decimals).tolist()


class TextClassifier(TextMetricsMixin):

    CROSS_CHANNEL_METRICS = {'burrows_cosine_disagreement'}

    _CORPUS_METRICS = CROSS_CHANNEL_METRICS | {
        FIGHTING_WORDS_PEAK_METRIC,
        FIGHTING_WORDS_FLOOR_METRIC,
        FIGHTING_WORDS_PEAK_CENTROID_METRIC,
        FIGHTING_WORDS_FLOOR_CENTROID_METRIC,
    }

    BURROWS_FUNCTION_WORDS = tuple(sorted({
        'i', 'you', 'he', 'it', 'we', 'they',
        'me', 'him', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
        'myself', 'yourself', 'himself', 'themselves',
        'be', 'is', 'am', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing',
        'will', 'would', 'should', 'can', 'could', 'may', 'might', 'must',
        'to', 'of', 'and', 'a', 'in', 'that', 'for', 'not', 'on', 'with', 'as', 'at',
        'but', 'by', 'from', 'or', 'an', 'if', 'about',
        'after', 'because', 'any', 'than', 'into', 'through', 'before',
        'this', 'these', 'those', 'there', 'what', 'so', 'who', 'which', 'when',
        'no', 'only', 'other', 'some', 'then', 'now', 'look', 'come', 'go', 'get',
        'take', 'see', 'think', 'also', 'use', 'how', 'even', 'want', 'give',
        'most', 'each', 'every', 'both', 'few', 'more', 'such', 'own', 'same', 'too',
        'very', 'just', 'here', 'why', 'once', 'the',
        'oh', 'yeah', 'yes', 'okay', 'like', 'well', 'uh', 'um', 'right',
        'mhm', 'next', 'again',
    }))

    BURROWS_CONTRACTIONS = {
        "i'm": "i am", "i've": "i have", "i'd": "i would", "i'll": "i will",
        "you're": "you are", "you've": "you have", "you'd": "you would", "you'll": "you will",
        "he's": "he is", "he'd": "he would", "he'll": "he will",
        "she's": "he is", "he'd": "he would", "he'll": "he will",
        "it's": "it is", "it'd": "it would", "it'll": "it will",
        "we're": "we are", "we've": "we have", "we'd": "we would", "we'll": "we will",
        "they're": "they are", "they've": "they have", "they'd": "they would", "they'll": "they will",
        "that's": "that is", "that'd": "that would", "that'll": "that will",
        "there's": "there is", "there'd": "there would", "there'll": "there will",
        "here's": "here is", "what's": "what is", "who's": "who is", "how's": "how is",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "can not", "couldn't": "could not", "won't": "will not", "wouldn't": "would not",
        "shouldn't": "should not", "mustn't": "must not", "let's": "let us",
        "y'all": "you all",
        "her": "him",
        "she": "he",
        "herself": "himself",
        "okay": "ok",
    }

    @classmethod
    def get_word_count_metric_names(cls) -> List[str]:
        selected = set()
        for metric_name, config in MetricConfig.METRICS.items():
            compute_method = config.get('compute_method', '')
            if metric_name.endswith('_count'):
                selected.add(metric_name)
                continue
            if compute_method.startswith('_compute'):
                method = getattr(cls, compute_method, None)
                if method is None:
                    continue
                try:
                    src = inspect.getsource(method)
                except Exception:
                    src = ''
                if 'PATTERNS' in src or 're.' in src:
                    selected.add(metric_name)
        return sorted(selected)

    def _target_metrics(
        self,
        include_global_metrics: bool = False,
        skip_centroid_duplicates: bool = False,
        skip_video_centroid_only: bool = False,
    ) -> List[str]:
        if getattr(self, 'no_metrics', False):
            return []

        if self.word_counts_only:
            names = self.get_word_count_metric_names()
        else:
            names = MetricConfig.get_metric_names()

        fw = (
            FIGHTING_WORDS_PEAK_METRIC,
            FIGHTING_WORDS_FLOOR_METRIC,
            FIGHTING_WORDS_PEAK_CENTROID_METRIC,
            FIGHTING_WORDS_FLOOR_CENTROID_METRIC,
        )
        if not (include_global_metrics and getattr(self, 'compute_fighting_words', False)):
            names = [n for n in names if n not in fw]

        names = [n for n in names if n not in self.CROSS_CHANNEL_METRICS]

        if not self.processor.external_documents:
            names = [
                n for n in names
                if not MetricConfig.METRICS.get(n, {}).get('sentence_metric', False)
            ]

        if skip_centroid_duplicates:
            names = [n for n in names if not MetricConfig.is_centroid_duplicate(n)]
        if skip_video_centroid_only:
            names = [n for n in names if not MetricConfig.is_video_centroid_only(n)]

        return names

    @classmethod
    def _burrows_profile(cls, tokens: List[str]) -> np.ndarray:
        profile = np.zeros(len(cls.BURROWS_FUNCTION_WORDS), dtype=np.float64)
        if not tokens:
            return profile

        word_indices = {word: i for i, word in enumerate(cls.BURROWS_FUNCTION_WORDS)}

        expanded = []
        for token in tokens:
            expanded.extend(cls.BURROWS_CONTRACTIONS.get(token, token).split())

        for token in expanded:
            idx = word_indices.get(token)
            if idx is not None:
                profile[idx] += 1.0

        return profile / len(expanded)

    @classmethod
    def _burrows_dist(
        cls,
        profiles: Dict[str, np.ndarray],
        texts: List[str],
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        valid = [i for i, text in enumerate(texts) if text in profiles]
        if len(valid) < 2:
            return None, []

        mat = np.vstack([profiles[texts[i]] for i in valid])
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=0)
        sd[sd == 0] = 1e-12
        z = (mat - mu) / sd

        return cdist(z, z, 'cityblock') / z.shape[1], valid

    @classmethod
    def _burrows_scores(
        cls,
        dist: Optional[np.ndarray],
        valid: List[int],
        emb: np.ndarray,
        texts: List[str],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {text: float('nan') for text in texts}
        if dist is None or len(valid) < 3:
            return scores

        norm = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sem_dist = 1.0 - norm @ norm.T

        for row, ti in enumerate(valid):
            peers = [p for p in range(len(valid)) if p != row]
            if len(peers) < 2:
                continue
            sr = rankdata(dist[row, peers], method='average')
            mr = rankdata(
                sem_dist[ti, [valid[p] for p in peers]],
                method='average',
            )
            sc = sr - sr.mean()
            mc = mr - mr.mean()
            denom = np.linalg.norm(sc) * np.linalg.norm(mc)
            rho = float(sc @ mc / denom) if denom else 0.0
            scores[texts[ti]] = float(1.0 - rho)

        return scores

    @classmethod
    def _semantic_capitals(
        cls,
        emb: np.ndarray,
        labels: List[Optional[int]],
        texts: List[str],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"world": None, "byId": {}}
        if not texts or emb.shape[0] == 0:
            return out

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = emb / norms

        gc = emb.mean(axis=0)
        gn = float(np.linalg.norm(gc))
        if gn > 0:
            gc = gc / gn
        out["world"] = texts[int(np.argmax(normalized @ gc))]

        members_by_id: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            if lab is None:
                continue
            try:
                cid = int(lab)
            except (TypeError, ValueError):
                continue
            members_by_id.setdefault(cid, []).append(i)

        for cid, members in members_by_id.items():
            if not members:
                continue
            sub = emb[members]
            centroid = sub.mean(axis=0)
            cn = float(np.linalg.norm(centroid))
            if cn > 0:
                centroid = centroid / cn
            sub_n = sub / norms[members]
            best = int(np.argmax(sub_n @ centroid))
            out["byId"][str(cid)] = texts[members[best]]

        return out

    @classmethod
    def _burrows_capitals(
        cls,
        dist: Optional[np.ndarray],
        valid: List[int],
        labels: List[Optional[int]],
        texts: List[str],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"world": None, "byId": {}}
        if dist is None or not valid:
            return out

        def row_means(sub: np.ndarray) -> np.ndarray:
            finite = np.isfinite(sub)
            sums = np.where(finite, sub, 0.0).sum(axis=1)
            counts = finite.sum(axis=1)
            return np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)

        out["world"] = texts[valid[int(np.argmin(row_means(dist)))]]

        cluster_pos: Dict[int, List[int]] = {}
        for pos, ti in enumerate(valid):
            lab = labels[ti]
            if lab is None:
                continue
            try:
                cid = int(lab)
            except (TypeError, ValueError):
                continue
            cluster_pos.setdefault(cid, []).append(pos)

        for cid, positions in cluster_pos.items():
            if not positions:
                continue
            sub = dist[np.ix_(positions, positions)]
            best = int(np.argmin(row_means(sub)))
            out["byId"][str(cid)] = texts[valid[positions[best]]]

        return out

    def _load_burrows_profile(
        self, text: str, filter_kwargs: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        filters = {k: v for k, v in filter_kwargs.items() if v is not None}
        ignore_ids = filter_kwargs.get('ignore_ids', set())

        if self.centroid_mode == 'video':
            files = self.processor._get_filtered_files(text, filters, ignore_ids)
            if not files:
                return None
            result = self.processor.load_video_centroids(
                text, self.centroid_videos, filters, ignore_ids,
                max_accumulated_tokens=self.stat_word_count,
                precomputed_files=files,
                need_embeddings=False,
                need_stat_corpus=True,
                need_centroid_tokens=True,
            )
            if not result:
                return None
            stat_tokens, centroid_tokens = result[2], result[3]
            tokens = stat_tokens if stat_tokens else centroid_tokens
            return self._burrows_profile(tokens) if tokens else None

        result = self.processor.get_tokens(text, filters, ignore_ids)
        if not result:
            return None
        tokens = result[0]
        if self.centroid_mode == 'word' and self.centroid_words > 0:
            tokens = tokens[-self.centroid_words:]
        return self._burrows_profile(tokens)

    @classmethod
    def _expand_burrows_dist(
        cls,
        n: int,
        distances: np.ndarray,
        valid: List[int],
    ) -> np.ndarray:
        full = np.full((n, n), np.nan, dtype=np.float64)
        for r, row in enumerate(valid):
            for c, col in enumerate(valid):
                full[row, col] = distances[r, c]
        return full

    @classmethod
    def _expand_burrows_distance_matrix(
        cls,
        n: int,
        distances: np.ndarray,
        valid: List[int],
    ) -> np.ndarray:
        full = np.full((n, n), np.nan, dtype=np.float64)
        for r_idx, row in enumerate(valid):
            for c_idx, col in enumerate(valid):
                full[row, col] = distances[r_idx, c_idx]

        finite_mask = np.isfinite(full)
        if finite_mask.any():
            fill_value = float(np.nanmax(full)) * 1.2
        else:
            fill_value = 1.0

        full[np.isnan(full)] = fill_value
        np.fill_diagonal(full, 0.0)

        full = (full + full.T) * 0.5
        np.fill_diagonal(full, 0.0)

        return full.astype(np.float32)

    def _burrows_reduce_dimensions(
        self,
        distances: np.ndarray,
        n: int,
    ) -> np.ndarray:
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)

        if n == 1:
            return np.zeros((1, 2), dtype=np.float32)

        if n == 2:
            d = float(distances[0, 1])
            return np.array([[0.0, 0.0], [d, 0.0]], dtype=np.float32)

        n_neighbors = min(5 if n <= 30 else 50, n - 1)
        if self.umap_neighbors is not None:
            n_neighbors = min(self.umap_neighbors, n - 1)
        n_neighbors = max(2, n_neighbors)

        reducer = umap.UMAP(
            n_components=2,
            metric='precomputed',
            n_neighbors=n_neighbors,
            min_dist=self.umap_min_dist,
            init='spectral',
            random_state=20,
            n_jobs=1,
            verbose=False,
            n_epochs=self.umap_epochs,
            low_memory=False,
        )
        return reducer.fit_transform(distances)

    def _metric_is_eligible(self, name: str) -> bool:
        return MetricConfig.is_metric_eligible(
            name,
            enable_spacy=self.enable_spacy,
            needs_perplexity_model=self.processor.needs_perplexity_model,
            enable_ngram_entropy=self.enable_ngram_entropy,
            fast_mode=self.fast_mode,
        )

    def _skipped(self, name: str) -> bool:
        return not self._metric_is_eligible(name)

    def _metric_needs_compute(self, name: str, cached_metrics: Dict[str, Any]) -> bool:
        if name not in cached_metrics:
            return self._metric_is_eligible(name)
        if cached_metrics[name] is not None:
            return False
        return self._metric_is_eligible(name)

    @staticmethod
    def _summarize(names: List[str], max_items: Optional[int] = 16) -> str:
        if not names:
            return "none"
        uniq = sorted(set(names))
        if max_items is None or max_items <= 0 or len(uniq) <= max_items:
            return ', '.join(uniq)
        return f"{', '.join(uniq[:max_items])} ... (+{len(uniq) - max_items} more)"

    def _print_summary(self, values: Dict[str, Any], source: str) -> None:
        if not values:
            return

        done = sorted(n for n, v in values.items() if v is not None)
        skipped = sorted(n for n, v in values.items() if v is None)

        print(f"    Metrics finished: {len(done)}/{len(values)} [{source}]")

        embed_m = [n for n in done if 'embedding' in n]
        wc_m = [
            n for n in done
            if _is_word_count_metric_config(n, MetricConfig.METRICS.get(n, {}))
        ]
        other = [n for n in done if n not in set(wc_m) | set(embed_m)]

        if wc_m:
            print(f"      • Word counts: {len(wc_m)}")
        if embed_m:
            print(f"      • Embedding metrics: {len(embed_m)}")
        if other:
            print(f"      • Other metrics: {len(other)}")

        category_buckets: Dict[str, List[str]] = defaultdict(list)
        for metric_name in done:
            category = MetricConfig.METRICS.get(metric_name, {}).get(
                'category', 'Uncategorized'
            )
            category_buckets[str(category)].append(metric_name)

        for category_name in sorted(category_buckets):
            names_in_category = category_buckets[category_name]
            print(f"      • {category_name}: {len(names_in_category)}")

        if skipped:
            print(f"      • Skipped/unavailable: {len(skipped)}")

    @staticmethod
    def _embed_device() -> str:
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        batch = max(1, int(self.embedding_batch_size))
        while True:
            try:
                return self.model.encode(
                    texts,
                    batch_size=batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            except RuntimeError as e:
                msg = str(e)
                oom = (
                    'Invalid buffer size' in msg
                    or 'out of memory' in msg.lower()
                    or 'CUDA out of memory' in msg
                )
                if not oom or batch <= 1:
                    raise
                batch = max(1, batch // 2)
                self.embedding_batch_size = batch
                print(
                    f"    Warning: embedding encode memory pressure, "
                    f"retrying with batch_size={batch}"
                )

    def __init__(
        self,
        use_cache: bool = True,
        active_metric: Optional[str] = None,
        enable_spacy: bool = False,
        enable_ngram_entropy: bool = False,
        fast_mode: bool = False,
        word_counts_only: bool = False,
        force_latest_cache: bool = False,
        embedding_model: Optional[str] = None,
        embedding_chunk_size: Optional[int] = None,
        embedding_min_chunk: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
        no_metrics: bool = False,
        compute_fighting_words: bool = False,
        compute_fighting_words_floor: Optional[bool] = None,
        centroid_mode: Optional[str] = None,
        centroid_videos: int = 10,
        centroid_words: int = 1000000,
        stat_word_count: int = 1000000,
        cluster_method: str = 'hdbscan',
        leiden_resolution_min: float = 0.3,
        leiden_resolution_max: float = 2.0,
        leiden_k_neighbors: int = 20,
        leiden_whiten: bool = True,
        umap_neighbors: Optional[int] = None,
        umap_min_dist: float = 0.1,
        umap_epochs: int = 200,
        tomotopy_num_topics: int = 0,
        tomotopy_iterations: int = 1000,
        tomotopy_threads: Optional[int] = None,
        tomotopy_max_tokens_per_channel: Optional[int] = None,
        change_fword_to_stupid: bool = False,
    ):
        self.embedding_model_name = resolve_embedding_model_name(embedding_model)
        if huggingface_local_files_only(self.embedding_model_name):
            os.environ['HF_HUB_OFFLINE'] = '1'
        try:
            self.model = SentenceTransformer(
                self.embedding_model_name,
                local_files_only=huggingface_local_files_only(
                    self.embedding_model_name
                ),
            )
        except OSError as exc:
            offline = huggingface_local_files_only(self.embedding_model_name)
            mode = (
                'The model is not available in the local Hugging Face cache.'
                if offline
                else 'Hugging Face could not download the model.'
            )
            raise RuntimeError(
                f'Unable to load embedding model '
                f'"{self.embedding_model_name}". {mode} '
                'Set ATLAS_HF_LOCAL_FILES_ONLY=0 with internet access to '
                'download it, or pre-cache the model before running Centroids.'
            ) from exc
        profile = get_embedding_model_profile(self.embedding_model_name)
        self.embedding_device = self._embed_device()

        detected_ctx = int(getattr(self.model, 'max_seq_length', 0) or 0)
        self.embedding_context_window = (
            detected_ctx if detected_ctx > 0 else profile['context_window']
        )

        practical = max(16, int(profile.get(
            f"practical_max_tokens_{self.embedding_device}",
            profile['chunk_target'],
        )))

        if embedding_chunk_size is None:
            chunk = profile['chunk_target']
            if self.embedding_context_window > 0:
                chunk = min(chunk, max(16, self.embedding_context_window - 8))
            chunk = min(int(chunk), practical)
        else:
            chunk = int(embedding_chunk_size)
            if self.embedding_context_window > 0:
                chunk = min(chunk, max(16, self.embedding_context_window - 8))

        if embedding_min_chunk is None:
            embedding_min_chunk = profile['min_chunk']
        if chunk is not None and embedding_min_chunk is not None:
            embedding_min_chunk = min(int(embedding_min_chunk), int(chunk))

        self.embedding_batch_size = int(embedding_batch_size or profile['batch_size'])
        if self.embedding_device == 'cpu':
            self.embedding_batch_size = min(self.embedding_batch_size, 8)
        elif self.embedding_device == 'mps':
            self.embedding_batch_size = min(self.embedding_batch_size, 16)

        self.change_fword_to_stupid = bool(change_fword_to_stupid)

        self.embedding_cache_signature: Dict[str, Any] = {
            'embedding_model': self.embedding_model_name,
            'embedding_context_window': int(self.embedding_context_window),
            'embedding_chunk_size': int(chunk),
            'embedding_min_chunk': int(embedding_min_chunk),
            'change_fword_to_stupid': self.change_fword_to_stupid,
        }

        print(
            f"Embedding model: {self.embedding_model_name} "
            f"| device={self.embedding_device} "
            f"| context={self.embedding_context_window} "
            f"| chunk={int(chunk)} "
            f"| min_chunk={int(embedding_min_chunk)} "
            f"| batch={self.embedding_batch_size} "
            f"| fword->stupid={self.change_fword_to_stupid}"
        )

        self.base_dir = Path(__file__).resolve().parents[1]
        self.cache = SimpleCache(self.base_dir / "cache" / "metrics") if use_cache else None
        self.active_metric = active_metric
        self.enable_spacy = enable_spacy
        self.enable_ngram_entropy = enable_ngram_entropy
        self.fast_mode = fast_mode
        self.word_counts_only = word_counts_only
        self.no_metrics = no_metrics
        self.force_latest_cache = force_latest_cache
        self.compute_fighting_words = (
            compute_fighting_words
            if compute_fighting_words_floor is None
            else compute_fighting_words_floor
        )

        self._fighting_words_unigrams: Dict[str, Counter] = {}
        self._fighting_words_token_counts: Dict[str, int] = {}
        self._fighting_words_unigrams_centroid: Dict[str, Counter] = {}
        self._fighting_words_token_counts_centroid: Dict[str, int] = {}
        self._fighting_words_peak_unigrams: Dict[str, str] = {}
        self._fighting_words_floor_unigrams: Dict[str, str] = {}
        self._fighting_words_peak_unigrams_centroid: Dict[str, str] = {}
        self._fighting_words_floor_unigrams_centroid: Dict[str, str] = {}
        self._centroid_equals_stat: Dict[str, bool] = {}
        self._current_text_name: Optional[str] = None
        self._current_video_count: int = 0

        self.centroid_mode = centroid_mode
        self.centroid_videos = centroid_videos
        self.centroid_words = centroid_words
        self.stat_word_count = stat_word_count

        self.tomotopy_num_topics: int = int(tomotopy_num_topics or 0)
        self.tomotopy_iterations: int = max(1, int(tomotopy_iterations or 1000))
        self.tomotopy_threads: int = max(1, int(tomotopy_threads or min(8, os.cpu_count() or 1)))
        if tomotopy_max_tokens_per_channel and int(tomotopy_max_tokens_per_channel) > 0:
            self.tomotopy_max_tokens_per_channel: Optional[int] = int(tomotopy_max_tokens_per_channel)
        else:
            self.tomotopy_max_tokens_per_channel = None
        self.tomotopy_topic_proportions: Dict[str, Dict[int, float]] = {}
        self.tomotopy_topic_labels: Dict[int, str] = {}

        self.cluster_method = (cluster_method or 'hdbscan').lower()
        self.leiden_resolution_min = float(leiden_resolution_min)
        self.leiden_resolution_max = float(leiden_resolution_max)
        self.leiden_k_neighbors = int(leiden_k_neighbors)
        self.leiden_whiten = bool(leiden_whiten)
        self.umap_neighbors = None if umap_neighbors is None else int(umap_neighbors)
        self.umap_min_dist = float(umap_min_dist)
        self.umap_epochs = int(umap_epochs)
        self.cluster_metric_labels: Dict[str, str] = {}
        self.burrows_cluster_metric_labels: Dict[str, str] = {}

        needs_perplexity = (
            active_metric and MetricConfig.requires_perplexity_model(active_metric)
        )

        self.processor = TextProcessor(
            self.base_dir,
            chunk_size=int(chunk),
            min_chunk=int(embedding_min_chunk),
            needs_perplexity_model=needs_perplexity,
            enable_spacy=enable_spacy,
            model_tokenizer=getattr(self.model, 'tokenizer', None),
            model_context_window=self.embedding_context_window,
            change_fword_to_stupid=self.change_fword_to_stupid,
        )

        self.processor._model = self.model
        self.processor.embedding_batch_size = self.embedding_batch_size
        self.processor.embedding_model_name_for_cache = self.embedding_model_name
        self.processor._encode_fn = self._encode

        self.embeddings: Dict[str, np.ndarray] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
        self.burrows_similarity_matrix: Optional[np.ndarray] = None
        self.cluster_variant_labels: Dict[str, List[Optional[int]]] = {}
        self.burrows_cluster_variant_labels: Dict[str, List[Optional[int]]] = {}
        self._scaler_cache = None
        self._video_centroid_cache: Dict[str, Tuple[List[str], List[np.ndarray], List[str]]] = {}

        self._runtime_cache_tokens_ref: Optional[List[str]] = None
        self._runtime_cache_len: int = -1
        self._runtime_cache_text: Optional[str] = None
        self._runtime_cache_text_lower: Optional[str] = None
        self._runtime_cache_text_bytes: Optional[bytes] = None
        self._runtime_cache_counter: Optional[Counter] = None
        self._runtime_cache_pattern_counts: Dict[Any, int] = {}
        self._runtime_cache_pattern_prefixes: Dict[int, Optional[str]] = {}
        self._runtime_cache_dynamic_counts: Dict[str, float] = {}
        self._runtime_cache_spacy_stats: Optional[Dict[str, int]] = None
        self._runtime_cache_oov_split: Optional[Tuple[float, float, float, float]] = None
        self._runtime_cache_lzma_full: Optional[Tuple[int, int]] = None
        self._runtime_cache_lzma_windows: Optional[List[Tuple[int, int, int]]] = None
        self._runtime_cache_pieces: Optional[Counter] = None
        self._runtime_cache_edges: Optional[Tuple] = None

        self.validate_metric_wiring()

    def _reset_fw(self) -> None:
        for attr in (
            '_fighting_words_peak_unigrams',
            '_fighting_words_floor_unigrams',
            '_fighting_words_peak_unigrams_centroid',
            '_fighting_words_floor_unigrams_centroid',
        ):
            setattr(self, attr, {})
        self._fw_peak_cache: Dict[str, Optional[float]] = {}
        self._fw_floor_cache: Dict[str, Optional[float]] = {}
        self._fw_peak_centroid_cache: Dict[str, Optional[float]] = {}
        self._fw_floor_centroid_cache: Dict[str, Optional[float]] = {}

    @classmethod
    def validate_metric_wiring(cls) -> None:
        bad = []
        for name, cfg_ in MetricConfig.METRICS.items():
            if name in cls._CORPUS_METRICS:
                continue
            m = cfg_.get('compute_method', '')
            if m == '_compute_dynamic_token_count':
                fb = DYNAMIC_COUNT_FALLBACK_METHODS.get(name)
                ok = (
                    name in SIMPLE_COLOR_PATTERN_MAP
                    or name in DYNAMIC_PATTERN_METRIC_MAP
                    or name in DYNAMIC_TOKEN_SET_METRIC_MAP
                    or name in (
                        'right_count', 'unique_words_count', 'unique_oov_count',
                        'total_oov_count', 'unique_non_oov_count', 'total_non_oov_count',
                    )
                    or (fb and fb != '_compute_dynamic_token_count' and hasattr(cls, fb))
                )
            elif m.startswith('_compute'):
                ok = hasattr(cls, m)
            else:
                ok = m in ('calculate_perplexity', 'compute_vader_sentiment')
            if not ok:
                bad.append(name)
        if bad:
            print(
                f"WARNING: {len(bad)} metrics have no resolvable compute path: "
                f"{sorted(bad)}"
            )

    def _compute_avg_words_per_video(self, tokens: List[str]) -> Optional[float]:
        n = self._current_video_count
        if not tokens or n <= 0:
            return None
        return len(tokens) / n

    @staticmethod
    def _compute_burrows_cosine_disagreement(tokens: List[str]) -> None:
        return None

    def _precompute_fw(
        self,
        texts: List[str],
        filter_kwargs: Dict[str, Any],
        ignore_ids: set,
    ) -> None:
        self._fw_peak_cache = {}
        self._fw_floor_cache = {}
        self._fw_peak_centroid_cache = {}
        self._fw_floor_centroid_cache = {}

        if not texts:
            return

        filters = {k: v for k, v in filter_kwargs.items() if v is not None}

        unigrams: Dict[str, Counter] = {}
        counts: Dict[str, int] = {}
        unigrams_c: Dict[str, Counter] = {}
        counts_c: Dict[str, int] = {}

        print(f"  Fighting-words: pre-scanning {len(texts)} texts...")

        for n_seen, tx in enumerate(texts, start=1):
            if n_seen == 1 or n_seen % 50 == 0 or n_seen == len(texts):
                print(f"    Pre-scan: {n_seen}/{len(texts)}")

            try:
                result = self.processor.get_tokens(tx, filters, ignore_ids)
            except Exception as exc:
                print(f"    Warning: pre-scan tokenize failed for {tx}: {exc}")
                continue

            if not result:
                continue
            tokens = result[0]
            if not tokens:
                continue

            unigrams[tx] = Counter(tokens)
            counts[tx] = len(tokens)

            if self.centroid_mode == 'video':
                try:
                    files = self.processor._get_filtered_files(tx, filters, ignore_ids)
                except Exception:
                    files = None

                if files:
                    try:
                        cap = self.tomotopy_max_tokens_per_channel
                        accumulate = (
                            min(self.stat_word_count, cap)
                            if cap and self.stat_word_count
                            else (cap or self.stat_word_count)
                        )
                        cres = self.processor.load_video_centroids(
                            tx, self.centroid_videos, filters, ignore_ids,
                            max_accumulated_tokens=accumulate,
                            precomputed_files=files,
                            need_embeddings=False,
                            need_stat_corpus=False,
                            need_centroid_tokens=True,
                        )
                    except Exception:
                        cres = None

                    if cres and cres[3]:
                        unigrams_c[tx] = Counter(cres[3])
                        counts_c[tx] = len(cres[3])
                    else:
                        unigrams_c[tx] = unigrams[tx]
                        counts_c[tx] = counts[tx]
                else:
                    unigrams_c[tx] = unigrams[tx]
                    counts_c[tx] = counts[tx]
            elif self.centroid_mode == 'word':
                tc = tokens
                if self.centroid_words and len(tokens) > self.centroid_words:
                    tc = tokens[-self.centroid_words:]
                unigrams_c[tx] = Counter(tc)
                counts_c[tx] = len(tc)
            else:
                unigrams_c[tx] = unigrams[tx]
                counts_c[tx] = counts[tx]

        if not unigrams:
            print("  Fighting-words precompute: no texts tokenized; skipping")
            return

        self._fighting_words_unigrams = unigrams
        self._fighting_words_token_counts = counts

        peak_values, peak_uni = self._fw_values(
            texts, FIGHTING_WORDS_PEAK_RANK, unigrams, counts
        )
        floor_values, floor_uni = self._fw_values(
            texts, FIGHTING_WORDS_FLOOR_RANK, unigrams, counts
        )
        self._fw_peak_cache = peak_values
        self._fw_floor_cache = floor_values
        self._fighting_words_peak_unigrams = peak_uni
        self._fighting_words_floor_unigrams = floor_uni

        any_separate = any(
            unigrams_c.get(tx) is not unigrams.get(tx)
            for tx in texts if tx in unigrams
        )

        if any_separate:
            self._fighting_words_unigrams_centroid = unigrams_c
            self._fighting_words_token_counts_centroid = counts_c
            peak_c, peak_uni_c = self._fw_values(
                texts, FIGHTING_WORDS_PEAK_RANK, unigrams_c, counts_c
            )
            floor_c, floor_uni_c = self._fw_values(
                texts, FIGHTING_WORDS_FLOOR_RANK, unigrams_c, counts_c
            )
            self._fw_peak_centroid_cache = peak_c
            self._fw_floor_centroid_cache = floor_c
            self._fighting_words_peak_unigrams_centroid = peak_uni_c
            self._fighting_words_floor_unigrams_centroid = floor_uni_c
        else:
            self._fighting_words_unigrams_centroid = unigrams
            self._fighting_words_token_counts_centroid = counts
            self._fw_peak_centroid_cache = dict(peak_values)
            self._fw_floor_centroid_cache = dict(floor_values)
            self._fighting_words_peak_unigrams_centroid = dict(peak_uni)
            self._fighting_words_floor_unigrams_centroid = dict(floor_uni)

        n_peak = sum(1 for v in self._fw_peak_cache.values() if v is not None)
        n_floor = sum(1 for v in self._fw_floor_cache.values() if v is not None)
        print(
            f"  Fighting-words precompute: peak {n_peak}/{len(texts)}, "
            f"floor {n_floor}/{len(texts)}"
        )

    def _fw_values(
        self,
        texts: List[str],
        rank: int,
        unigrams: Optional[Dict[str, Counter]] = None,
        counts: Optional[Dict[str, int]] = None,
        use_centroid: bool = False,
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
        if not texts:
            return {}, {}

        if unigrams is None:
            unigrams = (
                self._fighting_words_unigrams_centroid
                if use_centroid
                else self._fighting_words_unigrams
            )
        if counts is None:
            counts = (
                self._fighting_words_token_counts_centroid
                if use_centroid
                else self._fighting_words_token_counts
            )

        counters: Dict[str, Counter] = {}
        token_counts: Dict[str, int] = {}

        for text in texts:
            c = unigrams.get(text)
            n = counts.get(text, 0)
            if c is None or n <= 0:
                continue
            counters[text] = c
            token_counts[text] = n

        if not counters:
            return {text: None for text in texts}, {text: None for text in texts}

        total_counts = Counter()
        total_tokens = 0
        for text, c in counters.items():
            total_counts.update(c)
            total_tokens += token_counts[text]

        floors: Dict[str, Optional[float]] = {}
        top_unigrams: Dict[str, Optional[str]] = {}

        for text in texts:
            focus = counters.get(text)
            n_focus = token_counts.get(text, 0)
            if focus is None or n_focus <= 0:
                floors[text] = None
                top_unigrams[text] = None
                continue

            n_rest = total_tokens - n_focus
            if n_rest <= 0:
                floors[text] = None
                top_unigrams[text] = None
                continue

            rest = total_counts - focus
            candidates = [
                phrase for phrase, c in focus.items()
                if c >= FIGHTING_WORDS_FLOOR_MIN_COUNT
            ]
            if not candidates:
                floors[text] = None
                top_unigrams[text] = None
                continue

            V = len(candidates)
            z_score_pairs: List[Tuple[float, str]] = []

            for phrase in candidates:
                f_focus = focus[phrase]
                f_rest = rest.get(phrase, 0)
                p_focus = (
                    (f_focus + FIGHTING_WORDS_FLOOR_ALPHA)
                    / (n_focus + FIGHTING_WORDS_FLOOR_ALPHA * V)
                )
                p_rest = (
                    (f_rest + FIGHTING_WORDS_FLOOR_ALPHA)
                    / (n_rest + FIGHTING_WORDS_FLOOR_ALPHA * V)
                )
                log_odds = math.log(p_focus) - math.log(p_rest)
                variance = (
                    (1.0 / (f_focus + FIGHTING_WORDS_FLOOR_ALPHA))
                    + (1.0 / (f_rest + FIGHTING_WORDS_FLOOR_ALPHA))
                )
                z_score = log_odds / math.sqrt(variance)
                z_score_pairs.append((z_score, phrase))

            if not z_score_pairs:
                floors[text] = None
                top_unigrams[text] = None
                continue

            z_score_pairs.sort(reverse=True, key=lambda x: x[0])
            rank_index = min(rank, len(z_score_pairs)) - 1
            floors[text] = float(z_score_pairs[rank_index][0])
            top_unigrams[text] = z_score_pairs[rank_index][1]

        return floors, top_unigrams

    def _compute_fighting_words_values(
        self,
        texts: List[str],
        rank: int,
        use_centroid: bool = False,
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
        return self._fw_values(texts, rank, use_centroid=use_centroid)

    def _compute_fighting_words_floor_values(
        self,
        texts: List[str],
        use_centroid: bool = False,
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
        return self._fw_values(
            texts, FIGHTING_WORDS_FLOOR_RANK, use_centroid=use_centroid
        )

    def _get_centroid_tokens_for_tomotopy(
        self,
        text: str,
        filters: Dict,
        ignore_ids: set,
    ) -> List[str]:
        if self.centroid_mode == 'video':
            files = self.processor._get_filtered_files(text, filters, ignore_ids)
            if not files:
                return []

            cap = self.tomotopy_max_tokens_per_channel
            accumulate = (
                min(self.stat_word_count, cap) if cap and self.stat_word_count else (cap or self.stat_word_count)
            )
            result = self.processor.load_video_centroids(
                text, self.centroid_videos, filters, ignore_ids,
                max_accumulated_tokens=accumulate,
                precomputed_files=files,
                need_embeddings=False,
                need_stat_corpus=False,
                need_centroid_tokens=True,
            )
            return (result[3] or []) if result else []

        result = self.processor.get_tokens(text, filters, ignore_ids)
        if not result:
            return []
        tokens = result[0]
        if self.centroid_mode == 'word' and self.centroid_words and self.centroid_words > 0:
            tokens = tokens[-self.centroid_words:]
        return tokens

    def _train_tomotopy_topics(
        self,
        texts: List[str],
        ignore_ids: set,
        filter_kwargs: Dict[str, Any],
    ) -> None:
        if self.tomotopy_num_topics <= 0:
            return

        if not texts:
            return

        cache_dir = self.base_dir / "cache" / "tomotopy"
        cache_dir.mkdir(parents=True, exist_ok=True)
        sig_path = cache_dir / "signature.json"
        proportions_path = cache_dir / "proportions.json"
        labels_path = cache_dir / "labels.json"

        filter_sig: Dict[str, Any] = {}
        for key in sorted(CACHE_FILTER_KEYS):
            if key in ('centroid_mode', 'centroid_videos', 'centroid_words', 'stat_word_count'):
                continue
            v = filter_kwargs.get(key)
            if v is None:
                continue
            filter_sig[key] = str(v)

        sig = {
            'num_topics': int(self.tomotopy_num_topics),
            'iterations': int(self.tomotopy_iterations),
            'threads': int(self.tomotopy_threads),
            'texts': sorted(texts),
            'ignore_ids': sorted(str(v) for v in ignore_ids),
            'centroid_mode': self.centroid_mode,
            'centroid_videos': int(self.centroid_videos or 0),
            'centroid_words': int(self.centroid_words or 0),
            'stat_word_count': int(self.stat_word_count or 0),
            'filters': filter_sig,
            'change_fword_to_stupid': bool(getattr(self, 'change_fword_to_stupid', False)),
        }

        if sig_path.exists() and proportions_path.exists():
            try:
                with open(sig_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                if saved == sig:
                    with open(proportions_path, 'r', encoding='utf-8') as f:
                        props = json.load(f)
                    self.tomotopy_topic_proportions = {
                        tx: {int(k): float(v) for k, v in topics.items()}
                        for tx, topics in props.items()
                    }
                    if labels_path.exists():
                        try:
                            with open(labels_path, 'r', encoding='utf-8') as f:
                                labels = json.load(f)
                            self.tomotopy_topic_labels = {
                                int(k): str(v) for k, v in labels.items()
                            }
                        except Exception:
                            self.tomotopy_topic_labels = {}
                    print(
                        f"  Tomotopy topics loaded from cache "
                        f"({len(self.tomotopy_topic_proportions)} texts, "
                        f"{self.tomotopy_num_topics} topics)"
                    )
                    return
            except Exception as exc:
                print(f"  Warning: failed to load tomotopy cache ({exc}); retraining")

        try:
            import tomotopy as tp
        except ImportError:
            print("  Tomotopy not installed; skipping topic modeling.")
            print("  Install it with: pip install tomotopy")
            return

        print(
            f"  Training Tomotopy LDA: {self.tomotopy_num_topics} topics, "
            f"{self.tomotopy_iterations} iterations, "
            f"{self.tomotopy_threads} threads"
        )

        filters = {k: v for k, v in filter_kwargs.items() if v is not None}
        corpus = []
        doc_names = []

        for n_seen, tx in enumerate(texts, start=1):
            if n_seen == 1 or n_seen % 50 == 0 or n_seen == len(texts):
                print(f"    Gathering tomotopy tokens: {n_seen}/{len(texts)}")
            try:
                tokens = self._get_centroid_tokens_for_tomotopy(tx, filters, ignore_ids)
            except Exception as exc:
                print(f"    Warning: could not gather tokens for {tx}: {exc}")
                continue
            if not tokens:
                continue
            corpus.append(tokens)
            doc_names.append(tx)

        if not corpus:
            print("  Tomotopy skipped: no texts produced centroid tokens")
            return

        model = tp.LDAModel(
            k=self.tomotopy_num_topics,
            alpha=0.1,
            eta=0.01,
            min_cf=1,
            min_df=1,
            rm_top=0,
            seed=20,
        )

        for tokens in corpus:
            model.add_doc(tokens)

        print("    Training model...")
        for i in range(self.tomotopy_iterations):
            model.train(1)
            if (i + 1) % 100 == 0:
                print(f"    Iteration {i + 1}/{self.tomotopy_iterations}")

        proportions: Dict[str, Dict[int, float]] = {}
        for i, tx in enumerate(doc_names):
            if i >= len(corpus):
                continue
            topic_dist = model.get_doc_topics(i, min_prob=0.0)
            proportions[tx] = {int(tid): float(prob) for tid, prob in topic_dist}

        self.tomotopy_topic_proportions = proportions

        self.tomotopy_topic_labels = {}
        for tid in range(self.tomotopy_num_topics):
            topic_words = model.get_topic_words(tid, top_n=8)
            self.tomotopy_topic_labels[tid] = ' '.join(topic_words)

        try:
            with open(sig_path, 'w', encoding='utf-8') as f:
                json.dump(sig, f)
            with open(proportions_path, 'w', encoding='utf-8') as f:
                json.dump(self.tomotopy_topic_proportions, f)
            with open(labels_path, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in self.tomotopy_topic_labels.items()}, f)
        except Exception as exc:
            print(f"  Warning: failed to write tomotopy cache: {exc}")

        matched = sum(1 for tx in texts if tx in self.tomotopy_topic_proportions)
        print(
            f"  Tomotopy topics trained; matched {matched}/{len(texts)} texts "
            f"({len(self.tomotopy_topic_proportions)} docs parsed)"
        )

    def _ensure_runtime_cache_for_tokens(self, tokens: List[str]) -> None:
        if (
            getattr(self, '_runtime_cache_tokens_ref', None) is not tokens
            or getattr(self, '_runtime_cache_len', -1) != len(tokens)
        ):
            self._runtime_cache_tokens_ref = tokens
            self._runtime_cache_len = len(tokens)
            self._runtime_cache_text = None
            self._runtime_cache_text_lower = None
            self._runtime_cache_text_bytes = None
            self._runtime_cache_counter = None
            self._runtime_cache_pattern_counts = {}
            self._runtime_cache_pattern_prefixes = {}
            self._runtime_cache_dynamic_counts = {}
            self._runtime_cache_spacy_stats = None
            self._runtime_cache_oov_split = None
            self._runtime_cache_lzma_full = None
            self._runtime_cache_lzma_windows = None
            self._runtime_cache_pieces = None
            self._runtime_cache_edges = None

    def _clear_runtime_cache(self) -> None:
        self._runtime_cache_tokens_ref = None
        self._runtime_cache_len = -1
        self._runtime_cache_text = None
        self._runtime_cache_text_lower = None
        self._runtime_cache_text_bytes = None
        self._runtime_cache_counter = None
        self._runtime_cache_pattern_counts = {}
        self._runtime_cache_pattern_prefixes = {}
        self._runtime_cache_dynamic_counts = {}
        self._runtime_cache_spacy_stats = None
        self._runtime_cache_oov_split = None
        self._runtime_cache_lzma_full = None
        self._runtime_cache_lzma_windows = None
        self._runtime_cache_pieces = None
        self._runtime_cache_edges = None

    def _get_runtime_text(self, tokens: List[str]) -> str:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_text is None:
            self._runtime_cache_text = ' '.join(tokens)
        return self._runtime_cache_text

    def _get_runtime_text_bytes(self, tokens: List[str]) -> bytes:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_text_bytes is None:
            self._runtime_cache_text_bytes = self._get_runtime_text(tokens).encode('utf-8', errors='ignore')
        return self._runtime_cache_text_bytes

    def _get_runtime_text_lower(self, tokens: List[str]) -> str:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_text_lower is None:
            self._runtime_cache_text_lower = self._get_runtime_text(tokens).lower()
        return self._runtime_cache_text_lower

    @staticmethod
    def _extract_required_literal_prefix(pattern_obj: Any) -> Optional[str]:
        src = getattr(pattern_obj, 'pattern', None)
        if src is None:
            return None
        if isinstance(src, bytes):
            try:
                src = src.decode('utf-8', errors='ignore')
            except Exception:
                return None
        if not isinstance(src, str):
            return None

        depth, i, n = 0, 0, len(src)
        while i < n:
            ch = src[i]
            if ch == '\\':
                i += 2
                continue
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth -= 1
            elif ch == '|' and depth == 0:
                return None
            i += 1

        i = 0
        while i < n and src[i] in '^$':
            i += 1
        while i + 1 < n and src[i] == '\\' and src[i + 1] == 'b':
            i += 2

        chars: List[str] = []
        while i < n and src[i].isalnum():
            chars.append(src[i].lower())
            i += 1
        if chars and i < n and src[i] in '?*{':
            chars.pop()

        prefix = ''.join(chars)
        return prefix if len(prefix) >= 4 else None

    def _get_runtime_token_counter(self, tokens: List[str]) -> Counter:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_counter is None:
            self._runtime_cache_counter = Counter(tokens)
        return self._runtime_cache_counter

    def _get_runtime_word_pieces(self, tokens: List[str]) -> Counter:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_pieces is None:
            pieces: Counter = Counter()
            for tok, c in self._get_runtime_token_counter(tokens).items():
                for w in _WORD_RUN.findall(tok):
                    pieces[w] += c
            self._runtime_cache_pieces = pieces
        return self._runtime_cache_pieces

    def _runtime_edges(
        self, tokens: List[str]
    ) -> Tuple[List[str], List[str], Dict[int, Counter]]:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_edges is None:
            first, last = {}, {}
            for tok in self._get_runtime_token_counter(tokens):
                runs = _WORD_RUN.findall(tok)
                first[tok] = runs[0] if runs else ''
                last[tok] = runs[-1] if runs else ''
            self._runtime_cache_edges = (
                [last[t] for t in tokens],
                [first[t] for t in tokens],
                {},
            )
        return self._runtime_cache_edges

    def _get_runtime_ngram_counts(self, tokens: List[str], k: int) -> Counter:
        last_seq, first_seq, by_k = self._runtime_edges(tokens)
        c = by_k.get(k)
        if c is None:
            mids = [tokens[j:] for j in range(1, k - 1)]
            c = Counter(zip(last_seq, *mids, first_seq[k - 1:]))
            by_k[k] = c
        return c

    def _fast_count(self, tokens: List[str], terms: Dict[int, frozenset]) -> int:
        total = 0
        for k, words in terms.items():
            if k == 1:
                pieces = self._get_runtime_word_pieces(tokens)
                total += sum(pieces.get(w[0], 0) for w in words)
            else:
                c = self._get_runtime_ngram_counts(tokens, k)
                total += sum(c.get(w, 0) for w in words)
        return total

    def _get_runtime_spacy_stats(self, tokens: List[str]) -> Optional[Dict[str, int]]:
        if not self.enable_spacy:
            return None

        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_spacy_stats is not None:
            return self._runtime_cache_spacy_stats

        if cfg.spacy_nlp is None:
            try:
                import spacy
                cfg.spacy_nlp = spacy.load("en_core_web_sm")
            except (ModuleNotFoundError, ImportError, OSError):
                from helpers.atlas_config import _install_spacy_model
                _install_spacy_model()
                if cfg.spacy_nlp is None:
                    return None
            except Exception:
                return None

        try:
            text = self._get_runtime_text(tokens)
            if len(text) + 1 > getattr(cfg.spacy_nlp, 'max_length', 1_000_000):
                cfg.spacy_nlp.max_length = len(text) + 1
            doc = cfg.spacy_nlp(text, disable=['parser', 'ner'])
            content_tags = {'NOUN', 'VERB', 'ADJ', 'ADV'}
            total_valid = 0
            total_content = 0
            gerund = 0
            imperative = 0
            for token in doc:
                if not token.is_punct and not token.is_space:
                    total_valid += 1
                    if token.pos_ in content_tags:
                        total_content += 1
                if token.tag_ == 'VBG':
                    gerund += 1
                if token.pos_ == 'VERB' and token.tag_ == 'VB':
                    if (
                        token.i == 0
                        or doc[token.i - 1].is_punct
                        or doc[token.i - 1].pos_ == 'CCONJ'
                    ):
                        imperative += 1
            stats = {
                'total_valid': int(total_valid),
                'total_content': int(total_content),
                'gerund_count': int(gerund),
                'imperative_count': int(imperative),
            }
            self._runtime_cache_spacy_stats = stats
            return stats
        except Exception:
            return None

    def _compute_pattern_count(self, tokens: List[str], patterns: List) -> float:
        if not tokens:
            return 0.0

        self._ensure_runtime_cache_for_tokens(tokens)
        count = 0
        text = self._get_runtime_text(tokens)
        pattern_cache: Dict[Tuple[int, bool], int] = self._runtime_cache_pattern_counts
        prefix_cache: Dict[int, Optional[str]] = self._runtime_cache_pattern_prefixes

        pending = list(patterns) if isinstance(patterns, (list, tuple, set)) else [patterns]
        while pending:
            pat = pending.pop()
            if isinstance(pat, (list, tuple, set)):
                pending.extend(pat)
                continue
            if not hasattr(pat, 'findall'):
                continue

            terms = _pure_terms(pat)
            if terms is not None:
                count += self._fast_count(tokens, terms)
                continue

            pid = id(pat)
            if pid not in prefix_cache:
                prefix_cache[pid] = self._extract_required_literal_prefix(pat)
            req = prefix_cache.get(pid)
            if req:
                if req not in self._get_runtime_text_lower(tokens):
                    continue

            try:
                key = (id(pat), False)
                c = pattern_cache.get(key)
                if c is None:
                    c = len(pat.findall(text))
                    pattern_cache[key] = c
                count += c
            except TypeError:
                key = (id(pat), True)
                c = pattern_cache.get(key)
                if c is None:
                    c = len(pat.findall(self._get_runtime_text_bytes(tokens)))
                    pattern_cache[key] = c
                count += c

        return float(count)

    def _compute_dynamic_token_count(self, tokens: List[str], metric_name: str) -> float:
        if not tokens:
            return 0.0

        self._ensure_runtime_cache_for_tokens(tokens)
        cache: Dict[str, float] = self._runtime_cache_dynamic_counts
        if metric_name in cache:
            return cache[metric_name]

        if metric_name in SIMPLE_COLOR_PATTERN_MAP:
            v = self._compute_pattern_count(tokens, SIMPLE_COLOR_PATTERN_MAP[metric_name])
            cache[metric_name] = v
            return v

        if metric_name == 'right_count':
            base = self._compute_pattern_count(tokens, DYNAMIC_PATTERN_METRIC_MAP['right_count'])
            text = self._get_runtime_text(tokens)
            ar = len(re.findall(r'\ball\s+right\b', text, re.IGNORECASE))
            v = float(max(0.0, base - ar))
            cache[metric_name] = v
            return v

        if metric_name in DYNAMIC_PATTERN_METRIC_MAP:
            v = self._compute_pattern_count(tokens, DYNAMIC_PATTERN_METRIC_MAP[metric_name])
            cache[metric_name] = v
            return v

        if metric_name == 'unique_words_count':
            return self._compute_unique_words_count(tokens)

        if metric_name in (
            'unique_oov_count', 'total_oov_count',
            'unique_non_oov_count', 'total_non_oov_count',
        ):
            uo, to, un, tn = self._compute_oov_split_counts(tokens)
            if metric_name == 'unique_oov_count':
                return uo
            if metric_name == 'total_oov_count':
                return to
            if metric_name == 'unique_non_oov_count':
                return un
            return tn

        forms = DYNAMIC_TOKEN_SET_METRIC_MAP.get(metric_name)
        if forms is not None:
            tc = self._get_runtime_token_counter(tokens)
            v = float(sum(tc.get(f, 0) for f in forms))
            cache[metric_name] = v
            return v

        fb = DYNAMIC_COUNT_FALLBACK_METHODS.get(metric_name)
        if fb and fb != '_compute_dynamic_token_count':
            method = getattr(self, fb, None)
            if method is not None:
                v = method(tokens)
                cache[metric_name] = float(v)
                return v

        return 0.0

    def _compute_unique_words_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        uniq = set()
        for token in tokens:
            cleaned = token.lower().strip("'\".,!?;:()[]{}")
            if len(cleaned) < 2:
                continue
            if not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", cleaned):
                continue
            uniq.add(cleaned)
        return float(len(uniq))

    def _compute_oov_split_counts(
        self, tokens: List[str]
    ) -> Tuple[float, float, float, float]:
        if not tokens:
            return 0.0, 0.0, 0.0, 0.0

        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_oov_split is not None:
            return self._runtime_cache_oov_split

        try:
            global enchant_dict, enchant_word_check_cache
            if not ENCHANT_AVAILABLE:
                self._runtime_cache_oov_split = (0.0, 0.0, 0.0, 0.0)
                return self._runtime_cache_oov_split
            if enchant_dict is None:
                try:
                    enchant_dict = enchant.Dict("en_US")
                except Exception:
                    self._runtime_cache_oov_split = (0.0, 0.0, 0.0, 0.0)
                    return self._runtime_cache_oov_split

            counts = self._get_runtime_token_counter(tokens)
            if not counts:
                self._runtime_cache_oov_split = (0.0, 0.0, 0.0, 0.0)
                return self._runtime_cache_oov_split

            cache = enchant_word_check_cache
            if len(cache) > 1_000_000:
                cache.clear()
            check = enchant_dict.check

            total_oov = 0
            unique_oov = 0
            total_non = 0
            unique_non = 0
            for cleaned, count in counts.items():
                L = len(cleaned)
                if L == 0 or L > 30 or not cleaned.isascii():
                    total_oov += count
                    unique_oov += 1
                    continue
                cand = cleaned.replace("'", "") if "'" in cleaned else cleaned
                if not cand or not cand.isalpha():
                    total_oov += count
                    unique_oov += 1
                    continue
                val = cache.get(cleaned)
                if val is None:
                    try:
                        val = bool(check(cleaned))
                    except Exception:
                        val = False
                    cache[cleaned] = val
                if val:
                    total_non += count
                    unique_non += 1
                else:
                    total_oov += count
                    unique_oov += 1

            self._runtime_cache_oov_split = (
                float(unique_oov), float(total_oov),
                float(unique_non), float(total_non),
            )
            return self._runtime_cache_oov_split
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def _compute_oov_counts(self, tokens: List[str]) -> Tuple[float, float]:
        try:
            uo, to, _, _ = self._compute_oov_split_counts(tokens)
            return uo, to
        except Exception:
            return 0.0, 0.0

    def _all_metrics(self, tokens: List[str], chunks: List[str]) -> Dict[str, Any]:
        names = [
            n for n in self._target_metrics()
            if not MetricConfig.is_centroid_metric(n)
        ]
        return self._metrics(tokens, chunks, names)

    def _metric_subset(
        self, tokens: List[str], chunks: List[str], metric_names: List[str]
    ) -> Dict[str, Any]:
        return self._metrics(tokens, chunks, metric_names)

    def _compute_specific_metrics(
        self,
        tokens: List[str],
        chunks: List[str],
        chunk_embs: Optional[np.ndarray],
        metric_names: List[str],
    ) -> Dict[str, Any]:
        return self._metrics(tokens, chunks, metric_names)

    def _metrics(
        self, tokens: List[str], chunks: List[str], metric_names: List[str]
    ) -> Dict[str, Any]:
        metrics = {}
        total = len(metric_names)
        interval = 25 if total >= 100 else (10 if total >= 30 else 0)
        started = time.time()

        joined: Optional[str] = None
        freq: Optional[Counter] = None
        lex: Optional[LexicalRichness] = None

        try:
            for index, name in enumerate(metric_names, start=1):
                if name not in MetricConfig.METRICS:
                    continue
                cfg_ = MetricConfig.METRICS[name]
                if self._skipped(name):
                    metrics[name] = None
                else:
                    method = cfg_.get('compute_method', '')
                    t0 = time.time()
                    if name in ('word_burstiness', 'hapax_ratio', 'dis_ratio') and freq is None:
                        freq = Counter(tokens) if tokens else Counter()
                    if name in ('MATTR', 'yules_k', 'MATTR_centroid') and lex is None:
                        if joined is None:
                            joined = ' '.join(tokens) if tokens else ''
                        lex = LexicalRichness(joined) if joined else None
                    metrics[name] = self._one_metric(name, cfg_, tokens, chunks, freq, lex)
                    elapsed = time.time() - t0
                    if elapsed >= 8.0:
                        print(f"      slow metric: {name} ({elapsed:.1f}s, method={method})")
                if interval and (index % interval == 0 or index == total):
                    print(
                        f"      progress: {index}/{total} metrics "
                        f"({time.time() - started:.1f}s)"
                    )
            return metrics
        finally:
            self._clear_runtime_cache()

    def _one_metric(
        self,
        name: str,
        cfg_: Dict,
        tokens: List[str],
        chunks: List[str],
        freq: Optional[Counter],
        lex: Optional[LexicalRichness],
    ) -> Any:
        try:
            m = cfg_['compute_method']
            if m.startswith('_compute'):
                method = getattr(self, m)
                if name.startswith('letter_') and name.endswith('_count'):
                    return method(tokens, letter=name.split('_')[1])
                if m == '_compute_dynamic_token_count':
                    return method(tokens, metric_name=name)
                if name in ('word_burstiness', 'hapax_ratio', 'dis_ratio'):
                    return method(tokens, freq_counter=freq)
                if name in ('MATTR', 'yules_k', 'MATTR_centroid'):
                    return method(tokens, lex=lex)
                if m == '_compute_MTLD_score':
                    return method(tokens)
                kw = {k: cfg_[k] for k in ('lexicon_key', 'phrase_key') if k in cfg_}
                return method(tokens, **kw)
            if m == 'calculate_perplexity':
                return self.processor.calculate_perplexity(chunks)
            if m == 'compute_vader_sentiment':
                return self.processor.compute_vader_sentiment(chunks)[
                    cfg_.get('result_index', 0)
                ]
            print(f"    Warning: Unknown compute method '{m}' for '{name}'")
            return None
        except Exception as e:
            print(f"    Error computing {name}: {e}")
            return None

    @staticmethod
    def _low_token_result(
        mean_emb: np.ndarray, token_count: int, file_count: int
    ) -> Dict[str, Any]:
        return {
            'mean_emb': mean_emb,
            'token_count': token_count,
            'chunk_count': 0,
            'file_count': file_count,
            'low_token_text': True,
        }

    @staticmethod
    def _token_lists_are_identical(a: List[str], b: List[str]) -> bool:
        if a is b:
            return True
        if a is None or b is None:
            return False
        if len(a) != len(b):
            return False
        return a == b

    def _cache_filters(
        self,
        filters: Dict,
        centroid_extras: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        cf = {k: v for k, v in filters.items() if k in CACHE_FILTER_KEYS}
        if centroid_extras:
            cf.update(centroid_extras)
        cf.update(self.embedding_cache_signature)
        cf['metric_signature'] = hashlib.md5(
            ','.join(sorted(MetricConfig.get_metric_names())).encode('utf-8')
        ).hexdigest()[:12]
        return cf

    def process_text(
        self,
        text: str,
        ignore_ids: Optional[set] = None,
        **filter_kwargs,
    ) -> Optional[Dict]:
        if ignore_ids is None:
            ignore_ids = set()

        self._current_video_count = 0

        if not validate_text_name(text):
            print(f"\nError: Invalid text name '{text}' rejected for security")
            return None

        print(f"Processing: {text}")
        filters = {k: v for k, v in filter_kwargs.items() if v is not None}
        cache_filters = self._cache_filters(filters)

        safe = sanitize_filename(text)
        txt_dir = self.processor.base_dir / "data/input" / safe / "txt_files"
        if not txt_dir.exists() and not self.processor.has_external_document(text):
            print(f"  Skipped: {text} (no txt_files directory)")
            return None

        if self.centroid_mode:
            return self._centroid_text(text, filters, ignore_ids, **filter_kwargs)

        result = self.processor.get_tokens(text, filters, ignore_ids)
        if not result:
            return None

        tokens, txt_dir, file_count, video_ids = result
        style_profile = self._burrows_profile(tokens)
        emb_only_low = filter_kwargs.get('embeddings_only_low_token', False)

        self._centroid_equals_stat[text] = True

        min_total = filters.get('min_tokens_total', 0)
        if len(tokens) < min_total:
            if emb_only_low:
                print(
                    f"  Low tokens: {text} ({len(tokens):,} tokens, "
                    f"minimum is {min_total:,}) - generating embeddings only"
                )
                cached_emb = cached_mean = cached_chunks = None
                if self.cache:
                    cached = self.cache.load(
                        text, cache_filters, video_ids,
                        enable_spacy=self.enable_spacy,
                        needs_perplexity_model=self.processor.needs_perplexity_model,
                        enable_ngram_entropy=self.enable_ngram_entropy,
                        fast_mode=self.fast_mode,
                        expected_token_count=len(tokens),
                        embeddings_only=True,
                        force_latest_cache=self.force_latest_cache,
                    )
                    if cached is not None:
                        cached_emb = cached.get('embeddings')
                        cached_mean = cached.get('mean_emb')
                        cached_chunks = cached.get('chunks')

                if cached_chunks is not None and cached_emb is not None:
                    chunks = cached_chunks
                    chunk_embs = cached_emb
                    mean_emb = (
                        cached_mean
                        if cached_mean is not None
                        else np.mean(chunk_embs, axis=0).astype(np.float32)
                    )
                    print(f"  Cached: {text} (using cached embeddings)")
                else:
                    chunks = self.processor.chunk(tokens)
                    if not chunks:
                        print(f"  Skipped: {text} (no chunks generated)")
                        return None
                    chunk_embs = self._encode(chunks)
                    mean_emb = np.mean(chunk_embs, axis=0).astype(np.float32)

                self.embeddings[text] = mean_emb

                rd = {
                    'mean_emb': mean_emb,
                    'token_count': len(tokens),
                    'chunk_count': len(chunks),
                    'file_count': file_count,
                    'low_token_text': True,
                }
                if self.cache and (cached_chunks is None or cached_emb is None):
                    self.cache.save(
                        text, cache_filters, rd, video_ids,
                        len(tokens), len(chunks),
                        chunk_embs=chunk_embs, mean_emb=mean_emb, chunks=chunks,
                        enable_spacy=self.enable_spacy,
                        needs_perplexity_model=self.processor.needs_perplexity_model,
                        enable_ngram_entropy=self.enable_ngram_entropy,
                        fast_mode=self.fast_mode,
                    )
                print(
                    f"  Complete: {text} ({len(tokens):,} tokens, "
                    f"{len(chunks)} chunks, {file_count} files) [embeddings only]"
                )
                return rd

            print(
                f"  Skipped: {text} ({len(tokens):,} tokens, minimum is {min_total:,})"
            )
            return None

        cached_metrics: Dict[str, Any] = {}
        missing: List[str] = []
        cached_emb = cached_mean = cached_chunks = None
        target = self._target_metrics(skip_centroid_duplicates=True)

        if self.cache:
            cached = self.cache.load(
                text, cache_filters, video_ids,
                enable_spacy=self.enable_spacy,
                needs_perplexity_model=self.processor.needs_perplexity_model,
                enable_ngram_entropy=self.enable_ngram_entropy,
                fast_mode=self.fast_mode,
                expected_token_count=len(tokens),
                embeddings_only=self.no_metrics,
                force_latest_cache=self.force_latest_cache,
            )
            if cached is not None:
                cached_metrics = cached.get('metrics', {})
                cached_emb = cached.get('embeddings')
                cached_mean = cached.get('mean_emb')
                cached_chunks = cached.get('chunks')
                missing = [
                    m for m in target
                    if self._metric_needs_compute(m, cached_metrics)
                ]
            else:
                missing = list(target)

        if cached_chunks is not None and cached_emb is not None:
            chunks = cached_chunks
            chunk_embs = cached_emb
            mean_emb = cached_mean
            print(f"  Cached: {text} (using cached embeddings)")
        else:
            chunks = self.processor.chunk(tokens)
            if not chunks:
                print(f"  Skipped: {text} (no chunks generated)")
                return None
            chunk_embs = self._encode(chunks)
            mean_emb = np.mean(chunk_embs, axis=0).astype(np.float32)

        self.embeddings[text] = mean_emb

        report_values: Dict[str, Any] = {}
        report_src = 'computed'

        if missing:
            print(f"    -> Computing {len(missing)} new metrics")
            new = self._metric_subset(tokens, chunks, missing)
            all_metrics = {**cached_metrics, **new}
            report_values = new
        elif cached_metrics:
            all_metrics = cached_metrics
            report_values = cached_metrics
            report_src = 'cache'
        else:
            all_metrics = self._all_metrics(tokens, chunks)
            report_values = all_metrics

        if not self.no_metrics:
            self._print_summary(report_values, report_src)

        rd = {
            'mean_emb': mean_emb,
            'token_count': len(tokens),
            'chunk_count': len(chunks),
            'file_count': file_count,
            '_burrows_rate_profile': style_profile,
            **all_metrics,
        }

        if self.cache:
            self.cache.save(
                text, cache_filters, rd, video_ids,
                len(tokens), len(chunks),
                chunk_embs=chunk_embs, mean_emb=mean_emb, chunks=chunks,
                enable_spacy=self.enable_spacy,
                needs_perplexity_model=self.processor.needs_perplexity_model,
                enable_ngram_entropy=self.enable_ngram_entropy,
                fast_mode=self.fast_mode,
            )

        comp = all_metrics.get('compression')
        if isinstance(comp, (float, int)):
            print(
                f"  Complete: {text} ({len(tokens):,} tokens, {len(chunks)} chunks, "
                f"{file_count} files, CR: {_format_trimmed_decimal(float(comp), 4)})"
            )
        else:
            print(
                f"  Complete: {text} ({len(tokens):,} tokens, "
                f"{len(chunks)} chunks, {file_count} files)"
            )

        return rd

    def _centroid_text(
        self,
        text: str,
        filters: Dict,
        ignore_ids: Optional[set] = None,
        **filter_kwargs,
    ):
        if ignore_ids is None:
            ignore_ids = set()

        print(f"Processing (centroid mode={self.centroid_mode}): {text}")

        cache_filters = self._cache_filters(
            filters,
            centroid_extras={
                'centroid_mode': self.centroid_mode,
                'centroid_videos': int(self.centroid_videos),
                'centroid_words': int(self.centroid_words),
                'stat_word_count': int(self.stat_word_count),
            },
        )

        if self.centroid_mode == 'video':
            return self._centroid_video(text, filters, ignore_ids, cache_filters)
        if self.centroid_mode == 'word':
            return self._centroid_word(text, filters, ignore_ids, cache_filters)

        print(f"  Skipped: {text} (unknown centroid mode: {self.centroid_mode})")
        return None

    def _centroid_video(
        self, text: str, filters: Dict, ignore_ids: set, cache_filters: Dict
    ):
        min_stat = (
            int(self.stat_word_count)
            if (self.stat_word_count and self.stat_word_count > 0)
            else 0
        )
        cap = min_stat if min_stat > 0 else None
        enforce_min = (not self.no_metrics) and min_stat > 0

        files = self.processor._get_filtered_files(text, filters, ignore_ids)
        if not files:
            print(f"  Skipped: {text} (no txt_files directory or no files passed filters)")
            return None

        vids_for_cache = [extract_video_id(f.name) for _, f in files]
        n_files = len(vids_for_cache)

        target = self._target_metrics()
        stat_names = [m for m in target if not MetricConfig.is_centroid_metric(m)]
        centroid_names = [m for m in target if MetricConfig.is_centroid_metric(m)]

        cached_metrics: Dict[str, Any] = {}
        text_emb: Optional[np.ndarray] = None
        cached_low: Optional[bool] = None
        missing: List[str] = list(target)

        ckw = dict(
            enable_spacy=self.enable_spacy,
            needs_perplexity_model=self.processor.needs_perplexity_model,
            enable_ngram_entropy=self.enable_ngram_entropy,
            fast_mode=self.fast_mode,
        )

        if self.cache:
            cached = self.cache.load(
                text, cache_filters, vids_for_cache,
                embeddings_only=self.no_metrics,
                force_latest_cache=self.force_latest_cache,
                expected_token_count=cap,
                **ckw,
            )
            if cached is not None and cached.get('mean_emb') is not None:
                text_emb = cached['mean_emb']
                cached_metrics = cached.get('metrics', {}) or {}
                cached_tokens = cached.get('token_count', 0)
                cached_low = bool(enforce_min and cached_tokens < min_stat)

                needed = (
                    [m for m in target if MetricConfig.is_centroid_metric(m)]
                    if cached_low
                    else list(target)
                )
                missing = [
                    m for m in needed
                    if self._metric_needs_compute(m, cached_metrics)
                ]

                if self.no_metrics or (not missing and not self.compute_fighting_words):
                    if not self.no_metrics:
                        self._print_summary(cached_metrics, 'cache')
                    result_metrics = dict(cached_metrics)
                    if cached_low:
                        for m in stat_names:
                            result_metrics.pop(m, None)
                    self.embeddings[text] = text_emb
                    print(f"  Complete: {text} ({cached_tokens:,} tokens, cached)")
                    return {
                        'mean_emb': text_emb,
                        'token_count': cached_tokens,
                        'chunk_count': 0,
                        'file_count': n_files,
                        'low_token_text': bool(cached_low),
                        '_centroid_only': bool(cached_low),
                        **({} if self.no_metrics else result_metrics),
                    }

                uncached_by_fast_mode = [
                    name for name in missing
                    if self.fast_mode and name in ('ngram_entropy_2', 'ngram_entropy_3')
                ]
                if uncached_by_fast_mode:
                    print(
                        "    -> Not persisted because --fast-mode excludes: "
                        f"{self._summarize(uncached_by_fast_mode, max_items=None)}"
                    )

        result = self.processor.load_video_centroids(
            text, self.centroid_videos, filters, ignore_ids,
            max_accumulated_tokens=cap,
            precomputed_files=files,
            need_embeddings=text_emb is None,
            need_stat_corpus=True,
            need_centroid_tokens=True,
        )
        if not result:
            print(f"  Skipped: {text} (no video centroids)")
            return None

        video_ids, video_embeddings, stat_tokens, centroid_tokens, avg_words = result

        if text_emb is None:
            if not video_embeddings:
                print(f"  Skipped: {text} (no embeddings)")
                return None
            text_emb = np.mean(video_embeddings, axis=0).astype(np.float32)
            video_embeddings.clear()

        self.embeddings[text] = text_emb

        centroid_eq = self._token_lists_are_identical(centroid_tokens, stat_tokens)
        self._centroid_equals_stat[text] = centroid_eq
        is_low = bool(enforce_min and len(stat_tokens) < min_stat)

        self._current_video_count = len(video_ids)

        chunks: List[str] = []
        stat_metrics: Dict[str, Any] = {}
        centroid_metrics: Dict[str, Any] = {}

        if not self.no_metrics and centroid_tokens:
            c_missing = [m for m in missing if m in centroid_names]
            if centroid_eq and not is_low:
                c_missing = [
                    m for m in c_missing
                    if not MetricConfig.is_centroid_duplicate(m)
                ]
            if is_low:
                for m in missing:
                    if m in ('ngram_entropy_2', 'ngram_entropy_3') and m not in c_missing:
                        c_missing.append(m)
            if c_missing:
                needs_chunks = any(
                    MetricConfig.METRICS.get(m, {}).get('compute_method')
                    in ('calculate_perplexity', 'compute_vader_sentiment')
                    for m in c_missing
                )
                c_chunks = (
                    self.processor.chunk(centroid_tokens) if needs_chunks else []
                )
                print(
                    f"    -> Computing {len(c_missing)} centroid metrics "
                    f"from {len(centroid_tokens):,} tokens"
                )
                centroid_metrics = self._metrics(
                    centroid_tokens, c_chunks, c_missing
                )

        if (
            'avg_words_per_video' in centroid_names
            and 'avg_words_per_video' not in cached_metrics
        ):
            centroid_metrics['avg_words_per_video'] = avg_words

        self._current_video_count = 0

        if not self.no_metrics and not is_low and stat_tokens:
            s_missing = [m for m in missing if m in stat_names]
            if s_missing:
                needs_chunks = any(
                    MetricConfig.METRICS.get(m, {}).get('compute_method')
                    in ('calculate_perplexity', 'compute_vader_sentiment')
                    for m in s_missing
                )
                s_chunks = self.processor.chunk(stat_tokens) if needs_chunks else []
                if needs_chunks:
                    chunks = s_chunks
                print(
                    f"    -> Computing {len(s_missing)} stat metrics "
                    f"from {len(stat_tokens):,} tokens"
                )
                stat_metrics = self._metrics(stat_tokens, s_chunks, s_missing)

        new_metrics = {**stat_metrics, **centroid_metrics}
        if new_metrics:
            self._print_summary(new_metrics, 'computed')

        all_metrics = {**cached_metrics, **new_metrics}
        if is_low:
            for m in stat_names:
                all_metrics.pop(m, None)

        tokens_for_burrows = stat_tokens if stat_tokens else centroid_tokens

        rd = {
            'mean_emb': text_emb,
            'token_count': len(stat_tokens),
            'chunk_count': len(chunks),
            'file_count': n_files,
            '_burrows_rate_profile': self._burrows_profile(tokens_for_burrows),
            '_centroid_equals_stat': centroid_eq,
            '_centroid_only': is_low,
            **all_metrics,
        }

        if self.cache:
            self.cache.save(
                text, cache_filters, rd, vids_for_cache,
                len(stat_tokens), len(chunks),
                chunk_embs=None, mean_emb=text_emb, chunks=None,
                **ckw,
            )

        if is_low:
            print(f"  Complete: {text} ({len(stat_tokens):,} tokens, centroid metrics only)")
        else:
            print(
                f"  Complete: {text} ({len(stat_tokens):,} tokens, "
                f"{len(video_ids) or n_files} videos)"
            )

        return rd

    def _centroid_word(
        self, text: str, filters: Dict, ignore_ids: set, cache_filters: Dict
    ):
        result = self.processor.get_tokens(text, filters, ignore_ids)
        if not result:
            return None

        tokens, txt_dir, file_count, video_ids = result

        centroid_tokens = (
            tokens[-self.centroid_words:]
            if len(tokens) > self.centroid_words
            else tokens
        )
        stat_tokens = tokens
        centroid_eq = self._token_lists_are_identical(centroid_tokens, stat_tokens)
        self._centroid_equals_stat[text] = centroid_eq

        chunks = self.processor.chunk(centroid_tokens)
        if not chunks and centroid_tokens:
            chunks = [' '.join(centroid_tokens)]
        if not chunks:
            print(f"  Skipped: {text} (no chunks generated)")
            return None

        chunk_embs = self._encode(chunks)
        text_emb = np.mean(chunk_embs, axis=0).astype(np.float32)

        self.embeddings[text] = text_emb

        all_metrics = {}

        if not self.no_metrics:
            target = self._target_metrics(
                skip_centroid_duplicates=centroid_eq,
                skip_video_centroid_only=True,
            )
            stat_names = [m for m in target if not MetricConfig.is_centroid_metric(m)]
            centroid_names = [m for m in target if MetricConfig.is_centroid_metric(m)]

            if target:
                n_tokens = len(centroid_tokens)
                uniq = len(set(centroid_tokens))
                ttr = uniq / n_tokens if n_tokens else 0.0
                print(
                    f"    -> Stats corpus: {n_tokens:,} tokens, "
                    f"{uniq:,} unique (TTR={ttr:.4f}), "
                    f"centroid_words_cap={self.centroid_words:,}, "
                    f"chunks={len(chunks):,}"
                )
                print(
                    f"    -> Computing {len(target)} metrics "
                    f"from {len(centroid_tokens):,} tokens"
                )

                s_chunks: List[str] = []
                if any(
                    MetricConfig.METRICS.get(m, {}).get('compute_method')
                    in ('calculate_perplexity', 'compute_vader_sentiment')
                    for m in stat_names
                ):
                    s_chunks = self.processor.chunk(stat_tokens) or [' '.join(stat_tokens)]

                stat_metrics = (
                    self._metrics(stat_tokens, s_chunks, stat_names)
                    if stat_names else {}
                )

                c_chunks: List[str] = []
                if any(
                    MetricConfig.METRICS.get(m, {}).get('compute_method')
                    in ('calculate_perplexity', 'compute_vader_sentiment')
                    for m in centroid_names
                ):
                    c_chunks = self.processor.chunk(centroid_tokens) or [' '.join(centroid_tokens)]

                centroid_metrics = (
                    self._metrics(centroid_tokens, c_chunks, centroid_names)
                    if centroid_names else {}
                )

                all_metrics = {**stat_metrics, **centroid_metrics}
                self._print_summary(all_metrics, 'computed')

        rd = {
            'mean_emb': text_emb,
            'token_count': len(stat_tokens),
            'chunk_count': len(chunks),
            'file_count': file_count,
            'chunk_embs': chunk_embs,
            '_burrows_rate_profile': self._burrows_profile(centroid_tokens),
            '_centroid_equals_stat': centroid_eq,
            **all_metrics,
        }

        if self.cache:
            self.cache.save(
                text, cache_filters, rd, video_ids,
                len(stat_tokens), len(chunks),
                chunk_embs=chunk_embs, mean_emb=text_emb, chunks=chunks,
                enable_spacy=self.enable_spacy,
                needs_perplexity_model=self.processor.needs_perplexity_model,
                enable_ngram_entropy=self.enable_ngram_entropy,
                fast_mode=self.fast_mode,
            )

        print(f"  Complete: {text} ({len(stat_tokens):,} tokens, {len(chunks)} chunks)")
        return rd

    def reduce_dims(self, embeddings: np.ndarray, n_texts: int) -> np.ndarray:
        if self._scaler_cache is None:
            self._scaler_cache = StandardScaler()

        embeddings = np.ascontiguousarray(
            self._scaler_cache.fit_transform(embeddings), dtype=np.float32
        )

        if n_texts < 3:
            if embeddings.shape[1] >= 2:
                return embeddings[:, :2]
            pad = np.zeros((embeddings.shape[0], 2 - embeddings.shape[1]), dtype=np.float32)
            return np.hstack([embeddings, pad])

        if self.umap_neighbors is None:
            k = min(5 if n_texts <= 30 else 50, n_texts - 1)
        else:
            k = min(self.umap_neighbors, n_texts - 1)
        k = max(2, k)

        return umap.UMAP(
            n_components=2,
            metric='cosine',
            n_neighbors=k,
            min_dist=self.umap_min_dist,
            init='spectral',
            random_state=20,
            n_jobs=1,
            verbose=False,
            n_epochs=self.umap_epochs,
            low_memory=False,
        ).fit_transform(embeddings)

    def _cluster_default(
        self,
        embeddings: np.ndarray,
        idx: List[int],
        random_seed: int = 42,
        variant_index: int = 0,
    ) -> Tuple[List[Optional[int]], str]:
        labels_out: List[Optional[int]] = [None] * len(embeddings)
        if not idx:
            return labels_out, 'HDBSCAN'
        if len(idx) == 1:
            labels_out[idx[0]] = 0
            return labels_out, 'HDBSCAN'

        sub = embeddings[idx]

        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = sub / norms

        min_size = max(2, min(20, len(idx) // 20 or 2))
        min_samples = max(1, min_size // 2)

        inp = normalized
        if variant_index > 0:
            try:
                rng = np.random.default_rng(random_seed + variant_index * 1009)
                std = float(np.std(normalized))
                jitter = rng.normal(
                    0.0, max(1e-6, std * 1e-2), size=normalized.shape
                ).astype(np.float32)
                inp = normalized + jitter
            except Exception:
                inp = normalized

        labels: Optional[np.ndarray] = None
        algo = 'HDBSCAN'
        if SklearnHDBSCAN is not None:
            try:
                labels = SklearnHDBSCAN(
                    min_cluster_size=min_size,
                    min_samples=min_samples,
                    metric='euclidean',
                ).fit_predict(inp)
            except Exception as exc:
                print(f"    Warning: HDBSCAN failed ({exc})")
                labels = None

        if labels is None or np.all(labels == -1):
            algo = 'KMeans'
            fallback = max(2, min(14, int(np.sqrt(len(idx)))))
            print(
                f"    Note: HDBSCAN returned all-noise or failed; "
                f"falling back to KMeans (variant {variant_index}, seed {random_seed})"
            )
            labels = KMeans(
                n_clusters=fallback,
                random_state=random_seed,
                n_init=10,
            ).fit_predict(inp)

        for k, ti in enumerate(idx):
            labels_out[ti] = int(labels[k])
        return labels_out, algo

    def _cluster_leiden(
        self,
        embeddings: np.ndarray,
        idx: List[int],
        resolution: float = 1.0,
        random_seed: int = 42,
    ) -> Tuple[List[Optional[int]], str]:
        labels_out: List[Optional[int]] = [None] * len(embeddings)
        if not idx:
            return labels_out, 'HDBSCAN'
        if len(idx) == 1:
            labels_out[idx[0]] = 0
            return labels_out, 'HDBSCAN'

        if not LEIDEN_AVAILABLE:
            print("    Warning: leidenalg/igraph not available — falling back to HDBSCAN")
            return self._cluster_default(embeddings, idx, random_seed=random_seed)

        try:
            from sklearn.decomposition import PCA
            from sklearn.neighbors import kneighbors_graph
        except ImportError as exc:
            print(f"    Warning: sklearn dependency missing for Leiden ({exc}) — falling back to HDBSCAN")
            return self._cluster_default(embeddings, idx, random_seed=random_seed)

        X = embeddings[idx].astype(np.float32)
        n = X.shape[0]

        if self.leiden_whiten and X.shape[1] > 1:
            k = min(256, X.shape[1] - 1, n - 1)
            if k >= 2:
                try:
                    X = PCA(
                        n_components=k, whiten=True, random_state=random_seed,
                    ).fit_transform(X).astype(np.float32)
                except Exception as exc:
                    print(f"    Warning: PCA whitening failed ({exc}); using raw embeddings")

        k_eff = max(2, min(int(self.leiden_k_neighbors), n - 1))
        A = None
        for metric in ('cosine', 'euclidean'):
            try:
                A = kneighbors_graph(
                    X, k_eff, mode='connectivity',
                    metric=metric, include_self=False,
                )
                A = A.maximum(A.T)
                break
            except Exception as exc:
                print(f"    Warning: k-NN graph with metric={metric} failed ({exc})")
                A = None

        if A is None:
            print("    Warning: could not build k-NN graph — falling back to HDBSCAN")
            return self._cluster_default(embeddings, idx, random_seed=random_seed)

        try:
            sources, targets = A.nonzero()
            edges = [(int(s), int(t)) for s, t in zip(sources, targets) if s < t]
            if not edges:
                print("    Warning: k-NN graph produced no edges — falling back to HDBSCAN")
                return self._cluster_default(embeddings, idx, random_seed=random_seed)
            g = ig.Graph(n=n, edges=edges, directed=False)
            partition = la.find_partition(
                g,
                la.RBConfigurationVertexPartition,
                resolution_parameter=float(resolution),
                seed=random_seed,
            )
            labels = np.asarray(partition.membership, dtype=int)
        except Exception as exc:
            print(f"    Warning: Leiden failed ({exc}) — falling back to HDBSCAN")
            return self._cluster_default(embeddings, idx, random_seed=random_seed)

        for k, ti in enumerate(idx):
            labels_out[ti] = int(labels[k])
        return labels_out, 'Leiden'

    def _cluster_variants(
        self,
        embeddings: np.ndarray,
        idx: List[int],
        variant_count: int = 11,
        base_seed: int = 42,
    ) -> Dict[str, List[Optional[int]]]:
        if self.cluster_method == 'leiden':
            return self._cluster_variants_leiden(
                embeddings, idx, variant_count=variant_count, base_seed=base_seed,
            )

        total = max(1, int(variant_count))
        variants: Dict[str, List[Optional[int]]] = {}
        labels_map: Dict[str, str] = {}
        for vi in range(total):
            key = 'cluster' if vi == 0 else f'cluster_{vi + 1}'
            labels, algo = self._cluster_default(
                embeddings, idx,
                random_seed=base_seed + vi, variant_index=vi,
            )
            variants[key] = labels
            labels_map[key] = (
                f'{algo} (default)' if vi == 0
                else f'{algo} variant {vi + 1}'
            )
        self.cluster_metric_labels = labels_map
        return variants

    def _cluster_variants_leiden(
        self,
        embeddings: np.ndarray,
        idx: List[int],
        variant_count: int = 11,
        base_seed: int = 42,
    ) -> Dict[str, List[Optional[int]]]:
        total = max(1, int(variant_count))
        if total == 1:
            resolutions = [float(self.leiden_resolution_min)]
        else:
            resolutions = np.linspace(
                self.leiden_resolution_min,
                self.leiden_resolution_max,
                total,
            ).tolist()

        variants: Dict[str, List[Optional[int]]] = {}
        labels_map: Dict[str, str] = {}
        for i, res in enumerate(resolutions):
            key = 'cluster' if i == 0 else f'cluster_{i + 1}'
            labels, algo = self._cluster_leiden(
                embeddings, idx, resolution=float(res), random_seed=base_seed,
            )
            variants[key] = labels
            if algo == 'Leiden':
                labels_map[key] = f'Leiden r={res:.2f}'
            else:
                labels_map[key] = f'{algo} r={res:.2f} (Leiden fallback)'
        self.cluster_metric_labels = labels_map
        return variants

    def export_csv(self, texts: List[str], output: str, n: int = 100):
        if self.similarity_matrix is None:
            return

        print(f"\nExporting to {output}...")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Text']
            for i in range(1, n + 1):
                header.extend([f'Closest {i}', f'Closest {i} Similarity'])
            for i in range(1, n + 1):
                header.extend([f'Farthest {i}', f'Farthest {i} Similarity'])
            if self.compute_fighting_words:
                header.extend(['Fighting Words Peak Unigram', 'Fighting Words Floor Unigram'])
            writer.writerow(header)

            for i, text in enumerate(texts):
                sims = self.similarity_matrix[i].copy()
                sorted_idx = np.argsort(sims)[::-1]
                sorted_idx = [k for k in sorted_idx if k != i]
                avail = min(n, len(sorted_idx))
                closest = [(texts[k], sims[k]) for k in sorted_idx[:avail]]
                farthest = [(texts[k], sims[k]) for k in sorted_idx[-avail:][::-1]]

                row = [text]
                for j in range(n):
                    if j < len(closest):
                        row.extend([closest[j][0], _format_trimmed_decimal(closest[j][1], 4)])
                    else:
                        row.extend(['', ''])
                for j in range(n):
                    if j < len(farthest):
                        row.extend([farthest[j][0], _format_trimmed_decimal(farthest[j][1], 4)])
                    else:
                        row.extend(['', ''])
                if self.compute_fighting_words:
                    row.extend([
                        self._fighting_words_peak_unigrams.get(text, ''),
                        self._fighting_words_floor_unigrams.get(text, '')
                    ])
                writer.writerow(row)

        print(f"  Exported: {output_path.absolute()}")

    def export_metrics_csv(
        self,
        texts: List[str],
        metrics_data: Dict[str, np.ndarray],
        output: str,
    ):
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metric_names = [
            name for name in MetricConfig.get_metric_names()
            if name in metrics_data and metrics_data[name] is not None
        ]
        channel_mode = not self.processor.external_documents

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Text', *metric_names])
            for index, text in enumerate(texts):
                label = text
                if channel_mode and not METRICS_CSV_FULL_CHANNEL_NAMES:
                    label = text[:5]
                row = [label]
                for metric_name in metric_names:
                    value = metrics_data[metric_name][index]
                    row.append(
                        '' if value is None or np.isnan(value)
                        else _format_trimmed_decimal(float(value), 6)
                    )
                writer.writerow(row)

        print(f"  Exported metrics: {output_path.absolute()}")

    def _cluster_label(self, key: str) -> str:
        if key in self.cluster_metric_labels:
            return self.cluster_metric_labels[key]
        if key in self.burrows_cluster_metric_labels:
            return self.burrows_cluster_metric_labels[key]
        if key == 'cluster':
            return 'Cluster (default)'
        if key.startswith('cluster_'):
            idx = key.split('_', 1)[1]
            return f'Cluster {idx}'
        if key == 'burrows_cluster':
            return 'Burrows Cluster (default)'
        if key.startswith('burrows_cluster_'):
            idx = key.rsplit('_', 1)[-1]
            return f'Burrows Cluster {idx}'
        return key

    def visualize(
        self,
        reduced: np.ndarray,
        texts: List[str],
        metrics_data: Dict[str, np.ndarray],
        active_metric: Optional[str] = None,
        output_html: str = ".html",
        low_token_texts: Optional[set] = None,
        focus_text: Optional[str] = None,
        semantic_embeddings: Optional[List[List[float]]] = None,
        burrows_similarity: Optional[List[List[float]]] = None,
        fighting_words_unigrams: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        burrows_reduced: Optional[List[List[float]]] = None,
        burrows_function_words: Optional[List[str]] = None,
        burrows_word_frequencies: Optional[Dict[str, List[Optional[float]]]] = None,
        capital_cities: Optional[Dict[str, Any]] = None,
    ):
        low_token_texts = low_token_texts or set()
        if focus_text is None:
            focus_text = getattr(self, 'active_focus_text', '') or ''
        semantic_embeddings = semantic_embeddings or []
        burrows_similarity = burrows_similarity or []
        fighting_words_unigrams = fighting_words_unigrams or {}
        burrows_reduced = burrows_reduced or []
        burrows_function_words = burrows_function_words or []
        burrows_word_frequencies = burrows_word_frequencies or {}
        capital_cities = capital_cities or {}

        hover_texts, hover_panel_palettes, hover_text_palettes = (
            self._build_hover_texts(texts, metrics_data, reduced, low_token_texts)
        )

        cluster_metric_keys = [
            key for key in metrics_data.keys()
            if key == 'cluster' or key.startswith('cluster_')
            or key == 'burrows_cluster' or key.startswith('burrows_cluster_')
        ]
        cluster_metric_keys.sort(
            key=lambda key: (
                0 if key == 'cluster' else
                1 if key.startswith('cluster_') else
                2 if key == 'burrows_cluster' else
                3
            ,)
        )

        cluster_keys = [
            k for k in metrics_data.keys()
            if k == 'cluster' or k.startswith('cluster_')
            or k == 'burrows_cluster' or k.startswith('burrows_cluster_')
        ]
        cluster_keys.sort(key=lambda k: (
            0 if k == 'cluster' else
            1 if k.startswith('cluster_') else
            2 if k == 'burrows_cluster' else 3
        ))

        if (
            active_metric
            and active_metric in MetricConfig.METRICS
            and active_metric in metrics_data
            and metrics_data[active_metric] is not None
        ):
            cfg_ = MetricConfig.METRICS[active_metric]
            values = metrics_data[active_metric]
            color_label = cfg_.get('name', active_metric)
            color_scale = cfg_.get('colorscale', UNIFIED_COLORSCALE)
            suffix = cfg_.get('title_suffix', color_label)
            title = f"Text Semantic Map colored by {suffix}"
            color_data = [float(values[i]) for i in range(len(texts))]
        else:
            default_key = cluster_keys[0] if cluster_keys else 'cluster'
            clusters = metrics_data.get(default_key)
            color_data = (
                [float(clusters[i]) for i in range(len(texts))]
                if clusters is not None
                else [float('nan')] * len(texts)
            )
            color_label = 'Cluster'
            color_scale = 'turbo'
            title = 'Mapping Texts with Semantics :O'
            active_metric = default_key

        is_cluster = (
            active_metric == 'cluster'
            or active_metric.startswith('cluster_')
            or active_metric == 'burrows_cluster'
            or active_metric.startswith('burrows_cluster_')
        )
        if is_cluster:
            kept = [v for v in color_data if not np.isnan(v)]
            lo = np.nanmin(kept) if kept else 0.0
            hi = np.nanmax(kept) if kept else 1.0
        else:
            kept = [v for v in color_data if not np.isnan(v)]
            lo = np.nanmin(kept) if kept else 0.0
            hi = np.nanmax(kept) if kept else 1.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode='markers+text',
            text=texts,
            textposition="middle center",
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=15,
                color=color_data,
                colorscale=color_scale,
                showscale=True,
                cmin=lo,
                cmax=hi,
                colorbar=dict(title=color_label),
                line=dict(width=1, color='black'),
            ),
            textfont=dict(size=7, color='black'),
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            hovermode='closest',
            showlegend=False,
            height=800,
            width=1200,
            font=dict(family='Arial, sans-serif'),
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        output_path = Path(output_html).absolute()

        plotly_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        plotly_html = re.sub(r'\s+integrity="[^"]*"', '', plotly_html)
        plotly_html = re.sub(r'\s+crossorigin="[^"]*"', '', plotly_html)

        metrics_json: Dict[str, Any] = {}
        titles_json: Dict[str, str] = {}

        for ck in cluster_keys:
            values = metrics_data.get(ck)
            if values is None:
                metrics_json[ck] = [None for _ in range(len(texts))]
            else:
                labels = [None] * len(texts)
                for i in range(len(texts)):
                    if not np.isnan(values[i]):
                        labels[i] = int(values[i])
                metrics_json[ck] = labels
            titles_json[ck] = f'Text Semantic Clusters ({self._cluster_label(ck)})'

        names_dict, scales_dict, keys_list = self._metric_meta(cluster_keys)

        for name, values in metrics_data.items():
            if values is None:
                continue
            if (
                name == 'cluster'
                or name.startswith('cluster_')
                or name == 'burrows_cluster'
                or name.startswith('burrows_cluster_')
            ):
                continue
            metrics_json[name] = [float(values[i]) for i in range(len(texts))]
            cfg_ = MetricConfig.METRICS.get(name, {})
            titles_json[name] = (
                f"Text Semantic Clusters colored by "
                f"{cfg_.get('title_suffix', name)}"
            )

        metric_names_dict = {}
        metric_colorscale_dict = {}
        metric_keys_list = []

        for cluster_metric_key in cluster_metric_keys:
            metric_names_dict[cluster_metric_key] = self._cluster_label(cluster_metric_key)
            metric_colorscale_dict[cluster_metric_key] = 'Turbo'
            metric_keys_list.append(cluster_metric_key)

        non_contextual_metric_items = [
            (metric_name, MetricConfig.METRICS[metric_name])
            for metric_name in MetricConfig.get_metric_names()
            if MetricConfig.METRICS[metric_name].get('category') != 'Contextual Descriptions'
        ]

        for metric_name, metric_config in non_contextual_metric_items:
            metric_names_dict[metric_name] = metric_config.get('name', metric_name)
            metric_colorscale_dict[metric_name] = metric_config.get(
                'colorscale', UNIFIED_COLORSCALE
            )
            metric_keys_list.append(metric_name)

        metrics_json_str = safe_json_dumps(metrics_json)
        titles_json_str = safe_json_dumps(titles_json)
        texts_json = safe_json_dumps(texts)

        template_path = Path(__file__).parent / "output_files" / "template.html"
        script_path = Path(__file__).parent / "output_files" / "script.js"

        template = template_path.read_text(encoding="utf-8")
        script = script_path.read_text(encoding="utf-8")

        values = {
            "__PLOTLY_HTML__": plotly_html,
            "__FOCUS_TEXT__": json.dumps(focus_text),
            "__METRICS_JSON__": metrics_json_str,
            "__TITLES_JSON__": titles_json_str,
            "__TEXTS_JSON__": texts_json,
            "__HOVER_PANEL_PALETTES_JSON__": safe_json_dumps(hover_panel_palettes),
            "__HOVER_TEXT_PALETTES_JSON__": safe_json_dumps(hover_text_palettes),
            "__METRIC_NAMES_JSON__": json.dumps(metric_names_dict),
            "__METRIC_COLORScales_JSON__": json.dumps(metric_colorscale_dict),
            "__METRIC_KEYS_JSON__": json.dumps(metric_keys_list),
            "__SEMANTIC_EMBEDDINGS_JSON__": safe_json_dumps(semantic_embeddings),
            "__BURROWS_SIMILARITY_JSON__": safe_json_dumps(burrows_similarity),
            "__LOW_TOKEN_TEXTS_JSON__": json.dumps(
                [tx for tx in texts if tx in low_token_texts]
            ),
            "__FIGHTING_WORDS_UNIGRAMS_JSON__": safe_json_dumps(fighting_words_unigrams),
            "__ACTIVE_METRIC_JSON__": json.dumps(active_metric or "cluster"),
            "__BURROWS_REDUCED_JSON__": safe_json_dumps(burrows_reduced),
            "__BURROWS_FUNCTION_WORDS_JSON__": json.dumps(list(burrows_function_words)),
            "__BURROWS_WORD_FREQUENCIES_JSON__": safe_json_dumps(burrows_word_frequencies),
            "__CAPITAL_CITIES_JSON__": safe_json_dumps(capital_cities),
        }

        custom_script = script
        for k, v in values.items():
            custom_script = custom_script.replace(k, v)

        output_dir = Path(output_html).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        script_out = output_dir / "script.js"

        with open(script_out, 'w', encoding='utf-8') as f:
            f.write(custom_script)

        custom = template
        for k, v in values.items():
            custom = custom.replace(k, v)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(custom)

        print(f"\nExported interactive visualization to: {output_path}")
        print("Opening in browser...")
        time.sleep(0.1)
        try:
            webbrowser.open(f'file://{output_path}')
        except Exception as e:
            print(f"Could not automatically open browser: {e}")
            print(f"Please manually open: {output_path}")

    def _build_hover_texts(self, texts, metrics_data, reduced, low_token_texts):
        exclude_from_hover = {
            'latinate_word_ratio', 'coordinate_clause_ratio', 'subordinate_clause_ratio',
            'imperative_exclamation_count', 'academic_word_density',
            'article_count_raw', 'conjunction_count_raw', 'vulnerability',
            'family_exclamations', 'pronoun_article_ratio',
            'deictic_spatial_temporal', 'elaboration_explanation_ratio',
            'narration_continuation_ratio',
            'insult', 'bodily_humor', 'dis_ratio',
            'word_burstiness', 'semantic_disparity', 'guppa', 'pronoun_switching',
            'compression', 'moving_lzma_cr', 'normalized_compression_ratio',
            'window_normalized_lzma_cr',
            'vader_positivity', 'vader_volatility',
            'vader_positivity_centroid', 'vader_volatility_centroid',
            'fighting_words_peak_z1', 'fighting_words_floor_z1000',
            'fighting_words_peak_z1_centroid', 'fighting_words_floor_z1000_centroid',
        }
        exclude_from_hover.update({
            metric_name for metric_name in MetricConfig.METRICS
            if metric_name.endswith('_count')
            or metric_name.endswith('_density')
            or metric_name.startswith('tomotopy_topic_')
        })

        def wrap_long_lines(text: str, max_length: int = 120) -> str:
            lines = text.split('<br>')
            wrapped_lines = []
            for line in lines:
                if len(line) > max_length:
                    words = line.split()
                    current_line = []
                    current_length = 0
                    for word in words:
                        word_len = len(word) + (1 if current_line else 0)
                        if current_length + word_len > max_length and current_line:
                            wrapped_lines.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                        else:
                            current_line.append(word)
                            current_length += word_len

                    if current_line:
                        wrapped_lines.append(' '.join(current_line))
                else:
                    wrapped_lines.append(line)

            return '<br>'.join(wrapped_lines)

        def read_palette(text: str, filenames: tuple) -> list:
            channel_dir = (
                self.processor.base_dir
                / "data/input"
                / sanitize_filename(text)
            )
            palette_path = next(
                (channel_dir / filename for filename in filenames
                 if (channel_dir / filename).exists()),
                None,
            )
            if palette_path is None:
                return []

            try:
                colors = [
                    line.strip()
                    for line in palette_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeError) as exc:
                print(f"Warning: unable to read {palette_path}: {exc}")
                return []

            if not 2 <= len(colors) <= 4 or any(
                not _HEX_COLOR_RE.fullmatch(c) for c in colors
            ):
                print(
                    f"Warning: {palette_path} must contain 2-4 hex colours; ignoring."
                )
                return []
            return colors

        hover_texts = []
        hover_panel_palettes = []
        hover_text_palettes = []

        for i, text in enumerate(texts):
            tag = ''
            try:
                tag_path = (
                    self.processor.base_dir
                    / "data/input"
                    / sanitize_filename(text)
                    / "tag.txt"
                )
                if tag_path.exists():
                    tag = tag_path.read_text(encoding='utf-8').strip()
            except (OSError, UnicodeError):
                pass

            col = f"<b>{text}</b><br>"
            if tag:
                col += f"{escape_html_attribute(tag)}<br>"
            col += (
                f"X: {_format_trimmed_decimal(reduced[i, 0], 4)}<br>"
                f"Y: {_format_trimmed_decimal(reduced[i, 1], 4)}<br>"
            )

            for name, values in metrics_data.items():
                if (
                    values is None
                    or name in exclude_from_hover
                    or name.startswith('cluster')
                    or name.startswith('burrows_cluster')
                ):
                    continue
                if text in low_token_texts and not MetricConfig.is_centroid_duplicate(name):
                    continue
                v = values[i]
                if v is None or np.isnan(v):
                    continue
                display = MetricConfig.METRICS.get(name, {}).get('name', name)
                line = self._fmt_hover(name, display, v)
                if line:
                    col += line

            hover_texts.append(wrap_long_lines(col))
            hover_panel_palettes.append(
                read_palette(text, ("hover_panel.txt", "hover_gradient.txt"))
            )
            hover_text_palettes.append(
                read_palette(text, ("hover_text.txt",))
            )

        return hover_texts, hover_panel_palettes, hover_text_palettes

    def _fmt_hover(self, name: str, display: str, value) -> Optional[str]:
        if name in (
            'MTLD', 'TTR', 'MATTR', 'yules_k',
            'hapax_ratio',
            'avg_word_length', 'semantic_disparity',
            'burrows_cosine_disagreement',
            'MTLD_centroid', 'MATTR_centroid',
            'ngram_entropy_2', 'ngram_entropy_3',
        ):
            return f"{display}: {_format_trimmed_decimal(value, 4)}<br>"
        if name in ('perplexity', 'tfidf_distinctiveness'):
            return f"{display}: {value:.2f}<br>"
        if name == 'flesch_kincaid':
            return f"{display}: {value:.1f}<br>"
        if isinstance(value, float) and not float(value).is_integer():
            return f"{display}: {_format_trimmed_decimal(value, 4)}<br>"
        return f"{display}: {int(value)}<br>"

    def _metric_meta(
        self, cluster_keys: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, Any], List[str]]:
        names: Dict[str, str] = {}
        scales: Dict[str, Any] = {}
        keys: List[str] = []

        for k in cluster_keys:
            names[k] = self._cluster_label(k)
            scales[k] = 'Turbo'
            keys.append(k)

        for name in MetricConfig.get_metric_names():
            cfg_ = MetricConfig.METRICS[name]
            if cfg_.get('category') == 'Contextual Descriptions':
                continue
            names[name] = cfg_.get('name', name)
            scales[name] = cfg_.get('colorscale', UNIFIED_COLORSCALE)
            keys.append(name)

        return names, scales, keys

    def _build_burrows_word_frequencies(
        self,
        successful: List[str],
        style_profiles: Dict[str, np.ndarray],
    ) -> Dict[str, List[Optional[float]]]:
        words = list(self.BURROWS_FUNCTION_WORDS)
        n_words = len(words)
        freqs: Dict[str, List[Optional[float]]] = {w: [] for w in words}

        for tx in successful:
            profile = style_profiles.get(tx)
            for w_idx, w in enumerate(words):
                if profile is None or w_idx >= n_words:
                    freqs[w].append(None)
                else:
                    freqs[w].append(float(profile[w_idx]))

        return freqs

    def _build_burrows_layout(
        self,
        successful: List[str],
        style_profiles: Dict[str, np.ndarray],
    ) -> Optional[List[List[float]]]:
        if len(style_profiles) < 2:
            return None

        distances, valid = self._burrows_dist(style_profiles, successful)
        if distances is None or not valid:
            return None

        full = self._expand_burrows_distance_matrix(
            len(successful), distances, valid
        )

        try:
            reduced = self._burrows_reduce_dimensions(full, len(successful))
        except Exception as exc:
            print(f"  Warning: Burrows UMAP failed ({exc}); experimental tab disabled")
            return None

        return [
            [float(reduced[i, 0]), float(reduced[i, 1])]
            for i in range(len(successful))
        ]

    def _cluster_burrows_variants(
        self,
        distances: Optional[np.ndarray],
        valid: List[int],
        variant_count: int,
        base_seed: int = 42,
    ) -> Dict[str, List[Optional[int]]]:
        labels_by_key: Dict[str, List[Optional[int]]] = {}
        self.burrows_cluster_metric_labels = {}
        n = len(self.successful)
        if distances is None or len(valid) < 2:
            return labels_by_key

        compact = np.asarray(distances, dtype=np.float64)
        compact = np.nan_to_num(compact, nan=0.0, posinf=1e6, neginf=0.0)
        compact = np.maximum((compact + compact.T) * 0.5, 0.0)
        np.fill_diagonal(compact, 0.0)
        valid_count = len(valid)

        def expand(labels: np.ndarray) -> List[Optional[int]]:
            result: List[Optional[int]] = [None] * n
            for index, text_index in enumerate(valid):
                result[text_index] = int(labels[index])
            return result

        if self.cluster_method != 'leiden':
            # ---- HDBSCAN branch ----
            # Per-variant jitter on the distance matrix ensures the 5 variants
            # are actually different cluster assignments (HDBSCAN is otherwise
            # deterministic and would produce identical labels every time).
            min_cluster_size = max(2, min(20, valid_count // 20 or 2))
            min_samples = max(1, min_cluster_size // 2)
            pos = compact[compact > 0]
            base_scale = float(np.std(pos)) * 0.02 if pos.size else 1e-6
            for variant_index in range(max(1, int(variant_count))):
                key = 'burrows_cluster' if variant_index == 0 else f'burrows_cluster_{variant_index + 1}'
                jittered = compact
                if variant_index > 0:
                    rng = np.random.default_rng(base_seed + variant_index * 1009)
                    jit = rng.normal(0.0, max(1e-9, base_scale), size=compact.shape)
                    jit = (jit + jit.T) * 0.5
                    np.fill_diagonal(jit, 0.0)
                    jittered = np.maximum(compact + jit, 0.0)
                    np.fill_diagonal(jittered, 0.0)

                labels = None
                if SklearnHDBSCAN is not None:
                    try:
                        labels = SklearnHDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric='precomputed',
                        ).fit_predict(jittered)
                    except Exception as exc:
                        print(f"    Warning: Burrows HDBSCAN failed ({exc})")
                if labels is None or np.all(labels == -1):
                    labels = KMeans(
                        n_clusters=max(2, min(14, int(np.sqrt(valid_count)))),
                        random_state=base_seed + variant_index,
                        n_init=10,
                    ).fit_predict(jittered)
                labels_by_key[key] = expand(np.asarray(labels))
            return labels_by_key

        # ---- Leiden branch ----
        if not LEIDEN_AVAILABLE:
            return labels_by_key

        try:
            from sklearn.neighbors import kneighbors_graph
        except ImportError:
            return labels_by_key

        k_eff = max(2, min(int(self.leiden_k_neighbors), valid_count - 1))
        try:
            graph = kneighbors_graph(
                compact, k_eff, mode='connectivity',
                metric='precomputed', include_self=False,
            )
            graph = graph.maximum(graph.T)
            sources, targets = graph.nonzero()
            edges = [(int(s), int(t)) for s, t in zip(sources, targets) if s < t]
            if not edges:
                return labels_by_key
        except Exception as exc:
            print(f"    Warning: Burrows Leiden graph failed ({exc})")
            return labels_by_key

        resolutions = (
            [1.0] if variant_count <= 1 else np.linspace(
                self.leiden_resolution_min,
                self.leiden_resolution_max,
                max(1, int(variant_count)),
            ).tolist()
        )
        for index, resolution in enumerate(resolutions):
            key = 'burrows_cluster' if index == 0 else f'burrows_cluster_{index + 1}'
            try:
                partition = la.find_partition(
                    ig.Graph(n=valid_count, edges=edges, directed=False),
                    la.RBConfigurationVertexPartition,
                    resolution_parameter=float(resolution),
                    seed=base_seed,
                )
                labels_by_key[key] = expand(np.asarray(partition.membership, dtype=int))
                self.burrows_cluster_metric_labels[key] = (
                    f'Burrows Leiden r={float(resolution):.2f}'
                )
            except Exception as exc:
                print(f"    Warning: Burrows Leiden failed ({exc})")
                break
        return labels_by_key

    def run(
        self,
        texts: List[str],
        output_csv: str,
        output_html: str,
        ignore_ids: Optional[set] = None,
        capital_cities: bool = False,
        **filter_kwargs,
    ):
        if ignore_ids is None:
            ignore_ids = set()

        texts = sorted(texts, key=str.lower)
        print(f"Processing {len(texts)} texts...")

        # ---- Fighting-words precompute (item 1) ----
        # Must happen BEFORE the per-text processing loop, so that the
        # corpus-wide unigram counts are available. The harvest block further
        # down then reads straight from the _fw_*_cache attributes.
        if self.compute_fighting_words and not self.no_metrics:
            self._reset_fw()
            self._precompute_fw(texts, filter_kwargs, ignore_ids)

        successful: List[str] = []
        low_token_texts = set()
        style_profiles: Dict[str, np.ndarray] = {}
        metric_names = self._target_metrics(include_global_metrics=True)
        metrics_storage: Dict[str, Dict[str, Any]] = {n: {} for n in metric_names}

        for tx in texts:
            self._current_text_name = tx
            result = self.process_text(tx, ignore_ids=ignore_ids, **filter_kwargs)
            if result is None:
                continue

            if result.get('low_token_text', False):
                low_token_texts.add(tx)
            style_profile = result.get('_burrows_rate_profile')
            if style_profile is None:
                fallback_filters = dict(filter_kwargs)
                fallback_filters['ignore_ids'] = ignore_ids
                style_profile = self._load_burrows_profile(tx, fallback_filters)

            if style_profile is not None:
                style_profiles[tx] = style_profile

            if not result.get('low_token_text', False):
                for name in metric_names:
                    if name in result:
                        metrics_storage[name][tx] = result[name]
            else:
                for name in metric_names:
                    if name in result and MetricConfig.is_centroid_duplicate(name):
                        metrics_storage[name][tx] = result[name]

            successful.append(tx)

        if not successful:
            print("\nError: No texts successfully processed")
            return

        self.successful = successful

        print(f"\nSuccessfully processed {len(successful)}/{len(texts)} texts")

        if self.tomotopy_num_topics > 0 and successful:
            try:
                self._train_tomotopy_topics(successful, ignore_ids, filter_kwargs)
            except Exception as exc:
                print(f"  Warning: Tomotopy topic modeling failed ({exc}); continuing without topics")

        # ---- Fighting-words harvest (item 1) ----
        # Read the precomputed caches, filtered to successful texts. This is
        # what populates metrics_storage for the 4 fighting-word metrics.
        if self.compute_fighting_words and not self.no_metrics:
            for name, cache in (
                (FIGHTING_WORDS_PEAK_METRIC, self._fw_peak_cache),
                (FIGHTING_WORDS_FLOOR_METRIC, self._fw_floor_cache),
                (FIGHTING_WORDS_PEAK_CENTROID_METRIC, self._fw_peak_centroid_cache),
                (FIGHTING_WORDS_FLOOR_CENTROID_METRIC, self._fw_floor_centroid_cache),
            ):
                slot = metrics_storage.setdefault(name, {})
                if name not in metric_names:
                    metric_names.append(name)
                for tx in successful:
                    slot[tx] = cache.get(tx)

            n_peak = sum(1 for v in self._fw_peak_cache.values() if v is not None)
            n_floor = sum(1 for v in self._fw_floor_cache.values() if v is not None)
            print(
                f"  Fighting-words metrics: peak {n_peak}/{len(texts)}, "
                f"floor {n_floor}/{len(texts)} "
                f"(rank peak={FIGHTING_WORDS_PEAK_RANK}, "
                f"rank floor={FIGHTING_WORDS_FLOOR_RANK}, "
                f"min_count={FIGHTING_WORDS_FLOOR_MIN_COUNT})"
            )

        emb_matrix = np.vstack(
            [self.embeddings[tx] for tx in successful]
        ).astype(np.float32)

        cluster_idx = list(range(len(successful)))
        requested = filter_kwargs.get('cluster_variants')
        if requested is None:
            variant_count = 1 if len(successful) > 2000 else 11
        else:
            variant_count = max(1, int(requested))

        if self.cluster_method == 'leiden':
            print(
                f"  Clustering method: Leiden | k={self.leiden_k_neighbors} | "
                f"resolutions={self.leiden_resolution_min:.2f}-"
                f"{self.leiden_resolution_max:.2f} | "
                f"variants={variant_count} | whiten={self.leiden_whiten}"
            )
        else:
            print(f"  Clustering method: HDBSCAN | variants={variant_count}")

        cluster_labels = self._cluster_variants(
            emb_matrix, cluster_idx, variant_count=variant_count, base_seed=42,
        )
        self.cluster_variant_labels = cluster_labels

        metrics_data: Dict[str, Optional[np.ndarray]] = {}
        for name in metric_names:
            values = [metrics_storage[name].get(tx) for tx in successful]
            if any(v is not None for v in values):
                metrics_data[name] = np.array(
                    [v if v is not None else np.nan for v in values],
                    dtype=float,
                )
            else:
                metrics_data[name] = None

        for key, labels in cluster_labels.items():
            metrics_data[key] = np.array(
                [float(l) if l is not None else np.nan for l in labels],
                dtype=float,
            )

        self.similarity_matrix = cosine_similarity(emb_matrix)

        # --- distance from center (semantic) ---
        # Mean cosine distance of each text from the corpus centroid embedding.
        semantic_center = emb_matrix.mean(axis=0, keepdims=True)
        semantic_distances = 1.0 - cosine_similarity(
            emb_matrix, semantic_center
        ).flatten()
        metrics_data['distance_from_center_semantic'] = semantic_distances
        MetricConfig.METRICS['distance_from_center_semantic'] = {
            'name': 'Distance from Center (Semantic)',
            'title_suffix': 'Semantic distance from corpus centroid',
            'category': 'Centroid',
        }

        # --- distance from center (burrows) ---
        # Mean absolute deviation of each text's function-word rate profile
        # from the corpus mean profile.
        if style_profiles:
            style_matrix = np.vstack(
                [style_profiles[tx] for tx in successful if tx in style_profiles]
            )
            style_center = style_matrix.mean(axis=0, keepdims=True)
            burrows_center_distances: List[float] = []
            for tx in successful:
                if tx in style_profiles:
                    d = float(np.mean(np.abs(style_profiles[tx] - style_center)))
                else:
                    d = np.nan
                burrows_center_distances.append(d)
            metrics_data['distance_from_center_burrows'] = np.array(
                burrows_center_distances, dtype=float
            )
            MetricConfig.METRICS['distance_from_center_burrows'] = {
                'name': 'Distance from Center (Burrows)',
                'title_suffix': 'Burrows distance from corpus centroid',
                'category': 'Centroid',
            }

        burrows_distances, burrows_valid = self._burrows_dist(
            style_profiles, successful
        )

        if not self.no_metrics:
            disagreement = self._burrows_scores(
                burrows_distances, burrows_valid, emb_matrix, successful
            )
            metrics_data['burrows_cosine_disagreement'] = np.array(
                [disagreement.get(t, float('nan')) for t in successful],
                dtype=float,
            )

        normalized_embeddings = emb_matrix / np.maximum(
            np.linalg.norm(emb_matrix, axis=1, keepdims=True), 1e-12
        )
        self.export_csv(successful, output_csv)

        if self.tomotopy_topic_proportions:
            attached = 0
            for tid in range(self.tomotopy_num_topics):
                key = f'tomotopy_topic_{tid}'
                values = np.array([
                    self.tomotopy_topic_proportions.get(tx, {}).get(tid, np.nan)
                    for tx in successful
                ], dtype=float)
                if np.all(np.isnan(values)):
                    continue
                metrics_data[key] = values
                label_words = self.tomotopy_topic_labels.get(tid, '')
                MetricConfig.METRICS[key] = {
                    'name': f'Topic {tid}'
                            + (f' ({label_words})' if label_words else ''),
                    'title_suffix': f'LDA topic {tid} proportion',
                    'compute_method': '_compute_tomotopy_topic',
                    'category': 'Topic Modeling',
                }
                attached += 1
            if attached:
                print(f"  Attached {attached} Tomotopy topic metrics to visualization")

        if filter_kwargs.get('get_metrics_csv', False):
            out = Path(output_csv)
            out = out.with_name(f'{out.stem}_metrics{out.suffix}')
            self.export_metrics_csv(successful, metrics_data, str(out))

        n_success = len(successful)
        self.burrows_similarity_matrix = np.zeros((n_success, n_success), dtype=np.float32)
        for i in range(n_success):
            self.burrows_similarity_matrix[i, i] = 1.0

        if burrows_distances is not None and burrows_valid:
            for rp, ri in enumerate(burrows_valid):
                for cp, ci in enumerate(burrows_valid):
                    if rp == cp:
                        continue
                    d = float(burrows_distances[rp, cp])
                    self.burrows_similarity_matrix[ri, ci] = 1.0 / (1.0 + d)

        burrows_cluster_variants = self._cluster_burrows_variants(
            burrows_distances,
            burrows_valid,
            variant_count=(variant_count if self.cluster_method == 'leiden' else 5),
            base_seed=42,
        )
        for metric_key, labels in burrows_cluster_variants.items():
            metrics_data[metric_key] = np.array(
                [float(label) if label is not None else np.nan for label in labels],
                dtype=float,
            )
        self.burrows_cluster_variant_labels = burrows_cluster_variants

        burrows_reduced = self._build_burrows_layout(successful, style_profiles)
        burrows_word_frequencies = self._build_burrows_word_frequencies(
            successful, style_profiles
        )
        burrows_function_words = list(self.BURROWS_FUNCTION_WORDS)

        if burrows_reduced is not None:
            print(
                f"  Burrows experimental map ready "
                f"({len(style_profiles)}/{n_success} texts have style profiles)"
            )
        else:
            print(
                "  Burrows experimental map unavailable "
                "(fewer than two texts have style profiles); "
                "Experimental tab will be hidden"
            )

        capitals: Dict[str, Any] = {}
        if capital_cities:
            for key, labels in cluster_labels.items():
                capitals[key] = self._semantic_capitals(emb_matrix, labels, successful)
            for key, labels in burrows_cluster_variants.items():
                capitals[key] = self._burrows_capitals(
                    burrows_distances, burrows_valid, labels, successful
                )
            total = sum(1 for c in capitals.values() if c.get('world') is not None)
            print(f"  Capital cities computed for {total} cluster variant(s)")

        reduced = self.reduce_dims(emb_matrix, n_success)
        focus_text = filter_kwargs.get('focus_text', '')

        self.visualize(
            reduced,
            successful,
            metrics_data,
            self.active_metric,
            output_html,
            low_token_texts,
            focus_text=focus_text,
            # Item 3: round to 5 decimals to keep the exported HTML payload small
            semantic_embeddings=_round_list(normalized_embeddings, 5),
            burrows_similarity=_round_list(self.burrows_similarity_matrix, 5),
            fighting_words_unigrams={
                FIGHTING_WORDS_PEAK_METRIC: self._fighting_words_peak_unigrams,
                FIGHTING_WORDS_FLOOR_METRIC: self._fighting_words_floor_unigrams,
                FIGHTING_WORDS_PEAK_CENTROID_METRIC: self._fighting_words_peak_unigrams_centroid,
                FIGHTING_WORDS_FLOOR_CENTROID_METRIC: self._fighting_words_floor_unigrams_centroid,
            },
            burrows_reduced=burrows_reduced,
            burrows_function_words=burrows_function_words,
            burrows_word_frequencies=burrows_word_frequencies,
            capital_cities=capitals,
        )


def _load_external(args, parser):
    docs: Dict[str, Tuple[str, str]] = {}

    def label(raw: str, i: int) -> str:
        lbl = sanitize_filename(raw)
        if len(lbl) > 240:
            d = hashlib.sha256(lbl.encode('utf-8')).hexdigest()[:16]
            lbl = f"{lbl[:220]}_{d}"
        return lbl or f"document_{i}"

    if args.nltk_corpus:
        try:
            from nltk import corpus as nltk_corpus
            reader = getattr(nltk_corpus, args.nltk_corpus)
            fileids = list(reader.fileids())
        except (ImportError, AttributeError, LookupError) as exc:
            parser.error(f"could not load NLTK corpus '{args.nltk_corpus}': {exc}")

        for i, fid in enumerate(fileids):
            try:
                content = reader.raw(fid)
            except Exception as exc:
                print(f"Warning: skipped NLTK document {fid!r}: {exc}")
                continue

            k = label(
                f"nltk_{args.nltk_corpus}_{fid}".replace('/', '__').replace('\\', '__'),
                i,
            )
            if k in docs:
                k = f"{k}_{i}"
            sid = hashlib.sha256(content.encode('utf-8')).hexdigest()
            docs[k] = (content, f"nltk:{args.nltk_corpus}:{fid}:{sid}")

        if not docs:
            parser.error(f"NLTK corpus '{args.nltk_corpus}' contains no readable documents")

    if args.txt_directory:
        root = Path(args.txt_directory).expanduser().resolve()
        if not root.is_dir():
            parser.error(f"text directory does not exist: {root}")

        for fp in sorted(root.rglob('*.txt')):
            if not fp.is_file():
                continue
            rel = fp.relative_to(root)
            if len(rel.parts) - 1 > args.txt_depth:
                continue
            try:
                content = fp.read_text(encoding='utf-8')
                st = fp.stat()
            except (OSError, UnicodeDecodeError) as exc:
                print(f"Warning: skipped {fp}: {exc}")
                continue
            k = label(
                f"txt_{rel.as_posix()}".replace('/', '__'),
                len(docs),
            )
            if k in docs:
                k = f"{k}_{len(docs)}"
            sid = f"file:{fp}:{st.st_mtime_ns}:{st.st_size}"
            docs[k] = (content, sid)

        if not docs:
            parser.error(f"no readable .txt files found within {args.txt_depth} levels of {root}")

    return docs


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Text Classifier - Cluster texts by semantic similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python atlas.py text1 text2 text3\n"
            "  python atlas.py text1 text2 --niche-words <niche>\n"
            "  python atlas.py --cluster-method leiden --cluster-variants 8\n\n"
            "Available metrics for --color-by:\n  "
        ) + ", ".join(MetricConfig.get_metric_names()),
    )

    parser.add_argument('texts', nargs='*', help='Text folder names to process')

    src = parser.add_mutually_exclusive_group()
    src.add_argument('--nltk-corpus', metavar='NAME', help='Compare every text in an installed NLTK corpus')
    src.add_argument('--txt-directory', metavar='PATH', help='Compare every .txt file under a directory')

    parser.add_argument('--txt-depth', type=int, default=3,
                        help='Maximum number of subdirectory levels to scan with --txt-directory (default: 3)')
    parser.add_argument('--date-from', help='Include only videos from this date onwards (YYYYMMDD)')
    parser.add_argument('--date-to', help='Include only videos up to this date (YYYYMMDD)')
    parser.add_argument('--duration-from', help='Minimum video duration (HH:MM:SS)')
    parser.add_argument('--duration-to', help='Maximum video duration (HH:MM:SS)')
    parser.add_argument('--exclude-live', action='store_true', help='Exclude livestream videos')
    parser.add_argument('--min-tokens-per-file', type=int,
                        help='Skip individual transcript files with fewer than N tokens')
    parser.add_argument('--min-tokens-total', '--min-words', type=int, dest='min_tokens_total',
                        help='Skip texts with fewer than N total tokens')
    parser.add_argument('--embeddings-only-low-token', action='store_true',
                        dest='embeddings_only_low_token',
                        help='Generate embeddings for texts below min-tokens-total, but skip metric calculations')
    parser.add_argument('--token-limit', type=int, help='Use only the newest N tokens per text')
    parser.add_argument(
        '--text-token-limit', type=int, dest='text_token_limit',
        help='Use only the first N tokens per text',
    )
    parser.add_argument(
        '--per-video-token-limit', type=int, dest='per_video_token_limit',
        help='Use only the first N tokens per video before truncating each file',
    )
    parser.add_argument(
        '--ignore-urls', type=str, dest='ignore_urls',
        help='Path to txt file with YouTube URLs or video IDs to ignore (one per line)',
    )

    parser.add_argument(
        '--output-csv', default='data/output/text_similarities.csv',
        help='Output CSV file path',
    )
    parser.add_argument(
        '--get-metrics-csv', action='store_true',
        help='Also export one row per text with metric columns to a *_metrics.csv file',
    )
    parser.add_argument(
        '--output-html', default='data/output/CentroidsMap.html',
        help='Output HTML visualization file path',
    )
    parser.add_argument('--no-cache', action='store_true', help='Disable embedding cache')
    parser.add_argument(
        '--force-latest-cache', action='store_true',
        help='Bypass cache freshness checks and use newest cache file per text immediately',
    )
    parser.add_argument(
        '--clear-cache', nargs='?', const='all', metavar='TEXT',
        help='Clear cache for specific text or all texts',
    )

    parser.add_argument(
        '--color-by', choices=None,
        help='Metric to use for coloring the visualization',
    )
    parser.add_argument(
        '--calculate-perplexity', action='store_true',
        help='Enable perplexity calculation (slower, loads language model)',
    )
    parser.add_argument(
        '--compute-ngram-entropy', action='store_true',
        help='Enable n-gram cross-entropy metrics (slower for large token counts)',
    )
    parser.add_argument('--fast-mode', action='store_true', help='Skip expensive metrics (fastest mode)')
    parser.add_argument('--word-counts-only', action='store_true',
                        help='Compute/cache only count-style and regex-pattern metrics')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Skip all metric calculations and only run embeddings/clustering/UMAP')
    parser.add_argument('--capital-cities', action='store_true', dest='capital_cities',
                        help='Mark the most representative text (cluster centroid / medoid) of '
                             'each cluster as a pentagon, and the corpus centroid / medoid as the '
                             'yellow-outlined "capital of the world".')
    parser.add_argument(
        '--change-fword-to-stupid', action='store_true', dest='change_fword_to_stupid',
        help='Replace standalone "fuck" tokens with "stupid" before embedding and metric '
             'computation, so vulgarity does not dominate semantic comparisons. Derived forms '
             '("fucker", "fucking", etc.) are left untouched. The flag is mixed into cache '
             'signatures, so runs with and without it stay in separate cache slots.',
    )
    parser.add_argument('--umap-neighbors', type=int,
                        help='UMAP neighborhood size; defaults to 5 for datasets up to 30 items, otherwise 50')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                        help='Minimum distance between points in UMAP space (default: 0.1; use 0 for tighter groups)')
    parser.add_argument('--umap-epochs', type=int, default=200,
                        help='Number of UMAP optimization epochs (default: 200)')
    parser.add_argument('--cluster-variants', type=int, default=None,
                        help='Number of clustering variants to generate (minimum: 1). Defaults to 11 '
                             'for datasets up to 2,000 items and 1 for larger datasets.')
    parser.add_argument('--cluster-method', choices=['hdbscan', 'leiden'], default='hdbscan',
                        help='Clustering algorithm to use (default: hdbscan).')
    parser.add_argument('--leiden-resolution-min', type=float, default=1.15,
                        help='Minimum Leiden resolution parameter (default: 1.15)')
    parser.add_argument('--leiden-resolution-max', type=float, default=3.0,
                        help='Maximum Leiden resolution parameter (default: 3.0)')
    parser.add_argument('--leiden-k-neighbors', type=int, default=20,
                        help='Number of nearest neighbors for the Leiden k-NN graph (default: 20)')
    parser.add_argument('--no-leiden-whiten', action='store_true', dest='no_leiden_whiten',
                        help='Disable PCA whitening before k-NN graph construction')
    parser.add_argument('--embedding-model', default=SBERT_MODEL,
                        help='Embedding model name')
    parser.add_argument('--embedding-chunk-size', type=int,
                        help='Override embedding chunk size (in model tokens)')
    parser.add_argument('--embedding-min-chunk', type=int,
                        help='Override minimum embedding chunk size (in model tokens)')
    parser.add_argument('--embedding-batch-size', type=int,
                        help='Override embedding batch size for SentenceTransformer encode()')
    parser.add_argument('--trained-model-path', type=str, dest='trained_model_path',
                        help='Path to a custom fine-tuned SBERT model')
    parser.add_argument('--compute-fighting-words', '--compute-fighting-words-floor',
                        dest='compute_fighting_words', action='store_true',
                        help='Enable fighting-words peak (rank 1) and floor (rank 1000) metrics. '
                             'Requires a corpus-wide pre-scan (all texts are tokenized before '
                             'the main pass).')
    parser.add_argument('--centroid-mode', choices=['video', 'word'],
                        help='Use centroids for text representation (video or word)')
    parser.add_argument('--centroid-videos', type=int, default=10,
                        help='Number of videos to use for video centroid computation (default: 10)')
    parser.add_argument('--centroid-words', type=int, default=1000000,
                        help='Number of words to use for word centroid computation (default: 1,000,000)')
    parser.add_argument('--stat-word-count', type=int, default=None,
                        help='Number of words to use for statistical metrics when using video centroids')
    parser.add_argument('--focus-text', type=str, dest='focus_text',
                        help='Focus text for similarity mapping')
    parser.add_argument('--tomotopy-topics', type=int, default=0,
                        help='Train a Tomotopy LDA model with this many topics (requires tomotopy installed; 0 disables).')
    parser.add_argument('--tomotopy-iterations', type=int, default=1000,
                        help='Gibbs sampling iterations for Tomotopy (default: 1000).')
    parser.add_argument('--tomotopy-threads', type=int, default=None,
                        help='Threads for Tomotopy train (default: min(8, CPU count)).')
    parser.add_argument('--export-gameologist', action='store_true',
                        help='Export data for the gameologist sister project (requires 1500+ channels)')
    parser.add_argument('--gameologist-dir', type=str, default=None,
                        help='Path to gameologist directory (default: ../gameologist)')
    parser.add_argument('--niche-words', action='append', default=None, metavar='NICHE',
                        help='Enable niche word-count metric sets (repeatable).')

    return parser


def _apply_niche(args, parser):
    if not args.niche_words:
        return
    for niche in dict.fromkeys(args.niche_words):
        mod = None
        for module_name in (f'atlas_{niche}', f'metricsmixin.atlas_{niche}'):
            try:
                mod = importlib.import_module(module_name)
                break
            except ImportError:
                continue
        if mod is None:
            parser.error(f"unsupported niche '{niche}': No module named 'atlas_{niche}'")
        register = getattr(mod, f'register_{niche}_metrics', None)
        if register is None:
            parser.error(
                f"module '{mod.__name__}' has no function 'register_{niche}_metrics'"
            )
        register(MetricConfig, TextClassifier)


def _export_gameologist(args, classifier):
    if not GAMEOLOGIST_AVAILABLE:
        print("Error: gameologist export requested but atlas_gameologist module not available")
        return

    texts = list(getattr(classifier, 'successful', None) or classifier.embeddings.keys())
    n = len(texts)

    if n == 0:
        print("Error: no channels processed successfully; nothing to export to gameologist")
        return

    sim = classifier.similarity_matrix
    burrows = classifier.burrows_similarity_matrix
    if sim is None or burrows is None:
        print("Error: similarity matrices not available; skipping gameologist export")
        return
    if sim.shape[0] != n or burrows.shape[0] != n:
        print(
            f"Error: matrix shape mismatch (successful={n}, "
            f"semantic={sim.shape[0]}, burrows={burrows.shape[0]})"
        )
        return

    if n < 1500:
        print(
            f"Warning: gameologist usually targets 1500+ channels; "
            f"exporting {n} channels instead. The export will still work, "
            f"but the sister project is designed around a larger corpus."
        )

    if args.gameologist_dir:
        gdir = Path(args.gameologist_dir).expanduser().resolve()
    else:
        gdir = Path(__file__).resolve().parents[1] / "gameologist"

    exporter = GameologistExporter(
        centroids_dir=Path(__file__).resolve().parents[1],
        gameologist_dir=gdir,
    )

    try:
        exporter.export_all(
            texts=texts,
            cluster_labels=None,
            similarity_matrix=sim,
            burrows_similarity_matrix=burrows,
            cluster_method=args.cluster_method,
            leiden_resolution=1.5,
            cluster_variant_labels=classifier.cluster_variant_labels,
            cluster_metric_labels=classifier.cluster_metric_labels,
            burrows_cluster_variant_labels=classifier.burrows_cluster_variant_labels,
            burrows_cluster_metric_labels=classifier.burrows_cluster_metric_labels,
        )
    except Exception as exc:
        import traceback
        print(f"Error: gameologist export failed: {exc}")
        traceback.print_exc()


def main():
    parser = _arg_parser()
    args = parser.parse_args()

    if args.umap_neighbors is not None and args.umap_neighbors < 2:
        parser.error('--umap-neighbors must be at least 2')
    if not 0.0 <= args.umap_min_dist <= 1.0:
        parser.error('--umap-min-dist must be between 0 and 1')
    if args.umap_epochs < 1:
        parser.error('--umap-epochs must be positive')
    if args.tomotopy_topics < 0:
        parser.error('--tomotopy-topics must be zero or greater')

    if args.tomotopy_threads is not None and args.tomotopy_threads <= 0:
        parser.error('--tomotopy-threads must be greater than zero')

    if args.txt_depth < 0:
        parser.error('--txt-depth must be zero or greater')

    external = _load_external(args, parser)

    if external and args.centroid_mode == 'video':
        parser.error('external document sources support --centroid-mode word, not video')

    if args.stat_word_count is None:
        args.stat_word_count = args.token_limit if args.token_limit else 1_000_000

    _apply_niche(args, parser)

    valid = set(MetricConfig.get_metric_names())
    if args.color_by is not None and args.color_by not in valid:
        parser.error(f"--color-by '{args.color_by}' is not a registered metric")

    if args.cluster_method == 'leiden' and not LEIDEN_AVAILABLE:
        parser.error(
            "--cluster-method leiden requires the 'leidenalg' and 'igraph' packages. "
            "Install them with: pip install leidenalg python-igraph"
        )

    if args.clear_cache:
        cache_dir = Path(__file__).resolve().parents[1] / "cache" / "metrics"
        SimpleCache(cache_dir).clear(
            None if args.clear_cache == 'all' else args.clear_cache
        )
        if not args.texts:
            return

    if external:
        args.texts = list(external)
        print(f"Loaded {len(args.texts)} external documents")
    elif not args.texts:
        input_dir = Path(__file__).resolve().parents[1] / "data" / "input"
        if input_dir.exists():
            args.texts = [
                d.name for d in input_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]
            if args.texts:
                print(f"Auto-discovered {len(args.texts)} texts from {input_dir}")
            else:
                parser.error(f"No texts found in {input_dir}. Use --help for usage information.")
                return
        else:
            parser.error(
                "No texts specified and data/input directory not found in expected "
                "locations. Use --help for usage information."
            )
            return

    active = args.color_by

    if args.no_metrics and active:
        parser.error("--no-metrics cannot be used with --color-by")

    if (
        args.word_counts_only
        and active
        and active not in TextClassifier.get_word_count_metric_names()
    ):
        parser.error(
            "--word-counts-only only supports --color-by metrics in the count/regex subset"
        )

    if (
        active in (
            FIGHTING_WORDS_PEAK_METRIC, FIGHTING_WORDS_FLOOR_METRIC,
            FIGHTING_WORDS_PEAK_CENTROID_METRIC, FIGHTING_WORDS_FLOOR_CENTROID_METRIC,
        )
        and not args.compute_fighting_words
    ):
        parser.error(f"--color-by {active} requires --compute-fighting-words")

    needs_ppl = (
        (not args.no_metrics)
        and (args.calculate_perplexity or (active == 'perplexity'))
    )
    enable_spacy = (
        (not args.no_metrics)
        and (active and MetricConfig.requires_spacy_model(active))
    )
    enable_ngram = (
        (not args.no_metrics)
        and (args.compute_ngram_entropy or (active and MetricConfig.requires_ngram_entropy(active)))
    )

    model = args.embedding_model
    if args.trained_model_path:
        tm = Path(args.trained_model_path)
        if not tm.exists():
            parser.error(f"Trained model not found at: {args.trained_model_path}")
        model = str(tm.resolve())
        print(f"[INFO] Using trained model: {model}")

    classifier = TextClassifier(
        use_cache=not args.no_cache,
        active_metric=active,
        enable_spacy=enable_spacy,
        enable_ngram_entropy=enable_ngram,
        fast_mode=args.fast_mode,
        word_counts_only=args.word_counts_only,
        force_latest_cache=args.force_latest_cache,
        embedding_model=model,
        embedding_chunk_size=args.embedding_chunk_size,
        embedding_min_chunk=args.embedding_min_chunk,
        embedding_batch_size=args.embedding_batch_size,
        no_metrics=args.no_metrics,
        compute_fighting_words=args.compute_fighting_words,
        centroid_mode=args.centroid_mode,
        centroid_videos=args.centroid_videos,
        centroid_words=args.centroid_words,
        stat_word_count=args.stat_word_count,
        cluster_method=args.cluster_method,
        leiden_resolution_min=args.leiden_resolution_min,
        leiden_resolution_max=args.leiden_resolution_max,
        leiden_k_neighbors=args.leiden_k_neighbors,
        leiden_whiten=not args.no_leiden_whiten,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_epochs=args.umap_epochs,
        tomotopy_num_topics=args.tomotopy_topics,
        tomotopy_iterations=args.tomotopy_iterations,
        tomotopy_threads=args.tomotopy_threads,
        change_fword_to_stupid=args.change_fword_to_stupid,
    )

    # Defensive: make sure the requested clustering method actually reached the
    # classifier. This catches edge cases where argparse or an external wrapper
    # might not pass the flag through, which would otherwise silently fall back
    # to HDBSCAN.
    if getattr(classifier, 'cluster_method', 'hdbscan') != args.cluster_method:
        print(
            f"[WARNING] Requested --cluster-method {args.cluster_method!r} but "
            f"TextClassifier.cluster_method is "
            f"{getattr(classifier, 'cluster_method', None)!r}; forcing the "
            f"requested value."
        )
        classifier.cluster_method = args.cluster_method

    classifier.active_focus_text = args.focus_text if args.focus_text else ''
    if external:
        classifier.processor.set_external_documents(external)

    if needs_ppl:
        classifier.processor.needs_perplexity_model = True
        classifier.processor._initialize_perplexity_model('distilgpt2')

    ignore_ids = load_ignore_urls(args.ignore_urls)

    classifier.run(
        args.texts, args.output_csv, args.output_html,
        ignore_ids=ignore_ids,
        capital_cities=args.capital_cities,
        date_from=args.date_from, date_to=args.date_to,
        duration_from=args.duration_from, duration_to=args.duration_to,
        exclude_live=args.exclude_live, token_limit=args.token_limit,
        text_token_limit=args.text_token_limit,
        per_video_token_limit=args.per_video_token_limit,
        min_tokens_total=args.min_tokens_total,
        min_tokens_per_file=args.min_tokens_per_file,
        embeddings_only_low_token=args.embeddings_only_low_token,
        cluster_variants=args.cluster_variants,
        focus_text=args.focus_text,
        get_metrics_csv=args.get_metrics_csv,
    )

    if args.export_gameologist:
        _export_gameologist(args, classifier)


if __name__ == "__main__":
    main()