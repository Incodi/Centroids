from atlas_config import *  # noqa
from atlas_config import (
    _format_trimmed_decimal,
    _is_word_count_metric_config,
)
from atlas_cache import SimpleCache
from atlas_metrics import TextMetricsMixin
from atlas_processor import TextProcessor
import importlib
import atlas_config as cfg
from scipy.stats import rankdata

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
        if m and src.startswith('\\b(?:') == src.endswith(')\\b'):
            lower = bool(pattern.flags & re.IGNORECASE)
            alts = {tuple(w.lower() if lower else w for w in a.split(' '))
                    for a in m.group(1).split('|')}
            if not _ambiguous(alts):
                by_k: Dict[int, set] = {}
                for a in alts:
                    by_k.setdefault(len(a), set()).add(a)
                terms = {k: frozenset(v) for k, v in by_k.items()}
    _pure_terms_cache[pattern] = terms
    return terms


class TextClassifier(TextMetricsMixin):

    CROSS_CHANNEL_METRICS = {'burrows_cosine_disagreement'}
    BURROWS_FUNCTION_WORDS = tuple(sorted({
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
        'myself', 'yourself', 'himself', 'herself', 'themselves',
        'be', 'is', 'am', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing',
        'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
        'to', 'of', 'and', 'a', 'in', 'that', 'for', 'not', 'on', 'with', 'as', 'at',
        'but', 'by', 'from', 'or', 'an', 'up', 'out', 'if', 'about', 'over',
        'after', 'because', 'any', 'than', 'into', 'through', 'down', 'before', 'between',
        'this', 'these', 'those', 'there', 'what', 'so', 'who', 'which', 'when',
        'no', 'nor', 'only', 'other', 'some', 'then', 'now', 'look', 'come', 'go', 'get',
        'take', 'see', 'think', 'also', 'back', 'use', 'how', 'even', 'want', 'give',
        'most', 'each', 'every', 'both', 'few', 'more', 'such', 'own', 'same', 'too',
        'very', 'just', 'here', 'why', 'once', 'the',
        'oh', 'yeah', 'yes', 'okay', 'ok', 'like', 'well', 'uh', 'um', 'right',
        'mhm', 'next', 'again',
    }))
    BURROWS_CONTRACTIONS = {
        "i'm": "i am", "i've": "i have", "i'd": "i would", "i'll": "i will",
        "you're": "you are", "you've": "you have", "you'd": "you would", "you'll": "you will",
        "he's": "he is", "he'd": "he would", "he'll": "he will",
        "she's": "she is", "she'd": "she would", "she'll": "she will",
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
    }

    @staticmethod
    def get_word_count_metric_names() -> List[str]:
        selected = set()

        for metric_name, config in MetricConfig.METRICS.items():
            compute_method = config.get('compute_method', '')

            if metric_name.endswith('_count'):
                selected.add(metric_name)
                continue

            if compute_method == 'compute_insult_density':
                selected.add(metric_name)
                continue

            if compute_method.startswith('_compute'):
                method = getattr(TextClassifier, compute_method, None)
                if method is None:
                    continue
                try:
                    src = inspect.getsource(method)
                except Exception:
                    src = ''

                if 'PATTERNS' in src or 're.' in src:
                    selected.add(metric_name)

        return sorted(selected)

    def _get_target_metric_names(self, include_global_metrics: bool = False,
                                 skip_centroid_duplicates: bool = False,
                                 skip_video_centroid_only: bool = False) -> List[str]:
        if getattr(self, 'no_metrics', False):
            return []
        if self.word_counts_only:
            names = self.get_word_count_metric_names()
        else:
            names = MetricConfig.get_metric_names()

        if include_global_metrics:
            if not getattr(self, 'compute_fighting_words', False):
                names = [name for name in names if name not in (FIGHTING_WORDS_PEAK_METRIC, FIGHTING_WORDS_FLOOR_METRIC,
                                                                  FIGHTING_WORDS_PEAK_CENTROID_METRIC, FIGHTING_WORDS_FLOOR_CENTROID_METRIC)]
        else:
            names = [name for name in names if name not in (FIGHTING_WORDS_PEAK_METRIC, FIGHTING_WORDS_FLOOR_METRIC,
                                                              FIGHTING_WORDS_PEAK_CENTROID_METRIC, FIGHTING_WORDS_FLOOR_CENTROID_METRIC)]

        names = [name for name in names if name not in self.CROSS_CHANNEL_METRICS]

        if skip_centroid_duplicates:
            names = [name for name in names if not MetricConfig.is_centroid_duplicate(name)]

        if skip_video_centroid_only:
            names = [name for name in names if not MetricConfig.is_video_centroid_only(name)]

        return names

    @classmethod
    def _burrows_rate_profile(cls, tokens: List[str]) -> np.ndarray:
        profile = np.zeros(len(cls.BURROWS_FUNCTION_WORDS), dtype=np.float64)
        if not tokens:
            return profile
        word_indices = {word: index for index, word in enumerate(cls.BURROWS_FUNCTION_WORDS)}
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.extend(cls.BURROWS_CONTRACTIONS.get(token, token).split())
        for token in expanded_tokens:
            index = word_indices.get(token)
            if index is not None:
                profile[index] += 1.0
        return profile / len(expanded_tokens)

    @staticmethod
    def _compute_burrows_cosine_disagreement(tokens: List[str]) -> None:
        return None

    @classmethod
    def _burrows_distance_matrix(
        cls,
        style_profiles: Dict[str, np.ndarray],
        texts: List[str],
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        valid = [index for index, text in enumerate(texts) if text in style_profiles]
        if len(valid) < 2:
            return None, []
        profile_matrix = np.vstack([style_profiles[texts[index]] for index in valid])
        means = profile_matrix.mean(axis=0)
        std = profile_matrix.std(axis=0, ddof=0)
        std[std == 0] = 1e-12
        z_matrix = (profile_matrix - means) / std
        distances = np.mean(np.abs(z_matrix[:, None, :] - z_matrix[None, :, :]), axis=2)
        return distances, valid

    @classmethod
    def _burrows_cosine_scores(
        cls,
        style_profiles: Dict[str, np.ndarray],
        embeddings: np.ndarray,
        texts: List[str],
    ) -> Dict[str, float]:
        style_distances, valid = cls._burrows_distance_matrix(style_profiles, texts)
        scores = {text: 0.0 for text in texts}
        if style_distances is None or len(valid) < 3:
            return scores

        normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        semantic_distances = 1.0 - normalized @ normalized.T
        for row, text_index in enumerate(valid):
            peers = [peer for peer in range(len(valid)) if peer != row]
            if len(peers) < 2:
                continue
            style_ranks = rankdata(style_distances[row, peers], method='average')
            semantic_ranks = rankdata(semantic_distances[text_index, [valid[peer] for peer in peers]], method='average')
            style_centered = style_ranks - style_ranks.mean()
            semantic_centered = semantic_ranks - semantic_ranks.mean()
            denominator = np.linalg.norm(style_centered) * np.linalg.norm(semantic_centered)
            rho = float(style_centered @ semantic_centered / denominator) if denominator else 0.0
            scores[texts[text_index]] = float(1.0 - rho)
        return scores

    def _load_burrows_rate_profile(self, text: str, filter_kwargs: Dict[str, Any]) -> Optional[np.ndarray]:
        filters = {key: value for key, value in filter_kwargs.items() if value is not None}
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
            return self._burrows_rate_profile(result[3]) if result[3] else None

        result = self.processor.get_tokens(text, filters, ignore_ids)
        if not result:
            return None
        tokens = result[0]
        if self.centroid_mode == 'word' and self.centroid_words > 0:
            tokens = tokens[-self.centroid_words:]
        return self._burrows_rate_profile(tokens)

    def _metric_is_skipped(self, name: str) -> bool:
        return not MetricConfig.is_metric_eligible(
            name,
            enable_spacy=self.enable_spacy,
            needs_perplexity_model=self.processor.needs_perplexity_model,
            enable_ngram_entropy=self.enable_ngram_entropy,
            fast_mode=self.fast_mode,
        )

    def _metric_needs_compute(self, name: str, cached_metrics: Dict[str, Any]) -> bool:
        if name not in cached_metrics:
            return True
        if cached_metrics[name] is not None:
            return False
        return MetricConfig.is_metric_eligible(
            name,
            enable_spacy=self.enable_spacy,
            needs_perplexity_model=self.processor.needs_perplexity_model,
            enable_ngram_entropy=self.enable_ngram_entropy,
            fast_mode=self.fast_mode,
        )

    @staticmethod
    def _summarize_metric_names(metric_names: List[str], max_items: Optional[int] = 16) -> str:
        if not metric_names:
            return "none"
        names = sorted(set(metric_names))
        if max_items is None or max_items <= 0 or len(names) <= max_items:
            return ', '.join(names)
        shown = ', '.join(names[:max_items])
        return f"{shown} ... (+{len(names) - max_items} more)"

    def _print_metric_completion_summary(self, metric_values: Dict[str, Any], source: str) -> None:
        if not metric_values:
            return

        completed = sorted(name for name, value in metric_values.items() if value is not None)
        skipped = sorted(name for name, value in metric_values.items() if value is None)

        print(f"    Metrics finished: {len(completed)}/{len(metric_values)} [{source}]")

        embedding_metrics = [name for name in completed if 'embedding' in name]
        word_count_metrics = [
            name for name in completed
            if _is_word_count_metric_config(name, MetricConfig.METRICS.get(name, {}))
        ]

        grouped_names = set(word_count_metrics) | set(embedding_metrics)
        other_metrics = [name for name in completed if name not in grouped_names]

        if word_count_metrics:
            print(f"      • Word counts ({len(word_count_metrics)}): {self._summarize_metric_names(word_count_metrics)}")
        if embedding_metrics:
            print(f"      • Embedding metrics ({len(embedding_metrics)}): {self._summarize_metric_names(embedding_metrics)}")
        if other_metrics:
            print(f"      • Other metrics ({len(other_metrics)}): {self._summarize_metric_names(other_metrics)}")

        category_buckets: Dict[str, List[str]] = defaultdict(list)
        for metric_name in completed:
            category = MetricConfig.METRICS.get(metric_name, {}).get('category', 'Uncategorized')
            category_buckets[str(category)].append(metric_name)

        linguistic_categories = {
            'Core Linguistic Metrics',
            'Lexical Diversity',
            'Syntactic Patterns',
            'Discourse Markers',
            'Common Phrases',
        }
        for category_name in sorted(category_buckets):
            names_in_category = category_buckets[category_name]
            full_list = category_name in linguistic_categories
            summary = self._summarize_metric_names(
                names_in_category,
                max_items=None if full_list else 16,
            )
            print(f"      • {category_name} ({len(names_in_category)}): {summary}")

        if skipped:
            print(f"      • Skipped/unavailable ({len(skipped)}): {self._summarize_metric_names(skipped)}")

    @staticmethod
    def _detect_embedding_device() -> str:
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _encode_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        batch_size = max(1, int(self.embedding_batch_size))
        while True:
            try:
                return self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            except RuntimeError as e:
                message = str(e)
                memory_like_error = (
                    'Invalid buffer size' in message
                    or 'out of memory' in message.lower()
                    or 'CUDA out of memory' in message
                )
                if not memory_like_error or batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.embedding_batch_size = batch_size
                print(f"    Warning: embedding encode memory pressure, retrying with batch_size={batch_size}")

    def __init__(self, use_cache: bool = True, active_metric: Optional[str] = None, enable_spacy: bool = False, enable_ngram_entropy: bool = False, fast_mode: bool = False, word_counts_only: bool = False, force_latest_cache: bool = False, embedding_model: Optional[str] = None, embedding_chunk_size: Optional[int] = None, embedding_min_chunk: Optional[int] = None, embedding_batch_size: Optional[int] = None, no_metrics: bool = False, compute_fighting_words: bool = False, compute_fighting_words_floor: Optional[bool] = None, centroid_mode: Optional[str] = None, centroid_videos: int = 10, centroid_words: int = 1000000, stat_word_count: int = 1000000, cluster_method: str = 'hdbscan', leiden_resolution_min: float = 0.3, leiden_resolution_max: float = 2.0, leiden_k_neighbors: int = 20, leiden_whiten: bool = True, umap_neighbors: Optional[int] = None, umap_min_dist: float = 0.1, umap_epochs: int = 50, mallet_num_topics: int = 0):
        self.embedding_model_name = resolve_embedding_model_name(embedding_model)
        self.model = SentenceTransformer(self.embedding_model_name)
        profile = get_embedding_model_profile(self.embedding_model_name)
        self.embedding_device = self._detect_embedding_device()

        detected_context_window = int(getattr(self.model, 'max_seq_length', 0) or 0)
        self.embedding_context_window = detected_context_window if detected_context_window > 0 else profile['context_window']

        practical_key = f"practical_max_tokens_{self.embedding_device}"
        practical_max_tokens = int(profile.get(practical_key, profile['chunk_target']))
        practical_max_tokens = max(16, practical_max_tokens)

        if embedding_chunk_size is None:
            embedding_chunk_size = profile['chunk_target']
            if self.embedding_context_window > 0:
                embedding_chunk_size = min(embedding_chunk_size, max(16, self.embedding_context_window - 8))
            embedding_chunk_size = min(int(embedding_chunk_size), practical_max_tokens)
        else:
            embedding_chunk_size = int(embedding_chunk_size)
            if self.embedding_context_window > 0:
                embedding_chunk_size = min(embedding_chunk_size, max(16, self.embedding_context_window - 8))

        if embedding_min_chunk is None:
            embedding_min_chunk = profile['min_chunk']
        if embedding_chunk_size is not None and embedding_min_chunk is not None:
            embedding_min_chunk = min(int(embedding_min_chunk), int(embedding_chunk_size))

        self.embedding_batch_size = int(embedding_batch_size or profile['batch_size'])
        if self.embedding_device == 'cpu':
            self.embedding_batch_size = min(self.embedding_batch_size, 8)
        elif self.embedding_device == 'mps':
            self.embedding_batch_size = min(self.embedding_batch_size, 16)

        self.embedding_cache_signature: Dict[str, Any] = {
            'embedding_model': self.embedding_model_name,
            'embedding_context_window': int(self.embedding_context_window),
            'embedding_chunk_size': int(embedding_chunk_size),
            'embedding_min_chunk': int(embedding_min_chunk),
        }

        print(
            f"Embedding model: {self.embedding_model_name} | device={self.embedding_device} | context={self.embedding_context_window} "
            f"| chunk={int(embedding_chunk_size)} | min_chunk={int(embedding_min_chunk)} "
            f"| batch={self.embedding_batch_size}"
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
        self._centroid_equals_stat: Dict[str, bool] = {}

        self.centroid_mode = centroid_mode
        self.centroid_videos = centroid_videos
        self.centroid_words = centroid_words
        self.stat_word_count = stat_word_count

        self.mallet_num_topics: int = int(mallet_num_topics or 0)
        self.mallet_topic_proportions: Dict[str, Dict[int, float]] = {}
        self.mallet_topic_labels: Dict[int, str] = {}

        self.cluster_method = (cluster_method or 'hdbscan').lower()
        self.leiden_resolution_min = float(leiden_resolution_min)
        self.leiden_resolution_max = float(leiden_resolution_max)
        self.leiden_k_neighbors = int(leiden_k_neighbors)
        self.leiden_whiten = bool(leiden_whiten)
        self.umap_neighbors = None if umap_neighbors is None else int(umap_neighbors)
        self.umap_min_dist = float(umap_min_dist)
        self.umap_epochs = int(umap_epochs)
        self.cluster_metric_labels: Dict[str, str] = {}

        needs_perplexity = (active_metric and
                           MetricConfig.requires_perplexity_model(active_metric))

        self.processor = TextProcessor(
            self.base_dir,
            chunk_size=int(embedding_chunk_size),
            min_chunk=int(embedding_min_chunk),
            needs_perplexity_model=needs_perplexity,
            enable_spacy=enable_spacy,
            model_tokenizer=getattr(self.model, 'tokenizer', None),
            model_context_window=self.embedding_context_window,
        )

        self.processor._model = self.model
        self.processor.embedding_batch_size = self.embedding_batch_size
        self.processor.embedding_model_name_for_cache = self.embedding_model_name
        self.processor._encode_fn = self._encode_embeddings

        self.embeddings: Dict[str, np.ndarray] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
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

    @classmethod
    def validate_metric_wiring(cls) -> None:
        bad = []
        for name, cfg_ in MetricConfig.METRICS.items():
            m = cfg_.get('compute_method', '')
            if m == '_compute_dynamic_token_count':
                fb = DYNAMIC_COUNT_FALLBACK_METHODS.get(name)
                ok = (name in SIMPLE_COLOR_PATTERN_MAP
                      or name in DYNAMIC_PATTERN_METRIC_MAP
                      or name in DYNAMIC_TOKEN_SET_METRIC_MAP
                      or name in ('right_count', 'unique_words_count', 'unique_oov_count',
                                  'total_oov_count', 'unique_non_oov_count', 'total_non_oov_count')
                      or (fb and fb != '_compute_dynamic_token_count' and hasattr(cls, fb)))
            elif m.startswith('_compute'):
                ok = hasattr(cls, m)
            else:
                ok = m in ('compute_insult_density', 'calculate_perplexity', 'compute_vader_sentiment')
            if not ok:
                bad.append(name)
        if bad:
            print(f"WARNING: {len(bad)} metrics have no resolvable compute path: {sorted(bad)}")

    def _compute_fighting_words_peak_placeholder(self, tokens: List[str]) -> Optional[float]:
        return None

    def _compute_fighting_words_floor_placeholder(self, tokens: List[str]) -> Optional[float]:
        return None

    def _compute_fighting_words_peak_centroid_placeholder(self, tokens: List[str]) -> Optional[float]:
        return None

    def _compute_fighting_words_floor_centroid_placeholder(self, tokens: List[str]) -> Optional[float]:
        return None

    def _compute_avg_words_per_video(self, tokens: List[str]) -> Optional[float]:
        return None

    def _compute_fighting_words_values(self, texts: List[str], rank: int,
                                       use_centroid: bool = False) -> Dict[str, Optional[float]]:
        if not texts:
            return {}

        source_unigrams = self._fighting_words_unigrams_centroid if use_centroid else self._fighting_words_unigrams
        source_counts = self._fighting_words_token_counts_centroid if use_centroid else self._fighting_words_token_counts

        counters: Dict[str, Counter] = {}
        token_counts: Dict[str, int] = {}
        for text in texts:
            counter = source_unigrams.get(text)
            token_count = source_counts.get(text, 0)
            if counter is None or token_count <= 0:
                continue
            counters[text] = counter
            token_counts[text] = token_count

        if not counters:
            return {text: None for text in texts}

        total_counts = Counter()
        total_tokens = 0
        for text, counter in counters.items():
            total_counts.update(counter)
            total_tokens += token_counts[text]

        floors: Dict[str, Optional[float]] = {}
        for text in texts:
            focus_counts = counters.get(text)
            focus_token_count = token_counts.get(text, 0)
            if focus_counts is None or focus_token_count <= 0:
                floors[text] = None
                continue

            rest_token_count = total_tokens - focus_token_count
            if rest_token_count <= 0:
                floors[text] = None
                continue

            rest_counts = total_counts - focus_counts
            candidates = [phrase for phrase, count in focus_counts.items() if count >= FIGHTING_WORDS_FLOOR_MIN_COUNT]
            if not candidates:
                floors[text] = None
                continue

            V = len(candidates)
            z_scores: List[float] = []
            for phrase in candidates:
                f_focus = focus_counts[phrase]
                f_rest = rest_counts.get(phrase, 0)

                if f_rest <= 0:
                    continue

                p_focus = (f_focus + FIGHTING_WORDS_FLOOR_ALPHA) / (focus_token_count + FIGHTING_WORDS_FLOOR_ALPHA * V)
                p_rest = (f_rest + FIGHTING_WORDS_FLOOR_ALPHA) / (rest_token_count + FIGHTING_WORDS_FLOOR_ALPHA * V)

                log_odds = math.log(p_focus) - math.log(p_rest)
                variance = (1.0 / (f_focus + FIGHTING_WORDS_FLOOR_ALPHA)) + (1.0 / (f_rest + FIGHTING_WORDS_FLOOR_ALPHA))
                z_scores.append(log_odds / math.sqrt(variance))

            if not z_scores:
                floors[text] = None
                continue

            z_scores.sort(reverse=True)
            rank_index = min(rank, len(z_scores)) - 1
            floors[text] = float(z_scores[rank_index])

        return floors

    def _compute_fighting_words_floor_values(self, texts: List[str],
                                             use_centroid: bool = False) -> Dict[str, Optional[float]]:
        return self._compute_fighting_words_values(texts, FIGHTING_WORDS_FLOOR_RANK,
                                                    use_centroid=use_centroid)

    def _get_centroid_tokens_for_mallet(self, text: str, filters: Dict,
                                        ignore_ids: set) -> List[str]:
        if self.centroid_mode == 'video':
            files = self.processor._get_filtered_files(text, filters, ignore_ids)
            if not files:
                return []
            result = self.processor.load_video_centroids(
                text, self.centroid_videos, filters, ignore_ids,
                max_accumulated_tokens=self.stat_word_count,
                precomputed_files=files,
                need_embeddings=False,
                need_stat_corpus=False,
                need_centroid_tokens=True,
            )
            if not result:
                return []
            return result[3] or []

        result = self.processor.get_tokens(text, filters, ignore_ids)
        if not result:
            return []
        tokens = result[0]
        if self.centroid_mode == 'word' and self.centroid_words and self.centroid_words > 0:
            tokens = tokens[-self.centroid_words:]
        return tokens

    def _train_mallet_topics(self, texts: List[str], ignore_ids: set,
                             filter_kwargs: Dict[str, Any]) -> None:
        import shutil
        import subprocess

        if self.mallet_num_topics <= 0:
            return

        if not texts:
            return

        mallet_binary: Optional[str] = shutil.which('mallet')
        if not mallet_binary:
            mallet_home = os.environ.get('MALLET_HOME')
            if mallet_home:
                candidate = Path(mallet_home) / 'bin' / 'mallet'
                if candidate.exists():
                    mallet_binary = str(candidate)
        if not mallet_binary:
            print("  MALLET not found; skipping topic modeling. "
                  "Install it (brew install mallet, or set MALLET_HOME to a "
                  "MALLET install directory).")
            return

        cache_dir = self.base_dir / "cache" / "mallet"
        cache_dir.mkdir(parents=True, exist_ok=True)
        input_txt = cache_dir / "texts_input.txt"
        mallet_bin_file = cache_dir / "texts.mallet"
        doc_topics = cache_dir / "doc_topics.txt"
        topic_keys = cache_dir / "topic_keys.txt"
        sig_path = cache_dir / "signature.json"
        proportions_path = cache_dir / "proportions.json"
        labels_path = cache_dir / "labels.json"
        doc_map_path = cache_dir / "doc_map.json"

        filter_sig: Dict[str, Any] = {}
        for key in sorted(CACHE_FILTER_KEYS):
            if key in ('centroid_mode', 'centroid_videos', 'centroid_words', 'stat_word_count'):
                continue
            value = filter_kwargs.get(key)
            if value is None:
                continue
            filter_sig[key] = str(value)

        cache_signature = {
            'num_topics': int(self.mallet_num_topics),
            'texts': sorted(texts),
            'ignore_ids': sorted(str(v) for v in ignore_ids),
            'centroid_mode': self.centroid_mode,
            'centroid_videos': int(self.centroid_videos or 0),
            'centroid_words': int(self.centroid_words or 0),
            'stat_word_count': int(self.stat_word_count or 0),
            'filters': filter_sig,
        }

        if sig_path.exists() and proportions_path.exists():
            try:
                with open(sig_path, 'r', encoding='utf-8') as f:
                    saved_sig = json.load(f)
                if saved_sig == cache_signature:
                    with open(proportions_path, 'r', encoding='utf-8') as f:
                        loaded_props = json.load(f)
                    self.mallet_topic_proportions = {
                        tx: {int(k): float(v) for k, v in topics.items()}
                        for tx, topics in loaded_props.items()
                    }
                    if labels_path.exists():
                        try:
                            with open(labels_path, 'r', encoding='utf-8') as f:
                                loaded_labels = json.load(f)
                            self.mallet_topic_labels = {
                                int(k): str(v) for k, v in loaded_labels.items()
                            }
                        except Exception:
                            self.mallet_topic_labels = {}
                    print(f"  MALLET topics loaded from cache "
                          f"({len(self.mallet_topic_proportions)} texts, "
                          f"{self.mallet_num_topics} topics)")
                    return
            except Exception as exc:
                print(f"  Warning: failed to load MALLET cache ({exc}); retraining")

        print(f"  Training MALLET LDA: {self.mallet_num_topics} topics "
              f"(streaming input across {len(texts)} texts)")
        filters = {k: v for k, v in filter_kwargs.items() if v is not None}
        written = 0
        doc_map: Dict[str, str] = {}

        try:
            with open(input_txt, 'w', encoding='utf-8') as f:
                for tx in texts:
                    try:
                        tokens = self._get_centroid_tokens_for_mallet(tx, filters, ignore_ids)
                    except Exception as exc:
                        print(f"    Warning: could not gather tokens for {tx}: {exc}")
                        continue
                    if not tokens:
                        continue
                    safe_id = f"tx_{written:06d}"
                    doc_map[safe_id] = tx
                    f.write(safe_id)
                    f.write('\t')
                    f.write(' '.join(tokens))
                    f.write('\n')
                    written += 1
                    del tokens
        except OSError as exc:
            print(f"  MALLET input write failed: {exc}")
            return

        if written == 0:
            print("  MALLET skipped: no texts produced centroid tokens")
            return

        print(f"    Wrote {written} documents to {input_txt.name}")

        mallet_memory = os.environ.get('MALLET_MEMORY', '2g')
        run_env = {**os.environ, 'MALLET_MEMORY': mallet_memory}

        try:
            subprocess.run(
                [mallet_binary, 'import-file',
                 '--input', str(input_txt),
                 '--output', str(mallet_bin_file),
                 '--keep-sequence',
                 '--remove-stopwords'],
                check=True, capture_output=True, env=run_env,
            )
            subprocess.run(
                [mallet_binary, 'train-topics',
                 '--input', str(mallet_bin_file),
                 '--num-topics', str(self.mallet_num_topics),
                 '--num-iterations', '1000',
                 '--optimize-interval', '10',
                 '--output-doc-topics', str(doc_topics),
                 '--output-topic-keys', str(topic_keys)],
                check=True, capture_output=True, env=run_env,
            )
        except subprocess.CalledProcessError as exc:
            stderr = ''
            if getattr(exc, 'stderr', None):
                try:
                    stderr = exc.stderr.decode(errors='ignore')[:600]
                except Exception:
                    stderr = repr(exc.stderr)[:600]
            print(f"  MALLET training failed: {stderr or exc}")
            return
        except FileNotFoundError as exc:
            print(f"  MALLET binary not executable: {exc}")
            return
        except Exception as exc:
            print(f"  MALLET training failed: {exc}")
            return

        def _all_float(tokens: List[str]) -> bool:
            if not tokens:
                return False
            for tok in tokens:
                try:
                    float(tok)
                except ValueError:
                    return False
            return True

        proportions: Dict[str, Dict[int, float]] = {}
        try:
            with open(doc_topics, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    source_idx = -1
                    for i, part in enumerate(parts):
                        if part in doc_map:
                            source_idx = i
                            break
                    if source_idx < 0:
                        if parts[0] in doc_map:
                            source_idx = 0
                        else:
                            continue

                    source = parts[source_idx]
                    text_name = doc_map.get(source, source)
                    tail = parts[source_idx + 1:]

                    text_topics: Dict[int, float] = {}

                    if _all_float(tail):
                        for topic_id, token in enumerate(tail):
                            text_topics[topic_id] = float(token)
                    elif len(tail) == 1 and ':' in tail[0]:
                        for entry in tail[0].split():
                            if ':' not in entry:
                                continue
                            head, sep, val = entry.partition(':')
                            try:
                                text_topics[int(head)] = float(val)
                            except ValueError:
                                continue

                    if text_topics:
                        proportions[text_name] = text_topics
        except OSError as exc:
            print(f"  MALLET doc-topics read failed: {exc}")
            return

        if not proportions:
            print("  MALLET parse produced 0 documents; previewing doc-topics.txt:")
            try:
                with open(doc_topics, 'r', encoding='utf-8') as f:
                    for _ in range(5):
                        preview = f.readline()
                        if not preview:
                            break
                        print(f"    {preview.rstrip(chr(10))!r}")
            except Exception as preview_exc:
                print(f"    (could not preview file: {preview_exc})")

        self.mallet_topic_proportions = proportions

        self.mallet_topic_labels = {}
        try:
            with open(topic_keys, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        continue
                    try:
                        topic_id = int(parts[0])
                    except ValueError:
                        continue
                    top_words = ' '.join(parts[2].split()[:8])
                    self.mallet_topic_labels[topic_id] = top_words
        except OSError:
            pass

        try:
            with open(sig_path, 'w', encoding='utf-8') as f:
                json.dump(cache_signature, f)
            with open(proportions_path, 'w', encoding='utf-8') as f:
                json.dump(self.mallet_topic_proportions, f)
            with open(labels_path, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in self.mallet_topic_labels.items()}, f)
            with open(doc_map_path, 'w', encoding='utf-8') as f:
                json.dump(doc_map, f)
        except Exception as exc:
            print(f"  Warning: failed to write MALLET cache: {exc}")

        matched = sum(1 for tx in texts if tx in self.mallet_topic_proportions)
        print(f"  MALLET topics trained; matched {matched}/{len(texts)} texts "
              f"({len(self.mallet_topic_proportions)} docs parsed)")

    def _ensure_runtime_cache_for_tokens(self, tokens: List[str]) -> None:
        if (getattr(self, '_runtime_cache_tokens_ref', None) is not tokens
                or getattr(self, '_runtime_cache_len', -1) != len(tokens)):
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
        text = getattr(self, '_runtime_cache_text', None)
        if text is None:
            text = ' '.join(tokens)
            self._runtime_cache_text = text
        return text

    def _get_runtime_text_bytes(self, tokens: List[str]) -> bytes:
        self._ensure_runtime_cache_for_tokens(tokens)
        text_bytes = getattr(self, '_runtime_cache_text_bytes', None)
        if text_bytes is None:
            text_bytes = self._get_runtime_text(tokens).encode('utf-8', errors='ignore')
            self._runtime_cache_text_bytes = text_bytes
        return text_bytes

    def _get_runtime_text_lower(self, tokens: List[str]) -> str:
        self._ensure_runtime_cache_for_tokens(tokens)
        text_lower = getattr(self, '_runtime_cache_text_lower', None)
        if text_lower is None:
            text_lower = self._get_runtime_text(tokens).lower()
            self._runtime_cache_text_lower = text_lower
        return text_lower

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
        n = len(src)

        depth, i = 0, 0
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
        counter = getattr(self, '_runtime_cache_counter', None)
        if counter is None:
            counter = Counter(tokens)
            self._runtime_cache_counter = counter
        return counter

    def _get_runtime_word_pieces(self, tokens: List[str]) -> Counter:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_pieces is None:
            pieces: Counter = Counter()
            for tok, c in self._get_runtime_token_counter(tokens).items():
                for w in _WORD_RUN.findall(tok):
                    pieces[w] += c
            self._runtime_cache_pieces = pieces
        return self._runtime_cache_pieces

    def _runtime_edges(self, tokens: List[str]) -> Tuple[List[str], List[str], Dict[int, Counter]]:
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
        cached_stats = getattr(self, '_runtime_cache_spacy_stats', None)
        if cached_stats is not None:
            return cached_stats

        if cfg.spacy_nlp is None:
            try:
                import spacy
                cfg.spacy_nlp = spacy.load("en_core_web_sm")
            except (ModuleNotFoundError, ImportError, OSError):
                from umapper_config import _install_spacy_model
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
            gerund_count = 0
            imperative_count = 0

            for token in doc:
                if not token.is_punct and not token.is_space:
                    total_valid += 1
                    if token.pos_ in content_tags:
                        total_content += 1

                if token.tag_ == 'VBG':
                    gerund_count += 1

                if token.pos_ == 'VERB' and token.tag_ == 'VB':
                    if token.i == 0 or doc[token.i - 1].is_punct or doc[token.i - 1].pos_ == 'CCONJ':
                        imperative_count += 1

            stats = {
                'total_valid': int(total_valid),
                'total_content': int(total_content),
                'gerund_count': int(gerund_count),
                'imperative_count': int(imperative_count),
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
        pattern_cache: Dict[Tuple[int, bool], int] = getattr(self, '_runtime_cache_pattern_counts', {})
        prefix_cache: Dict[int, Optional[str]] = getattr(self, '_runtime_cache_pattern_prefixes', {})
        pending = list(patterns) if isinstance(patterns, (list, tuple, set)) else [patterns]

        while pending:
            pattern = pending.pop()

            if isinstance(pattern, (list, tuple, set)):
                pending.extend(pattern)
                continue

            if not hasattr(pattern, 'findall'):
                continue

            terms = _pure_terms(pattern)
            if terms is not None:
                count += self._fast_count(tokens, terms)
                continue

            pattern_id = id(pattern)
            if pattern_id not in prefix_cache:
                prefix_cache[pattern_id] = self._extract_required_literal_prefix(pattern)
            required_prefix = prefix_cache.get(pattern_id)
            if required_prefix:
                text_lower = self._get_runtime_text_lower(tokens)
                if required_prefix not in text_lower:
                    continue

            try:
                cache_key = (id(pattern), False)
                cached_count = pattern_cache.get(cache_key)
                if cached_count is None:
                    cached_count = len(pattern.findall(text))
                    pattern_cache[cache_key] = cached_count
                count += cached_count
            except TypeError:
                cache_key = (id(pattern), True)
                cached_count = pattern_cache.get(cache_key)
                if cached_count is None:
                    text_bytes = self._get_runtime_text_bytes(tokens)
                    cached_count = len(pattern.findall(text_bytes))
                    pattern_cache[cache_key] = cached_count
                count += cached_count

        return float(count)

    def _compute_dynamic_token_count(self, tokens: List[str], metric_name: str) -> float:
        if not tokens:
            return 0.0

        self._ensure_runtime_cache_for_tokens(tokens)
        dynamic_cache: Dict[str, float] = getattr(self, '_runtime_cache_dynamic_counts', {})
        cached_dynamic = dynamic_cache.get(metric_name)
        if cached_dynamic is not None:
            return cached_dynamic

        if metric_name in SIMPLE_COLOR_PATTERN_MAP:
            value = self._compute_pattern_count(tokens, SIMPLE_COLOR_PATTERN_MAP[metric_name])
            dynamic_cache[metric_name] = value
            return value

        if metric_name == 'right_count':
            base_count = self._compute_pattern_count(tokens, DYNAMIC_PATTERN_METRIC_MAP['right_count'])
            text = self._get_runtime_text(tokens)
            all_right_phrase_count = len(re.findall(r'\ball\s+right\b', text, re.IGNORECASE))
            value = float(max(0.0, base_count - all_right_phrase_count))
            dynamic_cache[metric_name] = value
            return value

        if metric_name in DYNAMIC_PATTERN_METRIC_MAP:
            value = self._compute_pattern_count(tokens, DYNAMIC_PATTERN_METRIC_MAP[metric_name])
            dynamic_cache[metric_name] = value
            return value

        if metric_name == 'unique_words_count':
            return self._compute_unique_words_count(tokens)

        if metric_name in ('unique_oov_count', 'total_oov_count', 'unique_non_oov_count', 'total_non_oov_count'):
            unique_oov, total_oov, unique_non_oov, total_non_oov = self._compute_oov_split_counts(tokens)
            if metric_name == 'unique_oov_count':
                return unique_oov
            if metric_name == 'total_oov_count':
                return total_oov
            if metric_name == 'unique_non_oov_count':
                return unique_non_oov
            return total_non_oov

        token_forms = DYNAMIC_TOKEN_SET_METRIC_MAP.get(metric_name)
        if token_forms is not None:
            token_counter = self._get_runtime_token_counter(tokens)
            value = float(sum(token_counter.get(token_form, 0) for token_form in token_forms))
            dynamic_cache[metric_name] = value
            return value

        fallback_method_name = DYNAMIC_COUNT_FALLBACK_METHODS.get(metric_name)
        if fallback_method_name and fallback_method_name != '_compute_dynamic_token_count':
            fallback_method = getattr(self, fallback_method_name, None)
            if fallback_method is not None:
                value = fallback_method(tokens)
                dynamic_cache[metric_name] = float(value)
                return value

        return 0.0

    def _compute_unique_words_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        unique_words = set()
        for token in tokens:
            cleaned = token.lower().strip("'\".,!?;:()[]{}")
            if len(cleaned) < 2:
                continue
            if not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", cleaned):
                continue
            unique_words.add(cleaned)
        return float(len(unique_words))

    def _compute_oov_split_counts(self, tokens: List[str]) -> Tuple[float, float, float, float]:
        if not tokens:
            return 0.0, 0.0, 0.0, 0.0

        self._ensure_runtime_cache_for_tokens(tokens)
        cached = self._runtime_cache_oov_split
        if cached is not None:
            return cached

        try:
            global enchant_dict, enchant_word_check_cache

            if not ENCHANT_AVAILABLE:
                result = (0.0, 0.0, 0.0, 0.0)
                self._runtime_cache_oov_split = result
                return result

            if enchant_dict is None:
                try:
                    enchant_dict = enchant.Dict("en_US")
                except Exception:
                    result = (0.0, 0.0, 0.0, 0.0)
                    self._runtime_cache_oov_split = result
                    return result

            normalized_counts = self._get_runtime_token_counter(tokens)
            if not normalized_counts:
                result = (0.0, 0.0, 0.0, 0.0)
                self._runtime_cache_oov_split = result
                return result

            cache = enchant_word_check_cache
            if len(cache) > 1_000_000:
                cache.clear()

            check = enchant_dict.check

            total_oov = 0
            unique_oov = 0
            total_non_oov = 0
            unique_non_oov = 0

            for cleaned, count in normalized_counts.items():
                length = len(cleaned)
                if length == 0 or length > 30 or not cleaned.isascii():
                    total_oov += count
                    unique_oov += 1
                    continue

                candidate = cleaned.replace("'", "") if "'" in cleaned else cleaned
                if not candidate or not candidate.isalpha():
                    total_oov += count
                    unique_oov += 1
                    continue

                cached_val = cache.get(cleaned)
                if cached_val is None:
                    try:
                        cached_val = bool(check(cleaned))
                    except Exception:
                        cached_val = False
                    cache[cleaned] = cached_val

                if cached_val:
                    total_non_oov += count
                    unique_non_oov += 1
                else:
                    total_oov += count
                    unique_oov += 1

            result = (
                float(unique_oov),
                float(total_oov),
                float(unique_non_oov),
                float(total_non_oov),
            )
            self._runtime_cache_oov_split = result
            return result
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def _compute_oov_counts(self, tokens: List[str]) -> Tuple[float, float]:
        try:
            unique_oov, total_oov, _, _ = self._compute_oov_split_counts(tokens)
            return unique_oov, total_oov
        except Exception:
            return 0.0, 0.0

    def _compute_all_metrics(self, tokens: List[str], chunks: List[str],
                             chunk_embs: Optional[np.ndarray] = None) -> Dict[str, any]:
        all_metric_names = [
            name for name in self._get_target_metric_names()
            if not MetricConfig.is_centroid_metric(name)
        ]
        return self._compute_metrics(tokens, chunks, all_metric_names)

    def _compute_metrics_subset(self, tokens: List[str], chunks: List[str],
                                metric_names: List[str]) -> Dict[str, any]:
        return self._compute_metrics(tokens, chunks, metric_names)

    def _compute_specific_metrics(self, tokens: List[str], chunks: List[str],
                                   chunk_embs: Optional[np.ndarray],
                                   metric_names: List[str]) -> Dict[str, any]:
        return self._compute_metrics(tokens, chunks, metric_names)

    def _compute_metrics(self, tokens: List[str], chunks: List[str],
                         metric_names: List[str]) -> Dict[str, any]:
        metrics = {}
        total_metrics = len(metric_names)
        progress_interval = 25 if total_metrics >= 100 else 10 if total_metrics >= 30 else 0
        started_at = time.time()

        joined_text: Optional[str] = None
        freq_counter: Optional[Counter] = None
        lex_obj: Optional[LexicalRichness] = None

        try:
            for index, metric_name in enumerate(metric_names, start=1):
                if metric_name not in MetricConfig.METRICS:
                    continue

                config = MetricConfig.METRICS[metric_name]

                if self._metric_is_skipped(metric_name):
                    metrics[metric_name] = None
                else:
                    method_name = config.get('compute_method', '')
                    metric_started_at = time.time()

                    if metric_name in ('word_burstiness', 'hapax_ratio', 'dis_ratio') and freq_counter is None:
                        freq_counter = Counter(tokens) if tokens else Counter()

                    if metric_name in ('MATTR', 'yules_k', 'MATTR_centroid') and lex_obj is None:
                        if joined_text is None:
                            joined_text = ' '.join(tokens) if tokens else ''
                        lex_obj = LexicalRichness(joined_text) if joined_text else None

                    metrics[metric_name] = self._compute_single_metric(
                        metric_name, config, tokens, chunks, freq_counter, lex_obj
                    )

                    metric_elapsed = time.time() - metric_started_at
                    if metric_elapsed >= 8.0:
                        print(f"      slow metric: {metric_name} ({metric_elapsed:.1f}s, method={method_name})")

                if progress_interval and (index % progress_interval == 0 or index == total_metrics):
                    elapsed = time.time() - started_at
                    print(f"      progress: {index}/{total_metrics} metrics ({elapsed:.1f}s)")

            return metrics
        finally:
            self._clear_runtime_cache()

    def _compute_single_metric(self, metric_name: str, config: Dict, tokens: List[str],
                               chunks: List[str],
                               freq_counter: Counter, lex_obj: Optional[LexicalRichness]) -> any:
        try:
            method_name = config['compute_method']

            if method_name.startswith('_compute'):
                method = getattr(self, method_name)
                if metric_name.startswith('letter_') and metric_name.endswith('_count'):
                    letter = metric_name.split('_')[1]
                    return method(tokens, letter=letter)
                elif method_name == '_compute_dynamic_token_count':
                    return method(tokens, metric_name=metric_name)
                elif metric_name in ('word_burstiness', 'hapax_ratio', 'dis_ratio'):
                    return method(tokens, freq_counter=freq_counter)
                elif metric_name in ('MATTR', 'yules_k', 'MATTR_centroid'):
                    return method(tokens, lex=lex_obj)
                elif method_name == '_compute_MTLD_score':
                    return method(tokens)
                else:
                    metric_kwargs = {
                        key: config[key]
                        for key in ('lexicon_key', 'phrase_key')
                        if key in config
                    }
                    return method(tokens, **metric_kwargs)
            elif method_name == 'compute_insult_density':
                return self.processor.compute_insult_density(tokens)
            elif method_name == 'calculate_perplexity':
                return self.processor.calculate_perplexity(chunks)
            elif method_name == 'compute_vader_sentiment':
                vader_results = self.processor.compute_vader_sentiment(chunks)
                return vader_results[config.get('result_index', 0)]
            else:
                print(f"    Warning: Unknown compute method '{method_name}' for '{metric_name}'")
                return None
        except Exception as e:
            print(f"    Error computing {metric_name}: {e}")
            return None

    @staticmethod
    def _low_token_result(mean_emb: np.ndarray, token_count: int, file_count: int) -> Dict[str, Any]:
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

    def process_text(self, text: str, ignore_ids: Optional[set] = None, **filter_kwargs) -> Optional[Dict]:
        if ignore_ids is None:
            ignore_ids = set()

        if not validate_text_name(text):
            print(f"\nError: Invalid text name '{text}' rejected for security")
            return None

        print(f"Processing: {text}")
        filters = {k: v for k, v in filter_kwargs.items() if v is not None}
        cache_filters = {k: v for k, v in filters.items() if k in CACHE_FILTER_KEYS}
        cache_filters.update(self.embedding_cache_signature)

        safe_text = sanitize_filename(text)
        txt_dir = self.processor.base_dir / "data/input" / safe_text / "txt_files"
        if not txt_dir.exists() and not self.processor.has_external_document(text):
            print(f"  Skipped: {text} (no txt_files directory)")
            return None

        if self.centroid_mode:
            return self._process_text_centroid(text, filters, ignore_ids, **filter_kwargs)

        result = self.processor.get_tokens(text, filters, ignore_ids)
        if not result:
            return None

        tokens, txt_dir, file_count, video_ids = result
        centroid_tokens = tokens
        style_rate_profile = self._burrows_rate_profile(centroid_tokens)
        embeddings_only_low_token = filter_kwargs.get('embeddings_only_low_token', False)

        self._centroid_equals_stat[text] = True

        if self.compute_fighting_words:
            self._fighting_words_unigrams[text] = Counter(tokens)
            self._fighting_words_token_counts[text] = len(tokens)
            self._fighting_words_unigrams_centroid[text] = Counter(centroid_tokens)
            self._fighting_words_token_counts_centroid[text] = len(centroid_tokens)

        min_tokens_total = filters.get('min_tokens_total', 0)
        if len(tokens) < min_tokens_total:
            if embeddings_only_low_token:
                print(f"  Low tokens: {text} ({len(tokens):,} tokens, minimum is {min_tokens_total:,}) - generating embeddings only")

                cached_embeddings = None
                cached_mean_emb = None
                cached_chunks = None
                if self.cache:
                    cached = self.cache.load(text, cache_filters, video_ids,
                                             enable_spacy=self.enable_spacy,
                                             needs_perplexity_model=self.processor.needs_perplexity_model,
                                             enable_ngram_entropy=self.enable_ngram_entropy,
                                             fast_mode=self.fast_mode,
                                             expected_token_count=len(tokens),
                                             embeddings_only=True,
                                             force_latest_cache=self.force_latest_cache)
                    if cached is not None:
                        cached_embeddings = cached.get('embeddings')
                        cached_mean_emb = cached.get('mean_emb')
                        cached_chunks = cached.get('chunks')

                if cached_chunks is not None and cached_embeddings is not None:
                    chunks = cached_chunks
                    chunk_embs = cached_embeddings
                    mean_emb = cached_mean_emb if cached_mean_emb is not None else np.mean(chunk_embs, axis=0).astype(np.float32)
                    print(f"  Cached: {text} (using cached embeddings)")
                else:
                    chunks = self.processor.chunk(tokens)
                    if not chunks:
                        print(f"  Skipped: {text} (no chunks generated)")
                        return None
                    chunk_embs = self._encode_embeddings(chunks)
                    mean_emb = np.mean(chunk_embs, axis=0).astype(np.float32)

                result_dict = {
                    'mean_emb': mean_emb,
                    'token_count': len(tokens),
                    'chunk_count': len(chunks),
                    'file_count': file_count,
                    'low_token_text': True
                }

                if self.cache and (cached_chunks is None or cached_embeddings is None):
                    self.cache.save(text, cache_filters, result_dict, video_ids, len(tokens), len(chunks),
                                    chunk_embs=chunk_embs, mean_emb=mean_emb, chunks=chunks,
                                    enable_spacy=self.enable_spacy,
                                    needs_perplexity_model=self.processor.needs_perplexity_model,
                                    enable_ngram_entropy=self.enable_ngram_entropy,
                                    fast_mode=self.fast_mode)

                print(f"  Complete: {text} ({len(tokens):,} tokens, {len(chunks)} chunks, {file_count} files) [embeddings only]")
                return result_dict
            else:
                print(f"  Skipped: {text} ({len(tokens):,} tokens, minimum is {min_tokens_total:,})")
                return None

        cached_metrics = {}
        missing_metrics = []
        cached_embeddings = None
        cached_mean_emb = None
        cached_chunks = None
        target_metric_names = self._get_target_metric_names(skip_centroid_duplicates=True)

        if self.cache:
            cached = self.cache.load(text, cache_filters, video_ids,
                                     enable_spacy=self.enable_spacy,
                                     needs_perplexity_model=self.processor.needs_perplexity_model,
                                     enable_ngram_entropy=self.enable_ngram_entropy,
                                     fast_mode=self.fast_mode,
                                     expected_token_count=len(tokens),
                                     embeddings_only=self.no_metrics,
                                     force_latest_cache=self.force_latest_cache)
            if cached is not None:
                cached_metrics = cached.get('metrics', {})
                cached_embeddings = cached.get('embeddings')
                cached_mean_emb = cached.get('mean_emb')
                cached_chunks = cached.get('chunks')
                missing_metrics = [m for m in target_metric_names if self._metric_needs_compute(m, cached_metrics)]
            else:
                missing_metrics = list(target_metric_names)

        if cached_chunks is not None and cached_embeddings is not None:
            chunks = cached_chunks
            chunk_embs = cached_embeddings
            mean_emb = cached_mean_emb
            print(f"  Cached: {text} (using cached embeddings)")
        else:
            chunks = self.processor.chunk(tokens)
            if not chunks:
                print(f"  Skipped: {text} (no chunks generated)")
                return None
            chunk_embs = self._encode_embeddings(chunks)
            mean_emb = np.mean(chunk_embs, axis=0).astype(np.float32)

        metric_report_values: Dict[str, Any] = {}
        metric_report_source = 'computed'

        if missing_metrics:
            print(f"    -> Computing {len(missing_metrics)} new metrics")
            new_metrics = self._compute_metrics_subset(tokens, chunks, missing_metrics)
            all_metrics = {**cached_metrics, **new_metrics}
            metric_report_values = new_metrics
            metric_report_source = 'computed'
        else:
            if cached_metrics:
                all_metrics = cached_metrics
                metric_report_values = cached_metrics
                metric_report_source = 'cache'
            else:
                all_metrics = self._compute_all_metrics(tokens, chunks, chunk_embs)
                metric_report_values = all_metrics
                metric_report_source = 'computed'

        if not self.no_metrics:
            self._print_metric_completion_summary(metric_report_values, metric_report_source)

        result_dict = {
            'mean_emb': mean_emb,
            'token_count': len(tokens),
            'chunk_count': len(chunks),
            'file_count': file_count,
            '_burrows_rate_profile': style_rate_profile,
            **all_metrics
        }

        if self.cache:
            self.cache.save(text, cache_filters, result_dict, video_ids, len(tokens), len(chunks),
                            chunk_embs=chunk_embs, mean_emb=mean_emb, chunks=chunks,
                            enable_spacy=self.enable_spacy,
                            needs_perplexity_model=self.processor.needs_perplexity_model,
                            enable_ngram_entropy=self.enable_ngram_entropy,
                            fast_mode=self.fast_mode)

        compression = all_metrics.get('compression')
        if isinstance(compression, (float, int)):
            print(
                f"  Complete: {text} ({len(tokens):,} tokens, {len(chunks)} chunks, {file_count} files, "
                f"CR: {_format_trimmed_decimal(float(compression), 4)})"
            )
        else:
            print(f"  Complete: {text} ({len(tokens):,} tokens, {len(chunks)} chunks, {file_count} files)")

        return result_dict

    def _process_text_centroid(self, text: str, filters: Dict,
                                ignore_ids: Optional[set] = None, **filter_kwargs):
        if ignore_ids is None:
            ignore_ids = set()
        print(f"Processing (centroid mode={self.centroid_mode}): {text}")

        cache_filters = {k: v for k, v in filters.items() if k in CACHE_FILTER_KEYS}
        cache_filters['centroid_mode'] = self.centroid_mode
        cache_filters['centroid_videos'] = int(self.centroid_videos)
        cache_filters['centroid_words'] = int(self.centroid_words)
        cache_filters['stat_word_count'] = int(self.stat_word_count)
        cache_filters.update(self.embedding_cache_signature)

        if self.centroid_mode == 'video':
            min_stat_tokens = (
                int(self.stat_word_count)
                if (self.stat_word_count and self.stat_word_count > 0)
                else 0
            )
            token_cap = min_stat_tokens if min_stat_tokens > 0 else None
            enforce_min_tokens = (not self.no_metrics) and min_stat_tokens > 0

            all_filtered_files = self.processor._get_filtered_files(
                text, filters, ignore_ids
            )
            if not all_filtered_files:
                print(f"  Skipped: {text} (no txt_files directory or no files passed filters)")
                return None
            video_ids_for_cache = [
                extract_video_id(f.name) for _, f in all_filtered_files
            ]
            n_files = len(video_ids_for_cache)

            target_metric_names = self._get_target_metric_names()
            stat_metric_names = [m for m in target_metric_names if not MetricConfig.is_centroid_metric(m)]
            centroid_metric_names = [m for m in target_metric_names if MetricConfig.is_centroid_metric(m)]

            cached_metrics: Dict[str, Any] = {}
            text_emb: Optional[np.ndarray] = None
            cached_is_low_token: Optional[bool] = None
            missing_metrics: List[str] = list(target_metric_names)
            cache_kwargs = dict(
                enable_spacy=self.enable_spacy,
                needs_perplexity_model=self.processor.needs_perplexity_model,
                enable_ngram_entropy=self.enable_ngram_entropy,
                fast_mode=self.fast_mode,
            )

            if self.cache:
                cached = self.cache.load(
                    text, cache_filters, video_ids_for_cache,
                    embeddings_only=self.no_metrics,
                    force_latest_cache=self.force_latest_cache, **cache_kwargs,
                )
                if cached is not None and cached.get('mean_emb') is not None:
                    text_emb = cached['mean_emb']
                    cached_metrics = cached.get('metrics', {}) or {}
                    cached_tokens = cached.get('token_count', 0)
                    cached_is_low_token = bool(
                        enforce_min_tokens and cached_tokens < min_stat_tokens
                    )

                    if cached_is_low_token:
                        needed_metrics = [
                            m for m in target_metric_names
                            if MetricConfig.is_centroid_metric(m)
                        ]
                    else:
                        needed_metrics = list(target_metric_names)

                    missing_metrics = [
                        m for m in needed_metrics
                        if self._metric_needs_compute(m, cached_metrics)
                    ]

                    if self.no_metrics or (not missing_metrics and not self.compute_fighting_words):
                        if not self.no_metrics:
                            self._print_metric_completion_summary(cached_metrics, 'cache')
                        result_metrics = dict(cached_metrics)
                        if cached_is_low_token:
                            for m in stat_metric_names:
                                result_metrics.pop(m, None)
                        print(f"  Complete: {text} ({cached_tokens:,} tokens, cached)")

                        return {
                            'mean_emb': text_emb,
                            'token_count': cached_tokens,
                            'chunk_count': 0,
                            'file_count': n_files,
                            'low_token_text': bool(cached_is_low_token),
                            '_centroid_only': bool(cached_is_low_token),
                            **({} if self.no_metrics else result_metrics),
                        }

                    print(f"    -> {len(missing_metrics)} metrics missing; computing only those")

            result = self.processor.load_video_centroids(
                text, self.centroid_videos, filters, ignore_ids,
                max_accumulated_tokens=token_cap,
                precomputed_files=all_filtered_files,
                need_embeddings=text_emb is None,
                need_stat_corpus=not self.no_metrics,
                need_centroid_tokens=not self.no_metrics,
            )
            if not result:
                print(f"  Skipped: {text} (no video centroids)")
                return None
            video_ids, video_embeddings, stat_tokens, centroid_tokens, avg_words_per_video = result

            if text_emb is None:
                if not video_embeddings:
                    print(f"  Skipped: {text} (no embeddings)")
                    return None
                text_emb = np.mean(video_embeddings, axis=0).astype(np.float32)
                video_embeddings.clear()

            centroid_equals_stat = self._token_lists_are_identical(centroid_tokens, stat_tokens)
            self._centroid_equals_stat[text] = centroid_equals_stat
            is_low_token = bool(enforce_min_tokens and len(stat_tokens) < min_stat_tokens)

            if self.compute_fighting_words:
                if not is_low_token:
                    self._fighting_words_unigrams[text] = Counter(stat_tokens)
                    self._fighting_words_token_counts[text] = len(stat_tokens)

                if centroid_equals_stat and not is_low_token:
                    self._fighting_words_unigrams_centroid[text] = self._fighting_words_unigrams[text]
                    self._fighting_words_token_counts_centroid[text] = self._fighting_words_token_counts[text]
                else:
                    self._fighting_words_unigrams_centroid[text] = Counter(centroid_tokens)
                    self._fighting_words_token_counts_centroid[text] = len(centroid_tokens)

            chunks: List[str] = []
            stat_metrics: Dict[str, Any] = {}
            centroid_metrics: Dict[str, Any] = {}

            if not self.no_metrics and centroid_tokens:
                centroid_missing = [m for m in missing_metrics if m in centroid_metric_names]
                if centroid_equals_stat and not is_low_token:
                    centroid_missing = [
                        m for m in centroid_missing
                        if not MetricConfig.is_centroid_duplicate(m)
                    ]

                if is_low_token:
                    for m in missing_metrics:
                        if m in ('ngram_entropy_2', 'ngram_entropy_3') and m not in centroid_missing:
                            centroid_missing.append(m)

                if centroid_missing:
                    needs_centroid_chunks = any(
                        MetricConfig.METRICS.get(m, {}).get('compute_method')
                        in ('calculate_perplexity', 'compute_vader_sentiment')
                        for m in centroid_missing
                    )
                    centroid_chunks = self.processor.chunk(centroid_tokens) if needs_centroid_chunks else []
                    print(f"    -> Computing {len(centroid_missing)} centroid metrics from {len(centroid_tokens):,} tokens")
                    centroid_metrics = self._compute_metrics(centroid_tokens, centroid_chunks, centroid_missing)

            if 'avg_words_per_video' in centroid_metric_names and 'avg_words_per_video' not in cached_metrics:
                centroid_metrics['avg_words_per_video'] = avg_words_per_video

            if not self.no_metrics and not is_low_token and stat_tokens:
                stat_missing = [m for m in missing_metrics if m in stat_metric_names]
                if stat_missing:
                    needs_stat_chunks = any(
                        MetricConfig.METRICS.get(m, {}).get('compute_method')
                        in ('calculate_perplexity', 'compute_vader_sentiment')
                        for m in stat_missing
                    )
                    stat_chunks = self.processor.chunk(stat_tokens) if needs_stat_chunks else []
                    if needs_stat_chunks:
                        chunks = stat_chunks
                    print(f"    -> Computing {len(stat_missing)} stat metrics from {len(stat_tokens):,} tokens")
                    stat_metrics = self._compute_metrics(stat_tokens, stat_chunks, stat_missing)

            new_metrics = {**stat_metrics, **centroid_metrics}
            if new_metrics:
                self._print_metric_completion_summary(new_metrics, 'computed')

            all_metrics = {**cached_metrics, **new_metrics}
            if is_low_token:
                for m in stat_metric_names:
                    all_metrics.pop(m, None)

            result_dict = {
                'mean_emb': text_emb,
                'token_count': len(stat_tokens),
                'chunk_count': len(chunks),
                'file_count': n_files,
                '_burrows_rate_profile': self._burrows_rate_profile(centroid_tokens),
                '_centroid_equals_stat': centroid_equals_stat,
                '_centroid_only': is_low_token,
                **all_metrics,
            }

            if self.cache:
                self.cache.save(text, cache_filters, result_dict, video_ids_for_cache,
                                len(stat_tokens), len(chunks), chunk_embs=None,
                                mean_emb=text_emb, chunks=None, **cache_kwargs)

            if is_low_token:
                print(f"  Complete: {text} ({len(stat_tokens):,} tokens, centroid metrics only)")
            else:
                print(f"  Complete: {text} ({len(stat_tokens):,} tokens, {len(video_ids) or n_files} videos)")
            return result_dict

        elif self.centroid_mode == 'word':
            result = self.processor.get_tokens(text, filters, ignore_ids)
            if not result:
                return None
            tokens, txt_dir, file_count, video_ids = result

            word_centroid_tokens = (
                tokens[-self.centroid_words:]
                if len(tokens) > self.centroid_words
                else tokens
            )
            stat_tokens = tokens
            centroid_tokens = word_centroid_tokens
            centroid_equals_stat = self._token_lists_are_identical(centroid_tokens, stat_tokens)
            self._centroid_equals_stat[text] = centroid_equals_stat

            if self.compute_fighting_words:
                self._fighting_words_unigrams[text] = Counter(stat_tokens)
                self._fighting_words_token_counts[text] = len(stat_tokens)
                if centroid_equals_stat:
                    self._fighting_words_unigrams_centroid[text] = self._fighting_words_unigrams[text]
                    self._fighting_words_token_counts_centroid[text] = self._fighting_words_token_counts[text]
                else:
                    self._fighting_words_unigrams_centroid[text] = Counter(centroid_tokens)
                    self._fighting_words_token_counts_centroid[text] = len(centroid_tokens)

            chunks = self.processor.chunk(word_centroid_tokens)
            if not chunks and word_centroid_tokens:
                chunks = [' '.join(word_centroid_tokens)]
            if not chunks:
                print(f"  Skipped: {text} (no chunks generated)")
                return None

            chunk_embs = self._encode_embeddings(chunks)
            text_emb = np.mean(chunk_embs, axis=0).astype(np.float32)

            all_metrics = {}
            if not self.no_metrics:
                target_metric_names = self._get_target_metric_names(
                    skip_centroid_duplicates=centroid_equals_stat,
                    skip_video_centroid_only=True,
                )
                stat_metric_names = [m for m in target_metric_names if not MetricConfig.is_centroid_metric(m)]
                centroid_metric_names = [m for m in target_metric_names if MetricConfig.is_centroid_metric(m)]

                if target_metric_names:
                    corpus_tokens = len(word_centroid_tokens)
                    unique_tokens = len(set(word_centroid_tokens))
                    tt_ratio = (
                        unique_tokens / corpus_tokens
                        if corpus_tokens else 0.0
                    )
                    print(
                        f"    -> Stats corpus: {corpus_tokens:,} tokens, "
                        f"{unique_tokens:,} unique (TTR={tt_ratio:.4f}), "
                        f"centroid_words_cap={self.centroid_words:,}, "
                        f"chunks={len(chunks):,}"
                    )
                    print(
                        f"    -> Computing {len(target_metric_names)} metrics "
                        f"from {len(word_centroid_tokens):,} tokens"
                    )

                    stat_chunks_for_metrics: List[str] = []
                    needs_stat_chunks = any(
                        MetricConfig.METRICS.get(m, {}).get('compute_method') in ('calculate_perplexity', 'compute_vader_sentiment')
                        for m in stat_metric_names
                    )
                    if needs_stat_chunks:
                        stat_chunks_for_metrics = self.processor.chunk(stat_tokens) or [' '.join(stat_tokens)]

                    stat_metrics = self._compute_metrics(stat_tokens, stat_chunks_for_metrics, stat_metric_names) if stat_metric_names else {}

                    centroid_chunks_for_metrics: List[str] = []
                    needs_centroid_chunks = any(
                        MetricConfig.METRICS.get(m, {}).get('compute_method') in ('calculate_perplexity', 'compute_vader_sentiment')
                        for m in centroid_metric_names
                    )
                    if needs_centroid_chunks:
                        centroid_chunks_for_metrics = self.processor.chunk(centroid_tokens) or [' '.join(centroid_tokens)]

                    centroid_metrics = self._compute_metrics(centroid_tokens, centroid_chunks_for_metrics, centroid_metric_names) if centroid_metric_names else {}

                    all_metrics = {**stat_metrics, **centroid_metrics}
                    self._print_metric_completion_summary(all_metrics, 'computed')

            result_dict = {
                'mean_emb': text_emb,
                'token_count': len(stat_tokens),
                'chunk_count': len(chunks),
                'file_count': file_count,
                'chunk_embs': chunk_embs,
                '_burrows_rate_profile': self._burrows_rate_profile(centroid_tokens),
                '_centroid_equals_stat': centroid_equals_stat,
                **all_metrics
            }

            if self.cache:
                self.cache.save(
                    text, cache_filters, result_dict, video_ids,
                    len(stat_tokens), len(chunks),
                    chunk_embs=chunk_embs, mean_emb=text_emb, chunks=chunks,
                    enable_spacy=self.enable_spacy,
                    needs_perplexity_model=self.processor.needs_perplexity_model,
                    enable_ngram_entropy=self.enable_ngram_entropy,
                    fast_mode=self.fast_mode,
                )

            print(f"  Complete: {text} ({len(stat_tokens):,} tokens, {len(chunks)} chunks)")
            return result_dict

        else:
            print(f"  Skipped: {text} (unknown centroid mode: {self.centroid_mode})")
            return None

    def reduce_dimensions(self, embeddings: np.ndarray, n_texts: int) -> np.ndarray:
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
            n_neighbors = min(5 if n_texts <= 30 else 50, n_texts - 1)
        else:
            n_neighbors = min(self.umap_neighbors, n_texts - 1)
        n_neighbors = max(2, n_neighbors)

        reducer = umap.UMAP(
            n_components=2, metric='cosine', n_neighbors=n_neighbors,
            min_dist=self.umap_min_dist, init='spectral', random_state=20, n_jobs=1,
            verbose=False, n_epochs=self.umap_epochs, low_memory=False
        )
        return reducer.fit_transform(embeddings)

    def _cluster_embeddings_default(self, embeddings: np.ndarray, full_token_indices: List[int], random_seed: int = 42) -> List[Optional[int]]:
        cluster_labels: List[Optional[int]] = [None] * len(embeddings)

        if not full_token_indices:
            return cluster_labels

        if len(full_token_indices) == 1:
            cluster_labels[full_token_indices[0]] = 0
            return cluster_labels

        full_token_embeddings = embeddings[full_token_indices]
        min_cluster_size = max(2, min(20, len(full_token_indices) // 20 or 2))
        min_samples = max(1, min_cluster_size // 2)

        labels: Optional[np.ndarray] = None

        try:
            rng = np.random.default_rng(random_seed)
            embedding_std = float(np.std(full_token_embeddings))
            jitter_scale = max(1e-8, embedding_std * 1e-4)
            jitter = rng.normal(0.0, jitter_scale, size=full_token_embeddings.shape).astype(np.float32)
            clustering_input = full_token_embeddings + jitter
        except Exception:
            clustering_input = full_token_embeddings

        try:
            if SklearnHDBSCAN is not None:
                labels = SklearnHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean'
                ).fit_predict(clustering_input)
        except Exception:
            labels = None

        if labels is None or np.all(labels == -1):
            fallback_clusters = max(2, min(14, int(np.sqrt(len(full_token_indices)))))
            labels = KMeans(n_clusters=fallback_clusters, random_state=random_seed, n_init=10).fit_predict(clustering_input)

        for idx, text_index in enumerate(full_token_indices):
            cluster_labels[text_index] = int(labels[idx])

        return cluster_labels

    def _cluster_embeddings_leiden(
        self,
        embeddings: np.ndarray,
        full_token_indices: List[int],
        resolution: float = 1.0,
        random_seed: int = 42,
    ) -> List[Optional[int]]:
        cluster_labels: List[Optional[int]] = [None] * len(embeddings)

        if not full_token_indices:
            return cluster_labels

        if len(full_token_indices) == 1:
            cluster_labels[full_token_indices[0]] = 0
            return cluster_labels

        if not LEIDEN_AVAILABLE:
            print("    Warning: leidenalg/igraph not available — falling back to HDBSCAN")
            return self._cluster_embeddings_default(embeddings, full_token_indices, random_seed=random_seed)

        try:
            from sklearn.decomposition import PCA
            from sklearn.neighbors import kneighbors_graph
        except ImportError as exc:
            print(f"    Warning: sklearn dependency missing for Leiden ({exc}) — falling back to HDBSCAN")
            return self._cluster_embeddings_default(embeddings, full_token_indices, random_seed=random_seed)

        X = embeddings[full_token_indices].astype(np.float32)
        n = X.shape[0]

        if self.leiden_whiten and X.shape[1] > 1:
            n_components = min(256, X.shape[1] - 1, n - 1)
            if n_components >= 2:
                try:
                    X = PCA(
                        n_components=n_components,
                        whiten=True,
                        random_state=random_seed,
                    ).fit_transform(X).astype(np.float32)
                except Exception as exc:
                    print(f"    Warning: PCA whitening failed ({exc}); using raw embeddings")

        k_eff = max(2, min(int(self.leiden_k_neighbors), n - 1))
        A = None
        for metric_name in ('cosine', 'euclidean'):
            try:
                A = kneighbors_graph(
                    X, k_eff, mode='connectivity',
                    metric=metric_name, include_self=False,
                )
                A = A.maximum(A.T)
                break
            except Exception as exc:
                print(f"    Warning: k-NN graph with metric={metric_name} failed ({exc})")
                A = None
        if A is None:
            print("    Warning: could not build k-NN graph — falling back to HDBSCAN")
            return self._cluster_embeddings_default(embeddings, full_token_indices, random_seed=random_seed)

        try:
            sources, targets = A.nonzero()
            edges = [(int(s), int(t)) for s, t in zip(sources, targets) if s < t]
            if not edges:
                print("    Warning: k-NN graph produced no edges — falling back to HDBSCAN")
                return self._cluster_embeddings_default(embeddings, full_token_indices, random_seed=random_seed)

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
            return self._cluster_embeddings_default(embeddings, full_token_indices, random_seed=random_seed)

        for idx, text_index in enumerate(full_token_indices):
            cluster_labels[text_index] = int(labels[idx])

        return cluster_labels

    def _cluster_embeddings_variants(
        self,
        embeddings: np.ndarray,
        full_token_indices: List[int],
        variant_count: int = 11,
        base_seed: int = 42,
    ) -> Dict[str, List[Optional[int]]]:
        if self.cluster_method == 'leiden':
            return self._cluster_embeddings_variants_leiden(
                embeddings, full_token_indices,
                variant_count=variant_count, base_seed=base_seed,
            )

        total_variants = max(1, int(variant_count))
        variants: Dict[str, List[Optional[int]]] = {}

        for variant_index in range(total_variants):
            metric_key = 'cluster' if variant_index == 0 else f'cluster_{variant_index + 1}'
            seed = base_seed + variant_index
            variants[metric_key] = self._cluster_embeddings_default(
                embeddings,
                full_token_indices,
                random_seed=seed,
            )

        return variants

    def _cluster_embeddings_variants_leiden(
        self,
        embeddings: np.ndarray,
        full_token_indices: List[int],
        variant_count: int = 11,
        base_seed: int = 42,
    ) -> Dict[str, List[Optional[int]]]:
        total_variants = max(1, int(variant_count))

        if total_variants == 1:
            resolutions = [1.0]
        else:
            resolutions = np.linspace(
                self.leiden_resolution_min,
                self.leiden_resolution_max,
                total_variants,
            ).tolist()

        variants: Dict[str, List[Optional[int]]] = {}
        labels_map: Dict[str, str] = {}

        for index, resolution in enumerate(resolutions):
            metric_key = 'cluster' if index == 0 else f'cluster_{index + 1}'
            variants[metric_key] = self._cluster_embeddings_leiden(
                embeddings,
                full_token_indices,
                resolution=float(resolution),
                random_seed=base_seed,
            )
            labels_map[metric_key] = f'Leiden r={resolution:.2f}'

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
            writer.writerow(header)

            for i, text in enumerate(texts):
                sims = self.similarity_matrix[i].copy()
                sims[i] = -1
                sorted_idx = np.argsort(sims)[::-1]

                n_avail = min(n, len(texts) - 1)
                closest = [(texts[idx], sims[idx]) for idx in sorted_idx[:n_avail]]
                farthest = [(texts[idx], sims[idx]) for idx in sorted_idx[-n_avail:][::-1]]

                row = [text]
                for j in range(n):
                    row.extend([closest[j][0], _format_trimmed_decimal(closest[j][1], 4)] if j < len(closest) else ['', ''])
                for j in range(n):
                    row.extend([farthest[j][0], _format_trimmed_decimal(farthest[j][1], 4)] if j < len(farthest) else ['', ''])
                writer.writerow(row)

        print(f"  Exported: {output_path.absolute()}")

    def visualize(self, reduced: np.ndarray, texts: List[str],
                metrics_data: Dict[str, np.ndarray],
                active_metric: Optional[str] = None,
                output_html: str = ".html",
                low_token_texts: Optional[set] = None,
                enable_annotations: bool = False,
                focus_text: Optional[str] = None,
                semantic_embeddings: Optional[List[List[float]]] = None,
                burrows_similarity: Optional[List[List[float]]] = None):
        if low_token_texts is None:
            low_token_texts = set()
        if focus_text is None:
            focus_text = self.active_focus_text if hasattr(self, 'active_focus_text') else ''
        if semantic_embeddings is None:
            semantic_embeddings = []
        if burrows_similarity is None:
            burrows_similarity = []

        exclude_from_hover = {
            'latinate_word_ratio', 'coordinate_clause_ratio', 'subordinate_clause_ratio', 'imperative_exclamation_density', 'academic_word_density',
            'article_count_raw', 'conjunction_count_raw', 'vulnerability', 'family_exclamations', 'pronoun_article_ratio',
            'imperative_exclamation_density', 'deictic_spatial_temporal', 'elaboration_explanation_ratio', 'narration_continuation_ratio',
            'insult', 'bodily_humor', 'dis_ratio',
            'word_burstiness', 'semantic_disparity', 'guppa', 'pronoun_switching',
            'compression', 'moving_lzma_cr', 'normalized_compression_ratio', 'window_normalized_lzma_cr',
            'vader_positivity', 'vader_volatility', 'vader_positivity_centroid', 'vader_volatility_centroid',
            'fighting_words_floor_z1000', 'fighting_words_floor_z1000_centroid',
        }
        exclude_from_hover.update({
            metric_name for metric_name in MetricConfig.METRICS
            if metric_name.endswith('_count') or metric_name.endswith('_density') or metric_name.startswith('mallet_topic_')
        })

        def wrap_long_lines(text: str, max_length: int = 75) -> str:
            lines = text.split('<br>')
            wrapped_lines = []
            for line in lines:
                clean_line = line.replace('<b>', '').replace('</b>', '')
                if len(clean_line) > max_length:
                    words = clean_line.split(' ')
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

        hover_texts = []
        programmer_notes = {}

        for i, text in enumerate(texts):
            stats_column = (
                f"<b>{text}</b><br>"
                f"X: {_format_trimmed_decimal(reduced[i, 0], 4)}<br>"
                f"Y: {_format_trimmed_decimal(reduced[i, 1], 4)}<br>"
            )

            is_low_token = text in low_token_texts
            if not is_low_token:
                for metric_name, metric_values in metrics_data.items():
                    if metric_values is not None and metric_name not in exclude_from_hover and not metric_name.startswith('cluster'):
                        per_text_value = metric_values[i]
                        if per_text_value is None or np.isnan(per_text_value):
                            continue
                        display_name = MetricConfig.METRICS.get(metric_name, {}).get('name', metric_name)
                        if metric_name in [
                            'MTLD', 'TTR', 'MATTR', 'yules_k',
                            'lexical_density', 'hapax_ratio',
                            'avg_word_length', 'semantic_disparity', 'burrows_cosine_disagreement',
                            'MTLD_centroid', 'MATTR_centroid',
                        ]:
                            stats_column += f"{display_name}: {_format_trimmed_decimal(metric_values[i], 4)}<br>"
                        elif metric_name in ['ngram_entropy_2', 'ngram_entropy_3']:
                            stats_column += f"{display_name}: {_format_trimmed_decimal(metric_values[i], 4)}<br>"
                        elif metric_name in ['perplexity', 'tfidf_distinctiveness']:
                            stats_column += f"{display_name}: {metric_values[i]:.2f}<br>"
                        elif metric_name == 'flesch_kincaid':
                            stats_column += f"{display_name}: {metric_values[i]:.1f}<br>"
                        else:
                            stats_column += f"{display_name}: {int(metric_values[i])}<br>"

            if enable_annotations:
                message = None
                try:
                    base_dir = self.processor.base_dir
                    safe_text = sanitize_filename(text)
                    tx_dir = base_dir / "data/input" / safe_text
                    message = None

                    for fname in ["additional_message.txt", "hover_message.txt", "message.txt", "note.txt"]:
                        if not fname.endswith('.txt'):
                            continue
                        fpath = tx_dir / fname
                        if fpath.exists():
                            with open(fpath, 'r', encoding='utf-8') as mf:
                                message = mf.read().strip()
                            break

                    if message is None:
                        meta_path = tx_dir / "metadata.json"
                        if meta_path.exists():
                            try:
                                with open(meta_path, 'r', encoding='utf-8') as mf:
                                    meta = json.load(mf)
                                if isinstance(meta, dict):
                                    for key in ["additional_message", "hover_message", "message", "note"]:
                                        if key in meta and isinstance(meta[key], str) and meta[key].strip():
                                            message = meta[key].strip()
                                            break
                                elif isinstance(meta, list):
                                    for item in meta:
                                        if not isinstance(item, dict):
                                            continue
                                        if item.get("type") == "text" or item.get("id") == "text":
                                            for key in ["additional_message", "hover_message", "message", "note"]:
                                                if key in item and isinstance(item[key], str) and item[key].strip():
                                                    message = item[key].strip()
                                                    break
                                            if message:
                                                break
                                        if item.get("scope") == "text":
                                            for key in ["additional_message", "hover_message", "message", "note"]:
                                                if key in item and isinstance(item[key], str) and item[key].strip():
                                                    message = item[key].strip()
                                                    break
                                            if message:
                                                break
                            except Exception:
                                pass

                    if message:
                        programmer_notes[text] = message
                        text_content = f"{stats_column}<span style='font-weight: bold;'>Click dot for Annotation</span>"
                    else:
                        text_content = stats_column
                except Exception:
                    text_content = stats_column
            else:
                text_content = stats_column

            text_content = wrap_long_lines(text_content)
            hover_texts.append(text_content)

        full_token_indices = list(range(len(texts)))
        full_token_reduced = reduced

        cluster_metric_keys = [
            key for key in metrics_data.keys()
            if key == 'cluster' or key.startswith('cluster_')
        ]
        cluster_metric_keys.sort(key=lambda key: 1 if key == 'cluster' else int(key.split('_')[1]))

        if active_metric and active_metric in MetricConfig.METRICS and active_metric in metrics_data and metrics_data[active_metric] is not None:
            metric_config = MetricConfig.METRICS[active_metric]
            metric_values = metrics_data[active_metric]
            full_token_values = [metric_values[i] for i in full_token_indices] if full_token_indices else metric_values
            color_label = metric_config.get('name', active_metric)
            color_scale = metric_config.get('colorscale', UNIFIED_COLORSCALE)
            title_suffix = metric_config.get('title_suffix', color_label)
            title = f"Text Semantic Map colored by {title_suffix}"
            color_data = [float(metric_values[i]) if i in full_token_indices else float('nan') for i in range(len(texts))]
        else:
            default_cluster_key = cluster_metric_keys[0] if cluster_metric_keys else 'cluster'
            precomputed_clusters = metrics_data.get(default_cluster_key)
            if precomputed_clusters is not None:
                color_data = [float(precomputed_clusters[i]) for i in range(len(texts))]
            else:
                color_data = [float('nan')] * len(texts)
            color_label = 'Cluster'
            color_scale = 'turbo'
            title = 'Mapping Texts with Semantics :O'
            active_metric = default_cluster_key

        if active_metric == 'cluster' or active_metric.startswith('cluster_'):
            color_data_for_range = [value for value in color_data if not np.isnan(value)]
            if color_data_for_range:
                explicit_min = np.nanmin(color_data_for_range)
                explicit_max = np.nanmax(color_data_for_range)
            else:
                explicit_min = 0.0
                explicit_max = 1.0
        elif full_token_indices:
            color_data_for_range = [
                color_data[i] for i in full_token_indices
                if not (color_data[i] is None or np.isnan(color_data[i]))
            ]
            if color_data_for_range:
                explicit_min = np.nanmin(color_data_for_range)
                explicit_max = np.nanmax(color_data_for_range)
            else:
                explicit_min = 0.0
                explicit_max = 1.0
        else:
            non_nan = [v for v in color_data if not np.isnan(v)]
            explicit_min = np.nanmin(non_nan) if non_nan else 0.0
            explicit_max = np.nanmax(non_nan) if non_nan else 1.0

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=reduced[:, 0], y=reduced[:, 1],
            mode='markers+text', text=texts, textposition="middle center",
            hovertext=hover_texts, hoverinfo="text",
            marker=dict(
                size=15,
                color=color_data,
                colorscale=color_scale,
                showscale=True,
                cmin=explicit_min,
                cmax=explicit_max,
                colorbar=dict(
                    title=color_label
                ),
                line=dict(width=1, color='black')
            ),
            textfont=dict(size=7, color='black')
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title="UMAP Component 1", yaxis_title="UMAP Component 2",
            hovermode='closest', showlegend=False, height=800, width=1200,
            font=dict(family='Arial, sans-serif')
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        output_path = Path(output_html).absolute()

        html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)

        html_str = re.sub(r'\s+integrity="[^"]*"', '', html_str)
        html_str = re.sub(r'\s+crossorigin="[^"]*"', '', html_str)

        metrics_json = {}
        titles_json = {}

        for cluster_metric_key in cluster_metric_keys:
            cluster_values = metrics_data.get(cluster_metric_key)
            if cluster_values is None:
                metrics_json[cluster_metric_key] = [None for _ in range(len(texts))]
            else:
                cluster_labels = [None] * len(texts)
                for i in range(len(texts)):
                    value = cluster_values[i]
                    if not np.isnan(value):
                        cluster_labels[i] = int(value)
                metrics_json[cluster_metric_key] = cluster_labels
            if cluster_metric_key in self.cluster_metric_labels:
                cluster_title_suffix = self.cluster_metric_labels[cluster_metric_key]
            else:
                cluster_title_suffix = cluster_metric_key.replace('_', ' ').title()
            titles_json[cluster_metric_key] = f'Text Semantic Clusters ({cluster_title_suffix})'

        metric_names_dict = {}
        metric_colorscale_dict = {}
        metric_keys_list = []

        for cluster_metric_key in cluster_metric_keys:
            if cluster_metric_key in self.cluster_metric_labels:
                metric_names_dict[cluster_metric_key] = self.cluster_metric_labels[cluster_metric_key]
            elif cluster_metric_key == 'cluster':
                metric_names_dict[cluster_metric_key] = 'Cluster 1 (default)'
            else:
                cluster_index = int(cluster_metric_key.split('_')[1])
                metric_names_dict[cluster_metric_key] = f'Cluster {cluster_index}'
            metric_colorscale_dict[cluster_metric_key] = 'Turbo'
            metric_keys_list.append(cluster_metric_key)

        non_contextual_metric_items = [
            (metric_name, MetricConfig.METRICS[metric_name])
            for metric_name in MetricConfig.get_metric_names()
            if MetricConfig.METRICS[metric_name].get('category') != 'Contextual Descriptions'
        ]

        for metric_name, metric_config in non_contextual_metric_items:
            metric_names_dict[metric_name] = metric_config.get('name', metric_name)
            metric_colorscale_dict[metric_name] = metric_config.get('colorscale', UNIFIED_COLORSCALE)
            metric_keys_list.append(metric_name)

        for metric_name, metric_values in metrics_data.items():
            if metric_values is not None:
                if metric_name == 'cluster' or metric_name.startswith('cluster_'):
                    continue
                metric_list = [float(metric_values[i]) if i in full_token_indices else None for i in range(len(texts))]
                metrics_json[metric_name] = metric_list
                metric_config = MetricConfig.METRICS.get(metric_name, {})
                titles_json[metric_name] = f"Text Semantic Clusters colored by {metric_config.get('title_suffix', metric_name)}"

        metrics_json_str = safe_json_dumps(metrics_json)
        titles_json_str = safe_json_dumps(titles_json)

        texts_json = safe_json_dumps(texts)

        programmer_notes_json = safe_json_dumps({tx: note for tx, note in programmer_notes.items()})
        annotations_enabled_json = safe_json_dumps(bool(enable_annotations))

        def _build_custom_style_and_script(
            metrics_json_str: str,
            titles_json_str: str,
            active_metric: str,
            texts_json: str,
            metric_names_dict: Dict[str, str],
            metric_colorscale_dict: Dict[str, str],
            metric_keys_list: List[str],
            programmer_notes_json: str,
            annotations_enabled_json: str,
            semantic_embeddings_json: str,
            burrows_similarity_json: str,
            texts: List[str],
            low_token_texts: set,
            focus_text: str,
        ) -> str:
            return f'''
    <style>
    * {{
        font-family: Arial, sans-serif;
    }}
    body, html, div, span, input, button, select, option, textarea {{
        font-family: Arial, sans-serif;
    }}

    .js-plotly-plot .hoverlayer,
    .plotly .hoverlayer {{
        z-index: 20050;
        pointer-events: none;
    }}

    @supports (paint-order: stroke) {{
        .plotly-stroke-text, .correlationStrokeText {{
            stroke: white;
            stroke-width: 1.5;
            stroke-opacity: 1;
            paint-order: stroke;
        }}
    }}

    .panelDragHandle {{
        cursor: move;
        user-select: none;
        -webkit-user-select: none;
    }}

    #noteModal {{
        display: none;
        position: fixed;
        z-index: 10000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
    }}
    #noteModalContent {{
        background-color: white;
        margin: 50px auto;
        padding: 10px;
        border: 0.5px solid black;
        width: 60%;
        max-height: 70vh;
        overflow-y: auto;
    }}
    #noteModalClose {{
        float: right;
        font-size: 20px;
        cursor: pointer;
    }}
    #noteModalBody {{
        white-space: pre-wrap;
        word-wrap: break-word;
    }}

    .plotly-graph-div:not(#correlationPlot) {{
        margin-left: auto;
        margin-right: auto;
        transform: translate(-60px, 5px);
    }}
    .js-plotly-plot:not(#correlationPlot) {{
        margin-left: auto;
        margin-right: auto;
    }}

    #textList {{
        max-height: 600px;
        overflow-y: auto;
    }}
    #textList div {{
        padding: 2px;
        cursor: pointer;
        font-size: 11px;
    }}
    #textList div:hover {{
        background-color: #ddd;
    }}
    #textList div.focusedTextItem {{
        background-color: #ffeb3b;
        font-weight: bold;
    }}
    #textList div.focusedTextItem:hover {{
        background-color: #ffe066;
    }}
    .searchableMetricInput {{
        width: 100%;
        box-sizing: border-box;
        margin: 4px 0;
        font-size: 11px;
    }}
    .searchableMetricSelect {{
        width: 100%;
        font-size: 11px;
    }}

    #correlationContainer {{
        display: none;
    }}
    #correlationPlot {{
    }}
    #correlationPlot .modebar {{
        left: 50%;
        right: auto;
        transform: translateX(-50%);
    }}
    #correlationControls {{
        position: fixed;
        left: 10px;
        top: 45px;
        background: white;
        padding: 5px;
        border: 0.5px solid black;
        z-index: 9999;
        font-size: 12px;
    }}
    #correlationControls button {{
        margin-top: 5px;
    }}
    #correlationControls select {{
        font-size: 11px;
    }}
    #correlationStats {{
        position: fixed;
        right: 10px;
        top: 45px;
        background: white;
        padding: 12px;
        border: 0.5px solid black;
        z-index: 9999;
        min-width: 100px;
        font-size: 10px;
        line-height: 1.4;
    }}
    #correlationStats strong {{
        font-size: 11px;
        display: block;
        margin-bottom: 8px;
        padding-bottom: 4px;
    }}
    </style>

    <script>
        const CONFIG = {{
            highlightSize: 80,
            normalSize: 15,
            focusStarSize: 40,
            dimmedOpacity: 0.2,
            highlightDuration: 2000,
            textOutlineDelay: 100,
            pointFontSize: 7,
            colorScaleZoomRatio: 0.25,
            regressionLinePoints: 100,
            defaultColorScale: 'rdbu_r',
            defaultFocusText: {json.dumps(focus_text)}
        }};

        const metricsData = {metrics_json_str};
        const chartTitles = {titles_json_str};
        const textNames = {texts_json};
        const metricDisplayNames = {json.dumps(metric_names_dict)};
        const metricColorScales = {json.dumps(metric_colorscale_dict)};
        const CLUSTER_TURBO_COLORSCALE = [
            [0.0, '#30123b'],
            [0.08, '#4145ab'],
            [0.16, '#4675ed'],
            [0.24, '#39a2fc'],
            [0.32, '#1bcfd4'],
            [0.40, '#24eca6'],
            [0.48, '#61fc6c'],
            [0.56, '#a4fc3b'],
            [0.64, '#d1e834'],
            [0.72, '#f3c63a'],
            [0.80, '#fe9b2d'],
            [0.88, '#f36315'],
            [0.96, '#d93806'],
            [1.0, '#7a0403']
        ];
        const metricKeys = {json.dumps(metric_keys_list)};
        const textAnnotations = {programmer_notes_json};
        const annotationsEnabled = {annotations_enabled_json};
        const semanticEmbeddings = {semantic_embeddings_json};
        const burrowsSimilarity = {burrows_similarity_json};
        const lowTokenTexts = {json.dumps([tx for tx in texts if tx in low_token_texts])};
        const FOCUS_OVERLAY_META = 'focusOverlay';
        const CORR_FOCUS_OVERLAY_META = 'corrFocusOverlay';

        let currentMetric = {json.dumps(active_metric)};
        let viewMode = 'umap';
        let correlationXMetric = metricKeys[0];
        let correlationYMetric = metricKeys.length > 1 ? metricKeys[1] : metricKeys[0];
        let umapLabelsVisible = true;
        let correlationLabelsVisible = true;
        let hideGrayPoints = false;
        let currentColorDataForPlot = [];
        let currentColorscale = CLUSTER_TURBO_COLORSCALE;
        let currentCmin = 0;
        let currentCmax = 1;
        let focusedTextName = null;
        let focusedTextIndex = -1;
        let semanticLineModeEnabled = false;
        let semanticLineCount = 10;
        let mainSemanticRefreshToken = 0;
        let correlationSemanticRefreshToken = 0;
        const clusterMetricKeys = metricKeys.filter(key => key === 'cluster' || key.startsWith('cluster_'));
        let focusClusterOnly = false;
        let focusClusterMetric = clusterMetricKeys.includes('cluster')
            ? 'cluster'
            : (clusterMetricKeys.length ? clusterMetricKeys[0] : null);
        let focusSimilarityMode = 'cosine';
        const textNameLookup = new Map(
            textNames.map((name, index) => [name.toLowerCase(), {{ name, index }}])
        );

        function addTextOutlines(selector, className) {{
            selector = selector || '.scatterlayer .textpoint text';
            className = className || 'plotly-stroke-text';
            const textElements = document.querySelectorAll(selector);
            textElements.forEach(element => element.classList.add(className));
        }}

        function showAnnotationModal(textName) {{
            if (!annotationsEnabled || !textAnnotations[textName]) {{
                return;
            }}

            const modal = document.getElementById('noteModal');
            const body = document.getElementById('noteModalBody');

            while (body.firstChild) {{
                body.removeChild(body.firstChild);
            }}

            const title = document.createElement('strong');
            title.textContent = `${{textName}} annotation:`;
            body.appendChild(title);
            body.appendChild(document.createElement('hr'));

            const content = document.createElement('div');
            content.style.whiteSpace = 'pre-wrap';
            content.textContent = textAnnotations[textName];
            body.appendChild(content);
            modal.style.display = 'block';
        }}

        function calculateMetricRange(metricArray) {{
            const validValues = metricArray.filter(value => value !== null && !isNaN(value));
            if  validValues.length === 0) {{
                return null;
            }}
            return {{
                min: Math.min(...validValues),
                max: Math.max(...validValues)
            }};
        }}

        function makeDraggable(panel) {{
            const handle = panel.querySelector('.panelDragHandle');
            if (!handle) return;

            let dragging = false;
            let startX = 0, startY = 0, startLeft = 0, startTop = 0;

            function onMove(e) {{
                if (!dragging) return;
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                panel.style.left = (startLeft + dx) + 'px';
                panel.style.top = (startTop + dy) + 'px';
            }}

            function onUp() {{
                if (!dragging) return;
                dragging = false;
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            }}

            handle.addEventListener('mousedown', (e) => {{
                const tag = (e.target.tagName || '').toUpperCase();
                if (tag === 'INPUT' || tag === 'SELECT' || tag === 'BUTTON' ||
                    tag === 'TEXTAREA' || tag === 'A') {{
                    return;
                }}
                e.preventDefault();
                e.stopPropagation();

                const rect = panel.getBoundingClientRect();
                panel.style.left = rect.left + 'px';
                panel.style.top = rect.top + 'px';
                panel.style.right = 'auto';
                panel.style.bottom = 'auto';

                startX = e.clientX;
                startY = e.clientY;
                startLeft = rect.left;
                startTop = rect.top;
                dragging = true;

                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
            }});
        }}

        function createControlPanel(id, position, borderStyle, content, options) {{
            options = options || {{}};
            const panel = document.createElement('div');
            panel.id = id;
            const {{display, ...pos}} = position;

            const draggable = options.draggable !== false;
            const resizable = options.resizable !== false;
            const zIndex = options.zIndex || 9999;

            Object.assign(panel.style, {{
                position: 'fixed',
                background: 'white',
                padding: '5px',
                border: borderStyle || '0.5px solid black',
                zIndex: String(zIndex),
                resize: resizable ? 'both' : 'none',
                overflow: resizable ? 'auto' : 'visible',
                minWidth: options.minWidth || '120px',
                minHeight: options.minHeight || '30px',
                boxSizing: 'border-box',
                maxHeight: '90vh',
                ...pos
            }});

            if (draggable) {{
                const handle = document.createElement('div');
                handle.className = 'panelDragHandle';
                handle.textContent = options.dragLabel || '⋮⋮';
                handle.style.cssText = [
                    'margin: -5px -5px 5px -5px',
                    'padding: 1px 5px',
                    'background: #f0f0f0',
                    'border-bottom: 0.5px solid #ccc',
                    'font-size: 9px',
                    'color: #555',
                    'text-align: center',
                    'letter-spacing: 2px',
                    'line-height: 1'
                ].join(';');
                panel.appendChild(handle);
            }}

            const body = document.createElement('div');
            body.className = 'panelBody';
            if (content) {{
                body.innerHTML = content;
            }}
            panel.appendChild(body);

            if (display) {{
                panel.style.display = display;
            }}
            document.body.appendChild(panel);

            if (draggable) {{
                makeDraggable(panel);
            }}

            return panel;
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            const mainPlot = document.querySelector('.plotly-graph-div');
            if (!mainPlot) {{
                return;
            }}

            setTimeout(() => addTextOutlines(), CONFIG.textOutlineDelay);

            if (annotationsEnabled) {{
                const modal = document.createElement('div');
                modal.id = 'noteModal';
                modal.innerHTML = '<div id="noteModalContent">' +
                    '<span id="noteModalClose">&times;</span>' +
                    '<div id="noteModalBody"></div>' +
                    '</div>';
                document.body.appendChild(modal);

                document.getElementById('noteModalClose').onclick = () => {{
                    modal.style.display = 'none';
                }};

                window.addEventListener('click', (event) => {{
                    if (event.target === modal) {{
                        modal.style.display = 'none';
                    }}
                }});

                mainPlot.on('plotly_click', (data) => {{
                    if (data.points && data.points[0]) {{
                        const textName = textNames[data.points[0].pointIndex];
                        showAnnotationModal(textName);
                    }}
                }});
            }}

            const plotObserver = new MutationObserver(() => {{
                addTextOutlines();
            }});
            plotObserver.observe(mainPlot, {{ childList: true, subtree: true }});

            const toggleViewPanel = createControlPanel('toggleViewPanel',
                {{ left: '10px', top: '5px', zIndex: 10001 }},
                '0.5px solid darkblue',
                '<button id="toggleViewBtn" style="font-weight: bold;">Switch to Correlation View</button>',
                {{ resizable: false, zIndex: 10001, dragLabel: 'view' }}
            );

            createControlPanel('metricPanel',
                {{ left: '10px', top: '45px' }},
                '0.5px solid black',
                '<b>Color Metric</b><br/>' +
                '<input id="metricSearchInput" class="searchableMetricInput" type="text" placeholder="Search metrics..."><br/>' +
                '<select id="metricSelect" class="searchableMetricSelect"></select><br/>' +
                '<button id="toggleLabelsBtn" style="margin-top: 5px; font-size: 10px; padding: 2px 6px;">Hide Labels</button><br/>' +
                '<button id="toggleGrayPointsBtn" style="margin-top: 5px; font-size: 10px; padding: 2px 6px;">Hide Gray Points</button>',
                {{ zIndex: 9999, minWidth: '220px', dragLabel: 'color metric' }}
            );

            const focusButtonPanel = createControlPanel('focusButtonPanel',
                {{ left: '260px', top: '5px', zIndex: 10050 }},
                '0.5px solid black',
                '<button id="openFocusPanelBtn">Focus Text</button>' +
                '<span id="focusMetricPercentile" style="margin-left: 8px; font-size: 10px; font-weight: 500;"></span>',
                {{ resizable: false, zIndex: 10050, dragLabel: 'focus' }}
            );
            const focusPanel = createControlPanel('focusPanel',
                {{ left: '260px', top: '39px', display: 'none', zIndex: 10050 }},
                '0.5px solid black',
                '<b>Focus Text</b><br/>' +
                '<input id="focusTextInput" type="text" size="20" placeholder="Enter text name"><br/>' +
                '<label style="font-size: 10px;">Similarity: <select id="focusSimilarityModeSelect" style="font-size: 10px;">' +
                    '<option value="cosine">Cosine</option>' +
                    '<option value="burrows">Burrows</option>' +
                '</select></label><br/>' +
                '<label style="font-size: 10px;"><input id="focusClusterOnlyCheckbox" type="checkbox"> Hide texts outside focused text cluster</label><br/>' +
                '<label style="font-size: 10px;">Clustering: <select id="focusClusterMetricSelect" style="font-size: 10px;"></select></label><br/>' +
                '<label style="font-size: 10px;"><input id="semanticLinesCheckbox" type="checkbox"> Show semantic connections</label><br/>' +
                '<label style="font-size: 10px;">Connections: <input id="semanticLinesSlider" type="range" min="1" max="50" value="10"> <span id="semanticLinesValue">10</span></label><br/>' +
                '<button id="applyFocusBtn" style="margin-top: 5px;">Apply</button> ' +
                '<button id="clearFocusBtn" style="margin-top: 5px;">Clear</button>',
                {{ zIndex: 10050, minWidth: '240px', dragLabel: 'focus text' }}
            );

            const toggleRect = toggleViewPanel.getBoundingClientRect();
            const horizontalGap = 8;
            const focusButtonLeft = toggleRect.left + toggleRect.width + horizontalGap;
            focusButtonPanel.style.left = `${{focusButtonLeft}}px`;
            focusButtonPanel.style.top = `${{toggleRect.top}}px`;

            focusPanel.style.left = `${{focusButtonLeft}}px`;
            focusPanel.style.top = `${{toggleRect.top + toggleRect.height + 6}}px`;

            createControlPanel('colorPanel',
                {{ right: '10px', top: '5px' }},
                '0.5px solid black',
                '<b>Color Scale</b><br/>' +
                'Min: <input id="colorMin" type="number" step="any" size="8"><br/>' +
                'Max: <input id="colorMax" type="number" step="any" size="8"><br/>' +
                '<button id="applyColorBtn">Apply</button> ' +
                '<button id="resetColorBtn">Reset</button><br/>' +
                '<button id="zoomInColorBtn">Zoom In</button> ' +
                '<button id="zoomOutColorBtn">Zoom Out</button>',
                {{ zIndex: 9999, minWidth: '160px', dragLabel: 'color scale' }}
            );

            createControlPanel('textPanel',
                {{ right: '10px', top: '135px', width: '180px' }},
                '0.5px solid black',
                '<b id="textListHeading">Texts</b><br/>' +
                '<select id="textSortOrder">' +
                '<option value="alpha-asc">A-Z</option>' +
                '<option value="alpha-desc">Z-A</option>' +
                '<option value="metric-asc">Low to High</option>' +
                '<option value="metric-desc" selected>High to Low</option>' +
                '</select>' +
                '<div id="textList"></div>',
                {{ zIndex: 9999, minWidth: '180px', dragLabel: 'texts' }}
            );

            const metricSelect = document.getElementById('metricSelect');
            const metricSearchInput = document.getElementById('metricSearchInput');
            let xMetricSearchInput = null;
            let yMetricSearchInput = null;

            const metricOptionCatalog = metricKeys
                .filter(key => key in metricsData && metricsData[key] !== null)
                .map(key => {{
                    const displayLabel = metricDisplayNames[key] || key;
                    return {{
                        key,
                        hasData: true,
                        displayLabel,
                        searchBlob: `${{key}} ${{displayLabel}}`.toLowerCase()
                    }};
                }});

            function repopulateMetricSelect(selectEl, selectedValue, filterText) {{
                if (!selectEl) return;
                const normalizedFilter = (filterText || '').trim().toLowerCase();
                const previousValue = selectedValue || selectEl.value;
                const filtered = metricOptionCatalog.filter(item =>
                    !normalizedFilter || item.searchBlob.includes(normalizedFilter)
                );

                selectEl.innerHTML = '';
                filtered.forEach(item => {{
                    const option = document.createElement('option');
                    option.value = item.key;
                    option.text = item.displayLabel;
                    option.disabled = !item.hasData;
                    selectEl.appendChild(option);
                }});

                if (filtered.length === 0) {{
                    const noResults = document.createElement('option');
                    noResults.value = '';
                    noResults.text = 'No matching metrics';
                    noResults.disabled = true;
                    noResults.selected = true;
                    selectEl.appendChild(noResults);
                    return;
                }}

                const hasPrevious = filtered.some(item => item.key === previousValue && item.hasData);
                if (hasPrevious) {{
                    selectEl.value = previousValue;
                }} else {{
                    const firstEnabled = filtered.find(item => item.hasData);
                    if (firstEnabled) {{
                        selectEl.value = firstEnabled.key;
                    }}
                }}
            }}

            repopulateMetricSelect(metricSelect, currentMetric, '');
            if (metricSelect.value) {{
                currentMetric = metricSelect.value;
            }}

            const metricRangeCache = {{}};

            function buildFocusedMarkerSymbols(pointCount, focusedIndex) {{
                return new Array(pointCount).fill('circle');
            }}

            function buildFocusedMarkerSizes(pointCount, focusedIndex) {{
                return new Array(pointCount).fill(CONFIG.normalSize);
            }}

            function buildMarkerSymbols() {{
                return buildFocusedMarkerSymbols(textNames.length, focusedTextIndex);
            }}

            function buildMarkerSizes() {{
                return buildFocusedMarkerSizes(textNames.length, focusedTextIndex);
            }}

            function buildMarkerOpacities() {{
                let focusedClusterValue = null;
                let selectedClusterData = null;

                if (focusClusterOnly && focusClusterMetric && focusedTextIndex >= 0) {{
                    selectedClusterData = metricsData[focusClusterMetric] || null;
                    if (selectedClusterData && focusedTextIndex < selectedClusterData.length) {{
                        const value = selectedClusterData[focusedTextIndex];
                        if (value !== null && value !== undefined && !Number.isNaN(value)) {{
                            focusedClusterValue = value;
                        }}
                    }}
                }}

                return textNames.map((_, index) => {{
                    if (index === focusedTextIndex) {{
                        return 1.0;
                    }}

                    const value = currentColorDataForPlot[index];
                    const isGrayPoint = value === null || value === undefined || Number.isNaN(value);
                    if (hideGrayPoints && isGrayPoint) {{
                        return 0.0;
                    }}

                    if (focusedClusterValue !== null && selectedClusterData) {{
                        const clusterValue = selectedClusterData[index];
                        const isInFocusedCluster = clusterValue !== null && clusterValue !== undefined && !Number.isNaN(clusterValue) && clusterValue === focusedClusterValue;
                        if (!isInFocusedCluster) {{
                            return 0.0;
                        }}
                    }}

                    return 1.0;
                }});
            }}

            function buildMainFocusOverlayTrace() {{
                return {{
                    x: [], y: [],
                    mode: 'markers+text',
                    type: 'scatter',
                    meta: FOCUS_OVERLAY_META,
                    hoverinfo: 'skip',
                    showlegend: false,
                    text: [],
                    textposition: 'middle center',
                    textfont: {{ size: 9, color: '#000' }},
                    marker: {{
                        symbol: 'star',
                        size: CONFIG.focusStarSize,
                        color: '#ffd400',
                        line: {{ color: '#111', width: 2 }}
                    }}
                }};
            }}

            function ensureMainFocusOverlay() {{
                if (!mainPlot.data.some(t => t.meta === FOCUS_OVERLAY_META)) {{
                    return Plotly.addTraces(mainPlot, buildMainFocusOverlayTrace());
                }}
                return Promise.resolve();
            }}

            function bringMainFocusOverlayToFront() {{
                const idx = mainPlot.data.findIndex(t => t.meta === FOCUS_OVERLAY_META);
                if (idx >= 0 && idx !== mainPlot.data.length - 1) {{
                    return Plotly.moveTraces(mainPlot, [idx], [mainPlot.data.length - 1]);
                }}
                return Promise.resolve();
            }}

            function readBaseColor(plotObj, index) {{
                const base = plotObj._fullData && plotObj._fullData[0];
                if (!base) return null;
                const colors = base.marker && base.marker.color;
                if (Array.isArray(colors) || (colors && ArrayBuffer.isView(colors))) {{
                    if (index >= 0 && index < colors.length) return colors[index];
                }} else if (typeof colors === 'string') {{
                    return colors;
                }}
                return null;
            }}

            function updateMainFocusOverlay() {{
                const idx = mainPlot.data.findIndex(t => t.meta === FOCUS_OVERLAY_META);
                if (idx < 0) return Promise.resolve();

                if (focusedTextIndex < 0 || !focusedTextName) {{
                    return Plotly.restyle(mainPlot, {{
                        x: [[]], y: [[]], text: [[]]
                    }}, [idx]);
                }}

                const xv = getPlotCoordinate(mainPlot, 0, 'x', focusedTextIndex);
                const yv = getPlotCoordinate(mainPlot, 0, 'y', focusedTextIndex);

                const resolved = readBaseColor(mainPlot, focusedTextIndex);

                if (resolved && typeof resolved === 'string') {{
                    return Plotly.restyle(mainPlot, {{
                        x: [[xv]],
                        y: [[yv]],
                        text: [[focusedTextName]],
                        'marker.color': [[resolved]]
                    }}, [idx]);
                }}

                const focusValue = currentColorDataForPlot[focusedTextIndex];
                const isEmpty = focusValue === null || focusValue === undefined || Number.isNaN(focusValue);
                if (isEmpty) {{
                    return Plotly.restyle(mainPlot, {{
                        x: [[xv]], y: [[yv]], text: [[focusedTextName]],
                        'marker.color': [['#808080']]
                    }}, [idx]);
                }}
                return Plotly.restyle(mainPlot, {{
                    x: [[xv]], y: [[yv]], text: [[focusedTextName]],
                    'marker.color': [[focusValue]],
                    'marker.colorscale': [currentColorscale],
                    'marker.cmin': [currentCmin],
                    'marker.cmax': [currentCmax]
                }}, [idx]);
            }}

            function refreshFocusedTextMarker() {{
                Plotly.restyle(mainPlot, {{
                    'marker.opacity': [buildMarkerOpacities()]
                }});

                ensureMainFocusOverlay()
                    .then(() => updateMainFocusOverlay())
                    .then(() => bringMainFocusOverlayToFront())
                    .then(() => refreshSemanticConnections());
            }}

            function populateFocusClusterMetricSelect() {{
                const select = document.getElementById('focusClusterMetricSelect');
                if (!select) {{
                    return;
                }}

                select.innerHTML = '';
                clusterMetricKeys.forEach(metricKey => {{
                    const option = document.createElement('option');
                    option.value = metricKey;
                    option.text = metricDisplayNames[metricKey] || metricKey;
                    select.appendChild(option);
                }});

                if (focusClusterMetric && clusterMetricKeys.includes(focusClusterMetric)) {{
                    select.value = focusClusterMetric;
                }} else if (clusterMetricKeys.length > 0) {{
                    focusClusterMetric = clusterMetricKeys[0];
                    select.value = focusClusterMetric;
                }}
            }}

            function getPlotCoordinate(plot, traceIndex, axis, pointIndex) {{
                const fullTrace = plot._fullData && plot._fullData[traceIndex];
                const dataTrace = plot.data && plot.data[traceIndex];
                const values = (fullTrace && fullTrace[axis]) || (dataTrace && dataTrace[axis]);
                return values ? values[pointIndex] : undefined;
            }}

            function semanticConnectionColor(rank, total) {{
                const ratio = total <= 1 ? 0 : rank / (total - 1);
                const red = Math.round(235 * (1 - ratio) + 70 * ratio);
                const green = Math.round(35 * (1 - ratio) + 55 * ratio);
                const blue = Math.round(45 * (1 - ratio) + 150 * ratio);
                return `rgb(${{red}}, ${{green}}, ${{blue}})`;
            }}

            function getCosineNeighbors(focusIndex) {{
                if (focusIndex < 0 || focusIndex >= semanticEmbeddings.length) return [];
                const focusEmbedding = semanticEmbeddings[focusIndex];
                return textNames
                    .map((text, index) => {{
                        if (index === focusIndex || !semanticEmbeddings[index]) return null;
                        const embedding = semanticEmbeddings[index];
                        let similarity = 0;
                        const length = Math.min(focusEmbedding.length, embedding.length);
                        for (let dimension = 0; dimension < length; dimension++) {{
                            similarity += focusEmbedding[dimension] * embedding[dimension];
                        }}
                        return {{ text, index, similarity }};
                    }})
                    .filter(Boolean)
                    .sort((a, b) => b.similarity - a.similarity);
            }}

            function getBurrowsNeighbors(focusIndex) {{
                if (focusIndex < 0 || focusIndex >= burrowsSimilarity.length) return [];
                const row = burrowsSimilarity[focusIndex];
                if (!row) return [];
                return textNames
                    .map((text, index) => {{
                        if (index === focusIndex) return null;
                        const similarity = row[index];
                        if (similarity === null || similarity === undefined || Number.isNaN(similarity)) return null;
                        return {{ text, index, similarity }};
                    }})
                    .filter(Boolean)
                    .sort((a, b) => b.similarity - a.similarity);
            }}

            function getFocusNeighbors(focusIndex) {{
                if (focusSimilarityMode === 'burrows') {{
                    return getBurrowsNeighbors(focusIndex);
                }}
                return getCosineNeighbors(focusIndex);
            }}

            function removeSemanticConnectionTraces(plot) {{
                const indices = [];
                plot.data.forEach((trace, index) => {{
                    if (trace.meta === 'semanticConnection') indices.push(index);
                }});
                if (indices.length) return Plotly.deleteTraces(plot, indices.reverse());
                return Promise.resolve();
            }}

            async function refreshSemanticConnections() {{
                const refreshToken = ++mainSemanticRefreshToken;
                await removeSemanticConnectionTraces(mainPlot);
                if (refreshToken !== mainSemanticRefreshToken) return;
                if (!semanticLineModeEnabled || focusedTextIndex < 0 || !focusedTextName) {{
                    await bringMainFocusOverlayToFront();
                    return;
                }}
                const neighbors = getFocusNeighbors(focusedTextIndex).slice(0, semanticLineCount);
                const traces = neighbors.map((item, rank) => {{
                    return {{ x: [getPlotCoordinate(mainPlot, 0, 'x', focusedTextIndex), getPlotCoordinate(mainPlot, 0, 'x', item.index)],
                        y: [getPlotCoordinate(mainPlot, 0, 'y', focusedTextIndex), getPlotCoordinate(mainPlot, 0, 'y', item.index)],
                        mode: 'lines', type: 'scatter', meta: 'semanticConnection', hoverinfo: 'skip',
                        showlegend: false,
                        line: {{ color: semanticConnectionColor(rank, neighbors.length), width: 2 }} }};
                }}).filter(Boolean);
                if (traces.length && refreshToken === mainSemanticRefreshToken) {{
                    await Plotly.addTraces(mainPlot, traces);
                }}
                await bringMainFocusOverlayToFront();
            }}

            function syncCurrentColorDataFromPlot() {{
                const plotData = mainPlot && mainPlot.data && mainPlot.data[0];
                if (!plotData || !plotData.marker) {{
                    currentColorDataForPlot = [];
                    return;
                }}
                const markerColors = plotData.marker.color;
                if (Array.isArray(markerColors)) {{
                    currentColorDataForPlot = markerColors.slice();
                }} else if (ArrayBuffer.isView(markerColors)) {{
                    currentColorDataForPlot = Array.from(markerColors);
                }} else {{
                    currentColorDataForPlot = new Array(textNames.length).fill(markerColors);
                }}
            }}

            function ordinalSuffix(n) {{
                const value = Math.round(n);
                const tens = value % 100;
                if (tens >= 11 && tens <= 13) {{
                    return `${{value}}th`;
                }}
                switch (value % 10) {{
                    case 1: return `${{value}}st`;
                    case 2: return `${{value}}nd`;
                    case 3: return `${{value}}rd`;
                    default: return `${{value}}th`;
                }}
            }}

            function formatTrimmedNumber(value, digits = 4) {{
                if (value === null || value === undefined || Number.isNaN(value)) {{
                    return 'N/A';
                }}
                const fixed = Number(value).toFixed(digits);
                const trimmed = fixed
                    .replace(/\\.?0+$/, (match) => match.startsWith('.') ? '' : '')
                    .replace(/(\\.\\d*?[1-9])0+/, '$1');
                return trimmed === '-0' ? '0' : trimmed;
            }}

            function calculateFocusedMetricPercentile(metricKey, textIndex) {{
                const metricValues = metricsData[metricKey];
                if (!metricValues || textIndex < 0 || textIndex >= metricValues.length) {{
                    return null;
                }}

                const focusValue = metricValues[textIndex];
                if (focusValue === null || isNaN(focusValue) || focusValue <= 0) {{
                    return {{ percentile: null, validCount: 0, focusValue }};
                }}

                const validValues = metricValues.filter(v => v !== null && !isNaN(v));
                if (!validValues.length) {{
                    return {{ percentile: null, validCount: 0, focusValue }};
                }}

                let lessThan = 0;
                let equalTo = 0;
                const EPSILON = 1e-9;
                for (const value of validValues) {{
                    if (value < focusValue - EPSILON) lessThan += 1;
                    else if (Math.abs(value - focusValue) <= EPSILON) equalTo += 1;
                }}

                const percentile = ((lessThan + (equalTo * 0.5)) / validValues.length) * 100;
                return {{ percentile, validCount: validValues.length, focusValue }};
            }}

            function updateFocusMetricPercentileLabel() {{
                const percentileEl = document.getElementById('focusMetricPercentile');
                if (!percentileEl) {{
                    return;
                }}

                if (!focusedTextName || focusedTextIndex < 0) {{
                    percentileEl.textContent = '';
                    return;
                }}

                const metricLabel = metricDisplayNames[currentMetric] || currentMetric;
                const percentileInfo = calculateFocusedMetricPercentile(currentMetric, focusedTextIndex);

                if (!percentileInfo || percentileInfo.percentile === null) {{
                    percentileEl.textContent = `${{metricLabel}}: no non-zero score`;
                    return;
                }}

                const pctLabel = ordinalSuffix(percentileInfo.percentile);
                const focusValue = percentileInfo.focusValue;
                const displayCount = Number.isInteger(focusValue) || Math.abs(focusValue - Math.round(focusValue)) < 1e-9
                    ? `${{Math.round(focusValue)}}`
                    : formatTrimmedNumber(focusValue, 4);
                percentileEl.textContent = `${{metricLabel}}: ${{pctLabel}} percentile (${{displayCount}})`;
            }}

            function setFocusedText(textInput) {{
                const normalizedInput = (textInput || '').trim();
                if (!normalizedInput) {{
                    focusedTextName = null;
                    focusedTextIndex = -1;
                    refreshFocusedTextMarker();
                    refreshFocusedTextInCorrelationPlot();
                    updateTextList();
                    updateFocusMetricPercentileLabel();
                    return;
                }}

                const match = textNameLookup.get(normalizedInput.toLowerCase());
                if (!match) {{
                    alert(`Who even is "${{normalizedInput}}"??? I dunno that text. Who they be???`);
                    return;
                }}

                focusedTextName = match.name;
                focusedTextIndex = match.index;
                refreshFocusedTextMarker();
                refreshFocusedTextInCorrelationPlot();
                updateTextList();
                updateFocusMetricPercentileLabel();
            }}

            function updateTextList() {{
                const sortOrder = document.getElementById('textSortOrder').value;
                const textListContainer = document.getElementById('textList');
                const currentMetricData = metricsData[currentMetric];

                const heading = document.getElementById('textListHeading');
                if (semanticLineModeEnabled && focusedTextName) {{
                    const modeLabel = focusSimilarityMode === 'burrows' ? 'Stylistically (Burrows)' : 'Semantically';
                    heading.textContent = `Most ${{modeLabel}} Similar to ${{focusedTextName}}:`;
                    textListContainer.innerHTML = '';
                    getFocusNeighbors(focusedTextIndex).forEach((item, position) => {{
                        const listItem = document.createElement('div');
                        const scoreLabel = focusSimilarityMode === 'burrows' ? 'burrows' : 'cosine';
                        listItem.textContent = `${{position + 1}}. ${{item.text}} (${{scoreLabel}}: ${{formatTrimmedNumber(item.similarity, 4)}})`;
                        textListContainer.appendChild(listItem);
                    }});
                    return;
                }}
                heading.textContent = 'Texts';

                const textsWithValues = textNames
                    .map((name, index) => ({{
                        name: name,
                        index: index,
                        value: currentMetricData ? currentMetricData[index] : null
                    }}))
                    .filter(text =>
                        text.value !== null &&
                        text.value !== undefined &&
                        !Number.isNaN(text.value)
                    );

                const sortFunctions = {{
                    'alpha-asc': (a, b) => a.name.localeCompare(b.name),
                    'alpha-desc': (a, b) => b.name.localeCompare(a.name),
                    'metric-asc': (a, b) => (a.value || 0) - (b.value || 0),
                    'metric-desc': (a, b) => (b.value || 0) - (a.value || 0)
                }};

                if (sortFunctions[sortOrder]) {{
                    textsWithValues.sort(sortFunctions[sortOrder]);
                }}

                textListContainer.innerHTML = '';

                textsWithValues.forEach((text, position) => {{
                    const listItem = document.createElement('div');
                    const valueDisplay = formatTrimmedNumber(text.value, 4);
                    listItem.textContent = `${{position + 1}}. ${{text.name}} (${{valueDisplay}})`;
                    listItem.title = 'Click to highlight on graph';
                    if (text.index === focusedTextIndex) {{
                        listItem.classList.add('focusedTextItem');
                    }}

                    listItem.addEventListener('click', () => {{
                        const sizes = buildMarkerSizes();
                        const baseOpacities = buildMarkerOpacities();
                        const opacities = baseOpacities.map(opacity => opacity === 0 ? 0 : CONFIG.dimmedOpacity);
                        sizes[text.index] = CONFIG.highlightSize;
                        opacities[text.index] = 1.0;
                        Plotly.restyle(mainPlot, {{
                            'marker.size': [sizes],
                            'marker.opacity': [opacities]
                        }});

                        setTimeout(() => {{
                            Plotly.restyle(mainPlot, {{
                                'marker.size': [buildMarkerSizes()],
                                'marker.opacity': [buildMarkerOpacities()]
                            }});
                        }}, CONFIG.highlightDuration);
                    }});

                    textListContainer.appendChild(listItem);
                }});
            }}

            function updateColorScaleInputs() {{
                const plotData = mainPlot.data[0];
                if (!plotData) {{
                    return;
                }}
                document.getElementById('colorMin').value = plotData.marker.cmin;
                document.getElementById('colorMax').value = plotData.marker.cmax;
            }}

            function applyColorScale(minValue, maxValue) {{
                Plotly.restyle(mainPlot, {{
                    'marker.cmin': minValue,
                    'marker.cmax': maxValue
                }}).then(() => {{
                    currentCmin = minValue;
                    currentCmax = maxValue;
                    refreshFocusedTextMarker();
                }});
            }}

            document.getElementById('focusTextInput').value = CONFIG.defaultFocusText;
            populateFocusClusterMetricSelect();
            syncCurrentColorDataFromPlot();

            (function initColorStateFromTrace() {{
                const base = mainPlot.data[0];
                if (base && base.marker) {{
                    if (base.marker.colorscale !== undefined) {{
                        currentColorscale = base.marker.colorscale;
                    }}
                    if (base.marker.cmin !== undefined && base.marker.cmin !== null) {{
                        currentCmin = base.marker.cmin;
                    }}
                    if (base.marker.cmax !== undefined && base.marker.cmax !== null) {{
                        currentCmax = base.marker.cmax;
                    }}
                }}
            }})();

            if (CONFIG.defaultFocusText) {{
                setFocusedText(CONFIG.defaultFocusText);
            }} else {{
                refreshFocusedTextMarker();
            }}
            updateColorScaleInputs();
            updateTextList();
            updateFocusMetricPercentileLabel();

            const initialRange = calculateMetricRange(metricsData[currentMetric]);
            if (initialRange) {{
                metricRangeCache[currentMetric] = initialRange;
            }}

            metricSearchInput.addEventListener('input', function() {{
                const previousMetric = metricSelect.value;
                repopulateMetricSelect(metricSelect, previousMetric, this.value);
                if (metricSelect.value && metricSelect.value !== previousMetric) {{
                    metricSelect.dispatchEvent(new Event('change'));
                }}
            }});

            metricSelect.addEventListener('change', function() {{
                const selectedMetric = this.value;
                const metricColorData = metricsData[selectedMetric];
                const chartTitle = chartTitles[selectedMetric] || 'Text Semantic Clusters';

                if (!metricColorData) {{
                    return;
                }}

                const colorDataForPlot = metricColorData.map(value => value === null ? NaN : value);

                const valueRange = calculateMetricRange(metricColorData);
                if (!valueRange) {{
                    return;
                }}

                const selectedColorScale = (selectedMetric === 'cluster' || selectedMetric.startsWith('cluster_'))
                    ? CLUSTER_TURBO_COLORSCALE
                    : (metricColorScales[selectedMetric] || CONFIG.defaultColorScale);

                currentMetric = selectedMetric;
                currentColorDataForPlot = colorDataForPlot;
                currentColorscale = selectedColorScale;
                currentCmin = valueRange.min;
                currentCmax = valueRange.max;
                metricRangeCache[selectedMetric] = valueRange;

                Promise.all([
                    Plotly.restyle(mainPlot, {{
                        'marker.color': [colorDataForPlot],
                        'marker.colorscale': [selectedColorScale],
                        'marker.autocolorscale': [false],
                        'marker.cmin': valueRange.min,
                        'marker.cmax': valueRange.max,
                        'marker.colorbar.title': metricDisplayNames[selectedMetric] || selectedMetric
                    }}),
                    Plotly.relayout(mainPlot, {{'title.text': chartTitle}})
                ]).then(() => {{
                    updateColorScaleInputs();
                    updateTextList();
                    updateFocusMetricPercentileLabel();
                    refreshFocusedTextMarker();
                }});
            }});

            document.getElementById('applyColorBtn').addEventListener('click', () => {{
                const minValue = parseFloat(document.getElementById('colorMin').value);
                const maxValue = parseFloat(document.getElementById('colorMax').value);

                if (isNaN(minValue) || isNaN(maxValue) || minValue >= maxValue) {{
                    alert('Dang it! You just did something impossible! Min can never be greater than Max!');
                    return;
                }}

                applyColorScale(minValue, maxValue);
            }});

            document.getElementById('resetColorBtn').addEventListener('click', () => {{
                const cachedRange = metricRangeCache[currentMetric];
                if (!cachedRange) {{
                    return;
                }}

                document.getElementById('colorMin').value = cachedRange.min;
                document.getElementById('colorMax').value = cachedRange.max;
                applyColorScale(cachedRange.min, cachedRange.max);
            }});

            document.getElementById('zoomInColorBtn').addEventListener('click', () => {{
                const currentMin = parseFloat(document.getElementById('colorMin').value);
                const currentMax = parseFloat(document.getElementById('colorMax').value);
                const range = currentMax - currentMin;
                const center = (currentMin + currentMax) / 2;

                const newMin = center - range * CONFIG.colorScaleZoomRatio;
                const newMax = center + range * CONFIG.colorScaleZoomRatio;

                document.getElementById('colorMin').value = newMin;
                document.getElementById('colorMax').value = newMax;
                applyColorScale(newMin, newMax);
            }});

            document.getElementById('zoomOutColorBtn').addEventListener('click', () => {{
                const currentMin = parseFloat(document.getElementById('colorMin').value);
                const currentMax = parseFloat(document.getElementById('colorMax').value);
                const range = currentMax - currentMin;
                const center = (currentMin + currentMax) / 2;

                const newMin = center - range;
                const newMax = center + range;

                document.getElementById('colorMin').value = newMin;
                document.getElementById('colorMax').value = newMax;
                applyColorScale(newMin, newMax);
            }});

            document.getElementById('textSortOrder').addEventListener('change', updateTextList);

            document.getElementById('openFocusPanelBtn').addEventListener('click', () => {{
                const panel = document.getElementById('focusPanel');
                panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            }});

            document.getElementById('applyFocusBtn').addEventListener('click', () => {{
                const userInput = document.getElementById('focusTextInput').value;
                setFocusedText(userInput);
            }});

            document.getElementById('clearFocusBtn').addEventListener('click', () => {{
                document.getElementById('focusTextInput').value = '';
                setFocusedText('');
            }});

            document.getElementById('focusTextInput').addEventListener('keydown', (event) => {{
                if (event.key === 'Enter') {{
                    setFocusedText(event.target.value);
                }}
            }});

            document.getElementById('focusSimilarityModeSelect').addEventListener('change', (event) => {{
                focusSimilarityMode = event.target.value;
                refreshFocusedTextMarker();
                refreshFocusedTextInCorrelationPlot();
                updateTextList();
            }});

            document.getElementById('focusClusterOnlyCheckbox').addEventListener('change', (event) => {{
                focusClusterOnly = !!event.target.checked;
                refreshFocusedTextMarker();
            }});

            document.getElementById('focusClusterMetricSelect').addEventListener('change', (event) => {{
                focusClusterMetric = event.target.value;
                refreshFocusedTextMarker();
            }});

            document.getElementById('semanticLinesCheckbox').addEventListener('change', (event) => {{
                semanticLineModeEnabled = !!event.target.checked;
                refreshFocusedTextMarker();
                refreshFocusedTextInCorrelationPlot();
                updateTextList();
            }});

            document.getElementById('semanticLinesSlider').addEventListener('input', (event) => {{
                semanticLineCount = Math.max(1, Math.min(50, Number(event.target.value) || 1));
                document.getElementById('semanticLinesValue').textContent = semanticLineCount;
                if (semanticLineModeEnabled) {{
                    refreshSemanticConnections();
                    refreshCorrelationSemanticConnections();
                    updateTextList();
                }}
            }});

            document.getElementById('toggleLabelsBtn').addEventListener('click', function() {{
                if (umapLabelsVisible) {{
                    Plotly.restyle(mainPlot, {{'mode': 'markers'}});
                    this.textContent = 'Show Labels';
                }} else {{
                    Plotly.restyle(mainPlot, {{'mode': 'markers+text'}});
                    this.textContent = 'Hide Labels';
                }}
                umapLabelsVisible = !umapLabelsVisible;
            }});

            document.getElementById('toggleGrayPointsBtn').addEventListener('click', function() {{
                hideGrayPoints = !hideGrayPoints;
                this.textContent = hideGrayPoints ? 'Show Gray Points' : 'Hide Gray Points';
                refreshFocusedTextMarker();
            }});

            const correlationContainer = document.createElement('div');
            correlationContainer.id = 'correlationContainer';
            Object.assign(correlationContainer.style, {{
                display: 'none',
                position: 'fixed',
                top: '0',
                left: '0',
                width: '100%',
                height: '100%',
                background: '#ffffff',
                zIndex: '9990',
                justifyContent: 'center',
                alignItems: 'center',
                overflow: 'auto'
            }});
            correlationContainer.innerHTML = '<div id="correlationPlot" style="width:1200px; height:800px;"></div>';
            document.body.appendChild(correlationContainer);

            const correlationControls = createControlPanel('correlationControls',
                {{ left: '10px', top: '45px', display: 'none' }},
                '0.5px solid black',
                '<b>Correlation Analysis</b><br/>' +
                '<b>X-Axis</b><br/>' +
                '<input id="xMetricSearchInput" class="searchableMetricInput" type="text" placeholder="Search X metric..."><br/>' +
                '<select id="xMetricSelect" class="searchableMetricSelect"></select><br/>' +
                '<b>Y-Axis</b><br/>' +
                '<input id="yMetricSearchInput" class="searchableMetricInput" type="text" placeholder="Search Y metric..."><br/>' +
                '<select id="yMetricSelect" class="searchableMetricSelect"></select><br/>' +
                '<b>Regression</b><br/>' +
                '<select id="regressionType">' +
                '<option value="linear">Linear</option>' +
                '<option value="polynomial">Polynomial</option>' +
                '<option value="logistic">Logistic</option>' +
                '</select><br/>' +
                '<button id="updateCorrelationBtn" style="margin-top: 5px;">Update Plot</button><br/>' +
                '<button id="toggleCorrelationLabelsBtn" style="margin-top: 5px;">Toggle Labels</button>',
                {{ zIndex: 9999, minWidth: '220px', dragLabel: 'correlation analysis' }}
            );

            xMetricSearchInput = document.getElementById('xMetricSearchInput');
            yMetricSearchInput = document.getElementById('yMetricSearchInput');

            const correlationStats = createControlPanel('correlationStats',
                {{ right: '10px', top: '45px', display: 'none' }},
                '0.5px solid black',
                '',
                {{ zIndex: 9999, minWidth: '220px', dragLabel: 'correlation stats' }}
            );

            function initializeCorrelationDropdowns() {{
                const xSelect = document.getElementById('xMetricSelect');
                const ySelect = document.getElementById('yMetricSelect');
                if (!xSelect || !ySelect) return;
                repopulateMetricSelect(xSelect, correlationXMetric, (xMetricSearchInput && xMetricSearchInput.value) || '');
                repopulateMetricSelect(ySelect, correlationYMetric, (yMetricSearchInput && yMetricSearchInput.value) || '');

                if (xSelect.value) {{
                    correlationXMetric = xSelect.value;
                }}
                if (ySelect.value) {{
                    correlationYMetric = ySelect.value;
                }}
            }}

            function calculatePearsonCorrelation(xValues, yValues) {{
                const validPairs = xValues
                    .map((x, i) => [x, yValues[i]])
                    .filter(pair =>
                        pair[0] !== null && pair[1] !== null &&
                        !isNaN(pair[0]) && !isNaN(pair[1])
                    );

                if (validPairs.length < 2) {{
                    return null;
                }}

                const meanX = validPairs.reduce((sum, pair) => sum + pair[0], 0) / validPairs.length;
                const meanY = validPairs.reduce((sum, pair) => sum + pair[1], 0) / validPairs.length;

                let numerator = 0;
                let denominatorX = 0;
                let denominatorY = 0;

                validPairs.forEach(pair => {{
                    const deviationX = pair[0] - meanX;
                    const deviationY = pair[1] - meanY;
                    numerator += deviationX * deviationY;
                    denominatorX += deviationX * deviationX;
                    denominatorY += deviationY * deviationY;
                }});

                if (denominatorX === 0 || denominatorY === 0) {{
                    return null;
                }}

                return numerator / Math.sqrt(denominatorX * denominatorY);
            }}

            function calculateMean(values) {{
                if (!values || values.length === 0) return null;
                return values.reduce((sum, value) => sum + value, 0) / values.length;
            }}

            function calculateVariance(values) {{
                if (!values || values.length === 0) return null;
                const mean = calculateMean(values);
                const sq = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0);
                return sq / values.length;
            }}

            function calculateStdDev(values) {{
                const variance = calculateVariance(values);
                return variance === null ? null : Math.sqrt(variance);
            }}

            function calculateLinearRegression(xValues, yValues) {{
                const n = xValues.length;
                let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

                for (let i = 0; i < n; i++) {{
                    sumX += xValues[i];
                    sumY += yValues[i];
                    sumXY += xValues[i] * yValues[i];
                    sumX2 += xValues[i] * xValues[i];
                }}

                const denom = (n * sumX2 - sumX * sumX);
                const slope = denom === 0 ? 0 : (n * sumXY - sumX * sumY) / denom;
                const intercept = n === 0 ? 0 : (sumY - slope * sumX) / n;

                return {{ slope, intercept }};
            }}

            function calculatePolynomialRegression(xValues, yValues) {{
                const n = xValues.length;
                let sumX = 0, sumY = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0, sumXY = 0, sumX2Y = 0;

                for (let i = 0; i < n; i++) {{
                    const x = xValues[i];
                    const y = yValues[i];
                    sumX += x;
                    sumY += y;
                    sumX2 += x * x;
                    sumX3 += x * x * x;
                    sumX4 += x * x * x * x;
                    sumXY += x * y;
                    sumX2Y += x * x * y;
                }}

                let matrix = [
                    [n, sumX, sumX2],
                    [sumX, sumX2, sumX3],
                    [sumX2, sumX3, sumX4]
                ];
                let vector = [sumY, sumXY, sumX2Y];

                for (let i = 0; i < 3; i++) {{
                    let maxRow = i;
                    for (let j = i + 1; j < 3; j++) {{
                        if (Math.abs(matrix[j][i]) > Math.abs(matrix[maxRow][i])) {{
                            maxRow = j;
                        }}
                    }}

                    [matrix[i], matrix[maxRow]] = [matrix[maxRow], matrix[i]];
                    [vector[i], vector[maxRow]] = [vector[maxRow], vector[i]];

                    for (let j = i + 1; j < 3; j++) {{
                        const factor = matrix[j][i] / matrix[i][i];
                        for (let k = i; k < 3; k++) {{
                            matrix[j][k] -= factor * matrix[i][k];
                        }}
                        vector[j] -= factor * vector[i];
                    }}
                }}

                let coefficients = [0, 0, 0];
                for (let i = 2; i >= 0; i--) {{
                    coefficients[i] = vector[i];
                    for (let j = i + 1; j < 3; j++) {{
                        coefficients[i] -= matrix[i][j] * coefficients[j];
                    }}
                    coefficients[i] /= matrix[i][i];
                }}

                return {{
                    a: coefficients[2],
                    b: coefficients[1],
                    c: coefficients[0]
                }};
            }}

            function calculateLogisticRegression(xValues, yValues) {{
                const minY = Math.min(...yValues);
                const maxY = Math.max(...yValues);
                if (maxY === minY) {{
                    return null;
                }}
                const normalizedY = yValues.map(y => (y - minY) / (maxY - minY));

                const linearFit = calculateLinearRegression(xValues, normalizedY);

                return {{
                    alpha: linearFit.intercept,
                    beta: linearFit.slope,
                    minY: minY,
                    maxY: maxY
                }};
            }}

            function getRegressionModel(xValues, yValues, regressionType) {{
                if (!xValues || xValues.length < 2) {{
                    return null;
                }}

                if (regressionType === 'linear') {{
                    return calculateLinearRegression(xValues, yValues);
                }}
                if (regressionType === 'polynomial') {{
                    return calculatePolynomialRegression(xValues, yValues);
                }}
                if (regressionType === 'logistic') {{
                    return calculateLogisticRegression(xValues, yValues);
                }}
                return null;
            }}

            function predictRegressionY(x, regressionType, model) {{
                if (!model) return null;

                if (regressionType === 'linear') {{
                    return model.slope * x + model.intercept;
                }}
                if (regressionType === 'polynomial') {{
                    return model.a * x * x + model.b * x + model.c;
                }}
                if (regressionType === 'logistic') {{
                    const normalized = 1 / (1 + Math.exp(-(model.alpha + model.beta * x)));
                    return normalized * (model.maxY - model.minY) + model.minY;
                }}
                return null;
            }}

            function buildResidualRanking(texts, xValues, yValues, regressionType, model) {{
                if (!model) return [];

                const ranked = [];
                for (let i = 0; i < xValues.length; i++) {{
                    const predicted = predictRegressionY(xValues[i], regressionType, model);
                    if (predicted === null || isNaN(predicted)) continue;
                    const residual = yValues[i] - predicted;
                    ranked.push({{
                        text: texts[i],
                        residual,
                        absResidual: Math.abs(residual)
                    }});
                }}

                ranked.sort((a, b) => b.absResidual - a.absResidual);
                return ranked;
            }}

            function generateRegressionLine(xValues, yValues, regressionType) {{
                if (xValues.length < 2) {{
                    return null;
                }}

                const minX = Math.min(...xValues);
                const maxX = Math.max(...xValues);

                const lineX = Array.from(
                    {{ length: CONFIG.regressionLinePoints + 1 }},
                    (_, i) => minX + (maxX - minX) * i / CONFIG.regressionLinePoints
                );

                let lineY = [];

                if (regressionType === 'linear') {{
                    const regression = calculateLinearRegression(xValues, yValues);
                    lineY = lineX.map(x => regression.slope * x + regression.intercept);
                }}
                else if (regressionType === 'polynomial') {{
                    const regression = calculatePolynomialRegression(xValues, yValues);
                    lineY = lineX.map(x => regression.a * x * x + regression.b * x + regression.c);
                }}
                else if (regressionType === 'logistic') {{
                    const regression = calculateLogisticRegression(xValues, yValues);
                    lineY = lineX.map(x => {{
                        const normalized = 1 / (1 + Math.exp(-(regression.alpha + regression.beta * x)));
                        return normalized * (regression.maxY - regression.minY) + regression.minY;
                    }});
                }}

                return {{ x: lineX, y: lineY }};
            }}

            function buildCorrFocusOverlayTrace() {{
                return {{
                    x: [], y: [],
                    mode: 'markers+text',
                    type: 'scatter',
                    meta: CORR_FOCUS_OVERLAY_META,
                    hoverinfo: 'skip',
                    showlegend: false,
                    text: [],
                    textposition: 'middle center',
                    textfont: {{ size: 9, color: '#000' }},
                    marker: {{
                        symbol: 'star',
                        size: CONFIG.focusStarSize,
                        color: '#ffd400',
                        line: {{ color: '#111', width: 2 }}
                    }}
                }};
            }}

            function ensureCorrFocusOverlay() {{
                const correlationPlot = document.getElementById('correlationPlot');
                if (!correlationPlot || !correlationPlot.data) return Promise.resolve();
                if (!correlationPlot.data.some(t => t.meta === CORR_FOCUS_OVERLAY_META)) {{
                    return Plotly.addTraces(correlationPlot, buildCorrFocusOverlayTrace());
                }}
                return Promise.resolve();
            }}

            function bringCorrFocusOverlayToFront() {{
                const correlationPlot = document.getElementById('correlationPlot');
                if (!correlationPlot || !correlationPlot.data) return Promise.resolve();
                const idx = correlationPlot.data.findIndex(t => t.meta === CORR_FOCUS_OVERLAY_META);
                if (idx >= 0 && idx !== correlationPlot.data.length - 1) {{
                    return Plotly.moveTraces(correlationPlot, [idx], [correlationPlot.data.length - 1]);
                }}
                return Promise.resolve();
            }}

            function refreshFocusedTextInCorrelationPlot() {{
                try {{
                    const correlationPlot = document.getElementById('correlationPlot');
                    if (!correlationPlot || !correlationPlot.data || !correlationPlot.data[0]) {{
                        return;
                    }}

                    const scatterTexts = correlationPlot.data[0].text || [];
                    const pointIndex = focusedTextName ? scatterTexts.indexOf(focusedTextName) : -1;

                    ensureCorrFocusOverlay().then(() => {{
                        const overlayIdx = correlationPlot.data.findIndex(t => t.meta === CORR_FOCUS_OVERLAY_META);
                        if (overlayIdx < 0) return Promise.resolve();

                        if (pointIndex < 0) {{
                            return Plotly.restyle(correlationPlot, {{
                                x: [[]], y: [[]], text: [[]]
                            }}, [overlayIdx]);
                        }}

                        const xv = correlationPlot.data[0].x[pointIndex];
                        const yv = correlationPlot.data[0].y[pointIndex];

                        const resolved = readBaseColor(correlationPlot, pointIndex);

                        if (resolved && typeof resolved === 'string') {{
                            return Plotly.restyle(correlationPlot, {{
                                x: [[xv]], y: [[yv]],
                                text: [[focusedTextName]],
                                'marker.color': [[resolved]]
                            }}, [overlayIdx]);
                        }}
                        return Plotly.restyle(correlationPlot, {{
                            x: [[xv]], y: [[yv]],
                            text: [[focusedTextName]]
                        }}, [overlayIdx]);
                    }}).then(() => bringCorrFocusOverlayToFront())
                      .then(() => refreshCorrelationSemanticConnections());
                }} catch (err) {{
                    console.error('refreshFocusedTextInCorrelationPlot failed:', err);
                }}
            }}

            async function refreshCorrelationSemanticConnections() {{
                const correlationPlot = document.getElementById('correlationPlot');
                if (!correlationPlot || !correlationPlot.data || !correlationPlot.data[0]) return;
                const refreshToken = ++correlationSemanticRefreshToken;
                await removeSemanticConnectionTraces(correlationPlot);
                if (refreshToken !== correlationSemanticRefreshToken) return;
                if (!semanticLineModeEnabled || !focusedTextName) {{
                    await bringCorrFocusOverlayToFront();
                    return;
                }}
                const scatterTexts = correlationPlot.data[0].text || [];
                const focusPoint = scatterTexts.indexOf(focusedTextName);
                if (focusPoint < 0) {{
                    await bringCorrFocusOverlayToFront();
                    return;
                }}
                const focusIndex = textNameLookup.get(focusedTextName)?.index;
                if (focusIndex === undefined) {{
                    await bringCorrFocusOverlayToFront();
                    return;
                }}
                const neighbors = getFocusNeighbors(focusIndex).slice(0, semanticLineCount);
                const traces = neighbors.map((item, rank) => {{
                    const targetPoint = scatterTexts.indexOf(item.text);
                    if (targetPoint < 0) return null;
                    return {{ x: [correlationPlot.data[0].x[focusPoint], correlationPlot.data[0].x[targetPoint]],
                        y: [correlationPlot.data[0].y[focusPoint], correlationPlot.data[0].y[targetPoint]],
                        mode: 'lines', type: 'scatter', meta: 'semanticConnection', hoverinfo: 'skip',
                        showlegend: false,
                        line: {{ color: semanticConnectionColor(rank, neighbors.length), width: 2 }} }};
                }}).filter(Boolean);
                if (traces.length && refreshToken === correlationSemanticRefreshToken) {{
                    await Plotly.addTraces(correlationPlot, traces);
                }}
                await bringCorrFocusOverlayToFront();
            }}

            function updateCorrelationPlot() {{
                try {{
                    const xSelect = document.getElementById('xMetricSelect');
                    const ySelect = document.getElementById('yMetricSelect');
                    const regressionSelect = document.getElementById('regressionType');

                    if (!xSelect || !ySelect || !regressionSelect) {{
                        console.error('[correlation] missing dropdowns',
                            {{ xSelect: !!xSelect, ySelect: !!ySelect, regressionSelect: !!regressionSelect }});
                        return;
                    }}

                    correlationXMetric = xSelect.value;
                    correlationYMetric = ySelect.value;

                    const xData = metricsData[correlationXMetric];
                    const yData = metricsData[correlationYMetric];

                    if (!xData || !yData) {{
                        alert('WHAT DID YOU DO!? You selected a metric that has no data! HOW?!?!?');
                        return;
                    }}

                    const scatterTexts = [];
                    const scatterX = [];
                    const scatterY = [];
                    const scatterColors = [];

                    for (let i = 0; i < xData.length; i++) {{
                        if (xData[i] !== null && yData[i] !== null &&
                            !isNaN(xData[i]) && !isNaN(yData[i])) {{
                            scatterTexts.push(textNames[i]);
                            scatterX.push(xData[i]);
                            scatterY.push(yData[i]);
                            scatterColors.push(i);
                        }}
                    }}

                    const scatterTrace = {{
                        x: scatterX,
                        y: scatterY,
                        mode: 'markers+text',
                        type: 'scatter',
                        name: '',
                        text: scatterTexts,
                        textposition: 'middle center',
                        textfont: {{
                            size: CONFIG.pointFontSize,
                            color: 'black'
                        }},
                        marker: {{
                            size: CONFIG.normalSize,
                            color: scatterColors,
                            colorscale: 'Viridis',
                            showscale: false,
                            line: {{ color: 'black', width: 1 }}
                        }},
                        hovertext: scatterTexts,
                        hoverinfo: 'text+x+y'
                    }};

                    const correlation = calculatePearsonCorrelation(xData, yData);

                    let title = `${{metricDisplayNames[correlationXMetric]}} vs ${{metricDisplayNames[correlationYMetric]}}`;
                    if (correlation !== null) {{
                        title += `<br><sub>Pearson r = ${{correlation.toFixed(3)}}</sub>`;
                    }}

                    const layout = {{
                        title: {{ text: title, x: 0.5, xanchor: 'center' }},
                        xaxis: {{
                            title: metricDisplayNames[correlationXMetric],
                            showgrid: true,
                            gridwidth: 1,
                            gridcolor: '#e0e0e0'
                        }},
                        yaxis: {{
                            title: metricDisplayNames[correlationYMetric],
                            showgrid: true,
                            gridwidth: 1,
                            gridcolor: '#e0e0e0'
                        }},
                        plot_bgcolor: '#f8f9fa',
                        paper_bgcolor: 'white',
                        hovermode: 'closest',
                        showlegend: false,
                        height: 800,
                        width: 1200,
                        font: {{ family: 'Arial, sans-serif' }},
                        margin: {{ t: 100, l: 80, r: 80, b: 80 }}
                    }};

                    const regressionType = regressionSelect.value;
                    const regressionData = generateRegressionLine(scatterX, scatterY, regressionType);
                    const regressionModel = getRegressionModel(scatterX, scatterY, regressionType);

                    let traces = [scatterTrace];

                    if (regressionData) {{
                        traces.push({{
                            x: regressionData.x,
                            y: regressionData.y,
                            mode: 'lines',
                            type: 'scatter',
                            name: `${{regressionType}} fit`,
                            line: {{
                                color: 'rgba(220, 53, 69, 0.2)',
                                width: 3,
                                dash: 'solid'
                            }},
                            hoverinfo: 'skip',
                            showlegend: false
                        }});
                    }}

                    Plotly.newPlot('correlationPlot', traces, layout).then(() => {{
                        refreshFocusedTextInCorrelationPlot();
                    }}).catch(err => console.error('[correlation] Plotly.newPlot failed:', err));

                    setTimeout(() => {{
                        addTextOutlines('#correlationPlot .scatterlayer .textpoint text', 'correlationStrokeText');
                    }}, CONFIG.textOutlineDelay);

                    const statsPanel = document.getElementById('correlationStats');
                    const statsBody = statsPanel ? statsPanel.querySelector('.panelBody') : null;
                    const statsTarget = statsBody || statsPanel;
                    if (!statsTarget) {{
                        return;
                    }}

                    while (statsTarget.firstChild) {{
                        statsTarget.removeChild(statsTarget.firstChild);
                    }}

                    const heading = document.createElement('strong');
                    heading.textContent = 'Correlation Statistics';
                    statsTarget.appendChild(heading);

                    const wrapper = document.createElement('div');
                    wrapper.style.marginTop = '8px';

                    const nLine = document.createElement('div');
                    nLine.style.marginBottom = '6px';
                    nLine.textContent = `Shample Size: ${{scatterX.length}}`;
                    wrapper.appendChild(nLine);

                    if (correlation !== null) {{
                        const rSquared = formatTrimmedNumber(correlation * correlation, 4);
                        const rValue = formatTrimmedNumber(correlation, 4);
                        const absCorrelation = Math.abs(correlation);

                        let strength = '';
                        if (absCorrelation >= 0.7) {{
                            strength = ' (Strong)';
                        }} else if (absCorrelation >= 0.4) {{
                            strength = ' (Moderate)';
                        }} else {{
                            strength = ' (Weak)';
                        }}

                        const rLine = document.createElement('div');
                        rLine.style.marginBottom = '6px';
                        rLine.textContent = `Pearson r: ${{rValue}}${{strength}}`;

                        const r2Line = document.createElement('div');
                        r2Line.style.marginBottom = '6px';
                        r2Line.textContent = `R²: ${{rSquared}}`;

                        wrapper.appendChild(rLine);
                        wrapper.appendChild(r2Line);
                    }} else {{
                        const rLine = document.createElement('div');
                        rLine.style.marginBottom = '6px';
                        rLine.textContent = 'Pearson r: unavailable';
                        wrapper.appendChild(rLine);
                    }}

                    const xVar = calculateVariance(scatterX);
                    const yVar = calculateVariance(scatterY);
                    const xStd = calculateStdDev(scatterX);
                    const yStd = calculateStdDev(scatterY);

                    const xStatsLine = document.createElement('div');
                    xStatsLine.style.marginBottom = '6px';
                    xStatsLine.textContent = `X variance/std: ${{formatTrimmedNumber(xVar, 4)}} / ${{formatTrimmedNumber(xStd, 4)}}`;

                    const yStatsLine = document.createElement('div');
                    yStatsLine.style.marginBottom = '6px';
                    yStatsLine.textContent = `Y variance/std: ${{formatTrimmedNumber(yVar, 4)}} / ${{formatTrimmedNumber(yStd, 4)}}`;

                    wrapper.appendChild(xStatsLine);
                    wrapper.appendChild(yStatsLine);

                    const residualRanking = buildResidualRanking(scatterTexts, scatterX, scatterY, regressionType, regressionModel);
                    if (residualRanking.length) {{
                        const topOutliers = residualRanking.slice(0, 5);
                        const leastOutliers = [...residualRanking].slice(-5).reverse();

                        const outlierHeading = document.createElement('div');
                        outlierHeading.style.marginTop = '10px';
                        outlierHeading.style.marginBottom = '4px';
                        outlierHeading.style.fontWeight = '600';
                        outlierHeading.textContent = 'Greatest Outliers';
                        wrapper.appendChild(outlierHeading);

                        topOutliers.forEach((item, idx) => {{
                            const line = document.createElement('div');
                            line.style.marginBottom = '3px';
                            line.textContent = `${{idx + 1}}. ${{item.text}} (|res|=${{formatTrimmedNumber(item.absResidual, 4)}})`;
                            wrapper.appendChild(line);
                        }});

                        const leastHeading = document.createElement('div');
                        leastHeading.style.marginTop = '10px';
                        leastHeading.style.marginBottom = '4px';
                        leastHeading.style.fontWeight = '600';
                        leastHeading.textContent = 'Least Outliers';
                        wrapper.appendChild(leastHeading);

                        leastOutliers.forEach((item, idx) => {{
                            const line = document.createElement('div');
                            line.style.marginBottom = '3px';
                            line.textContent = `${{idx + 1}}. ${{item.text}} (|res|=${{formatTrimmedNumber(item.absResidual, 4)}})`;
                            wrapper.appendChild(line);
                        }});
                    }}

                    statsTarget.appendChild(wrapper);
                }} catch (err) {{
                    console.error('[correlation] updateCorrelationPlot failed:', err);
                    const statsPanel = document.getElementById('correlationStats');
                    const statsBody = statsPanel ? statsPanel.querySelector('.panelBody') : null;
                    if (statsBody) {{
                        const msg = (err && err.message) ? err.message : String(err);
                        const stack = (err && err.stack) ? err.stack : '';
                        statsBody.innerHTML =
                            '<b style="color:#c00;">Correlation error</b><br>' +
                            '<div style="font-size:11px; margin:4px 0;"><b>' +
                            msg.replace(/&/g, '&amp;').replace(/</g, '&lt;') +
                            '</b></div>' +
                            '<pre style="font-size:9px; white-space:pre-wrap; margin:0;">' +
                            stack.replace(/&/g, '&amp;').replace(/</g, '&lt;') +
                            '</pre>';
                    }}
                }}
            }}

            initializeCorrelationDropdowns();

            function setViewMode(mode) {{
                try {{
                    const isCorrelation = (mode === 'correlation');

                    ['metricPanel', 'colorPanel', 'textPanel'].forEach(id => {{
                        const el = document.getElementById(id);
                        if (el) el.style.display = isCorrelation ? 'none' : 'block';
                    }});

                    const corrControls = document.getElementById('correlationControls');
                    const corrStats = document.getElementById('correlationStats');

                    if (isCorrelation) {{
                        correlationContainer.style.display = 'flex';
                        if (corrControls) corrControls.style.display = 'block';
                        if (corrStats) corrStats.style.display = 'block';
                    }} else {{
                        correlationContainer.style.display = 'none';
                        if (corrControls) corrControls.style.display = 'none';
                        if (corrStats) corrStats.style.display = 'none';
                    }}

                    viewMode = mode;
                }} catch (err) {{
                    console.error('setViewMode failed:', err);
                }}
            }}

            document.getElementById('toggleViewBtn').addEventListener('click', function() {{
                if (viewMode === 'umap') {{
                    setViewMode('correlation');
                    updateCorrelationPlot();
                    this.textContent = 'Switch to UMAP View';
                }} else {{
                    setViewMode('umap');
                    this.textContent = 'Switch to Correlation View';
                }}
            }});

            document.getElementById('updateCorrelationBtn').addEventListener('click', updateCorrelationPlot);

            if (xMetricSearchInput) xMetricSearchInput.addEventListener('input', initializeCorrelationDropdowns);
            if (yMetricSearchInput) yMetricSearchInput.addEventListener('input', initializeCorrelationDropdowns);

            document.getElementById('regressionType').addEventListener('change', updateCorrelationPlot);

            document.getElementById('toggleCorrelationLabelsBtn').addEventListener('click', function() {{
                correlationLabelsVisible = !correlationLabelsVisible;
                const correlationPlot = document.getElementById('correlationPlot');
                const displayMode = correlationLabelsVisible ? 'markers+text' : 'markers';

                Plotly.restyle(correlationPlot, {{'mode': displayMode}}, [0]);
                this.textContent = correlationLabelsVisible ? 'Hide Labels' : 'Show Labels';

                if (correlationLabelsVisible) {{
                    setTimeout(() => {{
                        addTextOutlines('#correlationPlot .scatterlayer .textpoint text', 'correlationStrokeText');
                    }}, CONFIG.textOutlineDelay);
                }}
            }});

        }});
    </script>
    '''

        custom_style_and_script = _build_custom_style_and_script(
            metrics_json_str,
            titles_json_str,
            active_metric or 'cluster',
            texts_json,
            metric_names_dict,
            metric_colorscale_dict,
            metric_keys_list,
            programmer_notes_json,
            annotations_enabled_json,
            safe_json_dumps(semantic_embeddings),
            safe_json_dumps(burrows_similarity),
            texts,
            low_token_texts,
            focus_text,
        )

        html_str = html_str.replace('</body>', custom_style_and_script + '</body>')

        title_tag = f'<title>{escape_html_attribute("Centroid plot")}</title>'
        html_str = html_str.replace('</head>', title_tag + '\n</head>')

        if not html_str.strip().endswith('</html>'):
            html_str = html_str.rstrip() + '\n</html>'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_str)

        print(f"\nExported interactive visualization to: {output_path}")
        print(f"Opening in browser...")

        import time
        time.sleep(0.1)

        try:
            webbrowser.open(f'file://{output_path}')
        except Exception as e:
            print(f"Could not automatically open browser: {e}")
            print(f"Please manually open: {output_path}")

    def run(self, texts: List[str], output_csv: str, output_html: str, ignore_ids: Optional[set] = None, **filter_kwargs):
        if ignore_ids is None:
            ignore_ids = set()

        texts = sorted(texts, key=str.lower)

        print(f"Processing {len(texts)} texts...")

        if self.compute_fighting_words:
            self._fighting_words_unigrams = {}
            self._fighting_words_token_counts = {}
            self._fighting_words_unigrams_centroid = {}
            self._fighting_words_token_counts_centroid = {}

        successful = []
        low_token_texts = set()
        style_profiles: Dict[str, np.ndarray] = {}
        metric_names = self._get_target_metric_names(include_global_metrics=True)
        metrics_storage = {name: {} for name in metric_names}
        annotations_enabled = filter_kwargs.get('annotations', False)

        for tx in texts:
            result = self.process_text(tx, ignore_ids=ignore_ids, **filter_kwargs)
            if result is not None:
                self.embeddings[tx] = result['mean_emb']

                if result.get('low_token_text', False):
                    low_token_texts.add(tx)
                else:
                    style_profile = result.get('_burrows_rate_profile')
                    if style_profile is None and not self.no_metrics:
                        fallback_filters = dict(filter_kwargs)
                        fallback_filters['ignore_ids'] = ignore_ids
                        style_profile = self._load_burrows_rate_profile(tx, fallback_filters)
                    if style_profile is not None:
                        style_profiles[tx] = style_profile
                    for metric_name in metric_names:
                        if metric_name in result:
                            metrics_storage[metric_name][tx] = result[metric_name]

                successful.append(tx)

        if not successful:
            print("\nError: No texts successfully processed")
            return

        print(f"\nSuccessfully processed {len(successful)}/{len(texts)} texts")

        if self.mallet_num_topics > 0 and successful:
            try:
                self._train_mallet_topics(
                    successful,
                    ignore_ids,
                    filter_kwargs,
                )
            except Exception as exc:
                print(f"  Warning: MALLET topic modeling failed ({exc}); continuing without topics")

        if self.compute_fighting_words:
            fighting_word_texts = successful

            peak_values = self._compute_fighting_words_values(
                fighting_word_texts, FIGHTING_WORDS_PEAK_RANK, use_centroid=False
            )
            floor_values = self._compute_fighting_words_values(
                fighting_word_texts, FIGHTING_WORDS_FLOOR_RANK, use_centroid=False
            )
            for metric_name, values in (
                (FIGHTING_WORDS_PEAK_METRIC, peak_values),
                (FIGHTING_WORDS_FLOOR_METRIC, floor_values),
            ):
                for tx in fighting_word_texts:
                    metrics_storage[metric_name][tx] = values.get(tx)

            any_separate_centroid = any(
                not self._centroid_equals_stat.get(tx, True) for tx in fighting_word_texts
            )
            if any_separate_centroid:
                peak_values_c = self._compute_fighting_words_values(
                    fighting_word_texts, FIGHTING_WORDS_PEAK_RANK, use_centroid=True
                )
                floor_values_c = self._compute_fighting_words_values(
                    fighting_word_texts, FIGHTING_WORDS_FLOOR_RANK, use_centroid=True
                )
                for metric_name, values in (
                    (FIGHTING_WORDS_PEAK_CENTROID_METRIC, peak_values_c),
                    (FIGHTING_WORDS_FLOOR_CENTROID_METRIC, floor_values_c),
                ):
                    for tx in fighting_word_texts:
                        metrics_storage[metric_name][tx] = values.get(tx)

            print(
                f"  Fighting-words metrics: peak computed "
                f"{sum(value is not None for value in peak_values.values())}/{len(fighting_word_texts)} "
                f"(rank={FIGHTING_WORDS_PEAK_RANK}); floor computed "
                f"{sum(value is not None for value in floor_values.values())}/{len(fighting_word_texts)} "
                f"(rank={FIGHTING_WORDS_FLOOR_RANK}, min_count={FIGHTING_WORDS_FLOOR_MIN_COUNT})"
            )

        emb_matrix = np.vstack([self.embeddings[tx] for tx in successful]).astype(np.float32)

        cluster_indices = list(range(len(successful)))
        requested_cluster_variants = filter_kwargs.get('cluster_variants')
        if requested_cluster_variants is None:
            cluster_variant_count = 1 if len(successful) > 2000 else 11
        else:
            cluster_variant_count = max(1, int(requested_cluster_variants))
        if self.cluster_method == 'leiden':
            print(
                f"  Clustering method: Leiden | k={self.leiden_k_neighbors} | "
                f"resolutions={self.leiden_resolution_min:.2f}-{self.leiden_resolution_max:.2f} | "
                f"variants={cluster_variant_count} | whiten={self.leiden_whiten}"
            )
        else:
            print(f"  Clustering method: HDBSCAN | variants={cluster_variant_count}")
        cluster_variant_labels = self._cluster_embeddings_variants(
            emb_matrix,
            cluster_indices,
            variant_count=cluster_variant_count,
            base_seed=42,
        )

        metrics_data = {}
        for metric_name in metric_names:
            metric_values = [metrics_storage[metric_name].get(tx) for tx in successful]
            if any(v is not None for v in metric_values):
                metrics_data[metric_name] = np.array(
                    [v if v is not None else np.nan for v in metric_values],
                    dtype=float,
                )
            else:
                metrics_data[metric_name] = None

        for metric_key, labels in cluster_variant_labels.items():
            metrics_data[metric_key] = np.array(
                [float(label) if label is not None else np.nan for label in labels],
                dtype=float
            )

        self.similarity_matrix = cosine_similarity(emb_matrix)

        if not self.no_metrics:
            disagreement = self._burrows_cosine_scores(style_profiles, emb_matrix, successful)
            metrics_data['burrows_cosine_disagreement'] = np.array(
                [disagreement.get(text, 0.0) for text in successful],
                dtype=float,
            )

            burrows_distances, burrows_valid = self._burrows_distance_matrix(style_profiles, successful)
        else:
            burrows_distances, burrows_valid = None, []

        normalized_embeddings = emb_matrix / np.maximum(
            np.linalg.norm(emb_matrix, axis=1, keepdims=True), 1e-12
        )
        self.export_csv(successful, output_csv)

        if self.mallet_topic_proportions:
            attached = 0
            for topic_id in range(self.mallet_num_topics):
                metric_key = f'mallet_topic_{topic_id}'
                values = np.array([
                    self.mallet_topic_proportions.get(tx, {}).get(topic_id, np.nan)
                    for tx in successful
                ], dtype=float)
                if np.all(np.isnan(values)):
                    continue
                metrics_data[metric_key] = values
                label_words = self.mallet_topic_labels.get(topic_id, '')
                MetricConfig.METRICS[metric_key] = {
                    'name': f'MALLET Topic {topic_id}'
                            + (f' ({label_words})' if label_words else ''),
                    'title_suffix': f'MALLET LDA topic {topic_id} proportion',
                    'compute_method': '_compute_mallet_topic',
                    'category': 'Topic Modeling',
                }
                attached += 1
            if attached:
                print(f"  Attached {attached} MALLET topic metrics to visualization")

        n_success = len(successful)
        burrows_similarity_matrix = np.zeros((n_success, n_success), dtype=np.float32)
        for i in range(n_success):
            burrows_similarity_matrix[i, i] = 1.0
        if burrows_distances is not None and burrows_valid:
            for row_pos, row_text_idx in enumerate(burrows_valid):
                for col_pos, col_text_idx in enumerate(burrows_valid):
                    if row_pos == col_pos:
                        continue
                    d = float(burrows_distances[row_pos, col_pos])
                    burrows_similarity_matrix[row_text_idx, col_text_idx] = 1.0 / (1.0 + d)

        reduced = self.reduce_dimensions(emb_matrix, len(successful))
        focus_text = filter_kwargs.get('focus_text', '')
        self.visualize(
            reduced,
            successful,
            metrics_data,
            self.active_metric,
            output_html,
            low_token_texts,
            enable_annotations=annotations_enabled,
            focus_text=focus_text,
            semantic_embeddings=normalized_embeddings.tolist(),
            burrows_similarity=burrows_similarity_matrix.tolist(),
        )


def main():
    parser = argparse.ArgumentParser(
        description='Text Classifier - Cluster texts by semantic similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic clustering with default coloring
  python umapper.py text1 text2 text3

  # Color by VADER sentiment
  python umapper.py text1 text2 --color-by vader_compound

    # Enable an optional niche word-count module
    python umapper.py text1 text2 --niche-words <niche>

  # Leiden clustering with a resolution sweep
  python umapper.py --cluster-method leiden --cluster-variants 8

Available metrics for --color-by:
  """ + ", ".join(MetricConfig.get_metric_names())
    )

    parser.add_argument('texts', nargs='*', help='Text folder names to process')

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--nltk-corpus', metavar='NAME',
                              help='Compare every text in an installed NLTK corpus (for example: gutenberg or reuters)')
    source_group.add_argument('--txt-directory', metavar='PATH',
                              help='Compare every .txt file under a directory')
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
    parser.add_argument('--embeddings-only-low-token', action='store_true', dest='embeddings_only_low_token',
                       help='Generate embeddings for texts below min-tokens-total, but skip metric calculations')
    parser.add_argument('--annotations', action='store_true',
                       help='Enable annotation loading and click-to-view notes in the HTML output')
    parser.add_argument('--token-limit', type=int,
                       help='Use only the newest N tokens per text')
    parser.add_argument('--text-token-limit', type=int, dest='text_token_limit',
                       help='Use only the first N tokens per text')
    parser.add_argument('--per-video-token-limit', type=int, dest='per_video_token_limit',
                       help='Use only the first N tokens per video before truncating each file')
    parser.add_argument('--ignore-urls', type=str, dest='ignore_urls',
                       help='Path to txt file with YouTube URLs or video IDs to ignore (one per line)')

    parser.add_argument('--output-csv', default='data/input/text_similarities.csv',
                       help='Output CSV file path')
    parser.add_argument('--output-html', default='data/input/SBERTClustersHD.html',
                       help='Output HTML visualization file path')
    parser.add_argument('--no-cache', action='store_true', help='Disable embedding cache')
    parser.add_argument('--force-latest-cache', action='store_true',
                       help='Bypass cache freshness checks and use newest cache file per text immediately')
    parser.add_argument('--clear-cache', nargs='?', const='all', metavar='TEXT',
                       help='Clear cache for specific text or all texts')

    parser.add_argument('--color-by', choices=None,
                       help='Metric to use for coloring the visualization')
    parser.add_argument('--calculate-perplexity', action='store_true',
                       help='Enable perplexity calculation (slower, loads language model)')
    parser.add_argument('--enable-spacy', action='store_true',
                       help='Enable spaCy-dependent metrics such as lexical_density (slower, loads 40MB model)')
    parser.add_argument('--compute-ngram-entropy', action='store_true',
                       help='Enable n-gram cross-entropy metrics (slower for large token counts)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Skip expensive metrics (fastest mode)')
    parser.add_argument('--word-counts-only', action='store_true',
                       help='Compute/cache only count-style and regex-pattern metrics')
    parser.add_argument('--no-metrics', action='store_true',
                       help='Skip all metric calculations and only run embeddings/clustering/UMAP')
    parser.add_argument('--umap-neighbors', type=int,
                       help='UMAP neighborhood size; defaults to 5 for datasets up to 30 items, otherwise 50')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                       help='Minimum distance between points in UMAP space (default: 0.1; use 0 for tighter groups)')
    parser.add_argument('--umap-epochs', type=int, default=50,
                       help='Number of UMAP optimization epochs (default: 50)')
    parser.add_argument('--cluster-variants', type=int, default=None,
                       help='Number of clustering variants to generate (minimum: 1). Defaults to 11 '
                           'for datasets up to 1,500 items and 1 for larger datasets. For HDBSCAN '
                           'these are seed variations; for Leiden these are resolution steps between '
                           '--leiden-resolution-min and --leiden-resolution-max.')

    parser.add_argument('--cluster-method', choices=['hdbscan', 'leiden'], default='hdbscan',
                       help='Clustering algorithm to use (default: hdbscan). '
                            '"leiden" runs community detection on a k-NN graph built from '
                            '(optionally whitened) embeddings and produces a resolution sweep. '
                            'Great for large datasets (1500+ items) with many small clusters. ')
    parser.add_argument('--leiden-resolution-min', type=float, default=1.15,
                       help='Minimum Leiden resolution parameter (default: 1.15)')
    parser.add_argument('--leiden-resolution-max', type=float, default=3.0,
                       help='Maximum Leiden resolution parameter (default: 3.0)')
    parser.add_argument('--leiden-k-neighbors', type=int, default=20,
                       help='Number of nearest neighbors for the Leiden k-NN graph (default: 20)')
    parser.add_argument('--no-leiden-whiten', action='store_true', dest='no_leiden_whiten',
                       help='Disable PCA whitening before k-NN graph construction '
                       '(use in large (1500+ item) homogenous datasets like gaming corpus for best results)')

    parser.add_argument('--embedding-model', default=SBERT_MODEL,
                       help='Embedding model name (e.g., all-MiniLM-L12-v2, all-mpnet-base-v2, Alibaba-NLP/gte-modernbert-base, thenlper/gte-small)')
    parser.add_argument('--embedding-chunk-size', type=int,
                       help='Override embedding chunk size (in model tokens)')
    parser.add_argument('--embedding-min-chunk', type=int,
                       help='Override minimum embedding chunk size (in model tokens)')
    parser.add_argument('--embedding-batch-size', type=int,
                       help='Override embedding batch size for SentenceTransformer encode()')
    parser.add_argument('--trained-model-path', type=str, dest='trained_model_path',
                       help='Path to a custom fine-tuned SBERT model (e.g., my_finetuned_sbert). If provided, this overrides --embedding-model')
    parser.add_argument('--compute-fighting-words', '--compute-fighting-words-floor',
                       dest='compute_fighting_words', action='store_true',
                       help='Enable fighting-words peak (rank 1) and floor (rank 1000) metrics')
    parser.add_argument('--centroid-mode', choices=['video', 'word'],
                       help='Use centroids for text representation (video or word)')
    parser.add_argument('--centroid-videos', type=int, default=10,
                       help='Number of videos to use for video centroid computation (default: 10)')
    parser.add_argument('--centroid-words', type=int, default=1000000,
                       help='Number of words to use for word centroid computation (default: 1,000,000)')
    parser.add_argument('--stat-word-count', type=int, default=None,
                       help='Number of words to use for statistical metrics when using video centroids '
                            '(defaults to --token-limit if set, else 1,000,000)')
    parser.add_argument('--focus-text', type=str, dest='focus_text',
                       help='Focus text for similarity mapping')

    parser.add_argument('--mallet-topics', type=int, default=0,
                       help='Train a MALLET LDA model with this many topics '
                            '(requires MALLET installed; 0 disables). '
                            'Set MALLET_MEMORY env var (e.g. 12g) for large corpora.')

    parser.add_argument('--niche-words', action='append', default=None,
                        metavar='NICHE',
                        help='Enable niche word-count metric sets (repeatable).')

    args = parser.parse_args()

    if args.umap_neighbors is not None and args.umap_neighbors < 2:
        parser.error('--umap-neighbors must be at least 2')
    if not 0.0 <= args.umap_min_dist <= 1.0:
        parser.error('--umap-min-dist must be between 0 and 1')
    if args.umap_epochs < 1:
        parser.error('--umap-epochs must be positive')

    if args.mallet_topics < 0:
        parser.error('--mallet-topics must be zero or greater')

    if args.txt_depth < 0:
        parser.error('--txt-depth must be zero or greater')

    external_documents: Dict[str, Tuple[str, str]] = {}

    def document_label(raw_label: str, index: int) -> str:
        label = sanitize_filename(raw_label)
        if len(label) > 240:
            digest = hashlib.sha256(label.encode('utf-8')).hexdigest()[:16]
            label = f"{label[:220]}_{digest}"
        return label or f"document_{index}"

    if args.nltk_corpus:
        try:
            from nltk import corpus as nltk_corpus
            corpus_reader = getattr(nltk_corpus, args.nltk_corpus)
            fileids = list(corpus_reader.fileids())
        except (ImportError, AttributeError, LookupError) as exc:
            parser.error(f"could not load NLTK corpus '{args.nltk_corpus}': {exc}")
        for index, fileid in enumerate(fileids):
            try:
                text_content = corpus_reader.raw(fileid)
            except Exception as exc:
                print(f"Warning: skipped NLTK document {fileid!r}: {exc}")
                continue
            label = document_label(
                f"nltk_{args.nltk_corpus}_{fileid}".replace('/', '__').replace('\\', '__'),
                index,
            )
            if label in external_documents:
                label = f"{label}_{index}"
            source_id = hashlib.sha256(text_content.encode('utf-8')).hexdigest()
            external_documents[label] = (text_content, f"nltk:{args.nltk_corpus}:{fileid}:{source_id}")
        if not external_documents:
            parser.error(f"NLTK corpus '{args.nltk_corpus}' contains no readable documents")

    if args.txt_directory:
        input_root = Path(args.txt_directory).expanduser().resolve()
        if not input_root.is_dir():
            parser.error(f"text directory does not exist: {input_root}")
        for filepath in sorted(input_root.rglob('*.txt')):
            if not filepath.is_file():
                continue
            relative_path = filepath.relative_to(input_root)
            if len(relative_path.parts) - 1 > args.txt_depth:
                continue
            try:
                text_content = filepath.read_text(encoding='utf-8')
                stat = filepath.stat()
            except (OSError, UnicodeDecodeError) as exc:
                print(f"Warning: skipped {filepath}: {exc}")
                continue
            label = document_label(f"txt_{relative_path.as_posix()}".replace('/', '__'), len(external_documents))
            if label in external_documents:
                label = f"{label}_{len(external_documents)}"
            source_id = f"file:{filepath}:{stat.st_mtime_ns}:{stat.st_size}"
            external_documents[label] = (text_content, source_id)
        if not external_documents:
            parser.error(f"no readable .txt files found within {args.txt_depth} levels of {input_root}")

    if external_documents and args.centroid_mode == 'video':
        parser.error('external document sources support --centroid-mode word, not video')

    if args.stat_word_count is None:
        args.stat_word_count = args.token_limit if args.token_limit else 1_000_000

    if args.niche_words:
        for niche in dict.fromkeys(args.niche_words):
            try:
                niche_module = importlib.import_module(f'umapper_{niche}')
                register = getattr(niche_module, f'register_{niche}_metrics')
            except (ImportError, AttributeError) as exc:
                parser.error(f"unsupported niche '{niche}': {exc}")
            register(MetricConfig, TextClassifier)

    valid_metric_names = set(MetricConfig.get_metric_names())
    if args.color_by is not None and args.color_by not in valid_metric_names:
        parser.error(
            f"--color-by '{args.color_by}' is not a registered metric"
        )

    if args.cluster_method == 'leiden' and not LEIDEN_AVAILABLE:
        parser.error(
            "--cluster-method leiden requires the 'leidenalg' and 'igraph' packages. "
            "Install them with: pip install leidenalg python-igraph"
        )

    if args.clear_cache:
        cache_dir = Path(__file__).resolve().parents[1] / "cache" / "metrics"
        SimpleCache(cache_dir).clear(None if args.clear_cache == 'all' else args.clear_cache)
        if not args.texts:
            return

    if external_documents:
        args.texts = list(external_documents)
        print(f"Loaded {len(args.texts)} external documents")
    elif not args.texts:
        input_dir = Path(__file__).resolve().parents[1] / "data" / "input"

        if input_dir.exists():
            args.texts = [d.name for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if args.texts:
                print(f"Auto-discovered {len(args.texts)} texts from {input_dir}")
            else:
                parser.error(f"No texts found in {input_dir}. Use --help for usage information.")
                return
        else:
            parser.error("No texts specified and data/input directory not found in expected locations. Use --help for usage information.")
            return

    active_metric = args.color_by

    if args.no_metrics and active_metric:
        parser.error("--no-metrics cannot be used with --color-by")

    if args.word_counts_only and active_metric and active_metric not in TextClassifier.get_word_count_metric_names():
        parser.error("--word-counts-only only supports --color-by metrics in the count/regex subset")

    if active_metric in (FIGHTING_WORDS_PEAK_METRIC, FIGHTING_WORDS_FLOOR_METRIC,
                          FIGHTING_WORDS_PEAK_CENTROID_METRIC, FIGHTING_WORDS_FLOOR_CENTROID_METRIC) \
            and not args.compute_fighting_words:
        parser.error(f"--color-by {active_metric} requires --compute-fighting-words")

    needs_perplexity = (not args.no_metrics) and (args.calculate_perplexity or (active_metric == 'perplexity'))

    enable_spacy = (not args.no_metrics) and (args.enable_spacy or (active_metric and MetricConfig.requires_spacy_model(active_metric)))

    enable_ngram_entropy = (not args.no_metrics) and (args.compute_ngram_entropy or (active_metric and MetricConfig.requires_ngram_entropy(active_metric)))

    embedding_model = args.embedding_model
    if args.trained_model_path:
        trained_model_path = Path(args.trained_model_path)
        if not trained_model_path.exists():
            parser.error(f"Trained model not found at: {args.trained_model_path}")
        embedding_model = str(trained_model_path.resolve())
        print(f"[INFO] Using trained model: {embedding_model}")

    classifier = TextClassifier(
        use_cache=not args.no_cache,
        active_metric=active_metric,
        enable_spacy=enable_spacy,
        enable_ngram_entropy=enable_ngram_entropy,
        fast_mode=args.fast_mode,
        word_counts_only=args.word_counts_only,
        force_latest_cache=args.force_latest_cache,
        embedding_model=embedding_model,
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
        mallet_num_topics=args.mallet_topics,
    )

    classifier.active_focus_text = args.focus_text if args.focus_text else ''
    if external_documents:
        classifier.processor.set_external_documents(external_documents)

    if needs_perplexity:
        classifier.processor.needs_perplexity_model = True
        classifier.processor._initialize_perplexity_model('distilgpt2')

    ignore_ids = load_ignore_urls(args.ignore_urls)

    classifier.run(
        args.texts, args.output_csv, args.output_html,
        ignore_ids=ignore_ids,
        date_from=args.date_from, date_to=args.date_to,
        duration_from=args.duration_from, duration_to=args.duration_to,
        exclude_live=args.exclude_live, token_limit=args.token_limit,
        text_token_limit=args.text_token_limit,
        per_video_token_limit=args.per_video_token_limit,
        min_tokens_total=args.min_tokens_total,
        min_tokens_per_file=args.min_tokens_per_file,
        embeddings_only_low_token=args.embeddings_only_low_token,
        cluster_variants=args.cluster_variants,
        annotations=args.annotations,
        focus_text=args.focus_text
    )


if __name__ == "__main__":
    main()