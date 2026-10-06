from atlas_config import *  # noqa


class SimpleCache:

    def __init__(self, cache_dir: str = "cache/metrics"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        self._filter_hash_cache = {}
        self._eligible_cache = {}

    def _cache_path(self, text: str, filters: Dict) -> Tuple[Path, Path]:
        safe_text = sanitize_filename(text)

        filter_key = frozenset(filters.items())
        if filter_key not in self._filter_hash_cache:
            sorted_filters = dict(sorted(filters.items()))
            filter_str = json_dumps(sorted_filters)
            self._filter_hash_cache[filter_key] = hashlib.md5(filter_str.encode()).hexdigest()[:8]
        filter_hash = self._filter_hash_cache[filter_key]
        base_name = f"{safe_text}_{filter_hash}"
        json_path = self.cache_dir / f"{base_name}.json"
        npy_path = self.cache_dir / f"{base_name}.npy"
        return json_path, npy_path

    def _fingerprint(self, video_ids: List[str]) -> str:
        if not video_ids:
            return ""
        vid_str = ','.join(sorted(video_ids))
        return hashlib.md5(vid_str.encode()).hexdigest()

    def _get_eligible_metrics(self, enable_spacy: bool, needs_perplexity_model: bool,
                              enable_ngram_entropy: bool, fast_mode: bool) -> set:
        config_key = (enable_spacy, needs_perplexity_model, enable_ngram_entropy, fast_mode)
        if config_key not in self._eligible_cache:
            self._eligible_cache[config_key] = {
                m for m in MetricConfig.get_metric_names()
                if MetricConfig.is_metric_eligible(m, enable_spacy, needs_perplexity_model, enable_ngram_entropy, fast_mode)
            }
        return self._eligible_cache[config_key]

    def load(self, text: str, filters: Dict, video_ids: List[str],
             enable_spacy: bool = False,
             needs_perplexity_model: bool = False,
             enable_ngram_entropy: bool = False,
             fast_mode: bool = False,
             expected_token_count: Optional[int] = None,
             embeddings_only: bool = False,
             force_latest_cache: bool = False) -> Optional[Dict]:
        if not validate_text_name(text):
            print(f"  Warning: Invalid text name '{text}' rejected for security")
            return None

        json_path, npy_path = self._cache_path(text, filters)
        fingerprint = None if force_latest_cache else self._fingerprint(video_ids)

        if force_latest_cache:
            safe_text = sanitize_filename(text)
            candidates = []
            for candidate in self.cache_dir.glob(f"{safe_text}_*.json"):
                try:
                    mtime = candidate.stat().st_mtime
                except OSError:
                    continue
                candidates.append((mtime, candidate))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                json_path = candidates[0][1]
                npy_path = self.cache_dir / f"{json_path.stem}.npy"

        if json_path in self._memory_cache:
            cached_data, cached_fp = self._memory_cache[json_path]
            if force_latest_cache or cached_fp == fingerprint:
                print(f"  Memory cache hit: {text}")
                if not embeddings_only:
                    if force_latest_cache:
                        return {
                            'metrics': cached_data.get('metrics', {}),
                            'missing_metrics': [],
                            'embeddings': cached_data.get('embeddings'),
                            'mean_emb': cached_data.get('mean_emb'),
                            'chunks': cached_data.get('chunks')
                        }

                    eligible = self._get_eligible_metrics(enable_spacy, needs_perplexity_model, enable_ngram_entropy, fast_mode)
                    cache_version = cached_data.get('version')
                    cache_stale = cache_version != METRICS_VERSION
                    cached_metrics = set(cached_data.get('metrics', {}).keys())
                    missing = list(eligible) if cache_stale else list(eligible - cached_metrics)

                    if cache_stale:
                        print(f"    -> stale metric cache (v{cache_version} != v{METRICS_VERSION}); recomputing metrics")
                        return {
                            'metrics': {},
                            'missing_metrics': missing,
                            'embeddings': cached_data.get('embeddings'),
                            'mean_emb': cached_data.get('mean_emb'),
                            'chunks': cached_data.get('chunks')
                        }

                    if missing:
                        print(f"    -> {len(missing)} new metrics to compute")

                    return {
                        'metrics': cached_data.get('metrics', {}),
                        'missing_metrics': missing,
                        'embeddings': cached_data.get('embeddings'),
                        'mean_emb': cached_data.get('mean_emb'),
                        'chunks': cached_data.get('chunks')
                    }

                return {
                    'metrics': {},
                    'missing_metrics': [],
                    'embeddings': cached_data.get('embeddings'),
                    'mean_emb': cached_data.get('mean_emb'),
                    'chunks': cached_data.get('chunks')
                }

        if not json_path.exists():
            if force_latest_cache:
                return None
            safe_text = sanitize_filename(text)
            fallback_found = False
            if expected_token_count is not None:
                for candidate in self.cache_dir.glob(f"{safe_text}_*.json"):
                    try:
                        with open(candidate, 'rb') as f:
                            file_data = f.read()
                        cached = json_loads(file_data)
                        if cached.get('fingerprint') != fingerprint:
                            continue
                        if cached.get('tokens') != expected_token_count:
                            continue
                        if cached.get('version') != METRICS_VERSION:
                            continue
                        json_path = candidate
                        npy_path = self.cache_dir / f"{candidate.stem}.npy"
                        fallback_found = True
                        break
                    except Exception:
                        continue
            if (not fallback_found) and embeddings_only and expected_token_count is not None:
                for candidate in self.cache_dir.glob(f"{safe_text}_*.json"):
                    try:
                        with open(candidate, 'rb') as f:
                            file_data = f.read()
                        cached = json_loads(file_data)
                        if cached.get('tokens') != expected_token_count:
                            continue
                        if cached.get('version') != METRICS_VERSION:
                            continue
                        candidate_npy = self.cache_dir / f"{candidate.stem}.npy"
                        if not candidate_npy.exists():
                            continue
                        json_path = candidate
                        npy_path = candidate_npy
                        fallback_found = True
                        break
                    except Exception:
                        continue
            if not fallback_found:
                return None

        try:
            import time
            t0 = time.time()
            if 'cached' in locals():
                load_time = time.time() - t0
            else:
                with open(json_path, 'rb') as f:
                    file_data = f.read()
                cached = json_loads(file_data)
                load_time = time.time() - t0

            if not force_latest_cache and cached.get('fingerprint') != fingerprint:
                return None

            cache_version = cached.get('version')
            cache_stale = (cache_version != METRICS_VERSION) and not force_latest_cache

            embeddings_data = None
            if npy_path.exists():
                try:
                    emb_data = np.load(npy_path, allow_pickle=True).item()
                    embeddings_data = {
                        'embeddings': emb_data['chunk_embs'],
                        'mean_emb': emb_data['mean_emb'],
                        'chunks': emb_data['chunks']
                    }
                except Exception as e:
                    print(f"  Warning: Failed to load embeddings: {e}")

            cached_with_embs = {**cached, **(embeddings_data or {})}

            memory_fp = cached.get('fingerprint') if cached.get('fingerprint') is not None else (fingerprint or "")
            self._memory_cache[json_path] = (cached_with_embs, memory_fp)

            if len(self._memory_cache) > 50:
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]

            emb_status = " + embeddings" if embeddings_data else ""
            print(f"  Cached: {text} ({cached.get('tokens', 0):,} tokens{emb_status})")

            if not embeddings_only:
                if force_latest_cache:
                    return {
                        'metrics': cached.get('metrics', {}),
                        'missing_metrics': [],
                        'embeddings': embeddings_data['embeddings'] if embeddings_data else None,
                        'mean_emb': embeddings_data['mean_emb'] if embeddings_data else None,
                        'chunks': embeddings_data['chunks'] if embeddings_data else None,
                        'token_count': cached.get('tokens', 0)
                    }

                eligible = self._get_eligible_metrics(enable_spacy, needs_perplexity_model, enable_ngram_entropy, fast_mode)
                cached_metrics = set(cached.get('metrics', {}).keys())
                missing = list(eligible) if cache_stale else list(eligible - cached_metrics)

                if cache_stale:
                    print(f"    -> stale metric cache (v{cache_version} != v{METRICS_VERSION}); recomputing metrics")
                    return {
                        'metrics': {},
                        'missing_metrics': missing,
                        'embeddings': embeddings_data['embeddings'] if embeddings_data else None,
                        'mean_emb': embeddings_data['mean_emb'] if embeddings_data else None,
                        'chunks': embeddings_data['chunks'] if embeddings_data else None,
                        'token_count': cached.get('tokens', 0)
                    }

                if missing:
                    print(f"    -> {len(missing)} new metrics to compute")

                return {
                    'metrics': cached.get('metrics', {}),
                    'missing_metrics': missing,
                    'embeddings': embeddings_data['embeddings'] if embeddings_data else None,
                    'mean_emb': embeddings_data['mean_emb'] if embeddings_data else None,
                    'chunks': embeddings_data['chunks'] if embeddings_data else None,
                    'token_count': cached.get('tokens', 0)
                }

            return {
                'metrics': {},
                'missing_metrics': [],
                'embeddings': embeddings_data['embeddings'] if embeddings_data else None,
                'mean_emb': embeddings_data['mean_emb'] if embeddings_data else None,
                'chunks': embeddings_data['chunks'] if embeddings_data else None,
                'token_count': cached.get('tokens', 0)
            }
        except Exception as e:
            print(f"  Cache load error: {e}")
            return None

    def save(self, text: str, filters: Dict, result_dict: Dict, video_ids: List[str],
             token_count: int, chunk_count: int,
             chunk_embs: np.ndarray = None, mean_emb: np.ndarray = None, chunks: List[str] = None,
             enable_spacy: bool = False,
             needs_perplexity_model: bool = False,
             enable_ngram_entropy: bool = False,
             fast_mode: bool = False):
        if not validate_text_name(text):
            print(f"  Warning: Invalid text name '{text}' rejected for security")
            return

        json_path, npy_path = self._cache_path(text, filters)
        try:
            fast_mode_excluded = MetricConfig.FAST_MODE_EXCLUDED

            metrics = {}
            for k, v in result_dict.items():
                if k not in MetricConfig.METRICS or v is None:
                    continue
                if k in ('ngram_entropy_2', 'ngram_entropy_3') and not enable_ngram_entropy:
                    continue
                if k in MetricConfig.FAST_MODE_EXCLUDED and fast_mode:
                    continue
                if isinstance(v, (np.integer, np.floating)):
                    metrics[k] = float(v)
                else:
                    metrics[k] = v

            cache_data = {
                'version': METRICS_VERSION,
                'tokens': token_count,
                'chunks': chunk_count,
                'fingerprint': self._fingerprint(video_ids),
                'metrics': metrics
            }

            if mean_emb is not None:
                np.save(npy_path, {
                    'chunk_embs': chunk_embs,
                    'mean_emb': mean_emb,
                    'chunks': chunks or []
                })

            memory_data = cache_data.copy()
            if chunk_embs is not None:
                memory_data['embeddings'] = chunk_embs
                memory_data['mean_emb'] = mean_emb
                memory_data['chunks'] = chunks
            self._memory_cache[json_path] = (memory_data, cache_data['fingerprint'])

            if len(self._memory_cache) > 50:
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]

            with open(json_path, 'wb') as f:
                f.write(json_dumps(cache_data).encode('utf-8') if JSON_LIB != 'orjson' else orjson.dumps(cache_data))
        except Exception as e:
            print(f"  Cache save error: {e}")

    def clear(self, text: Optional[str] = None):
        if text and not validate_text_name(text):
            print(f"Warning: Invalid text name '{text}' rejected for security")
            return
        safe_text = sanitize_filename(text) if text else None
        pattern = f"{safe_text}_*" if safe_text else "*"
        for ext in ['.json', '.npy']:
            for f in list(self.cache_dir.glob(f"{pattern}{ext}")):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
        if text:
            keys_to_remove = [k for k in self._memory_cache if isinstance(k, Path) and text in k.name]
            for k in keys_to_remove:
                del self._memory_cache[k]
        else:
            self._memory_cache.clear()
        print(f"Cleared cache for {text or 'all texts'}")