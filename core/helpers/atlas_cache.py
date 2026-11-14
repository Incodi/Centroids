import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from helpers.atlas_config import *  # noqa
from helpers.atlas_config import (
    JSON_LIB,
    METRICS_VERSION,
    MetricConfig,
    json_dumps,
    json_loads,
    sanitize_filename,
    validate_text_name,
)

try:
    import orjson
except ImportError:
    orjson = None


class SimpleCache:
    """Filesystem-backed per-text metric cache.

    Lenient by design:

    * Filter keys that don't affect tokenization (notably ``metric_signature``)
      are excluded from the cache path hash.  Adding/removing metrics reuses
      the same file and simply appends missing entries.
    * ``METRICS_VERSION`` bumps no longer invalidate previously cached metrics;
      the caller only recomputes metrics that are truly absent from the cache.
    * ``save`` merges with any existing cache for the same corpus (same
      ``tokens`` count) that it can find — including one reached through the
      fallback scanner — so a run with e.g. ``--fast-mode`` cannot wipe
      slow-mode metrics computed earlier.
    * The fallback path scan works even when the caller passes
      ``expected_token_count=None`` (used by centroid modes, where the
      accumulated stat corpus can be slightly below the requested cap due to
      chunk boundaries).  When the caller supplies a token count, it's used
      for exact matching on ``load`` and on the ``save`` merge.
    * When a fallback cache is used, it's promoted (copied) to the current
      path so future runs take the fast path.
    """

    # Keys that must NOT participate in the cache-path hash.  These don't
    # affect the token stream, so different values can share the same file.
    _PATH_EXCLUDED_KEYS = frozenset({'metric_signature'})

    def __init__(self, cache_dir: str = "cache/metrics"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        self._filter_hash_cache = {}
        self._eligible_cache = {}

    def _path_filters(self, filters: Dict) -> Dict:
        """Filters used for the on-disk filename hash (excludes non-token keys)."""
        return {
            k: v for k, v in filters.items()
            if k not in self._PATH_EXCLUDED_KEYS
        }

    def _cache_path(self, text: str, filters: Dict) -> Tuple[Path, Path]:
        safe_text = sanitize_filename(text)
        path_filters = self._path_filters(filters)

        try:
            filter_key = frozenset(path_filters.items())
        except TypeError:
            # Unhashable filter value — degrade to a JSON-string key
            filter_key = json_dumps(dict(sorted(path_filters.items())))

        if filter_key not in self._filter_hash_cache:
            sorted_filters = dict(sorted(path_filters.items()))
            filter_str = json_dumps(sorted_filters)
            self._filter_hash_cache[filter_key] = (
                hashlib.md5(filter_str.encode()).hexdigest()[:8]
            )

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

    def _get_eligible_metrics(
        self,
        enable_spacy: bool,
        needs_perplexity_model: bool,
        enable_ngram_entropy: bool,
        fast_mode: bool,
    ) -> set:
        config_key = (
            enable_spacy,
            needs_perplexity_model,
            enable_ngram_entropy,
            fast_mode,
        )

        if config_key not in self._eligible_cache:
            self._eligible_cache[config_key] = {
                m for m in MetricConfig.get_metric_names()
                if MetricConfig.is_metric_eligible(
                    m,
                    enable_spacy,
                    needs_perplexity_model,
                    enable_ngram_entropy,
                    fast_mode,
                )
            }

        return self._eligible_cache[config_key]

    def load(
        self,
        text: str,
        filters: Dict,
        video_ids: List[str],
        enable_spacy: bool = False,
        needs_perplexity_model: bool = False,
        enable_ngram_entropy: bool = False,
        fast_mode: bool = False,
        expected_token_count: Optional[int] = None,
        embeddings_only: bool = False,
        force_latest_cache: bool = False,
        force_cache_same_stat: bool = False,
    ) -> Optional[Dict]:
        if not validate_text_name(text):
            print(f"  Warning: Invalid text name '{text}' rejected for security")
            return None

        # Path the current filter set maps to.  If the fallback is used, the
        # file actually loaded may live at a different path; we promote it to
        # this canonical location after a successful load so future runs hit
        # the fast path directly.
        canonical_json_path, canonical_npy_path = self._cache_path(text, filters)
        json_path, npy_path = canonical_json_path, canonical_npy_path
        fingerprint = None if force_latest_cache else self._fingerprint(video_ids)

        if force_latest_cache:
            json_path, npy_path = self._pick_latest_cache_files(text)

        memory_hit = self._try_memory_cache(
            json_path,
            fingerprint,
            text,
            force_latest_cache,
            embeddings_only,
            enable_spacy,
            needs_perplexity_model,
            enable_ngram_entropy,
            fast_mode,
        )
        if memory_hit is not None:
            return memory_hit

        used_fallback = False
        if not json_path.exists():
            if force_latest_cache:
                return None

            fallback = self._find_fallback_cache(
                text, filters, fingerprint, expected_token_count,
                embeddings_only,
                force_cache_same_stat=force_cache_same_stat,
            )
            if fallback is None:
                return None

            json_path, npy_path = fallback
            used_fallback = True

        try:
            if 'cached' not in locals():
                with open(json_path, 'rb') as f:
                    file_data = f.read()
                cached = json_loads(file_data)

            if not force_latest_cache and cached.get('fingerprint') != fingerprint:
                return None

            # NOTE: cache_version is read for logging only.  A METRICS_VERSION
            # bump no longer invalidates previously cached metrics — only
            # metrics that are actually missing get recomputed.
            cache_version = cached.get('version')
            if cache_version != METRICS_VERSION:
                print(
                    f"  Note: {text} cached at v{cache_version} "
                    f"(current v{METRICS_VERSION}); keeping existing metrics, "
                    f"only missing ones will be recomputed"
                )

            embeddings_data = self._load_embeddings_data(npy_path)
            cached_with_embs = {**cached, **(embeddings_data or {})}

            memory_fp = (
                cached.get('fingerprint')
                if cached.get('fingerprint') is not None
                else (fingerprint or "")
            )
            self._store_memory_cache(json_path, cached_with_embs, memory_fp)

            emb_status = " + embeddings" if embeddings_data else ""
            if used_fallback:
                print(
                    f"  Cached (fallback path): {text} "
                    f"({cached.get('tokens', 0):,} tokens{emb_status})"
                )
                # Promote so next run hits the canonical path directly.
                self._promote_cache(json_path, npy_path, canonical_json_path, canonical_npy_path)
            else:
                print(f"  Cached: {text} ({cached.get('tokens', 0):,} tokens{emb_status})")

            if not embeddings_only:
                return self._build_cache_result(
                    cached,
                    embeddings_data,
                    force_latest_cache,
                    enable_spacy,
                    needs_perplexity_model,
                    enable_ngram_entropy,
                    fast_mode,
                )

            return {
                'metrics': {},
                'missing_metrics': [],
                'embeddings': embeddings_data['embeddings'] if embeddings_data else None,
                'mean_emb': embeddings_data['mean_emb'] if embeddings_data else None,
                'chunks': embeddings_data['chunks'] if embeddings_data else None,
                'token_count': cached.get('tokens', 0),
            }
        except Exception as e:
            print(f"  Cache load error: {e}")
            return None

    def _promote_cache(
        self,
        old_json: Path,
        old_npy: Path,
        new_json: Path,
        new_npy: Path,
    ) -> None:
        """Copy a fallback cache file to the canonical path for this filter set."""
        try:
            if old_json != new_json and old_json.exists():
                shutil.copy2(old_json, new_json)
            if old_npy != new_npy and old_npy.exists():
                shutil.copy2(old_npy, new_npy)
        except Exception as exc:
            print(f"  Warning: cache promotion failed ({exc})")

    def _pick_latest_cache_files(self, text: str) -> Tuple[Path, Path]:
        safe_text = sanitize_filename(text)
        candidates = []

        for candidate in self.cache_dir.glob(f"{safe_text}_*.json"):
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, candidate))

        if not candidates:
            return self._cache_path(text, {})

        candidates.sort(key=lambda item: item[0], reverse=True)
        json_path = candidates[0][1]
        npy_path = self.cache_dir / f"{json_path.stem}.npy"
        return json_path, npy_path

    def _try_memory_cache(
        self,
        json_path: Path,
        fingerprint: Optional[str],
        text: str,
        force_latest_cache: bool,
        embeddings_only: bool,
        enable_spacy: bool,
        needs_perplexity_model: bool,
        enable_ngram_entropy: bool,
        fast_mode: bool,
    ) -> Optional[Dict]:
        if json_path not in self._memory_cache:
            return None

        cached_data, cached_fp = self._memory_cache[json_path]
        if not (force_latest_cache or cached_fp == fingerprint):
            return None

        print(f"  Memory cache hit: {text}")

        if embeddings_only:
            return {
                'metrics': {},
                'missing_metrics': [],
                'embeddings': cached_data.get('embeddings'),
                'mean_emb': cached_data.get('mean_emb'),
                'chunks': cached_data.get('chunks'),
            }

        if force_latest_cache:
            return {
                'metrics': cached_data.get('metrics', {}),
                'missing_metrics': [],
                'embeddings': cached_data.get('embeddings'),
                'mean_emb': cached_data.get('mean_emb'),
                'chunks': cached_data.get('chunks'),
            }

        eligible = self._get_eligible_metrics(
            enable_spacy,
            needs_perplexity_model,
            enable_ngram_entropy,
            fast_mode,
        )
        cached_metrics = set(cached_data.get('metrics', {}).keys())
        # Lenient: only metrics never computed before are "missing".  Cached
        # values are kept verbatim regardless of METRICS_VERSION.
        missing = list(eligible - cached_metrics)

        if missing:
            print(f"    -> {len(missing)} new metrics to compute")

        return {
            'metrics': cached_data.get('metrics', {}),
            'missing_metrics': missing,
            'embeddings': cached_data.get('embeddings'),
            'mean_emb': cached_data.get('mean_emb'),
            'chunks': cached_data.get('chunks'),
        }

    def _find_fallback_cache(
        self,
        text: str,
        filters: Optional[Dict],
        fingerprint: Optional[str],
        expected_token_count: Optional[int],
        embeddings_only: bool,
        force_cache_same_stat: bool = False,
    ) -> Optional[Tuple[Path, Path]]:
        """Look for any cache file for `text` that matches the current corpus.

        Lenient rules:

        * If the cache stores ``filters``, they must match the current
          tokenization-affecting filters (metric_signature excluded).  Legacy
          caches without a ``filters`` snapshot are accepted on trust.
        * If ``expected_token_count`` is known, the cached token count must
          match exactly.  If it is ``None`` (centroid modes), we accept any
          token count — the accumulated stat corpus can be slightly below the
          requested cap because of chunk boundaries, and enforcing equality
          there was causing spurious cache misses that changed which videos
          contributed to a channel's embedding.
        * Among matching candidates, prefer newer files, and prefer files that
          carry a ``filters`` snapshot (they can be cross-checked) over legacy
          files.
        """
        safe_text = sanitize_filename(text)
        want_fword_sub = bool((filters or {}).get('change_fword_to_stupid', False))
        want_path_filters = self._path_filters(filters) if filters else None

        def filter_match(cached: Dict) -> bool:
            cached_filters = cached.get('filters')
            if cached_filters is None:
                return True  # legacy cache without filter snapshot
            if want_path_filters is None:
                return True
            if force_cache_same_stat:
                # --force-cache-same-stat: treat a cache as compatible even when
                # centroid_videos differs. Only centroid_videos is relaxed;
                # stat_word_count / centroid_mode / everything else must match.
                cached_cmp = {
                    k: v for k, v in cached_filters.items() if k != 'centroid_videos'
                }
                want_cmp = {
                    k: v for k, v in want_path_filters.items() if k != 'centroid_videos'
                }
                return cached_cmp == want_cmp
            return cached_filters == want_path_filters

        def token_match(cached_tokens: Optional[int]) -> bool:
            if force_cache_same_stat:
                return True
            if expected_token_count is None:
                return True
            return cached_tokens == expected_token_count

        # Collect candidates: priority (has filters snapshot) + mtime.
        scored: List[Tuple[int, float, Path, Path]] = []

        for candidate in self.cache_dir.glob(f"{safe_text}_*.json"):
            try:
                with open(candidate, 'rb') as f:
                    cached = json_loads(f.read())

                if cached.get('fingerprint') != fingerprint:
                    continue
                if not token_match(cached.get('tokens')):
                    continue
                if bool(cached.get('change_fword_to_stupid', False)) != want_fword_sub:
                    continue
                if not filter_match(cached):
                    continue

                has_filters = cached.get('filters') is not None
                priority = 2 if has_filters else 1
                try:
                    mtime = candidate.stat().st_mtime
                except OSError:
                    mtime = 0.0
                scored.append((
                    priority,
                    mtime,
                    candidate,
                    self.cache_dir / f"{candidate.stem}.npy",
                ))
            except Exception:
                continue

        if scored:
            # Prefer higher-priority (filters snapshot present), then newest.
            scored.sort(key=lambda c: (-c[0], -c[1]))
            _, _, json_path, npy_path = scored[0]
            return json_path, npy_path

        if embeddings_only:
            for candidate in self.cache_dir.glob(f"{safe_text}_*.json"):
                try:
                    with open(candidate, 'rb') as f:
                        cached = json_loads(f.read())

                    if not token_match(cached.get('tokens')):
                        continue
                    if bool(cached.get('change_fword_to_stupid', False)) != want_fword_sub:
                        continue
                    if not filter_match(cached):
                        continue

                    candidate_npy = self.cache_dir / f"{candidate.stem}.npy"
                    if not candidate_npy.exists():
                        continue

                    return candidate, candidate_npy
                except Exception:
                    continue

        return None

    def _load_embeddings_data(self, npy_path: Path) -> Optional[Dict]:
        if not npy_path.exists():
            return None

        try:
            emb_data = np.load(npy_path, allow_pickle=True).item()
            return {
                'embeddings': emb_data['chunk_embs'],
                'mean_emb': emb_data['mean_emb'],
                'chunks': emb_data['chunks'],
            }
        except Exception as e:
            print(f"  Warning: Failed to load embeddings: {e}")
            return None

    def _store_memory_cache(self, json_path: Path, data: Dict, fingerprint: str) -> None:
        self._memory_cache[json_path] = (data, fingerprint)

        if len(self._memory_cache) > 50:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

    def _build_cache_result(
        self,
        cached: Dict,
        embeddings_data: Optional[Dict],
        force_latest_cache: bool,
        enable_spacy: bool,
        needs_perplexity_model: bool,
        enable_ngram_entropy: bool,
        fast_mode: bool,
    ) -> Dict:
        emb = embeddings_data['embeddings'] if embeddings_data else None
        mean_emb = embeddings_data['mean_emb'] if embeddings_data else None
        chunks = embeddings_data['chunks'] if embeddings_data else None

        if force_latest_cache:
            return {
                'metrics': cached.get('metrics', {}),
                'missing_metrics': [],
                'embeddings': emb,
                'mean_emb': mean_emb,
                'chunks': chunks,
                'token_count': cached.get('tokens', 0),
            }

        eligible = self._get_eligible_metrics(
            enable_spacy,
            needs_perplexity_model,
            enable_ngram_entropy,
            fast_mode,
        )
        cached_metrics = set(cached.get('metrics', {}).keys())
        # Lenient: only metrics that have never been computed are "missing".
        # METRICS_VERSION bumps no longer nuke the metric dict.
        missing = list(eligible - cached_metrics)

        if missing:
            print(f"    -> {len(missing)} new metrics to compute")

        return {
            'metrics': cached.get('metrics', {}),
            'missing_metrics': missing,
            'embeddings': emb,
            'mean_emb': mean_emb,
            'chunks': chunks,
            'token_count': cached.get('tokens', 0),
        }

    def save(
        self,
        text: str,
        filters: Dict,
        result_dict: Dict,
        video_ids: List[str],
        token_count: int,
        chunk_count: int,
        chunk_embs: np.ndarray = None,
        mean_emb: np.ndarray = None,
        chunks: List[str] = None,
        enable_spacy: bool = False,
        needs_perplexity_model: bool = False,
        enable_ngram_entropy: bool = False,
        fast_mode: bool = False,
        force_cache_same_stat: bool = False,
    ):
        if not validate_text_name(text):
            print(f"  Warning: Invalid text name '{text}' rejected for security")
            return

        json_path, npy_path = self._cache_path(text, filters)
        fingerprint = self._fingerprint(video_ids)

        try:
            new_metrics = self._collect_metrics_for_cache(
                result_dict, enable_ngram_entropy, fast_mode
            )

            existing_metrics: Dict[str, Any] = {}
            existing_tokens: Optional[int] = None

            # Try the canonical path first.
            if json_path.exists():
                try:
                    with open(json_path, 'rb') as f:
                        existing_data = json_loads(f.read())
                    existing_metrics = existing_data.get('metrics', {}) or {}
                    existing_tokens = existing_data.get('tokens')
                except Exception as exc:
                    print(
                        f"  Warning: could not read existing cache at "
                        f"{json_path.name} ({exc})"
                    )
            else:
                # No file at the canonical path — look for one via the
                # fallback scanner so that metrics from an older path hash
                # (e.g. from before metric_signature was excluded) are
                # preserved across the promotion.  Exact token match only:
                # merging metrics across corpora with different token counts
                # would be wrong.
                fallback = self._find_fallback_cache(
                    text, filters, fingerprint, token_count,
                    embeddings_only=False,
                    force_cache_same_stat=force_cache_same_stat,
                )
                if fallback is not None:
                    try:
                        with open(fallback[0], 'rb') as f:
                            existing_data = json_loads(f.read())
                        existing_metrics = existing_data.get('metrics', {}) or {}
                        existing_tokens = existing_data.get('tokens')
                    except Exception:
                        pass

            # Only merge when the underlying token stream is identical.
            if existing_tokens is not None and existing_tokens == token_count:
                metrics = {**existing_metrics, **new_metrics}
                token_count_for_cache = token_count
            elif force_cache_same_stat and existing_tokens is not None:
                # --force-cache-same-stat: user opted into treating the cache as the
                # same stat corpus even though the token count differs (typically
                # because only centroid_videos changed). Merge rather than wipe, and
                # preserve the existing token count so the cache identity is stable.
                metrics = {**existing_metrics, **new_metrics}
                token_count_for_cache = existing_tokens
            else:
                metrics = new_metrics
                token_count_for_cache = token_count

            cache_data = {
                'version': METRICS_VERSION,
                'tokens': token_count_for_cache,
                'chunks': chunk_count,
                'fingerprint': fingerprint,
                'metrics': metrics,
                # Snapshot the tokenization-affecting filters so the fallback
                # scanner can verify compatibility without re-deriving them.
                'filters': self._path_filters(filters),
            }
            if filters.get('change_fword_to_stupid'):
                cache_data['change_fword_to_stupid'] = True

            if mean_emb is not None:
                np.save(
                    npy_path,
                    {
                        'chunk_embs': chunk_embs,
                        'mean_emb': mean_emb,
                        'chunks': chunks or [],
                    },
                )

            memory_data = cache_data.copy()
            if chunk_embs is not None:
                memory_data['embeddings'] = chunk_embs
                memory_data['mean_emb'] = mean_emb
                memory_data['chunks'] = chunks

            self._store_memory_cache(json_path, memory_data, cache_data['fingerprint'])

            self._write_json_cache(json_path, cache_data)
        except Exception as e:
            print(f"  Cache save error: {e}")

    def _collect_metrics_for_cache(
        self,
        result_dict: Dict,
        enable_ngram_entropy: bool,
        fast_mode: bool,
    ) -> Dict:
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

        return metrics

    def _write_json_cache(self, json_path: Path, cache_data: Dict) -> None:
        with open(json_path, 'wb') as f:
            if JSON_LIB == 'orjson' and orjson is not None:
                f.write(orjson.dumps(cache_data))
            else:
                f.write(json_dumps(cache_data).encode('utf-8'))

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
            keys_to_remove = [
                k for k in self._memory_cache
                if isinstance(k, Path) and text in k.name
            ]
            for k in keys_to_remove:
                del self._memory_cache[k]
        else:
            self._memory_cache.clear()

        print(f"Cleared cache for {text or 'all texts'}")