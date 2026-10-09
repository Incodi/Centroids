import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atlas_config import *  # noqa
from atlas_config import (
    TOKEN_REGEX,
    _install_spacy_model,
    extract_video_id,
    hms_to_seconds,
    perplexity_model,
    perplexity_tokenizer,
    sanitize_filename,
    vader_analyzer,
)
import atlas_config as cfg

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextProcessor:

    @staticmethod
    def _is_live(meta: Dict) -> bool:
        return (
            str(meta.get('was_live')) == 'True'
            or str(meta.get('is_live')) == 'True'
        )

    def __init__(
        self,
        base_dir: Path,
        chunk_size: Optional[int] = None,
        min_chunk: Optional[int] = None,
        needs_perplexity_model: bool = False,
        model_name: str = "DistilGPT2",
        enable_spacy: bool = False,
        model_tokenizer=None,
        model_context_window: Optional[int] = None,
    ):
        self.base_dir = base_dir
        self.external_documents: Dict[str, Tuple[str, str]] = {}
        self.model_tokenizer = model_tokenizer
        self.model_context_window = model_context_window
        self._token_length_cache: Dict[str, int] = {}
        self._model = None
        self.embedding_batch_size = 256

        self.embedding_model_name_for_cache: str = ''
        self.video_centroid_cache_dir = self.base_dir / "cache" / "video_centroids"
        self.video_centroid_cache_dir.mkdir(parents=True, exist_ok=True)

        self._encode_fn = None

        inferred_chunk_size = chunk_size
        inferred_min_chunk = min_chunk

        if inferred_chunk_size is None:
            if self.model_context_window and self.model_context_window > 16:
                inferred_chunk_size = max(16, int(self.model_context_window * 0.85))
            else:
                inferred_chunk_size = 256

        if inferred_min_chunk is None:
            inferred_min_chunk = max(16, int(inferred_chunk_size * 0.65))

        self.chunk_size = int(inferred_chunk_size)
        self.min_chunk = int(min(inferred_min_chunk, self.chunk_size))
        self.needs_perplexity_model = needs_perplexity_model
        self.enable_spacy = enable_spacy

        if enable_spacy:
            self._initialize_spacy_model()

        if needs_perplexity_model:
            self._initialize_perplexity_model(model_name)

    def set_external_documents(self, documents: Dict[str, Tuple[str, str]]) -> None:
        self.external_documents = dict(documents)

    def has_external_document(self, text: str) -> bool:
        return text in self.external_documents

    def get_source_text(self, text: str) -> Optional[str]:
        external = self.external_documents.get(text)
        return external[0] if external is not None else None

    def _initialize_spacy_model(self):
        if cfg.spacy_nlp is not None:
            return

        try:
            import spacy
            print("Loading spaCy model...")
            cfg.spacy_nlp = spacy.load('en_core_web_sm')
        except ImportError:
            print("spaCy not installed. Attempting to install...")
            _install_spacy_model()
        except OSError:
            print("spaCy model not found. Attempting to download...")
            _install_spacy_model()
        except Exception as e:
            print(f"Warning: Could not load spaCy model: {e}")
            cfg.spacy_nlp = None

    def _initialize_perplexity_model(self, model_name: str):
        global perplexity_model, perplexity_tokenizer

        if perplexity_model is not None:
            return

        print("Loading perplexity model...")
        perplexity_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
        perplexity_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
        )
        perplexity_model.eval()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        perplexity_model.to(device)
        print(f"Perplexity model loaded on {device}")

    def tokenize(self, text: str) -> List[str]:
        text = TOKEN_REGEX.sub(' ', text)
        return text.lower().split()

    def _video_centroid_signature(self, txt_path: Path) -> Dict[str, Any]:
        try:
            stat = txt_path.stat()
            file_sig = [int(stat.st_mtime), int(stat.st_size)]
        except OSError:
            file_sig = None

        return {
            'model': self.embedding_model_name_for_cache or '',
            'chunk_size': int(self.chunk_size),
            'min_chunk': int(self.min_chunk),
            'file_sig': file_sig,
        }

    def _video_centroid_cache_paths(self, video_id: str) -> Tuple[Path, Path]:
        safe = sanitize_filename(video_id) or 'unknown'
        return (
            self.video_centroid_cache_dir / f"{safe}.npy",
            self.video_centroid_cache_dir / f"{safe}.json",
        )

    def _load_video_centroid_from_cache(
        self, video_id: str, txt_path: Path
    ) -> Optional[np.ndarray]:
        npy_path, meta_path = self._video_centroid_cache_paths(video_id)

        if not npy_path.exists() or not meta_path.exists():
            return None

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                saved_sig = json.load(f)

            if saved_sig != self._video_centroid_signature(txt_path):
                return None

            emb = np.load(npy_path, allow_pickle=False)
            return emb.astype(np.float32, copy=False)
        except Exception:
            return None

    def _save_video_centroid_to_cache(
        self, video_id: str, txt_path: Path, embedding: np.ndarray
    ) -> None:
        npy_path, meta_path = self._video_centroid_cache_paths(video_id)

        try:
            tmp_npy = npy_path.with_name(npy_path.name + '.tmp')
            with open(tmp_npy, 'wb') as f:
                np.save(f, embedding.astype(np.float32, copy=False))
            tmp_npy.replace(npy_path)

            tmp_meta = meta_path.with_name(meta_path.name + '.tmp')
            with open(tmp_meta, 'w', encoding='utf-8') as f:
                json.dump(self._video_centroid_signature(txt_path), f)
            tmp_meta.replace(meta_path)
        except Exception as e:
            if not getattr(self, '_warned_cache_write', False):
                print(f"    Warning: video centroid cache write failed: {e}")
                self._warned_cache_write = True

    def _get_filtered_files(
        self,
        text: str,
        filters: Dict,
        ignore_ids: Optional[set] = None,
    ) -> Optional[List[Tuple[str, Path]]]:
        if ignore_ids is None:
            ignore_ids = set()

        safe_text = sanitize_filename(text)
        txt_dir = self.base_dir / "data/input" / safe_text / "txt_files"
        if not txt_dir.exists():
            return None

        metadata = self.load_metadata(text)
        min_dur = hms_to_seconds(filters.get('duration_from', ''))
        max_dur = hms_to_seconds(filters.get('duration_to', ''))
        date_filter_active = bool(
            filters.get('date_from') or filters.get('date_to')
        )

        files_with_dates: List[Tuple[str, Path]] = []

        for f in txt_dir.glob("*.txt"):
            vid = extract_video_id(f.name)
            if vid in ignore_ids:
                continue

            meta = metadata.get(vid, {})

            upload = meta.get('upload_date')
            if date_filter_active and not upload:
                continue
            upload = upload or '99999999'

            if filters.get('date_from') and upload < filters['date_from']:
                continue
            if filters.get('date_to') and upload > filters['date_to']:
                continue
            if min_dur and meta.get('duration', 0) < min_dur:
                continue
            if max_dur and meta.get('duration', float('inf')) > max_dur:
                continue
            if filters.get('exclude_live') and self._is_live(meta):
                continue

            files_with_dates.append((upload, f))

        if not files_with_dates:
            return None

        files_with_dates.sort()
        return files_with_dates

    def load_video_centroids(
        self,
        text: str,
        n_videos: int,
        filters: Dict,
        ignore_ids: Optional[set] = None,
        max_accumulated_tokens: Optional[int] = None,
        use_video_cache: bool = True,
        precomputed_files: Optional[List[Tuple[str, Path]]] = None,
        need_embeddings: bool = True,
        need_stat_corpus: bool = True,
        need_centroid_tokens: bool = True,
    ) -> Optional[Tuple[List[str], List[np.ndarray], List[str], List[str], float]]:
        if ignore_ids is None:
            ignore_ids = set()

        if precomputed_files is not None:
            all_files = precomputed_files
        else:
            all_files = self._get_filtered_files(text, filters, ignore_ids)

        if not all_files:
            print(f"  Skipped: {text} (no txt_files directory or no files passed filters)")
            return None

        target_stat_tokens = (
            int(max_accumulated_tokens)
            if (max_accumulated_tokens and max_accumulated_tokens > 0)
            else None
        )

        video_ids: List[str] = []
        video_embeddings: List[np.ndarray] = []
        centroid_tokens_by_path: Dict[Path, List[str]] = {}
        centroid_files: List[Tuple[str, Path]] = []
        retain_tokens_pass1 = bool(need_stat_corpus or need_centroid_tokens)

        if n_videos > 0 and (need_embeddings or need_centroid_tokens):
            centroid_files = all_files[-n_videos:]

        if need_embeddings:
            model = getattr(self, '_model', None)
            encode_fn = getattr(self, '_encode_fn', None)

            if model is None or encode_fn is None:
                print(f"    Warning: No embedding model available")
                return None

            if centroid_files:
                (
                    video_ids,
                    video_embeddings,
                    centroid_tokens_by_path,
                ) = self._load_video_centroids_pass(
                    centroid_files,
                    use_video_cache=use_video_cache,
                    retain_tokens=retain_tokens_pass1,
                )

        elif need_centroid_tokens and centroid_files:
            for _, filepath in centroid_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    tokens = self.tokenize(text_content)
                    del text_content
                except Exception as e:
                    print(f"    Warning: {filepath.name}: {e}")
                    continue

                if tokens:
                    centroid_tokens_by_path[filepath] = tokens

        centroid_tokens: List[str] = []
        centroid_word_counts: List[int] = []

        if need_centroid_tokens and centroid_files:
            for _, filepath in centroid_files:
                tkns = centroid_tokens_by_path.get(filepath)
                if tkns:
                    centroid_tokens.extend(tkns)
                    centroid_word_counts.append(len(tkns))

        avg_words_per_video = (
            float(sum(centroid_word_counts) / len(centroid_word_counts))
            if centroid_word_counts
            else 0.0
        )

        all_tokens: List[str] = []
        if need_stat_corpus:
            all_tokens = self._load_stat_corpus_pass(
                all_files,
                target_stat_tokens,
                reused_tokens_by_path=centroid_tokens_by_path,
            )

        centroid_tokens_by_path.clear()

        if not need_embeddings:
            return [], [], all_tokens, centroid_tokens, avg_words_per_video

        if not video_ids:
            return None

        return (
            video_ids,
            video_embeddings,
            all_tokens,
            centroid_tokens,
            avg_words_per_video,
        )

    def _batch_token_lengths(self, tokens: List[str]) -> None:
        if self.model_tokenizer is None or not tokens:
            return

        unique_tokens = set(tokens)
        if not unique_tokens:
            return

        cache = self._token_length_cache
        missing = [tok for tok in unique_tokens if tok not in cache]
        if not missing:
            return

        batch_size = 1024

        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]

            try:
                encoded = self.model_tokenizer(
                    batch,
                    add_special_tokens=False,
                    padding=False,
                )
                input_ids = (
                    encoded.get('input_ids')
                    if isinstance(encoded, dict)
                    else None
                )
                if input_ids is None:
                    input_ids = getattr(encoded, 'input_ids', None)

                if input_ids and not isinstance(input_ids[0], (list, tuple)):
                    input_ids = [input_ids]

                if not input_ids or len(input_ids) != len(batch):
                    raise ValueError("unexpected batch tokenizer output shape")

                for tok, ids in zip(batch, input_ids):
                    cache[tok] = max(1, len(ids))
            except Exception:
                for tok in batch:
                    try:
                        enc = self.model_tokenizer(tok, add_special_tokens=False)
                        ids = enc.get('input_ids', [])
                        if ids and isinstance(ids[0], (list, tuple)):
                            ids = ids[0]
                        cache[tok] = max(1, len(ids))
                    except Exception:
                        cache[tok] = 1

    def _load_video_centroids_pass(
        self,
        centroid_files: List[Tuple[str, Path]],
        use_video_cache: bool,
        retain_tokens: bool,
    ) -> Tuple[List[str], List[np.ndarray], Dict[Path, List[str]]]:
        encode_fn = getattr(self, '_encode_fn', None)
        if encode_fn is None:
            print(f"    Warning: No embedding model available")
            return [], [], {}

        video_ids: List[str] = []
        video_embeddings: List[Optional[np.ndarray]] = []
        tokens_by_path: Dict[Path, List[str]] = {}
        pending_batches: List[Tuple[int, List[str], Path]] = []
        cache_hits = 0
        cache_misses = 0

        for _, filepath in reversed(centroid_files):
            vid = extract_video_id(filepath.name)

            cached_emb: Optional[np.ndarray] = None
            if use_video_cache:
                cached_emb = self._load_video_centroid_from_cache(vid, filepath)

            if cached_emb is not None and not retain_tokens:
                cache_hits += 1
                video_ids.append(vid)
                video_embeddings.append(cached_emb)
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                tokens = self.tokenize(text_content)
                del text_content
            except Exception as e:
                print(f"    Warning: {filepath.name}: {e}")
                continue

            if not tokens:
                if cached_emb is not None:
                    cache_hits += 1
                    video_ids.append(vid)
                    video_embeddings.append(cached_emb)
                continue

            if retain_tokens:
                tokens_by_path[filepath] = tokens

            if cached_emb is not None:
                cache_hits += 1
                video_ids.append(vid)
                video_embeddings.append(cached_emb)
                continue

            cache_misses += 1
            chunks = self.chunk(tokens)
            if chunks:
                position = len(video_ids)
                video_ids.append(vid)
                video_embeddings.append(None)
                pending_batches.append((position, chunks, filepath))
            del chunks

        if pending_batches:
            MAX_CHUNKS_PER_GROUP = 50_000
            groups: List[List[Tuple[int, List[str], Path]]] = []
            current_group: List[Tuple[int, List[str], Path]] = []
            current_chunks = 0

            for entry in pending_batches:
                entry_chunks = len(entry[1])
                if current_group and current_chunks + entry_chunks > MAX_CHUNKS_PER_GROUP:
                    groups.append(current_group)
                    current_group = []
                    current_chunks = 0
                current_group.append(entry)
                current_chunks += entry_chunks

            if current_group:
                groups.append(current_group)

            for group in groups:
                all_chunks: List[str] = []
                boundaries: List[Tuple[int, int, int, Path]] = []

                for position, chunks, filepath in group:
                    start = len(all_chunks)
                    all_chunks.extend(chunks)
                    boundaries.append((position, start, len(all_chunks), filepath))

                try:
                    all_embs = np.asarray(encode_fn(all_chunks), dtype=np.float32)

                    for position, start, end, filepath in boundaries:
                        if end <= start:
                            continue
                        video_emb = np.mean(all_embs[start:end], axis=0).astype(np.float32)
                        video_embeddings[position] = video_emb
                        if use_video_cache:
                            self._save_video_centroid_to_cache(
                                video_ids[position], filepath, video_emb
                            )

                    del all_embs
                except RuntimeError as e:
                    print(
                        f"    Warning: batched encode failed ({e}); "
                        f"falling back to per-video"
                    )
                    for position, chunks, filepath in group:
                        try:
                            chunk_embs = np.asarray(encode_fn(chunks), dtype=np.float32)
                            video_emb = np.mean(chunk_embs, axis=0).astype(np.float32)
                            video_embeddings[position] = video_emb
                            if use_video_cache:
                                self._save_video_centroid_to_cache(
                                    video_ids[position], filepath, video_emb
                                )
                            del chunk_embs
                        except Exception as inner_e:
                            print(
                                f"    Warning: per-video encode failed for "
                                f"{filepath.name}: {inner_e}"
                            )

        if cache_hits or cache_misses:
            print(f"    Video centroids: {cache_hits} cached, {cache_misses} computed")

        valid_pairs = [
            (vid, emb)
            for vid, emb in zip(video_ids, video_embeddings)
            if emb is not None
        ]

        if not valid_pairs:
            return [], [], tokens_by_path

        return (
            [vid for vid, _ in valid_pairs],
            [emb for _, emb in valid_pairs],
            tokens_by_path,
        )

    def _load_stat_corpus_pass(
        self,
        all_files: List[Tuple[str, Path]],
        target_stat_tokens: Optional[int],
        reused_tokens_by_path: Optional[Dict[Path, List[str]]] = None,
    ) -> List[str]:
        if reused_tokens_by_path is None:
            reused_tokens_by_path = {}

        stat_chunks_newest_first: List[List[str]] = []
        stat_token_total = 0

        for _, filepath in reversed(all_files):
            if target_stat_tokens is not None and stat_token_total >= target_stat_tokens:
                break

            tokens = reused_tokens_by_path.pop(filepath, None)

            if tokens is None:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    tokens = self.tokenize(text_content)
                    del text_content
                except Exception as e:
                    print(f"    Warning: {filepath.name}: {e}")
                    continue

            if not tokens:
                continue

            stat_chunks_newest_first.append(tokens)
            stat_token_total += len(tokens)

        stat_chunks_newest_first.reverse()
        all_tokens: List[str] = []
        for chunk in stat_chunks_newest_first:
            all_tokens.extend(chunk)
        stat_chunks_newest_first.clear()

        if target_stat_tokens is not None and len(all_tokens) > target_stat_tokens:
            all_tokens = all_tokens[-target_stat_tokens:]

        return all_tokens

    def chunk(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []

        if self.model_tokenizer is None:
            return [
                ' '.join(tokens[i:i + self.chunk_size])
                for i in range(0, len(tokens), self.chunk_size)
                if len(tokens[i:i + self.chunk_size]) > 0
            ]

        max_model_tokens = max(8, int(self.chunk_size))
        min_model_tokens = max(1, int(self.min_chunk))

        chunks: List[str] = []
        current_words: List[str] = []
        current_piece_count = 0

        self._batch_token_lengths(tokens)
        token_length_cache = self._token_length_cache

        for token in tokens:
            token_piece_len = token_length_cache.get(token)
            if token_piece_len is None:
                token_piece_len = 1
                token_length_cache[token] = 1

            if token_piece_len > max_model_tokens:
                if current_words:
                    chunks.append(' '.join(current_words))
                current_words = []
                current_piece_count = 0
                if token_piece_len >= 1:
                    chunks.append(token)
                continue

            if current_words and (current_piece_count + token_piece_len > max_model_tokens):
                chunks.append(' '.join(current_words))
                current_words = [token]
                current_piece_count = token_piece_len
            else:
                current_words.append(token)
                current_piece_count += token_piece_len

        if current_words:
            chunks.append(' '.join(current_words))

        return chunks

    def load_metadata(self, text: str) -> Dict:
        safe_text = sanitize_filename(text)
        meta_file = self.base_dir / "data/input" / safe_text / "metadata.json"

        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                return {v['id']: v for v in json.load(f) if 'id' in v}
        except Exception:
            return {}

    def calculate_perplexity(
        self,
        text_or_chunks: Any,
        max_length: int = 1024,
        sample_chunks: int = 6,
    ) -> Optional[float]:
        if not self.needs_perplexity_model:
            return None

        if isinstance(text_or_chunks, list):
            chunks = [str(chunk) for chunk in text_or_chunks if chunk]
            if not chunks:
                return float('inf')

            if len(chunks) <= sample_chunks:
                selected_chunks = chunks
            else:
                selected_chunks = []
                last_idx = len(chunks) - 1
                for sample_idx in range(sample_chunks):
                    idx = int(round(sample_idx * last_idx / max(1, sample_chunks - 1)))
                    selected_chunks.append(chunks[idx])

            text = ' '.join(selected_chunks)
        else:
            text = str(text_or_chunks or '')

        if len(text) > 24000:
            text = text[:24000]

        if not text or not text.strip():
            return float('inf')

        try:
            encodings = perplexity_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encodings.input_ids.to(perplexity_model.device)

            with torch.no_grad():
                outputs = perplexity_model(input_ids, labels=input_ids)
                loss = outputs.loss

            perplexity = torch.exp(loss).item()
            return perplexity
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')

    def compute_vader_sentiment(self, chunks: List[str]) -> Tuple[float, float, float]:
        if not chunks:
            return 0.0, 0.0, 0.0

        sentiments = []
        positive_count = 0
        negative_count = 0

        for chunk in chunks:
            scores = vader_analyzer.polarity_scores(chunk)
            compound = scores['compound']
            sentiments.append(compound)

            if compound > 0.05:
                positive_count += 1
            elif compound < -0.05:
                negative_count += 1

        compound_mean = float(np.mean(sentiments)) if sentiments else 0.0

        total_polar = positive_count + negative_count
        positivity_ratio = (positive_count / total_polar) if total_polar > 0 else 0.5

        sentiment_volatility = float(np.std(sentiments)) if sentiments else 0.0

        return compound_mean, positivity_ratio, sentiment_volatility

    def get_tokens(
        self,
        text: str,
        filters: Dict,
        ignore_ids: Optional[set] = None,
    ) -> Optional[Tuple[List[str], Path, int, List[str]]]:
        if ignore_ids is None:
            ignore_ids = set()

        external = self.external_documents.get(text)
        if external is not None:
            return self._get_tokens_from_external(text, filters, external)

        safe_text = sanitize_filename(text)
        txt_dir = self.base_dir / "data/input" / safe_text / "txt_files"
        if not txt_dir.exists():
            print(f"  Skipped: {text} (no txt_files directory)")
            return None

        txt_files = list(txt_dir.glob("*.txt"))
        if not txt_files:
            print(f"  Skipped: {text} (no transcript files)")
            return None

        files_with_dates = self._filter_txt_files(
            txt_files, text, filters, ignore_ids
        )
        if not files_with_dates:
            print(f"  Skipped: {text} (no files after video-level filters)")
            return None

        files_with_dates.sort()

        return self._collect_tokens_from_files(
            text, txt_dir, files_with_dates, filters
        )

    def _get_tokens_from_external(
        self,
        text: str,
        filters: Dict,
        external: Tuple[str, str],
    ) -> Optional[Tuple[List[str], Path, int, List[str]]]:
        text_content, source_id = external
        file_tokens = self.tokenize(text_content)

        min_tokens_per_file = filters.get('min_tokens_per_file', 0)
        if len(file_tokens) < min_tokens_per_file:
            print(f"  Skipped: {text} (below {min_tokens_per_file} tokens)")
            return None

        per_video_token_limit = filters.get('per_video_token_limit')
        if per_video_token_limit and per_video_token_limit > 0:
            file_tokens = file_tokens[:per_video_token_limit]

        text_token_limit = filters.get('text_token_limit')
        if text_token_limit and text_token_limit > 0:
            file_tokens = file_tokens[:text_token_limit]

        token_limit = filters.get('token_limit')
        if token_limit and token_limit > 0:
            file_tokens = file_tokens[-token_limit:]

        if not file_tokens:
            print(f"  Skipped: {text} (no tokens after filtering)")
            return None

        return file_tokens, self.base_dir, 1, [source_id]

    def _filter_txt_files(
        self,
        txt_files: List[Path],
        text: str,
        filters: Dict,
        ignore_ids: set,
    ) -> List[Tuple[str, Path]]:
        metadata = self.load_metadata(text)
        min_dur = hms_to_seconds(filters.get('duration_from', ''))
        max_dur = hms_to_seconds(filters.get('duration_to', ''))
        date_filter_active = bool(
            filters.get('date_from') or filters.get('date_to')
        )

        files_with_dates: List[Tuple[str, Path]] = []
        files_ignored = 0

        for f in txt_files:
            vid_id = extract_video_id(f.name)

            if vid_id in ignore_ids:
                files_ignored += 1
                continue

            meta = metadata.get(vid_id, {})

            upload = meta.get('upload_date')
            if date_filter_active and not upload:
                continue
            upload = upload or '99999999'

            if filters.get('date_from') and upload < filters['date_from']:
                continue
            if filters.get('date_to') and upload > filters['date_to']:
                continue
            if min_dur and meta.get('duration', 0) < min_dur:
                continue
            if max_dur and meta.get('duration', float('inf')) > max_dur:
                continue
            if filters.get('exclude_live') and self._is_live(meta):
                continue

            files_with_dates.append((upload, f))

        if files_ignored > 0:
            print(f"    Ignored {files_ignored} file(s) in ignore list")

        return files_with_dates

    def _collect_tokens_from_files(
        self,
        text: str,
        txt_dir: Path,
        files_with_dates: List[Tuple[str, Path]],
        filters: Dict,
    ) -> Optional[Tuple[List[str], Path, int, List[str]]]:
        min_tokens_per_file = filters.get('min_tokens_per_file', 0)
        per_video_token_limit = filters.get('per_video_token_limit')
        text_token_limit = filters.get('text_token_limit')

        all_tokens: List[str] = []
        video_token_spans: List[Tuple[str, int, int]] = []
        files_processed = 0
        files_skipped = 0
        text_limit_applied = False

        for _, filepath in files_with_dates:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                file_tokens = self.tokenize(text_content)

                if (
                    per_video_token_limit
                    and per_video_token_limit > 0
                    and len(file_tokens) > per_video_token_limit
                ):
                    file_tokens = file_tokens[:per_video_token_limit]

                if len(file_tokens) < min_tokens_per_file:
                    files_skipped += 1
                    continue

                video_id = extract_video_id(filepath.name)
                start_idx = len(all_tokens)
                all_tokens.extend(file_tokens)
                end_idx = len(all_tokens)
                video_token_spans.append((video_id, start_idx, end_idx))
                files_processed += 1

                if (
                    text_token_limit
                    and text_token_limit > 0
                    and len(all_tokens) >= text_token_limit
                ):
                    all_tokens = all_tokens[:text_token_limit]
                    text_limit_applied = True
                    break
            except Exception as e:
                print(f"    Warning: {filepath.name}: {e}")

        if files_skipped > 0:
            print(
                f"    Skipped {files_skipped} file(s) below "
                f"{min_tokens_per_file} tokens"
            )

        if not all_tokens:
            print(f"  Skipped: {text} (no tokens after per-file filtering)")
            return None

        total_tokens_before_limit = len(all_tokens)
        selected_start_idx = 0
        selected_end_idx = total_tokens_before_limit

        token_limit = filters.get('token_limit')

        if (
            text_token_limit
            and text_token_limit > 0
            and len(all_tokens) > text_token_limit
        ):
            all_tokens = all_tokens[:text_token_limit]
            text_limit_applied = True
            selected_end_idx = len(all_tokens)
        elif token_limit and token_limit > 0 and len(all_tokens) > token_limit:
            selected_start_idx = total_tokens_before_limit - token_limit
            all_tokens = all_tokens[-token_limit:]
            print(f"    Applied token limit: kept newest {token_limit:,} tokens")

        selected_video_ids = [
            video_id
            for video_id, start_idx, end_idx in video_token_spans
            if end_idx > selected_start_idx and start_idx < selected_end_idx
        ]

        if text_limit_applied:
            print(
                f"    Applied text token limit: kept first "
                f"{text_token_limit:,} tokens"
            )

        return all_tokens, txt_dir, files_processed, selected_video_ids