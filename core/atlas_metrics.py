from atlas_config import *  # noqa
from atlas_config import _install_spacy_model
import atlas_config as cfg
import re
import math
import lzma
import numpy as np
from typing import List, Optional
from collections import Counter
from lexicalrichness import LexicalRichness
from typing import Dict, List, Optional, Tuple


class TextMetricsMixin:

    def _get_sentence_metrics_text(self) -> str:
        text_name = getattr(self, '_current_text_name', None)
        processor = getattr(self, 'processor', None)
        if not text_name or processor is None:
            return ''
        return processor.get_source_text(text_name) or ''

    @staticmethod
    def _sentence_word_lists(text: str) -> List[List[str]]:
        sentences = re.split(r'[.!?]+(?:\s+|$)', text)
        return [
            re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", sentence)
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _estimate_syllables(word: str) -> int:
        normalized = re.sub(r'[^a-z]', '', word.lower())
        if not normalized:
            return 0
        groups = len(re.findall(r'[aeiouy]+', normalized))
        if normalized.endswith('e') and groups > 1:
            groups -= 1
        return max(1, groups)

    def _compute_average_sentence_length(self, tokens: List[str]) -> float:
        sentences = self._sentence_word_lists(self._get_sentence_metrics_text())
        lengths = [len(sentence) for sentence in sentences if sentence]
        return float(np.mean(lengths)) if lengths else 0.0

    def _compute_readability(self, tokens: List[str]) -> float:
        sentences = self._sentence_word_lists(self._get_sentence_metrics_text())
        words = [word for sentence in sentences for word in sentence]
        if not sentences or not words:
            return 0.0
        average_sentence_length = len(words) / len(sentences)
        average_syllables = sum(self._estimate_syllables(word) for word in words) / len(words)
        return float(206.835 - (1.015 * average_sentence_length) - (84.6 * average_syllables))

    def _compute_TTR(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        unique_types = len(set(tokens))
        total_tokens = len(tokens)
        return float(unique_types / total_tokens) if total_tokens > 0 else 0.0

    def _compute_MTLD_score(self, chunks: List[str]) -> float:
        if not chunks:
            return 0.0
        blob = ' '.join(chunks)
        try:
            return LexicalRichness(blob).mtld(threshold=0.72)
        except Exception:
            return 0.0

    @staticmethod
    def _fast_mattr(tokens: List[str], w: int = 25) -> float:
        n = len(tokens)
        if n < w:
            return 0.0
        counts: Dict[str, int] = {}
        distinct = 0
        total = 0
        for i, t in enumerate(tokens):
            c = counts.get(t, 0)
            if c == 0:
                distinct += 1
            counts[t] = c + 1
            if i >= w:
                old = tokens[i - w]
                c = counts[old] - 1
                counts[old] = c
                if c == 0:
                    distinct -= 1
            if i >= w - 1:
                total += distinct
        return total / ((n - w + 1) * w)

    def _compute_MATTR(self, tokens: List[str], lex: Optional[LexicalRichness] = None) -> float:
        if not tokens or len(tokens) < 25:
            return 0.0
        try:
            return float(self._fast_mattr(tokens, w=25))
        except Exception:
            return 0.0

    def _compute_yules_k(self, tokens: List[str], lex: Optional[LexicalRichness] = None) -> float:
        if not tokens or len(tokens) < 10:
            return 0.0
        try:
            if lex is None:
                lex = LexicalRichness(' '.join(tokens))
            return float(lex.yulek)
        except Exception:
            return 0.0

    def _compute_avg_word_length(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        try:
            total_chars = sum(len(token) for token in tokens)
            return float(total_chars / len(tokens)) if tokens else 0.0
        except Exception:
            return 0.0

    def _lzma_full(self, tokens: List[str]) -> Tuple[int, int]:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_lzma_full is None:
            blob = self._get_runtime_text_bytes(tokens)
            if blob:
                self._runtime_cache_lzma_full = (len(lzma.compress(blob)), len(blob))
            else:
                self._runtime_cache_lzma_full = (0, 0)
        return self._runtime_cache_lzma_full

    def _lzma_windows(self, tokens: List[str]) -> List[Tuple[int, int, int]]:
        self._ensure_runtime_cache_for_tokens(tokens)
        if self._runtime_cache_lzma_windows is None:
            out: List[Tuple[int, int, int]] = []
            chunk_size = 5000
            for i in range(0, len(tokens), chunk_size):
                ct = tokens[i:i + chunk_size]
                if len(ct) < 4000:
                    continue
                blob = ' '.join(ct).encode('utf-8')
                if blob:
                    out.append((len(lzma.compress(blob)), len(blob), len(ct)))
            self._runtime_cache_lzma_windows = out
        return self._runtime_cache_lzma_windows

    def _compute_lzma_compression_ratio(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        c, raw = self._lzma_full(tokens)
        return float(c / raw) if raw else 0.0

    def _compute_moving_lzma_compression_ratio(self, tokens: List[str]) -> float:
        if not tokens or len(tokens) < 100:
            return 0.0
        windows = self._lzma_windows(tokens)
        if not windows:
            return 0.0
        return float(np.mean([c / raw for c, raw, _ in windows]))

    def _compute_byte_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        blob = ' '.join(tokens).encode('utf-8')
        if not blob:
            return 0.0
        return float(len(blob))

    def _compute_normalized_compression_ratio(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        c, raw = self._lzma_full(tokens)
        if not raw:
            return 0.0
        word_count = len(tokens)
        if word_count == 0:
            return 0.0
        expected_uncompressed = (word_count / 1_000_000) * 4_500_000
        return float(c / expected_uncompressed)

    def _compute_window_normalized_lzma_cr(self, tokens: List[str]) -> float:
        if not tokens or len(tokens) < 100:
            return 0.0
        windows = self._lzma_windows(tokens)
        if not windows:
            return 0.0
        return float(np.mean([c / ((n / 1_000_000) * 4_500_000) for c, _, n in windows]))

    def _compute_ngram_entropy(self, tokens: List[str], n: int = 2) -> float:
        if not tokens or len(tokens) < n:
            return 0.0
        try:
            vocab_size = len(set(tokens)) + 2
            padded = ['<s>'] * (n - 1) + tokens + ['</s>']
            ngrams = Counter([tuple(padded[i:i + n]) for i in range(len(padded) - n + 1)])
            contexts = Counter([tuple(padded[i:i + n - 1]) for i in range(len(padded) - n + 1)])
            log_prob_sum = 0.0
            total_positions = len(padded) - n + 1
            for i in range(n - 1, len(padded)):
                word = padded[i]
                context = tuple(padded[i - n + 1:i])
                ngram = context + (word,)
                count_ngram = ngrams[ngram]
                count_context = contexts[context]
                prob = (count_ngram + 1) / (count_context + vocab_size)
                log_prob_sum += math.log2(prob)
            cross_entropy = -log_prob_sum / total_positions if total_positions > 0 else 0.0
            return float(cross_entropy)
        except Exception:
            return 0.0

    def _compute_ngram_entropy_2(self, tokens: List[str]) -> float:
        return self._compute_ngram_entropy(tokens, n=2)

    def _compute_ngram_entropy_3(self, tokens: List[str]) -> float:
        return self._compute_ngram_entropy(tokens, n=3)

    def _compute_dev_specialty_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, DEV_SPECIALTY_PATTERNS)

    def _compute_custom_marker_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, CUSTOM_MARKER_PATTERNS)

    def _compute_smore_markers_2_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, SMORE_MARKERS_2_PATTERNS)

    def _compute_smore_markers_3_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, HYPO_PATTERNS)

    def _compute_pronoun_volatility(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        token_to_group = PRONOUN_VOLATILITY_TOKEN_TO_GROUP
        strip_chars = "'\".,!?;:()[]{}"
        transitions = 0
        prev_group: Optional[str] = None
        for token in tokens:
            current_group = token_to_group.get(token.lower().strip(strip_chars))
            if current_group is None:
                continue
            if prev_group is not None and current_group != prev_group:
                transitions += 1
            prev_group = current_group
        return float((transitions / len(tokens)) * 100.0) if tokens else 0.0

    def _compute_word_burstiness(self, tokens: List[str], freq_counter: Optional[Counter] = None) -> float:
        if not tokens or len(tokens) < 10:
            return 0.0
        try:
            word_counts = freq_counter if freq_counter is not None else Counter(tokens)
            sorted_freq = sorted(word_counts.values())
            if len(sorted_freq) < 2:
                return 0.0
            n = len(sorted_freq)
            total_sum = np.sum(sorted_freq)
            numerator = 2 * np.sum(np.arange(1, n + 1) * sorted_freq)
            gini = numerator / (n * total_sum) - (n + 1) / n
            return float(np.clip(gini, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_unique_bigrams(self, tokens: List[str]) -> float:
        if not tokens or len(tokens) < 2:
            return 0.0
        bigrams = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
        return float(len(bigrams))

    def _compute_unique_trigrams(self, tokens: List[str]) -> float:
        if not tokens or len(tokens) < 3:
            return 0.0
        trigrams = {(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)}
        return float(len(trigrams))

    def _compute_moving_unique_bigrams(self, tokens: List[str]) -> float:
        if not tokens or len(tokens) < 10:
            return 0.0
        chunk_size = 5000
        bigram_counts: List[float] = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            if len(chunk_tokens) < 4000:
                continue
            if len(chunk_tokens) >= 2:
                bigrams = {(chunk_tokens[j], chunk_tokens[j + 1]) for j in range(len(chunk_tokens) - 1)}
                bigram_counts.append(float(len(bigrams)))
        return float(np.mean(bigram_counts)) if bigram_counts else 0.0

    def _compute_moving_unique_trigrams(self, tokens: List[str]) -> float:
        if not tokens or len(tokens) < 15:
            return 0.0
        chunk_size = 5000
        trigram_counts: List[float] = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            if len(chunk_tokens) < 4000:
                continue
            if len(chunk_tokens) >= 3:
                trigrams = {(chunk_tokens[j], chunk_tokens[j + 1], chunk_tokens[j + 2]) for j in range(len(chunk_tokens) - 2)}
                trigram_counts.append(float(len(trigrams)))
        return float(np.mean(trigram_counts)) if trigram_counts else 0.0

    def _compute_hapax_ratio(self, tokens: List[str], freq_counter: Optional[Counter] = None) -> float:
        if not tokens or len(tokens) < 10:
            return 0.0
        try:
            word_counts = freq_counter if freq_counter is not None else Counter(tokens)
            hapax_count = sum(1 for count in word_counts.values() if count == 1)
            unique_count = len(word_counts)
            return float(hapax_count / unique_count) if unique_count > 0 else 0.0
        except Exception:
            return 0.0

    def _compute_dis_ratio(self, tokens: List[str], freq_counter: Optional[Counter] = None) -> float:
        if not tokens or len(tokens) < 10:
            return 0.0
        try:
            word_counts = freq_counter if freq_counter is not None else Counter(tokens)
            dis_count = sum(1 for count in word_counts.values() if count == 2)
            unique_count = len(word_counts)
            return float(dis_count / unique_count) if unique_count > 0 else 0.0
        except Exception:
            return 0.0

    def _get_spacy_word_embedding(self, word: str) -> Optional[np.ndarray]:
        if not cfg.spacy_nlp:
            return None
        try:
            lex = cfg.spacy_nlp.vocab[word]
            if lex is not None and getattr(lex, 'has_vector', False):
                return lex.vector.copy()
        except Exception:
            pass
        return None

    def _compute_semantic_disparity(self, tokens: List[str]) -> float:
        if not tokens or len(tokens) < 20:
            return 0.0

        if cfg.spacy_nlp is None:
            try:
                import spacy
                cfg.spacy_nlp = spacy.load("en_core_web_md")
            except (ModuleNotFoundError, ImportError):
                _install_spacy_model("en_core_web_md")
                if cfg.spacy_nlp is None:
                    return 0.0
            except OSError:
                _install_spacy_model("en_core_web_md")
                if cfg.spacy_nlp is None:
                    return 0.0
            except Exception:
                return 0.0

        try:
            unique_words = list(set(tokens))
            if len(unique_words) < 10:
                return 0.0

            embeddings = []
            for word in unique_words:
                if word in spacy_embedding_cache:
                    vec = spacy_embedding_cache[word]
                else:
                    vec = self._get_spacy_word_embedding(word)
                    spacy_embedding_cache[word] = vec
                if vec is not None:
                    embeddings.append(vec)

            if len(embeddings) < 10:
                return 0.0

            emb = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms

            n = emb.shape[0]
            if n < 2:
                return 0.0

            s = emb.sum(axis=0, dtype=np.float64)
            sq = float(np.sum(np.square(emb), dtype=np.float64))
            mean_sim = (float(s @ s) - sq) / (n * (n - 1))
            return float(1.0 - mean_sim)
        except Exception:
            return 0.0

    def _compute_self_mention(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return float(sum(1 for token in tokens if token in FIRST_PERSON_PRONOUNS))

    def _compute_hedge_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        hedge_count = 0
        text = self._get_runtime_text(tokens)
        for pattern in HEDGE_PATTERNS:
            hedge_count += len(pattern.findall(text))
        return float(hedge_count)

    def _compute_subordinate_clause_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return float(sum(1 for token in tokens if token in SUBORDINATORS))

    def _compute_coordinate_clause_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return float(sum(1 for token in tokens if token in COORDINATORS))

    def _compute_latinate_word_ratio(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        latinate_count = sum(1 for token in tokens if len(token) >= 5 and any(token.endswith(suf) for suf in LATINATE_SUFFIXES))
        return float(latinate_count)

    def _compute_discussion_commentary_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, DISCUSSION_COMMENTARY_PATTERNS)

    def _compute_article_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        text = self._get_runtime_text(tokens)
        text_marked = re.sub(r"\bwhat\s+the\b", "__WHAT_THE__", text, flags=re.IGNORECASE)
        deduplicated = re.sub(
            r"\b(the|a|an)\b(\s+(?:the|a|an)\b)+",
            r"\1",
            text_marked,
            flags=re.IGNORECASE
        )
        raw = sum(len(p.findall(deduplicated)) for p in ARTICLE_PATTERNS)
        return float(raw)

    def _compute_article_count_raw(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        text = self._get_runtime_text(tokens)
        deduplicated = re.sub(
            r"\b(the|a|an)\b(\s+(?:the|a|an)\b)+",
            r"\1",
            text,
            flags=re.IGNORECASE
        )
        return float(sum(len(p.findall(deduplicated)) for p in ARTICLE_PATTERNS))

    def _compute_demonstrative_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, DEMONSTRATIVE_PATTERNS)

    def _compute_possessive_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        count = 0
        for tok in tokens:
            lowered = tok.lower()
            if lowered in POSSESSIVE_TOKENS:
                count += 1
            elif lowered.endswith("'s") or lowered.endswith("’s"):
                count += 1
        return float(count)

    def _compute_quantifier_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return float(sum(1 for tok in tokens if tok.lower() in QUANTIFIER_TOKENS))

    def _compute_gerund_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0

        fallback_exclusions = {
            'thing', 'things', 'something', 'anything', 'nothing', 'everything',
            'morning', 'evening', 'afternoon', 'ceiling', 'flooring', 'building',
            'darling', 'sterling', 'during', 'spring', 'string', 'king', 'ring'
        }

        def _fallback_count() -> float:
            count = 0
            for tok in tokens:
                clean = tok.lower().strip("'\".,!?;:()[]{}")
                if len(clean) < 5:
                    continue
                if not clean.isalpha():
                    continue
                if not clean.endswith('ing'):
                    continue
                if clean in fallback_exclusions:
                    continue
                count += 1
            return float(count)

        try:
            stats = self._get_runtime_spacy_stats(tokens)
            if stats is None:
                return _fallback_count()
            return float(stats.get('gerund_count', 0))
        except Exception:
            return _fallback_count()

    def _compute_pronoun_article_ratio(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        text = self._get_runtime_text(tokens)
        pronoun_count = 0
        for pattern in PRONOUN_DEICTIC_PATTERNS:
            pronoun_count += len(pattern.findall(text))
        article_count = self._compute_article_count(tokens)
        return float(pronoun_count / (article_count + 1))

    def _compute_imperative_exclamation_count(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        try:
            text = self._get_runtime_text(tokens)
            exclamation_count = 0
            for pattern in EXCLAMATION_INTERJECTION_PATTERNS:
                exclamation_count += len(pattern.findall(text))
            exclamation_marks = text.count('!')
            stats = self._get_runtime_spacy_stats(tokens)
            imperative_count = int(stats.get('imperative_count', 0)) if stats is not None else 0
            return float(exclamation_count + imperative_count + exclamation_marks)
        except Exception:
            text = self._get_runtime_text(tokens)
            count = 0
            for pattern in EXCLAMATION_INTERJECTION_PATTERNS:
                count += len(pattern.findall(text))
            count += text.count('!')
            return float(count)

    def _compute_deictic_spatial_temporal(self, tokens: List[str]) -> float:
        if not tokens or len(tokens) < 10:
            return 0.0
        text = self._get_runtime_text(tokens)
        deictic_count = 0
        for pattern in DEICTIC_SPATIAL_TEMPORAL_PATTERNS:
            deictic_count += len(pattern.findall(text))
        return float((deictic_count / len(tokens)) * 1000) if tokens else 0.0

    def _compute_elaboration_explanation_ratio(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        text = self._get_runtime_text(tokens)
        elaboration_count = 0
        for pattern in ELABORATION_EXPLANATION_PATTERNS:
            elaboration_count += len(pattern.findall(text))
        return float((elaboration_count / len(tokens)) * 1000) if tokens else 0.0

    def _compute_narration_continuation_ratio(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        text = self._get_runtime_text(tokens)
        narration_count = 0
        for pattern in NARRATION_CONTINUATION_PATTERNS:
            narration_count += len(pattern.findall(text))
        return float((narration_count / len(tokens)) * 1000) if tokens else 0.0

    def _compute_family_exclamations(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, FAMILY_EXCLAMATION_PATTERNS)

    def _compute_vulnerability_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, VULNERABILITY_PATTERNS)

    def _compute_simile_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, SIMILE_PATTERNS)

    def _compute_color_count(self, tokens: List[str]) -> float:
        return self._compute_pattern_count(tokens, COLOR_PATTERNS)