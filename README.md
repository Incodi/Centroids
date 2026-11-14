
# Centroids

An advanced text analysis and visualization tool for comparing linguistic features across large text corpora. Built with YouTube channel analysis in mind, but supports NLTK corpora and custom text directories.

Check out the [wiki!](https://github.com/Incodi/Centroids/wiki)

## Table of Contents

- [Using the demo](#using-the-demo)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data layout](#data-layout)
  - [Example channel metadata](#example-channel-metadata)
  - [Converting downloaded subtitles](#converting-downloaded-subtitles)
  - [Subtitle Downloader GUI](#subtitle-downloader-gui)
- [CLI Usage](#cli-usage)
  - [Basic Syntax](#basic-syntax)
  - [Data Source Options](#data-source-options)
  - [Filtering Options (YouTube)](#filtering-options-youtube)
  - [Token Limits](#token-limits)
  - [Centroid Modes](#centroid-modes)
  - [Metric Computation Options](#metric-computation-options)
  - [Clustering & Dimensionality Reduction](#clustering--dimensionality-reduction)
  - [Embedding Model](#embedding-model)
  - [Output & Caching](#output--caching)
  - [Advanced Features](#advanced-features)
- [Examples](#examples)
- [Major Linguistic Features](#major-linguistic-features)
  - [MALLET Topic Modeling](#mallet-topic-modeling)
  - [Tomotopy Topic Modeling](#tomotopy-topic-modeling)
  - [Burrows Delta Stylistic Analysis](#burrows-delta-stylistic-analysis)
  - [SBERT (Sentence-BERT) Semantic Embeddings](#sbert-sentence-bert-semantic-embeddings)
  - [Focus Text Display](#focus-text-display)
- [Metric Categories](#metric-categories)
- [Output Files](#output-files)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [TODOs](#todos)

## Using the demo

First check the doctor file to check if you have everything installed:

```bash
python core/atlas_doctor.py
# or if you have python 3:
python3 core/atlas_doctor.py
```

Then when everything is installed try the demo:

```bash
python core/atlas_demo.py
# or if you have python 3:
python3 core/atlas_demo.py
```

## Key Features

- **Interactive HTML Visualization**: Plotly-based interactive scatter plots for exploring text similarities.
- **Extensive Linguistic Metrics**: 120+ metrics including sentiment analysis, lexical diversity, discourse markers, and stylometric features.
- **Advanced Topic Modeling**: MALLET LDA integration for topic discovery.
- **Tomotopy Topic Modeling**: Lightweight, pure‑Python LDA alternative to MALLET with no Java dependency.
- **GUI Subtitle Downloader**: Built‑in Tkinter interface for downloading YouTube subtitles and managing channel data.
- **Stylistic Analysis**: Burrows Delta method for function‑word based authorship/style analysis.
- **Semantic Embeddings**: SBERT (Sentence-BERT) for semantic similarity mapping.
- **Focus Text Display**: Interactive similarity mapping to highlight relationships between texts.
- **Smart Caching**: Multi-level caching system to avoid redundant computations.
- **Centroid Modes**: Two aggregation strategies – by video count or by word count.
- **Clustering**: HDBSCAN and Leiden algorithms with parameter sweeps.
- **UMAP Dimensionality Reduction**: Configurable parameters for visualization.

## Installation

```bash
pip install sentence-transformers scikit-learn umap-learn plotly torch transformers vaderSentiment lexicalrichness orjson
pip install spacy && python -m spacy download en_core_web_sm   # Optional, for lexical_density metric
pip install pyenchant                                         # Optional, for OOV word metrics
pip install python-igraph leidenalg                           # Optional, for Leiden clustering
pip install tomotopy                                          # Optional, for Tomotopy topic modeling
```

For MALLET topic modeling, download and install [MALLET](http://mallet.cs.umass.edu/). Set the `MALLET_HOME` environment variable or ensure `mallet` binary is in your PATH.

## Data layout

For YouTube-style input, the tool expects one directory per "text" (channel, book, document), containing a `txt_files/` subfolder of `.txt` files (one per video or chapter):

```
data/input/
  my_channel/
    txt_files/
      video1 [Video ID].txt
      video2 [Video ID].txt
      ...
    metadata.json          (optional — YouTube-style fields)
    tag.txt               (optional — short text shown under the channel name in hover)
    hover_panel.txt       (optional — 2-4 hex colors for a top-to-bottom hover gradient, decorate hover for favorite channels)
    hover_text.txt        (optional — 2-4 hex colors: text outline, text, then optional accents)
    archive.txt
  another_channel/
    txt_files/
      ...
```

### Example channel metadata

The format was created and modified after downloading from yt-dlp; Youtube Data API has a different format. For now, the filters look for this format.

```json
[
  {
    "id": "dQw4w9WgXcQ",
    "title": "I put my favorite channels into Centroids!",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "upload_date": "20250905",
    "duration": 1917,
    "view_count": 12474,
    "like_count": 897,
    "comment_count": 65,
    "was_live": false,
    "is_live": false,
    "channel_name": "thename",
    "channel_id": "UCjs2omu0U4AO0kxhGIzfuni",
    "subscriber_count": 120,
    "category": "Science & Technology"
  },
  {
    "id": "dQw4w9WgXcR",
    "title": "What Youtuber is the most mathematically similar Youtuber to MrBeast?",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcR",
    "upload_date": "20250902",
    "duration": 1147,
    "view_count": 14395,
    "like_count": 1100,
    "comment_count": 87,
    "was_live": false,
    "is_live": false,
    "channel_name": "thename",
    "channel_id": "UCjs2omu0U4AO0kxhGIzfuni",
    "subscriber_count": 120,
    "category": "Science & Technology"
  }
]
```

Each `.txt` file is treated as one document. In video-centroid mode, the N most recent documents become the text's centroid; in word-centroid mode, the last N words across all documents do. For non-YouTube use, `--txt-directory` and `--nltk-corpus` accept alternative sources without this layout.

### Converting downloaded subtitles

The subtitle downloader writes WebVTT files under each channel's `vtt_files/` directory. Convert all available channels to VTX and TXT files with:

```bash
python3 core/batch_converter.py
```

To process selected channels or remove source VTT files only after verified conversion:

```bash
python3 core/batch_converter.py --channels channel_one channel_two
python3 core/batch_converter.py --delete-vtt
```

### Subtitle Downloader GUI

Centroids includes a built‑in Tkinter GUI for downloading YouTube subtitles and organising channel data. Launch it with:

```bash
python3 core/atlas_downloader.py
```

The GUI provides:

- A channel‑list manager (add, remove, reorder channels).
- Configurable subtitle language (`en` by default) and download rate limits.
- Automatic conversion of downloaded VTT files to VTX and TXT (via `batch_converter.py`).
- Integration with the `yt-dlp` downloader, including cookie and impersonation support.
- Progress logging and error handling.

Use the GUI when you need an interactive way to fetch subtitles before running the analysis pipeline.

## CLI Usage

### Basic Syntax

```bash
python3 core/atlas.py [text_folder_names...] [OPTIONS]
```

### Data Source Options

| Option | Description |
|--------|-------------|
| `texts` | Text folder names to process (default: YouTube channels). |
| `--nltk-corpus NAME` | Compare texts from an NLTK corpus (e.g., `gutenberg`, `reuters`). |
| `--txt-directory PATH` | Compare `.txt` files from a directory (includes nested folders). |
| `--txt-depth N` | Max subdirectory depth for `--txt-directory` (default: 3). |

### Filtering Options (YouTube)

| Option | Description |
|--------|-------------|
| `--date-from YYYYMMDD` | Include only videos from this date onwards. |
| `--date-to YYYYMMDD` | Include only videos up to this date. |
| `--duration-from HH:MM:SS` | Minimum video duration. |
| `--duration-to HH:MM:SS` | Maximum video duration. |
| `--exclude-live` | Exclude livestream videos. |
| `--ignore-urls PATH` | Path to file with YouTube URLs/IDs to ignore (one per line). |

### Token Limits

| Option | Description |
|--------|-------------|
| `--token-limit N` | Use only the newest N tokens per text. |
| `--text-token-limit N` | Use only the first N tokens per text. |
| `--per-video-token-limit N` | Use only the first N tokens per video. |
| `--min-tokens-per-file N` | Skip files with fewer than N tokens. |
| `--min-tokens-total N` | Skip texts with fewer than N total tokens (alias: `--min-words`). |
| `--embeddings-only-low-token` | Generate embeddings for low‑token texts, skip metrics. |

### Centroid Modes

| Option | Description |
|--------|-------------|
| `--centroid-mode {video,word}` | `video`: aggregate embeddings from the N most recent videos; `word`: aggregate from the last N words in the corpus. |
| `--centroid-videos N` | Number of videos for video centroid (default: 10). |
| `--centroid-words N` | Number of words for word centroid (default: 1,000,000). |
| `--stat-word-count N` | Words for statistical metrics with video centroids. |

### Metric Computation Options

| Option | Description |
|--------|-------------|
| `--no-metrics` | Skip all metric calculations (embeddings/clustering only). |
| `--fast-mode` | Skip expensive metrics (fastest mode). |
| `--word-counts-only` | Compute only count‑style and regex‑pattern metrics. |
| `--calculate-perplexity` | Enable GPT‑2 perplexity calculation (slower). |
| `--compute-ngram-entropy` | Enable n‑gram cross‑entropy metrics (slower). |
| `--compute-fighting-words` | Enable fighting‑words peak/floor metrics. |

### Clustering & Dimensionality Reduction

| Option | Description |
|--------|-------------|
| `--cluster-method {hdbscan,leiden}` | Clustering algorithm (default: hdbscan). |
| `--leiden-resolution-min FLOAT` | Minimum Leiden resolution (default: 1.15). |
| `--leiden-resolution-max FLOAT` | Maximum Leiden resolution (default: 3.0). |
| `--leiden-k-neighbors N` | k‑NN graph neighbors for Leiden (default: 20). |
| `--no-leiden-whiten` | Disable PCA whitening before k‑NN (use for homogeneous datasets). |
| `--umap-neighbors N` | UMAP neighborhood size (auto: 5 for ≤30 items, else 50). |
| `--umap-min-dist FLOAT` | Minimum distance between points (default: 0.1; use 0 for tight groups). |
| `--umap-epochs N` | UMAP optimization epochs (default: 50). |
| `--cluster-variants N` | Number of clustering variants (default: 11 for ≤2000 items, else 1). |

### Embedding Model

| Option | Description |
|--------|-------------|
| `--embedding-model NAME` | SBERT model name (default: `BAAI/bge-small-en-v1.5`). Examples: `all-mpnet-base-v2`, `thenlper/gte-small`, `Alibaba-NLP/gte-modernbert-base`. |
| `--embedding-chunk-size N` | Override chunk size in model tokens. |
| `--embedding-min-chunk N` | Override minimum chunk size. |
| `--embedding-batch-size N` | Override batch size for encoding. |
| `--trained-model-path PATH` | Path to custom fine‑tuned SBERT model. |

### Output & Caching

| Option | Description |
|--------|-------------|
| `--output-csv PATH` | Output CSV file path (default: `data/input/text_similarities.csv`). |
| `--output-html PATH` | Output HTML visualization (default: `data/input/SBERTClustersHD.html`). |
| `--no-cache` | Disable embedding cache. |
| `--force-latest-cache` | Use newest cache file without freshness checks. |
| `--clear-cache [TEXT]` | Clear cache for specific text or all texts. |
| `--annotations` | Enable annotation loading and click‑to‑view notes. |

Each channel directory may also contain an optional `tag.txt`. Its short contents appear directly below the channel name in the visualization hover text. For custom hover styling, add `hover_panel.txt` and/or `hover_text.txt` to a channel directory. Put one `#RGB` or `#RRGGBB` colour per line (2‑4 colours). Panel colours blend from top to bottom in listed order. The first hover‑text colour is its outline and the second is the text colour. The panel border uses the point's current cluster or metric colour, so it remains identifiable when switching metrics. `hover_gradient.txt` is also accepted as a legacy alias for `hover_panel.txt`. Invalid palette files are ignored with a warning.

### Advanced Features

| Option | Description |
|--------|-------------|
| `--mallet-topics N` | Train MALLET LDA model with N topics (0 disables). Set `MALLET_MEMORY` env var (e.g., `12g`) for large corpora. |
| `--tomotopy-topics N` | Train Tomotopy LDA model with N topics (0 disables). |
| `--tomotopy-iterations N` | Number of Tomotopy training iterations (default: 1000). |
| `--tomotopy-threads N` | Number of Tomotopy threads (default: min(8, CPU count)). |
| `--tomotopy-max-tokens-per-channel N` | Limit tokens per channel for Tomotopy training. |
| `--niche-words NICHE` | Enable niche word‑count metric sets (repeatable). |
| `--focus-text TEXT` | Focus text for similarity mapping. |
| `--color-by METRIC` | Metric to use for colouring the visualization. |

## Examples

### YouTube Channel Analysis (Video Centroid)

```bash
python3 core/atlas.py \
  --date-from 20200101 \
  --duration-from 8:00 \
  --exclude-live \
  --centroid-mode video \
  --centroid-videos 200 \
  --stat-word-count 1000000 \
  --fast-mode \
  --embeddings-only-low-token
```

### YouTube Channel Analysis (Word Centroid)

```bash
python3 core/atlas.py \
  --date-from 20200101 \
  --duration-from 8:00 \
  --exclude-live \
  --centroid-mode word \
  --token-limit 1000000 \
  --stat-word-count 1000000 \
  --fast-mode \
  --embeddings-only-low-token
```

### NLTK Corpus Analysis

```bash
python3 core/atlas.py --nltk-corpus gutenberg --no-metrics
python3 core/atlas.py --nltk-corpus reuters --centroid-mode word
```

### Custom Text Directory

```bash
python3 core/atlas.py --txt-directory ./documents --txt-depth 2
```

### Small Corpora (Tight UMAP)

```bash
python3 core/atlas.py --nltk-corpus gutenberg \
  --umap-neighbors 5 --umap-min-dist 0
```

### Tomotopy Topic Modeling

```bash
python3 core/atlas.py --tomotopy-topics 20 --tomotopy-iterations 2000
```

## Major Linguistic Features

### MALLET Topic Modeling

MALLET (MAchine Learning for LanguagE Toolkit) LDA (Latent Dirichlet Allocation) automatically discovers thematic topics in your corpus. Make sure your corpus is not too big as there will be complications. Training can take days if run on a giant corpus.

**Usage**: `--mallet-topics N` where N is the number of topics.

**How it works**:

1. Tokenizes centroid tokens from each text.
2. Trains an LDA model with N topics across all texts.
3. Computes topic proportions for each text.
4. Extracts top words for each topic label.

**Output**: Topic proportions are available in the HTML visualization and can be used for colouring.

**Performance**: For large corpora, set `MALLET_MEMORY` environment variable (e.g., `MALLET_MEMORY=12g`) to allocate more memory.

### Tomotopy Topic Modeling

Tomotopy is a lightweight, pure‑Python LDA implementation that avoids the Java dependency of MALLET. It is ideal for users who want topic modelling without installing additional runtimes.

**Usage**: `--tomotopy-topics N` where N is the number of topics (0 disables).

**Additional options**:

- `--tomotopy-iterations N` – training iterations (default: 1000).
- `--tomotopy-threads N` – number of threads (default: min(8, CPU count)).
- `--tomotopy-max-tokens-per-channel N` – limit tokens per channel for training.

**Output**: Topic proportions and labels are included in the HTML visualization, analogous to MALLET output.

### Burrows Delta Stylistic Analysis

Burrows Delta is a stylometric method that analyzes function word usage to identify authorship or stylistic differences.

**Key Features**:

- Uses a curated list of function words (pronouns, articles, prepositions, auxiliary verbs, fillers).
- Expands contractions (e.g., "I'm" → "I am").
- Computes normalized frequency profiles for each text.
- Calculates stylistic distance matrices.
- Provides `burrows_cosine_disagreement` metric comparing stylistic vs semantic similarity.

**Burrows Function Words (You can customize this list in `core/atlas.py` `BURROWS_FUNCTION_WORDS`)**:

- Pronouns: I, you, he, she, it, we, they, me, him, her, us, them
- Possessives: my, your, his, its, our, their, mine, yours, ours, theirs
- Auxiliary verbs: be, is, am, are, was, were, been, being, have, has, had, do, does, did
- Modals: will, would, shall, should, can, could, may, might, must
- Prepositions/conjunctions: to, of, and, a, in, that, for, not, on, with, as, at, but, by, from, or, an
- Fillers: oh, yeah, yes, okay, like, well, uh, um, right, mhm

**Focus Text Integration**: In the HTML visualization, you can select "Burrows" as the similarity mode when using the Focus Text feature to find texts with similar stylistic profiles.

### SBERT (Sentence-BERT) Semantic Embeddings

SBERT generates dense vector representations of text that capture semantic meaning, enabling similarity‑based clustering and visualization.

**Supported Models**:

- `BAAI/bge-small-en-v1.5` (default): Fast, good balance of speed/accuracy. Great accuracy for a small model.
- `all-MiniLM-L12-v2`: Fast, good balance of speed/accuracy.
- `all-mpnet-base-v2`: Higher accuracy, slower.
- `thenlper/gte-small`: Excellent performance, efficient.
- `Alibaba-NLP/gte-modernbert-base`: Large context window (8192 tokens), state‑of‑the‑art.

**Embedding Process**:

1. Text is chunked into segments (size depends on model context window).
2. Each chunk is encoded using the SBERT model.
3. Chunk embeddings are averaged to create a text‑level embedding.
4. Embeddings are used for:
   - UMAP dimensionality reduction.
   - Clustering (HDBSCAN/Leiden).
   - Cosine similarity calculations.
   - Focus text neighbour finding.

**Performance Tips**:

- Use `--fast-mode` for quick iterations.
- `gte-modernbert-base` is recommended for large contexts.
- CPU: batch size auto‑limited to 8.
- MPS (Apple Silicon): batch size auto‑limited to 16.
- CUDA: higher batch sizes available.

### Focus Text Display

The Focus Text feature enables interactive similarity mapping to explore relationships between texts.

**CLI Usage**: `--focus-text TEXT_NAME`

**HTML Interface**:

1. Click the "Focus Text" button in the visualization.
2. Enter a text name to focus on.
3. Choose similarity mode:
   - **Cosine**: Semantic similarity (SBERT embeddings).
   - **Burrows**: Stylistic similarity (function‑word profiles).
4. Optionally enable "Hide texts outside focused text cluster" to explore how clustering relates to semantic neighborhoods.
5. Identify outliers or unexpected connections.

## Metric Categories

### Core Linguistic Metrics

- **TTR**: Type‑Token Ratio.
- **Fighting Words**: Monroe log‑odds z‑scores for distinctive vocabulary.
- **Compression**: LZMA compression ratios (normalized and windowed).
- **N‑gram Entropy**: 2‑gram and 3‑gram cross‑entropy.
- **MTLD/MATTR**: Lexical diversity measures.
- **Average Words per Video**: For video centroid mode.

### Lexical Diversity

- **Yule's K**: Vocabulary richness statistic.
- **Average Word Length**: Character‑level measure.
- **Word Burstiness**: Gini coefficient of word frequency distribution.
- **Semantic Disparity**: Spread of word meanings.
- **Unique Bigrams/Trigrams**: N‑gram diversity.
- **Hapax/Dis Legomena**: Words used once or twice.

### Sentiment & Tone

- **VADER Positivity**: Positive sentiment ratio.
- **VADER Volatility**: Sentiment variance.
- **Perplexity**: GPT‑2 language model perplexity (requires model).

### Syntactic Patterns

- **Self‑Mention Count**: First‑person pronouns.
- **Hedge Word Count**: Uncertainty markers (maybe, perhaps, probably).
- **Subordinate/Coordinate Clause Counts**: Conjunction usage.
- **Latinate Word Ratio**: Greek/Latin morphological patterns.

### Discourse Markers

- **Family Exclamations**: G‑rated exclamations.
- **Vulnerability Count**: Expressions of uncertainty/weakness.
- **Simile Count**: Comparative expressions.
- **Color Mentions**: Color word usage.
- **Discussion Commentary**: Dialogue markers.
- **Articles/Demonstratives/Possessives**: Determiner patterns.
- **Quantifiers**: Quantity expressions.
- **Gerunds**: Present participles.
- **Deictic Spatial/Temporal**: Pointing language (here, there, now, then).
- **Elaboration/Explanation**: Explanatory structures.
- **Narration/Continuation**: Action‑based narrative markers.

### Cross‑Modal Similarity

- **Burrows/Cosine Disagreement**: Disagreement between stylistic and semantic similarity.

## Output Files

### CSV Output (`--output-csv`)

Contains all computed metrics for each text, with columns for each metric and text name.

### HTML Output (`--output-html`)

Interactive Plotly visualization with:

- UMAP scatter plot coloured by selected metric.
- Hover tooltips with text details.
- Cluster assignments.
- Focus text panel for similarity mapping.
- Metric selection dropdown.
- Clustering variant selector.
- Annotation support (if enabled).

## Performance Tips

1. **Use caching**: The cache system dramatically speeds up repeated runs.
2. **Fast mode**: `--fast-mode` skips expensive metrics for quick iterations.
3. **Word counts only**: `--word-counts-only` for rapid regex‑based metrics.
4. **Limit tokens**: Use `--token-limit` to focus on recent content.
5. **Batch size**: Auto‑adjusted for device, but can be overridden.
6. **MALLET memory**: Increase `MALLET_MEMORY` for large topic models.
7. **UMAP parameters**: Reduce `--umap-epochs` for faster visualization.

## Troubleshooting

- **CUDA out of memory**: Batch size auto‑reduces; try smaller `--embedding-batch-size`.
- **spaCy not found**: Run `pip install spacy && python -m spacy download en_core_web_sm`.
- **Hugging Face model not found**: Centroids uses the local Hugging Face cache when available and downloads missing model files automatically. Set `ATLAS_HF_LOCAL_FILES_ONLY=1` to require cache‑only loading for offline use. Alternatively, pre‑cache the configured embedding model (and `distilgpt2` when using `--calculate-perplexity`) and copy the cache to your environment.

## TODOs

- Guide for understanding the structure of data needed to use the YouTube channel part of the program.
- Design CSVs to work well with PowerBI and Tableau.
- Learn about PyTest and Playwright to test functions and web display.
- Actually getting findings from this tool.
- Improve the UI and add tutorials; it is currently confusing for broad use.
- **Problem**: Installing Centroids is painful because of mixed dependencies such as Python packages, Java/MALLET, NLTK data, spaCy models, torch/SBERT. Maybe use Docker to wrap it into a container? `pyproject.toml` + shell script to create venv and download models. Consider replacing MALLET with Tomotopy to drop the Java dependency.
```
