# Centroids

An advanced text analysis and visualization tool for comparing linguistic features across large text corpora. Built with YouTube channel analysis in mind, but supports NLTK corpora and custom text directories.

## Using the demo

First check the doctor file to check if you have everything installed:

```bash
python core/atlas_doctor.py 
```

or if you have python 3:

```bash
python3 core/atlas_doctor.py 
```

Then when everything is installed try the demo:

```bash
python core/atlas_demo.py 
```

or if you have python 3:

```bash
python3 core/atlas_demo.py 
```

## Key Features

- **Interactive HTML Visualization**: Plotly-based interactive scatter plots for exploring text similarities
- **Extensive Linguistic Metrics**: 120+ metrics including sentiment analysis, lexical diversity, discourse markers, and stylometric features
- **Advanced Topic Modeling**: MALLET LDA integration for topic discovery
- **Stylistic Analysis**: Burrows Delta method for function-word based authorship/style analysis
- **Semantic Embeddings**: SBERT (Sentence-BERT) for semantic similarity mapping
- **Focus Text Display**: Interactive similarity mapping to highlight relationships between texts
- **Smart Caching**: Multi-level caching system to avoid redundant computations
- **Centroid Modes**: Two aggregation strategies - by video count or by word count
- **Clustering**: HDBSCAN and Leiden algorithms with parameter sweeps
- **UMAP Dimensionality Reduction**: Configurable parameters for visualization

## Installation

```bash
pip install sentence-transformers scikit-learn umap-learn plotly torch transformers vaderSentiment lexicalrichness orjson
pip install spacy && python -m spacy download en_core_web_sm  # Optional, for lexical_density metric
pip install pyenchant  # Optional, for OOV word metrics
pip install python-igraph leidenalg  # Optional, for Leiden clustering
```

For MALLET topic modeling, download and install [MALLET](http://mallet.cs.umass.edu/). Set the `MALLET_HOME` environment variable or ensure `mallet` binary is in your PATH.

## Data layout

For YouTube-style input, the tool expects one directory per "text" (channel, book, document), containing a `txt_files/` subfolder of `.txt` files (one per video or chapter):

```
data/input/
  my_channel/
    txt_files/
      video1.txt
      video2.txt
      ...
    metadata.json       (optional — YouTube-style fields)
    tag.txt             (optional — short text shown under the channel name in hover)
  another_channel/
    txt_files/
      ...
```

Each `.txt` file is treated as one document. In video-centroid mode, the N most recent documents become the text's centroid; in word-centroid mode, the last N words across all documents do.

For non-YouTube use, `--txt-directory` and `--nltk-corpus` accept alternative sources without this layout.

## CLI Usage

### Basic Syntax

```bash
python3 core/atlas.py [text_folder_names...] [OPTIONS]
```

### Data Source Options

```
texts                    Text folder names to process (default: YouTube channels)
--nltk-corpus NAME       Compare texts from an NLTK corpus (e.g., gutenberg, reuters)
--txt-directory PATH     Compare .txt files from a directory (includes nested folders)
--txt-depth N            Max subdirectory depth for --txt-directory (default: 3)
```

### Filtering Options (YouTube)

```
--date-from YYYYMMDD     Include only videos from this date onwards
--date-to YYYYMMDD       Include only videos up to this date
--duration-from HH:MM:SS Minimum video duration
--duration-to HH:MM:SS   Maximum video duration
--exclude-live           Exclude livestream videos
--ignore-urls PATH       Path to file with YouTube URLs/IDs to ignore (one per line)
```

### Token Limits

```
--token-limit N          Use only the newest N tokens per text
--text-token-limit N     Use only the first N tokens per text
--per-video-token-limit N Use only the first N tokens per video
--min-tokens-per-file N  Skip files with fewer than N tokens
--min-tokens-total N     Skip texts with fewer than N total tokens (alias: --min-words)
--embeddings-only-low-token Generate embeddings for low-token texts, skip metrics
```

### Centroid Modes

Centroid modes control how text embeddings are computed from multiple documents:

```
--centroid-mode {video,word}
  video: Aggregate embeddings from the N most recent videos
  word:  Aggregate from the last N words in the corpus

--centroid-videos N      Number of videos for video centroid (default: 10)
--centroid-words N       Number of words for word centroid (default: 1,000,000)
--stat-word-count N      Words for statistical metrics with video centroids
```

### Metric Computation Options

```
--no-metrics             Skip all metric calculations (embeddings/clustering only)
--fast-mode              Skip expensive metrics (fastest mode)
--word-counts-only       Compute only count-style and regex-pattern metrics
--calculate-perplexity   Enable GPT-2 perplexity calculation (slower)
--compute-ngram-entropy  Enable n-gram cross-entropy metrics (slower)
--compute-fighting-words Enable fighting-words peak/floor metrics
--get-metrics-csv        Also write a <output-csv>_metrics.csv file with one row per text
```

The metrics CSV uses metric names as its first row and one text per row. In channel mode,
text labels are limited to five characters by default; set `METRICS_CSV_FULL_CHANNEL_NAMES`
in `core/atlas_config.py` to `True` to keep full channel names. External text sources
(`--nltk-corpus` and `--txt-directory`) also provide `readability` (Flesch Reading Ease)
and `average_sentence_length` metrics.

### Clustering & UMAP

```
--cluster-method {hdbscan,leiden}
  hdbscan: Density-based clustering (default)
  leiden:  Community detection on k-NN graph (great for large datasets)

--cluster-variants N      Number of clustering variants (default: 11 for ≤2000 items, else 1)
--umap-neighbors N       UMAP neighborhood size (auto: 5 for ≤30 items, else 50)
--umap-min-dist FLOAT     Minimum distance between points (default: 0.1; use 0 for tight groups)
--umap-epochs N          UMAP optimization epochs (default: 50)

--leiden-resolution-min FLOAT  Minimum Leiden resolution (default: 1.15)
--leiden-resolution-max FLOAT  Maximum Leiden resolution (default: 3.0)
--leiden-k-neighbors N        k-NN graph neighbors for Leiden (default: 20)
--no-leiden-whiten            Disable PCA whitening before k-NN (use for homogeneous datasets)
```

### Embedding Model

```
--embedding-model NAME    SBERT model name (default: all-MiniLM-L12-v2)
  Examples: all-mpnet-base-v2, thenlper/gte-small, Alibaba-NLP/gte-modernbert-base
--embedding-chunk-size N  Override chunk size in model tokens
--embedding-min-chunk N   Override minimum chunk size
--embedding-batch-size N  Override batch size for encoding
--trained-model-path PATH Path to custom fine-tuned SBERT model
```

### Output & Caching

```
--output-csv PATH         Output CSV file path (default: data/input/text_similarities.csv)
--output-html PATH        Output HTML visualization (default: data/input/SBERTClustersHD.html)
--no-cache                Disable embedding cache
--force-latest-cache      Use newest cache file without freshness checks
--clear-cache [TEXT]      Clear cache for specific text or all texts
--annotations             Enable annotation loading and click-to-view notes
```

Each channel directory may also contain an optional `tag.txt`. Its short contents
appear directly below the channel name in the visualization hover text.

### Advanced Features

```
--mallet-topics N         Train MALLET LDA model with N topics (0 disables)
                          Set MALLET_MEMORY env var (e.g., 12g) for large corpora
--niche-words NICHE       Enable niche word-count metric sets (repeatable)
--focus-text TEXT         Focus text for similarity mapping
--color-by METRIC         Metric to use for coloring the visualization
```

## Examples

### YouTube Channel Analysis (Video Centroid)

```bash
python3 core/atlas.py \
  --date-from 20200101 --duration-from 8:00 --exclude-live \
  --centroid-mode video \
  --centroid-videos 200 \
  --stat-word-count 1000000 \
  --fast-mode --embeddings-only-low-token
```

### YouTube Channel Analysis (Word Centroid)

```bash
python3 core/atlas.py \
  --date-from 20200101 --duration-from 8:00 --exclude-live \
  --centroid-mode word \
  --token-limit 1000000 \
  --stat-word-count 1000000 \
  --fast-mode --embeddings-only-low-token
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

### MALLET Topic Modeling

```bash
MALLET_MEMORY=8g python3 core/atlas.py \
  --mallet-topics 20 \
  --centroid-mode word
```

### Leiden Clustering for Large Datasets

```bash
python3 core/atlas.py \
  --cluster-method leiden \
  --cluster-variants 8 \
  --leiden-resolution-min 0.3 \
  --leiden-resolution-max 2.0
```

## Major Linguistic Features

### MALLET Topic Modeling

MALLET (MAchine Learning for LanguagE Toolkit) LDA (Latent Dirichlet Allocation) automatically discovers thematic topics in your corpus.

**Usage**: `--mallet-topics N` where N is the number of topics

**How it works**:
1. Tokenizes centroid tokens from each text
2. Trains an LDA model with N topics across all texts
3. Computes topic proportions for each text
4. Extracts top words for each topic label

**Output**: Topic proportions are available in the HTML visualization and can be used for coloring.

**Performance**: For large corpora, set `MALLET_MEMORY` environment variable (e.g., `MALLET_MEMORY=12g`) to allocate more memory.

### Burrows Delta Stylistic Analysis

Burrows Delta is a stylometric method that analyzes function word usage to identify authorship or stylistic differences.

**Key Features**:
- Uses a curated list of function words (pronouns, articles, prepositions, auxiliary verbs, fillers)
- Expands contractions (e.g., "I'm" → "I am")
- Computes normalized frequency profiles for each text
- Calculates stylistic distance matrices
- Provides `burrows_cosine_disagreement` metric comparing stylistic vs semantic similarity

**Burrows Function Words (You can customize this list in `core/atlas.py BURROWS_FUNCTION_WORDS`)**:
- Pronouns: I, you, he, she, it, we, they, me, him, her, us, them
- Possessives: my, your, his, its, our, their, mine, yours, ours, theirs
- Auxiliary verbs: be, is, am, are, was, were, been, being, have, has, had, do, does, did
- Modals: will, would, shall, should, can, could, may, might, must
- Prepositions/conjunctions: to, of, and, a, in, that, for, not, on, with, as, at, but, by, from, or, an
- Fillers: oh, yeah, yes, okay, like, well, uh, um, right, mhm

**Focus Text Integration**: In the HTML visualization, you can select "Burrows" as the similarity mode when using the Focus Text feature to find texts with similar stylistic profiles.

### SBERT (Sentence-BERT) Semantic Embeddings

SBERT generates dense vector representations of text that capture semantic meaning, enabling similarity-based clustering and visualization.

**Supported Models**:
- `bge-small-en-v1.5` (default): Fast, good balance of speed/accuracy. Great accuracy for a small model.
- `all-MiniLM-L12-v2` : Fast, good balance of speed/accuracy
- `all-mpnet-base-v2`: Higher accuracy, slower
- `thenlper/gte-small`: Excellent performance, efficient
- `Alibaba-NLP/gte-modernbert-base`: Large context window (8192 tokens), state-of-the-art

**Embedding Process**:
1. Text is chunked into segments (size depends on model context window)
2. Each chunk is encoded using the SBERT model
3. Chunk embeddings are averaged to create a text-level embedding
4. Embeddings are used for:
   - UMAP dimensionality reduction
   - Clustering (HDBSCAN/Leiden)
   - Cosine similarity calculations
   - Focus text neighbor finding

**Performance Tips**:
- Use `--fast-mode` for quick iterations
- `gte-modernbert-base` is recommended for large contexts
- CPU: batch size auto-limited to 8
- MPS (Apple Silicon): batch size auto-limited to 16
- CUDA: higher batch sizes available

### Focus Text Display

The Focus Text feature enables interactive similarity mapping to explore relationships between texts.

**CLI Usage**: `--focus-text TEXT_NAME`

**HTML Interface**:
1. Click the "Focus Text" button in the visualization
2. Enter a text name to focus on
3. Choose similarity mode:
   - **Cosine**: Semantic similarity (SBERT embeddings)
   - **Burrows**: Stylistic similarity (function word profiles)
4. Optionally enable "Hide texts outside focused text cluster" to filter
5. Select which clustering metric to use for cluster filtering

**Features**:
- Star marker highlights the focused text
- Nearby texts are shown with lines connecting to the focus
- Similarity scores displayed in percentile
- Cluster filtering to isolate related texts
- Dynamic updates as you switch texts

**Use Cases**:
- Find texts semantically similar to a specific video or document
- Compare stylistic similarities across authors or channels
- Explore how clustering relates to semantic neighborhoods
- Identify outliers or unexpected connections

## Metric Categories

### Core Linguistic Metrics
- **TTR**: Type-Token Ratio
- **Fighting Words**: Monroe log-odds z-scores for distinctive vocabulary
- **Compression**: LZMA compression ratios (normalized and windowed)
- **N-gram Entropy**: 2-gram and 3-gram cross-entropy
- **MTLD/MATTR**: Lexical diversity measures
- **Average Words per Video**: For video centroid mode

### Lexical Diversity
- **Yule's K**: Vocabulary richness statistic
- **Average Word Length**: Character-level measure
- **Word Burstiness**: Gini coefficient of word frequency distribution
- **Semantic Disparity**: Spread of word meanings 
- **Unique Bigrams/Trigrams**: N-gram diversity
- **Hapax/Dis Legomena**: Words used once or twice

### Sentiment & Tone
- **VADER Positivity**: Positive sentiment ratio
- **VADER Volatility**: Sentiment variance
- **Perplexity**: GPT-2 language model perplexity (requires model)

### Syntactic Patterns
- **Self-Mention Count**: First-person pronouns
- **Hedge Word Count**: Uncertainty markers (maybe, perhaps, probably)
- **Subordinate/Coordinate Clause Counts**: Conjunction usage
- **Latinate Word Ratio**: Greek/Latin morphological patterns

### Discourse Markers
- **Family Exclamations**: G-rated exclamations
- **Vulnerability Count**: Expressions of uncertainty/weakness
- **Simile Count**: Comparative expressions
- **Color Mentions**: Color word usage
- **Discussion Commentary**: Dialogue markers
- **Articles/Demonstratives/Possessives**: Determiner patterns
- **Quantifiers**: Quantity expressions
- **Gerunds**: Present participles
- **Deictic Spatial/Temporal**: Pointing language (here, there, now, then)
- **Elaboration/Explanation**: Explanatory structures
- **Narration/Continuation**: Action-based narrative markers

### Cross-Modal Similarity
- **Burrows/Cosine Disagreement**: Disagreement between stylistic and semantic similarity

## Output Files

### CSV Output (`--output-csv`)
Contains all computed metrics for each text, with columns for each metric and text name.

### HTML Output (`--output-html`)
Interactive Plotly visualization with:
- UMAP scatter plot colored by selected metric
- Hover tooltips with text details
- Cluster assignments
- Focus text panel for similarity mapping
- Metric selection dropdown
- Clustering variant selector
- Annotation support (if enabled)

## Performance Tips

1. **Use caching**: The cache system dramatically speeds up repeated runs
2. **Fast mode**: `--fast-mode` skips expensive metrics for quick iterations
3. **Word counts only**: `--word-counts-only` for rapid regex-based metrics
4. **Limit tokens**: Use `--token-limit` to focus on recent content
5. **Batch size**: Auto-adjusted for device, but can be overridden
6. **MALLET memory**: Increase `MALLET_MEMORY` for large topic models
7. **UMAP parameters**: Reduce `--umap-epochs` for faster visualization

## Troubleshooting

- **CUDA out of memory**: Batch size auto-reduces; try smaller `--embedding-batch-size`
- **spaCy not found**: Run `pip install spacy && python -m spacy download en_core_web_sm`
- **MALLET not found**: Install MALLET and set `MALLET_HOME` or add to PATH
- **Low token counts**: Use `--embeddings-only-low-token` to skip metrics
- **Large datasets**: Use Leiden clustering with `--no-leiden-whiten` for homogeneous corpora

## TODOs

Doctor file to help users download packages before running the program.

Demo file to help users test the program with a small dataset and test the features.

Guide for understanding the structure of data needed to use the YouTube channel part of the program.
