# Centroids
WIP More advanced version of my "atlas" project with many additional features. Created using AI peer programming with Github Copilot and Deepseek.

## Important Features different from atlas

Uses Plot.ly to make an interactive HTML output.

Multiple new creative methods like VADER sentiment, word counts of common words and function words, perplexity, average embedding distance between words.

Correlation mode to see correlation between metrics.

Caching system to prevent redoing processes all over again.

Fast metric compute for regex word-count metrics.

Two separate centroid modes to create a channel embedding, choose to accumulate by a number of videos or choose to accumulate by a number of words.

## Example

Centroid videos mode:

python3 core/atlas.py \
--date-from 20200101 --duration-from 8:00 --exclude-live \
--centroid-mode video \
--centroid-videos 200 \
--stat-word-count 1000000 \
--fast-mode --embeddings-only-low-token

Centroid words mode:

python3 core/atlas.py \
--date-from 20200101 --duration-from 8:00 --exclude-live \
--centroid-mode word \
--token-limit 1000000 \
--stat-word-count 1000000 \
--fast-mode --embeddings-only-low-token

WIP: CLI will be detailed in README when project is finished.

## Text and NLTK sources

The same metrics, embeddings, clustering, UMAP visualization, and CSV/HTML
outputs can compare individual texts from an installed NLTK corpus:

```bash
python3 core/atlas.py --nltk-corpus gutenberg --no-metrics
python3 core/atlas.py --nltk-corpus reuters --centroid-mode word
```

To compare `.txt` files from a directory, including files in nested folders,
use `--txt-directory`. The default maximum depth is 3; depth 0 includes only
files directly in the selected directory:

```bash
python3 core/atlas.py --txt-directory ./documents --txt-depth 2
```

External document sources support the default mode and `--centroid-mode word`.
YouTube channel processing remains the default when neither source option is
provided.

For small corpora such as Gutenberg, use a local UMAP neighborhood and tighter
point packing:

```bash
python3 core/atlas.py --nltk-corpus gutenberg \
	--umap-neighbors 5 --umap-min-dist 0
```

UMAP automatically uses 5 neighbors for datasets of 30 items or fewer. The
`--umap-neighbors`, `--umap-min-dist`, and `--umap-epochs` options override
that behavior when needed.

Currently in the process of proofreading code and debugging. 

