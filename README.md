# Centroids
WIP More advanced version of my "umapper" project with many additional features. Created using AI peer programming with Github Copilot and Deepseek.

## Important Features different from umapper

Uses Plot.ly to make an interactive HTML output.

Multiple new creative methods like VADER sentiment, word counts of common words and function words, perplexity, average embedding distance between words.

Correlation mode to see correlation between metrics.

Caching system to prevent redoing processes all over again.

Fast metric compute for regex word-count metrics.

Two separate centroid modes to create a channel embedding, choose to accumulate by a number of videos or choose to accumulate by a number of words.

## Example

Centroid videos mode:

python3 main/centroids/run.py \
--date-from 20200101 --duration-from 8:00 --exclude-live \
--centroid-mode video \
--centroid-videos 200 \
--stat-word-count 1000000 \
--fast-mode --embeddings-only-low-token

Centroid words mode:

python3 main/centroids/run.py \
--date-from 20200101 --duration-from 8:00 --exclude-live \
--centroid-mode word \
--token-limit 1000000 \
--stat-word-count 1000000 \
--fast-mode --embeddings-only-low-token

WIP: CLI will be detailed in README when project is finished.

Currently in the process of proofreading code and debugging. 

