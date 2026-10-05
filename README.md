# Centroids
WIP Data Analysis project. More advanced version of my "umapper" project with many, a lot of, additional features. 

## Important Features different from umapper

Uses Plot.ly to make an interactive HTML output.

Is not limited to just analyzing Youtube channels, it is compatible with directories of TXT files, and NTLK corpuses.

Multiple new creative methods like VADER sentiment, word counts of common words and function words, perplexity, average embedding distance between words.

Experimentation with multiple digital humanities tools and statistical methods such as MALLET, Burrows Delta, and Bayesian log-odds.

Correlation mode tab in HTML output to see correlation between metrics on a scatter plot. 

Efficient caching system to prevent redoing processes all over again.

Fast metric compute architecture for regex word-count metrics.

In the default Youtube channel analysis mode, there are two separate centroid modes to create a channel embedding, choose to accumulate by a number of videos (like 200 videos) or choose to accumulate by a number of words (like 1 million words).

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

## Methods

Created using AI peer programming with Github Copilot and Deepseek.

