
# Twitter/X Content Pipeline — Extraction, Preprocessing, Bot Detection & Engagement Lifecycle

### Project Overview

This repository covers a full pipeline for social-media content analysis:

1. **Extraction** (`Twitter_extraction_Final.ipynb`) — pulls tweets via the
   Tweepy API.
2. **Preprocessing** (`twitter_final_Preprocess.ipynb`, `Working.ipynb`) —
   cleans tweet text (lowercasing, removing mentions/hashtags/URLs/
   punctuation/emojis, stopword removal).
3. **Bot detection** (`src/`) — a trained supervised classifier that scores
   accounts as bot-like vs human-like from behavioral + text features.
4. **Engagement lifecycle analysis** (`src/`) — curve fitting and viral/
   normal classification of how a post's engagement evolves over time.
5. **Streaming simulation** (`src/`) — an incremental, per-event scoring
   loop demonstrating the processing pattern a real-time deployment would
   use.

---

## Honesty note on data sources ("Real-Time" and data provenance)

The Twitter/X API now requires a paid tier for any meaningful search,
history, or streaming access. This project does **not** have paid API
credentials, so:

* Stages 1–2 (extraction/preprocessing notebooks) show the real code for
  pulling and cleaning tweets, but were last run against small free-tier
  samples and are not re-run automatically here.
* Stages 3–5 (`src/`) are trained and evaluated on **synthetic but
  honestly-labeled data** — see `src/synthetic_accounts.py` and
  `src/engagement_lifecycle.py` for the exact generative model and label
  logic. Ground truth is programmatic (we know `is_bot` / `is_viral`
  because we generated it), which is what makes it possible to report real
  precision/recall/F1/AUC below.
* Nothing in this repo maintains an open connection to the live Twitter/X
  streaming API. The title's "Real-Time" refers to the **streaming-
  simulated / near-real-time processing pattern** implemented in
  `src/streaming_simulation.py` (incremental per-event feature updates and
  scoring), not a live production system. If you have paid API access, the
  same `AccountState` incremental-update pattern can be pointed at a real
  `tweepy.StreamingClient` with no change to the scoring logic.

This mirrors the same "honest synthetic data with documented ground truth"
approach used elsewhere in this author's portfolio.

---

## Repository Structure

```
Repo/
 ┣ Twitter_extraction_Final.ipynb      # Extracts tweets using Twitter API (Tweepy)
 ┣ twitter_final_Preprocess.ipynb      # Cleans and preprocesses tweet text
 ┣ Working.ipynb                       # Combines and demonstrates full working pipeline
 ┣ src/
 ┃ ┣ synthetic_accounts.py             # Synthetic bot/human account + post generator (documented ground truth)
 ┃ ┣ features.py                       # TF-IDF-based duplicate-content-rate feature extraction
 ┃ ┣ bot_classifier.py                 # Trains/evaluates GradientBoostingClassifier, saves metrics + model
 ┃ ┣ engagement_lifecycle.py           # Synthetic engagement time series + decay-curve fitting + viral detection
 ┃ ┗ streaming_simulation.py           # Incremental per-event bot-likelihood scoring ("near-real-time" demo)
 ┣ tests/
 ┃ ┣ test_features.py
 ┃ ┣ test_bot_classifier.py
 ┃ ┗ test_engagement_lifecycle.py
 ┣ outputs/                            # Generated metrics, plots, CSVs, trained model (committed for reference)
 ┣ requirements.txt
 ┗ README.md
```

---

## How to Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 1. Bot detection: builds synthetic dataset, trains classifier, writes outputs/
python -m src.bot_classifier

# 2. Engagement lifecycle: simulates post time series, fits decay curves,
#    classifies viral vs normal, writes outputs/
python -m src.engagement_lifecycle

# 3. Streaming simulation: incremental per-event scoring demo
python -m src.streaming_simulation

# Run tests
pytest tests/ -q
```

The extraction/preprocessing notebooks (`Twitter_extraction_Final.ipynb`,
`twitter_final_Preprocess.ipynb`) require your own Twitter Developer
credentials and are independent of the `src/` pipeline above; `src/`
consumes text in the same cleaned format those notebooks produce.

---

## Bot Detection — Method & Measured Results

**Features** (`src/synthetic_accounts.py`, `src/features.py`):

| Feature | Description |
|---|---|
| `account_age_days` | Synthetic account age |
| `follower_following_ratio` | Follower/following ratio |
| `posts_per_day` | Posting frequency |
| `posting_interval_cv` | Coefficient of variation of inter-post gaps (low = suspiciously regular timing) |
| `burstiness` | Fano-factor-style burst score |
| `avg_hashtags_per_post` | Average hashtags per post |
| `default_profile_ratio` | Proxy for default/unedited profile |
| `duplicate_content_rate` | TF-IDF cosine-similarity duplicate rate across an account's own posts |

The two populations (bot-like / human-like) are generated with
**deliberately overlapping distributions and per-feature noise** so the
task is not trivially separable (see the generator's parameters — bots and
humans both span the full range of most features, just with different
means).

**Model**: `GradientBoostingClassifier` (scikit-learn), trained on a
2,000-account synthetic dataset (35% bots), 75/25 stratified train/test
split.

**Measured held-out test metrics** (`outputs/bot_detection_metrics.json`,
n_test = 500):

| Metric | Value |
|---|---|
| Accuracy | 0.850 |
| Precision (bot) | 0.847 |
| Recall (bot) | 0.697 |
| F1 (bot) | 0.765 |
| ROC AUC | 0.922 |

Full classification report: `outputs/bot_detection_report.txt`.
Feature importances: `outputs/feature_importances.csv`.
ROC curve: `outputs/roc_curve.png`.

These numbers are exactly what you'd expect from a genuinely
non-trivial-but-learnable synthetic task: strong AUC, but recall well
below 1.0 because some synthetic bots are deliberately generated to look
"human-like" (low posting frequency, organic-looking timing) and vice
versa.

---

## Engagement Lifecycle Analysis — Method & Results

`src/engagement_lifecycle.py` simulates 300 posts' 72-hour cumulative
engagement curves using a logistic-growth-then-exponential-decay shape
(the standard pattern reported for social-media content diffusion), with
12% generated as "viral" (higher peak, slower decay/longer tail).

For each post we:
1. **Fit** the rise-decay model back out with `scipy.optimize.curve_fit`
   to recover peak time, peak height, decay rate, and half-life.
2. **Classify** viral vs normal with a rule based on peak height and decay
   half-life (sustained high engagement = viral).
3. **Cross-check** with KMeans clustering on the same shape features.
4. **Visualize** growth/peak/decay stages for representative posts
   (`outputs/engagement_lifecycle_examples.png`).

**Measured results** (`outputs/engagement_lifecycle_summary.json`, n=300):

| Metric | Value |
|---|---|
| Rule-based vs. cluster-based agreement | 0.937 |
| Rule-based recall vs. synthetic ground truth | 1.00 |
| Rule-based precision vs. synthetic ground truth | 0.90 |
| Viral posts flagged (rule-based) | 40 / true 36 |

Full per-post results: `outputs/engagement_lifecycle_results.csv`.

---

## Streaming Simulation

`src/streaming_simulation.py` replays synthetic tweets arriving one at a
time for 20 accounts, maintaining an `AccountState` object per account that
updates posting-interval statistics and duplicate-content rate
incrementally (not recomputed from scratch), then scores bot-likelihood
with the trained classifier after every arrival. This is the processing
pattern (incremental state + per-event scoring) a true real-time/streaming
deployment would need — see the honesty note above on why this is
simulated rather than connected to a live API.

---

## Limitations (worth saying out loud)

* All modeling in `src/` runs on **synthetic** data with programmatic
  ground truth, not real labeled Twitter/X bot accounts or real engagement
  telemetry — real-world performance would need validation against an
  actual labeled dataset (e.g. Botometer-labeled accounts) before any
  production use.
* The synthetic generator's feature distributions are hand-tuned to be
  "realistic but not trivial" based on published bot-detection literature
  patterns (regular timing, duplication, follower ratios), not fit to real
  data.
* The engagement-lifecycle curve shape (logistic growth + exponential
  decay) is a common simplification; real engagement curves can have
  multiple resurgence peaks (e.g. re-shares, news cycles) that this single-
  peak model does not capture.
* The streaming simulation processes synthetic events in a tight loop, not
  a real message queue/stream — it demonstrates the incremental-scoring
  pattern, not production streaming infrastructure.

---

## Tools & Libraries Used

* Python, pandas, numpy, scikit-learn, scipy, matplotlib, joblib, pytest
* Tweepy (Twitter API, extraction notebooks only)

---

## Author

**Kabilesh Rajaselvan**
M.Tech Integrated – VIT Chennai
GitHub: https://github.com/KabileshRajaselvan
