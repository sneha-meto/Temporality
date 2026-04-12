# Temporality

**Temporal Signals of Deception: Tense and Event Structure in Fake vs Verified News**

A research pipeline for the TCD group paper investigating whether temporal language patterns (tense distribution, event sequencing, narrative coherence) differ systematically between fake and real news articles.

---

## Hypotheses

| ID | Claim |
|----|-------|
| H1 | Fake news contains more temporal inconsistencies (mismatched event dates, anachronistic references) |
| H2 | Fake news shows higher tense entropy and more tense shifts across article sections |
| H3 | Fake news has lower narrative coherence, correlated with higher event-timeline misalignment |
| H4 | Fake news shows greater tense misalignment between structural sections (title/lead/body)

---

## Project Structure

```
Temporality/
├── run_pipeline.py               # Orchestrator — entry point
├── requirements.txt
├── src/
│   ├── utils.py                  # Paths, logging, shared constants & lexicons
│   ├── phase1_preprocessing.py   # ISOT loading, cleaning, topic KMeans, sectioning, splits
│   ├── phase2_tense_features.py  # spaCy tense/aspect extraction (→ H2)
│   ├── phase3_temporal_expressions.py  # Regex temporal tagger, connectives, misalignment (→ H1, H3)
│   ├── phase4_coherence.py       # Embedding + entity-grid coherence (→ H3)
│   └── phase5_analysis.py        # Mann-Whitney U, Spearman, LR+RF+BoW classification
├── data/
│   ├── raw/                      # Place True.csv and Fake.csv here
│   └── processed/                # Intermediate parquets written here
└── results/
    ├── figures/                  # PNG plots
    └── tables/                   # CSV result tables
```

---

## Setup

**Requirements:** Python 3.12, pip

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Dataset: ISOT Fake News Dataset

Dataset link: https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset

**Option A — Manual download (recommended):**
1. Download the dataset from the link above (requires a free Kaggle account).
2. Extract the archive and place the two CSV files in `data/raw/`:

```
Temporality/
└── data/
    └── raw/
        ├── True.csv
        └── Fake.csv
```

The filenames must be exactly `True.csv` and `Fake.csv` (case-sensitive).


## Running the Pipeline

```bash
# Recommended first run — smoke test on 500 articles
python run_pipeline.py --sample 500

# Full pipeline (all articles — slow due to spaCy)
python run_pipeline.py
```

### Partial runs

Once intermediate parquets exist in `data/processed/`, you can skip completed phases:

```bash
# Skip phases 1 and 2, re-run from phase 3 onwards
python run_pipeline.py --skip-phase1 --skip-phase2

# Run only one specific phase
python run_pipeline.py --only phase5
```

### All flags

| Flag | Description |
|------|-------------|
| `--sample N` | Process only N randomly sampled articles |
| `--skip-phase1` | Skip preprocessing (requires `corpus.parquet`) |
| `--skip-phase2` | Skip tense extraction (requires `features_tense.parquet`) |
| `--skip-phase3` | Skip temporal expressions (requires `features_temporal.parquet`) |
| `--skip-phase4` | Skip coherence scoring (requires `features_coherence.parquet`) |
| `--skip-phase5` | Skip analysis |
| `--only PHASE` | Run only `phase1`…`phase5` |

---

## Output

| Location | Contents |
|----------|----------|
| `results/tables/hypothesis_tests.csv` | Mann-Whitney U results for H1 & H2 |
| `results/tables/spearman_h3.csv` | Spearman correlations for H3 |
| `results/tables/classification_report.csv` | LR & RF classification metrics |
| `results/figures/` | Tense distribution plots, coherence score plots, etc. |

---

## Pipeline Phases

| Phase | Module | What it does |
|-------|--------|--------------|
| 1 | `phase1_preprocessing.py` | Load & clean ISOT, remove boilerplate, topic clustering (TF-IDF + KMeans k=5), section into title/lead/body, 70/10/20 stratified split |
| 2 | `phase2_tense_features.py` | spaCy dependency parse → tense/aspect per verb, tense entropy, shift rate, sectional alignment |
| 3 | `phase3_temporal_expressions.py` | Regex temporal tagger, temporal connective density, event-timeline misalignment score |
| 4 | `phase4_coherence.py` | Sentence-transformer embedding coherence + entity-grid coherence |
| 5 | `phase5_analysis.py` | Mann-Whitney U (H1, H2), Spearman (H3), logistic regression + random forest classification |

---

## Tool Stack

| Task | Tool |
|------|------|
| POS tagging, dependency parsing | spaCy `en_core_web_sm` |
| Temporal expression tagging | Pure-Python regex tagger |
| Sentence embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Topic clustering | scikit-learn KMeans |
| Statistical tests | `scipy.stats` |
| Classification | scikit-learn |
| Visualisation | matplotlib, seaborn |
