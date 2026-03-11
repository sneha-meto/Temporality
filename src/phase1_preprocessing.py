"""
Phase 1 – Dataset loading, cleaning, topic clustering, sectioning, and splitting.

Expected inputs : data/raw/True.csv  and  data/raw/Fake.csv  (ISOT dataset).
Outputs         : data/processed/corpus.parquet   (full cleaned corpus with splits)
                  data/processed/topic_model.pkl  (fitted KMeans model)
"""

import re
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.utils import (
    DATA_RAW, DATA_PROC, RANDOM_STATE, N_TOPICS, TOPIC_NAMES, get_logger,
)

warnings.filterwarnings("ignore")
log = get_logger(__name__)


# ── 1. Load ────────────────────────────────────────────────────────────────

def _kaggle_download() -> None:
    """Download ISOT dataset from Kaggle into DATA_RAW using kagglehub."""
    try:
        import kagglehub, shutil
        log.info("Downloading ISOT dataset via kagglehub…")
        src = Path(kagglehub.dataset_download("csmalarkodi/isot-fake-news-dataset"))
        log.info("kagglehub cache: %s", src)
        for name in ["True.csv", "Fake.csv"]:
            candidates = list(src.rglob(name))
            if not candidates:
                raise FileNotFoundError(f"{name} not found in kagglehub download at {src}")
            shutil.copy(candidates[0], DATA_RAW / name)
            log.info("Copied %s → %s", candidates[0], DATA_RAW / name)
    except ImportError:
        raise ImportError(
            "kagglehub is not installed. Run:  pip install kagglehub\n"
            "Or place True.csv / Fake.csv manually in:  " + str(DATA_RAW)
        )


def load_isot() -> pd.DataFrame:
    true_path = DATA_RAW / "True.csv"
    fake_path = DATA_RAW / "Fake.csv"

    if not true_path.exists() or not fake_path.exists():
        log.info("Dataset files not found in %s — attempting kagglehub download…", DATA_RAW)
        _kaggle_download()

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df["label"] = "real"
    fake_df["label"] = "fake"

    df = pd.concat([true_df, fake_df], ignore_index=True)
    log.info("Loaded %d articles (%d real, %d fake)",
             len(df), (df.label == "real").sum(), (df.label == "fake").sum())
    return df


# ── 2. Clean ───────────────────────────────────────────────────────────────

_REUTERS_RE = re.compile(
    r"^[A-Z\s\(\)]+\(Reuters\)\s*[-–]\s*", re.IGNORECASE
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _REUTERS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def clean_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("").apply(_clean_text)
    df["text"]  = df["text"].fillna("").apply(_clean_text)
    df["full_text"] = df["title"] + " " + df["text"]

    before = len(df)
    df["word_count"] = df["text"].apply(_word_count)
    df = df[(df["word_count"] >= 50) & (df["word_count"] <= 5000)]
    log.info("Length filter: %d → %d articles", before, len(df))

    # Drop exact-duplicate texts
    df = df.drop_duplicates(subset=["text"])
    log.info("After dedup: %d articles", len(df))

    return df.reset_index(drop=True)


# ── 3. Parse & filter dates ────────────────────────────────────────────────

def _parse_date(raw) -> pd.Timestamp | None:
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    try:
        return pd.Timestamp(date_parser.parse(str(raw), fuzzy=True))
    except Exception:
        return None


def filter_by_date(df: pd.DataFrame,
                   start: str = "2016-01-01",
                   end:   str = "2018-01-01") -> pd.DataFrame:
    df = df.copy()
    df["pub_date"] = df["date"].apply(_parse_date)

    before = len(df)
    df = df[df["pub_date"].notna()]
    df = df[(df["pub_date"] >= start) & (df["pub_date"] < end)]
    log.info("Date filter [%s, %s): %d → %d articles", start, end, before, len(df))
    return df.reset_index(drop=True)


# ── 4. Sectioning ──────────────────────────────────────────────────────────

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    """Simple rule-based sentence splitter (avoids spaCy dependency here)."""
    sentences = _SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def section_articles(df: pd.DataFrame, lead_n: int = 3) -> pd.DataFrame:
    """Add columns: title (already present), lead, body, sentences."""
    df = df.copy()

    leads, bodies, sent_lists = [], [], []
    for _, row in df.iterrows():
        sents = _split_sentences(row["text"])
        lead  = " ".join(sents[:lead_n])
        body  = " ".join(sents[lead_n:])
        leads.append(lead)
        bodies.append(body)
        sent_lists.append(sents)

    df["lead"]      = leads
    df["body"]      = bodies
    df["sentences"] = sent_lists
    return df


# ── 5. Topic clustering ────────────────────────────────────────────────────

def assign_topics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster articles into N_TOPICS topic buckets using TF-IDF + KMeans.
    Falls back gracefully if sentence-transformers is not installed.
    """
    log.info("Fitting topic model (TF-IDF + KMeans, k=%d)…", N_TOPICS)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english",
                                sublinear_tf=True)
        X = tfidf.fit_transform(df["full_text"].tolist())
        X_norm = normalize(X)
        km = KMeans(n_clusters=N_TOPICS, random_state=RANDOM_STATE,
                    n_init=10, max_iter=300)
        df = df.copy()
        df["topic_id"]   = km.fit_predict(X_norm)
        df["topic_name"] = df["topic_id"].map(
            {i: TOPIC_NAMES[i] for i in range(N_TOPICS)}
        )
        model_path = DATA_PROC / "topic_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"tfidf": tfidf, "km": km}, f)
        log.info("Topic model saved to %s", model_path)
    except Exception as exc:
        log.warning("Topic modelling failed (%s); assigning topic 'other'", exc)
        df = df.copy()
        df["topic_id"]   = 0
        df["topic_name"] = "other"

    counts = df.groupby(["topic_name", "label"]).size().unstack(fill_value=0)
    log.info("Topic distribution:\n%s", counts)
    return df


# ── 6. Train / dev / test split ────────────────────────────────────────────

def split_corpus(df: pd.DataFrame,
                 dev_size:  float = 0.10,
                 test_size: float = 0.20) -> pd.DataFrame:
    """
    Date-stratified split: sort by pub_date, then stratified by label+topic.
    Returns df with a new 'split' column (train/dev/test).
    """
    df = df.sort_values("pub_date").reset_index(drop=True)
    strat = df["label"] + "_" + df["topic_id"].astype(str)

    idx_train, idx_temp = train_test_split(
        df.index, test_size=dev_size + test_size,
        stratify=strat, random_state=RANDOM_STATE
    )
    strat_temp = strat.loc[idx_temp]
    rel_test = test_size / (dev_size + test_size)
    idx_dev, idx_test = train_test_split(
        idx_temp, test_size=rel_test,
        stratify=strat_temp, random_state=RANDOM_STATE
    )

    df = df.copy()
    df["split"] = "train"
    df.loc[idx_dev,  "split"] = "dev"
    df.loc[idx_test, "split"] = "test"

    for s in ["train", "dev", "test"]:
        sub = df[df["split"] == s]
        log.info("%-5s: %d articles (%d real, %d fake)",
                 s, len(sub),
                 (sub.label == "real").sum(),
                 (sub.label == "fake").sum())
    return df


# ── 7. Entry point ─────────────────────────────────────────────────────────

def run(start_date: str = "2016-01-01", end_date: str = "2018-01-01") -> pd.DataFrame:
    log.info("=== Phase 1: Preprocessing ===")

    df = load_isot()
    df = clean_corpus(df)
    df = filter_by_date(df, start=start_date, end=end_date)
    df = section_articles(df)
    df = assign_topics(df)
    df = split_corpus(df)

    out_path = DATA_PROC / "corpus.parquet"
    # Store sentences as joined string for parquet compatibility
    df["sentences_json"] = df["sentences"].apply(lambda s: "|||".join(s))
    df_save = df.drop(columns=["sentences"])
    df_save.to_parquet(out_path, index=False)
    log.info("Corpus saved to %s  (%d rows)", out_path, len(df_save))

    return df


if __name__ == "__main__":
    run()
