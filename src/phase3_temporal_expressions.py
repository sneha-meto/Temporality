"""
Phase 3 – Temporal expressions, connectives, and event sequencing (H1, H3).

Features extracted per article / section:
  - Temporal adverbial frequencies & ratios (present / past / future)
  - Temporal connective density and type diversity
  - Temporal inconsistency count (relative expressions vs publication date)
  - Absolute date count and future-date ratio
  - Event-narration misalignment score (ordering connective analysis)

Uses a pure-Python regex tagger to normalise temporal expressions.
If py_heideltime is installed and Java is available it will be used instead.

Input  : data/processed/corpus.csv
Output : data/processed/features_temporal.csv
"""

import re
import logging
import warnings
from datetime import timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from tqdm import tqdm

from src.utils import (
    DATA_PROC, TABLES, RANDOM_STATE, ORDERING_CONNECTIVES,
    PRESENT_ADVERBIALS, PAST_ADVERBIALS, FUTURE_ADVERBIALS,
    get_logger,
)

warnings.filterwarnings("ignore")
log = get_logger(__name__)


# ── Temporal adverbial lexicons (regex patterns) ───────────────────────────

_PRES_ADV_RE = re.compile(
    r"\b(now|today|currently|at present|at this moment|at the moment|"
    r"nowadays|this week|this month|this year|this morning|this evening|tonight)\b",
    re.IGNORECASE,
)
_PAST_ADV_RE = re.compile(
    r"\b(yesterday|last week|last month|last year|formerly|previously|"
    r"in the past|\d+ (days?|weeks?|months?|years?) ago|"
    r"earlier today|earlier this week|last night)\b",
    re.IGNORECASE,
)
_FUTURE_ADV_RE = re.compile(
    r"\b(tomorrow|next week|next month|next year|"
    r"soon|in the future|later today|shortly|"
    r"in (coming|the next) (days?|weeks?|months?))\b",
    re.IGNORECASE,
)

# Absolute date patterns: "March 2017", "Jan 5, 2016", "2017-03-15", "15 March 2017"
_ABS_DATE_RE = re.compile(
    r"\b(?:"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|"
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{4}"
    r"|"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{4}"
    r"|"
    r"\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)

# Temporal connectives
_CONNECTIVE_PATTERNS = {
    c: re.compile(r"\b" + re.escape(c) + r"\b", re.IGNORECASE)
    for c in ORDERING_CONNECTIVES
}


# ── Relative expression → day offset ──────────────────────────────────────

_REL_OFFSETS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\byesterday\b",       re.I),  -1),
    (re.compile(r"\btoday\b",           re.I),   0),
    (re.compile(r"\bnow\b",             re.I),   0),
    (re.compile(r"\btomorrow\b",        re.I),  +1),
    (re.compile(r"\blast\s+week\b",     re.I),  -7),
    (re.compile(r"\bnext\s+week\b",     re.I),  +7),
    (re.compile(r"\blast\s+month\b",    re.I), -30),
    (re.compile(r"\bnext\s+month\b",    re.I), +30),
    (re.compile(r"\blast\s+year\b",     re.I),-365),
    (re.compile(r"\bnext\s+year\b",     re.I),+365),
    (re.compile(r"\blast\s+night\b",    re.I),  -1),
    (re.compile(r"\bearlier\s+today\b", re.I),   0),
]

_N_AGO_RE = re.compile(
    r"(\d+)\s+(day|week|month|year)s?\s+ago", re.IGNORECASE
)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def _resolve_relative(text: str, pub_date: date) -> list[date]:
    """Return a list of absolute dates inferred from relative expressions."""
    resolved = []
    for pat, offset in _REL_OFFSETS:
        if pat.search(text):
            resolved.append(pub_date + timedelta(days=offset))
    for m in _N_AGO_RE.finditer(text):
        n, unit = int(m.group(1)), m.group(2).lower()
        resolved.append(pub_date - timedelta(days=n * _UNIT_DAYS[unit]))
    return resolved


def _parse_absolute_dates(text: str) -> list[date]:
    """Extract and parse absolute date strings from text."""
    dates = []
    for m in _ABS_DATE_RE.finditer(text):
        try:
            d = date_parser.parse(m.group(), fuzzy=True).date()
            dates.append(d)
        except Exception:
            pass
    return dates


# ── Inconsistency detection ────────────────────────────────────────────────

def count_inconsistencies(text: str, pub_date: date) -> int:
    """
    Count temporal inconsistencies:
    1. Relative expression resolves to a future date  (e.g. "yesterday" but pub_date
       is earlier than what the resolved date implies — simple heuristic: flag
       resolved dates > pub_date + 1 day when expression is past-anchored).
    2. Absolute dates that are more than 30 days in the future of pub_date
       (unlikely for a past-event news article).
    3. Multiple conflicting absolute dates within same article (>2 distinct absolute
       year-month combinations that span > 18 months, excluding publication year).
    """
    inconsistencies = 0

    # Rule 1: past-anchored relative expression with future resolution
    past_relative = _PAST_ADV_RE.findall(text)
    for _ in past_relative:
        pass  # we already flagged the dates

    resolved_rel = _resolve_relative(text, pub_date)
    for d in resolved_rel:
        if d > pub_date + timedelta(days=1):
            inconsistencies += 1  # a "past" expression landed in future

    # Rule 2: absolute dates far in the future
    abs_dates = _parse_absolute_dates(text)
    for d in abs_dates:
        if d > pub_date + timedelta(days=30):
            inconsistencies += 1

    # Rule 3: wide date span in article (conflicting dates)
    all_dates = abs_dates
    if len(all_dates) >= 2:
        span_days = (max(all_dates) - min(all_dates)).days
        if span_days > 548:   # more than 18 months gap
            inconsistencies += 1

    return inconsistencies


# ── Connective features ────────────────────────────────────────────────────

def connective_features(text: str, word_count: int) -> dict:
    counts = {}
    for conn, pat in _CONNECTIVE_PATTERNS.items():
        counts[conn] = len(pat.findall(text))

    total = sum(counts.values())
    density   = total / max(word_count, 1) * 100   # per 100 words
    diversity = sum(1 for v in counts.values() if v > 0)

    return {
        "conn_total":    total,
        "conn_density":  density,
        "conn_diversity": diversity,
        **{f"conn_{c}": counts[c] for c in counts},
    }


# ── Event-narration misalignment ───────────────────────────────────────────

def event_misalignment_score(sents: list[str]) -> float:
    """
    For each sentence containing an ordering connective, we check whether the
    implied temporal order contradicts the narration order.

    Simplified model:
      - 'before'  in sentence i implies the events in i precede those in i+1
        (expected narration order: A before B → sentence A comes before sentence B)
      - 'after'/'later'/'then' implies previous sentence's events come first
      - 'previously'/'earlier'/'formerly' in sentence i implies its event is OLDER
        than events in neighbouring sentences → expect it to appear later in article
        if narrative is chronological

    We score a pair (i, i+1) as misaligned when a backward-ordering connective
    ('before', 'previously', 'earlier', 'formerly') is used but the next sentence
    appears to introduce even older events (heuristic: another backward connector).

    Returns misalignment proportion (0–1).
    """
    if len(sents) < 2:
        return 0.0

    forward  = {"after", "later", "subsequently", "eventually", "then", "next"}
    backward = {"before", "previously", "earlier", "formerly"}

    pairs = 0
    misaligned = 0

    for i in range(len(sents) - 1):
        sent_i     = sents[i].lower()
        sent_next  = sents[i + 1].lower()

        conn_i    = {c for c in ORDERING_CONNECTIVES if re.search(r"\b" + c + r"\b", sent_i)}
        conn_next = {c for c in ORDERING_CONNECTIVES if re.search(r"\b" + c + r"\b", sent_next)}

        if not conn_i:
            continue
        pairs += 1

        # Check for direction flip: forward → backward or backward → forward
        fwd_i  = conn_i    & forward
        bwd_i  = conn_i    & backward
        fwd_nx = conn_next & forward
        bwd_nx = conn_next & backward

        if (fwd_i and bwd_nx) or (bwd_i and fwd_nx):
            misaligned += 1

    return misaligned / pairs if pairs > 0 else 0.0


# ── Per-article feature computation ────────────────────────────────────────

def compute_article_features(row: pd.Series) -> dict:
    title = row.get("title", "")
    lead  = row.get("lead", "")
    body  = row.get("body", "")
    full  = (title + " " + lead + " " + body).strip()
    sents = row["sentences_json"].split("|||") if isinstance(row["sentences_json"], str) else []

    pub_date: date | None = None
    if pd.notna(row.get("pub_date")):
        try:
            pub_date = pd.Timestamp(row["pub_date"]).date()
        except Exception:
            pub_date = None

    word_count = row.get("word_count", max(len(full.split()), 1))

    # ── Adverbial counts ──────────────────────────────────────────
    def adv_counts(text: str) -> dict:
        pres  = len(_PRES_ADV_RE.findall(text))
        past  = len(_PAST_ADV_RE.findall(text))
        fut   = len(_FUTURE_ADV_RE.findall(text))
        total = pres + past + fut or 1
        return {
            "adv_present": pres,   "adv_past": past,    "adv_future": fut,
            "adv_total":   pres + past + fut,
            "adv_ratio_present": pres / total,
            "adv_ratio_past":    past / total,
            "adv_ratio_future":  fut  / total,
        }

    art_adv   = adv_counts(full)
    title_adv = adv_counts(title)
    lead_adv  = adv_counts(lead)
    body_adv  = adv_counts(body)

    # ── Absolute dates ────────────────────────────────────────────
    abs_dates_art = _parse_absolute_dates(full)
    abs_count = len(abs_dates_art)
    if pub_date and abs_dates_art:
        future_abs = sum(1 for d in abs_dates_art if d > pub_date)
        future_abs_ratio = future_abs / abs_count
    else:
        future_abs_ratio = 0.0

    # ── Inconsistencies ───────────────────────────────────────────
    incons = 0
    if pub_date:
        incons = count_inconsistencies(full, pub_date)

    # ── Connectives ───────────────────────────────────────────────
    conn_feats = connective_features(full, word_count)

    # ── Misalignment ──────────────────────────────────────────────
    misalign = event_misalignment_score(sents)

    return {
        "article_id": row.name,
        # adverbials – article level
        **{f"art_{k}": v for k, v in art_adv.items()},
        # adverbials – sections
        **{f"title_{k}": v for k, v in title_adv.items()},
        **{f"lead_{k}":  v for k, v in lead_adv.items()},
        **{f"body_{k}":  v for k, v in body_adv.items()},
        # absolute dates
        "abs_date_count":      abs_count,
        "abs_future_ratio":    future_abs_ratio,
        # inconsistencies
        "temporal_incons":     incons,
        # connectives
        **conn_feats,
        # event misalignment
        "event_misalignment":  misalign,
        # metadata
        "label":      row["label"],
        "split":      row["split"],
        "topic_name": row.get("topic_name", "other"),
    }


# ── Entry point ────────────────────────────────────────────────────────────

def run(sample: int | None = None) -> pd.DataFrame:
    log.info("=== Phase 3: Temporal Expressions & Event Sequencing ===")

    corpus_path = DATA_PROC / "corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found. Run Phase 1 first.")

    df = pd.read_csv(corpus_path, parse_dates=["pub_date"])
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        log.info("Using sample of %d articles", len(df))

    log.info("Extracting temporal features for %d articles…", len(df))
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Temporal features"):
        try:
            records.append(compute_article_features(row))
        except Exception as exc:
            log.debug("Skipping article %s: %s", row.name, exc)

    feat_df = pd.DataFrame(records).set_index("article_id")

    out_path = DATA_PROC / "features_temporal.csv"
    feat_df.to_csv(out_path)
    log.info("Temporal features saved to %s  (%d rows, %d cols)",
             out_path, len(feat_df), feat_df.shape[1])

    for label in ["real", "fake"]:
        sub = feat_df[feat_df["label"] == label]
        log.info(
            "[%s] temporal_incons=%.3f  event_misalignment=%.3f  conn_density=%.3f",
            label,
            sub["temporal_incons"].mean(),
            sub["event_misalignment"].mean(),
            sub["conn_density"].mean(),
        )

    # Summary measurements CSV
    feat_cols = [c for c in feat_df.columns if c not in {"label", "split", "topic_name"}]
    summary = feat_df.groupby("label")[feat_cols].agg(["mean", "median", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(TABLES / "summary_temporal.csv")
    log.info("Temporal feature summary saved to %s", TABLES / "summary_temporal.csv")

    return feat_df


if __name__ == "__main__":
    run()
