"""Shared utilities: paths, logging, constants."""

import logging
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

for _p in [DATA_RAW, DATA_PROC, RESULTS, FIGURES, TABLES]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

# ── Constants ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Tense labels
PAST    = "past"
PRESENT = "present"
FUTURE  = "future"
TENSES  = [PAST, PRESENT, FUTURE]

# Aspect labels
SIMPLE              = "simple"
PROGRESSIVE         = "progressive"
PERFECT             = "perfect"
PERFECT_PROGRESSIVE = "perfect_progressive"
ASPECTS = [SIMPLE, PROGRESSIVE, PERFECT, PERFECT_PROGRESSIVE]

# Temporal connectives that encode ordering
ORDERING_CONNECTIVES = {
    "before":       "B_BEFORE_A",   # sentence mentions A; 'before' implies A precedes B
    "after":        "A_BEFORE_B",
    "later":        "A_BEFORE_B",
    "subsequently": "A_BEFORE_B",
    "eventually":   "A_BEFORE_B",
    "previously":   "B_BEFORE_A",
    "earlier":      "B_BEFORE_A",
    "formerly":     "B_BEFORE_A",
    "then":         "A_BEFORE_B",
    "next":         "A_BEFORE_B",
    "meanwhile":    "CONCURRENT",
    "simultaneously":"CONCURRENT",
}

# Temporal adverbial lexicons
PRESENT_ADVERBIALS = {
    "now", "today", "currently", "at present", "at this moment",
    "at the moment", "nowadays", "this week", "this month", "this year",
    "this morning", "this evening", "tonight",
}
PAST_ADVERBIALS = {
    "yesterday", "last week", "last month", "last year",
    "formerly", "previously", "in the past", "ago",
    "earlier today", "earlier this week", "last night",
}
FUTURE_ADVERBIALS = {
    "tomorrow", "next week", "next month", "next year",
    "soon", "in the future", "later today", "shortly",
    "in coming days", "in coming weeks",
}

# Topic labels for KMeans
N_TOPICS = 5
TOPIC_NAMES = ["politics_us", "world_intl", "economy_finance", "social_crime", "other"]
