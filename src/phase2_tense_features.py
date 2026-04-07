"""
Phase 2 – Tense & aspect feature extraction (H2).

For each article and each section (title / lead / body) we extract:
  - Tense distribution  : proportions of past / present / future verbs
  - Tense entropy       : −Σ p log p  over tense proportions
  - Dominant tense per sentence
  - Tense shift rate    : fraction of consecutive sentence pairs where dominant tense changes
  - Sectional alignment : L1 distance between tense distributions of (title vs body, lead vs body)
  - Aspect distribution : proportions of simple / progressive / perfect / perfect-progressive

Input  : data/processed/corpus.csv
Output : data/processed/features_tense.csv
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from src.utils import (
    DATA_PROC, TABLES, RANDOM_STATE, TENSES, ASPECTS,
    PAST, PRESENT, FUTURE,
    SIMPLE, PROGRESSIVE, PERFECT, PERFECT_PROGRESSIVE,
    get_logger,
)

warnings.filterwarnings("ignore")
log = get_logger(__name__)

# ── spaCy model ────────────────────────────────────────────────────────────

def _load_spacy():
    for model in ["en_core_web_trf", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model, disable=["ner"])
            log.info("Loaded spaCy model: %s", model)
            return nlp
        except OSError:
            continue
    raise OSError(
        "No spaCy English model found. Install one with:\n"
        "  python -m spacy download en_core_web_sm"
    )


# ── Tense classification logic ─────────────────────────────────────────────

_FUTURE_MODALS = {"will", "shall", "'ll"}
_PERF_HAVE     = {"have", "has", "had"}
_PROG_BE       = {"be", "am", "is", "are", "was", "were", "been", "being"}
_COND_MODALS   = {"would", "could", "should", "might", "may"}


def classify_verb(token, doc) -> tuple[str, str]:
    """
    Return (tense, aspect) for a finite verb token.
    tense  : past | present | future
    aspect : simple | progressive | perfect | perfect_progressive
    """
    tag = token.tag_   # Penn Treebank POS
    dep = token.dep_

    # Collect left auxiliaries in the same clause
    auxs = [
        c.lemma_.lower()
        for c in token.children
        if c.dep_ in ("aux", "auxpass")
    ]
    # Also include the token's own head if it's aux
    if token.dep_ in ("aux", "auxpass") and token.head.pos_ == "VERB":
        # We're looking at the aux itself; classify the head
        return None, None

    aux_set = set(auxs)

    # ── Aspect ────────────────────────────────────────────────────
    is_perfect_base     = bool(aux_set & _PERF_HAVE)
    is_progressive_base = bool(aux_set & _PROG_BE)

    if tag == "VBG" and is_progressive_base:
        # progressive or perfect-progressive
        if is_perfect_base:
            aspect = PERFECT_PROGRESSIVE
        else:
            aspect = PROGRESSIVE
    elif tag in ("VBN", "VBD") and is_perfect_base and not is_progressive_base:
        aspect = PERFECT
    else:
        aspect = SIMPLE

    # ── Tense ─────────────────────────────────────────────────────
    future_modal = aux_set & _FUTURE_MODALS
    going_to = (
        any(c.lower_ == "going" and c.tag_ == "VBG" for c in token.children)
    )

    if future_modal or going_to:
        tense = FUTURE
    elif tag in ("VBD", "VBN") and not is_perfect_base:
        tense = PAST
    elif tag in ("VBD",):
        tense = PAST
    elif tag in ("VBP", "VBZ"):
        tense = PRESENT
    elif tag == "VB":
        # bare infinitive – check context
        if aux_set & _FUTURE_MODALS:
            tense = FUTURE
        elif aux_set & _COND_MODALS:
            tense = PRESENT   # treat conditional as present-ish
        else:
            tense = PRESENT
    elif tag == "VBG":
        tense = PRESENT   # gerund / progressive base form
    elif tag == "VBN":
        # past participle without clear aux → assume past passive
        tense = PAST
    elif tag == "MD":
        # standalone modal
        lemma = token.lemma_.lower()
        tense = FUTURE if lemma in _FUTURE_MODALS else PRESENT
    else:
        tense = PRESENT  # fallback

    return tense, aspect


def extract_verb_features_from_sents(sents: list[str], nlp) -> dict:
    """
    Process a list of sentences and return all tense/aspect features.
    """
    tense_counts  = {t: 0 for t in TENSES}
    aspect_counts = {a: 0 for a in ASPECTS}
    dominant_per_sent: list[str | None] = []

    for sent in sents:
        if not sent.strip():
            dominant_per_sent.append(None)
            continue

        doc = nlp(sent)
        sent_tenses = {t: 0 for t in TENSES}

        for token in doc:
            if token.pos_ not in ("VERB", "AUX") or token.dep_ in ("aux", "auxpass"):
                continue
            # Skip non-finite tokens (infinitive markers, etc.)
            if token.tag_ not in ("VBD", "VBN", "VBP", "VBZ", "VB", "VBG", "MD"):
                continue

            t, a = classify_verb(token, doc)
            if t is None:
                continue
            tense_counts[t]  += 1
            aspect_counts[a] += 1
            sent_tenses[t]   += 1

        total = sum(sent_tenses.values())
        if total == 0:
            dominant_per_sent.append(None)
        else:
            dominant_per_sent.append(max(sent_tenses, key=sent_tenses.get))

    return tense_counts, aspect_counts, dominant_per_sent


def tense_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {t: 0.0 for t in TENSES}
    return {t: counts[t] / total for t in TENSES}


def tense_entropy(dist: dict[str, float]) -> float:
    vals = np.array([v for v in dist.values() if v > 0])
    return float(-np.sum(vals * np.log(vals + 1e-12)))


def tense_shift_rate(dominant: list[str | None]) -> float:
    valid = [d for d in dominant if d is not None]
    if len(valid) < 2:
        return 0.0
    shifts = sum(a != b for a, b in zip(valid[:-1], valid[1:]))
    return shifts / (len(valid) - 1)


def l1_dist(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    return float(sum(abs(dist_a.get(t, 0) - dist_b.get(t, 0)) for t in TENSES))


# ── Per-article feature computation ────────────────────────────────────────

def compute_article_features(row: pd.Series, nlp) -> dict:
    article_id = row.name
    sents = row["sentences_json"].split("|||") if isinstance(row["sentences_json"], str) else []

    # Determine section boundary
    lead_n = 3
    lead_sents = sents[:lead_n]
    body_sents = sents[lead_n:]

    title_text = row.get("title", "")
    title_sents = [title_text] if title_text else []

    # Extract per-section features
    def _feats(s: list[str], prefix: str) -> dict:
        if not s:
            zero_t = {t: 0.0 for t in TENSES}
            zero_a = {a: 0.0 for a in ASPECTS}
            return {
                **{f"{prefix}_prop_{t}": 0.0 for t in TENSES},
                f"{prefix}_entropy": 0.0,
                f"{prefix}_shift_rate": 0.0,
                **{f"{prefix}_aspect_{a}": 0.0 for a in ASPECTS},
            }
        tc, ac, dom = extract_verb_features_from_sents(s, nlp)
        dist_t = tense_distribution(tc)
        total_a = sum(ac.values()) or 1
        return {
            **{f"{prefix}_prop_{t}": dist_t[t] for t in TENSES},
            f"{prefix}_entropy": tense_entropy(dist_t),
            f"{prefix}_shift_rate": tense_shift_rate(dom),
            **{f"{prefix}_aspect_{a}": ac[a] / total_a for a in ASPECTS},
        }

    title_feats = _feats(title_sents, "title")
    lead_feats  = _feats(lead_sents,  "lead")
    body_feats  = _feats(body_sents,  "body")

    # Whole-article tense features (over all sentences)
    all_tc, all_ac, all_dom = extract_verb_features_from_sents(sents, nlp)
    all_dist = tense_distribution(all_tc)
    total_all_a = sum(all_ac.values()) or 1

    art_feats = {
        **{f"art_prop_{t}": all_dist[t] for t in TENSES},
        "art_entropy":    tense_entropy(all_dist),
        "art_shift_rate": tense_shift_rate(all_dom),
        **{f"art_aspect_{a}": all_ac[a] / total_all_a for a in ASPECTS},
    }

    # Sectional alignment: L1 distance between tense distributions
    title_dist = {t: title_feats[f"title_prop_{t}"] for t in TENSES}
    lead_dist  = {t: lead_feats[f"lead_prop_{t}"]   for t in TENSES}
    body_dist  = {t: body_feats[f"body_prop_{t}"]   for t in TENSES}

    align_feats = {
        "align_title_body": l1_dist(title_dist, body_dist),
        "align_lead_body":  l1_dist(lead_dist,  body_dist),
        "align_title_lead": l1_dist(title_dist, lead_dist),
    }

    return {
        "article_id": article_id,
        **title_feats,
        **lead_feats,
        **body_feats,
        **art_feats,
        **align_feats,
        "label":      row["label"],
        "split":      row["split"],
        "topic_name": row.get("topic_name", "other"),
    }


# ── Entry point ────────────────────────────────────────────────────────────

def run(sample: int | None = None) -> pd.DataFrame:
    log.info("=== Phase 2: Tense & Aspect Feature Extraction ===")

    corpus_path = DATA_PROC / "corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run Phase 1 first."
        )

    df = pd.read_csv(corpus_path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        log.info("Using sample of %d articles", len(df))

    nlp = _load_spacy()

    log.info("Extracting tense/aspect features for %d articles…", len(df))
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tense features"):
        try:
            records.append(compute_article_features(row, nlp))
        except Exception as exc:
            log.debug("Skipping article %s: %s", row.name, exc)

    feat_df = pd.DataFrame(records).set_index("article_id")

    out_path = DATA_PROC / "features_tense.csv"
    feat_df.to_csv(out_path)
    log.info("Tense features saved to %s  (%d rows, %d cols)",
             out_path, len(feat_df), feat_df.shape[1])

    # Quick summary
    for label in ["real", "fake"]:
        sub = feat_df[feat_df["label"] == label]
        log.info(
            "[%s] art_shift_rate=%.3f  art_entropy=%.3f  art_prop_past=%.3f",
            label,
            sub["art_shift_rate"].mean(),
            sub["art_entropy"].mean(),
            sub["art_prop_past"].mean(),
        )

    # Summary measurements CSV
    feat_cols = [c for c in feat_df.columns if c not in {"label", "split", "topic_name"}]
    summary = feat_df.groupby("label")[feat_cols].agg(["mean", "median", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary_path = TABLES / "summary_tense.csv"
    summary.to_csv(summary_path)
    log.info("Tense feature summary saved to %s", summary_path)

    return feat_df


if __name__ == "__main__":
    run()
