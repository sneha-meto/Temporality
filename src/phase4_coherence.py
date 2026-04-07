"""
Phase 4 – Coherence scoring (H3).

Two coherence measures per article:
  1. Embedding-based coherence
     – Encode each sentence with a sentence-transformer.
     – Compute cosine similarity between every adjacent sentence pair.
     – coherence_emb      = mean similarity
     – coherence_emb_std  = std of similarities (lower = more uniform flow)
     – coherence_emb_min  = minimum similarity (captures worst transitions)

  2. Entity-based coherence (simplified entity grid)
     – Extract named entities (spaCy) per sentence.
     – For each entity, record its presence pattern across sentences.
     – entity_repetition_rate  = mean entity mention density across sentences
     – entity_unique_ratio     = unique entities / total entity mentions
     – entity_grid_ss          = proportion of S→S transitions (entity keeps subject)

Input  : data/processed/corpus.csv
Output : data/processed/features_coherence.csv
"""

import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from src.utils import DATA_PROC, TABLES, RANDOM_STATE, get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)


# ── Model loading ──────────────────────────────────────────────────────────

def _load_spacy():
    for model in ["en_core_web_trf", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model)
            log.info("Loaded spaCy model: %s", model)
            return nlp
        except OSError:
            continue
    raise OSError(
        "No spaCy English model found. Run: python -m spacy download en_core_web_sm"
    )


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Loaded sentence-transformer: all-MiniLM-L6-v2")
        return model
    except ImportError:
        log.warning(
            "sentence-transformers not installed. "
            "Embedding coherence will be set to NaN. "
            "Install with: pip install sentence-transformers"
        )
        return None


# ── Embedding-based coherence ──────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embedding_coherence(sents: list[str], embedder) -> dict:
    if embedder is None or len(sents) < 2:
        return {
            "coherence_emb":     float("nan"),
            "coherence_emb_std": float("nan"),
            "coherence_emb_min": float("nan"),
        }

    sents_clean = [s for s in sents if s.strip()]
    if len(sents_clean) < 2:
        return {
            "coherence_emb":     float("nan"),
            "coherence_emb_std": float("nan"),
            "coherence_emb_min": float("nan"),
        }

    embs = embedder.encode(sents_clean, show_progress_bar=False,
                           convert_to_numpy=True)
    sims = [cosine_sim(embs[i], embs[i + 1]) for i in range(len(embs) - 1)]

    return {
        "coherence_emb":     float(np.mean(sims)),
        "coherence_emb_std": float(np.std(sims)),
        "coherence_emb_min": float(np.min(sims)),
    }


# ── Entity-based coherence ─────────────────────────────────────────────────

# Grammatical role labels from spaCy dep tags
_SUBJ_DEPS  = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DEPS   = {"dobj", "iobj", "pobj", "attr"}
_OTHER_DEPS = {"appos", "conj", "nmod", "compound"}   # X in Barzilay & Lapata


def _entity_role(ent, doc) -> str:
    """Return S / O / X for an entity span based on its syntactic head."""
    heads = {token.dep_ for token in ent}
    if heads & _SUBJ_DEPS:
        return "S"
    if heads & _OBJ_DEPS:
        return "O"
    if heads & _OTHER_DEPS:
        return "X"
    return "-"


def entity_coherence(sents: list[str], nlp) -> dict:
    if len(sents) < 2:
        return {
            "coherence_ent_rep":    0.0,
            "coherence_ent_unique": 0.0,
            "coherence_ent_ss":     0.0,
            "coherence_ent_so":     0.0,
        }

    # Build entity grid: entity_text → list of (sent_idx, role)
    entity_grid: dict[str, list[tuple[int, str]]] = defaultdict(list)
    total_mentions = 0

    for sent_idx, sent in enumerate(sents):
        if not sent.strip():
            continue
        doc = nlp(sent)
        for ent in doc.ents:
            canonical = ent.text.lower().strip()
            role = _entity_role(ent, doc)
            entity_grid[canonical].append((sent_idx, role))
            total_mentions += 1

    if not entity_grid or total_mentions == 0:
        return {
            "coherence_ent_rep":    0.0,
            "coherence_ent_unique": 0.0,
            "coherence_ent_ss":     0.0,
            "coherence_ent_so":     0.0,
        }

    # Entity repetition rate = entities that appear in > 1 sentence / total unique entities
    multi_sent = sum(
        1 for v in entity_grid.values()
        if len({i for i, _ in v}) > 1
    )
    unique_ents = len(entity_grid)
    rep_rate    = multi_sent / unique_ents

    # Unique entity ratio = unique entities / total mentions
    unique_ratio = unique_ents / total_mentions

    # Grid transition probabilities (S→S, S→O)
    # For each entity appearing in consecutive sentences, count role transitions
    ss_count = so_count = trans_count = 0

    for mentions in entity_grid.values():
        by_sent = defaultdict(list)
        for sent_idx, role in mentions:
            by_sent[sent_idx].append(role)

        sent_indices = sorted(by_sent)
        for a, b in zip(sent_indices[:-1], sent_indices[1:]):
            if b - a == 1:   # only consecutive sentences
                role_a = by_sent[a][0]   # first role in that sentence
                role_b = by_sent[b][0]
                trans_count += 1
                if role_a == "S" and role_b == "S":
                    ss_count += 1
                elif role_a == "S" and role_b == "O":
                    so_count += 1

    denom = trans_count or 1
    return {
        "coherence_ent_rep":    rep_rate,
        "coherence_ent_unique": unique_ratio,
        "coherence_ent_ss":     ss_count / denom,
        "coherence_ent_so":     so_count / denom,
    }


# ── Per-article feature computation ────────────────────────────────────────

def compute_article_features(row: pd.Series, nlp, embedder) -> dict:
    sents = row["sentences_json"].split("|||") if isinstance(row["sentences_json"], str) else []
    sents = [s for s in sents if s.strip()]

    emb_feats = embedding_coherence(sents, embedder)
    ent_feats = entity_coherence(sents, nlp)

    return {
        "article_id": row.name,
        **emb_feats,
        **ent_feats,
        "label":      row["label"],
        "split":      row["split"],
        "topic_name": row.get("topic_name", "other"),
    }


# ── Entry point ────────────────────────────────────────────────────────────

def run(sample: int | None = None) -> pd.DataFrame:
    log.info("=== Phase 4: Coherence Scoring ===")

    corpus_path = DATA_PROC / "corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found. Run Phase 1 first.")

    df = pd.read_csv(corpus_path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
        log.info("Using sample of %d articles", len(df))

    nlp      = _load_spacy()
    embedder = _load_embedder()

    log.info("Computing coherence scores for %d articles…", len(df))
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Coherence"):
        try:
            records.append(compute_article_features(row, nlp, embedder))
        except Exception as exc:
            log.debug("Skipping article %s: %s", row.name, exc)

    feat_df = pd.DataFrame(records).set_index("article_id")

    out_path = DATA_PROC / "features_coherence.csv"
    feat_df.to_csv(out_path)
    log.info("Coherence features saved to %s  (%d rows, %d cols)",
             out_path, len(feat_df), feat_df.shape[1])

    for label in ["real", "fake"]:
        sub = feat_df[feat_df["label"] == label]
        log.info(
            "[%s] coherence_emb=%.3f  coherence_ent_rep=%.3f  coherence_ent_ss=%.3f",
            label,
            sub["coherence_emb"].mean(),
            sub["coherence_ent_rep"].mean(),
            sub["coherence_ent_ss"].mean(),
        )

    # Summary measurements CSV
    feat_cols = [c for c in feat_df.columns if c not in {"label", "split", "topic_name"}]
    summary = feat_df.groupby("label")[feat_cols].agg(["mean", "median", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(TABLES / "summary_coherence.csv")
    log.info("Coherence feature summary saved to %s", TABLES / "summary_coherence.csv")

    return feat_df


if __name__ == "__main__":
    run()
