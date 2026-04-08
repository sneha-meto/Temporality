"""
Phase 5 – Statistical analysis, hypothesis testing, and classification.

Tests:
  H1 – temporal_incons is higher in fake news (Mann-Whitney U)
  H2 – art_shift_rate, art_entropy higher in fake news (Mann-Whitney U)
  H3 – event_misalignment correlates with lower coherence_emb in fake news
       (Spearman ρ, group difference test)
  H4 – fake news exhibits greater tense misalignment between structural
       sections (title/lead/body) than verified news (Mann-Whitney U)

Classification:
  - Logistic Regression  (temporal features only)
  - Random Forest        (temporal features only)
  - Logistic Regression  (BoW baseline)
  - Compared on accuracy, precision, recall, F1

Outputs saved to results/tables/ and results/figures/.

Input  : data/processed/features_tense.csv
         data/processed/features_temporal.csv
         data/processed/features_coherence.csv
         data/processed/corpus.csv  (for BoW baseline)
Output : results/tables/hypothesis_tests.csv
         results/tables/classification_results.csv
         results/tables/spearman_correlations.csv
         results/figures/*.png
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import DATA_PROC, RESULTS, FIGURES, TABLES, RANDOM_STATE, get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


# ── Load and merge feature matrices ───────────────────────────────────────

def load_features() -> pd.DataFrame:
    paths = {
        "tense":    DATA_PROC / "features_tense.csv",
        "temporal": DATA_PROC / "features_temporal.csv",
        "coherence":DATA_PROC / "features_coherence.csv",
    }
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing feature files: {missing}. "
            "Run Phases 1-4 first (run_pipeline.py)."
        )

    dfs = {}
    for name, path in paths.items():
        df = pd.read_csv(path, index_col=0)
        # Keep only feature cols (drop duplicated metadata)
        meta = [c for c in ["label", "split", "topic_name"] if c in df.columns]
        dfs[name] = df

    # Merge on index (article_id)
    base = dfs["tense"]
    for name in ["temporal", "coherence"]:
        other = dfs[name].drop(
            columns=[c for c in ["label", "split", "topic_name"]
                     if c in dfs[name].columns],
            errors="ignore"
        )
        base = base.join(other, how="inner", rsuffix=f"_{name}")

    log.info("Merged feature matrix: %d articles × %d features", *base.shape)
    return base


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all numeric feature columns (excluding metadata)."""
    meta = {"label", "split", "topic_name", "article_id"}
    return [c for c in df.columns if c not in meta and df[c].dtype in (float, "float64", int, "int64")]


# ── Mann-Whitney U test helper ─────────────────────────────────────────────

def mwu_test(df: pd.DataFrame, feature: str, group_col: str = "label",
             g1: str = "fake", g2: str = "real") -> dict:
    x = df.loc[df[group_col] == g1, feature].dropna()
    y = df.loc[df[group_col] == g2, feature].dropna()
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    # Rank-biserial correlation as effect size
    n1, n2 = len(x), len(y)
    r = 1 - (2 * stat) / (n1 * n2)
    return {
        "feature":   feature,
        "mean_fake": float(x.mean()),
        "mean_real": float(y.mean()),
        "U":         float(stat),
        "p_value":   float(p),
        "effect_r":  float(r),
        "sig":       "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")),
    }


# ── H1 – Temporal inconsistency ────────────────────────────────────────────

def test_H1(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── H1: Temporal Inconsistency ──")
    features = [
        "temporal_incons",
        "abs_date_count",
        "abs_future_ratio",
        "art_adv_total",
    ]
    features = [f for f in features if f in df.columns]
    rows = [mwu_test(df, f) for f in features]
    result = pd.DataFrame(rows)
    log.info("\n%s", result.to_string(index=False))
    return result


# ── H2 – Tense shifts ─────────────────────────────────────────────────────

def test_H2(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── H2: Tense Shifts ──")
    features = [
        "art_shift_rate",
        "art_entropy",
        "art_prop_past",
        "art_prop_present",
        "art_prop_future",
        "art_aspect_simple",
        "art_aspect_perfect",
        "art_aspect_progressive",
    ]
    features = [f for f in features if f in df.columns]
    rows = [mwu_test(df, f) for f in features]
    result = pd.DataFrame(rows)
    log.info("\n%s", result.to_string(index=False))
    return result


# ── H4 – Structural tense misalignment ────────────────────────────────────

def compute_h4_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute convention-based H4 misalignment features.

    The expected journalistic convention is: titles are present-tense,
    bodies and leads shift to past tense for reporting.  Misalignment means
    the body/lead *fails* to make that shift — i.e. stays present-dominant.

    Features
    --------
    body_present_excess  : body_prop_present - body_prop_past
                           Positive  → body is present-heavy  (fake pattern = misaligned)
                           Negative  → body is past-heavy     (real pattern = aligned)
    lead_present_excess  : lead_prop_present - lead_prop_past  (same logic for lead)
    h4_misalign          : mean of body_present_excess and lead_present_excess
                           Summary measure of convention violation.
    """
    df = df.copy()
    if {"body_prop_present", "body_prop_past"}.issubset(df.columns):
        df["body_present_excess"] = df["body_prop_present"] - df["body_prop_past"]
    if {"lead_prop_present", "lead_prop_past"}.issubset(df.columns):
        df["lead_present_excess"] = df["lead_prop_present"] - df["lead_prop_past"]
    if {"body_present_excess", "lead_present_excess"}.issubset(df.columns):
        df["h4_misalign"] = (df["body_present_excess"] + df["lead_present_excess"]) / 2
    return df


def test_H4(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── H4: Structural Tense Misalignment ──")
    features = [
        "body_present_excess",
        "lead_present_excess",
        "h4_misalign",
    ]
    features = [f for f in features if f in df.columns]
    rows = [mwu_test(df, f) for f in features]
    result = pd.DataFrame(rows)
    log.info("\n%s", result.to_string(index=False))
    return result


# ── H3 – Event sequencing & coherence ─────────────────────────────────────

def test_H3(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── H3: Event Sequencing & Coherence ──")
    misalign = "event_misalignment"
    coherence_cols = [c for c in df.columns if c.startswith("coherence_")]

    rows = []
    for coh in coherence_cols:
        for label in ["fake", "real"]:
            sub = df[df["label"] == label][[misalign, coh]].dropna()
            if len(sub) < 10:
                continue
            r, p = stats.spearmanr(sub[misalign], sub[coh])
            rows.append({
                "label":      label,
                "misalign":   misalign,
                "coherence":  coh,
                "spearman_r": float(r),
                "p_value":    float(p),
                "n":          len(sub),
                "sig":        "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")),
            })

    result = pd.DataFrame(rows)
    log.info("\n%s", result.to_string(index=False))
    return result


# ── Classification ─────────────────────────────────────────────────────────

def _scores(y_true, y_pred, y_prob=None) -> dict:
    s = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="fake"),
        "recall":    recall_score(y_true, y_pred, pos_label="fake"),
        "f1":        f1_score(y_true, y_pred, pos_label="fake"),
    }
    if y_prob is not None:
        try:
            s["auc"] = roc_auc_score(
                (np.array(y_true) == "fake").astype(int), y_prob
            )
        except Exception:
            pass
    return s


def classify(df: pd.DataFrame, corpus_df: pd.DataFrame | None = None) -> pd.DataFrame:
    log.info("── Classification ──")

    feat_cols = get_feature_columns(df)
    train = df[df["split"] == "train"]
    test  = df[df["split"] == "test"]

    X_tr = train[feat_cols].fillna(0).values
    y_tr = train["label"].values
    X_te = test[feat_cols].fillna(0).values
    y_te = test["label"].values

    if len(X_te) == 0:
        log.warning("Test set is empty — skipping classification")
        return pd.DataFrame()

    results = []

    # ── Logistic Regression (temporal features) ────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                   class_weight="balanced")),
    ])
    lr_pipe.fit(X_tr, y_tr)
    y_pred = lr_pipe.predict(X_te)
    y_prob = lr_pipe.predict_proba(X_te)[:, list(lr_pipe.classes_).index("fake")]
    s = _scores(y_te, y_pred, y_prob)
    results.append({"model": "LogReg (temporal)", **s})
    log.info("LogReg temporal:\n%s", classification_report(y_te, y_pred))

    # ── Random Forest (temporal features) ─────────────────────────
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, list(rf.classes_).index("fake")]
    s = _scores(y_te, y_pred, y_prob)
    results.append({"model": "RandomForest (temporal)", **s})
    log.info("RandomForest temporal:\n%s", classification_report(y_te, y_pred))

    # ── Feature importance ─────────────────────────────────────────
    importance = pd.Series(rf.feature_importances_, index=feat_cols)
    top = importance.nlargest(20)
    log.info("Top-20 temporal features:\n%s", top.to_string())
    top.to_csv(TABLES / "feature_importance.csv")

    # ── BoW baseline ───────────────────────────────────────────────
    if corpus_df is not None:
        try:
            corp_train = corpus_df[corpus_df["split"] == "train"]
            corp_test  = corpus_df[corpus_df["split"] == "test"]

            # Re-index to match merged df
            idx_tr = train.index.intersection(corp_train.index)
            idx_te = test.index.intersection(corp_test.index)

            if len(idx_te) > 10:
                tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True,
                                        stop_words="english")
                Xb_tr = tfidf.fit_transform(corp_train.loc[idx_tr, "full_text"].fillna(""))
                yb_tr = corp_train.loc[idx_tr, "label"].values
                Xb_te = tfidf.transform(corp_test.loc[idx_te, "full_text"].fillna(""))
                yb_te = corp_test.loc[idx_te, "label"].values

                bow_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                             class_weight="balanced")
                bow_clf.fit(Xb_tr, yb_tr)
                y_pred_bow = bow_clf.predict(Xb_te)
                y_prob_bow = bow_clf.predict_proba(Xb_te)[:, list(bow_clf.classes_).index("fake")]
                s = _scores(yb_te, y_pred_bow, y_prob_bow)
                results.append({"model": "LogReg (BoW baseline)", **s})
                log.info("BoW baseline:\n%s", classification_report(yb_te, y_pred_bow))
        except Exception as exc:
            log.warning("BoW baseline failed: %s", exc)

    result_df = pd.DataFrame(results)
    log.info("\n%s", result_df.to_string(index=False))
    return result_df


# ── Visualisations ─────────────────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, features: list[str],
                                filename: str = "feature_distributions.png"):
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = axes.flatten()

    for ax, feat in zip(axes, features):
        for label, colour in [("fake", "#e74c3c"), ("real", "#2ecc71")]:
            vals = df.loc[df["label"] == label, feat].dropna()
            ax.hist(vals, bins=40, alpha=0.55, color=colour, label=label, density=True)
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Temporal Feature Distributions: Fake vs Real", y=1.01)
    plt.tight_layout()
    out = FIGURES / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_coherence_vs_misalignment(df: pd.DataFrame):
    if "event_misalignment" not in df.columns:
        return
    coh_cols = [c for c in df.columns if c.startswith("coherence_")]
    if not coh_cols:
        return

    fig, axes = plt.subplots(1, len(coh_cols), figsize=(6 * len(coh_cols), 5))
    if len(coh_cols) == 1:
        axes = [axes]

    for ax, coh in zip(axes, coh_cols):
        for label, colour in [("fake", "#e74c3c"), ("real", "#2ecc71")]:
            sub = df[df["label"] == label][["event_misalignment", coh]].dropna()
            ax.scatter(sub["event_misalignment"], sub[coh],
                       alpha=0.25, s=8, color=colour, label=label)
        ax.set_xlabel("Event misalignment score")
        ax.set_ylabel(coh)
        ax.set_title(f"Misalignment vs {coh}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES / "coherence_vs_misalignment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_tense_distributions(df: pd.DataFrame):
    tense_cols = ["art_prop_past", "art_prop_present", "art_prop_future"]
    tense_cols = [c for c in tense_cols if c in df.columns]
    if not tense_cols:
        return

    means = df.groupby("label")[tense_cols].mean().T
    means.index = [c.replace("art_prop_", "").capitalize() for c in means.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(means))
    width = 0.35
    bars_r = ax.bar(x - width / 2, means["real"], width, label="Real",
                    color="#2ecc71", alpha=0.8)
    bars_f = ax.bar(x + width / 2, means["fake"], width, label="Fake",
                    color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Mean proportion")
    ax.set_title("Tense Distribution by Label")
    ax.legend()
    plt.tight_layout()
    out = FIGURES / "tense_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_classification_results(clf_df: pd.DataFrame):
    if clf_df.empty:
        return
    metrics = ["accuracy", "precision", "recall", "f1"]
    metrics = [m for m in metrics if m in clf_df.columns]

    melted = clf_df.melt(id_vars=["model"], value_vars=metrics,
                         var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Classification Performance: Temporal Features vs BoW Baseline")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    out = FIGURES / "classification_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_sectional_alignment(df: pd.DataFrame):
    align_cols = ["body_present_excess", "lead_present_excess", "h4_misalign"]
    align_cols = [c for c in align_cols if c in df.columns]
    if not align_cols:
        return

    means = df.groupby("label")[align_cols].mean().T
    labels_map = {
        "body_present_excess": "Body\n(present−past)",
        "lead_present_excess": "Lead\n(present−past)",
        "h4_misalign":         "Combined\nmisalignment",
    }
    means.index = [labels_map.get(c, c) for c in means.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(means))
    width = 0.35
    ax.bar(x - width / 2, means["real"], width, label="Real", color="#2ecc71", alpha=0.8)
    ax.bar(x + width / 2, means["fake"], width, label="Fake", color="#e74c3c", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Mean (present − past) proportion")
    ax.set_title("H4: Convention-Based Tense Misalignment (body/lead should be past-dominant)")
    ax.legend()
    plt.tight_layout()
    out = FIGURES / "sectional_alignment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_topic_analysis(df: pd.DataFrame):
    if "topic_name" not in df.columns or "art_shift_rate" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, feat in zip(axes, ["art_shift_rate", "temporal_incons"]):
        if feat not in df.columns:
            continue
        sub = df.groupby(["topic_name", "label"])[feat].mean().unstack()
        sub.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"], alpha=0.8)
        ax.set_title(f"{feat} by topic")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Temporal Features by Topic")
    plt.tight_layout()
    out = FIGURES / "topic_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


# ── Entry point ────────────────────────────────────────────────────────────

def run() -> None:
    log.info("=== Phase 5: Statistical Analysis & Classification ===")

    df = load_features()
    df = compute_h4_features(df)

    # Use test set for hypothesis tests; train+dev+test for correlations
    test_df  = df[df["split"] == "test"]
    all_df   = df

    # ── Hypothesis tests ──────────────────────────────────────────
    h1 = test_H1(test_df)
    h2 = test_H2(test_df)
    h3 = test_H3(all_df)
    h4 = test_H4(test_df)

    h1["hypothesis"] = "H1"
    h2["hypothesis"] = "H2"
    h4["hypothesis"] = "H4"
    hyp = pd.concat([h1, h2, h4], ignore_index=True)
    hyp.to_csv(TABLES / "hypothesis_tests.csv", index=False)
    log.info("Hypothesis tests saved to %s", TABLES / "hypothesis_tests.csv")

    h3.to_csv(TABLES / "spearman_correlations.csv", index=False)
    log.info("Spearman correlations saved to %s", TABLES / "spearman_correlations.csv")

    # ── Classification ────────────────────────────────────────────
    try:
        corpus_df = pd.read_csv(DATA_PROC / "corpus.csv")
    except Exception:
        corpus_df = None

    clf_df = classify(df, corpus_df)
    if not clf_df.empty:
        clf_df.to_csv(TABLES / "classification_results.csv", index=False)
        log.info("Classification results saved to %s", TABLES / "classification_results.csv")

    # ── Plots ─────────────────────────────────────────────────────
    h2_features = [
        "art_shift_rate", "art_entropy", "art_prop_past", "art_prop_present",
    ]
    h1_features = ["temporal_incons", "conn_density", "conn_diversity",
                   "event_misalignment", "abs_date_count"]
    vis_feats = [f for f in h2_features + h1_features if f in all_df.columns]

    plot_feature_distributions(all_df, vis_feats)
    plot_coherence_vs_misalignment(all_df)
    plot_tense_distributions(all_df)
    plot_sectional_alignment(all_df)
    plot_topic_analysis(all_df)
    if not clf_df.empty:
        plot_classification_results(clf_df)

    log.info("=== Phase 5 complete. Results in %s ===", RESULTS)


if __name__ == "__main__":
    run()
