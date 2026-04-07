"""
run_pipeline.py – Orchestrates all five phases of the Temporality project.

Usage
─────
  # Full pipeline (slow – processes all articles through spaCy)
  python run_pipeline.py

  # Quick smoke-test on a small sample
  python run_pipeline.py --sample 500

  # Skip phases already done
  python run_pipeline.py --skip-phase1 --skip-phase2

  # Run a single phase
  python run_pipeline.py --only phase3

Flags
─────
  --sample N        Process only N randomly sampled articles (for testing)
  --skip-phase1     Skip Phase 1 (assumes corpus.csv already exists)
  --skip-phase2     Skip Phase 2 (assumes features_tense.csv already exists)
  --skip-phase3     Skip Phase 3 (assumes features_temporal.csv already exists)
  --skip-phase4     Skip Phase 4 (assumes features_coherence.csv already exists)
  --skip-phase5     Skip Phase 5 (analysis only)
  --only PHASE      Run only the specified phase (phase1..phase5)
"""

import argparse
import sys
import time
import logging


def parse_args():
    p = argparse.ArgumentParser(description="Temporality pipeline runner")
    p.add_argument("--sample",       type=int,  default=None,
                   help="Number of articles to sample (default: all)")
    p.add_argument("--skip-phase1",  action="store_true")
    p.add_argument("--skip-phase2",  action="store_true")
    p.add_argument("--skip-phase3",  action="store_true")
    p.add_argument("--skip-phase4",  action="store_true")
    p.add_argument("--skip-phase5",  action="store_true")
    p.add_argument("--only",         type=str,  default=None,
                   choices=["phase1", "phase2", "phase3", "phase4", "phase5"])
    return p.parse_args()


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("pipeline")

    # Determine which phases to run
    run = {f"phase{i}": True for i in range(1, 6)}
    if args.only:
        run = {f"phase{i}": False for i in range(1, 6)}
        run[args.only] = True
    else:
        for i in range(1, 6):
            if getattr(args, f"skip_phase{i}", False):
                run[f"phase{i}"] = False

    total_start = time.time()

    # ── Phase 1 ────────────────────────────────────────────────────
    if run["phase1"]:
        log.info("━━━ Phase 1: Preprocessing ━━━")
        t0 = time.time()
        from src.phase1_preprocessing import run as p1
        p1()
        log.info("Phase 1 done in %s", _hms(time.time() - t0))
    else:
        log.info("Skipping Phase 1")

    # ── Phase 2 ────────────────────────────────────────────────────
    if run["phase2"]:
        log.info("━━━ Phase 2: Tense & Aspect Features ━━━")
        t0 = time.time()
        from src.phase2_tense_features import run as p2
        p2(sample=args.sample)
        log.info("Phase 2 done in %s", _hms(time.time() - t0))
    else:
        log.info("Skipping Phase 2")

    # ── Phase 3 ────────────────────────────────────────────────────
    if run["phase3"]:
        log.info("━━━ Phase 3: Temporal Expressions & Event Sequencing ━━━")
        t0 = time.time()
        from src.phase3_temporal_expressions import run as p3
        p3(sample=args.sample)
        log.info("Phase 3 done in %s", _hms(time.time() - t0))
    else:
        log.info("Skipping Phase 3")

    # ── Phase 4 ────────────────────────────────────────────────────
    if run["phase4"]:
        log.info("━━━ Phase 4: Coherence Scoring ━━━")
        t0 = time.time()
        from src.phase4_coherence import run as p4
        p4(sample=args.sample)
        log.info("Phase 4 done in %s", _hms(time.time() - t0))
    else:
        log.info("Skipping Phase 4")

    # ── Phase 5 ────────────────────────────────────────────────────
    if run["phase5"]:
        log.info("━━━ Phase 5: Statistical Analysis & Classification ━━━")
        t0 = time.time()
        from src.phase5_analysis import run as p5
        p5()
        log.info("Phase 5 done in %s", _hms(time.time() - t0))
    else:
        log.info("Skipping Phase 5")

    log.info("━━━ Pipeline complete in %s ━━━", _hms(time.time() - total_start))
    log.info("Results → results/tables/  and  results/figures/")


if __name__ == "__main__":
    main()
