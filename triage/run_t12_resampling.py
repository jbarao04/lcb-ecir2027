"""
t12_resampling.py
=================

Correction-policy budget sweep and paired query resampling for
"When Can an LLM-Judge Be Trusted?".

Produces every number in the budget-to-first-touch table plus bootstrap
confidence intervals for the endpoints, thresholds, areas and structural
correlations.

--------------------------------------------------------------------------
WHAT CHANGED RELATIVE TO THE PREVIOUS VERSION
--------------------------------------------------------------------------

C1. GAIN CONVENTION.  Everything is linear gain, gain(g) = g.
    The old `scores_oracle_error` used 2^g - 1, which is
    inconsistent with the evaluation and with the eps^T C eps identity.

C2. LEVERAGE NO LONGER READS THE GOLD RANKING.  The old code set
    `target_systems = gold_rank_full[:20]`, i.e. the human leaderboard.
    The method estimates the top twenty "from the grades available at
    the time".  `leverage` is now a genuinely adaptive policy:
    it starts from the all-LLM leaderboard and recomputes the target set
    after every batch.

C3. AREA SIGN BUG.  `ql_triage_sweep_boot` emits budgets in DESCENDING
    order.  `np.trapezoid` over a descending axis returns a negated area,
    and `np.interp` requires an increasing `xp`.  Both query-level oracle
    areas were wrong.  `area_between` now sorts both curves ascending.

C4. LARA IS NOW LARA.  Takehi et al. (SIGIR 2025) Sec. 4.2: "Logistic
    regression is used as the calibration model for LARA."  The old code
    used a bespoke ordinal torch net with an extra 0.1 * MSE pull toward
    the raw softmax, and applied a softmax over an already-normalised
    probability vector inside `_from_ordinal`.  Replaced with per-level
    logistic regression on pi, refit after every batch, selection by the
    CALIBRATED margin, and the remainder imputed with argmax of the
    calibrated distribution (Algorithm 1, lines 11-19).

C5. `naive` IS SEPARATED FROM `lara`.  The paper describes the
    judge-only policy as the raw margin between the top two grade
    probabilities.  That is LARA's *Naive* baseline (their Sec. 3.2), not
    LARA (their Sec. 3.3).  Both are run.  Decide which one Table
    `tab:budget-first-touch` should carry, and label it accordingly.

C6. MTF IS NOW CORMACK ET AL. (SIGIR 1998).  Their Sec. 6: a run that
    yields a relevant document has "its priority set to the maximum",
    otherwise "its priority is reduced".  The old deque rotation demoted a
    run to the very back on a single miss, which is a much larger
    demotion.  Priorities are now numeric.  Their rule that a document
    "previously judged relevant because it appeared in some other
    submission" also counts as a hit is implemented, without spending
    budget.  MTF is LOCAL (per topic) with round-robin over topics, which
    is the variant that "ensures that each topic receives a comparable
    number of judgements"; global MTF would exhaust one topic before
    touching the next and is meaningless for a per-query leaderboard.

C7. NEW POLICIES: `product_raw` and `product_cal`, the two empty rows of
    the table.  Both are adaptive and use the same top-20 leverage as the
    `leverage` row, so the three are directly comparable.

C8. CALIBRATED ERROR ESTIMATOR.  P(g_h | g_L, bin) confusion table, fitted
    leave-one-query-out within year, never across the v1/v2 boundary.
    Bin edges come from max_prob quantiles, which use no labels.  LOQO is
    exact and cheap because counts are additive: subtract the held-out
    query's counts from the global table.  Backoff 3 bins -> 2 bins ->
    grade only when cells fall below 20 examples, per Sec. 5.1.

C9. STRUCTURAL CORRELATIONS.  `score_bias_vs_score_shift_pearson` was
    computed with `spearmanr`.  Now Pearson, with Spearman reported
    alongside.  The agreement-vs-damage row now uses the Pearson target
    (the primary target) with the Spearman target as a robustness check.

C10. SPEED.  IDCG was recomputed once per (query, step, SYSTEM).  It
    depends only on the query and the step.  Now computed once per
    (query, step) and shared across systems.  DCG is updated
    incrementally, since a correction only moves a system whose top ten
    contains the corrected passage.  Leverage uses a sparse weight matrix.

C11. TIE-BREAKING.  Roughly 85-90 percent of pairs have C_pp = 0.  Sorting
    by leverage leaves an enormous tie group.  All static and adaptive
    scores are tie-broken by a fixed random key, so ties behave like
    random selection rather than like passage-id order.

C12. BOTH THRESHOLDS.  First touch (the table) and sustained (the
    appendix) are reported side by side.

C13. PAIRS OUTSIDE THE LLM SCORING SET are frozen at their human grade
    rather than dropped, so the mixed-grade leaderboard and the gold
    leaderboard use the same passage set.  The count is reported.

C14. MINIMAL TEST COLLECTIONS (Carterette, Allan and Sitaraman, SIGIR 2006)
    IS NOW A BASELINE.  Their Sec. 3 writes AP as a quadratic form
    AP = (1/|R|) sum_i sum_{j>=i} a_ij x_i x_j, with a_ij =
    1/max{rank(i), rank(j)}; the cross-terms exist only because of the
    1/|R| normaliser.  DCG@10 has no such normaliser and is LINEAR in the
    per-passage gains, so c_ij = 0 for i != j and their Algorithm 1 weight
    w_i^R = c_ii p + sum_{j in S} c_ij p^2 collapses to c_ii p.  Their
    selection rule max{p_i w_i^R, (1 - p_i) w_i^N} then reduces to
    c_ii * max{p_i^2, (1 - p_i)^2}, and under the neutral assumption they
    state ("if i is unjudged, p_i = 0.5") that factor is a constant 0.25.
    So on DCG@10 MTC ranks purely by c_ii.

    For a pair of systems c_ii = w_sp - w_tp.  Their multi-system rule is
    verbatim: "To extend that to ranking a set of systems, we use the
    document with the max weight in all pairs of systems", i.e.
    max_{s,t} |w_sp - w_tp| = max_s w_sp - min_s w_sp, the RANGE of the
    weight vector across the target systems.  Our leverage C_pp is the
    VARIANCE of that same vector.  Same object, different moment, so the
    comparison must be made.

    Two consequences, both asserted by the code rather than assumed.
    First, MTC is STATIC here: once the cross-terms vanish the weights no
    longer depend on any judgment, so there is nothing to update as labels
    arrive.  `mtc_range_all` is therefore the faithful published method.
    Second, because every w_sp > 0, min_s w_sp is zero for any passage
    that is not inside EVERY target system's top ten, so the range usually
    degenerates to the max.  `mtc_diag` measures that share and reports the
    Spearman correlation between range and variance, which is the number
    that justifies treating them as different signals.

    Cutoff note: Carterette used the top 100 per system and set the
    reciprocal rank to 0 outside it.  The analogue for nDCG@10 is the
    top 10, which is exactly the support of `build_pair_weight_matrix`, so
    the baseline reuses W unchanged and the range/variance comparison is
    made on identical inputs.

    Three policies are registered:
      mtc_range        range over the ADAPTIVE top-20, no error term.
                       Directly comparable to `leverage`: same target set,
                       same batching, only the moment differs.
      mtc_product_cal  the same range times E[eps^2]_cal.  Directly
                       comparable to `product_cal`.
      mtc_range_all    range over ALL systems, computed once.  The
                       faithful published MTC.

    RNG ISOLATION.  The MTC policies draw from their own deterministic
    RandomState, keyed by (seed, year, policy name) through zlib.crc32,
    and never from the shared `rng`.  Existing draws are also made
    unconditionally even when a policy is skipped by --policies.  Both
    together mean the random-baseline tables and the bootstrap indices are
    bit-identical to a run of the previous version, so the new rows can be
    pasted beside results you have already computed.

C15. TARGETING GRID.  See the block above TARGETING_SPECS.

C16. ONLINE CALIBRATION (A1).  `expected_sq_error_calibrated` is fitted
    leave-one-query-out on the human grades of every OTHER query of the
    year.  Those labels are not counted in any budget, while LARA -- the
    baseline it is compared against -- only ever sees labels it paid for.
    The comparison was tilted in our favour.

    `OnlineCalibratedError` is the budget-honest replacement.  It keeps
    the estimator identical in every other respect, so the only thing that
    changes is the training set:

      * bin edges: max_prob quantiles over the universe (label free), and
        the number of bins chosen by the SAME rule as the offline
        estimator, which depends only on the occupancy of (emitted grade,
        bin) cells and is therefore label free as well;
      * the table P(g_h | g_L, bin) is rebuilt after every batch from the
        pairs PURCHASED SO FAR and nothing else;
      * per-cell backoff: bin cell -> emitted grade only -> raw softmax,
        whenever the cell has fewer than CAL_MIN_CELL purchased examples;
      * before the first purchase every cell is empty, so the first batch
        is scored with the raw E[eps^2], the analogue of LARA's identity
        calibrator (their Algorithm 1, line 3).

    Human grades are read in exactly one place, `observe()`, and only for
    indices the policy has just bought.  A leakage test confirms it:
    rewriting every label outside the first t purchases leaves the first t
    + one batch of the ordering unchanged, and the same test FAILS for the
    offline estimator.

    New rows: `product_online` (THE HEADLINE), `mtc_product_online`,
    `meanw_online`, and the `prod_on_*` targeting family.  The offline
    rows are kept, relabelled, so the size of the leak can be reported.

C17. LARA(n = N) (B1).  Takehi et al., Sec. 3.4, verbatim: "Bundle the
    topics into groups, based on the number of assessors n.  Then, divide
    the budget B by the number of assessors; each group of topics would be
    assigned with the budget B/n.  Only sample from one group until the
    budget B/n is exhausted, then move on to the next group."  They
    recommend n = N.

    Their procedure is defined for ONE budget.  A prefix of a sequential
    ordering is not it: at 10 percent the prefix would have exhausted the
    first topics and never touched the rest, whereas their LARA(N) at that
    budget gives every topic B/N.  So `lara_nN` is scheduled by
    WATER-FILLING: every batch is split so that each topic's cumulative
    count stays equal (to within one), and a topic whose pool is exhausted
    passes its share to the others.  At every checkpoint the per-topic
    counts therefore match what their LARA(N) would buy at that budget.
    The calibrator is shared across topics (their A accumulates over
    groups), refit after every batch, and the remainder imputed with the
    calibrated argmax.  What can differ from a literal sequential run at a
    given budget is only the calibrator's HISTORY: here it is trained on
    labels spread over every topic, not on whichever topics come first.
    The per-topic counts, the selection rule and the imputation are theirs.

C18. BASELINES (B2, B3).  `depth_k` (classical depth-k pooling order:
    every rank-1 passage, then every rank-2, ...) was computed but never
    reported; it is now a table row.  `retrieval_count` is reported as an
    extra.  `oracle` (|eps|, error only) stays, and `product_oracle` is
    added: the headline policy with the TRUE eps^2 in place of the
    estimate.  It is the ceiling for any error estimator under our
    leverage, so product_online -> product_oracle is the gap attributable
    to error estimation and product_oracle -> oracle is what leverage buys
    even with perfect error knowledge.  Both oracles read labels and are
    never policies.

C19. MEAN-WEIGHT ABLATION (B4).  `stat="mean"`: mean_s w_sp over the
    target systems.  Same target set, same batching, same error term as
    the headline; only the SPREAD is removed.  A passage every top system
    ranks third has a high mean and zero variance -- and cannot reorder
    anything under a linear metric.  If the variance is doing the work,
    `meanw_online` loses to `product_online`.

C21. TOP-K MEMBERSHIP.  tau@20 (Clarke and Dietz's definition) orders the
    TRUE top 20 among themselves and ignores any system the collection
    wrongly lifts into the top 20.  A policy that corrects only the passages
    of the systems it targets could look good on tau@20 while promoting
    intruders.  Two keys close that gap and flow through every output:
    overlap_at_20 (share of the predicted top 20 that is truly top 20) and
    tau_union_20 (Kendall tau over the union of both top 20s).  Neither
    changes any existing number.

C20. FIXED-BUDGET STATISTICS.  First-touch budgets have intervals up to
    ~100 points wide.  Every bootstrap replicate now also records tau@20,
    tau_AP and tau_all at TARGET_BUDGETS, per policy and as within-sample
    paired differences.  Outputs: fixed_budget_ci.csv, and new rows in
    paired_differences.csv.  The RESTRICT caveat in (b) below applies.

--------------------------------------------------------------------------
DESIGN DECISIONS THE BOOTSTRAP MAKES, STATED EXPLICITLY
--------------------------------------------------------------------------

(a) DUPLICATED QUERIES.  A query drawn k times contributes k copies of its
    nDCG to the mean.  Its pairs appear once in the correction universe,
    so the budget denominator counts unique pairs.

(b) POLICY ORDERING.  RESTRICT for every policy: the acquisition ordering
    is computed once on the full data and then restricted to the sampled
    pairs.  Exact for the static policies, whose scores are per-pair.
    NOT exact for every adaptive run-aware policy, for `mtf`, `mm_ns`,
    `lara` and `lara_nN`, and for anything using the online calibrator,
    all of which carry cross-query state.  Intervals for those understate
    uncertainty.  This is stated in bootstrap_config.md.

(c) B.  Start at 200, time year one, raise to 1000 if it fits.

--------------------------------------------------------------------------
OUTPUTS  ->  results/t12_resampling/
--------------------------------------------------------------------------
    bootstrap_config.md      design decisions, timings, seeds
    budget_table.csv         the table, point estimates
    budget_table.tex         the same, LaTeX, ready to paste
    endpoints_ci.csv         all-LLM endpoints with intervals
    thresholds_ci.csv        first touch and sustained, with intervals
    area_ci.csv              area over random, and ratio to oracle
    paired_differences.csv   within-sample policy differences
    difference_by_budget.csv policy minus random at every budget
    structural_ci.csv        correlations, with the resampled unit named
    curves_full.csv          every policy curve on full data
    fixed_budget_ci.csv      tau@20 / tau_AP / tau_all at fixed budgets (C20)
    online_calibration_diag.csv  how fast the online estimator learns (C16)
    targeting_matrix.csv     C15 grid at fixed budgets
    targeting_diagonal.csv   C15 diagonal test
    REPORT.md                everything in prose
"""

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kendalltau, spearmanr, pearsonr, rankdata

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# v2_id_mapping.py sits at the repository root, one level above triage/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2_id_mapping import V2_YEARS, load_canonical_map, canonicalize_runs


# ---------------------------------------------------------------------------
#  COMPAT
# ---------------------------------------------------------------------------

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent     # repository root
OUTPUT_DIR = BASE_DIR / "results" / "t12_resampling"

SEED             = 42
B_INITIAL        = 200
B_FULL           = 1000
N_BUDGET_STEPS   = 101      # 0%, 1%, ..., 100%  -> the 1% grid of the paper
TAU_THRESHOLD    = 0.95
K_TOP            = 20       # tau@K and the size of the adaptive target set

# C15.  Every sweep records tau at each of these cutoffs, plus tau_all and
# tau_AP.  20 must stay in the list: it is the default metric for the
# thresholds and the main table.
TAU_KS           = (10, 20, 50)

# Every metric `ranking_metrics` emits.  Anything that aggregates or copies
# curve rows must iterate this, not a hand-written subset -- forgetting one
# leaves the random baseline with a missing key, which surfaces silently as
# NaN in every area and difference computed against it.
METRIC_KEYS      = (("tau_all", "tau_ap", "max_drop")
                    + tuple(f"tau_at_{k}" for k in TAU_KS)
                    # C21.  Membership of the top K_TOP, see compute_topk_overlap.
                    + (f"overlap_at_{K_TOP}", f"tau_union_{K_TOP}"))

# Fixed budgets at which the targeting matrix is read.  Deliberately NOT
# budget-to-threshold: that statistic is a crossing point of a non-monotone
# curve and its bootstrap interval runs to 100 in most cells, so it cannot
# carry a twelve-cell comparison.  tau at a fixed budget is bounded and
# well behaved, and "who is ahead at matched spend" is the question the
# targeting claim actually asks.
TARGET_BUDGETS   = (0.05, 0.10, 0.15, 0.20, 0.30)

# C20.  Metrics read at TARGET_BUDGETS in every bootstrap replicate.
FIXED_METRICS    = ("tau_at_20", "tau_ap", "tau_all",
                    f"tau_union_{K_TOP}", f"overlap_at_{K_TOP}")

# C16.  Points at which the online calibrator's estimate is snapshotted for
# the diagnostic.  Diagnostic only: nothing here feeds back into selection.
ONLINE_CHECKPOINTS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
BATCH_FRACTION   = 0.01     # adaptive policies re-score every 1% of the pool
N_RAND_TABLES    = 20       # distinct random orderings cycled over bootstraps

# MTF / MM-NS treat a passage as relevant at human grade >= 2.  TREC DL
# convention, and LARA's l/2 rule with l = 3 gives the same threshold.
RELEVANCE_THRESHOLD = 2

# Calibrated error estimator
CAL_MIN_CELL = 20
CAL_BIN_TRY  = (3, 2, 1)    # 1 means "emitted grade only"

# MaxMean non-stationary (optional, not in the paper table)
MM_NS_WINDOW  = 50
MM_NS_EPSILON = 0.1

YEARS_CFG = {
    2019: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl" / "2019" / "qrels.txt",
        "scores":   BASE_DIR / "grades" / "llama-3.1-8b_v1.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2019",
    },
    2020: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl" / "2020" / "qrels.txt",
        "scores":   BASE_DIR / "grades" / "llama-3.1-8b_v1.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2020",
    },
    2021: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2021" / "qrels_dedup.txt",
        "scores":   BASE_DIR / "grades" / "llama-3.1-8b_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2021",
    },
    2022: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2022" / "qrels_dedup.txt",
        "scores":   BASE_DIR / "grades" / "llama-3.1-8b_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2022",
    },
    2023: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2023" / "qrels_dedup.txt",
        "scores":   BASE_DIR / "grades" / "llama-3.1-8b_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2023",
    },
}

# LLM judges.  YEARS_CFG above carries the default (llama); set_judge swaps
# the grade files and the output directory for another one.
JUDGES = {"llama": "llama-3.1-8b", "qwen": "qwen2.5-7b"}


def set_judge(judge):
    """Point YEARS_CFG and OUTPUT_DIR at the grades of `judge`."""
    global OUTPUT_DIR
    stem = JUDGES[judge]
    for year, cfg in YEARS_CFG.items():
        v = "v2" if year in V2_YEARS else "v1"
        cfg["scores"] = BASE_DIR / "grades" / f"{stem}_{v}.jsonl"
    if judge != "llama":
        OUTPUT_DIR = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_{judge}"

CONFIDENCE_CSV = BASE_DIR / "results" / "spectral" / "confidence_passage_linear.csv"
DAMAGE_DIR     = BASE_DIR / "results" / "triage_impact"
SPECTRAL_INT   = BASE_DIR / "results" / "spectral" / "intermediates"
DDEC_CSV       = BASE_DIR / "results" / "spectral" / "damage_decoupling_linear.csv"

# Policy identifier -> the name used in the paper table and figures.
DISPLAY_NAME = {
    "random":       "Random",
    "naive":        "Confidence, raw margin (Naive)",
    "lara":         "Confidence, calibrated margin (LARA)",
    "mtf":          "Move-to-front pooling",
    "mtc_range_all": "Minimal test collections (all systems)",
    "mtc_range":    "Weight range, top-20 (MTC)",
    "mtc_product_cal": "Weight range x error, calibrated (MTC)",
    "lev_k10":        "Leverage, C over top-10",
    "lev_k20":        "Leverage, C over top-20",
    "lev_k50":        "Leverage, C over top-50",
    "lev_kall":       "Leverage, C over all systems",
    "lev_k20_static": "Leverage, C over top-20, fixed at budget 0",
    "mtc_k10":        "Range, C over top-10",
    "mtc_k20":        "Range, C over top-20",
    "mtc_k50":        "Range, C over top-50",
    "mtc_kall":       "Range, C over all systems",
    "mtc_k20_static": "Range, C over top-20, fixed at budget 0",
    "prod_k10":       "Product, offline cal., C over top-10",
    "prod_k20":       "Product, offline cal., C over top-20",
    "prod_kall":      "Product, offline cal., C over all systems",
    "prod_on_k10":    "Product, online cal., C over top-10",
    "prod_on_k20":    "Product, online cal., C over top-20",
    "prod_on_k50":    "Product, online cal., C over top-50",
    "prod_on_kall":   "Product, online cal., C over all systems",
    "prod_on_k20_static": "Product, online cal., C over top-20, fixed at budget 0",
    "leverage":     "Leverage",
    "product_raw":  "Product, raw",
    "product_cal":  "Product, offline cal. (LOQO labels outside budget)",
    "product_online": "Product, online cal. (purchased labels only)",
    "mtc_product_cal": "Weight range x error, offline cal. (MTC)",
    "mtc_product_online": "Weight range x error, online cal. (MTC)",
    "meanw_k20":    "Mean weight, top-20",
    "meanw_online": "Mean weight x error, online cal.",
    "lara_nN":      "Confidence, calibrated margin (LARA, n = N)",
    "oracle":       "Oracle (error magnitude)",
    "product_oracle": "Oracle (leverage x true error)",
    "mm_ns":        "MaxMean non-stationary",
    "depth_k":      "Depth-k pooling",
    "retrieval_count": "Retrieval count",
}

READS = {
    "random": "neither", "naive": "judge", "lara": "judge", "mtf": "runs",
    "lara_nN": "judge",
    "mtc_range_all": "runs", "mtc_range": "runs", "mtc_product_cal": "both",
    "mtc_product_online": "both",
    "lev_k10": "runs", "lev_k20": "runs", "lev_k50": "runs",
    "lev_kall": "runs", "lev_k20_static": "runs",
    "mtc_k10": "runs", "mtc_k20": "runs", "mtc_k50": "runs",
    "mtc_kall": "runs", "mtc_k20_static": "runs",
    "prod_k10": "both", "prod_k20": "both", "prod_kall": "both",
    "prod_on_k10": "both", "prod_on_k20": "both", "prod_on_k50": "both",
    "prod_on_kall": "both", "prod_on_k20_static": "both",
    "leverage": "runs", "product_raw": "both", "product_cal": "both",
    "product_online": "both", "meanw_k20": "runs", "meanw_online": "both",
    "oracle": "labels", "product_oracle": "labels",
    "mm_ns": "runs", "depth_k": "runs",
    "retrieval_count": "runs",
}

# Rows of the table, in order.  Published baselines first, then the
# ablation ladder that ends in the headline, then the offline-calibrated
# references (kept only to size the leak, C16), then the two oracles.
TABLE_POLICIES = ["random", "depth_k", "mtf", "naive", "lara", "lara_nN",
                  "mtc_range_all", "mtc_range", "mtc_product_online",
                  "leverage", "meanw_k20", "meanw_online", "product_raw",
                  "product_online",
                  "mtc_product_cal", "product_cal",
                  "oracle", "product_oracle"]

# Policies added in C14.  Kept as a set so the RNG-isolation logic and the
# --policies default can both refer to it.
MTC_POLICIES = ("mtc_range_all", "mtc_range", "mtc_product_cal",
                "mtc_product_online")

# ---------------------------------------------------------------------------
#  C16 / C18 / C19.  RUN-AWARE POLICIES OUTSIDE THE TARGETING GRID
# ---------------------------------------------------------------------------
#
# All adaptive over the estimated top-K_TOP, all with stable per-policy
# seeds, so none of them touches the shared RNG stream.
#
#   (name, stat, error)      error: None | "online" | "true"
RUN_AWARE_SPECS = [
    ("product_online",     "var",   "online"),    # the headline
    ("mtc_product_online", "range", "online"),    # MTC with our error term
    ("meanw_k20",          "mean",  None),        # B4, spread removed
    ("meanw_online",       "mean",  "online"),    # B4, spread removed, x error
    ("product_oracle",     "var",   "true"),      # B3, ceiling on the error term
]
RUN_AWARE_POLICIES = tuple(s[0] for s in RUN_AWARE_SPECS)

# ---------------------------------------------------------------------------
#  C15.  TARGETING GRID
# ---------------------------------------------------------------------------
#
# The question: is the budget reduction from 25-59% (C over all systems) to
# 10-24% (C over the top twenty) caused by TARGETING the estimand, or merely
# by the target set being RECOMPUTED as grades arrive?  `mtc_range_all` is
# static over all systems and `mtc_range` is adaptive over twenty, so the
# two explanations are confounded in the C14 results.
#
# Note the design is not a 2x2.  "Adaptive over all systems" is a null
# concept: the target set never changes, so C is constant across batches and
# the policy is identical to its static form.  There are three cells and one
# was missing -- static over the top twenty.  `*_k20_static` supplies it.
#
# The grid then goes further and sweeps K, so the targeting claim can be
# demonstrated rather than asserted: if the budget rises monotonically with
# K, and each policy is strongest on the tau@K matching its own target, the
# decomposition is steerable.  At K = all systems this doubles as the tau_all
# experiment, which is the defence of the appendix table where the top-20
# policies lose to move-to-front pooling on the full ranking.
#
# `k = None` means all systems, and is always static for the reason above.
# K is clamped to the system count, so K=50 is K=all in a 35-system year.
#
#   (name, stat, error, k, adaptive)
TARGETING_SPECS = [
    ("lev_k10",         "var",   None,     10,   True),
    ("lev_k20",         "var",   None,     20,   True),
    ("lev_k50",         "var",   None,     50,   True),
    ("lev_kall",        "var",   None,     None, False),
    ("lev_k20_static",  "var",   None,     20,   False),
    ("mtc_k10",         "range", None,     10,   True),
    ("mtc_k20",         "range", None,     20,   True),
    ("mtc_k50",         "range", None,     50,   True),
    ("mtc_kall",        "range", None,     None, False),
    ("mtc_k20_static",  "range", None,     20,   False),
    ("prod_k10",        "var",   "e_cal",  10,   True),
    ("prod_k20",        "var",   "e_cal",  20,   True),
    ("prod_kall",       "var",   "e_cal",  None, False),
    # C16.  The budget-honest product family, with the full grid so the
    # targeting claim can be made for the headline policy itself and not
    # only for its label-free components.  With an online error term a
    # "static" policy is still re-scored every batch -- only C is frozen.
    ("prod_on_k10",        "var", "e_online", 10,   True),
    ("prod_on_k20",        "var", "e_online", 20,   True),
    ("prod_on_k50",        "var", "e_online", 50,   True),
    ("prod_on_kall",       "var", "e_online", None, False),
    ("prod_on_k20_static", "var", "e_online", 20,   False),
]
TARGETING_POLICIES = tuple(s[0] for s in TARGETING_SPECS)
ONLINE_TARGETING = tuple(s[0] for s in TARGETING_SPECS if s[2] == "e_online")

# lev_k20 duplicates `leverage`, mtc_k20 duplicates `mtc_range` and mtc_kall
# duplicates `mtc_range_all` in every respect EXCEPT the tie-break jitter,
# which comes from the stable per-policy seed here and from the shared stream
# there.  Between 31 and 73 percent of pairs have C_pp = 0, so the tie-break
# is not cosmetic.  Running both is therefore a free sensitivity check: if
# lev_k20 and leverage land far apart, the headline budget is partly an
# artefact of how ties are broken, which is something to find out now.

# Extra signals, bootstrapped and reported but not table rows.
EXTRA_POLICIES = ["max_prob", "entropy", "retrieval_count"]

# --policies shortcuts.  "new" is everything added in C16-C19; run it with
# --out-suffix to get the new rows without touching existing results, but
# note that a paired difference is only computed when BOTH policies ran in
# the same invocation (see PAIRS in run_year).
ONLINE_POLICIES = ("product_online", "mtc_product_online", "meanw_online") \
                  + ONLINE_TARGETING
NEW_POLICIES = ONLINE_POLICIES + ("lara_nN", "depth_k", "retrieval_count",
                                  "oracle", "meanw_k20", "product_oracle")
POLICY_SHORTCUTS = {
    "mtc": MTC_POLICIES,
    "targeting": TARGETING_POLICIES,
    "online": ONLINE_POLICIES,
    "new": NEW_POLICIES,
    "table": tuple(TABLE_POLICIES),
}
KNOWN_POLICIES = (set(TABLE_POLICIES) | set(TARGETING_POLICIES)
                  | set(EXTRA_POLICIES) | set(RUN_AWARE_POLICIES) | {"mm_ns"})


# ---------------------------------------------------------------------------
#  LINEAR GAIN
# ---------------------------------------------------------------------------

def gain(g):
    """Linear gain.  NOT 2^g - 1."""
    return float(g)


# ---------------------------------------------------------------------------
#  LOADING
# ---------------------------------------------------------------------------

def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            g = int(parts[3])
            qrels[parts[0]][parts[2]] = max(g, 0)   # TREC uses -1 for "not judged"
    return dict(qrels)


def load_llm_data(jsonl_path, year_queries):
    """Return (grades, probs).  Probabilities are renormalised to sum to one,
    which is LARA Eq. (1)."""
    grades = defaultdict(dict)
    probs  = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["query_id"])
            if qid not in year_queries:
                continue
            pid = str(rec["passage_id"])
            grades[qid][pid] = int(rec["score"])
            p = np.array([float(rec["probs"].get(str(k), 0.0)) for k in range(4)],
                         dtype=np.float64)
            s = p.sum()
            probs[(qid, pid)] = p / s if s > 0 else np.full(4, 0.25)
    return dict(grades), probs


def load_system_runs(runs_dir):
    runs = {}
    for fname in sorted(os.listdir(runs_dir)):
        if not fname.endswith(".txt"):
            continue
        sname = fname[:-4]
        sr = defaultdict(list)
        with open(os.path.join(runs_dir, fname)) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 6:
                    continue
                sr[p[0]].append((int(p[3]), p[2]))
        for qid in sr:
            sr[qid].sort()
            seen, out = set(), []
            for _, pid in sr[qid]:
                if pid in seen:          # duplicate ids can survive v2 mapping
                    continue
                seen.add(pid)
                out.append(pid)
                if len(out) >= 1000:
                    break
            sr[qid] = out
        runs[sname] = dict(sr)
    return runs


# ---------------------------------------------------------------------------
#  nDCG, LINEAR GAIN, IDCG SHARED ACROSS SYSTEMS
# ---------------------------------------------------------------------------

_DISCOUNT = np.array([1.0 / math.log2(i + 2) for i in range(10)])


def idcg_from_counter(cnt, k=10):
    """Ideal DCG at cutoff k from a Counter of grades.  Linear gain."""
    out, i = 0.0, 0
    for g in (3, 2, 1, 0):
        m = cnt.get(g, 0)
        while m > 0 and i < k:
            out += gain(g) * _DISCOUNT[i]
            i += 1
            m -= 1
        if i >= k:
            break
    return out


def dcg_at_k(ranked_pids, grades_q, k=10):
    out = 0.0
    for i, p in enumerate(ranked_pids[:k]):
        out += gain(grades_q.get(p, 0)) * _DISCOUNT[i]
    return out


def ndcg_matrix_from_grades(grades_by_q, queries, system_names, sys_top10,
                            out=None, dirty=None):
    """nDCG@10 for every (system, query).  IDCG computed once per query.

    `dirty` is an iterable of query indices to refresh; everything else is
    left alone in `out`.  Pass dirty=None to refresh all.
    """
    n_sys, n_q = len(system_names), len(queries)
    if out is None:
        out = np.zeros((n_sys, n_q))
        dirty = range(n_q)
    if dirty is None:
        dirty = range(n_q)
    for qi in dirty:
        gq = grades_by_q.get(queries[qi], {})
        idcg = idcg_from_counter(Counter(gq.values()))
        if idcg <= 0:
            out[:, qi] = 0.0
            continue
        for si in range(n_sys):
            out[si, qi] = dcg_at_k(sys_top10[si][qi], gq) / idcg
    return out


def rank_systems(scores, system_names):
    return [n for n, _ in sorted(zip(system_names, scores), key=lambda x: (-x[1], x[0]))]


def compute_tau(gold, pred):
    rank_g = {n: i for i, n in enumerate(gold)}
    tau, _ = kendalltau(list(range(len(gold))), [rank_g[n] for n in pred])
    return float(tau) if not np.isnan(tau) else 1.0


def compute_tau_at_k(gold, pred, K=K_TOP):
    K = min(K, len(gold))
    top = set(gold[:K])
    go = [s for s in gold if s in top]
    pr = [s for s in pred if s in top]
    if len(pr) < 2:
        return 1.0
    gr = {n: i for i, n in enumerate(go)}
    tau, _ = kendalltau(list(range(K)), [gr[n] for n in pr])
    return float(tau) if not np.isnan(tau) else 1.0


def compute_max_drop(gold, pred):
    gr = {n: i for i, n in enumerate(gold)}
    pr = {n: i for i, n in enumerate(pred)}
    return max(pr[n] - gr[n] for n in gold)


def compute_tau_ap(gold, pred):
    """AP rank correlation, Yilmaz, Aslam and Robertson (SIGIR 2008).

        tau_AP = 2/(N-1) * sum_{i=2}^{N} C(i)/(i-1)  -  1

    Verbatim from the paper: "C(i) is the number of items above rank i and
    correctly ranked with respect to the item at rank i in list1", where
    list1 is the ESTIMATED ranking and list2 the actual one.  So `i` runs
    over positions in `pred` and correctness is judged against `gold`.

    Asymmetric by construction -- the paper states "the AP correlation
    coefficient is not a symmetric statistic.  It assumes that there is an
    actual ranked list (list2) of items and an estimated ranked list
    (list1)."  The weighting is therefore top-heavy on `pred`, which is
    what we want: the question is whether the systems the collection calls
    best really are best.

    Both rankings here come from `rank_systems`, which breaks score ties
    deterministically by name, so they are total orders and the
    ties-corrected tau_AP_b is not needed.

    O(N^2) but vectorised; N is the number of systems, at most 100.
    """
    N = len(pred)
    if N < 2:
        return 1.0
    gr = {n: i for i, n in enumerate(gold)}
    g = np.fromiter((gr[p] for p in pred), dtype=np.int64, count=N)
    # concord[i, j] is True when j sits above i in pred AND above it in gold
    concord = np.tril(g[None, :] < g[:, None], k=-1)
    C = concord.sum(axis=1)[1:]                 # C(i) for i = 2 .. N
    denom = np.arange(1, N)                     # i - 1 for i = 2 .. N
    return float(2.0 * float(np.sum(C / denom)) / (N - 1) - 1.0)


def compute_topk_overlap(gold, pred, K=K_TOP):
    """C21.  |true top-K  intersect  predicted top-K| / K.

    tau@K, as defined by Clarke and Dietz and implemented above, orders the
    TRUE top K among themselves and never looks at who else the predicted
    ranking puts there.  A system from outside the true top K that rises
    into the predicted top K -- an intruder -- is invisible to it.  This is
    the share of the predicted top K that belongs there; 1 - overlap is the
    intruder share.
    """
    K = min(K, len(gold))
    if K == 0:
        return 1.0
    return len(set(gold[:K]) & set(pred[:K])) / K


def compute_tau_union(gold, pred, K=K_TOP):
    """C21.  Kendall tau over the UNION of the true and predicted top K.

    Identical to tau@K when there are no intruders (the union is then the
    true top K).  Each intruder enters the comparison together with every
    true top-K system it was placed above, so intruders now cost
    concordance instead of being ignored.  A single top-of-ranking number
    that cannot be flattered by correcting only the systems one targets.
    """
    K = min(K, len(gold))
    S = set(gold[:K]) | set(pred[:K])
    go = [s for s in gold if s in S]
    pr = [s for s in pred if s in S]
    if len(pr) < 2:
        return 1.0
    gr = {n: i for i, n in enumerate(go)}
    tau, _ = kendalltau(list(range(len(pr))), [gr[n] for n in pr])
    return float(tau) if not np.isnan(tau) else 1.0


def ranking_metrics(gold, pred):
    """Every agreement measure recorded at one budget step.

    tau_at_20 is retained under exactly that key because it is the default
    metric for the thresholds and for the main table.  A K larger than the
    number of systems is clamped by compute_tau_at_k, so for a year with
    35 systems tau_at_50 is tau_all.

    Keys must match METRIC_KEYS exactly; boot_schedule_sweep's degenerate
    branch and every aggregation rely on it.
    """
    out = {"tau_all": compute_tau(gold, pred),
           "tau_ap": compute_tau_ap(gold, pred),
           "max_drop": compute_max_drop(gold, pred)}
    for k in TAU_KS:
        out[f"tau_at_{k}"] = compute_tau_at_k(gold, pred, k)
    out[f"overlap_at_{K_TOP}"] = compute_topk_overlap(gold, pred, K_TOP)
    out[f"tau_union_{K_TOP}"] = compute_tau_union(gold, pred, K_TOP)
    return out


# ---------------------------------------------------------------------------
#  AUXILIARY STRUCTURES
# ---------------------------------------------------------------------------

def build_sys_top10(runs, queries, system_names):
    return [
        {qi: runs.get(sn, {}).get(qid, [])[:10] for qi, qid in enumerate(queries)}
        for sn in system_names
    ]


def build_pair_weight_matrix(runs, queries, system_names, universe):
    """Sparse (n_pairs x n_sys) matrix of nDCG position weights.

    w[p, s] = 1 / log2(rank_s(p) + 1) if p is in s's top ten, else 0.
    """
    pair_index = {k: i for i, k in enumerate(universe)}
    rows, cols, vals = [], [], []
    for si, sn in enumerate(system_names):
        sr = runs.get(sn, {})
        for qid in queries:
            for r, pid in enumerate(sr.get(qid, [])[:10]):
                j = pair_index.get((qid, pid))
                if j is None:
                    continue
                rows.append(j)
                cols.append(si)
                vals.append(_DISCOUNT[r])
    W = sp.csr_matrix((vals, (rows, cols)),
                      shape=(len(universe), len(system_names)))
    return W.tocsc(), pair_index


def leverage_over(Wc, sys_idx):
    """C_pp = population variance of the position weight across the given
    systems, the diagonal of C.  Zeros count."""
    M = len(sys_idx)
    if M == 0:
        return np.zeros(Wc.shape[0])
    sub = Wc[:, list(sys_idx)]
    s1 = np.asarray(sub.sum(axis=1)).ravel()
    s2 = np.asarray(sub.multiply(sub).sum(axis=1)).ravel()
    return np.maximum(s2 / M - (s1 / M) ** 2, 0.0)


def minmax_over(Wc, sys_idx):
    """(max_s w_sp, min_s w_sp) across the given systems, counting the
    structural zeros.

    Computed explicitly with reduceat rather than through scipy's sparse
    min / max, so the result does not depend on how a given scipy version
    treats implicit zeros.

    Every stored weight is 1 / log2(rank + 1) with rank in [0, 9], hence
    strictly positive and at most 1.  Two facts follow and are used here:

      * max over the subset is the largest STORED value in the row, or 0
        when the row stores nothing for these systems;
      * min over the subset is 0 unless the row stores a value for EVERY
        one of the M systems, in which case it is the smallest stored one.
    """
    M = len(sys_idx)
    n = Wc.shape[0]
    if M == 0:
        return np.zeros(n), np.zeros(n)

    sub = Wc[:, list(sys_idx)].tocsr()
    counts = np.diff(sub.indptr)
    mx = np.zeros(n)
    mn_stored = np.zeros(n)

    nz = counts > 0
    if nz.any():
        # CSR data is stored row by row, so the start offsets of the
        # non-empty rows segment `data` exactly: an empty row contributes no
        # elements and therefore cannot split a segment.
        starts = sub.indptr[:-1][nz]
        mx[nz] = np.maximum.reduceat(sub.data, starts)
        mn_stored[nz] = np.minimum.reduceat(sub.data, starts)

    mn = np.where(counts == M, mn_stored, 0.0)
    return mx, mn


def range_over(Wc, sys_idx):
    """MTC's selection weight on a linear metric: max_s w_sp - min_s w_sp.

    Carterette, Allan and Sitaraman (SIGIR 2006).  See note C14 in the
    module docstring for the derivation from their Algorithm 1.  This is
    the first moment analogue of `leverage_over`, computed over the same
    systems and the same weight matrix.
    """
    mx, mn = minmax_over(Wc, sys_idx)
    return mx - mn


def mean_over(Wc, sys_idx):
    """C19.  Mean position weight across the given systems, zeros counted.

    The ablation of the SPREAD.  Same support as leverage and range (a pair
    scores zero unless some target system has it in its top ten), but a
    passage every target system places at the same rank scores highly here
    and zero under the variance -- and under a linear metric such a
    passage cannot change any system's position relative to another.
    """
    M = len(sys_idx)
    if M == 0:
        return np.zeros(Wc.shape[0])
    sub = Wc[:, list(sys_idx)]
    return np.asarray(sub.sum(axis=1)).ravel() / M


# stat name -> the per-pair statistic of the weight vector across the target
# systems.  `run_adaptive_run_aware` and the static targeting path both
# dispatch through this, so a new moment is one entry here.
SPREAD_FNS = {"var": leverage_over, "range": range_over, "mean": mean_over}


def _stable_seed(seed, year, name):
    """Deterministic RandomState seed from (seed, year, policy name).

    zlib.crc32 rather than hash(), because Python randomises string hashes
    per process unless PYTHONHASHSEED is fixed, which would make the MTC
    orderings irreproducible across runs.
    """
    key = f"{int(seed)}|{int(year)}|{name}".encode("utf-8")
    return int(zlib.crc32(key) % (2 ** 31 - 1))


def build_pool_depth(runs, queries, system_names, universe_set):
    depth, n_sys = {}, {}
    for sn in system_names:
        sr = runs.get(sn, {})
        for qid in queries:
            for ri, pid in enumerate(sr.get(qid, [])):
                k = (qid, pid)
                if k not in universe_set:
                    continue
                depth[k] = min(depth.get(k, 10 ** 9), ri + 1)
                n_sys[k] = n_sys.get(k, 0) + 1
    return depth, n_sys


# ---------------------------------------------------------------------------
#  EXPECTED SQUARED ERROR
# ---------------------------------------------------------------------------

def expected_sq_error_raw(universe, llm_grades, softmax_probs):
    """E[eps_p^2]_raw = sum_h pi_h(p) (g_L(p) - h)^2.  Linear gain.

    Label free.  Assumes the judge is calibrated, which Sec. 3.2 shows it
    is not: on a passage confidently graded 2, almost no mass sits on
    grade 1, so a human-grade-1 passage gets a near-zero error term.
    """
    out = np.empty(len(universe))
    grades_g = np.arange(4, dtype=np.float64)
    for i, (qid, pid) in enumerate(universe):
        pi = softmax_probs.get((qid, pid))
        if pi is None:
            out[i] = 0.0
            continue
        gl = float(llm_grades[qid][pid])
        out[i] = float(np.sum(pi * (gl - grades_g) ** 2))
    return out


def _quantile_bin_edges(x, n_bins):
    if n_bins <= 1:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(x, qs))
    return edges


def expected_sq_error_calibrated(universe, llm_grades, human_qrels, softmax_probs,
                                 min_cell=CAL_MIN_CELL, bin_try=CAL_BIN_TRY,
                                 verbose=True):
    """E[eps_p^2]_cal = sum_h P(g_h = h | g_L(p), bin(p)) (g_L(p) - h)^2.

    The table is fitted LEAVE-ONE-QUERY-OUT within the year, and never
    across the v1 / v2 boundary because each year is processed alone.

    Bin edges come from max_prob quantiles.  max_prob uses no human labels,
    so the edges leak nothing.  Leave-one-query-out is exact and cheap
    because counts are additive: subtract the held-out query's counts.

    Backoff, per Sec. 5.1: three bins, then two, then the emitted grade
    alone, whenever an occupied cell holds fewer than `min_cell` examples.
    A cell with no training mass at all falls back to the raw softmax.

    NOT label free.  It uses human grades on the other queries of the same
    year.  It is training free at inference only.
    """
    n = len(universe)
    maxp = np.array([float(np.max(softmax_probs.get(k, np.full(4, .25))))
                     for k in universe])
    gl   = np.array([llm_grades[q][p] for q, p in universe], dtype=int)
    gh   = np.array([human_qrels[q].get(p, 0) for q, p in universe], dtype=int)
    qid_of = np.array([q for q, _ in universe])

    chosen_bins, edges, bin_of, N = None, None, None, None
    for nb in bin_try:
        e = _quantile_bin_edges(maxp, nb)
        b = np.digitize(maxp, e) if nb > 1 else np.zeros(n, dtype=int)
        nb_eff = int(b.max()) + 1
        cnt = np.zeros((4, nb_eff, 4))
        np.add.at(cnt, (gl, b, gh), 1.0)
        occupied = cnt.sum(axis=2)
        ok = np.all((occupied == 0) | (occupied >= min_cell))
        if ok or nb == bin_try[-1]:
            chosen_bins, edges, bin_of, N = nb_eff, e, b, cnt
            if not ok and verbose:
                print(f"      calibration: forced to {nb_eff} bin(s); "
                      f"smallest occupied cell = {occupied[occupied > 0].min():.0f}")
            break

    if verbose:
        print(f"      calibration table: {chosen_bins} confidence bin(s), "
              f"min occupied cell = "
              f"{N.sum(axis=2)[N.sum(axis=2) > 0].min():.0f}")

    # Per-query counts, for the leave-one-out subtraction.
    per_q = defaultdict(lambda: np.zeros((4, chosen_bins, 4)))
    for i in range(n):
        per_q[qid_of[i]][gl[i], bin_of[i], gh[i]] += 1.0

    raw = expected_sq_error_raw(universe, llm_grades, softmax_probs)
    grades_g = np.arange(4, dtype=np.float64)
    out = np.empty(n)
    n_fallback = 0

    order = defaultdict(list)
    for i in range(n):
        order[qid_of[i]].append(i)

    for q, idxs in order.items():
        T = N - per_q[q]
        for i in idxs:
            row = T[gl[i], bin_of[i]]
            tot = row.sum()
            if tot < min_cell:
                row = T[gl[i]].sum(axis=0)      # grade-only backoff
                tot = row.sum()
            if tot < min_cell:
                out[i] = raw[i]                 # nothing to learn from
                n_fallback += 1
                continue
            P = row / tot
            out[i] = float(np.sum(P * (gl[i] - grades_g) ** 2))

    if verbose and n_fallback:
        print(f"      calibration: {n_fallback} pair(s) fell back to the raw "
              f"softmax ({100 * n_fallback / n:.2f}%)")
    return out, {"n_bins": chosen_bins, "n_fallback": int(n_fallback)}


def true_sq_error(universe, llm_grades, human_qrels):
    """eps_p^2 = (g_L(p) - g_h(p))^2, linear gain.  READS EVERY LABEL.

    Used only by `product_oracle` and by the online-calibration diagnostic.
    Never by a policy.
    """
    return np.array([(gain(llm_grades[q][p]) - gain(human_qrels[q].get(p, 0))) ** 2
                     for q, p in universe], dtype=np.float64)


def _cal_bin_structure(maxp, gl, min_cell=CAL_MIN_CELL, bin_try=CAL_BIN_TRY):
    """The bin choice of `expected_sq_error_calibrated`, reproduced WITHOUT
    human labels.

    That function tries 3, 2, 1 bins and keeps the first count whose every
    occupied (emitted grade, bin) cell holds at least `min_cell` pairs.
    Occupancy sums the count table over the human grade, so it depends on
    the judge's output alone.  Reproducing it here therefore gives the
    online estimator exactly the same bins as the offline one, and the two
    differ only in their training set.
    """
    n = len(maxp)
    for nb in bin_try:
        e = _quantile_bin_edges(maxp, nb)
        b = np.digitize(maxp, e) if nb > 1 else np.zeros(n, dtype=int)
        nb_eff = int(b.max()) + 1
        occ = np.zeros((4, nb_eff))
        np.add.at(occ, (gl, b), 1.0)
        ok = np.all((occ == 0) | (occ >= min_cell))
        if ok or nb == bin_try[-1]:
            return nb_eff, e, b
    raise RuntimeError("unreachable")


class OnlineCalibratedError:
    """C16.  E[eps_p^2] estimated from the labels PURCHASED SO FAR.

        E[eps_p^2] = sum_h P_t(g_h = h | g_L(p), bin(p)) (g_L(p) - h)^2

    where P_t is the empirical distribution over the pairs bought before
    batch t.  Same bins, same backoff and same minimum cell size as the
    offline `expected_sq_error_calibrated`; only the training set differs.

    Backoff, per pair, in order:
        'cell'   the (emitted grade, bin) cell has >= min_cell purchases
        'grade'  else the emitted grade, pooled over bins, has >= min_cell
        'raw'    else the raw softmax E[eps^2], as before any purchase

    LABEL DISCIPLINE.  `human_qrels` is stored but read in exactly one
    method, `observe`, and only at indices the caller has just purchased.
    Observing a pair twice raises.  `n_label_reads` counts the reads, so a
    caller can assert that it equals the number of purchases.

    A fresh instance is needed per policy run: the counts ARE the state.
    """

    LEVELS = ("cell", "grade", "raw")

    def __init__(self, universe, llm_grades, softmax_probs, human_qrels,
                 min_cell=CAL_MIN_CELL, bin_try=CAL_BIN_TRY, checkpoints=()):
        self.universe = universe
        n = len(universe)
        maxp = np.array([float(np.max(softmax_probs.get(k, np.full(4, .25))))
                         for k in universe])
        self.gl = np.array([llm_grades[q][p] for q, p in universe], dtype=int)
        self.n_bins, self.edges, self.bin_of = _cal_bin_structure(
            maxp, self.gl, min_cell, bin_try)
        self.raw = expected_sq_error_raw(universe, llm_grades, softmax_probs)
        self.min_cell = int(min_cell)
        self._sqerr = ((self.gl[:, None] - np.arange(4)[None, :]) ** 2
                       ).astype(np.float64)
        self.cnt = np.zeros((4, self.n_bins, 4))
        self.observed = np.zeros(n, dtype=bool)
        self._labels = human_qrels
        self.n_label_reads = 0
        self._cache, self._level = None, None
        self.checkpoints = tuple(sorted(checkpoints))
        self._next_cp = 0
        self.snapshots = []

    def observe(self, idx):
        """Record the human grades of pairs that have JUST been purchased."""
        for j in idx:
            j = int(j)
            if self.observed[j]:
                raise RuntimeError(f"pair {self.universe[j]} observed twice")
            q, p = self.universe[j]
            h = int(self._labels[q][p])
            self.n_label_reads += 1
            self.cnt[self.gl[j], self.bin_of[j], h] += 1.0
            self.observed[j] = True
        self._cache = None
        if self._next_cp < len(self.checkpoints):
            frac = float(self.observed.mean())
            # Half a percent of slack: a 1% batch is round(0.01 n) pairs,
            # which can land a hair under 1%.
            while (self._next_cp < len(self.checkpoints)
                   and frac >= self.checkpoints[self._next_cp] - 5e-3):
                est, lev = self._compute()
                self.snapshots.append({
                    "target_frac": self.checkpoints[self._next_cp],
                    "frac": frac, "estimate": est.copy(),
                    "level": lev.copy(), "unbought": ~self.observed})
                self._next_cp += 1

    def _compute(self):
        cell = self.cnt[self.gl, self.bin_of]              # (n, 4)
        tot_c = cell.sum(axis=1)
        grd = self.cnt.sum(axis=1)[self.gl]                # (n, 4)
        tot_g = grd.sum(axis=1)
        use_c = tot_c >= self.min_cell
        use_g = (~use_c) & (tot_g >= self.min_cell)
        est = self.raw.copy()
        if use_c.any():
            Pc = cell[use_c] / tot_c[use_c][:, None]
            est[use_c] = (Pc * self._sqerr[use_c]).sum(axis=1)
        if use_g.any():
            Pg = grd[use_g] / tot_g[use_g][:, None]
            est[use_g] = (Pg * self._sqerr[use_g]).sum(axis=1)
        level = np.where(use_c, 0, np.where(use_g, 1, 2))
        return est, level

    def current(self):
        """E[eps^2] for EVERY pair under the current table.  Cached until the
        next `observe`."""
        if self._cache is None:
            self._cache, self._level = self._compute()
        return self._cache

    def level_shares(self, mask=None):
        self.current()
        lv = self._level if mask is None else self._level[mask]
        if lv.size == 0:
            return {k: float("nan") for k in self.LEVELS}
        return {k: float((lv == i).mean()) for i, k in enumerate(self.LEVELS)}


# ---------------------------------------------------------------------------
#  STATIC SCORERS   (higher score = bought earlier)
# ---------------------------------------------------------------------------

def scores_naive_margin(universe, softmax_probs):
    """LARA's Naive method, their Sec. 3.2 and Eq. (2).

    m' = pi^{k'} - pi^{s'} on the RAW normalised probabilities.  Smallest
    margin first, so the score is the negated margin.
    """
    out = np.empty(len(universe))
    for i, k in enumerate(universe):
        pi = np.sort(softmax_probs.get(k, np.full(4, .25)))[::-1]
        out[i] = -(pi[0] - pi[1])
    return out


def scores_max_prob(universe, softmax_probs):
    return np.array([-float(np.max(softmax_probs.get(k, np.full(4, .25))))
                     for k in universe])


def scores_entropy(universe, softmax_probs):
    out = np.empty(len(universe))
    for i, k in enumerate(universe):
        p = np.clip(softmax_probs.get(k, np.full(4, .25)), 1e-12, 1.0)
        out[i] = float(-np.sum(p * np.log(p)))
    return out


def scores_oracle_error(universe, human_qrels, llm_grades):
    """|eps_p| under LINEAR gain.  eps_p = g_L(p) - g_h(p).

    Not a policy.  It is the denominator of the oracle-share numbers.
    """
    return np.array([abs(gain(llm_grades[q][p]) - gain(human_qrels[q].get(p, 0)))
                     for q, p in universe])


def scores_depth_k(universe, pool_depth, pool_nsys):
    return np.array([-pool_depth.get(k, 10 ** 9) + 1e-6 * pool_nsys.get(k, 0)
                     for k in universe])


def scores_retrieval_count(universe, pool_nsys):
    return np.array([float(pool_nsys.get(k, 0)) for k in universe])


def order_from_scores(universe, scores, rng):
    """Descending sort with a fixed random tie-break.

    Roughly 85-90 percent of pairs have C_pp = 0.  Without this, the tie
    group would be resolved by passage-id order, which is neither random
    nor meaningful.
    """
    jitter = rng.permutation(len(universe))
    idx = np.lexsort((jitter, -np.asarray(scores, dtype=np.float64)))
    return [universe[i] for i in idx]


# ---------------------------------------------------------------------------
#  PER-QUERY nDCG SCHEDULE FOR A FIXED ACQUISITION ORDER
# ---------------------------------------------------------------------------

def build_per_query_ndcg_table(ordering_qi, grades_start, human_qrels,
                               queries, system_names, sys_top10):
    """ndcg_qk[qi][k, si] = nDCG of system si on query qi after the first k
    pairs OF THAT QUERY (in acquisition order) have been bought.

    IDCG depends only on (query, step), so it is computed once and shared
    across systems.  DCG is updated incrementally: a correction moves a
    system only if the corrected passage sits in that system's top ten.
    """
    n_sys, n_q = len(system_names), len(queries)
    pairs_per_q = [[] for _ in range(n_q)]
    for qi, pid in ordering_qi:
        pairs_per_q[qi].append(pid)

    ndcg_qk = []
    for qi, qid in enumerate(queries):
        pids = pairs_per_q[qi]
        K = len(pids)
        g0 = dict(grades_start.get(qid, {}))
        hq = human_qrels.get(qid, {})

        cnt = Counter(g0.values())
        idcg = np.empty(K + 1)
        idcg[0] = idcg_from_counter(cnt)
        for k, pid in enumerate(pids, 1):
            old, new = g0[pid], hq[pid]
            cnt[old] -= 1
            cnt[new] += 1
            g0[pid] = new
            idcg[k] = idcg_from_counter(cnt)
        idcg_safe = np.where(idcg > 0, idcg, 1.0)

        step_of = {pid: k for k, pid in enumerate(pids, 1)}
        delta   = {pid: gain(hq[pid]) - gain(grades_start[qid][pid]) for pid in pids}
        base_g  = grades_start.get(qid, {})

        tab = np.zeros((K + 1, n_sys))
        for si in range(n_sys):
            top = sys_top10[si][qi]
            dcg = np.full(K + 1, sum(gain(base_g.get(p, 0)) * _DISCOUNT[r]
                                     for r, p in enumerate(top)))
            for r, pid in enumerate(top):
                k = step_of.get(pid)
                if k is not None:
                    dcg[k:] += _DISCOUNT[r] * delta[pid]
            tab[:, si] = np.where(idcg > 0, dcg / idcg_safe, 0.0)
        ndcg_qk.append(tab)

    return ndcg_qk, pairs_per_q


# ---------------------------------------------------------------------------
#  ADAPTIVE RUN-AWARE POLICIES:  leverage, product_raw, product_cal
# ---------------------------------------------------------------------------

def run_adaptive_run_aware(universe, pair_index, Wc, grades_start, human_qrels,
                           queries, system_names, sys_top10,
                           error_term=None, M=K_TOP, batch_size=100,
                           rng=None, verbose=False, stat="var",
                           error_model=None, fixed_target=None):
    """Buy in descending order of the chosen spread statistic, optionally
    times E[eps^2].

    `stat` selects the moment of the weight vector taken across the target
    systems:

        "var"    C_pp, the population variance.  Our leverage.
        "range"  max_s w_sp - min_s w_sp.  Minimal Test Collections
                 (Carterette et al., SIGIR 2006) on a linear metric; see
                 note C14 in the module docstring.
        "mean"   mean_s w_sp.  The spread ablation, note C19.

    Everything else is held identical between them, so a difference in
    the resulting budget is attributable to the moment alone.

    The error term is EITHER a fixed array (`error_term`) OR a model
    (`error_model`) that is queried before every batch and told what was
    bought after it.  The model is how the online calibrator (C16) enters;
    passing both is an error.

    The target set is the M systems currently ranked highest under the
    grades available at that moment.  It starts from the ALL-LLM
    leaderboard, so no human label enters the selection.  Purchases change
    the estimated ranking, which changes the target set, which changes the
    leverage used for the next batch.  This is the adaptive rule described
    in Sec. 4.5, not one fixed sorting.

    `fixed_target`, if given, is a set of system indices used for C at
    every batch instead of the current top M: the "static" arm of the
    targeting grid.  With a fixed error array that reproduces
    `order_from_scores` exactly (same permutation draw, same tie-break);
    it exists so that a static C can be combined with an error model that
    still changes every batch.

    Called with neither `error_model` nor `fixed_target`, this function
    executes exactly the statements of the previous version, so every
    existing policy is bit-identical.
    """
    if stat not in SPREAD_FNS:
        raise ValueError(f"stat must be one of {sorted(SPREAD_FNS)}, got {stat!r}")
    if error_term is not None and error_model is not None:
        raise ValueError("pass error_term OR error_model, not both")
    spread_fn = SPREAD_FNS[stat]
    rng = rng or np.random.RandomState(0)
    qi_map = {q: i for i, q in enumerate(queries)}
    grades = {q: dict(v) for q, v in grades_start.items()}

    ndcg = ndcg_matrix_from_grades(grades, queries, system_names, sys_top10)
    n_q = len(queries)

    unbought = np.ones(len(universe), dtype=bool)
    jitter = rng.permutation(len(universe)).astype(np.float64)
    jitter /= (jitter.max() + 1.0)          # strictly inside [0, 1)
    ordering = []

    C_fixed = (None if fixed_target is None
               else spread_fn(Wc, [int(s) for s in fixed_target]))

    while unbought.any():
        if C_fixed is None:
            top_idx = np.argsort(-ndcg.mean(axis=1), kind="stable")[:M]
            C = spread_fn(Wc, top_idx)
        else:
            C = C_fixed
        term = error_term if error_model is None else error_model.current()
        score = C if term is None else C * term

        cand = np.flatnonzero(unbought)
        # Rank by score, ties broken by the fixed jitter.
        key = np.lexsort((jitter[cand], -score[cand]))
        take = cand[key[:min(batch_size, cand.size)]]

        dirty = set()
        for j in take:
            qid, pid = universe[j]
            grades[qid][pid] = human_qrels[qid][pid]
            unbought[j] = False
            ordering.append((qi_map[qid], pid))
            dirty.add(qi_map[qid])

        if error_model is not None:
            error_model.observe(take)

        # A frozen target set never reads the leaderboard, so skip the work.
        if C_fixed is None:
            ndcg = ndcg_matrix_from_grades(grades, queries, system_names,
                                           sys_top10, out=ndcg, dirty=dirty)
        if verbose and len(ordering) % (20 * batch_size) < batch_size:
            print(".", end="", flush=True)

    return ordering


# ---------------------------------------------------------------------------
#  MOVE-TO-FRONT POOLING  (Cormack, Palmer and Clarke, SIGIR 1998, Sec. 6)
# ---------------------------------------------------------------------------

def run_mtf_policy(universe, human_qrels, queries, system_names, sys_top10,
                   runs, relevance_threshold=RELEVANCE_THRESHOLD):
    """Local MTF with numeric priorities.

    Cormack et al., Sec. 6: "The submissions themselves are prioritized,
    and the top-ranked document from the submission with the top priority
    is judged.  If it is judged relevant (or has been previously judged
    relevant because it appeared in some other submission) its priority is
    set to the maximum.  Otherwise, its priority is reduced."

    Three points the previous implementation missed.

      1. A miss REDUCES the priority by one.  It does not send the run to
         the back of the queue.  A run far ahead of the field stays ahead
         after a single miss.
      2. A hit sets the priority to the current maximum, so the run
         returns to the front.
      3. A document already judged still updates the priority, and costs
         no budget.  That is explicit in the paper.

    Local, not global.  Cormack's local variant "ensures that each topic
    receives a comparable number of judgements".  Global MTF would exhaust
    one topic before touching the next, which destroys a per-query
    leaderboard sweep at any budget below 100 percent.  Topics are visited
    round-robin, one judgement per visit.
    """
    universe_set = set(universe)
    qi_map = {q: i for i, q in enumerate(queries)}
    pids_of_q = defaultdict(set)
    for qid, pid in universe:
        pids_of_q[qid].add(pid)

    prio    = {qid: {sn: 0.0 for sn in system_names} for qid in queries}
    cursor  = {qid: {sn: 0 for sn in system_names} for qid in queries}
    judged  = {qid: {} for qid in queries}          # pid -> human grade
    remaining = defaultdict(int)
    for qid, pid in universe:
        remaining[qid] += 1

    ordering = []
    active = [q for q in queries if remaining[q] > 0]

    while active:
        still = []
        for qid in active:
            sr_prio = prio[qid]
            bought = False
            # Bounded: each inner pass advances at least one cursor.
            for _ in range(len(system_names) * 4):
                if remaining[qid] == 0:
                    break
                sn = max(system_names, key=lambda s: (sr_prio[s], s))
                pids = runs.get(sn, {}).get(qid, [])
                c = cursor[qid][sn]
                # Advance past pairs that are not part of the judged pool.
                while c < len(pids) and (qid, pids[c]) not in universe_set:
                    c += 1
                cursor[qid][sn] = c
                if c >= len(pids):
                    sr_prio[sn] = -1e18          # this run is exhausted here
                    continue
                pid = pids[c]
                cursor[qid][sn] = c + 1
                if pid in judged[qid]:
                    # Already judged elsewhere.  Update priority, spend nothing.
                    g = judged[qid][pid]
                    if g >= relevance_threshold:
                        sr_prio[sn] = max(sr_prio.values())
                    else:
                        sr_prio[sn] -= 1.0
                    continue
                g = human_qrels[qid][pid]
                judged[qid][pid] = g
                remaining[qid] -= 1
                ordering.append((qi_map[qid], pid))
                if g >= relevance_threshold:
                    sr_prio[sn] = max(sr_prio.values())
                else:
                    sr_prio[sn] -= 1.0
                bought = True
                break
            if remaining[qid] > 0:
                if bought or any(v > -1e17 for v in sr_prio.values()):
                    still.append(qid)
                else:
                    # No run reaches the leftover pairs.  Append them in a
                    # fixed order so the ordering covers the whole universe.
                    for pid in sorted(pids_of_q[qid] - set(judged[qid])):
                        judged[qid][pid] = human_qrels[qid][pid]
                        ordering.append((qi_map[qid], pid))
                        remaining[qid] -= 1
        active = still

    return ordering


# ---------------------------------------------------------------------------
#  MaxMean non-stationary  (Losada et al. 2016).  Optional.
# ---------------------------------------------------------------------------

def run_mm_ns_policy(universe, human_qrels, queries, system_names, runs,
                     window=MM_NS_WINDOW, epsilon=MM_NS_EPSILON, seed=SEED,
                     relevance_threshold=RELEVANCE_THRESHOLD):
    from collections import deque as _deque
    rng = np.random.RandomState(seed)
    universe_set = set(universe)
    qi_map = {q: i for i, q in enumerate(queries)}
    pids_of_q = defaultdict(set)
    for qid, pid in universe:
        pids_of_q[qid].add(pid)

    rw = {qid: {sn: _deque([0.5], maxlen=window) for sn in system_names}
          for qid in queries}
    cursor = {qid: {sn: 0 for sn in system_names} for qid in queries}
    judged = {qid: set() for qid in queries}
    remaining = defaultdict(int)
    for qid, pid in universe:
        remaining[qid] += 1

    ordering = []
    active = [q for q in queries if remaining[q] > 0]
    while active:
        still = []
        for qid in active:
            arms = [s for s in system_names if cursor[qid][s] is not None]
            if not arms or remaining[qid] == 0:
                continue
            if rng.random() < epsilon:
                sn = arms[rng.randint(len(arms))]
            else:
                sn = max(arms, key=lambda s: (sum(rw[qid][s]) / len(rw[qid][s]), s))
            pids = runs.get(sn, {}).get(qid, [])
            c = cursor[qid][sn]
            while c < len(pids) and ((qid, pids[c]) not in universe_set
                                     or pids[c] in judged[qid]):
                c += 1
            if c >= len(pids):
                cursor[qid][sn] = None
                still.append(qid)
                continue
            pid = pids[c]
            cursor[qid][sn] = c + 1
            judged[qid].add(pid)
            remaining[qid] -= 1
            ordering.append((qi_map[qid], pid))
            rw[qid][sn].append(1.0 if human_qrels[qid][pid] >= relevance_threshold
                               else 0.0)
            if remaining[qid] > 0:
                still.append(qid)
        active = [q for q in still if remaining[q] > 0
                  and any(cursor[q][s] is not None for s in system_names)]

    # Anything no run reaches, appended last.
    for qid in queries:
        for pid in sorted(pids_of_q[qid] - judged[qid]):
            ordering.append((qi_map[qid], pid))
    return ordering


# ---------------------------------------------------------------------------
#  LARA  (Takehi, Voorhees, Sakai and Soboroff, SIGIR 2025, Algorithm 1)
# ---------------------------------------------------------------------------

class LaraCalibrator:
    """Per-level calibration of the LLM probability vector.

    Their Sec. 3.3: "we propose to learn this calibration mapping for each
    relevance level j".  Their Sec. 4.2: "Logistic regression is used as
    the calibration model for LARA."

    One one-vs-rest logistic regression per relevance level, taking the
    four LLM probabilities as features, refit on all labels acquired so
    far, then normalised across levels.  Until at least two distinct human
    grades have been seen, the calibrator is the identity, which is
    Algorithm 1 line 3.
    """

    def __init__(self, n_cls=4, C=1.0, seed=SEED):
        self.n_cls = n_cls
        self.C = C
        self.seed = seed
        self.X, self.y = [], []
        self.models = None

    @property
    def active(self):
        return self.models is not None

    def add(self, x, label):
        self.X.append(np.asarray(x, dtype=np.float64))
        self.y.append(int(label))

    def refit(self):
        if len(self.y) < 2 or len(set(self.y)) < 2 or not SKLEARN_AVAILABLE:
            return
        X = np.vstack(self.X)
        y = np.asarray(self.y)
        models = {}
        for j in range(self.n_cls):
            t = (y == j).astype(int)
            if t.sum() == 0 or t.sum() == len(t):
                models[j] = float(t.mean())          # degenerate: constant rate
                continue
            m = LogisticRegression(C=self.C, max_iter=1000,
                                   random_state=self.seed)
            m.fit(X, t)
            models[j] = m
        self.models = models

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if not self.active:
            return X / np.clip(X.sum(axis=1, keepdims=True), 1e-12, None)
        P = np.zeros((X.shape[0], self.n_cls))
        for j, m in self.models.items():
            P[:, j] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
        P = np.clip(P, 1e-12, None)
        return P / P.sum(axis=1, keepdims=True)

    def margin(self, X):
        P = self.predict_proba(X)
        s = np.sort(P, axis=1)[:, ::-1]
        return s[:, 0] - s[:, 1]

    def argmax(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def run_lara_policy(universe, grades_start, human_qrels, softmax_probs,
                    queries, system_names, sys_top10, batch_size,
                    n_groups=1, seed=SEED, verbose=False):
    """LARA, Algorithm 1.

    Selection is by the CALIBRATED margin, not the raw one.  The remainder
    is imputed with argmax of the calibrated distribution.  That imputation
    is why LARA needs its own leaderboard schedule: unbought grades move
    every time the calibrator is refit, so a per-query correction table
    cannot represent it.

    `n_groups` is their LARA(n).  n = 1 is a single pool over all topics.
    n = N gives every topic its own budget share.  They recommend n = N.
    Default 1, the single pool.

    n > 1 is scheduled by WATER-FILLING, note C17: each batch is split so
    every group's cumulative count stays equal to within one, and an
    exhausted group passes its share on.  At every checkpoint each group
    has received B/n (capped at its pool), which is what their LARA(n)
    buys at budget B.  The previous version exhausted group 1 before
    touching group 2, so any prefix below 100 percent left later topics at
    zero -- not LARA(n) at that budget.  The n = 1 path is unchanged.

    Returns the acquisition ordering plus the checkpoint schedule of
    system-by-query nDCG under the mixed human / calibrated grades.
    """
    rng = np.random.RandomState(seed)
    qi_map = {q: i for i, q in enumerate(queries)}
    n_q, n_sys = len(queries), len(system_names)
    n_pairs = len(universe)

    X_all = np.vstack([softmax_probs.get(k, np.full(4, .25)) for k in universe])
    jitter = rng.permutation(n_pairs).astype(np.float64)
    jitter /= (jitter.max() + 1.0)

    if n_groups <= 1:
        groups = [np.arange(n_pairs)]
    else:
        gq = {q: i % n_groups for i, q in enumerate(queries)}
        # One pass over the universe instead of one per group: with n = N
        # the per-group comprehension was O(N x pairs) in Python.
        g_of = np.fromiter((gq[q] for q, _ in universe), dtype=np.int64,
                           count=n_pairs)
        groups = [np.flatnonzero(g_of == g) for g in range(n_groups)]
        groups = [g for g in groups if g.size]

    cal = LaraCalibrator(seed=seed)
    grades = {q: dict(v) for q, v in grades_start.items()}
    unbought = np.ones(n_pairs, dtype=bool)

    ndcg = ndcg_matrix_from_grades(grades, queries, system_names, sys_top10)
    sched_ndcg = [ndcg.copy()]
    sched_cum  = [np.zeros(n_q, dtype=int)]
    cum = np.zeros(n_q, dtype=int)
    ordering = []

    def _buy_refit_checkpoint(take):
        for j in take:
            qid, pid = universe[j]
            y = human_qrels[qid][pid]
            grades[qid][pid] = y
            cal.add(X_all[j], y)
            unbought[j] = False
            ordering.append((qi_map[qid], pid))
            cum[qi_map[qid]] += 1
        cal.refit()

        # Impute every unbought grade with the calibrated argmax.
        mixed = {q: dict(v) for q, v in grades.items()}
        rest = np.flatnonzero(unbought)
        if rest.size and cal.active:
            pred = cal.argmax(X_all[rest])
            for t, j in enumerate(rest):
                qid, pid = universe[j]
                mixed[qid][pid] = int(pred[t])

        sched_ndcg.append(ndcg_matrix_from_grades(mixed, queries, system_names,
                                                  sys_top10).copy())
        sched_cum.append(cum.copy())
        if verbose:
            print(".", end="", flush=True)

    if len(groups) == 1:
        g = groups[0]
        while unbought[g].any():
            cand = g[unbought[g]]
            m = cal.margin(X_all[cand])
            key = np.lexsort((jitter[cand], m))       # smallest margin first
            _buy_refit_checkpoint(cand[key[:min(batch_size, cand.size)]])
    else:
        # C17.  Water-filling across groups.  The group tie order is drawn
        # AFTER the jitter, so the n = 1 stream above is untouched.
        tie = rng.permutation(len(groups))
        sizes = np.array([g.size for g in groups], dtype=int)
        taken = np.zeros(len(groups), dtype=int)
        margin = np.empty(n_pairs)
        while unbought.any():
            room = sizes - taken
            alloc = _water_fill(taken, room,
                                min(batch_size, int(room.sum())), tie)
            rest = np.flatnonzero(unbought)
            margin[rest] = cal.margin(X_all[rest])
            parts = []
            for gi in np.flatnonzero(alloc):
                g = groups[gi]
                cand = g[unbought[g]]
                key = np.lexsort((jitter[cand], margin[cand]))
                parts.append(cand[key[:alloc[gi]]])
            taken += alloc
            _buy_refit_checkpoint(np.concatenate(parts))

    return (ordering, np.array(sched_ndcg), np.array(sched_cum, dtype=int))


def _water_fill(taken, room, n, tie):
    """Distribute `n` units one at a time, always to the group with the
    fewest units so far that still has room.  Ties by `tie` rank.

    Returns the per-group allocation.  After it, taken + alloc is as equal
    across non-exhausted groups as integers allow, which is "each group
    gets B/n" with the leftover of small groups redistributed.
    """
    import heapq
    alloc = np.zeros(len(taken), dtype=int)
    heap = [(int(taken[g]), int(tie[g]), g) for g in range(len(taken))
            if room[g] > 0]
    heapq.heapify(heap)
    for _ in range(int(n)):
        c, t, g = heapq.heappop(heap)
        alloc[g] += 1
        if alloc[g] < room[g]:
            heapq.heappush(heap, (c + 1, t, g))
    return alloc


# ---------------------------------------------------------------------------
#  SWEEPS
# ---------------------------------------------------------------------------

def boot_correction_sweep(ordering_qi, ndcg_qk, query_counts, system_names,
                          gold_ranking, n_q, budget_fracs):
    """Sweep for a policy whose unbought pairs keep their starting grade."""
    n_sys = len(system_names)
    boot_qs = np.flatnonzero(query_counts > 0)
    restr = [(qi, pid) for qi, pid in ordering_qi if query_counts[qi] > 0]
    n_restr = max(len(restr), 1)

    out, k_per_q, ptr = [], np.zeros(n_q, dtype=int), 0
    for b in budget_fracs:
        target = int(round(b * n_restr))
        while ptr < target:
            k_per_q[restr[ptr][0]] += 1
            ptr += 1
        s = np.zeros(n_sys)
        for qi in boot_qs:
            k = min(k_per_q[qi], ndcg_qk[qi].shape[0] - 1)
            s += query_counts[qi] * ndcg_qk[qi][k]
        s /= n_q
        r = rank_systems(s, system_names)
        out.append(dict(budget=float(b), **ranking_metrics(gold_ranking, r)))
    return out


def boot_schedule_sweep(sched_ndcg, sched_cum, query_counts, system_names,
                        gold_ranking, n_q, budget_fracs):
    """Sweep for a policy that also rewrites its unbought grades, i.e. LARA."""
    boot_qs = np.flatnonzero(query_counts > 0)
    n_boot = int(sched_cum[-1][boot_qs].sum())
    if n_boot == 0:
        d = {"tau_all": 1.0, "tau_ap": 1.0, "max_drop": 0}
        d.update({f"tau_at_{k}": 1.0 for k in TAU_KS})
        d.update({f"overlap_at_{K_TOP}": 1.0, f"tau_union_{K_TOP}": 1.0})
        assert set(d) == set(METRIC_KEYS), "degenerate row out of sync with METRIC_KEYS"
        return [dict(d, budget=float(b)) for b in budget_fracs]
    b_at = sched_cum[:, boot_qs].sum(axis=1) / n_boot

    out = []
    for b in budget_fracs:
        i = max(0, min(int(np.searchsorted(b_at, b, side="right")) - 1,
                       len(sched_ndcg) - 1))
        s = (sched_ndcg[i] * query_counts[None, :]).sum(axis=1) / n_q
        r = rank_systems(s, system_names)
        out.append(dict(budget=float(b), **ranking_metrics(gold_ranking, r)))
    return out


def ql_triage_sweep(ndcg_h, ndcg_l, order_qi, query_counts, system_names,
                    gold_ranking, n_q):
    """Query-level triage, budget expressed as the fraction of pairs bought.

    Emitted ASCENDING in budget.  The previous version emitted descending,
    which silently negated every area and broke np.interp.
    """
    n_sys = len(system_names)
    boot_qs = set(np.flatnonzero(query_counts > 0).tolist())
    restr = [qi for qi in order_qi if qi in boot_qs]
    n_restr = max(len(restr), 1)

    human_total = (ndcg_h * query_counts[None, :]).sum(axis=1)
    delta = ndcg_l - ndcg_h

    rows = []
    switch = np.zeros(n_sys)
    for k in range(n_restr + 1):
        s = (human_total + switch) / n_q
        r = rank_systems(s, system_names)
        rows.append(dict(budget=(n_restr - k) / n_restr,
                         **ranking_metrics(gold_ranking, r)))
        if k < n_restr:
            qi = restr[k]
            switch += delta[:, qi] * query_counts[qi]
    rows.sort(key=lambda d: d["budget"])
    return rows


# ---------------------------------------------------------------------------
#  THRESHOLDS AND AREAS
# ---------------------------------------------------------------------------

def threshold_first_touch(curve, metric="tau_at_20", thr=TAU_THRESHOLD):
    """Smallest budget at which the metric FIRST reaches thr.  The table."""
    for r in sorted(curve, key=lambda d: d["budget"]):
        if r[metric] >= thr:
            return r["budget"], True
    return None, False


def threshold_sustained(curve, metric="tau_at_20", thr=TAU_THRESHOLD):
    """Smallest budget b such that the metric is at or above thr at EVERY
    budget from b to the end of the sweep.  The appendix."""
    c = sorted(curve, key=lambda d: d["budget"])
    v = [r[metric] for r in c]
    idx = len(v)
    for i in range(len(v) - 1, -1, -1):
        if v[i] >= thr:
            idx = i
        else:
            break
    return (c[idx]["budget"], True) if idx < len(v) else (None, False)


def area_between(pol_curve, rand_curve, metric="tau_at_20"):
    """Trapezoidal area between a policy curve and the random mean curve.

    Both curves are sorted ascending in budget first.  np.interp needs an
    increasing xp, and the trapezoid of a descending axis is negated.
    """
    p = sorted(pol_curve, key=lambda d: d["budget"])
    r = sorted(rand_curve, key=lambda d: d["budget"])
    pb = np.array([x["budget"] for x in p])
    pv = np.array([x[metric] for x in p])
    rb = np.array([x["budget"] for x in r])
    rv = np.array([x[metric] for x in r])
    span = pb[-1] - pb[0]
    if span <= 0:
        return 0.0
    return float(_TRAPZ(pv - np.interp(pb, rb, rv), pb))


# ---------------------------------------------------------------------------
#  ONE YEAR
# ---------------------------------------------------------------------------

def run_year(year, cfg, B, rng_master, verbose=True, include_mm_ns=False,
             lara_groups=1, policies=None, seed=SEED):
    """`policies` is the set of policy names to actually compute.  None
    means all of them.

    Skipping a policy never changes any other policy's result.  The draws
    from the shared `rng` are made whether or not the policy is built, so
    the stream reaching the random tables and the bootstrap is unchanged,
    and the MTC policies added in C14 take their randomness from their own
    deterministic RandomState instead of from that stream.  `random` is
    always computed because it is the reference curve for every area and
    every by-budget difference.
    """
    t0 = time.time()
    rng = np.random.RandomState(rng_master.randint(0, 2 ** 31))

    if policies is None:
        want = lambda name: True                                  # noqa: E731
    else:
        keep = set(policies) | {"random"}
        want = lambda name: name in keep                          # noqa: E731

    human_qrels = load_qrels(cfg["qrels"])
    year_qs = set(human_qrels)
    llm_grades, softmax_probs = load_llm_data(cfg["scores"], year_qs)
    year_qs &= set(llm_grades)
    queries = sorted(year_qs)

    runs = load_system_runs(cfg["runs_dir"])
    if year in V2_YEARS:
        canonicalize_runs(runs, load_canonical_map())
    system_names = sorted(runs)
    n_q, n_sys = len(queries), len(system_names)

    universe = [(q, p) for q in queries for p in sorted(human_qrels[q])
                if p in llm_grades.get(q, {})]
    n_universe = len(universe)
    n_missing = sum(len(human_qrels[q]) for q in queries) - n_universe
    batch_size = max(1, int(round(BATCH_FRACTION * n_universe)))

    # Pairs the judge never scored keep their human grade in BOTH the gold
    # and the mixed leaderboards, so the two use the same passage set.
    grades_start = {q: {p: (llm_grades[q][p] if p in llm_grades.get(q, {})
                            else human_qrels[q][p])
                        for p in human_qrels[q]} for q in queries}

    if verbose:
        print(f"  {n_q} queries, {n_sys} systems, {n_universe} pairs, "
              f"batch {batch_size}, {n_missing} pair(s) without an LLM grade")

    sys_top10 = build_sys_top10(runs, queries, system_names)
    Wc, pair_index = build_pair_weight_matrix(runs, queries, system_names, universe)
    pool_depth, pool_nsys = build_pool_depth(runs, queries, system_names,
                                             set(universe))

    zero_lev = float((leverage_over(Wc, range(n_sys)) == 0).mean())
    if verbose:
        print(f"  {100 * zero_lev:.1f}% of pairs have C_pp = 0 over all runs "
              f"(the eligibility share)")

    ndcg_h = ndcg_matrix_from_grades(human_qrels, queries, system_names, sys_top10)
    ndcg_l = ndcg_matrix_from_grades(grades_start, queries, system_names, sys_top10)
    gold_full = rank_systems(ndcg_h.mean(axis=1), system_names)
    llm_full  = rank_systems(ndcg_l.mean(axis=1), system_names)
    point = ranking_metrics(gold_full, llm_full)
    if verbose:
        print(f"  all-LLM endpoint: tau_all={point['tau_all']:.3f}  "
              f"tau@20={point['tau_at_20']:.3f}  max_drop={point['max_drop']}")

    # ---- MTC diagnostic, C14 -------------------------------------------
    # Is the range the same signal as the variance?  Measured on the
    # top-20 set the adaptive policies START from, which is the all-LLM
    # leaderboard, so no human label enters.
    #
    # `spearman_nonzero` is the number to quote.  The pooled correlation
    # over all pairs is inflated by the 85-90 percent of pairs where both
    # statistics are exactly zero; restricting to pairs some system could
    # actually move is the honest comparison.
    top0 = np.argsort(-ndcg_l.mean(axis=1), kind="stable")[:K_TOP]
    lev0 = leverage_over(Wc, top0)
    mx0, mn0 = minmax_over(Wc, top0)
    rng0 = mx0 - mn0
    nzm = (lev0 > 0) | (rng0 > 0)

    def _sp(a, b):
        if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
            return float("nan")
        return float(spearmanr(a, b)[0])

    mtc_diag = {
        "spearman_all": _sp(lev0, rng0),
        "spearman_nonzero": _sp(lev0[nzm], rng0[nzm]),
        "kendall_nonzero": (float(kendalltau(lev0[nzm], rng0[nzm])[0])
                            if nzm.sum() >= 3 else float("nan")),
        "share_movable": float(nzm.mean()),
        # Where min > 0 the range is a genuine spread; elsewhere it is just
        # the max, i.e. "the best rank any target system gave this passage".
        "share_min_positive": float((mn0 > 0).mean()),
        "share_min_positive_of_movable": (float((mn0[nzm] > 0).mean())
                                          if nzm.any() else float("nan")),
        "n_top20_systems": int(len(top0)),
    }
    if verbose:
        print(f"  MTC diagnostic (top-20 at budget 0): "
              f"spearman(range, leverage) = {mtc_diag['spearman_nonzero']:.3f} "
              f"on the {100 * mtc_diag['share_movable']:.1f}% of pairs either "
              f"statistic can move")
        print(f"    min_s w > 0 for {100 * mtc_diag['share_min_positive']:.2f}% "
              f"of pairs, so the range is the max for the rest")

    # ---- error terms --------------------------------------------------
    if verbose:
        print("  error estimators...")
    e_raw = expected_sq_error_raw(universe, llm_grades, softmax_probs)
    e_cal, cal_info = expected_sq_error_calibrated(
        universe, llm_grades, human_qrels, softmax_probs, verbose=verbose)
    # ORACLE ONLY.  Feeds `product_oracle` and the online diagnostic.
    e_true = true_sq_error(universe, llm_grades, human_qrels)

    def _online(checkpoints=()):
        """A fresh online calibrator (C16).  One per policy run: its counts
        are its state, so sharing one would leak labels across policies."""
        return OnlineCalibratedError(universe, llm_grades, softmax_probs,
                                     human_qrels, checkpoints=checkpoints)

    _probe = _online()
    cal_info = dict(cal_info, online_n_bins=int(_probe.n_bins),
                    online_bins_match_offline=bool(
                        _probe.n_bins == cal_info["n_bins"]))
    if not cal_info["online_bins_match_offline"]:
        raise RuntimeError(
            f"online calibrator chose {_probe.n_bins} bins but the offline one "
            f"chose {cal_info['n_bins']}; the bin rule must be identical")
    del _probe

    # ---- orderings ----------------------------------------------------
    if verbose:
        print("  policy orderings...")
    qi_map = {q: i for i, q in enumerate(queries)}
    orderings, schedules = {}, {}

    def _static(name, s):
        # `s` is evaluated by the caller, BEFORE this body runs.  That
        # ordering matters: `random` passes rng.rand(...), which must be
        # drawn before the rng.randint below, exactly as in the previous
        # version.  The randint is drawn whether or not the policy is
        # built, so skipping one leaves the stream untouched for
        # everything downstream.
        s_rng = np.random.RandomState(rng.randint(0, 2 ** 31))
        if not want(name):
            return
        t = time.time()
        o = order_from_scores(universe, s, s_rng)
        orderings[name] = [(qi_map[q], p) for q, p in o]
        if verbose:
            print(f"    {name:<20s} {time.time() - t:5.1f}s")

    _static("random", rng.rand(n_universe))
    _static("naive", scores_naive_margin(universe, softmax_probs))
    _static("max_prob", scores_max_prob(universe, softmax_probs))
    _static("entropy", scores_entropy(universe, softmax_probs))
    _static("oracle", scores_oracle_error(universe, human_qrels, llm_grades))
    _static("depth_k", scores_depth_k(universe, pool_depth, pool_nsys))
    _static("retrieval_count", scores_retrieval_count(universe, pool_nsys))

    for name, term in [("leverage", None), ("product_raw", e_raw),
                       ("product_cal", e_cal)]:
        p_rng = np.random.RandomState(rng.randint(0, 2 ** 31))
        if not want(name):
            continue
        t = time.time()
        orderings[name] = run_adaptive_run_aware(
            universe, pair_index, Wc, grades_start, human_qrels, queries,
            system_names, sys_top10, error_term=term, M=K_TOP,
            batch_size=batch_size, stat="var", rng=p_rng)
        if verbose:
            print(f"    {name:<20s} {time.time() - t:5.1f}s")

    # ---- MTC, C14 ------------------------------------------------------
    # Own RandomState, keyed by (seed, year, name).  Nothing is drawn from
    # the shared `rng`, so adding these rows leaves every pre-existing
    # number bit-identical.
    if want("mtc_range_all"):
        t = time.time()
        # Static: with c_ij = 0 the MTC weights do not depend on any
        # judgment, so the published algorithm reduces to one fixed sort.
        o = order_from_scores(
            universe, range_over(Wc, range(n_sys)),
            np.random.RandomState(_stable_seed(seed, year, "mtc_range_all")))
        orderings["mtc_range_all"] = [(qi_map[q], p) for q, p in o]
        if verbose:
            print(f"    {'mtc_range_all':<20s} {time.time() - t:5.1f}s")

    for name, term in [("mtc_range", None), ("mtc_product_cal", e_cal)]:
        if not want(name):
            continue
        t = time.time()
        orderings[name] = run_adaptive_run_aware(
            universe, pair_index, Wc, grades_start, human_qrels, queries,
            system_names, sys_top10, error_term=term, M=K_TOP,
            batch_size=batch_size, stat="range",
            rng=np.random.RandomState(_stable_seed(seed, year, name)))
        if verbose:
            print(f"    {name:<20s} {time.time() - t:5.1f}s")

    # ---- online calibration, mean weight, product oracle: C16, C18, C19 --
    # Adaptive over the estimated top-K_TOP, stable per-policy seeds,
    # nothing drawn from the shared stream.
    online_models = {}
    for name, stat, ek in RUN_AWARE_SPECS:
        if not want(name):
            continue
        t = time.time()
        em = None
        if ek == "online":
            em = _online(ONLINE_CHECKPOINTS if name == "product_online" else ())
        orderings[name] = run_adaptive_run_aware(
            universe, pair_index, Wc, grades_start, human_qrels, queries,
            system_names, sys_top10,
            error_term=(e_true if ek == "true" else None), error_model=em,
            M=min(K_TOP, n_sys), batch_size=batch_size, stat=stat,
            rng=np.random.RandomState(_stable_seed(seed, year, name)))
        if em is not None:
            # Every label the calibrator read was a purchase, and every
            # purchase was read exactly once.
            assert em.n_label_reads == n_universe == int(em.observed.sum()), \
                f"{name}: {em.n_label_reads} label reads for {n_universe} pairs"
            online_models[name] = em
        if verbose:
            print(f"    {name:<20s} {time.time() - t:5.1f}s")

    # ---- targeting grid, C15 -------------------------------------------
    # Same RNG isolation as C14: stable per-policy seeds, nothing drawn
    # from the shared stream.
    targeting_meta = {}
    for name, stat, err_key, k, adaptive in TARGETING_SPECS:
        if not want(name):
            continue
        k_eff = n_sys if k is None else min(k, n_sys)
        # A target set covering every system cannot change as grades
        # arrive, so C is constant and the adaptive loop would be pure
        # overhead computing the identical ordering.
        adaptive_eff = bool(adaptive) and k_eff < n_sys
        p_rng = np.random.RandomState(_stable_seed(seed, year, name))
        t = time.time()
        if err_key == "e_online":
            # C16.  The error term changes every batch even when C does
            # not, so the static arm also runs the batch loop, with C
            # frozen at the budget-zero (all-LLM) top k.
            em = _online()
            fixed = (None if adaptive_eff else
                     np.argsort(-ndcg_l.mean(axis=1), kind="stable")[:k_eff])
            orderings[name] = run_adaptive_run_aware(
                universe, pair_index, Wc, grades_start, human_qrels, queries,
                system_names, sys_top10, error_model=em, M=k_eff,
                batch_size=batch_size, stat=stat, rng=p_rng,
                fixed_target=fixed)
            assert em.n_label_reads == n_universe
        elif adaptive_eff:
            term = {None: None, "e_cal": e_cal, "e_raw": e_raw}[err_key]
            orderings[name] = run_adaptive_run_aware(
                universe, pair_index, Wc, grades_start, human_qrels, queries,
                system_names, sys_top10, error_term=term, M=k_eff,
                batch_size=batch_size, stat=stat, rng=p_rng)
        else:
            # Static: C is computed once from the leaderboard available at
            # budget zero, which is the all-LLM one, then never updated.
            term = {None: None, "e_cal": e_cal, "e_raw": e_raw}[err_key]
            top0_k = np.argsort(-ndcg_l.mean(axis=1), kind="stable")[:k_eff]
            C0 = SPREAD_FNS[stat](Wc, top0_k)
            sc = C0 if term is None else C0 * term
            o = order_from_scores(universe, sc, p_rng)
            orderings[name] = [(qi_map[q], p) for q, p in o]
        targeting_meta[name] = {"stat": stat, "error": err_key or "none",
                                "target_k": k_eff,
                                "target_is_all": k_eff >= n_sys,
                                "adaptive": adaptive_eff,
                                "requested_k": "all" if k is None else k}
        if verbose:
            print(f"    {name:<20s} {time.time() - t:5.1f}s  "
                  f"(k={k_eff}{'' if k_eff < n_sys else '=all'}, "
                  f"{'adaptive' if adaptive_eff else 'static'})")

    if want("mtf"):
        t = time.time()
        orderings["mtf"] = run_mtf_policy(universe, human_qrels, queries,
                                          system_names, sys_top10, runs)
        if verbose:
            print(f"    {'mtf':<20s} {time.time() - t:5.1f}s")

    if include_mm_ns and want("mm_ns"):
        # Seeded from SEED + year, not from the shared stream.
        t = time.time()
        orderings["mm_ns"] = run_mm_ns_policy(universe, human_qrels, queries,
                                              system_names, runs,
                                              seed=SEED + year)
        if verbose:
            print(f"    {'mm_ns':<20s} {time.time() - t:5.1f}s")

    if want("lara"):
        # Seeded from SEED, not from the shared stream.
        t = time.time()
        lara_ord, lara_sched, lara_cum = run_lara_policy(
            universe, grades_start, human_qrels, softmax_probs, queries,
            system_names, sys_top10, batch_size, n_groups=lara_groups,
            seed=SEED)
        orderings["lara"] = lara_ord
        schedules["lara"] = (lara_sched, lara_cum)
        if verbose:
            print(f"    {'lara':<20s} {time.time() - t:5.1f}s")

    lara_nN_info = {}
    if want("lara_nN"):
        # C17.  One group per topic, water-filled.  Own stable seed.
        t = time.time()
        o_, s_, c_ = run_lara_policy(
            universe, grades_start, human_qrels, softmax_probs, queries,
            system_names, sys_top10, batch_size, n_groups=n_q,
            seed=_stable_seed(seed, year, "lara_nN"))
        orderings["lara_nN"] = o_
        schedules["lara_nN"] = (s_, c_)
        # How equal are the per-topic counts at each checkpoint?  Among
        # topics not yet exhausted the spread must be at most one.
        pool_q = np.bincount([qi_map[q] for q, _ in universe], minlength=n_q)
        worst = 0
        for row in c_:
            live = row < pool_q
            if live.sum() >= 2:
                worst = max(worst, int(row[live].max() - row[live].min()))
            # No live topic may trail an exhausted one by more than one: a
            # topic only exhausts while it is among the least served.
            if live.any() and (~live).any():
                assert row[~live].max() <= row[live].min() + 1, \
                    "a live topic was starved below an exhausted one"
        assert worst <= 1, f"lara_nN per-topic spread reached {worst}"
        lara_nN_info = {"max_live_spread": worst, "n_checkpoints": int(len(c_))}
        if verbose:
            print(f"    {'lara_nN':<20s} {time.time() - t:5.1f}s  "
                  f"(n = {n_q}, live-topic spread <= {worst})")

    for name, o in orderings.items():
        assert len(o) == n_universe, f"{name} ordering covers {len(o)}/{n_universe}"
        assert len(set(o)) == n_universe, f"{name} ordering has duplicates"

    # ---- C16 diagnostic: how well does the online estimator learn? -----
    # Measured on the pairs still UNBOUGHT at each checkpoint, which are
    # the pairs the estimate is actually used to choose between.  Uses the
    # true labels, so it is reported, never fed back.
    online_diag = []
    em = online_models.get("product_online")
    if em is not None:
        def _sp_safe(a, b):
            if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
                return float("nan")
            return float(spearmanr(a, b)[0])

        def _auc(score, y):
            y = np.asarray(y, dtype=bool)
            n1, n0 = int(y.sum()), int((~y).sum())
            if n1 == 0 or n0 == 0:
                return float("nan")
            r = rankdata(score)
            return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

        wrong = e_true > 0
        for snap in em.snapshots:
            U = snap["unbought"]
            if U.sum() < 3:
                continue
            est = snap["estimate"]
            lv = snap["level"][U]
            online_diag.append({
                "target_frac": snap["target_frac"], "frac_bought": snap["frac"],
                "n_unbought": int(U.sum()),
                "share_cell": float((lv == 0).mean()),
                "share_grade": float((lv == 1).mean()),
                "share_raw": float((lv == 2).mean()),
                "spearman_online_vs_offline": _sp_safe(est[U], e_cal[U]),
                "spearman_online_vs_true": _sp_safe(est[U], e_true[U]),
                "spearman_offline_vs_true": _sp_safe(e_cal[U], e_true[U]),
                "spearman_raw_vs_true": _sp_safe(e_raw[U], e_true[U]),
                "auc_online": _auc(est[U], wrong[U]),
                "auc_offline": _auc(e_cal[U], wrong[U]),
                "auc_raw": _auc(e_raw[U], wrong[U]),
                "mean_online": float(est[U].mean()),
                "mean_offline": float(e_cal[U].mean()),
                "mean_true": float(e_true[U].mean()),
            })
        if verbose and online_diag:
            last = [d for d in online_diag if d["target_frac"] <= 0.10]
            if last:
                d = last[-1]
                print(f"  online calibration at {100 * d['frac_bought']:.0f}%: "
                      f"AUC(err) online {d['auc_online']:.3f} / offline "
                      f"{d['auc_offline']:.3f} / raw {d['auc_raw']:.3f}; "
                      f"{100 * d['share_cell']:.0f}% of unbought pairs on a "
                      f"full cell")

    # ---- per-query nDCG schedules -------------------------------------
    if verbose:
        print("  per-query nDCG tables...")
    tables = {}
    for name, o in orderings.items():
        if name == "random":
            continue          # random gets its own bank of tables below
        tables[name] = build_per_query_ndcg_table(
            o, grades_start, human_qrels, queries, system_names, sys_top10)[0]

    rand_tables = [(orderings["random"], build_per_query_ndcg_table(
        orderings["random"], grades_start, human_qrels, queries, system_names,
        sys_top10)[0])]
    for _ in range(N_RAND_TABLES - 1):
        o = list(orderings["random"])
        rng.shuffle(o)
        rand_tables.append((o, build_per_query_ndcg_table(
            o, grades_start, human_qrels, queries, system_names, sys_top10)[0]))
    tables["random"] = rand_tables[0][1]

    precomp = time.time() - t0
    if verbose:
        print(f"  precompute {precomp:.1f}s")

    # ---- full-data curves ---------------------------------------------
    budgets = np.linspace(0, 1, N_BUDGET_STEPS)
    ones = np.ones(n_q)
    full_curves = {}

    # The random baseline is the MEAN over draws, per Sec. 4.4.  Averaging a
    # single draw would make the comparison depend on one shuffle.
    draws = [boot_correction_sweep(o, t, ones, system_names, gold_full, n_q,
                                   budgets) for o, t in rand_tables]
    rand_full = []
    for i, b in enumerate(budgets):
        row = {"budget": float(b)}
        for m in METRIC_KEYS:
            row[m] = float(np.mean([d[i][m] for d in draws]))
        rand_full.append(row)
    full_curves["random"] = rand_full

    if point["tau_at_20"] >= TAU_THRESHOLD:
        print(f"  WARNING: the all-LLM tau@20 for {year} is already at or "
              f"above {TAU_THRESHOLD}. Every policy will report a first-touch "
              f"budget of 0 and the table row is uninformative for this year.")
    for name in orderings:
        if name == "random":
            continue
        if name in schedules:
            full_curves[name] = boot_schedule_sweep(*schedules[name], ones,
                                                    system_names, gold_full,
                                                    n_q, budgets)
        else:
            full_curves[name] = boot_correction_sweep(
                orderings[name], tables[name], ones, system_names, gold_full,
                n_q, budgets)

    # ---- query-level orders -------------------------------------------
    dmg_path = DAMAGE_DIR / f"{year}_query_damage.csv"
    impact_order, reliab_order = [], []
    if dmg_path.exists():
        d = pd.read_csv(dmg_path, dtype={"query_id": str})
        col = "pearson" if "pearson" in d.columns else "spearman"
        m = {r["query_id"]: (r["damage_all"], r[col]) for _, r in d.iterrows()}
        ql = [q for q in queries if q in m]
        impact_order = [qi_map[q] for q in sorted(ql, key=lambda x: m[x][0])]
        reliab_order = [qi_map[q] for q in sorted(ql, key=lambda x: -m[x][1])]
    elif verbose:
        print(f"  no {dmg_path.name}; query-level oracles skipped")

    # ---- bootstrap -----------------------------------------------------
    if verbose:
        print(f"  bootstrap B={B} ", end="", flush=True)
    tb = time.time()

    pol_names = [p for p in TABLE_POLICIES if p in orderings] + \
                [p for p in TARGETING_POLICIES if p in orderings] + \
                [p for p in EXTRA_POLICIES if p in orderings]
    stor = {p: defaultdict(list) for p in pol_names + ["all_llm"]}
    stor_ql = {p: defaultdict(list) for p in ["damage_oracle", "reliability_oracle"]}
    paired = defaultdict(list)
    bystep = {p: defaultdict(list) for p in pol_names}

    PAIRS = [("leverage", "lara"), ("leverage", "naive"), ("mtf", "lara"),
             ("mtf", "naive"), ("product_cal", "leverage"),
             ("product_raw", "leverage"), ("product_cal", "lara"),
             # C14.  These three are the comparisons the MTC objection
             # turns on: same target set, same batching, only the moment
             # of the weight vector differs.
             ("leverage", "mtc_range"),
             ("product_cal", "mtc_product_cal"),
             ("leverage", "mtc_range_all"),
             # C16-C19.  The headline against every published baseline,
             # every ablation, the leaky reference and the ceiling.
             ("product_online", "random"),
             ("product_online", "depth_k"),
             ("product_online", "mtf"),
             ("product_online", "naive"),
             ("product_online", "lara"),
             ("product_online", "lara_nN"),
             ("product_online", "mtc_range_all"),
             ("product_online", "mtc_range"),
             ("product_online", "mtc_product_online"),
             ("product_online", "leverage"),
             ("product_online", "meanw_online"),
             ("product_online", "product_raw"),
             ("product_online", "product_cal"),
             ("product_oracle", "product_online"),
             ("leverage", "meanw_k20"),
             ("lara_nN", "lara"),
             ("prod_on_k20", "prod_on_kall"),
             ("prod_on_k20", "prod_on_k20_static")]

    # C20.  Index of each fixed budget on the sweep grid.
    fb_idx = {b: int(np.argmin(np.abs(budgets - b))) for b in TARGET_BUDGETS}

    def _fb_key(metric, b):
        return f"{metric}@{int(round(100 * b))}"

    for it in range(B):
        idx = rng.choice(n_q, size=n_q, replace=True)
        cnt = np.bincount(idx, minlength=n_q).astype(float)

        gold_b = rank_systems(ndcg_h @ cnt / n_q, system_names)
        llm_b  = rank_systems(ndcg_l @ cnt / n_q, system_names)
        for _m, _v in ranking_metrics(gold_b, llm_b).items():
            stor["all_llm"][_m].append(_v)

        ro, rt = rand_tables[it % N_RAND_TABLES]
        rand_curve = boot_correction_sweep(ro, rt, cnt, system_names, gold_b,
                                           n_q, budgets)
        curves = {"random": rand_curve}
        for name in pol_names:
            if name == "random":
                continue
            if name in schedules:
                curves[name] = boot_schedule_sweep(*schedules[name], cnt,
                                                   system_names, gold_b, n_q,
                                                   budgets)
            else:
                curves[name] = boot_correction_sweep(orderings[name],
                                                     tables[name], cnt,
                                                     system_names, gold_b, n_q,
                                                     budgets)

        rv20 = np.array([r["tau_at_20"] for r in rand_curve])
        for name, c in curves.items():
            if name not in stor:
                continue
            ft, ok_ft = threshold_first_touch(c)
            su, ok_su = threshold_sustained(c)
            stor[name]["first_touch"].append(ft)
            stor[name]["first_touch_ok"].append(ok_ft)
            stor[name]["sustained"].append(su)
            stor[name]["sustained_ok"].append(ok_su)
            stor[name]["area_tau20"].append(area_between(c, rand_curve, "tau_at_20"))
            stor[name]["area_tau_all"].append(area_between(c, rand_curve, "tau_all"))
            v20 = np.array([r["tau_at_20"] for r in c])
            for ti, bf in enumerate(budgets):
                bystep[name][bf].append(v20[ti] - rv20[ti])
            for b_fix, j in fb_idx.items():
                for m in FIXED_METRICS:
                    stor[name][_fb_key(m, b_fix)].append(c[j][m])

        for a, b_ in PAIRS:
            if a not in curves or b_ not in curves:
                continue
            if a not in stor or b_ not in stor:
                continue
            fa, oa = threshold_first_touch(curves[a])
            fb, ob = threshold_first_touch(curves[b_])
            if oa and ob:
                paired[(a, b_, "first_touch_tau20")].append(fa - fb)
            paired[(a, b_, "area_tau20")].append(
                stor[a]["area_tau20"][-1] - stor[b_]["area_tau20"][-1])
            for b_fix, j in fb_idx.items():
                for m in FIXED_METRICS:
                    paired[(a, b_, _fb_key(m, b_fix))].append(
                        curves[a][j][m] - curves[b_][j][m])

        if impact_order:
            qlc = np.where(cnt > 0, cnt, 0.0)
            gold_ql = rank_systems(ndcg_h @ qlc / max(qlc.sum(), 1), system_names)
            rnd = list(range(n_q))
            rng.shuffle(rnd)
            rand_ql = ql_triage_sweep(ndcg_h, ndcg_l, rnd, qlc, system_names,
                                      gold_ql, n_q)
            for nm, order in [("damage_oracle", impact_order),
                              ("reliability_oracle", reliab_order)]:
                c = ql_triage_sweep(ndcg_h, ndcg_l, order, qlc, system_names,
                                    gold_ql, n_q)
                stor_ql[nm]["area_tau20"].append(area_between(c, rand_ql, "tau_at_20"))
                stor_ql[nm]["area_tau_all"].append(area_between(c, rand_ql, "tau_all"))

        if verbose and (it + 1) % 50 == 0:
            print(f"{it + 1} ", end="", flush=True)

    boot_time = time.time() - tb
    if verbose:
        print(f"\n  bootstrap {boot_time:.1f}s ({boot_time / B:.2f}s/iter)")

    return {
        "year": year, "n_q": n_q, "n_sys": n_sys, "n_universe": n_universe,
        "n_missing": n_missing, "batch_size": batch_size,
        "zero_leverage_share": zero_lev, "cal_info": cal_info,
        "mtc_diag": mtc_diag, "targeting_meta": targeting_meta,
        "online_diag": online_diag, "lara_nN_info": lara_nN_info,
        "point": point, "full_curves": full_curves,
        "stor": stor, "stor_ql": stor_ql, "paired": dict(paired),
        "bystep": {k: dict(v) for k, v in bystep.items()},
        "precomp_time": precomp, "boot_time": boot_time,
        "policies": pol_names,
    }


# ---------------------------------------------------------------------------
#  STRUCTURAL CORRELATIONS
# ---------------------------------------------------------------------------

def run_structural_bootstrap(B, rng_master, years):
    rows = []
    rng = np.random.RandomState(rng_master.randint(0, 2 ** 31))
    ddec = pd.read_csv(DDEC_CSV, dtype={"query_id": str}) if DDEC_CSV.exists() else None

    for year in years:
        if ddec is not None:
            y = ddec[ddec["year"] == year]
            for col, label in [("R_q_pearson", "agreement_pearson"),
                               ("R_q_spearman", "agreement_spearman")]:
                if col not in y.columns:
                    continue
                d = y[[col, "D_q"]].dropna()
                R, D = d[col].values, d["D_q"].values
                if len(R) < 3:
                    continue
                boot = [spearmanr(R[s], D[s])[0] for s in
                        (rng.choice(len(R), len(R), True) for _ in range(B))]
                rows.append({"year": year,
                             "quantity": f"spearman({label}, damage)",
                             "resample_unit": "query",
                             "point": float(spearmanr(R, D)[0]),
                             "lo_2p5": np.nanpercentile(boot, 2.5),
                             "median": np.nanpercentile(boot, 50),
                             "hi_97p5": np.nanpercentile(boot, 97.5)})

        sys_path = SPECTRAL_INT / str(year) / "systems.parquet"
        if not sys_path.exists():
            continue
        sdf = pd.read_parquet(sys_path)
        n_s = len(sdf)
        pairs = [("score_bias", "displacement_dcg", "score_bias_vs_score_shift"),
                 ("score_bias", "baseline_quality", "score_bias_vs_system_quality")]
        for xa, xb, label in pairs:
            if xa not in sdf.columns or xb not in sdf.columns:
                continue
            a, b = sdf[xa].values, sdf[xb].values
            samples = [rng.choice(n_s, n_s, True) for _ in range(B)]
            # Pearson is the primary statistic.  Spearman is reported beside
            # it; the previous version computed Spearman under a Pearson label.
            for fn, tag in [(pearsonr, "pearson"), (spearmanr, "spearman")]:
                boot = [fn(a[s], b[s])[0] for s in samples]
                rows.append({"year": year, "quantity": f"{label}_{tag}",
                             "resample_unit": "system",
                             "point": float(fn(a, b)[0]),
                             "lo_2p5": np.nanpercentile(boot, 2.5),
                             "median": np.nanpercentile(boot, 50),
                             "hi_97p5": np.nanpercentile(boot, 97.5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  CI HELPERS
# ---------------------------------------------------------------------------

def pct(arr, p):
    a = [x for x in arr if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.percentile(a, p)) if a else np.nan


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=B_INITIAL)
    ap.add_argument("--years", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--judge", choices=sorted(JUDGES), default="llama",
                    help="LLM whose grades start the collection.  Output "
                         "goes to results/t12_resampling_<judge>/ for any "
                         "judge but llama.")
    ap.add_argument("--include-mm-ns", action="store_true")
    ap.add_argument("--lara-groups", type=int, default=1,
                    help="n for the `lara` row.  1 = single pool (the "
                         "default).  LARA(n=N) has its own row, `lara_nN`, "
                         "so this flag is only for other n.")
    ap.add_argument("--policies", nargs="+", default=None,
                    metavar="NAME",
                    help="Only compute these policies.  'random' is always "
                         "added because it is the reference curve for every "
                         "area and every by-budget difference.  Skipping a "
                         "policy does not change any other policy's numbers: "
                         "the shared RNG is advanced identically and every "
                         "policy added since C14 uses its own deterministic "
                         "seed.  A PAIRED difference needs both of its "
                         "policies in the same run.  Shortcuts: "
                         + "; ".join(f"'{k}' = {len(v)} rows"
                                     for k, v in POLICY_SHORTCUTS.items())
                         + ".  Known names: "
                         + ", ".join(sorted(KNOWN_POLICIES)))
    ap.add_argument("--out-suffix", default=None, metavar="TAG",
                    help="Write to results/.../t12_resampling_TAG/ instead "
                         "of the default directory, so an incremental run "
                         "cannot overwrite results you already have.")
    args = ap.parse_args()

    policies = args.policies
    if policies is not None:
        expanded = []
        for p in policies:
            if p in POLICY_SHORTCUTS:
                expanded.extend(POLICY_SHORTCUTS[p])
            else:
                expanded.append(p)
        known = KNOWN_POLICIES
        unknown = [p for p in expanded if p not in known]
        if unknown:
            raise SystemExit(f"unknown policy name(s): {', '.join(unknown)}\n"
                             f"known: {', '.join(sorted(known))}\n"
                             f"shortcuts: {', '.join(POLICY_SHORTCUTS)}")
        policies = sorted(set(expanded) | {"random"})

    need_lara = policies is None or bool({"lara", "lara_nN"} & set(policies))
    if need_lara and not SKLEARN_AVAILABLE:
        raise SystemExit("scikit-learn is required for the LARA calibrator. "
                         "pip install scikit-learn")

    global OUTPUT_DIR
    set_judge(args.judge)
    if args.out_suffix:
        OUTPUT_DIR = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_{args.out_suffix}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.RandomState(args.seed)
    years = args.years or sorted(YEARS_CFG)
    B = args.B

    print(f"=== t12_resampling  B={B}  years={years}  seed={args.seed} ===")
    print("    gain = LINEAR (g)\n")

    res_by_year, wall = {}, {}
    for i, year in enumerate(years):
        print(f"\n{'-' * 62}\n  {year}\n{'-' * 62}")
        r = run_year(year, YEARS_CFG[year], B, rng_master,
                     include_mm_ns=args.include_mm_ns,
                     lara_groups=args.lara_groups,
                     policies=policies, seed=args.seed)
        res_by_year[year] = r
        wall[year] = r["boot_time"] + r["precomp_time"]
        if i == 0:
            mins = wall[year] / 60
            print(f"\n  *** TIMING: {mins:.1f} min for B={B}. "
                  f"Five years at B=1000 projects to "
                  f"{5 * (r['boot_time'] / B) * 1000 / 60 + 5 * r['precomp_time'] / 60:.0f} min ***")
            if mins < 30 and B < B_FULL and len(years) == 1:
                print(f"  Under 30 min. Re-run with --B {B_FULL}.")

    print(f"\n{'-' * 62}\n  structural correlations\n{'-' * 62}")
    struct = run_structural_bootstrap(B, rng_master, years)

    # ---- endpoints -----------------------------------------------------
    endpoints = []
    ENDPOINT_METRICS = METRIC_KEYS
    for y, r in res_by_year.items():
        for m in ENDPOINT_METRICS:
            a = r["stor"]["all_llm"][m]
            endpoints.append({"year": y, "metric": m, "point": r["point"][m],
                              "lo_2p5": pct(a, 2.5), "median": pct(a, 50),
                              "hi_97p5": pct(a, 97.5)})
    pd.DataFrame(endpoints).to_csv(OUTPUT_DIR / "endpoints_ci.csv", index=False)

    # ---- thresholds ----------------------------------------------------
    thresholds, table_cells = [], {}
    for y, r in res_by_year.items():
        for pol in r["policies"]:
            row = {"year": y, "policy": pol, "display": DISPLAY_NAME.get(pol, pol),
                   "reads": READS.get(pol, "")}
            for kind in ("first_touch", "sustained"):
                a = r["stor"][pol][kind]
                ok = r["stor"][pol][kind + "_ok"]
                ft_full, ft_ok = (threshold_first_touch(r["full_curves"][pol])
                                  if kind == "first_touch"
                                  else threshold_sustained(r["full_curves"][pol]))
                span = pct(a, 97.5) - pct(a, 2.5)
                row.update({
                    f"{kind}_point": None if ft_full is None else 100 * ft_full,
                    f"{kind}_reached": bool(ft_ok),
                    f"{kind}_lo_2p5": 100 * pct(a, 2.5),
                    f"{kind}_median": 100 * pct(a, 50),
                    f"{kind}_hi_97p5": 100 * pct(a, 97.5),
                    f"{kind}_frac_never": 1 - (sum(ok) / max(len(ok), 1)),
                    f"{kind}_span_pp": 100 * span if not np.isnan(span) else np.nan,
                })
            thresholds.append(row)
            table_cells[(y, pol)] = row["first_touch_point"]
    thr_df = pd.DataFrame(thresholds)
    thr_df.to_csv(OUTPUT_DIR / "thresholds_ci.csv", index=False)

    # ---- the table -----------------------------------------------------
    # Only the policies actually computed get a row, so an incremental run
    # produces a paste-ready fragment rather than a table of dashes.
    ran = set()
    for r in res_by_year.values():
        ran.update(r["policies"])
    tab_rows = []
    for pol in [p for p in TABLE_POLICIES if p in ran]:
        row = {"policy": DISPLAY_NAME.get(pol, pol), "reads": READS.get(pol, "")}
        for y in years:
            v = table_cells.get((y, pol))
            row[str(y)] = "---" if v is None else f"{v:.0f}"
        tab_rows.append(row)
    tab_df = pd.DataFrame(tab_rows)
    tab_df.to_csv(OUTPUT_DIR / "budget_table.csv", index=False)

    tex = ["\\begin{table}[h]", "\\centering",
           "\\begin{tabular}{ll" + "c" * len(years) + "}", "\\hline",
           "Policy & Reads & " + " & ".join(str(y) for y in years) + " \\\\",
           "\\hline"]
    for r_ in tab_rows:
        tex.append(f"{r_['policy']} & {r_['reads']} & "
                   + " & ".join(r_[str(y)] for y in years) + " \\\\")
    tex += ["\\hline", "\\end{tabular}",
            "\\caption{Percentage of judged pairs verified before $\\tau@20$ "
            "first reaches 0.95.}",
            "\\label{tab:budget-first-touch}", "\\end{table}"]
    (OUTPUT_DIR / "budget_table.tex").write_text("\n".join(tex), encoding="utf-8")

    # ---- areas ---------------------------------------------------------
    areas = []
    for y, r in res_by_year.items():
        # The oracle is the denominator of ratio_to_oracle.  With
        # --policies it may not have been run, in which case the ratios
        # are simply absent rather than an error.
        _orc_store = r["stor"].get("oracle", {})
        orc = {m: (_orc_store.get(f"area_{m}", []) if _orc_store else [])
               for m in ("tau20", "tau_all")}
        # C18.  Second denominator: the headline with the true error term.
        _porc_store = r["stor"].get("product_oracle", {})
        porc = {m: (_porc_store.get(f"area_{m}", []) if _porc_store else [])
                for m in ("tau20", "tau_all")}

        def _ratios(a, o):
            return [a[i] / o[i] for i in range(min(len(a), len(o)))
                    if o[i] is not None and abs(o[i]) > 1e-12]

        for pol in r["policies"]:
            for metric, key in (("tau_at_20", "area_tau20"),
                                ("tau_all", "area_tau_all")):
                a = r["stor"][pol][key]
                if not a:
                    continue
                mk = "tau20" if metric == "tau_at_20" else "tau_all"
                ratios = _ratios(a, orc[mk])
                pratios = _ratios(a, porc[mk])
                areas.append({"year": y, "policy": pol, "metric": metric,
                              "area_vs_random": pct(a, 50),
                              "lo_2p5": pct(a, 2.5), "hi_97p5": pct(a, 97.5),
                              "ratio_to_oracle": pct(ratios, 50),
                              "ratio_lo": pct(ratios, 2.5),
                              "ratio_hi": pct(ratios, 97.5),
                              "ratio_to_product_oracle": pct(pratios, 50),
                              "pratio_lo": pct(pratios, 2.5),
                              "pratio_hi": pct(pratios, 97.5)})
        for nm in ("damage_oracle", "reliability_oracle"):
            for metric, key in (("tau_at_20", "area_tau20"),
                                ("tau_all", "area_tau_all")):
                a = r["stor_ql"][nm][key]
                if not a:
                    continue
                areas.append({"year": y, "policy": nm, "metric": metric,
                              "area_vs_random": pct(a, 50),
                              "lo_2p5": pct(a, 2.5), "hi_97p5": pct(a, 97.5),
                              "ratio_to_oracle": np.nan,
                              "ratio_lo": np.nan, "ratio_hi": np.nan,
                              "ratio_to_product_oracle": np.nan,
                              "pratio_lo": np.nan, "pratio_hi": np.nan})
    pd.DataFrame(areas).to_csv(OUTPUT_DIR / "area_ci.csv", index=False)

    # ---- fixed-budget points and intervals, C20 --------------------------
    def _curve_at(curve, b):
        bb = np.array([c["budget"] for c in curve])
        return curve[int(np.argmin(np.abs(bb - b)))]

    fixed_rows = []
    for y, r in res_by_year.items():
        for pol in r["policies"]:
            fc = r["full_curves"].get(pol)
            for b_fix in TARGET_BUDGETS:
                for m in FIXED_METRICS:
                    key = f"{m}@{int(round(100 * b_fix))}"
                    a = r["stor"][pol].get(key, [])
                    fixed_rows.append({
                        "year": y, "policy": pol,
                        "display": DISPLAY_NAME.get(pol, pol),
                        "reads": READS.get(pol, ""),
                        "metric": m, "budget_pct": int(round(100 * b_fix)),
                        "point": (float(_curve_at(fc, b_fix)[m])
                                  if fc else np.nan),
                        "lo_2p5": pct(a, 2.5), "median": pct(a, 50),
                        "hi_97p5": pct(a, 97.5)})
    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(OUTPUT_DIR / "fixed_budget_ci.csv", index=False)

    # ---- paired differences --------------------------------------------
    pairs_out = []
    for y, r in res_by_year.items():
        for (a, b_, q), d in r["paired"].items():
            if not d:
                continue
            lo, hi = pct(d, 2.5), pct(d, 97.5)
            # Full-data difference, available for the fixed-budget rows.
            point = np.nan
            if "@" in q:
                m, bp = q.split("@")
                ca, cb = r["full_curves"].get(a), r["full_curves"].get(b_)
                if ca and cb:
                    point = float(_curve_at(ca, int(bp) / 100)[m]
                                  - _curve_at(cb, int(bp) / 100)[m])
            pairs_out.append({"year": y, "policy_a": a, "policy_b": b_,
                              "quantity": q, "point": point,
                              "mean_diff": float(np.mean(d)),
                              "lo_2p5": lo, "hi_97p5": hi,
                              "excludes_zero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(pairs_out).to_csv(OUTPUT_DIR / "paired_differences.csv", index=False)

    # ---- online calibration diagnostic, C16 -----------------------------
    od_rows = []
    for y, r in res_by_year.items():
        for d in r.get("online_diag", []):
            od_rows.append(dict(year=y, policy="product_online", **d))
    od_df = pd.DataFrame(od_rows)
    if len(od_df):
        od_df.to_csv(OUTPUT_DIR / "online_calibration_diag.csv", index=False)

    # ---- difference by budget ------------------------------------------
    bystep_out = []
    for y, r in res_by_year.items():
        for pol, d in r["bystep"].items():
            for bf, arr in d.items():
                if not arr:
                    continue
                lo, hi = pct(arr, 2.5), pct(arr, 97.5)
                bystep_out.append({"year": y, "policy": pol, "metric": "tau_at_20",
                                   "budget_pct": round(100 * float(bf), 1),
                                   "mean_diff_vs_random": float(np.mean(arr)),
                                   "lo_2p5": lo, "hi_97p5": hi,
                                   "excludes_zero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(bystep_out).to_csv(OUTPUT_DIR / "difference_by_budget.csv",
                                    index=False)

    # ---- full-data curves ----------------------------------------------
    METRIC_COLS = METRIC_KEYS
    cur = []
    for y, r in res_by_year.items():
        for pol, c in r["full_curves"].items():
            for row in c:
                d = {"year": y, "policy": pol,
                     "budget_pct": round(100 * row["budget"], 1)}
                d.update({m: row.get(m) for m in METRIC_COLS})
                cur.append(d)
    pd.DataFrame(cur).to_csv(OUTPUT_DIR / "curves_full.csv", index=False)

    # ---- targeting matrix, C15 -----------------------------------------
    # Read at fixed budgets, not at a threshold crossing.  See the note on
    # TARGET_BUDGETS for why.
    tgt_rows = []
    for y, r in res_by_year.items():
        meta = r.get("targeting_meta", {})
        if not meta:
            continue
        for pol, m in meta.items():
            curve = r["full_curves"].get(pol)
            if not curve:
                continue
            bb = np.array([c["budget"] for c in curve])
            for want_b in TARGET_BUDGETS:
                j = int(np.argmin(np.abs(bb - want_b)))
                row = {"year": y, "policy": pol, "stat": m["stat"],
                       "error": m["error"], "target_k": m["target_k"],
                       "target_is_all": m["target_is_all"],
                       "adaptive": m["adaptive"],
                       "budget_pct": round(100 * curve[j]["budget"], 1)}
                row.update({mm: curve[j].get(mm) for mm in METRIC_COLS})
                tgt_rows.append(row)
    tgt_df = pd.DataFrame(tgt_rows)
    if len(tgt_df):
        tgt_df.to_csv(OUTPUT_DIR / "targeting_matrix.csv", index=False)

    # The diagonal test.  For each (year, budget, statistic family), which
    # target size wins on each evaluation cutoff?  If targeting works, the
    # winner on tau@10 is the k=10 policy, on tau@20 the k=20 policy, and
    # on tau_all the all-systems policy.
    diag_rows = []
    if len(tgt_df):
        for (y, stat, err, b), grp in tgt_df.groupby(
                ["year", "stat", "error", "budget_pct"]):
            for metric in ("tau_at_10", "tau_at_20", "tau_all"):
                sub = grp.dropna(subset=[metric])
                if sub.empty:
                    continue
                best = sub.loc[sub[metric].idxmax()]
                # Which target size SHOULD win on this metric?
                if metric == "tau_all":
                    predicted_ok = bool(best["target_is_all"])
                else:
                    k_needed = 10 if metric == "tau_at_10" else 20
                    predicted_ok = (int(best["target_k"]) == k_needed
                                    and not best["target_is_all"])
                diag_rows.append({
                    "year": y, "stat": stat, "error": err, "budget_pct": b,
                    "evaluated_on": metric, "winner": best["policy"],
                    "winner_target_k": int(best["target_k"]),
                    "winner_value": float(best[metric]),
                    "matches_prediction": predicted_ok})
        pd.DataFrame(diag_rows).to_csv(OUTPUT_DIR / "targeting_diagonal.csv",
                                       index=False)

    struct.to_csv(OUTPUT_DIR / "structural_ci.csv", index=False)

    # ---- config --------------------------------------------------------
    wide = thr_df[thr_df["first_touch_span_pp"] > 30][
        ["year", "policy", "first_touch_span_pp"]]

    cfg_md = [
        "# bootstrap_config.md", "",
        f"- judge: {args.judge} ({JUDGES[args.judge]})",
        f"- B: {B}", f"- seed: {args.seed}", f"- budget grid: {N_BUDGET_STEPS} steps",
        f"- threshold: tau@20 >= {TAU_THRESHOLD}", f"- K: {K_TOP}",
        f"- adaptive batch: {BATCH_FRACTION:.0%} of the pool",
        f"- random orderings cycled: {N_RAND_TABLES}",
        f"- LARA(n) for the `lara` row: {args.lara_groups}; "
        "`lara_nN` is n = number of topics, water-filled (C17)",
        "- gain: LINEAR, gain(g) = g",
        f"- fixed budgets (C20): {', '.join(f'{100 * b:.0f}%' for b in TARGET_BUDGETS)}",
        f"- calibration: min cell {CAL_MIN_CELL}, bins tried {CAL_BIN_TRY}", "",
        "## Calibration, C16", "",
        "`product_cal`, `mtc_product_cal` and `prod_k*` use the OFFLINE",
        "estimator, fitted leave-one-query-out on the human grades of every",
        "other query in the year.  Those labels are outside the budget.",
        "They are kept only to size the leak.", "",
        "`product_online`, `mtc_product_online`, `meanw_online` and",
        "`prod_on_*` use the ONLINE estimator: the table is rebuilt after",
        "every batch from purchased pairs only.  Same bins (asserted per",
        "year), same backoff, same minimum cell.  The first batch is scored",
        "with the raw softmax E[eps^2].  Every run asserts that the number",
        "of label reads equals the number of purchases.", "",
        "## Wall clock", ""]
    for y, t in wall.items():
        cfg_md.append(f"- {y}: {t / 60:.1f} min")
    cfg_md += ["", "## Per year", "",
               "| year | queries | systems | pairs | no LLM grade | C_pp = 0 | cal bins |",
               "|---|---|---|---|---|---|---|"]
    for y, r in res_by_year.items():
        cfg_md.append(f"| {y} | {r['n_q']} | {r['n_sys']} | {r['n_universe']} | "
                      f"{r['n_missing']} | {100 * r['zero_leverage_share']:.1f}% | "
                      f"{r['cal_info']['n_bins']} |")
    cfg_md += ["", "## Checks asserted during the run", "",
               "| year | online bins = offline bins | lara_nN live-topic spread "
               "| lara_nN checkpoints |", "|---|---|---|---|"]
    for y, r in res_by_year.items():
        li = r.get("lara_nN_info", {})
        cfg_md.append(
            f"| {y} | {r['cal_info'].get('online_bins_match_offline', '---')} "
            f"| {li.get('max_live_spread', '---')} "
            f"| {li.get('n_checkpoints', '---')} |")

    cfg_md += ["", "## MTC diagnostic (C14): range against variance", "",
               "Measured on the top-20 set the adaptive policies start from,",
               "which is the all-LLM leaderboard, so no human label enters.",
               "`spearman_nonzero` is the number to quote: the pooled",
               "correlation over all pairs is inflated by the large majority",
               "where both statistics are exactly zero.", "",
               "Where min_s w_sp = 0 the range is not a spread at all, it is",
               "just max_s w_sp, i.e. the best rank any target system gave the",
               "passage.  A low share of positive minima is the mechanical",
               "reason the two criteria can behave differently.", "",
               "| year | spearman (movable) | kendall (movable) | movable share "
               "| min > 0 | min > 0 of movable |",
               "|---|---|---|---|---|---|"]
    for y, r in res_by_year.items():
        d = r["mtc_diag"]
        cfg_md.append(
            f"| {y} | {d['spearman_nonzero']:.3f} | {d['kendall_nonzero']:.3f} "
            f"| {100 * d['share_movable']:.1f}% "
            f"| {100 * d['share_min_positive']:.2f}% "
            f"| {100 * d['share_min_positive_of_movable']:.2f}% |")

    cfg_md += ["", "## Design decision (a): duplicated queries", "",
               "A query drawn k times contributes k copies of its nDCG to the",
               "mean.  Its pairs appear once in the correction universe, so the",
               "budget denominator counts unique pairs.  The mean nDCG",
               "denominator stays at the full query count.", "",
               "## Design decision (b): policy ordering", "",
               "RESTRICT throughout.  The acquisition ordering is computed once",
               "on the full data and restricted to the sampled pairs.", "",
               "Exact for random, naive, max_prob, entropy, oracle, depth_k and",
               "retrieval_count, whose scores are per-pair quantities.", "",
               "NOT exact, and therefore UNDERSTATING uncertainty, for:", "",
               "- every adaptive run-aware policy (leverage, product_*,",
               "  mtc_range, mtc_product_*, meanw_*, lev_k*/mtc_k*/prod_k*",
               "  adaptive arms): the target top-K set is recomputed from the",
               "  current leaderboard, which spans queries.",
               "- every online-calibrated policy, static arms included: the",
               "  calibration table pools purchases across queries.",
               "- mtf, mm_ns: the run priority queue carries cross-query state.",
               "- lara, lara_nN: the calibrator is fitted on labels from every",
               "  query and its imputation rewrites every unbought grade.", "",
               "This applies equally to the fixed-budget intervals and paired",
               "differences of C20.", "",
               "## Design decision (c): B", "",
               f"Started at {B_INITIAL}.  Raise to {B_FULL} once year one fits in",
               "thirty minutes.", ""]

    if len(wide):
        cfg_md += ["## FLAG: first-touch intervals wider than 30 points", ""]
        for _, w in wide.iterrows():
            cfg_md.append(f"- {int(w['year'])}, {w['policy']}: "
                          f"{w['first_touch_span_pp']:.0f} points")
        cfg_md += ["", "These cells are not stable enough to quote as integers.", ""]

    (OUTPUT_DIR / "bootstrap_config.md").write_text("\n".join(cfg_md), encoding="utf-8")

    # ---- report --------------------------------------------------------
    rep = ["# REPORT.md, t12_resampling", "",
           f"B = {B}, seed = {args.seed}, linear gain, tau@20 threshold "
           f"{TAU_THRESHOLD}.", "",
           "## The table, first touch, percent of judged pairs", "",
           "| Policy | Reads | " + " | ".join(str(y) for y in years) + " |",
           "|" + "---|" * (2 + len(years))]
    for r_ in tab_rows:
        rep.append(f"| {r_['policy']} | {r_['reads']} | "
                   + " | ".join(r_[str(y)] for y in years) + " |")

    # ---- C20: fixed budgets ---------------------------------------------
    REPORT_BUDGETS = (10, 15, 20)
    if len(fixed_df):
        rep += ["", "## At fixed budgets (C20): full-data point [95% CI]", "",
                "First touch is a crossing statistic with intervals up to ~100",
                "points wide.  These are the numbers to compare policies on.",
                "Intervals are RESTRICT bootstrap over queries; they understate",
                "uncertainty for adaptive policies (bootstrap_config.md).", ""]
        for m, label in (("tau_at_20", "tau@20"), ("tau_ap", "tau_AP"),
                         ("tau_all", "tau_all"),
                         (f"tau_union_{K_TOP}", f"tau over true+predicted top-{K_TOP}"),
                         (f"overlap_at_{K_TOP}", f"top-{K_TOP} overlap")):
            for bp in REPORT_BUDGETS:
                sub = fixed_df[(fixed_df.metric == m) & (fixed_df.budget_pct == bp)]
                if sub.empty:
                    continue
                rep += [f"### {label} at {bp}%", "",
                        "| Policy | Reads | " + " | ".join(str(y) for y in years) + " |",
                        "|" + "---|" * (2 + len(years))]
                for pol in [p for p in TABLE_POLICIES if p in ran]:
                    cells = []
                    for y in years:
                        row = sub[(sub.year == y) & (sub.policy == pol)]
                        if row.empty:
                            cells.append("---")
                            continue
                        row = row.iloc[0]
                        cells.append(f"{row['point']:.3f} "
                                     f"[{row['lo_2p5']:.3f}, {row['hi_97p5']:.3f}]")
                    rep.append(f"| {DISPLAY_NAME.get(pol, pol)} | "
                               f"{READS.get(pol, '')} | " + " | ".join(cells) + " |")
                rep.append("")

    pdf = pd.DataFrame(pairs_out)
    if len(pdf):
        rep += ["## Paired differences at fixed budgets (C20)", "",
                "A minus B, full-data point [95% CI], computed within each",
                "replicate.  `*` marks an interval that excludes zero.", ""]
        order = []
        for a, b_ in [(r0["policy_a"], r0["policy_b"]) for r0 in pairs_out]:
            if (a, b_) not in order:
                order.append((a, b_))
        for m, label in (("tau_at_20", "tau@20"), ("tau_ap", "tau_AP"),
                         (f"tau_union_{K_TOP}", f"tau over true+predicted top-{K_TOP}")):
            for bp in (10, 15):
                q = f"{m}@{bp}"
                sub = pdf[pdf.quantity == q]
                if sub.empty:
                    continue
                rep += [f"### {label} at {bp}%", "",
                        "| A | B | " + " | ".join(str(y) for y in years) + " |",
                        "|" + "---|" * (2 + len(years))]
                for a, b_ in order:
                    cells, any_row = [], False
                    for y in years:
                        row = sub[(sub.year == y) & (sub.policy_a == a)
                                  & (sub.policy_b == b_)]
                        if row.empty:
                            cells.append("---")
                            continue
                        any_row = True
                        row = row.iloc[0]
                        star = "*" if row["excludes_zero"] else ""
                        cells.append(f"{row['point']:+.3f} "
                                     f"[{row['lo_2p5']:+.3f}, {row['hi_97p5']:+.3f}]{star}")
                    if any_row:
                        rep.append(f"| {a} | {b_} | " + " | ".join(cells) + " |")
                rep.append("")

    if len(od_df):
        rep += ["## Online calibration (C16): how fast does it learn?", "",
                "On the pairs still UNBOUGHT at each point, i.e. the pairs the",
                "estimate is used to choose between.  AUC = how well the",
                "estimate separates pairs the judge got wrong from pairs it got",
                "right.  `cell` = share of those pairs scored from a full",
                "(grade, bin) cell; the rest back off to grade-only or raw.", "",
                "| year | bought | cell | grade | raw | AUC online | AUC offline "
                "| AUC raw | spearman(online, offline) |",
                "|---|---|---|---|---|---|---|---|---|"]
        for _, d in od_df.iterrows():
            rep.append(f"| {int(d['year'])} | {100 * d['frac_bought']:.0f}% | "
                       f"{100 * d['share_cell']:.0f}% | "
                       f"{100 * d['share_grade']:.0f}% | "
                       f"{100 * d['share_raw']:.0f}% | "
                       f"{d['auc_online']:.3f} | {d['auc_offline']:.3f} | "
                       f"{d['auc_raw']:.3f} | "
                       f"{d['spearman_online_vs_offline']:.3f} |")
        rep.append("")

    rep += ["", "## First touch against sustained, with intervals", "",
            "| year | policy | first touch | 95% CI | sustained | never reached |",
            "|---|---|---|---|---|---|"]
    for _, r_ in thr_df.iterrows():
        ft = "---" if r_["first_touch_point"] is None else f"{r_['first_touch_point']:.0f}"
        su = "---" if r_["sustained_point"] is None else f"{r_['sustained_point']:.0f}"
        rep.append(f"| {int(r_['year'])} | {r_['policy']} | {ft} | "
                   f"[{r_['first_touch_lo_2p5']:.0f}, {r_['first_touch_hi_97p5']:.0f}] | "
                   f"{su} | {r_['first_touch_frac_never']:.2f} |")

    rep += ["", "## All-LLM endpoints", "",
            "| year | metric | point | 95% CI |", "|---|---|---|---|"]
    for e in endpoints:
        rep.append(f"| {e['year']} | {e['metric']} | {e['point']:.3f} | "
                   f"[{e['lo_2p5']:.3f}, {e['hi_97p5']:.3f}] |")

    rep += ["", "## Share of the oracle recovered, tau@20", "",
            "Sections 4.5 and 5.2 say judge confidence recovers roughly 40 to 50",
            "percent of the oracle's area.  The measured ratios:", "",
            "`oracle` sorts by the true |eps| alone.  `product_oracle` is the",
            "headline with the true eps^2 in place of the estimate (C18), so",
            "its ratio is the share of the achievable gain that the error",
            "ESTIMATOR recovers, holding leverage fixed.", "",
            "| year | policy | area | ratio to oracle | 95% CI "
            "| ratio to product_oracle | 95% CI |",
            "|---|---|---|---|---|---|---|"]
    for a in areas:
        if a["metric"] != "tau_at_20" or a["policy"] in (
                "random", "damage_oracle", "reliability_oracle"):
            continue
        if a["policy"] not in TABLE_POLICIES and a["policy"] not in EXTRA_POLICIES:
            continue
        rep.append(f"| {a['year']} | {a['policy']} | {a['area_vs_random']:.4f} | "
                   f"{a['ratio_to_oracle']:.3f} | "
                   f"[{a['ratio_lo']:.3f}, {a['ratio_hi']:.3f}] | "
                   f"{a['ratio_to_product_oracle']:.3f} | "
                   f"[{a['pratio_lo']:.3f}, {a['pratio_hi']:.3f}] |")

    if len(tgt_df):
        rep += ["", "## Targeting grid (C15)", "",
                "tau at FIXED budgets, not budget-to-threshold.  The claim is",
                "that the policy targeting a given set of systems is the one",
                "that does best at ranking those systems.", "",
                "### Is the gain from targeting or from adaptivity?", "",
                "Static top-20 against all-systems and against adaptive",
                "top-20, on tau@20.  If static top-20 sits near adaptive",
                "top-20, targeting explains the gain.  If it sits near",
                "all-systems, adaptivity does.", "",
                "| year | budget | family | all systems | top-20 static | "
                "top-20 adaptive |", "|---|---|---|---|---|---|"]
        # (stat, error) -> name prefix.  prod_on is the headline family (C16).
        FAMILIES = {("var", "none"): "lev", ("range", "none"): "mtc",
                    ("var", "e_online"): "prod_on"}
        for (y, stat, err, b), grp in tgt_df.groupby(
                ["year", "stat", "error", "budget_pct"]):
            pre = FAMILIES.get((stat, err))
            if pre is None:
                continue
            g = grp.set_index("policy")
            def _v(pol):
                return (f"{g.loc[pol, 'tau_at_20']:.3f}"
                        if pol in g.index else "---")
            rep.append(f"| {y} | {b:.0f}% | {pre} | {_v(pre + '_kall')} | "
                       f"{_v(pre + '_k20_static')} | {_v(pre + '_k20')} |")

        rep += ["", "### Diagonal test", "",
                "Share of (year, budget, family) cells where the winning",
                "policy is the one targeting the set being evaluated.", ""]
        dd = pd.DataFrame(diag_rows)
        if len(dd):
            rep += ["| evaluated on | cells | matches prediction |",
                    "|---|---|---|"]
            for metric, grp in dd.groupby("evaluated_on"):
                rep.append(f"| {metric} | {len(grp)} | "
                           f"{100 * grp['matches_prediction'].mean():.0f}% |")

    rep += ["", "## MTC diagnostic (C14)", "",
            "Spearman between the MTC range and the leverage variance over",
            "the same top-20 set, restricted to pairs either statistic can",
            "move.  A low value means the two criteria order the pool",
            "differently and the budget comparison below is measuring a real",
            "distinction, not noise.", "",
            "| year | spearman | kendall | movable share | min > 0 |",
            "|---|---|---|---|---|"]
    for y, r in res_by_year.items():
        d = r["mtc_diag"]
        rep.append(f"| {y} | {d['spearman_nonzero']:.3f} | "
                   f"{d['kendall_nonzero']:.3f} | "
                   f"{100 * d['share_movable']:.1f}% | "
                   f"{100 * d['share_min_positive']:.2f}% |")

    rep += ["", "## Paired differences, computed within each sample", "",
            "| year | A | B | quantity | mean | 95% CI | excludes zero |",
            "|---|---|---|---|---|---|---|"]
    for p in pairs_out:
        rep.append(f"| {p['year']} | {p['policy_a']} | {p['policy_b']} | "
                   f"{p['quantity']} | {p['mean_diff']:.4f} | "
                   f"[{p['lo_2p5']:.4f}, {p['hi_97p5']:.4f}] | "
                   f"{p['excludes_zero']} |")

    if len(struct):
        rep += ["", "## Structural correlations", "",
                "| year | quantity | unit | point | 95% CI |",
                "|---|---|---|---|---|"]
        for _, s in struct.iterrows():
            rep.append(f"| {int(s['year'])} | {s['quantity']} | "
                       f"{s['resample_unit']} | {s['point']:.3f} | "
                       f"[{s['lo_2p5']:.3f}, {s['hi_97p5']:.3f}] |")

    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")

    print(f"\nDone.  Outputs in {OUTPUT_DIR}")
    if len(wide):
        print(f"WARNING: {len(wide)} (year, policy) cells have a first-touch "
              f"interval wider than 30 points.  See bootstrap_config.md.")


if __name__ == "__main__":
    main()