"""
Leave-One-Group-Out Reusability Experiment for TREC DL.

For each year, each group is held out from the selection policy (building W
and estimating the top-20 target set) while remaining in the scoring set.
The exclusion effect is measured as a difference-in-differences on system rank.

Outputs to results/reusability/:
  reusability_per_system.csv   — per (year, policy, budget, group, system)
  reusability_effects.csv      — per (year, policy, budget, group, stratum)
  reusability_summary.csv      — aggregated with bootstrap CIs over groups
  reusability_config.md        — configuration log
  REPORT.md                    — summary table and reading
  reusability_scatter.png      — rank_in vs rank_out figure

Usage:
    python triage/run_reusability.py [--years 2019 2020 ...] [--pilot-only]
                                     [--policies prod_on_k20 lev_kall mtf random]
                                     [--out-suffix TAG]

ONLINE CALIBRATION (A1).  The headline policy is now `prod_on_k20`: leverage
over the adaptive top-20 times E[eps^2] from the OnlineCalibratedError of
run_t12_resampling (note C16), fitted after every batch on the pairs this
configuration has purchased and nothing else.  The previous headline,
`prod_k20`, multiplied by the offline leave-one-query-out estimate, which is
fitted on human grades outside the budget; it is still selectable with
--policies for continuity with earlier results.

Each configuration gets a FRESH calibrator.  That is required, not a
convenience: the calibrator's counts are built from the configuration's own
purchases, which differ when a group is held out, so the error term itself
is part of what the exclusion can change.
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_t12_resampling import (
    load_qrels, load_llm_data, load_system_runs,
    build_sys_top10, build_pair_weight_matrix,
    leverage_over, range_over,
    run_adaptive_run_aware, order_from_scores, run_mtf_policy,
    build_per_query_ndcg_table, boot_correction_sweep,
    ndcg_matrix_from_grades, rank_systems, ranking_metrics,
    expected_sq_error_calibrated, OnlineCalibratedError,
    _stable_seed,
    K_TOP, TARGET_BUDGETS, BATCH_FRACTION, SEED,
)

# run_t12_resampling has put the repository root on sys.path.
from v2_id_mapping import V2_YEARS, canonicalize_runs, load_canonical_map

BASE_DIR = Path(__file__).resolve().parent.parent     # repository root


# ── Configuration ──────────────────────────────────────────────────────

YEARS_CFG = {
    2019: {"qrels": str(BASE_DIR / "data_prep/data/trec-dl/2019/qrels.txt"),
           "scores": str(BASE_DIR / "grades" / "llama-3.1-8b_v1.jsonl"),
           "runs_dir": str(BASE_DIR / "data/system_runs/2019")},
    2020: {"qrels": str(BASE_DIR / "data_prep/data/trec-dl/2020/qrels.txt"),
           "scores": str(BASE_DIR / "grades" / "llama-3.1-8b_v1.jsonl"),
           "runs_dir": str(BASE_DIR / "data/system_runs/2020")},
    2021: {"qrels": str(BASE_DIR / "data_prep/data/trec-dl-v2/2021/qrels_dedup.txt"),
           "scores": str(BASE_DIR / "grades" / "llama-3.1-8b_v2.jsonl"),
           "runs_dir": str(BASE_DIR / "data/system_runs/2021")},
    2022: {"qrels": str(BASE_DIR / "data_prep/data/trec-dl-v2/2022/qrels_dedup.txt"),
           "scores": str(BASE_DIR / "grades" / "llama-3.1-8b_v2.jsonl"),
           "runs_dir": str(BASE_DIR / "data/system_runs/2022")},
    2023: {"qrels": str(BASE_DIR / "data_prep/data/trec-dl-v2/2023/qrels_dedup.txt"),
           "scores": str(BASE_DIR / "grades" / "llama-3.1-8b_v2.jsonl"),
           "runs_dir": str(BASE_DIR / "data/system_runs/2023")},
}

# mtf added: move-to-front pooling (Cormack et al. 1998) also reads the runs,
# so if it shows the same exclusion effect the finding is a property of
# run-aware selection in general rather than of this policy, which is a much
# stronger and more defensible claim.
POLICIES = ["prod_on_k20", "lev_kall", "mtf", "random"]

# The policy the figure, the headline strata, the correction-rate table and
# A5 are computed for.
HEADLINE = "prod_on_k20"

# Everything run_policy_ordering knows.  prod_k20 is the offline-calibrated
# predecessor (labels outside the budget), kept only for continuity.
ALL_POLICIES = ["prod_on_k20", "prod_k20", "lev_kall", "mtf", "random"]

# 0.50, 0.80 and 1.00 added.  1.00 is not a data point, it is an ASSERTION:
# at full budget every pair carries its human grade, so the collection is
# identical no matter what order it was bought in, so the ranking must equal
# the gold ranking and the exclusion effect must be EXACTLY zero for every
# policy.  Unlike A3 (see below) this exercises a policy whose ordering really
# does change across configurations, so it is the only end-to-end check that
# can actually fail.
BUDGET_FRACS = sorted(set(list(TARGET_BUDGETS) + [0.50, 0.80, 1.00]))

DEGENERATE_THRESHOLD = 0.5  # flag if group holds >50% of true top-20
OUT_DIR = BASE_DIR / "results" / "reusability"


# ── Load group mapping ─────────────────────────────────────────────────

def load_group_mapping(path=BASE_DIR / "data" / "run_groups.csv"):
    """Return per-year dicts: groups[year] = {group_id: [run_id, ...]}
    and run_to_group[year] = {run_id: group_id}."""
    df = pd.read_csv(path)
    groups = {}
    run_to_group = {}
    for year, grp in df.groupby("year"):
        groups[year] = {}
        run_to_group[year] = {}
        for _, row in grp.iterrows():
            gid = row["group_id"]
            rid = row["run_id"]
            groups[year].setdefault(gid, []).append(rid)
            run_to_group[year][rid] = gid
    return groups, run_to_group


# ── Core helpers ───────────────────────────────────────────────────────

def compute_ranks_dict(ndcg_scores, system_names):
    """Return {system_name: 1-based rank} from score array."""
    ranking = rank_systems(ndcg_scores, system_names)
    return {s: i + 1 for i, s in enumerate(ranking)}


def compute_correction_rate(ordering_qi, sys_top10, n_buy, queries, system_names):
    """For each system, fraction of top-10 passages (across queries) that
    were purchased in the first n_buy entries of the ordering."""
    purchased = set()
    for idx, (qi, pid) in enumerate(ordering_qi):
        if idx >= n_buy:
            break
        purchased.add((qi, pid))

    rates = []
    for si, sn in enumerate(system_names):
        total = 0
        hit = 0
        t10 = sys_top10[si]
        for qi in range(len(queries)):
            for pid in t10.get(qi, []):
                total += 1
                if (qi, pid) in purchased:
                    hit += 1
        rates.append(hit / total if total > 0 else 0.0)
    return rates


def run_policy_ordering(policy_name, universe, pair_index, Wc, grades_start,
                        human_qrels, queries, system_names_sel, sys_top10_sel,
                        e_cal, batch_size, year, qi_map, n_sys_sel,
                        runs=None, online_factory=None):
    """Generate the acquisition ordering for a policy using sys_sel.

    Every policy takes its tie-break jitter from a seed that does NOT depend
    on the held-out group.  That is deliberate: tie-breaking is then held
    fixed across configurations, so any change in the ordering is caused by
    the change in the scores, not by a different shuffle.
    """

    if policy_name == "random":
        rng = np.random.RandomState(_stable_seed(SEED, year, "reusability_random"))
        scores = rng.rand(len(universe))
        s_rng = np.random.RandomState(_stable_seed(SEED, year, "reusability_random_tie"))
        o = order_from_scores(universe, scores, s_rng)
        return [(qi_map[q], p) for q, p in o]

    elif policy_name == "lev_kall":
        # Static: C over all of sys_sel, no error term
        all_idx = list(range(n_sys_sel))
        C_pp = leverage_over(Wc, all_idx)
        p_rng = np.random.RandomState(_stable_seed(SEED, year, "reusability_lev_kall"))
        o = order_from_scores(universe, C_pp, p_rng)
        return [(qi_map[q], p) for q, p in o]

    elif policy_name == "prod_k20":
        # Adaptive: stat=var, error_term=e_cal, M=20
        p_rng = np.random.RandomState(_stable_seed(SEED, year, "reusability_prod_k20"))
        return run_adaptive_run_aware(
            universe, pair_index, Wc, grades_start, human_qrels, queries,
            system_names_sel, sys_top10_sel, error_term=e_cal, M=K_TOP,
            batch_size=batch_size, stat="var", rng=p_rng)

    elif policy_name == "prod_on_k20":
        # Adaptive: stat=var, ONLINE-calibrated error, M=20.  A fresh
        # calibrator per call, so no configuration sees another's labels.
        if online_factory is None:
            raise ValueError("prod_on_k20 requires `online_factory`")
        em = online_factory()
        p_rng = np.random.RandomState(_stable_seed(SEED, year, "reusability_prod_on_k20"))
        o = run_adaptive_run_aware(
            universe, pair_index, Wc, grades_start, human_qrels, queries,
            system_names_sel, sys_top10_sel, error_model=em, M=K_TOP,
            batch_size=batch_size, stat="var", rng=p_rng)
        assert em.n_label_reads == len(universe), \
            "online calibrator read a label it did not purchase"
        return o

    elif policy_name == "mtf":
        # Move-to-front pooling.  Reads the runs through `system_names_sel`
        # only, so held-out runs never enter the priority queue.
        if runs is None:
            raise ValueError("mtf requires `runs`")
        return run_mtf_policy(universe, human_qrels, queries,
                              system_names_sel, sys_top10_sel, runs)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def sweep_ranks_at_budgets(ordering_qi, grades_start, human_qrels, queries,
                           system_names_all, sys_top10_all, gold_ranking,
                           budget_fracs):
    """Run the budget sweep and return {budget: {system: rank}} and
    {budget: {system: ndcg_score}}."""
    n_q = len(queries)
    ndcg_qk, _ = build_per_query_ndcg_table(
        ordering_qi, grades_start, human_qrels, queries,
        system_names_all, sys_top10_all)
    query_counts = np.ones(n_q, dtype=float)

    rank_at_budget = {}
    ndcg_at_budget = {}
    metrics_at_budget = {}

    n_sys = len(system_names_all)
    for b in budget_fracs:
        # Manually replicate boot_correction_sweep logic for single budget
        restr = ordering_qi  # all queries active
        n_restr = len(restr)
        target = int(round(b * n_restr))
        k_per_q = np.zeros(n_q, dtype=int)
        for idx in range(target):
            k_per_q[restr[idx][0]] += 1
        s = np.zeros(n_sys)
        for qi in range(n_q):
            k = min(k_per_q[qi], ndcg_qk[qi].shape[0] - 1)
            s += ndcg_qk[qi][k]
        s /= n_q
        r = rank_systems(s, system_names_all)
        rank_at_budget[b] = {sys: i + 1 for i, sys in enumerate(r)}
        ndcg_at_budget[b] = {sys: float(s[si]) for si, sys in enumerate(system_names_all)}
        metrics_at_budget[b] = ranking_metrics(gold_ranking, r)

    return rank_at_budget, ndcg_at_budget, metrics_at_budget


# ── Pilot assertions ──────────────────────────────────────────────────

def run_pilot(year, universe, universe_set, grades_start, human_qrels, queries,
              system_names_all, sys_top10_all, runs, e_cal, gold_ranking,
              gold_top20_set, groups, run_to_group, qi_map, batch_size, log,
              online_factory=None):
    """Run pilot on one group with four assertions. Returns True if all pass."""
    log(f"\n{'='*60}")
    log(f"PILOT — Year {year}")
    log(f"{'='*60}")

    # Pick a group that has at least one member in the true top-20
    pilot_group = None
    for gid, members in groups[year].items():
        if any(m in gold_top20_set for m in members):
            pilot_group = gid
            break
    if pilot_group is None:
        # fallback: largest group
        pilot_group = max(groups[year], key=lambda g: len(groups[year][g]))
        log(f"  WARNING: no group has a member in the true top-20, using {pilot_group}")

    held_out_runs = set(groups[year][pilot_group])
    log(f"  Pilot group: {pilot_group} ({len(held_out_runs)} runs)")
    log(f"  Held-out runs: {sorted(held_out_runs)}")

    # Build sys_sel
    sys_sel = [s for s in system_names_all if s not in held_out_runs]
    sys_top10_sel = build_sys_top10(runs, queries, sys_sel)
    Wc_sel, pair_index_sel = build_pair_weight_matrix(runs, queries, sys_sel, universe)

    # ── A1: universe unchanged ──
    log("\n  A1 — Universe unchanged...")
    # Universe depends only on qrels and LLM grades, not on runs
    # Verify Wc_sel has same number of rows as the baseline
    assert Wc_sel.shape[0] == len(universe), \
        f"A1 FAIL: Wc_sel rows {Wc_sel.shape[0]} != universe {len(universe)}"
    log(f"    Wc_sel shape: {Wc_sel.shape} (rows = universe, cols = sys_sel)")
    log("    PASS")

    # ── A2: held-out systems absent from selection ──
    log("\n  A2 — Held-out absent from sys_sel...")
    for m in held_out_runs:
        assert m not in sys_sel, f"A2 FAIL: {m} in sys_sel"
    expected_cols = len(system_names_all) - len(held_out_runs)
    assert Wc_sel.shape[1] == expected_cols, \
        f"A2 FAIL: Wc_sel cols {Wc_sel.shape[1]} != expected {expected_cols}"
    log(f"    sys_sel has {len(sys_sel)} systems (all - {len(held_out_runs)} held out)")
    log(f"    W_sel has {Wc_sel.shape[1]} columns (matches)")
    log("    PASS")

    # ── A3: run-independent policy gives zero exclusion effect ──
    #
    # NOTE ON THE PREVIOUS VERSION.  It built the random ordering twice from
    # the same seed and asserted the two were equal, i.e. it asserted that
    # RandomState is deterministic.  It never passed sys_sel, Wc_sel or
    # sys_top10_sel to anything, so it exercised none of the configuration
    # machinery and could not fail.  Both orderings must now be produced by
    # the SAME call path the real experiment uses, with the held-out one
    # actually receiving the reduced selection structures.
    log("\n  A3 — Run-independent policy: zero exclusion effect...")
    Wc_all_pilot, pair_index_all_pilot = build_pair_weight_matrix(
        runs, queries, system_names_all, universe)

    ordering_base = run_policy_ordering(
        "random", universe, pair_index_all_pilot, Wc_all_pilot, grades_start,
        human_qrels, queries, system_names_all, sys_top10_all,
        e_cal, batch_size, year, qi_map, len(system_names_all), runs=runs,
        online_factory=online_factory)
    ordering_held = run_policy_ordering(
        "random", universe, pair_index_sel, Wc_sel, grades_start,
        human_qrels, queries, sys_sel, sys_top10_sel,
        e_cal, batch_size, year, qi_map, len(sys_sel), runs=runs,
        online_factory=online_factory)

    assert ordering_base == ordering_held, "A3 FAIL: random orderings differ"
    log("    Orderings identical through the real call path  PASS")

    ranks_base, _, _ = sweep_ranks_at_budgets(
        ordering_base, grades_start, human_qrels, queries,
        system_names_all, sys_top10_all, gold_ranking, BUDGET_FRACS)
    ranks_held, _, _ = sweep_ranks_at_budgets(
        ordering_held, grades_start, human_qrels, queries,
        system_names_all, sys_top10_all, gold_ranking, BUDGET_FRACS)

    for b in BUDGET_FRACS:
        for s in system_names_all:
            assert ranks_base[b][s] == ranks_held[b][s], \
                f"A3 FAIL: {s} rank differs at budget {b}"
    log("    Ranks identical at all budgets  PASS")
    log("    (weak by construction: `random` ignores the selection set, so "
        "this can only confirm the scoring path does not read it. A5 is the "
        "assertion that can actually fail.)")

    # ── A4: target set moved for at least one group ──
    log("\n  A4 — Target set moved...")
    # Baseline estimated top-20 (from LLM grades over all systems)
    ndcg_l_all = ndcg_matrix_from_grades(
        grades_start, queries, system_names_all, sys_top10_all)
    base_top20 = set(rank_systems(ndcg_l_all.mean(axis=1), system_names_all)[:K_TOP])

    # Held-out estimated top-20 (from LLM grades over sys_sel)
    ndcg_l_sel = ndcg_matrix_from_grades(
        grades_start, queries, sys_sel, sys_top10_sel)
    held_top20 = set(rank_systems(ndcg_l_sel.mean(axis=1), sys_sel)[:K_TOP])

    diff = base_top20.symmetric_difference(held_top20)
    if diff:
        log(f"    Estimated top-20 changed by {len(diff)} systems")
        log(f"    Removed from top-20: {base_top20 - held_top20}")
        log(f"    Entered top-20: {held_top20 - base_top20}")
        log("    PASS")
    else:
        # Check if ANY group causes a change
        any_moved = False
        for gid, members in groups[year].items():
            hr = set(members)
            ss = [s for s in system_names_all if s not in hr]
            st = build_sys_top10(runs, queries, ss)
            nl = ndcg_matrix_from_grades(grades_start, queries, ss, st)
            ht = set(rank_systems(nl.mean(axis=1), ss)[:K_TOP])
            if ht != base_top20 - hr:  # compare with eligible set
                any_moved = True
                break
        if any_moved:
            log(f"    Pilot group didn't move top-20, but group '{gid}' does")
            log("    PASS (at least one group causes movement)")
        else:
            log("    WARNING: no group changes the estimated top-20")
            log("    WEAK PASS (experiment has limited power)")

    # ── A5: full-budget convergence, with a policy that really does move ──
    #
    # At budget 1.0 every pair carries its human grade, so the collection is
    # identical regardless of the order it was bought in.  The ranking must
    # therefore equal the gold ranking exactly, and the exclusion effect must
    # be exactly zero.  The headline's ordering genuinely differs between the two
    # configurations, so unlike A3 this is a check that can fail: if any part
    # of the pipeline mixes the selection set into scoring, or if the budget
    # denominator drifts between configurations, this catches it.
    log(f"\n  A5 — Full-budget convergence ({HEADLINE})...")
    if 1.0 not in BUDGET_FRACS:
        log("    SKIPPED: budget 1.0 not in BUDGET_FRACS")
    else:
        ord_base_p = run_policy_ordering(
            HEADLINE, universe, pair_index_all_pilot, Wc_all_pilot,
            grades_start, human_qrels, queries, system_names_all,
            sys_top10_all, e_cal, batch_size, year, qi_map,
            len(system_names_all), runs=runs, online_factory=online_factory)
        ord_held_p = run_policy_ordering(
            HEADLINE, universe, pair_index_sel, Wc_sel, grades_start,
            human_qrels, queries, sys_sel, sys_top10_sel,
            e_cal, batch_size, year, qi_map, len(sys_sel), runs=runs,
            online_factory=online_factory)

        assert ord_base_p != ord_held_p, \
            (f"A5 FAIL: {HEADLINE} produced the SAME ordering with and without "
             "the held-out group, so the selection set is not reaching the "
             "policy and the whole experiment is measuring nothing")
        log(f"    {HEADLINE} orderings genuinely differ between configurations")

        rb, _, _ = sweep_ranks_at_budgets(
            ord_base_p, grades_start, human_qrels, queries,
            system_names_all, sys_top10_all, gold_ranking, [1.0])
        rh, _, _ = sweep_ranks_at_budgets(
            ord_held_p, grades_start, human_qrels, queries,
            system_names_all, sys_top10_all, gold_ranking, [1.0])

        gold_rank_map = {s: i + 1 for i, s in enumerate(gold_ranking)}
        for s in system_names_all:
            assert rb[1.0][s] == gold_rank_map[s], \
                f"A5 FAIL: baseline rank of {s} at budget 1.0 is not gold"
            assert rh[1.0][s] == gold_rank_map[s], \
                f"A5 FAIL: held-out rank of {s} at budget 1.0 is not gold"
        log("    At budget 1.0 both configurations reproduce the gold "
            "ranking exactly  PASS")

    log("\n  ALL PILOT ASSERTIONS PASSED")
    return True


# ── Main experiment ────────────────────────────────────────────────────

def run_year_reusability(year, cfg, groups, run_to_group, log):
    """Run the full leave-one-group-out experiment for one year."""
    t0 = time.time()
    log(f"\n{'='*60}")
    log(f"Year {year}")
    log(f"{'='*60}")

    # ── Load data ──
    human_qrels = load_qrels(cfg["qrels"])
    year_qs = set(human_qrels)
    llm_grades, softmax_probs = load_llm_data(cfg["scores"], year_qs)
    year_qs &= set(llm_grades)
    queries = sorted(year_qs)
    n_q = len(queries)
    qi_map = {q: i for i, q in enumerate(queries)}

    runs = load_system_runs(cfg["runs_dir"])
    if year in V2_YEARS:
        canonicalize_runs(runs, load_canonical_map())
    system_names_all = sorted(runs)
    n_sys = len(system_names_all)

    universe = [(q, p) for q in queries for p in sorted(human_qrels[q])
                if p in llm_grades.get(q, {})]
    n_universe = len(universe)
    batch_size = max(1, int(round(BATCH_FRACTION * n_universe)))

    grades_start = {q: {p: (llm_grades[q][p] if p in llm_grades.get(q, {})
                            else human_qrels[q][p])
                        for p in human_qrels[q]} for q in queries}

    log(f"  {n_q} queries, {n_sys} systems, {n_universe} pairs, batch {batch_size}")

    # ── Build all-system structures ──
    sys_top10_all = build_sys_top10(runs, queries, system_names_all)
    Wc_all, pair_index_all = build_pair_weight_matrix(
        runs, queries, system_names_all, universe)

    # ── Gold ranking (from full human grades, never changes) ──
    ndcg_h = ndcg_matrix_from_grades(human_qrels, queries, system_names_all, sys_top10_all)
    gold_scores = ndcg_h.mean(axis=1)
    gold_ranking = rank_systems(gold_scores, system_names_all)
    gold_ranks = {s: i + 1 for i, s in enumerate(gold_ranking)}
    gold_top20 = gold_ranking[:K_TOP]
    gold_top20_set = set(gold_top20)
    log(f"  True top-20: {gold_top20[:5]}... (gold ranking)")

    # ── e_cal: computed once, does not depend on runs ──
    e_cal, cal_info = expected_sq_error_calibrated(
        universe, llm_grades, human_qrels, softmax_probs, verbose=False)

    # ── Online calibrator factory (A1): one fresh instance per ordering ──
    def online_factory():
        return OnlineCalibratedError(universe, llm_grades, softmax_probs,
                                     human_qrels)

    # ── Year-level group info ──
    year_groups = groups.get(year, {})
    n_groups = len(year_groups)
    log(f"  {n_groups} groups: {sorted(year_groups.keys())}")

    # ── Results storage ──
    per_system_rows = []
    effect_rows = []

    # ── Run each policy ──
    for policy in POLICIES:
        log(f"\n  Policy: {policy}")
        tp = time.time()

        # ── Baseline configuration (sys_sel = sys_all) ──
        log(f"    Baseline...")
        t1 = time.time()
        ordering_base = run_policy_ordering(
            policy, universe, pair_index_all, Wc_all, grades_start,
            human_qrels, queries, system_names_all, sys_top10_all,
            e_cal, batch_size, year, qi_map, n_sys, runs=runs,
            online_factory=online_factory)
        ranks_in, ndcg_in, metrics_in = sweep_ranks_at_budgets(
            ordering_base, grades_start, human_qrels, queries,
            system_names_all, sys_top10_all, gold_ranking, BUDGET_FRACS)
        log(f"    Baseline done in {time.time()-t1:.1f}s")

        # Correction rates for baseline
        corr_rates_in = {}
        for b in BUDGET_FRACS:
            n_buy = int(round(b * len(ordering_base)))
            rates = compute_correction_rate(
                ordering_base, sys_top10_all, n_buy, queries, system_names_all)
            corr_rates_in[b] = {s: rates[si] for si, s in enumerate(system_names_all)}

        # ── Held-out configurations ──
        for gid in sorted(year_groups.keys()):
            held_out_runs = set(year_groups[gid])
            group_size = len(held_out_runs)

            # Build sys_sel
            sys_sel = [s for s in system_names_all if s not in held_out_runs]
            n_sys_sel = len(sys_sel)

            # Check for degenerate configuration
            n_held_in_top20 = len(held_out_runs & gold_top20_set)
            is_degenerate = n_held_in_top20 > DEGENERATE_THRESHOLD * K_TOP

            t1 = time.time()

            # Build selection structures
            sys_top10_sel = build_sys_top10(runs, queries, sys_sel)
            Wc_sel, pair_index_sel = build_pair_weight_matrix(
                runs, queries, sys_sel, universe)

            # Run policy with sys_sel for selection
            ordering_held = run_policy_ordering(
                policy, universe, pair_index_sel, Wc_sel, grades_start,
                human_qrels, queries, sys_sel, sys_top10_sel,
                e_cal, batch_size, year, qi_map, n_sys_sel, runs=runs,
                online_factory=online_factory)

            # Score all systems using the held-out ordering
            ranks_out, ndcg_out, metrics_out = sweep_ranks_at_budgets(
                ordering_held, grades_start, human_qrels, queries,
                system_names_all, sys_top10_all, gold_ranking, BUDGET_FRACS)

            # Correction rates for held-out configuration
            corr_rates_out = {}
            for b in BUDGET_FRACS:
                n_buy = int(round(b * len(ordering_held)))
                rates = compute_correction_rate(
                    ordering_held, sys_top10_all, n_buy, queries, system_names_all)
                corr_rates_out[b] = {s: rates[si] for si, s in enumerate(system_names_all)}

            elapsed = time.time() - t1

            # ── Compute displacements and effects ──
            for b in BUDGET_FRACS:
                disps_held = []
                disps_contrib = []
                disps_held_A = []  # stratum A: held-out in true top-20
                disps_held_B = []  # stratum B: held-out outside true top-20
                disps_contrib_A = []
                disps_contrib_B = []

                for si, s in enumerate(system_names_all):
                    r_in = ranks_in[b][s]
                    r_out = ranks_out[b][s]
                    disp = r_out - r_in
                    is_held = s in held_out_runs
                    in_top20 = s in gold_top20_set

                    per_system_rows.append({
                        "year": year,
                        "policy": policy,
                        "budget_pct": int(b * 100),
                        "held_out_group": gid,
                        "group_size": group_size,
                        "system": s,
                        "is_held_out": is_held,
                        "in_true_top20": in_top20,
                        "rank_in": r_in,
                        "rank_out": r_out,
                        "disp": disp,
                        "ndcg_in": ndcg_in[b][s],
                        "ndcg_out": ndcg_out[b][s],
                        "top10_correction_rate": corr_rates_out[b][s],
                        "is_degenerate": is_degenerate,
                    })

                    if is_held:
                        disps_held.append(disp)
                        if in_top20:
                            disps_held_A.append(disp)
                        else:
                            disps_held_B.append(disp)
                    else:
                        disps_contrib.append(disp)
                        if in_top20:
                            disps_contrib_A.append(disp)
                        else:
                            disps_contrib_B.append(disp)

                D_held = np.mean(disps_held) if disps_held else float("nan")
                D_contrib = np.mean(disps_contrib) if disps_contrib else float("nan")
                effect_all = D_held - D_contrib if disps_held else float("nan")

                # Stratum A (held-out in true top-20)
                D_held_A = np.mean(disps_held_A) if disps_held_A else float("nan")
                D_contrib_A = np.mean(disps_contrib_A) if disps_contrib_A else float("nan")
                effect_A = D_held_A - D_contrib_A if disps_held_A else float("nan")

                # Stratum B
                D_held_B = np.mean(disps_held_B) if disps_held_B else float("nan")
                D_contrib_B = np.mean(disps_contrib_B) if disps_contrib_B else float("nan")
                effect_B = D_held_B - D_contrib_B if disps_held_B else float("nan")

                for stratum, n_h, dh, dc, eff in [
                    ("all", len(disps_held), D_held, D_contrib, effect_all),
                    ("A_top20", len(disps_held_A), D_held_A, D_contrib_A, effect_A),
                    ("B_rest", len(disps_held_B), D_held_B, D_contrib_B, effect_B),
                ]:
                    effect_rows.append({
                        "year": year,
                        "policy": policy,
                        "budget_pct": int(b * 100),
                        "held_out_group": gid,
                        "group_size": group_size,
                        "stratum": stratum,
                        "n_held": n_h,
                        "D_held": dh,
                        "D_contrib": dc,
                        "effect": eff,
                        "is_degenerate": is_degenerate,
                    })

            log(f"    {gid:25s} ({group_size:2d} runs, {n_held_in_top20} in top-20)"
                f"  {elapsed:5.1f}s"
                f"{'  DEGENERATE' if is_degenerate else ''}")

        log(f"  Policy {policy} total: {time.time()-tp:.1f}s")

    elapsed_year = time.time() - t0
    log(f"\n  Year {year} total: {elapsed_year:.1f}s")

    return per_system_rows, effect_rows, {
        "year": year, "n_q": n_q, "n_sys": n_sys, "n_universe": n_universe,
        "n_groups": n_groups, "batch_size": batch_size,
        "cal_info": cal_info, "elapsed": elapsed_year,
        "gold_top20": gold_top20,
    }


# ── Summary with bootstrap over groups ─────────────────────────────────

def bootstrap_summary(df_effects, n_boot=5000, seed=SEED):
    """Bootstrap CIs resampling over groups (the unit of exclusion)."""
    rng = np.random.RandomState(seed)
    rows = []
    for (policy, budget, stratum), grp in df_effects.groupby(
            ["policy", "budget_pct", "stratum"]):
        # Exclude degenerate configs from summary
        grp_clean = grp[~grp["is_degenerate"]]
        effects = grp_clean["effect"].dropna().values
        n_groups = len(effects)
        if n_groups == 0:
            continue

        mean_eff = float(np.mean(effects))
        boot_means = []
        for _ in range(n_boot):
            idx = rng.choice(n_groups, size=n_groups, replace=True)
            boot_means.append(float(np.mean(effects[idx])))
        boot_means = np.array(boot_means)
        rows.append({
            "policy": policy,
            "budget_pct": budget,
            "stratum": stratum,
            "n_groups": n_groups,
            "mean_effect": mean_eff,
            "ci_2_5": float(np.percentile(boot_means, 2.5)),
            "ci_50": float(np.percentile(boot_means, 50)),
            "ci_97_5": float(np.percentile(boot_means, 97.5)),
        })
    return pd.DataFrame(rows)


# ── Figure ─────────────────────────────────────────────────────────────

def make_scatter_figure(df_per_system, out_path):
    """rank_in vs rank_out, held-out highlighted, one panel per budget."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    # Headline policy only for the figure
    df = df_per_system[df_per_system["policy"] == HEADLINE].copy()
    budgets = sorted(df["budget_pct"].unique())
    n_panels = len(budgets)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.5),
                             squeeze=False)
    axes = axes[0]

    for ax, budget in zip(axes, budgets):
        db = df[df["budget_pct"] == budget]
        contrib = db[~db["is_held_out"]]
        held = db[db["is_held_out"]]

        # Plot contributing systems
        ax.scatter(contrib["rank_in"], contrib["rank_out"],
                   s=6, alpha=0.15, c="gray", label="contributing", zorder=2)
        # Plot held-out systems
        ax.scatter(held["rank_in"], held["rank_out"],
                   s=12, alpha=0.5, c="tab:red", label="held out", zorder=3)
        # Diagonal
        lim = max(db["rank_in"].max(), db["rank_out"].max()) + 2
        ax.plot([0, lim], [0, lim], "k--", lw=0.5, zorder=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("rank (baseline)")
        ax.set_ylabel("rank (held-out)")
        ax.set_title(f"{budget}% budget")
        ax.set_aspect("equal")

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Leave-One-Group-Out: {HEADLINE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {out_path}")


# ── Reports ────────────────────────────────────────────────────────────

def write_config_log(out_dir, year_infos, pilot_ok, wall_clock, groups, log_lines):
    """Write reusability_config.md."""
    path = out_dir / "reusability_config.md"
    with open(path, "w") as f:
        f.write("# Reusability Experiment Configuration\n\n")
        f.write(f"Wall clock: {wall_clock:.1f}s\n\n")
        f.write(f"Pilot assertions: {'PASSED' if pilot_ok else 'FAILED'}\n\n")
        f.write("## Per-year\n\n")
        for info in year_infos:
            y = info["year"]
            f.write(f"### {y}\n")
            f.write(f"- Queries: {info['n_q']}, Systems: {info['n_sys']}, "
                    f"Pairs: {info['n_universe']}\n")
            f.write(f"- Groups: {info['n_groups']}, Batch size: {info['batch_size']}\n")
            f.write(f"- Calibration: {info['cal_info']}\n")
            f.write(f"- True top-20: {info['gold_top20'][:5]}...\n")
            f.write(f"- Time: {info['elapsed']:.1f}s\n\n")
            # Group sizes
            f.write("| Group | Size | In top-20 |\n|---|---|---|\n")
            top20_set = set(info["gold_top20"])
            for gid in sorted(groups[y]):
                members = groups[y][gid]
                n_top = len(set(members) & top20_set)
                f.write(f"| {gid} | {len(members)} | {n_top} |\n")
            f.write("\n")
        f.write("## Log\n\n```\n")
        f.write("\n".join(log_lines))
        f.write("\n```\n")


def write_report(out_dir, df_summary, df_effects, df_per_system=None):
    """Write REPORT.md."""
    path = out_dir / "REPORT.md"
    with open(path, "w") as f:
        f.write("# Leave-One-Group-Out Reusability Report\n\n")

        f.write("## Summary Table\n\n")
        f.write("Mean exclusion effect (rank positions) with 95% bootstrap CI "
                "over groups.\n\n")
        f.write("| Policy | Budget | Stratum | N groups | Effect | 95% CI |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in df_summary.iterrows():
            ci = f"[{row['ci_2_5']:.2f}, {row['ci_97_5']:.2f}]"
            f.write(f"| {row['policy']} | {row['budget_pct']}% "
                    f"| {row['stratum']} | {row['n_groups']} "
                    f"| {row['mean_effect']:.2f} | {ci} |\n")

        f.write("\n## Reading\n\n")

        # Stratum A headline
        headline = df_summary[
            (df_summary["stratum"] == "A_top20") &
            (df_summary["policy"] == HEADLINE)
        ]
        if len(headline) > 0:
            f.write("### Headline: Stratum A (held-out systems in the true top-20)\n\n")
            for _, row in headline.iterrows():
                n = int(row["n_groups"])
                if n < 30:
                    f.write(f"- **{row['budget_pct']}% budget**: effect = "
                            f"{row['mean_effect']:.2f} rank positions, "
                            f"95% CI {row['ci_2_5']:.2f} to {row['ci_97_5']:.2f} "
                            f"(N={n} groups — below 30, interpret with caution)\n")
                else:
                    f.write(f"- **{row['budget_pct']}% budget**: effect = "
                            f"{row['mean_effect']:.2f} rank positions, "
                            f"95% CI {row['ci_2_5']:.2f} to {row['ci_97_5']:.2f} "
                            f"(N={n} groups)\n")
        else:
            f.write(f"No Stratum A observations for {HEADLINE}.\n")

        # Stratum B, so the headline can be read against something
        f.write("\n### Stratum B (held-out systems OUTSIDE the true top-20)\n\n")
        f.write("If the effect appears only in Stratum A it is specific to "
                "the systems the paper's claim is about; if it is uniform it "
                "is a general property of the collection.\n\n")
        f.write("| Budget | N groups | Effect | 95% CI |\n|---|---|---|---|\n")
        sb = df_summary[(df_summary["stratum"] == "B_rest") &
                        (df_summary["policy"] == HEADLINE)]
        for _, row in sb.iterrows():
            f.write(f"| {row['budget_pct']}% | {int(row['n_groups'])} "
                    f"| {row['mean_effect']:.2f} "
                    f"| [{row['ci_2_5']:.2f}, {row['ci_97_5']:.2f}] |\n")

        # Policy contrast.  ALL FOUR policies, and the stratum is NAMED --
        # the previous version compared stratum "all" directly beneath a
        # headline computed on stratum A_top20 without saying so, which made
        # the two tables look contradictory.
        f.write("\n### Policy contrast — which selection signals show the "
                "effect?\n\n")
        f.write("`random` reads neither the runs nor the judge, so its effect "
                "is zero by construction and is a pipeline check, not a "
                "result. `mtf` reads the runs but is not targeted. If `mtf` "
                "shows the effect too, it is a property of run-aware "
                "selection in general rather than of this policy.\n\n")
        for stratum, label in (("A_top20", "Stratum A (true top-20)"),
                               ("all", "All held-out systems")):
            f.write(f"\n**{label}**\n\n")
            pols = [p for p in POLICIES]
            f.write("| Budget | " + " | ".join(pols) + " |\n")
            f.write("|" + "---|" * (len(pols) + 1) + "\n")
            for b in sorted(df_summary["budget_pct"].unique()):
                cells = []
                for p in pols:
                    row = df_summary[(df_summary["policy"] == p) &
                                     (df_summary["budget_pct"] == b) &
                                     (df_summary["stratum"] == stratum)]
                    cells.append(f"{row.iloc[0]['mean_effect']:.2f}"
                                 if len(row) else "---")
                f.write(f"| {b}% | " + " | ".join(cells) + " |\n")

        # A5 as a reported number, not just an assertion in the log.
        f.write("\n### Convergence check (budget 100%)\n\n")
        f.write("At full budget the collection is entirely human-graded and "
                "cannot depend on the acquisition order, so every effect here "
                "must be exactly 0.00. A non-zero value means the pipeline "
                "leaks the selection set into scoring.\n\n")
        conv = df_summary[df_summary["budget_pct"] == 100]
        if len(conv):
            f.write("| Policy | Stratum | Effect |\n|---|---|---|\n")
            for _, row in conv.iterrows():
                flag = "" if abs(row["mean_effect"]) < 1e-9 else "  **NON-ZERO**"
                f.write(f"| {row['policy']} | {row['stratum']} "
                        f"| {row['mean_effect']:.6f}{flag} |\n")
        else:
            f.write("Budget 100% not in BUDGET_FRACS — check not run.\n")

        # Correction rate: the MECHANISM.  Already stored per system in
        # reusability_per_system.csv but never surfaced, which left the sign
        # of the effect explained only by a plausible story.
        f.write("\n### Correction rate — the mechanism\n\n")
        f.write("Fraction of each system's top-10 passages purchased by the "
                "policy, held-out vs contributing, within the same "
                "configuration. A materially lower rate for held-out systems "
                "is the mechanism behind the effect; matching rates mean "
                "there is nothing to explain.\n\n")
        if df_per_system is not None and len(df_per_system):
            dcr = df_per_system[df_per_system["policy"] == HEADLINE]
            f.write("| Budget | held-out | contributing | difference |\n")
            f.write("|---|---|---|---|\n")
            for b in sorted(dcr["budget_pct"].unique()):
                sub = dcr[dcr["budget_pct"] == b]
                h = sub[sub["is_held_out"]]["top10_correction_rate"].mean()
                c = sub[~sub["is_held_out"]]["top10_correction_rate"].mean()
                f.write(f"| {b}% | {h:.4f} | {c:.4f} | {h - c:+.4f} |\n")
            f.write("\nSame, restricted to systems in the true top-20:\n\n")
            f.write("| Budget | held-out | contributing | difference |\n")
            f.write("|---|---|---|---|\n")
            for b in sorted(dcr["budget_pct"].unique()):
                sub = dcr[(dcr["budget_pct"] == b) & (dcr["in_true_top20"])]
                h = sub[sub["is_held_out"]]["top10_correction_rate"].mean()
                c = sub[~sub["is_held_out"]]["top10_correction_rate"].mean()
                f.write(f"| {b}% | {h:.4f} | {c:.4f} | {h - c:+.4f} |\n")

        # Group size correlation
        f.write("\n### Group size vs effect\n\n")
        df_eff_clean = df_effects[
            (~df_effects["is_degenerate"]) &
            (df_effects["stratum"] == "all") &
            (df_effects["policy"] == HEADLINE)
        ].dropna(subset=["effect"])
        if len(df_eff_clean) > 5:
            rho, p = spearmanr(df_eff_clean["group_size"], df_eff_clean["effect"])
            f.write(f"Spearman(group_size, effect) = {rho:.3f} (p={p:.3f}) "
                    f"over {len(df_eff_clean)} non-degenerate configs.\n")

        f.write("\n---\n*Generated by run_reusability.py*\n")


# ── Entry point ────────────────────────────────────────────────────────

def main():
    global POLICIES, OUT_DIR
    parser = argparse.ArgumentParser(description="LOGO reusability experiment")
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS_CFG.keys()))
    parser.add_argument("--pilot-only", action="store_true",
                        help="Run only the pilot assertions, then stop")
    parser.add_argument("--policies", nargs="+", default=None,
                        choices=ALL_POLICIES,
                        help=f"default: {' '.join(POLICIES)}")
    parser.add_argument("--out-suffix", default=None, metavar="TAG",
                        help="write to results/reusability_TAG/ so earlier "
                             "results are not overwritten")
    args = parser.parse_args()

    if args.policies:
        POLICIES = list(dict.fromkeys(args.policies))
    if HEADLINE not in POLICIES:
        raise SystemExit(f"{HEADLINE} must be among --policies: the report, "
                         f"the figure and pilot check A5 are built on it")
    if args.out_suffix:
        OUT_DIR = OUT_DIR.parent / f"{OUT_DIR.name}_{args.out_suffix}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    groups, run_to_group = load_group_mapping()
    log("Loaded group mapping")
    for y in args.years:
        yg = groups.get(y, {})
        log(f"  {y}: {len(yg)} groups, {sum(len(v) for v in yg.values())} runs")

    t_start = time.time()

    # ── Pilot ──
    # Run on first year that has a group with a top-20 member
    pilot_year = args.years[0]
    cfg = YEARS_CFG[pilot_year]
    human_qrels = load_qrels(cfg["qrels"])
    year_qs = set(human_qrels)
    llm_grades, softmax_probs = load_llm_data(cfg["scores"], year_qs)
    year_qs &= set(llm_grades)
    queries = sorted(year_qs)
    qi_map = {q: i for i, q in enumerate(queries)}

    runs = load_system_runs(cfg["runs_dir"])
    if pilot_year in V2_YEARS:
        canonicalize_runs(runs, load_canonical_map())
    system_names_all = sorted(runs)

    universe = [(q, p) for q in queries for p in sorted(human_qrels[q])
                if p in llm_grades.get(q, {})]
    batch_size = max(1, int(round(BATCH_FRACTION * len(universe))))

    grades_start = {q: {p: (llm_grades[q][p] if p in llm_grades.get(q, {})
                            else human_qrels[q][p])
                        for p in human_qrels[q]} for q in queries}

    sys_top10_all = build_sys_top10(runs, queries, system_names_all)
    ndcg_h = ndcg_matrix_from_grades(human_qrels, queries, system_names_all, sys_top10_all)
    gold_ranking = rank_systems(ndcg_h.mean(axis=1), system_names_all)
    gold_top20_set = set(gold_ranking[:K_TOP])

    e_cal, _ = expected_sq_error_calibrated(
        universe, llm_grades, human_qrels, softmax_probs, verbose=False)

    def online_factory():
        return OnlineCalibratedError(universe, llm_grades, softmax_probs,
                                     human_qrels)

    pilot_ok = run_pilot(
        pilot_year, universe, set(universe), grades_start, human_qrels, queries,
        system_names_all, sys_top10_all, runs, e_cal, gold_ranking,
        gold_top20_set, groups, run_to_group, qi_map, batch_size, log,
        online_factory=online_factory)

    if not pilot_ok:
        log("\nPILOT FAILED — aborting.")
        sys.exit(1)

    if args.pilot_only:
        log("\nPilot passed. --pilot-only set, stopping.")
        return

    # ── Full sweep ──
    all_per_system = []
    all_effects = []
    year_infos = []

    for year in args.years:
        ps, ef, info = run_year_reusability(
            year, YEARS_CFG[year], groups, run_to_group, log)
        all_per_system.extend(ps)
        all_effects.extend(ef)
        year_infos.append(info)

    # ── Write outputs ──
    log(f"\nWriting outputs to {OUT_DIR}/")

    df_ps = pd.DataFrame(all_per_system)
    df_ps.to_csv(OUT_DIR / "reusability_per_system.csv", index=False)
    log(f"  reusability_per_system.csv: {len(df_ps)} rows")

    df_eff = pd.DataFrame(all_effects)
    df_eff.to_csv(OUT_DIR / "reusability_effects.csv", index=False)
    log(f"  reusability_effects.csv: {len(df_eff)} rows")

    df_summary = bootstrap_summary(df_eff)
    df_summary.to_csv(OUT_DIR / "reusability_summary.csv", index=False)
    log(f"  reusability_summary.csv: {len(df_summary)} rows")

    make_scatter_figure(df_ps, OUT_DIR / "reusability_scatter.png")

    wall_clock = time.time() - t_start
    write_config_log(OUT_DIR, year_infos, pilot_ok, wall_clock, groups, log_lines)
    write_report(OUT_DIR, df_summary, df_eff, df_ps)
    log(f"\n  Total wall clock: {wall_clock:.1f}s")
    log("Done.")


if __name__ == "__main__":
    main()