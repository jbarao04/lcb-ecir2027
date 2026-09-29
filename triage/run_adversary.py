"""
run_adversary.py
================

D1, part 1.  Does a collection graded by LLM + targeted human correction
resist the gaming attack of Clarke and Dietz (EVIA 2025, arXiv 2412.17156)?

THE ATTACK WE REPRODUCE
-----------------------
Their run uwc1, Sec. 4.1: the team "pooled the top 20 documents from 15
preliminary runs", graded the pool with an LLM, judged the top-graded
passages pairwise with the LLM, and ranked with "LLM-based preference
judgments as the primary key, LLM-based relevance assessments as the
secondary key, and the reciprocal rank fusion of preliminary runs as a
tertiary tie-breaker".  Under automatic assessment it ranked 5th; under
manual assessment, 28th.

Our gamed run, per query:
  pool      union of the top-d passages of the source runs (d = 20 for uwc1's
            shape, 100 for the deep variants), restricted to pairs the LLM
            graded (the judged pool; see LIMITS)
  key 1     the LLM grade g_L                       (their "relevance assessments")
  key 2     the LLM expected grade  sum_h h * pi_h  (stands in for their pairwise
            preferences: a finer ordering of passages within the same grade.
            Their preferences were primary but only defined among top-graded
            passages, so grade-then-finer-signal is the same shape)
  key 3     reciprocal rank fusion over the source runs, k = 60
            (Cormack, Clarke and Buettcher, SIGIR 2009)
  key 4     passage id, so the run is deterministic

Six variants per year (source set x pool depth), so the result is not one
anecdote:
  top5_d20, top15_d20   the 5 / 15 systems ranked highest under the ALL-LLM
                        grades (the strongest runs as the gamer can see them),
                        top 20 of each -- uwc1's shape
  rand15_d20            15 systems drawn at random (stable seed), the closest
                        analogue of uwc1's mixed preliminary runs
  all_d20               every submitted run, top 20
  top15_d100, all_d100  the same with each source's top 100: the depth of
                        Clarke and Dietz's circularity simulation, where every
                        100-deep run was re-ranked by the judge.  Deeper pools
                        hold more passages the LLM likes and the assessors do
                        not, so these are the stronger attacks.

POWER.  A gamed run built from strong runs can simply BE the best system, in
which case no collection can over-rank it and the case carries no
information.  The report therefore also summarises the POWERED cases only
(all-LLM inflation <= POWER_MIN_INFLATION) as the share of the all-LLM
inflation that survives at each budget.

THE TWO CONDITIONS
------------------
  included   the gamed run is a submitted run: it is in the selection set, so
             it enters W, the estimated top-20, the depth-k pool and MTF's
             queue.  The collection is built WITH it.
  held_out   the gamed run arrives after the collection is built from the
             original runs.  Selection never sees it; it is only scored.
             (The reusability experiment's worst case.)

  held_out_topup
             held_out, then VERIFY ON SUBMISSION: when the new run is scored,
             every pair in its top 10 that the collection has not yet checked
             is checked by a human.  The collection is updated for everyone
             (a checked pair is checked for every system that retrieved it).
             The extra pairs are recorded as extra budget.  Not run for LARA,
             whose collection re-imputes unbought grades.

Random and LARA do not read the runs, so for them included and held_out are
the same collection -- asserted, not assumed.

WHAT IS MEASURED
----------------
For each (year, source set) case, each policy, condition and budget:
  rank of the gamed run in the collection, rank under the full human grades
  (the truth), and INFLATION = collection rank - true rank.  Negative means
  the collection ranks it better than it deserves: the attack worked.
  Budget 0 is the all-LLM collection (Clarke and Dietz's automatic
  condition).  Budget 100 is the full human collection, where inflation must
  be exactly zero -- asserted.
Also: the share of the gamed run's top-10 pairs that were human-checked
(the mechanism), and whether it sits in the predicted top 20 while outside
the true top 20.

STATISTICS
----------
Query bootstrap, B replicates, queries resampled independently within each
year.  The COLLECTION at each budget is the one built on the full query set;
only the evaluation queries are resampled.  (Unlike the RESTRICT scheme of
run_t12_resampling, the purchased set is not re-derived per replicate.  Both
understate uncertainty for adaptive policies.)  Pooled statistics average the
cases of all years inside each replicate.

LIMITS, to state in the paper
-----------------------------
  * The gamed run can only use passages the LLM graded, which here are the
    NIST-judged pairs.  A real gamer would also grade unjudged passages; in
    this framework those carry no LLM grade and score 0 under every
    collection, so they could not help it anyway.
  * Pairwise preferences are replaced by the expected grade.
  * The human corrections are NIST judgments made without seeing any LLM
    label, so there is no rubber-stamp effect (Dietz et al., principles
    paper).  A deployment must hide the LLM grade from assessors.

Usage
-----
    python triage/run_adversary.py                       # all years, B = 1000
    python triage/run_adversary.py --years 2019 --B 100  # quick look
    python triage/run_adversary.py --judge qwen          # -> results/adversary_qwen/
"""

import argparse
import importlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00)
# (source set, pool depth).  Depth 20: uwc1, "the top 20 documents".
# Depth 100: the circularity simulation re-ranked 100-deep runs.
VARIANTS = (("top5", 20), ("top15", 20), ("rand15", 20), ("all", 20),
            ("top15", 100), ("all", 100))
SOURCE_SETS = tuple(f"{s}_d{d}" for s, d in VARIANTS)
POWER_MIN_INFLATION = -2  # a case is "powered" if all-LLM over-ranks it by >= 2
RRF_K = 60               # Cormack, Clarke and Buettcher (2009)
GAMED_PREFIX = "zzz_GAMED_"   # sorts last, so any score tie goes AGAINST it
TOPUP = "held_out_topup"
CONDITIONS = ("included", "held_out", TOPUP)

# policy -> reads the runs?  (if not, included == held_out by construction)
POLICIES = {
    "random":         False,
    "depth_k":        True,
    "mtf":            True,
    "mtc_range_all":  True,
    "lara_nN":        False,
    "product_online": True,     # the headline: C over the adaptive top 20
    "prod_on_kall":   True,     # the same product, C over all systems
}


# ---------------------------------------------------------------------------
#  THE GAMED RUN
# ---------------------------------------------------------------------------

def build_gamed_run(runs, sources, queries, universe_set, llm_grades, probs,
                    depth, rrf_k=RRF_K):
    """{qid: [pid, ...]} ordered by (g_L, E_pi[h], RRF, pid), all descending
    except pid.  Only passages the LLM graded can enter."""
    hvals = np.arange(4, dtype=np.float64)
    out = {}
    for qid in queries:
        rrf = defaultdict(float)
        pool = set()
        for sn in sources:
            lst = runs.get(sn, {}).get(qid, [])
            for r, pid in enumerate(lst):
                if (qid, pid) not in universe_set:
                    continue
                rrf[pid] += 1.0 / (rrf_k + r + 1)
                if r < depth:
                    pool.add(pid)
        key = []
        for pid in pool:
            g = int(llm_grades[qid][pid])
            eh = float(np.dot(probs[(qid, pid)], hvals))
            key.append((-g, -eh, -rrf[pid], pid))
        key.sort()
        out[qid] = [k[3] for k in key]
    return out


def check_gamed_run(gr, queries, universe_set, llm_grades):
    """The construction must be exactly what the docstring says."""
    for qid in queries:
        lst = gr[qid]
        assert len(lst) == len(set(lst)), f"duplicate passages in gamed run, {qid}"
        assert all((qid, p) in universe_set for p in lst), "gamed run left the judged pool"
        g = [llm_grades[qid][p] for p in lst]
        assert all(g[i] >= g[i + 1] for i in range(len(g) - 1)), \
            f"gamed run not sorted by LLM grade, {qid}"


# ---------------------------------------------------------------------------
#  COLLECTIONS AT FIXED BUDGETS
# ---------------------------------------------------------------------------

def per_query_at_budgets(ordering_qi, ndcg_qk, n_q, budgets):
    """N[b] = (n_sys, n_q) per-query nDCG of the collection after the first
    round(b * n) purchases.  Identical to boot_correction_sweep with unit
    query weights (asserted in the pilot)."""
    n = len(ordering_qi)
    qseq = np.fromiter((qi for qi, _ in ordering_qi), dtype=np.int64, count=n)
    out, purchased_prefix = [], []
    for b in budgets:
        L = int(round(b * n))
        k = np.bincount(qseq[:L], minlength=n_q)
        cols = [ndcg_qk[qi][min(k[qi], ndcg_qk[qi].shape[0] - 1)] for qi in range(n_q)]
        out.append(np.stack(cols, axis=1))
        purchased_prefix.append(L)
    return np.stack(out), purchased_prefix


def lara_at_budgets(sched_ndcg, sched_cum, n_pairs, budgets):
    b_at = sched_cum.sum(axis=1) / n_pairs
    idx = [max(0, min(int(np.searchsorted(b_at, b, side="right")) - 1,
                      len(sched_ndcg) - 1)) for b in budgets]
    return np.stack([sched_ndcg[i] for i in idx]), idx


SCORE_DECIMALS = 10     # nDCG differences below 1e-10 are float noise


def rank_of(scores, names, target, name_less=None):
    """1-based rank of `target` under rank_systems' convention: descending
    score, ties by name ascending.

    Scores are rounded first.  The same nDCG computed by the incremental
    per-query table and by a direct recomputation can differ in the last
    bits; without rounding such a difference could break an exact tie the
    wrong way and make, e.g., the 100%-budget check fail spuriously.
    `name_less[i]` = names[i] < target, precomputed for speed.
    """
    sc = np.round(np.asarray(scores, dtype=np.float64), SCORE_DECIMALS)
    t = names.index(target)
    s = sc[t]
    if name_less is None:
        name_less = np.array([nm < target for nm in names])
    better = int(np.sum(sc > s))
    tied_before = int(np.sum((sc == s) & name_less))
    return better + tied_before + 1


# ---------------------------------------------------------------------------
#  ONE YEAR
# ---------------------------------------------------------------------------

def run_year(T, year, seed, log):
    t0 = time.time()
    cfg = T.YEARS_CFG[year]
    human = T.load_qrels(cfg["qrels"])
    qs = set(human)
    llm, probs = T.load_llm_data(cfg["scores"], qs)
    qs &= set(llm)
    queries = sorted(qs)
    n_q = len(queries)
    qi_map = {q: i for i, q in enumerate(queries)}
    runs = T.load_system_runs(cfg["runs_dir"])
    if year in T.V2_YEARS:
        T.canonicalize_runs(runs, T.load_canonical_map())
    sys_orig = sorted(runs)
    universe = [(q, p) for q in queries for p in sorted(human[q]) if p in llm.get(q, {})]
    universe_set = set(universe)
    n_pairs = len(universe)
    batch = max(1, int(round(T.BATCH_FRACTION * n_pairs)))
    grades_start = {q: {p: (llm[q][p] if p in llm.get(q, {}) else human[q][p])
                        for p in human[q]} for q in queries}
    log(f"\n=== {year}: {n_q} queries, {len(sys_orig)} systems, {n_pairs} pairs, batch {batch}")

    # ---- gamed runs ------------------------------------------------------
    top10_orig = T.build_sys_top10(runs, queries, sys_orig)
    ndcg_l_orig = T.ndcg_matrix_from_grades(grades_start, queries, sys_orig, top10_orig)
    llm_rank = T.rank_systems(ndcg_l_orig.mean(axis=1), sys_orig)
    rs = np.random.RandomState(T._stable_seed(seed, year, "gamed_rand15"))
    src_sets = {"top5": llm_rank[:5], "top15": llm_rank[:15],
                "rand15": sorted(rs.choice(sys_orig, size=min(15, len(sys_orig)), replace=False).tolist()),
                "all": list(sys_orig)}
    sources, gamed = {}, {}
    for (src, depth), v in zip(VARIANTS, SOURCE_SETS):
        sources[v] = src_sets[src]
        gr = build_gamed_run(runs, sources[v], queries, universe_set, llm, probs, depth)
        check_gamed_run(gr, queries, universe_set, llm)
        gamed[v] = gr
    gnames = {v: GAMED_PREFIX + v for v in SOURCE_SETS}
    assert not any(g in runs for g in gnames.values())

    runs_ext = dict(runs)
    for v in SOURCE_SETS:
        runs_ext[gnames[v]] = gamed[v]
    sys_score = sys_orig + [gnames[v] for v in SOURCE_SETS]      # scoring set
    top10_score = T.build_sys_top10(runs_ext, queries, sys_score)
    ndcg_h = T.ndcg_matrix_from_grades(human, queries, sys_score, top10_score)

    # universe must not change when a gamed run is added: it adds only
    # passages that are already judged.
    for v in SOURCE_SETS:
        for qid in queries:
            assert all((qid, p) in universe_set for p in gamed[v][qid][:10])

    # ---- orderings -------------------------------------------------------
    def seed_for(pol):
        # Same tie-break for both conditions and every case: a change in the
        # ordering is then caused by the change in the selection set only.
        return np.random.RandomState(T._stable_seed(seed, year, f"adv_{pol}"))

    def build_order(pol, sel_names, sel_runs):
        top10_sel = T.build_sys_top10(sel_runs, queries, sel_names)
        if pol == "random":
            r = seed_for("random")
            sc = r.rand(n_pairs)
            o = T.order_from_scores(universe, sc, r)
            return [(qi_map[q], p) for q, p in o], None
        if pol == "depth_k":
            dep, ns = T.build_pool_depth(sel_runs, queries, sel_names, universe_set)
            o = T.order_from_scores(universe, T.scores_depth_k(universe, dep, ns), seed_for(pol))
            return [(qi_map[q], p) for q, p in o], None
        if pol == "mtf":
            return T.run_mtf_policy(universe, human, queries, sel_names, top10_sel, sel_runs), None
        Wc, pidx = T.build_pair_weight_matrix(sel_runs, queries, sel_names, universe)
        assert Wc.shape == (n_pairs, len(sel_names))
        if pol == "mtc_range_all":
            o = T.order_from_scores(universe, T.range_over(Wc, range(len(sel_names))), seed_for(pol))
            return [(qi_map[q], p) for q, p in o], None
        if pol in ("product_online", "prod_on_kall"):
            em = T.OnlineCalibratedError(universe, llm, probs, human)
            o = T.run_adaptive_run_aware(
                universe, pidx, Wc, grades_start, human, queries, sel_names, top10_sel,
                error_model=em, M=T.K_TOP, batch_size=batch, stat="var", rng=seed_for(pol),
                fixed_target=(None if pol == "product_online" else range(len(sel_names))))
            assert em.n_label_reads == n_pairs
            return o, None
        raise ValueError(pol)

    # LARA does not read the runs: one ordering and one schedule per year,
    # scored over every system including all gamed runs at once.
    tl = time.time()
    lara_ord, lara_sched, lara_cum = T.run_lara_policy(
        universe, grades_start, human, probs, queries, sys_score, top10_score, batch,
        n_groups=n_q, seed=T._stable_seed(seed, year, "adv_lara_nN"))
    N_lara, lara_idx = lara_at_budgets(lara_sched, lara_cum, n_pairs, BUDGETS)
    # purchases actually made at the checkpoint each budget maps to
    lara_pref = [int(lara_cum[i].sum()) for i in lara_idx]
    assert lara_pref[0] == 0 and lara_pref[-1] == n_pairs
    log(f"  lara_nN {time.time() - tl:.0f}s")

    # held-out orderings: selection over the ORIGINAL systems only, shared by
    # every case of the year.
    held = {}
    for pol in POLICIES:
        if pol == "lara_nN":
            continue
        tp = time.time()
        held[pol], _ = build_order(pol, sys_orig, runs)
        assert len(held[pol]) == n_pairs and len(set(held[pol])) == n_pairs
        log(f"  held-out {pol:<15s} {time.time() - tp:.0f}s")

    def table_for(ordering, verify=False):
        ndcg_qk, _ = T.build_per_query_ndcg_table(ordering, grades_start, human,
                                                  queries, sys_score, top10_score)
        N, pref = per_query_at_budgets(ordering, ndcg_qk, n_q, BUDGETS)
        if verify:
            # Must be the collection run_t12_resampling evaluates.  Recompute
            # every system's score exactly as boot_correction_sweep does (a
            # running sum over queries of the prefix the budget buys) and
            # demand the SCORES agree to float precision.  Comparing the two
            # RANKINGS instead is wrong: TREC runs include near-duplicate
            # systems with equal scores, and a last-bit difference between
            # the two summation orders can swap them without the collection
            # differing at all.  (This script ranks on scores rounded to
            # SCORE_DECIMALS, so its own ranks are immune to that.)
            n = len(ordering)
            for bi, b in enumerate(BUDGETS):
                target = int(round(b * n))
                k_per_q = np.zeros(n_q, dtype=int)
                for qi, _ in ordering[:target]:
                    k_per_q[qi] += 1
                s_sweep = np.zeros(len(sys_score))
                for qi in range(n_q):
                    s_sweep += ndcg_qk[qi][min(k_per_q[qi], ndcg_qk[qi].shape[0] - 1)]
                s_sweep /= n_q
                gap = float(np.max(np.abs(s_sweep - N[bi].mean(axis=1))))
                assert gap < 1e-9, f"fixed-budget collection != sweep at {b}: max gap {gap:.2e}"
        return N, pref

    N_held = {pol: table_for(o, verify=(pol == "product_online")) for pol, o in held.items()}

    # ---- verify on submission (held out + top-up) ------------------------
    # For every held-out collection and budget: check the gamed run's
    # still-unchecked top-10 pairs.  The base collection is recomputed
    # directly from grades (not from the incremental table), and only the
    # queries the top-up touches are rescored.
    cols_of = {v: list(range(len(sys_orig))) + [sys_score.index(gnames[v])] for v in SOURCE_SETS}
    topup = {v: {} for v in SOURCE_SETS}
    tt = time.time()
    for pol, o in held.items():
        pref = N_held[pol][1]
        acc = {v: ([], []) for v in SOURCE_SETS}
        for bi, L in enumerate(pref):
            bought = set(o[:L])
            mixed = {q: dict(vv) for q, vv in grades_start.items()}
            for qi, pid in o[:L]:
                q = queries[qi]
                mixed[q][pid] = human[q][pid]
            base = T.ndcg_matrix_from_grades(mixed, queries, sys_score, top10_score)
            for v in SOURCE_SETS:
                add = [(qi_map[q], pid) for q in queries for pid in gamed[v][q][:10]
                       if (qi_map[q], pid) not in bought]
                dirty = sorted({qi for qi, _ in add})
                m2 = dict(mixed)                        # shallow: untouched queries shared
                for qi in dirty:
                    m2[queries[qi]] = dict(mixed[queries[qi]])
                for qi, pid in add:
                    m2[queries[qi]][pid] = human[queries[qi]][pid]
                Nq = T.ndcg_matrix_from_grades(m2, queries, sys_score, top10_score,
                                               out=base.copy(), dirty=dirty)
                acc[v][0].append(Nq[cols_of[v], :])
                acc[v][1].append(len(add))
        for v in SOURCE_SETS:
            topup[v][pol] = (np.stack(acc[v][0]), acc[v][1])
            assert acc[v][1][-1] == 0, "top-up must add nothing at 100% budget"
    log(f"  verify-on-submission collections {time.time() - tt:.0f}s")

    # ---- cases -------------------------------------------------------------
    cases = []
    for v in SOURCE_SETS:
        g = gnames[v]
        cols = list(range(len(sys_orig))) + [sys_score.index(g)]
        names = sys_orig + [g]
        case = {"year": year, "source_set": v, "gamed": g, "names": names, "cols": cols,
                "sources": sources[v], "N": {}, "prefix": {},
                "name_less": np.array([nm < g for nm in names])}
        gtop = [(qi_map[q], p) for q in queries for p in gamed[v][q][:10]]
        case["gamed_top10"] = gtop
        for pol, reads_runs in POLICIES.items():
            for cond in ("included", "held_out"):
                if pol == "lara_nN":
                    case["N"][(pol, cond)] = N_lara[:, cols, :]
                    case["prefix"][(pol, cond)] = (lara_ord, lara_pref)
                    continue
                if cond == "held_out" or not reads_runs:
                    N, pref = N_held[pol]
                    o = held[pol]
                    if cond == "included" and not reads_runs:
                        # run-independent: rebuild through the real call path
                        # with the gamed run present and demand identity.
                        o2, _ = build_order(pol, sys_orig + [g], {**runs, g: gamed[v]})
                        assert o2 == o, f"{pol}: included differs from held-out"
                    case["N"][(pol, cond)] = N[:, cols, :]
                    case["prefix"][(pol, cond)] = (o, pref)
                    continue
                tp = time.time()
                o, _ = build_order(pol, sys_orig + [g], {**runs, g: gamed[v]})
                assert len(o) == n_pairs and len(set(o)) == n_pairs
                if o == held[pol]:
                    log(f"  WARNING {v} {pol}: included ordering identical to held-out")
                N, pref = table_for(o)
                case["N"][(pol, cond)] = N[:, cols, :]
                case["prefix"][(pol, cond)] = (o, pref)
                log(f"  {v:<6s} included {pol:<15s} {time.time() - tp:.0f}s")
        case["ndcg_h"] = ndcg_h[cols, :]
        # Share of the gamed run's top-10 pairs human-checked at each budget.
        # Computed here so the orderings can be released.
        top = case["gamed_top10"]
        case["checked"] = {}
        for key, (o, pref) in case["prefix"].items():
            pos = {x: i for i, x in enumerate(o)}
            ranks = np.array([pos[x] for x in top]) if top else np.array([])
            case["checked"][key] = [float(np.mean(ranks < L)) if len(ranks) else 0.0 for L in pref]
        case["topup_extra"] = {}
        for pol, (Nt, extra) in topup[v].items():
            case["N"][(pol, TOPUP)] = Nt
            case["checked"][(pol, TOPUP)] = [1.0] * len(BUDGETS)
            case["topup_extra"][pol] = [e / n_pairs for e in extra]
        case["n_pairs"] = n_pairs
        del case["prefix"], case["gamed_top10"]
        cases.append(case)

    log(f"  {year} done in {(time.time() - t0) / 60:.1f} min")
    return cases, {"year": year, "n_q": n_q, "n_sys": len(sys_orig), "n_pairs": n_pairs,
                   "sources": {v: sources[v] for v in SOURCE_SETS}}


# ---------------------------------------------------------------------------
#  EVALUATION
# ---------------------------------------------------------------------------

def evaluate_case(case, cnt, n_q):
    """Ranks of the gamed run under query weights `cnt`."""
    names, g = case["names"], case["gamed"]
    nl = case["name_less"]
    gold = case["ndcg_h"] @ cnt / n_q
    r_gold = rank_of(gold, names, g, nl)
    out = {}
    for key, N in case["N"].items():
        S = N @ cnt / n_q                                  # (n_budgets, n_sys)
        for bi, b in enumerate(BUDGETS):
            r = rank_of(S[bi], names, g, nl)
            out[(key[0], key[1], b)] = (r, r - r_gold)
    return r_gold, out, (r_gold <= 20)


def correction_rate(case, key, bi):
    return case["checked"][key][bi]


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="run_t12_resampling")
    ap.add_argument("--judge", default="llama", choices=["llama", "qwen"])
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-suffix", default=None)
    args = ap.parse_args()

    T = importlib.import_module(args.module)
    T.set_judge(args.judge)
    suffix = args.out_suffix or (args.judge if args.judge != "llama" else None)
    out_dir = Path(T.BASE_DIR) / "results" / ("adversary" + (f"_{suffix}" if suffix else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(m=""):
        print(m, flush=True)
        lines.append(m)

    t_all = time.time()
    all_cases, infos = [], []
    for y in args.years:
        c, info = run_year(T, y, args.seed, log)
        all_cases.extend(c)
        infos.append(info)

    # ---- full-data evaluation and the assertions ----------------------------
    rows, checks = [], defaultdict(int)
    for case in all_cases:
        n_q = case["ndcg_h"].shape[1]
        r_gold, res, in_gold_top = evaluate_case(case, np.ones(n_q), n_q)
        r_llm = None
        for (pol, cond, b), (r, infl) in res.items():
            bi = BUDGETS.index(b)
            if b == 0.0 and cond != TOPUP:
                r_llm = r if r_llm is None else r_llm
                assert r == r_llm, "budget 0 must be the all-LLM collection for every policy"
                checks["budget0_equals_all_llm"] += 1
            if b == 1.0:
                assert infl == 0, f"{case['year']} {case['source_set']} {pol} {cond}: " \
                                  f"inflation {infl} at 100% budget"
                checks["budget100_zero_inflation"] += 1
            rows.append({"year": case["year"], "source_set": case["source_set"], "policy": pol,
                         "condition": cond, "budget_pct": int(round(100 * b)),
                         "rank_true": r_gold, "rank_collection": r, "inflation": infl,
                         "in_true_top20": in_gold_top, "in_pred_top20": r <= 20,
                         "intruder_top20": (r <= 20) and not in_gold_top,
                         "gamed_top10_checked": correction_rate(case, (pol, cond), bi),
                         "extra_budget_pct": (100 * case["topup_extra"][pol][bi]
                                              if cond == TOPUP else 0.0),
                         "n_systems": len(case["names"])})
        for pol, reads in POLICIES.items():
            if not reads:
                for b in BUDGETS:
                    a = res[(pol, "included", b)]; h = res[(pol, "held_out", b)]
                    assert a == h, f"{pol} is run-independent but conditions differ"
                    checks["run_independent_conditions_equal"] += 1
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "adversary_ranks.csv", index=False)

    cases_df = df[(df.budget_pct == 0) & (df.policy == "random") & (df.condition == "held_out")][
        ["year", "source_set", "rank_true", "rank_collection", "inflation", "n_systems"]].rename(
        columns={"rank_collection": "rank_all_llm", "inflation": "inflation_all_llm"})
    cases_df.to_csv(out_dir / "adversary_cases.csv", index=False)

    # ---- bootstrap ----------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    by_year = defaultdict(list)
    for case in all_cases:
        by_year[case["year"]].append(case)
    keys = sorted({(p, c, b) for case in all_cases for (p, c) in case["N"] for b in BUDGETS})
    boot = {k: [] for k in keys}
    tb = time.time()
    for it in range(args.B):
        acc = defaultdict(list)
        for y, cs in by_year.items():
            n_q = cs[0]["ndcg_h"].shape[1]
            cnt = np.bincount(rng.choice(n_q, n_q, replace=True), minlength=n_q).astype(float)
            for case in cs:
                _, res, _ = evaluate_case(case, cnt, n_q)
                for k, (_, infl) in res.items():
                    acc[k].append(infl)
                    if k[2] == 1.0:
                        assert infl == 0, "non-zero inflation at 100% inside a replicate"
        for k in keys:
            boot[k].append(float(np.mean(acc[k])))
    log(f"\nbootstrap B={args.B}: {time.time() - tb:.0f}s")

    summ = []
    for (pol, cond, b), arr in boot.items():
        sub = df[(df.policy == pol) & (df.condition == cond) & (df.budget_pct == int(round(100 * b)))]
        summ.append({"policy": pol, "condition": cond, "budget_pct": int(round(100 * b)),
                     "n_cases": len(sub), "mean_inflation": sub.inflation.mean(),
                     "lo_2p5": float(np.percentile(arr, 2.5)), "hi_97p5": float(np.percentile(arr, 97.5)),
                     "median_inflation": sub.inflation.median(),
                     "cases_inflated": int((sub.inflation < 0).sum()),
                     "cases_intruding_top20": int(sub.intruder_top20.sum()),
                     "mean_gamed_top10_checked": sub.gamed_top10_checked.mean(),
                     "mean_extra_budget_pct": sub.extra_budget_pct.mean()})
    S = pd.DataFrame(summ).sort_values(["condition", "policy", "budget_pct"])
    S.to_csv(out_dir / "adversary_summary.csv", index=False)

    # ---- report -------------------------------------------------------------
    rep = ["# Clarke & Dietz gaming attack against the hybrid collection", "",
           f"Judge module: `{args.module}`.  Years: {args.years}.  B = {args.B}.  "
           f"{len(all_cases)} cases (years x source sets).", "",
           "Inflation = rank in the collection minus true rank (full human grades).  "
           "Negative = ranked better than it deserves.  Budget 0 is the all-LLM "
           "collection.  Mean over cases, 95% query-bootstrap interval.", "",
           "## The attack under the all-LLM collection", "",
           "| year | source set | true rank | all-LLM rank | inflation | systems |",
           "|---|---|---|---|---|---|"]
    for _, r in cases_df.iterrows():
        rep.append(f"| {int(r.year)} | {r.source_set} | {int(r.rank_true)} | {int(r.rank_all_llm)} | "
                   f"{int(r.inflation_all_llm):+d} | {int(r.n_systems)} |")
    def srow(pol, cond, b):
        r = S[(S.policy == pol) & (S.condition == cond) & (S.budget_pct == int(round(100 * b)))]
        return None if r.empty else r.iloc[0]

    TITLES = {"included": "Gamed run INCLUDED in the selection",
              "held_out": "Gamed run HELD OUT of the selection",
              TOPUP: "HELD OUT, then VERIFY ON SUBMISSION (its unchecked top-10 pairs are checked)"}
    for cond in CONDITIONS:
        rep += ["", f"## {TITLES[cond]}", "",
                "| policy | " + " | ".join(f"{int(100 * b)}%" for b in BUDGETS) + " |",
                "|---|" + "---|" * len(BUDGETS)]
        for pol in POLICIES:
            cells = []
            for b in BUDGETS:
                r = srow(pol, cond, b)
                cells.append("—" if r is None else f"{r.mean_inflation:+.1f} [{r.lo_2p5:+.1f}, {r.hi_97p5:+.1f}]")
            rep.append(f"| {pol} | " + " | ".join(cells) + " |")
        if cond == TOPUP:
            rep += ["", "Extra budget spent on the top-up, % of the pool (mean over cases):", "",
                    "| policy | " + " | ".join(f"{int(100 * b)}%" for b in BUDGETS) + " |",
                    "|---|" + "---|" * len(BUDGETS)]
            for pol in POLICIES:
                cells = []
                for b in BUDGETS:
                    r = srow(pol, cond, b)
                    cells.append("—" if r is None else f"+{r.mean_extra_budget_pct:.2f}")
                rep.append(f"| {pol} | " + " | ".join(cells) + " |")
        else:
            rep += ["", f"Share of the gamed run's top-10 pairs checked by a human ({cond}):", "",
                    "| policy | " + " | ".join(f"{int(100 * b)}%" for b in BUDGETS) + " |",
                    "|---|" + "---|" * len(BUDGETS)]
            for pol in POLICIES:
                cells = []
                for b in BUDGETS:
                    r = srow(pol, cond, b)
                    cells.append("—" if r is None else f"{r.mean_gamed_top10_checked:.2f}")
                rep.append(f"| {pol} | " + " | ".join(cells) + " |")
        rep += ["", f"Cases where the gamed run sits in the predicted top 20 but not the true top 20 ({cond}):", "",
                "| policy | " + " | ".join(f"{int(100 * b)}%" for b in BUDGETS) + " |",
                "|---|" + "---|" * len(BUDGETS)]
        for pol in POLICIES:
            cells = []
            for b in BUDGETS:
                r = srow(pol, cond, b)
                cells.append("—" if r is None else str(r.cases_intruding_top20))
            rep.append(f"| {pol} | " + " | ".join(cells) + " |")
    # ---- powered cases: share of the all-LLM inflation that survives -------
    base = df[(df.budget_pct == 0) & (df.policy == "random") & (df.condition == "held_out")]
    powered = set(map(tuple, base[base.inflation <= POWER_MIN_INFLATION][["year", "source_set"]].values))
    pw_rows = []
    for (pol, cond, b), _ in boot.items():
        sub = df[(df.policy == pol) & (df.condition == cond) & (df.budget_pct == int(round(100 * b)))]
        sub = sub[[(y, v) in powered for y, v in zip(sub.year, sub.source_set)]]
        b0 = df[(df.policy == pol) & (df.condition == ("held_out" if cond == TOPUP else cond))
                & (df.budget_pct == 0)]
        b0 = b0[[(y, v) in powered for y, v in zip(b0.year, b0.source_set)]]
        pw_rows.append({"policy": pol, "condition": cond, "budget_pct": int(round(100 * b)),
                        "n_powered": len(sub),
                        "mean_inflation": sub.inflation.mean() if len(sub) else np.nan,
                        "share_surviving": (sub.inflation.sum() / b0.inflation.sum()) if len(sub) else np.nan})
    PW = pd.DataFrame(pw_rows)
    PW.to_csv(out_dir / "adversary_powered.csv", index=False)
    rep += ["", f"## Powered cases only (all-LLM inflation <= {POWER_MIN_INFLATION}): "
            f"{len(powered)} of {len(base)} cases", "",
            "Share of the all-LLM inflation that survives (1.00 = the attack is untouched, "
            "0.00 = fully removed).  Mean inflation in brackets.", ""]
    for cond in CONDITIONS:
        rep += [f"**{cond}**" + (" (share relative to the all-LLM inflation)" if cond == TOPUP else ""),
                "", "| policy | " + " | ".join(f"{int(100 * b)}%" for b in BUDGETS) + " |",
                "|---|" + "---|" * len(BUDGETS)]
        for pol in POLICIES:
            cells = []
            for b in BUDGETS:
                rr = PW[(PW.policy == pol) & (PW.condition == cond) & (PW.budget_pct == int(round(100 * b)))]
                r = None if rr.empty else rr.iloc[0]
                cells.append("—" if r is None or not r.n_powered else f"{r.share_surviving:.2f} ({r.mean_inflation:+.1f})")
            rep.append(f"| {pol} | " + " | ".join(cells) + " |")
        rep.append("")

    rep += ["", "## Checks asserted", ""] + [f"- {k}: {v} passed" for k, v in checks.items()] + [
        "- gamed runs sorted by LLM grade, no duplicates, inside the judged pool: passed",
        "- online calibrator label reads = purchases in every product run: passed",
        "- every ordering covers the universe exactly once: passed",
        "- zero inflation at 100% budget inside every bootstrap replicate: passed",
        "- verify-on-submission adds no pair at 100% budget: passed", "",
        f"Wall clock: {(time.time() - t_all) / 60:.1f} min", "", "## Source sets", ""]
    for info in infos:
        for v in SOURCE_SETS:
            srcs = info["sources"][v]
            rep.append(f"- {info['year']} {v}: {len(srcs)} runs"
                       + (f" ({', '.join(srcs)})" if len(srcs) <= 15 else ""))
    (out_dir / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    (out_dir / "log.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()