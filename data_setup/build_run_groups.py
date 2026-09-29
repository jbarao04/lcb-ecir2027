"""
Build run -> group mapping for the TREC DL reusability experiment.

Outputs:
  data/runs_by_year.csv       — canonical run identifiers per year (pipeline order)
  data/run_groups.csv         — year, run_id, participant, group_id, is_baseline, task, source
  data/nist_run_metadata.json — cached NIST Browser metadata
  results/run_groups_log.md   — detailed markdown report

Usage:
    python build_run_groups.py [--refresh-metadata] [--runs-dir data/system_runs]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from download_trec_dl_runs import fetch_run_metadata, TREC_EDITIONS

# ── Configuration ──────────────────────────────────────────────────────

YEARS = [2019, 2020, 2021, 2022, 2023]
EXPECTED_COUNTS = {2019: 37, 2020: 59, 2021: 63, 2022: 100, 2023: 35}
EXPECTED_TOTAL = 294

# Edition number -> year (inverse of TREC_EDITIONS)
YEAR_TO_EDITION = {y: n for n, y in TREC_EDITIONS.items()}

# Known joint submissions: group_id -> list of constituent teams
JOINT_SUBMISSIONS = {
    "naverloo": ["naver", "waterloo"],
}

# Manual overrides for runs that cannot be resolved automatically.
# Key: (year, run_id), Value: dict with participant, group_id, is_baseline
MANUAL_OVERRIDES = {
    # 2019: NIST "BASELINE" bucket -> Anserini/Waterloo organizer baselines
    (2019, "bm25base_ax_p"):   {"participant": "BASELINE", "group_id": "anserini", "is_baseline": True},
    (2019, "bm25base_p"):      {"participant": "BASELINE", "group_id": "anserini", "is_baseline": True},
    (2019, "bm25base_prf_p"):  {"participant": "BASELINE", "group_id": "anserini", "is_baseline": True},
    (2019, "bm25base_rm3_p"):  {"participant": "BASELINE", "group_id": "anserini", "is_baseline": True},
    (2019, "bm25tuned_ax_p"):  {"participant": "BASELINE", "group_id": "anserini", "is_baseline": True},
    (2019, "bm25tuned_p"):     {"participant": "BASELINE", "group_id": "anserini", "is_baseline": True},
    (2019, "bm25tuned_prf_p"): {"participant": "BASELINE", "group_id": "anserini", "is_baseline": True},
    (2019, "bm25tuned_rm3_p"): {"participant": "BASELINE", "group_id": "anserini", "is_baseline": True},
    # 2021: NIST "BASELINES" bucket -> actual teams
    # bl_bcai_* = BCAI baselines
    (2021, "bl_bcai_p_nn_rt"): {"participant": "BASELINES", "group_id": "bcai", "is_baseline": True},
    (2021, "bl_bcai_p_trad"):  {"participant": "BASELINES", "group_id": "bcai", "is_baseline": True},
    (2021, "bl_bcai_wloo_p"):  {"participant": "BASELINES", "group_id": "bcai", "is_baseline": True},
    # ielab baselines
    (2021, "ielab-robertav1"): {"participant": "BASELINES", "group_id": "ielab", "is_baseline": True},
    (2021, "ielab-robertav2"): {"participant": "BASELINES", "group_id": "ielab", "is_baseline": True},
    # Anserini/Waterloo organizer baselines
    (2021, "p_bm25"):         {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    (2021, "p_bm25rm3"):      {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    (2021, "p_fusion00"):     {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    (2021, "p_fusion10"):     {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    (2021, "p_tct0"):         {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    (2021, "p_tct1"):         {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    (2021, "p_unicoil0"):     {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    (2021, "paug_bm25"):      {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    (2021, "paug_bm25rm3"):   {"participant": "BASELINES", "group_id": "anserini", "is_baseline": True},
    # UoGTr (Glasgow) baselines
    (2021, "uogTrBasePD"):    {"participant": "BASELINES", "group_id": "uogtr", "is_baseline": True},
    (2021, "uogTrBasePDQ"):   {"participant": "BASELINES", "group_id": "uogtr", "is_baseline": True},
    # Cross-year name unification
    # yorku22 (2022) = yorku (2021) — same York University team
    (2022, "yorku22a"):       {"participant": "yorku22", "group_id": "yorku", "is_baseline": False},
    (2022, "yorku22b"):       {"participant": "yorku22", "group_id": "yorku", "is_baseline": False},
    # DOSSIER (2022) = TU Vienna — DOSSIER is a project name
    (2022, "tuvienna-pas-col"):    {"participant": "DOSSIER", "group_id": "tu_vienna", "is_baseline": False},
    (2022, "tuvienna-pas-unicol"): {"participant": "DOSSIER", "group_id": "tu_vienna", "is_baseline": False},
}


# ── Helpers ────────────────────────────────────────────────────────────

def get_local_runs(runs_dir, year):
    """Return sorted list of run_id strings for a year (filename sans .txt).

    Canonical order = sorted(stripped_names), matching
    system_names = sorted(runs.keys()) in run_spectral_linear.py:544.
    Note: sorted(filenames) then strip != strip then sort, because
    '.' and '-' have different ASCII order. The pipeline sorts KEYS
    (stripped names), so we do the same.
    """
    year_dir = os.path.join(runs_dir, str(year))
    fnames = [f for f in os.listdir(year_dir) if f.endswith(".txt")]
    return sorted(f[:-4] for f in fnames)


def canonicalize_group(participant):
    """Normalize participant name to a canonical group_id."""
    g = participant.strip().lower()
    g = re.sub(r'[^a-z0-9]', '_', g)
    g = re.sub(r'_+', '_', g).strip('_')
    return g


def infer_prefix_group(run_id):
    """Infer a candidate group_id from run_id prefix (for unmatched runs)."""
    known_prefixes = [
        'bm25base', 'bm25tuned', 'srchvrs', 'idst_bert', 'runid',
    ]
    rid_lower = run_id.lower()
    for prefix in known_prefixes:
        if rid_lower.startswith(prefix):
            return prefix
    # Fallback: split on _ or -, take first token
    parts = re.split(r'[-_]', run_id)
    if len(parts) >= 2:
        return parts[0].lower()
    return run_id.lower()


def is_baseline_by_participant(participant):
    """Check if a run is a baseline based on its NIST participant field.

    Only runs explicitly submitted under baseline-type participants are marked.
    Run-id patterns alone are NOT sufficient — e.g. p_bm25 in 2022 is a
    competitive h2oloo submission, not a baseline.
    """
    p = participant.strip().lower()
    # Runs submitted under the "anserini" participant are organizer baselines
    if p == "anserini":
        return True
    # Indri/Terrier baselines submitted under bl_rmit
    if p == "bl_rmit":
        return True
    return False


# ── Enhanced metadata fetcher ──────────────────────────────────────────

def fetch_run_metadata_enhanced(trec_n):
    """
    Fetch NIST metadata and also check for baseline/type indicators.
    Extends the base fetch_run_metadata to also parse **Type:** fields.
    """
    import urllib.request

    url = (
        f"https://raw.githubusercontent.com/usnistgov/trec-browser"
        f"/main/browser/src/docs/trec{trec_n}/deep/runs.md"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except Exception:
        content = ""

    # Parse for Type/Baseline fields per run
    type_map = {}
    current_run_id = None
    for line in content.split("\n"):
        m = re.match(r"^####\s+(\S+)", line)
        if m:
            current_run_id = m.group(1)
        if current_run_id:
            # Check for Type field
            mt = re.search(r'\*\*Type:\*\*\s+(.+)', line)
            if mt:
                type_map[current_run_id] = mt.group(1).strip()
            # Check for Baseline field
            mb = re.search(r'\*\*Baseline:\*\*\s+(.+)', line, re.IGNORECASE)
            if mb:
                type_map[current_run_id] = "baseline:" + mb.group(1).strip()

    # Get standard metadata
    runs = fetch_run_metadata(trec_n)

    # Enrich with type info
    for run in runs:
        if run["run_id"] in type_map:
            run["type_info"] = type_map[run["run_id"]]

    return runs


# ── Main logic ─────────────────────────────────────────────────────────

def build_run_groups(runs_dir, refresh_metadata):
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("TREC DL Run -> Group Mapping")
    log(f"Timestamp: {datetime.now().isoformat()}")
    log("=" * 70)

    # ── Step 1: Enumerate local runs and emit runs_by_year.csv ──────

    log("\n## Step 1: Enumerate local runs\n")

    all_runs_by_year = {}
    runs_by_year_rows = []

    for year in YEARS:
        run_ids = get_local_runs(runs_dir, year)
        all_runs_by_year[year] = run_ids

        # Assert: our order matches sorted(keys), which is how the pipeline
        # builds system_names (run_spectral_linear.py:544).
        assert run_ids == sorted(run_ids), \
            f"{year}: run_ids not in sorted order"

        for pos, rid in enumerate(run_ids):
            runs_by_year_rows.append({"year": year, "run_id": rid, "position": pos})

        n = len(run_ids)
        expected = EXPECTED_COUNTS[year]
        status = "OK" if n == expected else f"MISMATCH (expected {expected})"
        log(f"  {year}: {n} runs  [{status}]")

        # Also verify: sorted(filenames-then-strip) vs sorted(strip-then-sort)
        # These can differ (e.g. '.' vs '-' ASCII), but the pipeline uses
        # sorted(runs.keys()) i.e. strip-then-sort.  We document the diff.
        fnames_sorted = sorted(
            f for f in os.listdir(os.path.join(runs_dir, str(year)))
            if f.endswith(".txt")
        )
        listdir_order = [f[:-4] for f in fnames_sorted]
        if listdir_order != run_ids:
            n_diff = sum(1 for a, b in zip(listdir_order, run_ids) if a != b)
            log(f"    NOTE: sorted(filenames)-then-strip differs from"
                f" strip-then-sort at {n_diff} positions."
                f" Pipeline uses strip-then-sort (sorted keys).")

    total = sum(len(v) for v in all_runs_by_year.values())
    assert total == EXPECTED_TOTAL, \
        f"Total runs {total} != expected {EXPECTED_TOTAL}"
    log(f"\n  Total: {total} (expected {EXPECTED_TOTAL})  [OK]")

    # Write runs_by_year.csv
    rby_path = Path("data/runs_by_year.csv")
    rby_path.parent.mkdir(parents=True, exist_ok=True)
    df_rby = pd.DataFrame(runs_by_year_rows)
    df_rby.to_csv(rby_path, index=False)
    log(f"\n  Written: {rby_path}")
    log(f"  Identifier = filename sans .txt, order = Python sorted()")

    # ── Step 2: Fetch NIST metadata ────────────────────────────────

    log("\n## Step 2: Fetch NIST metadata\n")

    cache_path = Path("data/nist_run_metadata.json")
    if cache_path.exists() and not refresh_metadata:
        log(f"  Loading cached metadata from {cache_path}")
        with open(cache_path) as f:
            nist_data = json.load(f)
    else:
        log("  Fetching from NIST TREC Browser GitHub...")
        nist_data = {}
        for year in YEARS:
            trec_n = YEAR_TO_EDITION[year]
            log(f"    TREC-{trec_n} ({year})...")
            runs_meta = fetch_run_metadata_enhanced(trec_n)
            nist_data[str(year)] = runs_meta
            log(f"      {len(runs_meta)} total runs in metadata")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(nist_data, f, indent=2)
        log(f"\n  Cached to {cache_path}")

    # Summarize NIST metadata
    for year in YEARS:
        year_meta = nist_data.get(str(year), [])
        passage_runs = [r for r in year_meta
                        if r.get("task", "").startswith("passage")]
        doc_runs = [r for r in year_meta
                    if r.get("task", "") == "docs"]
        other = len(year_meta) - len(passage_runs) - len(doc_runs)
        log(f"  {year}: {len(passage_runs)} passage, {len(doc_runs)} docs"
            f", {other} other  (total: {len(year_meta)})")

    # ── Step 3: Join local runs to NIST metadata ───────────────────

    log("\n## Step 3: Join local runs to NIST metadata (exact match)\n")

    # Check for baseline indicators in NIST metadata
    # Note: NIST type_info is run type (auto/manual/feedback), NOT baseline indicator
    log("  NIST metadata has no explicit baseline indicator field.")
    log("  is_baseline is set via participant name (anserini, bl_rmit)")
    log("  and manual overrides for BASELINES-bucket runs.")

    all_group_rows = []
    all_unmatched_local = []
    all_unmatched_nist = []

    for year in YEARS:
        local_ids = set(all_runs_by_year[year])
        year_meta = nist_data.get(str(year), [])
        nist_passage = {
            r["run_id"]: r for r in year_meta
            if r.get("task", "").startswith("passage")
        }
        nist_ids = set(nist_passage.keys())

        matched = local_ids & nist_ids
        unmatched_local = sorted(local_ids - nist_ids)
        unmatched_nist = sorted(nist_ids - local_ids)

        log(f"  {year}: {len(local_ids)} local, {len(nist_passage)} NIST passage"
            f" -> {len(matched)} matched, {len(unmatched_local)} unmatched-local"
            f", {len(unmatched_nist)} unmatched-NIST")

        if unmatched_local:
            log(f"    Unmatched LOCAL: {unmatched_local}")
            for rid in unmatched_local:
                all_unmatched_local.append((year, rid))

        if unmatched_nist:
            log(f"    Unmatched NIST:  {unmatched_nist}")
            for rid in unmatched_nist:
                all_unmatched_nist.append((year, rid))

        # Build rows for matched runs
        for rid in sorted(matched):
            meta = nist_passage[rid]
            participant = meta.get("participant", "")

            # Manual overrides take priority (e.g. BASELINES bucket)
            if (year, rid) in MANUAL_OVERRIDES:
                ov = MANUAL_OVERRIDES[(year, rid)]
                all_group_rows.append({
                    "year": year,
                    "run_id": rid,
                    "participant": participant,
                    "group_id": ov["group_id"],
                    "is_baseline": ov.get("is_baseline", False),
                    "task": "passages",
                    "source": "manual",
                })
                continue

            # Check for undifferentiated BASELINES bucket
            if participant.upper() in ("BASELINES", "BASELINE", "ORGANIZER"):
                group_id = "_not_a_team"
                log(f"    WARNING: {rid} has participant='{participant}'"
                    f" (undifferentiated bucket, no manual override)")
            else:
                group_id = canonicalize_group(participant) if participant else ""

            # Determine is_baseline from participant name
            is_baseline = is_baseline_by_participant(participant)

            all_group_rows.append({
                "year": year,
                "run_id": rid,
                "participant": participant,
                "group_id": group_id,
                "is_baseline": is_baseline,
                "task": "passages",
                "source": "nist",
            })

        # Build rows for unmatched local runs
        for rid in unmatched_local:
            # Check manual overrides first
            if (year, rid) in MANUAL_OVERRIDES:
                ov = MANUAL_OVERRIDES[(year, rid)]
                all_group_rows.append({
                    "year": year,
                    "run_id": rid,
                    "participant": ov.get("participant", ""),
                    "group_id": ov["group_id"],
                    "is_baseline": ov.get("is_baseline", False),
                    "task": "passages",
                    "source": "manual",
                })
            else:
                group_id = infer_prefix_group(rid)
                all_group_rows.append({
                    "year": year,
                    "run_id": rid,
                    "participant": "",
                    "group_id": group_id,
                    "is_baseline": False,  # no participant info to judge
                    "task": "passages",
                    "source": "prefix",
                })

    # ── Step 4: Cross-year consistency checks ──────────────────────

    log("\n## Step 4: Cross-year consistency\n")

    # Group by group_id, check for variant participant strings
    gid_to_participants = defaultdict(set)
    gid_to_years = defaultdict(set)
    for row in all_group_rows:
        if row["participant"]:
            gid_to_participants[row["group_id"]].add(row["participant"])
        gid_to_years[row["group_id"]].add(row["year"])

    multi_name_groups = {
        gid: parts for gid, parts in gid_to_participants.items()
        if len(parts) > 1
    }
    if multi_name_groups:
        log("  Groups with variant participant names across years:")
        for gid, parts in sorted(multi_name_groups.items()):
            years = sorted(gid_to_years[gid])
            log(f"    {gid}: {sorted(parts)} (years: {years})")
    else:
        log("  All groups have consistent participant names across years.")

    # Multi-year groups
    multi_year = {
        gid: sorted(yrs) for gid, yrs in gid_to_years.items()
        if len(yrs) > 1
    }
    log(f"\n  Groups appearing in multiple years: {len(multi_year)}")
    for gid, yrs in sorted(multi_year.items()):
        log(f"    {gid}: {yrs}")

    # ── Step 5: Joint submission flags ─────────────────────────────

    log("\n## Step 5: Joint submission flags\n")

    for gid, constituents in JOINT_SUBMISSIONS.items():
        matching_rows = [r for r in all_group_rows if gid in r["group_id"].lower()]
        if matching_rows:
            log(f"  JOINT: {gid} ({' + '.join(constituents)})"
                f" — {len(matching_rows)} runs")
            for r in matching_rows:
                log(f"    {r['year']} {r['run_id']}")
        else:
            # Check if any run_id contains the joint name
            for r in all_group_rows:
                if gid in r["run_id"].lower():
                    log(f"  JOINT candidate in run_id: {r['year']} {r['run_id']}"
                        f" (group: {r['group_id']})")

    # ── Step 6: Write run_groups.csv ───────────────────────────────

    log("\n## Step 6: Write outputs\n")

    df = pd.DataFrame(all_group_rows)
    df = df.sort_values(["year", "run_id"]).reset_index(drop=True)

    out_path = Path("data/run_groups.csv")
    df.to_csv(out_path, index=False)
    log(f"  Written: {out_path} ({len(df)} rows)")

    # ── Step 7: Validate ───────────────────────────────────────────

    log("\n## Step 7: Validation\n")

    # Total rows
    assert len(df) == EXPECTED_TOTAL, \
        f"Row count {len(df)} != {EXPECTED_TOTAL}"
    log(f"  Total rows: {len(df)} == {EXPECTED_TOTAL}  [OK]")

    # Per-year counts
    for year in YEARS:
        n = len(df[df["year"] == year])
        expected = EXPECTED_COUNTS[year]
        assert n == expected, f"{year}: {n} != {expected}"
        log(f"  {year}: {n} runs  [OK]")

    # Every run has a group_id
    empty_gid = df[df["group_id"] == ""]
    if len(empty_gid) > 0:
        log(f"  WARNING: {len(empty_gid)} runs with empty group_id:")
        for _, r in empty_gid.iterrows():
            log(f"    {r['year']} {r['run_id']}")
    else:
        log("  All runs have non-empty group_id  [OK]")

    # No duplicate (year, run_id)
    dupes = df[df.duplicated(subset=["year", "run_id"], keep=False)]
    assert len(dupes) == 0, f"Duplicate (year, run_id) pairs: {len(dupes)}"
    log("  No duplicate (year, run_id) pairs  [OK]")

    # Source breakdown
    source_counts = df["source"].value_counts()
    log(f"\n  Source breakdown:")
    for src, cnt in source_counts.items():
        log(f"    {src}: {cnt}")

    # ── Step 8: Per-year summary tables ────────────────────────────

    log("\n## Step 8: Per-year group summary\n")

    for year in YEARS:
        dy = df[df["year"] == year]
        group_sizes = dy.groupby("group_id").size().sort_values(ascending=False)
        n_groups = len(group_sizes)
        largest_group = group_sizes.index[0]
        largest_size = group_sizes.iloc[0]
        share = largest_size / len(dy) * 100

        log(f"### {year}: {len(dy)} runs, {n_groups} groups")
        log(f"  Largest group: {largest_group} ({largest_size} runs, {share:.1f}%)")
        log(f"  Group sizes (descending):")
        for gid, sz in group_sizes.items():
            baseline_mark = ""
            n_base = dy[(dy["group_id"] == gid) & (dy["is_baseline"])].shape[0]
            if n_base > 0:
                baseline_mark = f"  [{n_base} baseline]"
            log(f"    {gid}: {sz}{baseline_mark}")
        log("")

    # ── Step 9: Top-20 systems per year ────────────────────────────

    log("\n## Step 9: Top-20 systems per year (by human nDCG)\n")
    log("  Top-20 defined as rank_systems(human_mean, system_names)[:20]")
    log("  from run_spectral_linear.py:245-247, 630")
    log("  Replicated here by sorting systems.parquet by human_score desc.\n")

    for year in YEARS:
        parquet_path = Path(f"results/spectral/intermediates/{year}/systems.parquet")
        if not parquet_path.exists():
            log(f"  {year}: systems.parquet not found, skipping top-20")
            continue

        df_sys = pd.read_parquet(parquet_path)
        df_sys = df_sys.sort_values("human_score", ascending=False).reset_index(drop=True)
        top20 = df_sys.head(20)

        dy = df[df["year"] == year].set_index("run_id")

        log(f"### {year} top-20:")
        log(f"  {'Rank':<5} {'System':<35} {'Group':<25} {'Source':<8} {'nDCG':<8}")
        log(f"  {'-'*5} {'-'*35} {'-'*25} {'-'*8} {'-'*8}")

        group_counts = defaultdict(int)
        for i, (_, row) in enumerate(top20.iterrows()):
            sys_name = row["system"]
            ndcg = row["human_score"]
            if sys_name in dy.index:
                grp = dy.loc[sys_name, "group_id"]
                src = dy.loc[sys_name, "source"]
            else:
                grp = "???"
                src = "???"
            group_counts[grp] += 1
            log(f"  {i+1:<5} {sys_name:<35} {grp:<25} {src:<8} {ndcg:.4f}")

        log(f"\n  Top-20 group distribution:")
        for grp, cnt in sorted(group_counts.items(), key=lambda x: -x[1]):
            log(f"    {grp}: {cnt}")
        log("")

    # ── Step 10: Prefix-inferred runs (for manual review) ─────────

    prefix_runs = df[df["source"] == "prefix"]
    if len(prefix_runs) > 0:
        log("\n## Prefix-inferred runs (NEEDS MANUAL REVIEW)\n")
        log(f"  {len(prefix_runs)} runs with source=prefix:")
        for _, r in prefix_runs.iterrows():
            log(f"    {r['year']} {r['run_id']:<40} -> group={r['group_id']}")

    # ── Write markdown log ─────────────────────────────────────────

    log_path = Path("results/run_groups_log.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Run Group Mapping Log\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("```\n")
        f.write("\n".join(log_lines))
        f.write("\n```\n")

    print(f"\n  Log written to {log_path}")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Build TREC DL run -> group mapping"
    )
    parser.add_argument(
        "--runs-dir", default="data/system_runs",
        help="Directory containing year subdirs with .txt run files",
    )
    parser.add_argument(
        "--refresh-metadata", action="store_true",
        help="Force re-fetch of NIST metadata (ignore cache)",
    )
    args = parser.parse_args()

    build_run_groups(args.runs_dir, args.refresh_metadata)


if __name__ == "__main__":
    main()
