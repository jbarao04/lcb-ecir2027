# Deduplicate TREC DL 2021-2023 qrels by keeping only cluster representatives

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run: pip install requests tqdm")
    sys.exit(1)


NEARDUPES_URL = "https://ir.nist.gov/msmarco-v2-passage-neardupes.txt.gz"
NEARDUPES_FILENAME = "msmarco-v2-passage-neardupes.txt.gz"


def download_file(url: str, dest: Path, description: str) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {description}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            pbar.update(len(chunk))


def load_duplicate_pids(neardupes_path: Path) -> set:
    """
    Parse the equivalence classes file and return the set of
    duplicate (non-representative) passage IDs.

    File format: three tab-separated fields per line:
        class_representative_id    passage_id    passage_text_snippet

    When field 1 != field 2, the passage is a duplicate.
    When field 1 == field 2, the passage is the representative.
    """
    duplicates = set()
    total_lines = 0

    print("  Parsing equivalence classes...")
    open_fn = gzip.open if str(neardupes_path).endswith(".gz") else open

    with open_fn(neardupes_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading neardupes", unit=" lines"):
            total_lines += 1
            parts = line.strip().split(None, 2)
            if len(parts) < 2:
                continue
            representative_id, passage_id = parts[0], parts[1]
            if representative_id != passage_id:
                duplicates.add(passage_id)

    print(f"  Total entries: {total_lines:,}")
    print(f"  Duplicate passage IDs: {len(duplicates):,}")
    return duplicates


def load_qrels(path: Path) -> list:
    """Load qrels as raw lines with parsed fields."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                entries.append({
                    "qid": parts[0],
                    "iter": parts[1],
                    "pid": parts[2],
                    "grade": int(parts[3]),
                    "raw": line.strip(),
                })
    return entries


def dedup_and_save(
    qrels_entries: list, duplicates: set, output_path: Path, year: str
) -> dict:
    """Filter out duplicate passages and save deduplicated qrels."""
    original_count = len(qrels_entries)
    kept = [e for e in qrels_entries if e["pid"] not in duplicates]
    removed_count = original_count - len(kept)

    # Write deduplicated qrels
    with open(output_path, "w", encoding="utf-8") as f:
        for e in kept:
            f.write(f"{e['qid']} {e['iter']} {e['pid']} {e['grade']}\n")

    # Compute per-query stats
    qrels_by_query_orig = defaultdict(int)
    qrels_by_query_dedup = defaultdict(int)
    for e in qrels_entries:
        qrels_by_query_orig[e["qid"]] += 1
    for e in kept:
        qrels_by_query_dedup[e["qid"]] += 1

    per_query_orig = list(qrels_by_query_orig.values())
    per_query_dedup = list(qrels_by_query_dedup.values())

    grade_dist = Counter(e["grade"] for e in kept)

    stats = {
        "year": year,
        "original_entries": original_count,
        "deduplicated_entries": len(kept),
        "removed": removed_count,
        "reduction_pct": round(100 * removed_count / original_count, 1)
        if original_count > 0
        else 0,
        "num_queries": len(qrels_by_query_dedup),
        "mean_per_query_original": round(
            sum(per_query_orig) / len(per_query_orig), 1
        ),
        "mean_per_query_dedup": round(
            sum(per_query_dedup) / len(per_query_dedup), 1
        ),
        "max_per_query_original": max(per_query_orig),
        "max_per_query_dedup": max(per_query_dedup),
        "grade_distribution": {str(k): v for k, v in sorted(grade_dist.items())},
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Deduplicate v2 qrels")
    parser.add_argument(
        "--data-dir", type=str, default="./data/trec-dl-v2"
    )
    args = parser.parse_args()
    base = Path(args.data_dir)
    downloads = base / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    # Step 1: Download equivalence classes
    print("\n=== Step 1: Downloading equivalence classes ===\n")
    neardupes_path = downloads / NEARDUPES_FILENAME
    download_file(NEARDUPES_URL, neardupes_path, "near-duplicate equivalence classes (~50 MB)")

    # Step 2: Parse duplicates
    print("\n=== Step 2: Parsing duplicate IDs ===\n")
    duplicates = load_duplicate_pids(neardupes_path)

    # Step 3: Deduplicate each year
    print("\n=== Step 3: Deduplicating qrels ===\n")

    all_stats = {}
    for year in ["2021", "2022", "2023"]:
        qrels_path = base / year / "qrels.txt"
        if not qrels_path.exists():
            print(f"  {year}: qrels.txt not found, skipping")
            continue

        dedup_path = base / year / "qrels_dedup.txt"
        entries = load_qrels(qrels_path)

        # Check how many are duplicates
        dupe_count = sum(1 for e in entries if e["pid"] in duplicates)

        if dupe_count == 0:
            print(f"  {year}: no duplicates found, copying as-is")
            with open(dedup_path, "w") as f:
                for e in entries:
                    f.write(f"{e['qid']} {e['iter']} {e['pid']} {e['grade']}\n")
            all_stats[year] = {
                "year": year,
                "original_entries": len(entries),
                "deduplicated_entries": len(entries),
                "removed": 0,
                "reduction_pct": 0,
            }
            continue

        stats = dedup_and_save(entries, duplicates, dedup_path, year)
        all_stats[year] = stats

        print(
            f"  {year}: {stats['original_entries']:,} -> "
            f"{stats['deduplicated_entries']:,} "
            f"(-{stats['reduction_pct']}%)"
        )
        print(
            f"         mean/query: {stats['mean_per_query_original']} -> "
            f"{stats['mean_per_query_dedup']}"
        )
        print(
            f"         max/query:  {stats['max_per_query_original']:,} -> "
            f"{stats['max_per_query_dedup']:,}"
        )
        print(f"         grades: {stats['grade_distribution']}")
        print()

    # Save summary
    summary_path = base / "dedup_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Print combined totals
    print("=== Summary ===\n")
    total_orig = sum(s.get("original_entries", 0) for s in all_stats.values())
    total_dedup = sum(s.get("deduplicated_entries", 0) for s in all_stats.values())
    total_queries = sum(s.get("num_queries", 0) for s in all_stats.values())

    print(f"  Total entries:  {total_orig:,} -> {total_dedup:,}")
    print(f"  Total removed:  {total_orig - total_dedup:,}")
    print(f"  Files saved:    <year>/qrels_dedup.txt for each year")
    print(f"  Summary saved:  {summary_path}")

    print(
        f"\n  Use qrels_dedup.txt instead of qrels.txt for LLM scoring."
    )
    print(
        "  The original withDupes qrels are preserved in qrels.txt."
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
