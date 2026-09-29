# Verify dataset integrity and demonstrate loading after setup

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_qrels(path: Path) -> dict:
    """Load qrels: {qid: {pid: grade}}"""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, pid, grade = parts
                qrels[qid][pid] = int(grade)
    return dict(qrels)


def load_queries(path: Path) -> dict:
    """Load queries: {qid: text}"""
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def load_judged_passages(path: Path) -> dict:
    """Load judged passages: {pid: text}"""
    passages = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            passages[obj["pid"]] = obj["passage"]
    return passages


def verify(data_dir: str) -> bool:
    """Run all verification checks. Returns True if all pass."""
    base = Path(data_dir)
    ok = True

    print("=== Verification ===\n")

    # Check files exist
    expected_files = [
        base / "2019" / "queries.tsv",
        base / "2019" / "qrels.txt",
        base / "2020" / "queries.tsv",
        base / "2020" / "qrels.txt",
        base / "judged_passages" / "judged_passages.jsonl",
        base / "summary.json",
        base / "corpus" / "collection.tsv",
    ]

    for f in expected_files:
        if f.exists():
            print(f"  [ok]   {f.relative_to(base)}")
        else:
            print(f"  [FAIL] {f.relative_to(base)} not found")
            ok = False

    if not ok:
        print("\nSome files are missing. Run setup_trec_dl.py first.")
        return False

    # Load data
    print("\n=== Loading data ===\n")

    qrels_2019 = load_qrels(base / "2019" / "qrels.txt")
    qrels_2020 = load_qrels(base / "2020" / "qrels.txt")
    queries_2019 = load_queries(base / "2019" / "queries.tsv")
    queries_2020 = load_queries(base / "2020" / "queries.tsv")
    passages = load_judged_passages(base / "judged_passages" / "judged_passages.jsonl")

    # Check query counts
    checks = [
        ("2019 judged queries", len(qrels_2019), 43, 43),
        ("2020 judged queries", len(qrels_2020), 50, 60),
        ("2019 query file", len(queries_2019), 40, 50),
        ("2020 query file", len(queries_2020), 50, 60),
    ]

    for name, actual, lo, hi in checks:
        status = "[ok]  " if lo <= actual <= hi else "[WARN]"
        print(f"  {status} {name}: {actual} (expected {lo}-{hi})")
        if not (lo <= actual <= hi):
            ok = False

    # Check all judged passages are available
    all_pids = set()
    for qrels in [qrels_2019, qrels_2020]:
        for judgments in qrels.values():
            all_pids.update(judgments.keys())

    missing = all_pids - set(passages.keys())
    if missing:
        print(f"\n  [WARN] {len(missing)} judged passages missing from JSONL")
        ok = False
    else:
        print(f"\n  [ok]   All {len(all_pids):,} judged passages found")

    # Check that judged queries exist in query files
    for year, qrels, queries in [
        ("2019", qrels_2019, queries_2019),
        ("2020", qrels_2020, queries_2020),
    ]:
        judged_qids = set(qrels.keys())
        query_qids = set(queries.keys())
        missing_queries = judged_qids - query_qids
        if missing_queries:
            print(f"  [WARN] {year}: {len(missing_queries)} judged queries not in query file")
            ok = False
        else:
            print(f"  [ok]   {year}: all judged queries have text")

    # Sample output
    print("\n=== Sample data ===\n")
    sample_qid = list(qrels_2019.keys())[0]
    sample_judgments = qrels_2019[sample_qid]
    sample_pid = list(sample_judgments.keys())[0]

    print(f"  Query ID:    {sample_qid}")
    print(f"  Query text:  {queries_2019.get(sample_qid, 'N/A')[:100]}...")
    print(f"  Judged docs: {len(sample_judgments)}")
    print(f"  Grade dist:  {dict(Counter(sample_judgments.values()))}")
    print(f"  Sample PID:  {sample_pid}")
    print(f"  Passage:     {passages.get(sample_pid, 'N/A')[:120]}...")

    # Grade distribution across all years
    print("\n=== Overall grade distribution ===\n")
    for year, qrels in [("2019", qrels_2019), ("2020", qrels_2020)]:
        all_grades = []
        for judgments in qrels.values():
            all_grades.extend(judgments.values())
        grade_dist = Counter(all_grades)
        total = len(all_grades)
        print(f"  {year} ({total:,} judgments):")
        for g in sorted(grade_dist.keys()):
            pct = 100 * grade_dist[g] / total
            print(f"    Grade {g}: {grade_dist[g]:>6,}  ({pct:5.1f}%)")

    if ok:
        print("\n=== All checks passed. Dataset is ready. ===")
    else:
        print("\n=== Some checks failed. Review warnings above. ===")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify TREC DL dataset setup")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/trec-dl",
        help="Root directory for the dataset",
    )
    args = parser.parse_args()
    success = verify(args.data_dir)
    sys.exit(0 if success else 1)
