# Download and organise TREC Deep Learning 2021-2023 data (MS MARCO v2 corpus)

import argparse
import gzip
import json
import os
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run: pip install requests tqdm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Download sources
# ---------------------------------------------------------------------------
SOURCES = {
    # Queries (from Microsoft Azure storage)
    "queries_2021": {
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv",
        "filename": "2021_queries.tsv",
        "description": "TREC DL 2021 queries",
    },
    "queries_2022": {
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_queries.tsv",
        "filename": "2022_queries.tsv",
        "description": "TREC DL 2022 queries",
    },
    "queries_2023": {
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_queries.tsv",
        "filename": "2023_queries.tsv",
        "description": "TREC DL 2023 queries",
    },
    # Qrels (from TREC/NIST)
    # 2021 uses "final" (no duplicate propagation needed).
    # 2022 and 2023 use "withDupes" (judgments propagated to near-duplicate passages).
    "qrels_2021": {
        "url": "https://trec.nist.gov/data/deep/2021.qrels.pass.final.txt",
        "filename": "2021.qrels.pass.final.txt",
        "description": "TREC DL 2021 passage qrels (final)",
    },
    "qrels_2022": {
        "url": "https://trec.nist.gov/data/deep/2022.qrels.pass.withDupes.txt",
        "filename": "2022.qrels.pass.withDupes.txt",
        "description": "TREC DL 2022 passage qrels (with duplicate propagation)",
    },
    "qrels_2023": {
        "url": "https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt",
        "filename": "2023.qrels.pass.withDupes.txt",
        "description": "TREC DL 2023 passage qrels (with duplicate propagation)",
    },
    # Corpus
    "corpus": {
        "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar",
        "filename": "msmarco_v2_passage.tar",
        "description": "MS MARCO v2 passage corpus (~20 GB)",
    },
}


def download_file(url: str, dest: Path, description: str) -> None:
    """Download a file with progress bar. Skips if already present."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return

    print(f"  Downloading {description}...")
    # Azure blob storage may need this header
    headers = {"X-Ms-Version": "2019-12-12"}
    response = requests.get(url, stream=True, headers=headers, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_tar(archive: Path, dest_dir: Path) -> Path:
    """Extract the passage corpus tar file."""
    corpus_dir = dest_dir / "msmarco_v2_passage"
    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        print(f"  [skip] {corpus_dir.name} already extracted")
        return corpus_dir

    print("  Extracting corpus archive (this will take a while)...")
    with tarfile.open(archive, "r") as tar:
        tar.extractall(path=dest_dir)

    return corpus_dir


def parse_qrels(path: Path) -> dict:
    """Parse TREC-format qrels. Returns: {qid: {pid: grade}}"""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, pid, grade = parts
            qrels[qid][pid] = int(grade)
    return dict(qrels)


def parse_queries(path: Path) -> dict:
    """Parse TSV queries. Returns: {qid: text}"""
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def parse_v2_passage_id(pid: str):
    """
    Parse a v2 passage ID to get file and byte position.

    Example: msmarco_passage_41_45753370
    -> bundle_num=41, byte_position=45753370
    -> file: msmarco_v2_passage/msmarco_passage_41
    """
    parts = pid.split("_")
    if len(parts) != 4 or parts[0] != "msmarco" or parts[1] != "passage":
        return None, None
    try:
        bundle_num = int(parts[2])
        byte_pos = int(parts[3])
        return bundle_num, byte_pos
    except ValueError:
        return None, None


def extract_judged_passages_v2(
    corpus_dir: Path, judged_pids: set, output_path: Path
) -> dict:
    """
    Extract judged passages from the v2 corpus using seek-based lookup.

    Each passage ID encodes its file and byte position, so we can read
    passages directly without scanning the entire corpus.

    Returns: {pid: passage_text}
    """
    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists, loading...")
        passages = {}
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                passages[obj["pid"]] = obj["passage"]
        return passages

    # Group passage IDs by bundle file for efficient I/O
    bundles = defaultdict(list)
    unparseable = []
    for pid in judged_pids:
        bundle_num, byte_pos = parse_v2_passage_id(pid)
        if bundle_num is not None:
            bundles[bundle_num].append((pid, byte_pos))
        else:
            unparseable.append(pid)

    if unparseable:
        print(f"  WARNING: {len(unparseable)} passage IDs could not be parsed")
        for pid in unparseable[:5]:
            print(f"    e.g. {pid}")

    print(
        f"  Extracting {len(judged_pids):,} passages from "
        f"{len(bundles)} bundle files..."
    )

    found = {}
    missing = []

    for bundle_num in tqdm(sorted(bundles.keys()), desc="Reading bundles"):
        bundle_file = corpus_dir / f"msmarco_passage_{bundle_num:02d}"
        if not bundle_file.exists():
            # Try without zero-padding
            bundle_file = corpus_dir / f"msmarco_passage_{bundle_num}"
        if not bundle_file.exists():
            print(f"  WARNING: Bundle file not found: {bundle_file.name}")
            missing.extend(pid for pid, _ in bundles[bundle_num])
            continue

        with open(bundle_file, "r", encoding="utf-8") as f:
            for pid, byte_pos in bundles[bundle_num]:
                try:
                    f.seek(byte_pos)
                    line = f.readline()
                    obj = json.loads(line)
                    passage_text = obj.get("passage", obj.get("body", ""))
                    found[pid] = passage_text
                except Exception as e:
                    missing.append(pid)

    if missing:
        print(f"  WARNING: {len(missing)} passages could not be extracted")

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pid, text in sorted(found.items()):
            json.dump({"pid": pid, "passage": text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"  Extracted {len(found):,} passages, {len(missing)} missing")
    return found


def collect_judged_pids(qrels_by_year: dict) -> set:
    """Collect all unique passage IDs across all years."""
    pids = set()
    for year_qrels in qrels_by_year.values():
        for judgments in year_qrels.values():
            pids.update(judgments.keys())
    return pids


def compute_summary(
    qrels_by_year: dict, queries_by_year: dict, judged_pids: set, found_pids: set
) -> dict:
    """Compute dataset summary statistics."""
    summary = {"years": {}, "total": {}}
    all_queries = 0
    all_judgments = 0

    for year in sorted(qrels_by_year.keys()):
        qrels = qrels_by_year[year]
        queries = queries_by_year.get(year, {})
        grade_counter = Counter()
        per_query_counts = []

        for qid, judgments in qrels.items():
            per_query_counts.append(len(judgments))
            grade_counter.update(judgments.values())

        year_info = {
            "num_queries_with_judgments": len(qrels),
            "num_queries_in_file": len(queries),
            "total_judgments": sum(per_query_counts),
            "unique_passages": len(
                set(p for j in qrels.values() for p in j)
            ),
            "grade_distribution": {
                str(k): v for k, v in sorted(grade_counter.items())
            },
            "judged_per_query_mean": round(
                sum(per_query_counts) / len(per_query_counts), 1
            )
            if per_query_counts
            else 0,
            "judged_per_query_min": min(per_query_counts) if per_query_counts else 0,
            "judged_per_query_max": max(per_query_counts) if per_query_counts else 0,
        }
        summary["years"][year] = year_info
        all_queries += len(qrels)
        all_judgments += year_info["total_judgments"]

    missing = judged_pids - found_pids
    summary["total"] = {
        "queries": all_queries,
        "judgments": all_judgments,
        "unique_judged_passages": len(judged_pids),
        "passages_found_in_corpus": len(found_pids),
        "passages_missing_from_corpus": len(missing),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Setup TREC DL 2021-2023 dataset (MS MARCO v2)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/trec-dl-v2",
        help="Root directory for the dataset (default: ./data/trec-dl-v2)",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Download only queries and qrels, skip the ~20 GB corpus download",
    )
    args = parser.parse_args()
    base = Path(args.data_dir)

    # Create directory structure
    dirs = {
        "downloads": base / "downloads",
        "corpus": base / "corpus",
        "2021": base / "2021",
        "2022": base / "2022",
        "2023": base / "2023",
        "judged": base / "judged_passages",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Step 1: Download queries and qrels (small files)
    # -------------------------------------------------------------------
    print("\n=== Step 1: Downloading queries and qrels ===")
    dl_dir = dirs["downloads"]
    for key, info in SOURCES.items():
        if key == "corpus":
            continue  # handle separately
        download_file(info["url"], dl_dir / info["filename"], info["description"])

    # -------------------------------------------------------------------
    # Step 2: Copy queries and qrels to year directories
    # -------------------------------------------------------------------
    print("\n=== Step 2: Organizing files ===")

    query_files = {
        "2021": "2021_queries.tsv",
        "2022": "2022_queries.tsv",
        "2023": "2023_queries.tsv",
    }
    qrel_files = {
        "2021": "2021.qrels.pass.final.txt",
        "2022": "2022.qrels.pass.withDupes.txt",
        "2023": "2023.qrels.pass.withDupes.txt",
    }

    for year in ["2021", "2022", "2023"]:
        # Queries (plain TSV, no extraction needed)
        src = dl_dir / query_files[year]
        dst = dirs[year] / "queries.tsv"
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Copied {src.name} -> {year}/queries.tsv")
        else:
            print(f"  [skip] {year}/queries.tsv already exists")

        # Qrels
        src = dl_dir / qrel_files[year]
        dst = dirs[year] / "qrels.txt"
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Copied {src.name} -> {year}/qrels.txt")
        else:
            print(f"  [skip] {year}/qrels.txt already exists")

    # -------------------------------------------------------------------
    # Step 3: Parse qrels and queries
    # -------------------------------------------------------------------
    print("\n=== Step 3: Parsing qrels and queries ===")

    qrels_by_year = {}
    queries_by_year = {}
    for year in ["2021", "2022", "2023"]:
        qrels_by_year[year] = parse_qrels(dirs[year] / "qrels.txt")
        queries_by_year[year] = parse_queries(dirs[year] / "queries.tsv")
        n_qrels = sum(len(j) for j in qrels_by_year[year].values())
        print(
            f"  {year}: {len(qrels_by_year[year])} judged queries, "
            f"{len(queries_by_year[year])} in file, "
            f"{n_qrels:,} total judgments"
        )

    judged_pids = collect_judged_pids(qrels_by_year)
    print(f"\n  Total unique judged passage IDs: {len(judged_pids):,}")

    # -------------------------------------------------------------------
    # Step 4: Download and extract corpus (if not skipped)
    # -------------------------------------------------------------------
    if args.skip_corpus:
        print("\n=== Step 4: Corpus download SKIPPED (--skip-corpus) ===")
        print("  Re-run without --skip-corpus to download the corpus")
        print("  and extract judged passages.")

        # Still compute partial summary
        summary = compute_summary(
            qrels_by_year, queries_by_year, judged_pids, set()
        )
        summary["note"] = "Corpus not yet downloaded. Passage texts unavailable."
    else:
        print("\n=== Step 4: Downloading corpus ===")
        corpus_info = SOURCES["corpus"]
        download_file(
            corpus_info["url"],
            dl_dir / corpus_info["filename"],
            corpus_info["description"],
        )

        print("\n=== Step 5: Extracting corpus ===")
        corpus_dir = extract_tar(dl_dir / "msmarco_v2_passage.tar", dirs["corpus"])

        print("\n=== Step 6: Extracting judged passages ===")
        passages = extract_judged_passages_v2(
            corpus_dir, judged_pids, dirs["judged"] / "judged_passages.jsonl"
        )

        summary = compute_summary(
            qrels_by_year, queries_by_year, judged_pids, set(passages.keys())
        )

    # -------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------
    summary_path = base / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("DATASET SUMMARY (MS MARCO v2, TREC DL 2021-2023)")
    print(f"{'='*60}")

    for year, info in summary["years"].items():
        print(f"\n  TREC DL {year}:")
        print(f"    Queries with judgments: {info['num_queries_with_judgments']}")
        print(f"    Total judgments:        {info['total_judgments']:,}")
        print(f"    Unique passages:        {info['unique_passages']:,}")
        print(f"    Judged/query (mean):    {info['judged_per_query_mean']}")
        print(f"    Judged/query (min):     {info['judged_per_query_min']}")
        print(f"    Judged/query (max):     {info['judged_per_query_max']}")
        print(f"    Grade distribution:     {info['grade_distribution']}")

    t = summary["total"]
    print(f"\n  TOTAL:")
    print(f"    Queries:                {t['queries']}")
    print(f"    Judgments:              {t['judgments']:,}")
    print(f"    Unique passages:        {t['unique_judged_passages']:,}")
    if t["passages_found_in_corpus"] > 0:
        print(f"    Found in corpus:        {t['passages_found_in_corpus']:,}")
        print(f"    Missing from corpus:    {t['passages_missing_from_corpus']}")

    if summary.get("note"):
        print(f"\n  NOTE: {summary['note']}")

    print(f"\n  Summary saved to: {summary_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
