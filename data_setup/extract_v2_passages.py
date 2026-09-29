# Extract judged passages from gzipped MS MARCO v2 bundle files

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def load_qrels(path: Path) -> dict:
    qrels = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, pid, grade = parts
                qrels[qid][pid] = int(grade)
    return dict(qrels)


def parse_v2_passage_id(pid: str):
    """
    Parse msmarco_passage_XX_YYYYYYYY -> (bundle_num, byte_position).
    The byte position refers to the UNCOMPRESSED file.
    """
    parts = pid.split("_")
    if len(parts) != 4 or parts[0] != "msmarco" or parts[1] != "passage":
        return None, None
    try:
        return int(parts[2]), int(parts[3])
    except ValueError:
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Extract judged passages from gzipped v2 bundles"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/trec-dl-v2",
    )
    args = parser.parse_args()
    base = Path(args.data_dir)

    corpus_dir = base / "corpus" / "msmarco_v2_passage"
    output_path = base / "judged_passages" / "judged_passages.jsonl"

    # Check if already done
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r") as f:
            count = sum(1 for _ in f)
        if count > 0:
            print(f"  {output_path} already has {count:,} passages.")
            print("  Delete it to re-extract. Exiting.")
            return

    # Collect all judged passage IDs from qrels
    print("=== Collecting judged passage IDs ===\n")
    all_pids = set()
    for year in ["2021", "2022", "2023"]:
        qrels_path = base / year / "qrels.txt"
        if qrels_path.exists():
            qrels = load_qrels(qrels_path)
            year_pids = set()
            for judgments in qrels.values():
                year_pids.update(judgments.keys())
            all_pids.update(year_pids)
            print(f"  {year}: {len(year_pids):,} unique passage IDs")

    print(f"\n  Total unique passage IDs: {len(all_pids):,}")

    # Group by bundle
    bundles = defaultdict(list)
    unparseable = []
    for pid in all_pids:
        bundle_num, byte_pos = parse_v2_passage_id(pid)
        if bundle_num is not None:
            bundles[bundle_num].append((pid, byte_pos))
        else:
            unparseable.append(pid)

    if unparseable:
        print(f"\n  WARNING: {len(unparseable)} IDs could not be parsed")

    print(f"  Passages spread across {len(bundles)} bundle files\n")

    # Check which gz files exist
    existing_gz = set()
    for f in corpus_dir.iterdir():
        if f.name.endswith(".gz") and f.name.startswith("msmarco_passage_"):
            num = f.name.replace("msmarco_passage_", "").replace(".gz", "")
            try:
                existing_gz.add(int(num))
            except ValueError:
                pass

    missing_bundles = set(bundles.keys()) - existing_gz
    if missing_bundles:
        print(f"  WARNING: {len(missing_bundles)} bundle .gz files not found")

    # Extract passages one bundle at a time
    print("=== Extracting passages (one bundle at a time) ===\n")

    found = {}
    errors = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for bundle_num in tqdm(sorted(bundles.keys()), desc="Processing bundles"):
        gz_path = corpus_dir / f"msmarco_passage_{bundle_num:02d}.gz"
        if not gz_path.exists():
            # Try without zero-padding
            gz_path = corpus_dir / f"msmarco_passage_{bundle_num}.gz"
        if not gz_path.exists():
            errors.extend(pid for pid, _ in bundles[bundle_num])
            continue

        # Decompress to temp file
        temp_path = corpus_dir / f"_temp_msmarco_passage_{bundle_num}"
        try:
            # Decompress
            with gzip.open(gz_path, "rb") as gz_in, open(temp_path, "wb") as tmp_out:
                while True:
                    chunk = gz_in.read(65536)
                    if not chunk:
                        break
                    tmp_out.write(chunk)

            # Seek and extract passages
            with open(temp_path, "r", encoding="utf-8") as f:
                for pid, byte_pos in bundles[bundle_num]:
                    try:
                        f.seek(byte_pos)
                        line = f.readline()
                        obj = json.loads(line)
                        found[pid] = obj.get("passage", obj.get("body", ""))
                    except Exception as e:
                        errors.append(pid)

        finally:
            # Always clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    # Write JSONL
    print(f"\n=== Writing output ===\n")
    with open(output_path, "w", encoding="utf-8") as f:
        for pid in sorted(found.keys()):
            json.dump({"pid": pid, "passage": found[pid]}, f, ensure_ascii=False)
            f.write("\n")

    print(f"  Extracted: {len(found):,} passages")
    print(f"  Errors:    {len(errors)}")
    print(f"  Output:    {output_path}")

    if errors:
        error_path = base / "judged_passages" / "extraction_errors.txt"
        with open(error_path, "w") as f:
            for pid in sorted(set(errors)):
                f.write(pid + "\n")
        print(f"  Error log: {error_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
