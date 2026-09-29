# Download TREC Deep Learning passage ranking system runs (2019-2023) from NIST

import argparse
import gzip
import os
import re
import sys
import time
import urllib.request
import urllib.error


# Map from TREC edition number to year
TREC_EDITIONS = {
    28: 2019,
    29: 2020,
    30: 2021,
    31: 2022,
    32: 2023,
}

# GitHub raw URL for the runs metadata files
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/usnistgov/trec-browser"
    "/main/browser/src/docs"
)


def fetch_run_metadata(trec_n):
    """
    Download and parse the runs.md file for a given TREC edition.
    Returns a list of dicts with keys: run_id, task, input_url, participant.
    """
    url = f"{GITHUB_RAW_BASE}/trec{trec_n}/deep/runs.md"
    print(f"  Fetching metadata from {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"  ERROR: HTTP {e.code} fetching {url}")
        return []
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

    runs = []
    current_run = {}

    for line in content.split("\n"):
        # New run block starts with #### run_name
        m = re.match(r"^####\s+(\S+)", line)
        if m:
            if current_run.get("run_id"):
                runs.append(current_run)
            current_run = {"run_id": m.group(1)}
            continue

        # Task field
        if "**Task:**" in line:
            if "passages" in line:
                current_run["task"] = "passages"
            elif "docs" in line:
                current_run["task"] = "docs"

        # Input URL
        m2 = re.search(r'\*\*`Input`\*\*\]\((https://[^)]+)\)', line)
        if m2:
            current_run["input_url"] = m2.group(1)

        # Participant
        m3 = re.search(r'\*\*Participant:\*\*\s+(\S+)', line)
        if m3:
            current_run["participant"] = m3.group(1)

    # Don't forget the last run
    if current_run.get("run_id"):
        runs.append(current_run)

    # For runs without an Input URL (e.g. 2023), construct it from the pattern
    for run in runs:
        if "input_url" not in run:
            run_id = run["run_id"]
            run["input_url"] = (
                f"https://trec.nist.gov/results/trec{trec_n}"
                f"/deep/input.{run_id}.gz"
            )
            run["url_constructed"] = True

    return runs


def download_run(url, output_path, decompress=True):
    """
    Download a single run file. If decompress=True, gunzip and save as plain text.
    Returns True on success, False on failure.
    """
    try:
        import base64
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        login = f"{os.environ['TREC_USER']}:{os.environ['TREC_PASSWORD']}"
        credentials = base64.b64encode(login.encode()).decode()
        req.add_header("Authorization", f"Basic {credentials}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            # Check for auth-required responses
            if resp.status == 401 or resp.status == 403:
                return False

            data = resp.read()

            if decompress and url.endswith(".gz"):
                try:
                    data = gzip.decompress(data)
                except gzip.BadGzipFile:
                    # File might not actually be gzipped
                    pass

            with open(output_path, "wb") as f:
                f.write(data)

            return True

    except urllib.error.HTTPError as e:
        if e.code == 401 or e.code == 403:
            print(f"    AUTH REQUIRED (HTTP {e.code}): {url}")
            print("    The TREC results archive may require credentials.")
            print("    See: https://trec.nist.gov/results.html")
        else:
            print(f"    HTTP ERROR {e.code}: {url}")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def validate_run_file(filepath, year):
    """
    Basic validation: check the file has TREC run format lines.
    Returns (n_queries, n_lines) or None if invalid.
    """
    queries = set()
    n_lines = 0
    try:
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    queries.add(parts[0])
                    n_lines += 1
    except Exception:
        return None

    if n_lines == 0:
        return None
    return len(queries), n_lines


def main():
    parser = argparse.ArgumentParser(
        description="Download TREC DL passage ranking system runs (2019-2023)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/system_runs",
        help="Output directory (default: data/system_runs)",
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=[2019, 2020, 2021, 2022, 2023],
        default=None,
        help="Download only a specific year (default: all years)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List runs without downloading",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between downloads to be polite (default: 0.5)",
    )
    args = parser.parse_args()

    missing = [v for v in ("TREC_USER", "TREC_PASSWORD") if v not in os.environ]
    if missing and not args.dry_run:
        sys.exit(f"Set {' and '.join(missing)} to the login of the TREC "
                 "results archive (https://trec.nist.gov/results.html).")

    # Determine which editions to process
    if args.year:
        editions = {
            n: y for n, y in TREC_EDITIONS.items() if y == args.year
        }
    else:
        editions = TREC_EDITIONS

    print("=" * 60)
    print("TREC Deep Learning Track: Passage Run Downloader")
    print("=" * 60)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for trec_n, year in sorted(editions.items()):
        print(f"\n--- TREC DL {year} (TREC-{trec_n}) ---")

        # Step 1: fetch and parse metadata
        all_runs = fetch_run_metadata(trec_n)
        passage_runs = [r for r in all_runs if r.get("task") == "passages"]

        print(f"  Found {len(passage_runs)} passage runs "
              f"(out of {len(all_runs)} total)")

        if args.dry_run:
            for r in passage_runs:
                marker = " [constructed]" if r.get("url_constructed") else ""
                print(f"    {r['run_id']:40s} {r.get('input_url', 'NO URL')}{marker}")
            continue

        # Step 2: create output directory
        year_dir = os.path.join(args.output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        # Step 3: download each passage run
        for i, run in enumerate(passage_runs):
            run_id = run["run_id"]
            url = run.get("input_url")
            output_path = os.path.join(year_dir, f"{run_id}.txt")

            if os.path.exists(output_path):
                total_skipped += 1
                continue

            if not url:
                print(f"    [{i+1}/{len(passage_runs)}] {run_id}: NO URL")
                total_failed += 1
                continue

            print(f"    [{i+1}/{len(passage_runs)}] {run_id}...", end=" ")

            ok = download_run(url, output_path)
            if ok:
                result = validate_run_file(output_path, year)
                if result:
                    n_q, n_l = result
                    print(f"OK ({n_q} queries, {n_l:,} lines)")
                    total_downloaded += 1
                else:
                    print("DOWNLOADED but validation failed (bad format?)")
                    total_failed += 1
            else:
                total_failed += 1

            time.sleep(args.delay)

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"Done. Downloaded: {total_downloaded}, "
              f"Skipped (existing): {total_skipped}, "
              f"Failed: {total_failed}")
        print(f"Output: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()