# Canonical-ID mapping for MS MARCO v2 near-duplicate passages (2021-2023)

import gzip
from pathlib import Path

_NEARDUPES_PATH = (
    Path(__file__).resolve().parent
    / "data_prep" / "data" / "trec-dl-v2" / "downloads"
    / "msmarco-v2-passage-neardupes.txt.gz"
)

# Years that use the v2 corpus and need canonicalization
V2_YEARS = {2021, 2022, 2023}

_cached_map = None


def load_canonical_map(path=None):
    """Load the near-duplicate equivalence classes and return a mapping
    {duplicate_id: representative_id} for every non-representative passage.

    Representatives map to themselves and are NOT included in the dict,
    so ``canonical_map.get(pid, pid)`` gives the canonical form for any ID.
    """
    global _cached_map
    if _cached_map is not None:
        return _cached_map

    p = Path(path) if path else _NEARDUPES_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Near-duplicates file not found: {p}\n"
            "Run  python data_setup/dedup_v2_qrels.py --data-dir data_prep/data/trec-dl-v2  to download it."
        )

    canonical_map = {}
    open_fn = gzip.open if str(p).endswith(".gz") else open
    with open_fn(p, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            rep_id, passage_id = parts[0], parts[1]
            if rep_id != passage_id:
                canonical_map[passage_id] = rep_id

    _cached_map = canonical_map
    return canonical_map


def canonicalize_runs(runs, canonical_map):
    """Remap passage IDs in system runs to their canonical representatives.

    For each (system, query) ranking:
      1. Map every passage ID through canonical_map.
      2. If two passages collapse to the same representative, keep the one
         with the better (earlier) rank.
      3. Re-compact the list so it stays dense (no gaps).

    Args:
        runs: {sys_name: {qid: [pid_list_by_rank]}}  — modified in-place
        canonical_map: {dup_id: rep_id}  from load_canonical_map()

    Returns:
        runs (same object, mutated)
    """
    for sys_name, sys_run in runs.items():
        for qid, pid_list in sys_run.items():
            seen = set()
            deduped = []
            for pid in pid_list:
                canonical = canonical_map.get(pid, pid)
                if canonical not in seen:
                    seen.add(canonical)
                    deduped.append(canonical)
            sys_run[qid] = deduped
    return runs
