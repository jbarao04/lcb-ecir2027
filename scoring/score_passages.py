#!/usr/bin/env python3
# LLM scoring pipeline: constrained decoding on a 0-3 scale with logprob extraction

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a search quality rater evaluating the relevance of web passages. "
    "Given a query and a passage, you must provide a score on an integer scale "
    "of 0 to 3 with the following meanings:\n\n"
    "0: The passage is not relevant to the query.\n"
    "1: The passage is related to the query but does not directly answer it.\n"
    "2: The passage has some answer for the query, but the answer may be a bit "
    "unclear, or hidden amongst extraneous information.\n"
    "3: The passage is dedicated to the query and contains the exact answer."
)

USER_TEMPLATE = "Query: {query}\nPassage: {passage}\n\nScore:"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_queries(path: str) -> Dict[str, str]:
    """Load queries from TSV file (qid<TAB>query)."""
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    logging.info("Loaded %d queries from %s", len(queries), path)
    return queries


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels in TREC format (qid 0 pid rel)."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, pid, rel = parts[0], parts[2], int(parts[3])
                qrels.setdefault(qid, {})[pid] = rel
    total_pairs = sum(len(v) for v in qrels.values())
    logging.info(
        "Loaded qrels: %d queries, %d judged pairs from %s",
        len(qrels), total_pairs, path,
    )
    return qrels


def load_passages(path: str) -> Dict[str, str]:
    """Load passages from JSONL file ({"pid": ..., "passage": ...})."""
    passages = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            passages[str(obj["pid"])] = obj["passage"]
    logging.info("Loaded %d passages from %s", len(passages), path)
    return passages


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------

def resolve_score_token_ids(tokenizer) -> Tuple[Dict[int, int], List[int]]:
    """
    Identify token IDs for score digits "0", "1", "2", "3".

    Checks both plain ("0") and space-prefixed (" 0") variants, since
    the tokenizer may encode them differently depending on context.

    Returns:
        token_to_score: mapping from token_id -> score value (0-3)
        allowed_ids:    list of all allowed token IDs (for the logits mask)
    """
    token_to_score: Dict[int, int] = {}
    for score in range(4):
        for variant in [str(score), f" {score}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_to_score[ids[0]] = score
                logging.debug(
                    "Score %d: variant %r -> token_id %d", score, variant, ids[0]
                )

    # Verify we have at least one token per score
    scores_found = set(token_to_score.values())
    if scores_found != {0, 1, 2, 3}:
        missing = {0, 1, 2, 3} - scores_found
        raise RuntimeError(
            f"Could not find token IDs for scores: {missing}. "
            "Check the tokenizer vocabulary."
        )

    allowed_ids = sorted(token_to_score.keys())
    logging.info(
        "Score token mapping: %s",
        {v: k for k, v in sorted(token_to_score.items(), key=lambda x: x[1])},
    )
    return token_to_score, allowed_ids


def build_logits_processor(allowed_ids: List[int]):
    """
    Returns a logits processor that masks all tokens except the allowed
    score tokens. Sets disallowed logits to -inf so the model can only
    produce a valid score.
    """
    allowed_set = set(allowed_ids)

    def processor(token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(logits, float("-inf"))
        for tid in allowed_ids:
            mask[tid] = 0.0
        return logits + mask

    return processor


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_chat_prompt(
    tokenizer, query: str, passage: str, max_passage_tokens: int = 512
) -> str:
    """
    Build a chat-formatted prompt using the model's chat template.
    Truncates passage if it exceeds max_passage_tokens.
    """
    # Truncate passage by tokens if needed
    passage_ids = tokenizer.encode(passage, add_special_tokens=False)
    if len(passage_ids) > max_passage_tokens:
        passage = tokenizer.decode(
            passage_ids[:max_passage_tokens], skip_special_tokens=True
        )
        logging.debug("Truncated passage to %d tokens", max_passage_tokens)

    user_content = USER_TEMPLATE.format(query=query, passage=passage)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_query(
    llm: LLM,
    tokenizer,
    query_id: str,
    query_text: str,
    passage_ids: List[str],
    passages: Dict[str, str],
    token_to_score: Dict[int, int],
    allowed_ids: List[int],
    max_passage_tokens: int = 512,
) -> List[dict]:
    """
    Score all judged passages for a single query.

    Returns a list of result dicts, one per passage:
        {
            "query_id": str,
            "passage_id": str,
            "score": int,
            "logprobs": {"0": float, "1": float, "2": float, "3": float},
            "probs": {"0": float, "1": float, "2": float, "3": float},
        }
    """
    # Build prompts for all passages in this query
    prompts = []
    valid_pids = []
    for pid in passage_ids:
        if pid not in passages:
            logging.warning(
                "Query %s: passage %s not found in passage file, skipping", query_id, pid
            )
            continue
        prompt = build_chat_prompt(
            tokenizer, query_text, passages[pid], max_passage_tokens
        )
        prompts.append(prompt)
        valid_pids.append(pid)

    if not prompts:
        logging.warning("Query %s: no valid passages to score", query_id)
        return []

    # Set up constrained decoding
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=20,  # request enough to capture all 4 score tokens
        logits_processors=[build_logits_processor(allowed_ids)],
    )

    # Generate scores (vLLM handles batching internally)
    outputs = llm.generate(prompts, sampling_params)

    # Extract scores and logprobs
    results = []
    for pid, output in zip(valid_pids, outputs):
        generated = output.outputs[0]

        # The generated token
        gen_token_id = generated.token_ids[0]

        # Determine score from token ID
        if gen_token_id in token_to_score:
            score = token_to_score[gen_token_id]
        else:
            logging.error(
                "Query %s, passage %s: unexpected token_id %d (token=%r). "
                "Falling back to logprobs argmax.",
                query_id, pid, gen_token_id, generated.text,
            )
            score = -1  # will be resolved below

        # Extract logprobs for score tokens from vLLM output
        # generated.logprobs is a list with one dict (one generated token)
        token_logprobs = generated.logprobs[0]  # dict: token_id -> Logprob

        raw_logprobs = {}  # score -> logprob
        for tid, score_val in token_to_score.items():
            if tid in token_logprobs:
                raw_logprobs[score_val] = token_logprobs[tid].logprob
            else:
                # Token not in top-k logprobs; assign a very low value
                raw_logprobs[score_val] = -100.0

        # If score was unresolved, use argmax of logprobs
        if score == -1:
            score = max(raw_logprobs, key=raw_logprobs.get)

        # Convert logprobs to probabilities via softmax over the 4 tokens
        # (Re-normalise in case some tokens were missing from top-k)
        max_lp = max(raw_logprobs.values())
        exp_vals = {s: math.exp(lp - max_lp) for s, lp in raw_logprobs.items()}
        total = sum(exp_vals.values())
        probs = {s: exp_vals[s] / total for s in range(4)}

        results.append(
            {
                "query_id": query_id,
                "passage_id": pid,
                "score": score,
                "logprobs": {str(s): round(raw_logprobs[s], 6) for s in range(4)},
                "probs": {str(s): round(probs[s], 6) for s in range(4)},
            }
        )

    return results


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> set:
    """Load set of completed query IDs from checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
        completed = set(data.get("completed_queries", []))
        logging.info("Resumed from checkpoint: %d queries already done", len(completed))
        return completed
    return set()


def save_checkpoint(checkpoint_path: str, completed: set):
    """Save set of completed query IDs to checkpoint file."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"completed_queries": sorted(completed)}, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score (query, passage) pairs with an LLM judge (0-3 scale)."
    )
    parser.add_argument("--queries", required=True, help="Path to queries TSV")
    parser.add_argument("--qrels", required=True, help="Path to qrels (TREC format)")
    parser.add_argument("--passages", required=True, help="Path to passages JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--max-passage-tokens",
        type=int,
        default=512,
        help="Maximum passage length in tokens (default: 512)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Load data ---
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    passages = load_passages(args.passages)

    # Determine which queries to score (intersection of queries and qrels)
    query_ids = sorted(set(queries.keys()) & set(qrels.keys()))
    logging.info("Queries to score: %d", len(query_ids))

    total_pairs = sum(len(qrels[qid]) for qid in query_ids)
    logging.info("Total (query, passage) pairs: %d", total_pairs)

    # Check for missing passages
    missing = 0
    for qid in query_ids:
        for pid in qrels[qid]:
            if pid not in passages:
                missing += 1
    if missing > 0:
        logging.warning(
            "%d judged passages not found in passage file. They will be skipped.", missing
        )

    # --- Load checkpoint ---
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = args.output + ".checkpoint.json"
    completed = load_checkpoint(checkpoint_path)

    remaining = [qid for qid in query_ids if qid not in completed]
    logging.info("Queries remaining: %d (of %d total)", len(remaining), len(query_ids))

    if not remaining:
        logging.info("All queries already scored. Nothing to do.")
        return

    # --- Initialize model ---
    logging.info("Loading model: %s", args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,  # sufficient for system + user prompt + passage
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Resolve score token IDs
    token_to_score, allowed_ids = resolve_score_token_ids(tokenizer)

    # --- Scoring loop ---
    logging.info("Starting scoring run")
    t_start = time.time()
    pairs_scored = 0

    # Open output file in append mode (for resumability)
    with open(args.output, "a", encoding="utf-8") as out_f:
        for i, qid in enumerate(remaining):
            query_text = queries[qid]
            passage_ids = sorted(qrels[qid].keys())

            logging.info(
                "Scoring query %d/%d: %s (%d passages)",
                i + 1, len(remaining), qid, len(passage_ids),
            )

            results = score_query(
                llm=llm,
                tokenizer=tokenizer,
                query_id=qid,
                query_text=query_text,
                passage_ids=passage_ids,
                passages=passages,
                token_to_score=token_to_score,
                allowed_ids=allowed_ids,
                max_passage_tokens=args.max_passage_tokens,
            )

            # Write results for this query
            for r in results:
                out_f.write(json.dumps(r) + "\n")
            out_f.flush()

            # Update checkpoint
            completed.add(qid)
            save_checkpoint(checkpoint_path, completed)
            pairs_scored += len(results)

            # Progress logging
            elapsed = time.time() - t_start
            rate = pairs_scored / elapsed if elapsed > 0 else 0
            logging.info(
                "  -> %d pairs scored so far (%.1f pairs/sec, %.1f min elapsed)",
                pairs_scored, rate, elapsed / 60,
            )

    elapsed_total = time.time() - t_start
    logging.info(
        "Done. Scored %d pairs across %d queries in %.1f minutes.",
        pairs_scored, len(remaining), elapsed_total / 60,
    )
    logging.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
