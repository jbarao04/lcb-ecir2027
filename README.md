# LCB: Choosing Which LLM Relevance Grades a Human Should Check

Code and LLM grades for a paper under review.

LCB (Leverage-Confidence Based selection) starts from a test collection graded entirely by an LLM. It sends query-passage pairs to a human assessor in batches of 1% of the pairs, choosing first the pairs with the largest leverage times expected squared error. The goal is a mixed test collection that orders the twenty best systems as the human grades do, with a small human budget.

## Contents

```
grades/            LLM grades and grade probabilities (Llama and Qwen)
data_setup/        Download and preparation of the TREC DL data
scoring/           LLM grading (GPU)
triage/            Selection policies, protocol and bootstrap
v2_id_mapping.py   Maps near-duplicate MS MARCO v2 passages to one id
```

## LLM grades

| File | Collections |
|---|---|
| `grades/llama-3.1-8b_v1.jsonl` | TREC DL 2019 and 2020 (MS MARCO v1) |
| `grades/llama-3.1-8b_v2.jsonl` | TREC DL 2021 to 2023 (MS MARCO v2) |
| `grades/qwen2.5-7b_v1.jsonl` | TREC DL 2019 and 2020 (MS MARCO v1) |
| `grades/qwen2.5-7b_v2.jsonl` | TREC DL 2021 to 2023 (MS MARCO v2) |

Each line is one graded pair, with the fields `query_id`, `passage_id`, `score` (the LLM grade, 0 to 3), `logprobs` and `probs` (the log-probability and probability of each grade). Passage texts are not included.

## LLM configuration

- **Models:** Llama-3.1-8B-Instruct (main) and Qwen2.5-7B-Instruct
- **Prompt:** UMBRELA (Upadhyay et al., 2024), grades on a 0 to 3 scale
- **Decoding:** constrained to the tokens {0, 1, 2, 3}, temperature 0
- **Output:** the grade and the probability of each of the four grades

## Data

The TREC data are not redistributed. The scripts expect them in this layout, relative to the repository root:

```
data_prep/data/trec-dl/<year>/queries.tsv, qrels.txt       2019, 2020
data_prep/data/trec-dl-v2/<year>/qrels_dedup.txt           2021 to 2023
data/system_runs/<year>/                                    submitted runs
data/run_groups.csv                                         run -> participating group
```

Run all commands from the repository root.

1. **2019 and 2020:** download the queries and qrels from the [TREC Deep Learning track](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html).
2. **2021 to 2023:** download the queries and qrels, then map near-duplicate passages to one id.
   ```
   python data_setup/setup_trec_dl_v2.py --data-dir data_prep/data/trec-dl-v2 --skip-corpus
   python data_setup/dedup_v2_qrels.py --data-dir data_prep/data/trec-dl-v2
   ```
3. **Submitted runs:** these require the login of the [TREC results archive](https://trec.nist.gov/results.html). Set it in `TREC_USER` and `TREC_PASSWORD`, then run:
   ```
   python data_setup/download_trec_dl_runs.py
   ```
4. **Run groups** (for the held-out systems experiment only): map each run to the group that submitted it, from the public TREC run metadata.
   ```
   python data_setup/build_run_groups.py
   ```

The MS MARCO passage corpora are needed only to grade the pairs again with an LLM. Drop `--skip-corpus` to download the v2 corpus.

## Reproducing the paper

1. **LLM grading** (optional, GPU; the grades in `grades/` can be used instead)
   ```
   python scoring/score_passages.py --queries <queries.tsv> --qrels <qrels.txt> --passages <passages.jsonl> --output <grades.jsonl>
   ```
   The default model is Llama-3.1-8B-Instruct. Add `--model Qwen/Qwen2.5-7B-Instruct` for Qwen.

2. **Main experiment** (LCB, Leverage, the error oracle and the seven baselines, 1000 bootstrap resamples; Table 2)
   ```
   python triage/run_t12_resampling.py --B 1000
   python triage/run_t12_resampling.py --B 1000 --judge qwen
   ```
   The first command uses the Llama grades and writes to `results/t12_resampling/`; the second uses the Qwen grades and writes to `results/t12_resampling_qwen/`. Every policy is computed by default, including LARA with n = N (the `lara_nN` row); `--lara-groups` only sets n for the separate `lara` row (default 1). `--years` restricts the run to some years.

3. **Gamed run** (30 gamed runs, Llama; Table 3)
   ```
   python triage/run_adversary.py
   ```
   Defaults: 1000 bootstrap resamples, all five years. Writes to `results/adversary/`.

4. **Held-out systems** (one group of systems held out at a time, 62 groups, Llama; needs `data/run_groups.csv`, see Data step 4)
   ```
   python triage/run_reusability.py
   ```
   Writes to `results/reusability/`.

Results are written to `results/`.

## Requirements

Python 3.11. Install with `pip install -r requirements.txt`. The main dependencies are numpy, scipy, pandas, scikit-learn, requests and tqdm. LLM grading also needs vLLM 0.6.6.post1, transformers 4.48.0 and one 24 GB GPU.
