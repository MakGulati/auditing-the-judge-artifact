#!/usr/bin/env python3
"""Measure how large --solve_max_tokens needs to be, instead of guessing.

The problem this solves is a censored measurement. A run at budget B tells you the
length of every solution SHORTER than B, and nothing at all about the ones that hit
it — they are all recorded as exactly B. So the existing raw.jsonl cannot answer
"what budget would have been enough"; the answer was truncated away with the text.

This script re-solves a sample of the previously-truncated problems at a deliberately
generous budget, measures where they actually stop, and reports the budget that would
cover a given fraction of the WHOLE dataset.

Two details that make the arithmetic non-obvious:

  * The truncated records are not a random sample. They are the upper tail, so their
    own p99 is far above the dataset's p99. Because every truncated solution is longer
    than the old budget and every completed one is shorter, the strata do not overlap
    and the combined quantile maps cleanly: to cover q of the dataset when a fraction
    `t` was truncated, you need the (q - (1-t)) / t quantile of the truncated stratum.
    Covering 99% of a dataset that was 24% truncated means the ~96th percentile of
    this sample — which is why the sample has to be large enough to have points there.

  * The measuring budget must not itself truncate, or the answer is censored again.
    The script reports how many sampled solutions hit the ceiling and refuses to
    recommend a number if any did.

Example:
  python calibrate_solve_budget.py --model google/gemma-3-12b-it --backend vllm \\
    --input results_gemma3_math_train/math/raw.jsonl --dataset math \\
    --sample 500 --probe_max_tokens 8192 --max-model-len 16384
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys

from dataset_registry import DATASETS, prompts_for
from metrics.correctness import DEFAULT_DATASET


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def truncated_records(records: list[dict], assume_budget: int, tokenizer) -> list[dict]:
    """The records whose solutions the old budget cut off.

    Prefers the recorded ``truncated`` flag. Files generated before that field existed
    have to be detected by length: a solution sitting exactly on the old budget was
    almost certainly cut off there. That inference is only sound if you tell the script
    what the old budget WAS, hence --assume_budget.
    """
    flagged = [r for r in records if r.get("truncated") is True]
    if flagged:
        return flagged
    if tokenizer is None:
        raise SystemExit(
            "[FATAL] no record carries a `truncated` flag, so truncation must be "
            "inferred from solution length — which needs the tokenizer. Pass --model "
            "(and --assume_budget for the budget that run used)."
        )
    print(f"[info] no `truncated` flags (file predates them); inferring from length "
          f"at --assume_budget {assume_budget}", flush=True)
    out = []
    for r in records:
        n = len(tokenizer(r.get("phase1_solution") or "",
                          add_special_tokens=False)["input_ids"])
        # A couple of tokens of slack: re-tokenizing decoded text is not exactly the
        # generation token count.
        if n >= assume_budget - 4:
            out.append(r)
    return out


def combined_quantile(sorted_lengths: list[int], truncated_fraction: float,
                      q: float) -> int | None:
    """Length covering `q` of the dataset, given `truncated_fraction` was censored.

    Returns None when `q` falls inside the un-truncated part, where the old budget
    already sufficed and this sample says nothing.
    """
    completed_fraction = 1.0 - truncated_fraction
    if q <= completed_fraction:
        return None
    within = (q - completed_fraction) / truncated_fraction
    idx = min(int(round(within * (len(sorted_lengths) - 1))), len(sorted_lengths) - 1)
    return sorted_lengths[idx]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="raw.jsonl from the truncated run")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--backend", default="vllm", choices=["vllm", "api", "transformers"])
    ap.add_argument("--sample", type=int, default=500,
                    help="how many truncated problems to re-solve. The recommended "
                         "budget comes from this sample's upper tail, so too few "
                         "points there makes the recommendation noisy (default 500).")
    ap.add_argument("--probe_max_tokens", type=int, default=8192,
                    help="budget to measure at. Must be generous enough that nothing "
                         "hits it, or the measurement is censored all over again.")
    ap.add_argument("--assume_budget", type=int, default=1024,
                    help="the budget the input run used, for files predating the "
                         "`truncated` flag")
    ap.add_argument("--coverage", type=float, default=0.99,
                    help="fraction of the dataset the recommended budget should cover")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92,
                    dest="gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    ap.add_argument("--max_concurrent", type=int, default=16)
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    if args.max_model_len and args.max_model_len <= args.probe_max_tokens:
        ap.error(f"--max-model-len {args.max_model_len} must exceed --probe_max_tokens "
                 f"{args.probe_max_tokens} with room for the prompt, or the measurement "
                 f"is censored by the context window instead of by the budget")

    records = load_records(args.input)
    tokenizer = None
    if not any(r.get("truncated") is True for r in records):
        from extraction_common import load_tokenizer
        if not args.model:
            ap.error("--model is required to infer truncation on a file with no flags")
        tokenizer = load_tokenizer(args.model)

    truncated = truncated_records(records, args.assume_budget, tokenizer)
    fraction = len(truncated) / len(records)
    print(f"input: {len(records)} records, {len(truncated)} truncated "
          f"({fraction:.1%})", flush=True)
    if not truncated:
        print("nothing was truncated; the current budget already suffices.")
        return
    if len(truncated) < args.sample:
        print(f"[info] only {len(truncated)} truncated records available; using all")
    sample = random.Random(args.seed).sample(truncated, min(args.sample, len(truncated)))

    # Re-solve at the generous budget.
    sys.argv = [sys.argv[0]]           # run_eval's parser must not see our flags
    from run_eval import build_backend
    backend = build_backend(argparse.Namespace(
        backend=args.backend, model=args.model, gguf_file="", tokenizer=None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=None, quantization=None,
        kv_cache_dtype=None, mistral_format=False, assistant_prefill="",
        max_concurrent=args.max_concurrent))
    prompts = prompts_for(args.dataset)()

    from extraction_common import load_tokenizer as _lt
    tok = tokenizer or _lt(args.model)
    sem = asyncio.Semaphore(args.max_concurrent)

    async def one(rec):
        async with sem:
            from pipeline.solver import solve_greedy
            text, answer, was_truncated = await solve_greedy(
                rec["problem"], prompts, backend, args.probe_max_tokens)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        return n, was_truncated, answer is not None

    from tqdm import tqdm
    results = []
    tasks = [one(r) for r in sample]
    bar = tqdm(total=len(tasks), desc=f"re-solving at {args.probe_max_tokens}")
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
        bar.update(1)
    bar.close()
    try:
        await backend.aclose()
    except Exception as exc:                                   # noqa: BLE001
        print(f"[WARN] backend.aclose() failed (ignored): {exc}", file=sys.stderr)

    lengths = sorted(n for n, _, _ in results)
    still_cut = sum(1 for _, t, _ in results if t)
    now_answered = sum(1 for _, _, ok in results if ok)
    print(f"\nre-solved {len(lengths)} previously-truncated problems at "
          f"{args.probe_max_tokens} tokens")
    print(f"  now produce an answer : {now_answered}/{len(lengths)} "
          f"({now_answered/len(lengths):.1%})")
    print(f"  still hit the ceiling : {still_cut}")
    print(f"  their true lengths    : median={lengths[len(lengths)//2]}  "
          f"p90={lengths[int(.90*len(lengths))]}  p99={lengths[int(.99*len(lengths))]}  "
          f"max={lengths[-1]}")

    report = {"input": args.input, "dataset": args.dataset, "model": args.model,
              "n_records": len(records), "n_truncated": len(truncated),
              "truncated_fraction": fraction, "n_sampled": len(lengths),
              "probe_max_tokens": args.probe_max_tokens, "still_truncated": still_cut,
              "sample_lengths": {"median": lengths[len(lengths) // 2],
                                 "p90": lengths[int(.90 * len(lengths))],
                                 "p99": lengths[int(.99 * len(lengths))],
                                 "max": lengths[-1]}}

    if still_cut:
        print(f"\n[FATAL] {still_cut} sampled solution(s) hit {args.probe_max_tokens} "
              f"tokens, so their true length is still unknown and any recommendation "
              f"would be a floor, not an answer. Re-run with a larger "
              f"--probe_max_tokens.")
        report["recommendation"] = None
    else:
        print(f"\nbudget needed to cover the WHOLE dataset "
              f"(truncated stratum was {fraction:.1%}):")
        for q in (0.95, 0.99, 0.999, 1.0):
            need = combined_quantile(lengths, fraction, q)
            label = f"  {q:.1%} of records"
            if need is None:
                print(f"{label}: already covered by the old budget")
            else:
                print(f"{label}: {need} tokens")
        rec = combined_quantile(lengths, fraction, args.coverage) or args.assume_budget
        # Round up to a power-of-two-ish value; engines and KV budgeting like round
        # numbers, and the extra headroom costs nothing when nothing reaches it.
        rounded = 1 << (int(rec) - 1).bit_length()
        report["recommendation"] = {"coverage": args.coverage, "raw": rec,
                                    "solve_max_tokens": rounded}
        print(f"\n  --solve_max_tokens {rounded}   (covers {args.coverage:.1%}, "
              f"rounded up from {rec})")
        print(f"\nAlso raise --max-model-len: the judge prompt embeds the whole "
              f"solution,\n  so it must hold (longest problem + {rounded} tokens). "
              f"Check the run's\n  report or measure the prompts before choosing.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
