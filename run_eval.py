#!/usr/bin/env python3
"""Entrypoint for the LLM self-judge evaluation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from dataset_registry import DATASETS, dataset_split, loader_for, prompts_for
from pipeline.backend import APIBackend, LLMBackend, TransformersBackend, VLLMBackend
from pipeline.solver import DEFAULT_SOLVE_MAX_TOKENS, solve_greedy
from pipeline.judge import judge_solution
from pipeline.majority import majority_vote_sample
from metrics.confusion import generate_report
from metrics.correctness import (DEFAULT_DATASET, answers_equal, label_policy_for,
                                 relabel_result)


# Number of leading problems hashed into the fingerprint. The loader shuffles then
# truncates, so the prefix is stable as --n_problems grows: a 100-problem pilot and a
# full 7473-problem run share this digest, while a different seed, split, or dataset
# revision changes it. Hashing all N would spuriously differ whenever N grew.
_FINGERPRINT_PROBES = 64


def problems_digest(examples: list[dict], k: int | None = None) -> str:
    """SHA-256 over the first ``k`` question texts — identifies the actual sampled set.

    Stronger than recording a dataset revision string: ``Dataset.shuffle(seed=...)``
    permutes via the ``datasets`` library's internal RNG, so a library upgrade can
    re-order the same underlying data. The questions themselves cannot.

    ``k`` defaults to ``_FINGERPRINT_PROBES``, but a run smaller than that hashes only
    what it has, so the width is recorded alongside the digest (``problems_fingerprint_n``)
    and a later, larger run re-hashes at the stored width before comparing. Without
    that, a 40-problem pilot could never be grown: its digest covers 40 questions and
    the resumed run's covers 64, so identical data would read as a different sample.
    """
    h = hashlib.sha256()
    for ex in examples[:_FINGERPRINT_PROBES if k is None else k]:
        h.update(str(ex.get("question", "")).encode())
        h.update(b"\0")
    return h.hexdigest()


def run_fingerprint(args: argparse.Namespace, examples: list[dict]) -> dict:
    """Identity of the run, so a resume can't silently mix incompatible records.

    ``idx`` is a position in the shuffled subset, not a stable dataset ID, so a
    resume against a different seed/split/dataset pairs old records with different
    problems. ``model``/``backend`` matter for the same reason one stage later:
    ``raw.jsonl`` carries no per-record provenance, so two models' generations can
    land in one file and be extracted with a single model. (The extractor's own
    ``Dump._check_meta`` already refuses that; this closes the same hole upstream.)
    """
    return {
        # which problems, in which order
        "dataset": args.dataset,
        "split": dataset_split(args.dataset),
        "seed": args.seed,
        "problems_sha256": problems_digest(examples),
        "problems_fingerprint_n": min(_FINGERPRINT_PROBES, len(examples)),
        # How many problems this invocation covers. NOT an identity field: the sample
        # may grow between resumes. It may not SHRINK — the digest only fingerprints a
        # 64-problem prefix, so a 100-record directory resumed with --n_problems 80
        # passes every other check while still reporting all 100 old records, i.e. a
        # sample size that silently disagrees with the one requested.
        "n_problems": len(examples),
        # which weights produced the text
        "model": args.model,
        "backend": args.backend,
        "gguf_file": args.gguf_file or None,
        "quantization": args.quantization or None,
        "kv_cache_dtype": args.kv_cache_dtype or None,
        # how the prompt was templated and how much the model was allowed to say
        "tokenizer": args.tokenizer or None,
        # Retained as a false-valued compatibility field because the archived
        # submission metadata fingerprints this schema version.
        "mistral_format": False,
        "assistant_prefill": args.assistant_prefill,
        "judge_max_tokens": args.judge_max_tokens,
        "solve_max_tokens": args.solve_max_tokens,
        "max_model_len": args.max_model_len,
        # record shape: k_solutions/k_answers length, and whether a majority exists
        "k_samples": args.k_samples,
        "label_policy": label_policy_for(args.dataset),
    }


# Everything in the fingerprint is an identity key except these: a mismatch means the
# existing records describe a different experiment and must not be resumed into.
#
#   - quantization / kv_cache_dtype / gguf_file change the weights, so they change the
#     generated text even though nothing about the problem set moved.
#   - tokenizer / assistant_prefill change how the prompt is
#     templated, and extraction reproduces that templating from the model id alone.
#   - judge_max_tokens changes the judge's reply budget, which changes verdicts;
#     max_model_len can truncate a prompt.
#   - k_samples sets the length of k_solutions/k_answers, so mixing values yields a
#     file whose majority statistics are computed over inconsistent k.
#
# label_policy is advisory: labels are recomputed from the raw answers at analysis
# time, so an old policy in the file changes nothing downstream.
#
# Purely operational knobs are deliberately NOT fingerprinted, because they cannot
# change what was recorded: gpu_memory_utilization, max_num_seqs, max_concurrent.
_ADVISORY_KEYS = {"label_policy"}

# Sample size, not identity: it may grow between resumes but never shrink, so it is
# compared with >= rather than ==. `problems_fingerprint_n` moves with it because the
# digest widens as a sub-64-problem run grows.
_MONOTONIC_KEYS = ("n_problems", "problems_fingerprint_n")


def check_fingerprint(meta_path: Path, current: dict, *, raw_exists: bool,
                      adopt_existing: bool = False, digest_at=None) -> None:
    """Refuse to resume into a directory that describes a different run.

    ``digest_at(k)`` re-hashes this invocation's problem set over its first ``k``
    questions; it is needed only when the stored fingerprint is narrower than the
    current one (a pilot smaller than ``_FINGERPRINT_PROBES`` being grown).
    """
    if not meta_path.exists():
        if raw_exists and not adopt_existing:
            raise SystemExit(
                f"[FATAL] {meta_path.parent} already holds raw.jsonl but no "
                f"{meta_path.name}, so the run that produced it cannot be identified.\n"
                f"        Writing a fingerprint now would stamp it with THIS invocation's "
                f"identity (split={current['split']!r}, seed={current['seed']}, "
                f"model={current['model']!r}) regardless of what actually generated it.\n"
                f"        Re-run with --adopt_existing if you are certain the settings "
                f"match, or use a fresh --output_dir."
            )
        if raw_exists:
            print(f"[WARN] adopting pre-existing {meta_path.parent}/raw.jsonl under the "
                  f"current settings on your say-so (--adopt_existing).", file=sys.stderr)
        meta_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        return

    stored = json.loads(meta_path.read_text())
    # --n_problems may legitimately grow between resumes; identity may not.
    drift, missing, shrunk, backfill = {}, {}, {}, {}
    comparable = dict(current)
    stored_width = stored.get("problems_fingerprint_n")
    if (digest_at is not None and stored_width
            and stored_width < current.get("problems_fingerprint_n", 0)):
        # The stored digest covers fewer questions than this run would hash. Compare on
        # the prefix they share, or growing a sub-64-problem pilot would always look
        # like a different problem set.
        comparable["problems_sha256"] = digest_at(stored_width)
    for k, v in comparable.items():
        if k in _ADVISORY_KEYS:
            continue
        if k not in stored:
            # A missing *identity* field is unknown, not matching, and backfilling it
            # would assert a provenance nobody checked. The size fields are neither:
            # recording this invocation's value cannot mislabel a record, it only means
            # the shrink check has nothing to compare against until the next resume.
            # (`main` independently refuses a sample smaller than the recorded idx
            # values, which is direct evidence and needs no stored field.)
            if k not in _MONOTONIC_KEYS:
                missing[k] = current[k]
            else:
                backfill[k] = current[k]
        elif k in _MONOTONIC_KEYS:
            if stored[k] > v:
                shrunk[k] = (stored[k], v)
        elif stored[k] != v:
            drift[k] = (stored[k], v)
    if shrunk:
        detail = "; ".join(f"{k}: existing={old!r} requested={new!r}"
                           for k, (old, new) in sorted(shrunk.items()))
        raise SystemExit(
            f"[FATAL] {meta_path} covers a LARGER sample than this invocation "
            f"({detail}).\n"
            f"        The problem-set digest only fingerprints the first "
            f"{_FINGERPRINT_PROBES} questions, so shrinking the sample passes every "
            f"other check while raw.jsonl keeps — and the report keeps counting — the "
            f"records beyond the new size.\n"
            f"        Use a fresh --output_dir for the smaller run."
        )
    if drift:
        detail = "; ".join(f"{k}: existing={old!r} requested={new!r}"
                           for k, (old, new) in sorted(drift.items()))
        raise SystemExit(
            f"[FATAL] {meta_path} describes a different run ({detail}).\n"
            f"        'idx' is a position in the shuffled subset and raw.jsonl carries no "
            f"per-record provenance, so resuming here would mix incompatible records.\n"
            f"        Use a fresh --output_dir, or delete {meta_path.parent} to restart."
        )
    # An identity field absent from an older fingerprint is UNKNOWN, not matching.
    # Backfilling it silently would stamp this invocation's value onto records that may
    # have been generated with a different one — the same failure the raw.jsonl-without-
    # run_meta.json check above refuses. Require the same explicit assertion.
    if missing and not adopt_existing:
        raise SystemExit(
            f"[FATAL] {meta_path} predates fingerprint field(s) "
            f"{', '.join(sorted(missing))}, so the value(s) used to generate the "
            f"existing records are unknown.\n"
            f"        This invocation would use "
            + "; ".join(f"{k}={v!r}" for k, v in sorted(missing.items())) + ".\n"
            f"        Recording those without verifying them would assert a provenance "
            f"that was never checked.\n"
            f"        Re-run with --adopt_existing if you are certain the settings match, "
            f"or use a fresh --output_dir."
        )
    if missing:
        print(f"[WARN] adopting {meta_path.name} field(s) {', '.join(sorted(missing))} "
              f"under the current settings on your say-so (--adopt_existing). They were "
              f"NOT verified against the existing records.", file=sys.stderr)
    policy_changed = stored.get("label_policy") != current["label_policy"]
    if policy_changed:
        print(f"[WARN] {meta_path} was written under label policy "
              f"{stored.get('label_policy')!r}; current is {current['label_policy']!r}. "
              f"Labels are recomputed at analysis time, so this is safe.", file=sys.stderr)
    # The sample grew: record the new size and, if the digest widened with it, the
    # wider digest — so the next resume compares against what is actually on disk.
    grew = {k: current[k] for k in _MONOTONIC_KEYS
            if k in stored and stored[k] < current[k]}
    if grew:
        grew["problems_sha256"] = current["problems_sha256"]
        print(f"[INFO] sample grew to n_problems={current['n_problems']} "
              f"(was {stored.get('n_problems', 'unknown')}); resuming and extending.",
              file=sys.stderr)
    if missing or policy_changed or grew or backfill:
        stored.update(missing)
        stored.update(backfill)
        stored.update(grew)
        stored["label_policy"] = current["label_policy"]
        meta_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")


def read_raw_records(raw_path: Path) -> tuple[dict[int, dict], dict[str, int]]:
    """Parse raw.jsonl defensively, keyed by idx. Returns (records, repair stats).

    Three things can be wrong with a file a killed run left behind:

      * a **partial final line**, if the process died between ``write`` and ``flush``.
        It is dropped here, and the caller rewrites the file so the next appended
        record is not concatenated onto the corrupt tail — which would destroy a
        *second*, previously valid record.
      * **duplicate idx values**, from a resume that ran before this repair existed.
        Both rows are then counted in the report while the raw->activation mapping
        keeps only one, so the two disagree. The successful record wins over an error
        record for the same idx; otherwise the last one does.
      * lines that parse but carry no ``idx``, which nothing downstream can align.
    """
    records: dict[int, dict] = {}
    stats = {"lines": 0, "malformed": 0, "duplicate": 0}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                record = json.loads(line)
                idx = record["idx"]
            except (json.JSONDecodeError, KeyError, TypeError):
                stats["malformed"] += 1
                continue
            if idx in records:
                stats["duplicate"] += 1
                if records[idx].get("error") is None and record.get("error") is not None:
                    continue
            records[idx] = record
    return records, stats


def rewrite_raw(raw_path: Path, records: dict[int, dict]) -> None:
    """Replace raw.jsonl with exactly ``records``, atomically and in idx order."""
    tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for idx in sorted(records):
            f.write(json.dumps(records[idx]) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, raw_path)


def _labelable_snapshot(gold_answer, dataset: str, truncated: bool | None) -> bool:
    """Best-effort `labelable` for the record being written.

    Analysis never trusts this: `relabel_result` recomputes it from the record. It is
    written correctly anyway so that a raw.jsonl inspected directly does not contradict
    the analysis that reads it.
    """
    from metrics.correctness import gold_is_parsable

    if truncated is True:
        return False
    return gold_is_parsable(gold_answer, dataset)


async def run_problem(
    idx: int,
    example: dict,
    gold_answer: str | None,
    prompts,
    backend: LLMBackend,
    k_samples: int,
    judge_max_tokens: int = 16,
    dataset: str = DEFAULT_DATASET,
    solve_max_tokens: int = DEFAULT_SOLVE_MAX_TOKENS,
) -> dict:
    problem = example["question"]

    # Phase 1 solve and Phase 2 k-sampling run concurrently; judge waits for solve.
    solve_task = asyncio.create_task(
        solve_greedy(problem, prompts, backend, solve_max_tokens))
    majority_task = asyncio.create_task(
        majority_vote_sample(problem, prompts, backend, k_samples,
                             solve_max_tokens)
    )

    try:
        phase1_solution, phase1_answer, phase1_truncated = await solve_task
        solve_correct = answers_equal(phase1_answer, gold_answer, dataset)

        judge_verdict = await judge_solution(problem, phase1_solution, prompts, backend,
                                             max_tokens=judge_max_tokens)

        k_solutions, k_answers, majority_answer, k_truncated = await majority_task
        majority_correct = answers_equal(majority_answer, gold_answer, dataset)
    except BaseException:
        # The two phases are independent tasks, so a failure in one leaves the other
        # running: its request keeps consuming a GPU slot or an API quota that nothing
        # is waiting for any more, and under repeated failures those orphans accumulate
        # for the rest of the run. Cancel and reap both before the error propagates.
        for task in (solve_task, majority_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(solve_task, majority_task, return_exceptions=True)
        raise

    return {
        "idx": idx,
        # Every consumer downstream (labelling, extraction, the probe) needs to know
        # which equivalence policy this record's answers obey. raw.jsonl carries no
        # other per-record provenance, so it is recorded per record, not just in
        # run_meta.json.
        "dataset": dataset,
        "problem": problem,
        "gold_answer": gold_answer,
        "phase1_solution": phase1_solution,
        "phase1_answer": phase1_answer,
        # True when the budget cut the solution off before it could write its
        # final answer. Such a record parses to no answer and would otherwise be
        # labelled a wrong answer, which is a claim about the model rather than
        # about our budget. None = the backend could not report it.
        "truncated": phase1_truncated,
        "solve_correct": solve_correct,
        "judge_verdict": judge_verdict,
        "k_solutions": k_solutions,
        "k_answers": k_answers,
        "majority_answer": majority_answer,
        "k_truncated": k_truncated,
        "majority_correct": majority_correct,
        # Advisory snapshot only; `metrics.correctness.is_labelable` recomputed at
        # analysis time is authoritative. Written from the same function here so the
        # file is not self-contradictory: the old expression was `gold_answer is not
        # None`, which called a budget-truncated record labelable when the analysis
        # correctly excluded it, and disagreed on 7 GSM8K-train, 197 MATH-train and
        # 117 MATH-test records.
        "labelable": _labelable_snapshot(gold_answer, dataset, phase1_truncated),
        "label_policy": label_policy_for(dataset),
    }


async def run_problem_safe(idx: int, example: dict, gold_answer: str | None, **kwargs) -> dict:
    try:
        return await run_problem(idx, example, gold_answer, **kwargs)
    except Exception as exc:
        print(f"\n[WARN] Problem {idx} failed: {exc}", file=sys.stderr)
        k = kwargs.get("k_samples", 0)
        dataset = kwargs.get("dataset", DEFAULT_DATASET)
        # The placeholder fields below are NOT observations. `error` is the marker
        # that makes this record unlabelable downstream (metrics.correctness.
        # is_labelable); without it an OOM would read as a genuine wrong answer that
        # the judge correctly rejected — a free true negative for the judge and a
        # trivially separable training example for the probe.
        return {
            "idx": idx,
            "dataset": dataset,
            "problem": example.get("question", ""),
            "gold_answer": gold_answer,
            "phase1_solution": "",
            "phase1_answer": None,
            "truncated": None,
            "solve_correct": False,
            "judge_verdict": None,
            "k_solutions": [""] * k,
            "k_answers": [None] * k,
            "k_truncated": [None] * k,
            "majority_answer": None,
            "majority_correct": False,
            # Generation raised, so this record is unlabelable under any policy.
            "labelable": False,
            "label_policy": label_policy_for(dataset),
            "error": str(exc),
        }


def build_backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend == "transformers":
        return TransformersBackend(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    if args.backend == "vllm":
        return VLLMBackend(
            model=args.model,
            gguf_file=args.gguf_file or None,
            tokenizer=args.tokenizer or None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len or None,
            max_num_seqs=args.max_num_seqs or None,
            quantization=args.quantization or None,
            kv_cache_dtype=args.kv_cache_dtype or None,
            assistant_prefill=args.assistant_prefill,
        )

    # API backend (default)
    from openai import AsyncOpenAI
    api_key = os.environ.get("OPENAI_API_KEY") or "dummy"
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(args.max_concurrent)
    return APIBackend(client, args.model, semaphore)


async def main() -> None:
    parser = argparse.ArgumentParser(description="LLM self-judge evaluation pipeline")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS),
                        help="'gsm8k' (numeric answers, $GSM8K_SPLIT) or 'math' "
                             "(Hendrycks MATH, LaTeX answers, $MATH_SPLIT / "
                             "$MATH_SUBJECTS / $MATH_LEVELS)")
    parser.add_argument("--n_problems", type=int, default=200)
    parser.add_argument("--k_samples", type=int, default=5)
    parser.add_argument(
        "--model",
        default="google/gemma-3-12b-it",
        help="HF repo ID (vllm backend) or model name for the API backend",
    )
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--seed", type=int, default=42)

    # Backend selection
    parser.add_argument(
        "--backend", default="vllm", choices=["vllm", "api", "transformers"],
        help="'vllm' runs the model in-process (GGUF/safetensors); 'api' calls an "
             "OpenAI-compatible HTTP endpoint; 'transformers' is a slower in-process "
             "Hugging Face fallback",
    )

    # vLLM-specific args
    parser.add_argument(
        "--gguf-file",
        default="",
        help="Filename of a .gguf quant within the HF repo to load (vllm backend only). "
             "Empty (default) loads the repo's HF safetensors. Only set this for an "
             "actual GGUF repository and filename.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF repo ID for the tokenizer (vllm backend). Defaults to --model.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        dest="gpu_memory_utilization")
    parser.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    parser.add_argument("--max-num-seqs", type=int, default=None, dest="max_num_seqs",
                        help="cap engine max concurrent seqs + sampler-warmup batch (vllm). "
                             "Set low (e.g. 16) for a big model on a small card to avoid a "
                             "warmup OOM even when weights+KV fit.")
    parser.add_argument("--quantization", default=None,
                        help="vllm weight quant, e.g. 'fp8' (online): ~halves weight VRAM so "
                             "the KV cache can hold a much larger context (4k->32k for a 14B "
                             "on a 32 GB card). Generation-only; extraction still loads bf16.")
    parser.add_argument("--kv-cache-dtype", default=None, dest="kv_cache_dtype",
                        help="vllm KV cache dtype, e.g. 'fp8' to ~halve KV memory for even "
                             "more context (stacks with --quantization).")
    parser.add_argument("--solve_max_tokens", type=int,
                        default=DEFAULT_SOLVE_MAX_TOKENS,
                        help="max tokens for a solution. The final answer is the "
                             "LAST thing written, so a solution cut off here has "
                             "none and is labelled wrong — a claim about the model, "
                             "not about the budget. 1024 fits GSM8K but truncates "
                             "~24%% of MATH; raise it (and --max-model-len, which "
                             "must also hold the judge prompt containing the whole "
                             "solution) for harder datasets.")
    parser.add_argument("--judge_max_tokens", type=int, default=16,
                        help="max tokens for the judge reply. Raise it for "
                             "justification-first judges whose YES/NO arrives at the "
                             "end of the reply.")
    parser.add_argument("--assistant_prefill", default="",
                        help="text appended after the chat template's generation prompt, "
                             "as if the assistant had already written it. vLLM backend "
                             "only.")
    # API-specific args
    parser.add_argument("--max_concurrent", type=int, default=8,
                        help="Max simultaneous requests (api backend only)")

    parser.add_argument("--retry_failed", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="on resume, re-run records whose generation raised (API "
                             "error, OOM, timeout). These are transient failures, but "
                             "they are written to raw.jsonl like any other record, so "
                             "without this a resume treats them as permanently done and "
                             "the evaluated sample silently shrinks. "
                             "--no-retry-failed keeps the old skip-everything behaviour.")

    parser.add_argument("--adopt_existing", action="store_true",
                        help="permit resuming when the existing run's provenance cannot be "
                             "verified: an output_dir with raw.jsonl but no run_meta.json, "
                             "or a run_meta.json that predates some fingerprint fields. In "
                             "both cases you are asserting the settings match the records "
                             "already on disk; nothing verifies it.")

    args = parser.parse_args()

    # Numeric knobs whose invalid values fail late and obscurely: --max_concurrent 0
    # deadlocks on a zero-capacity semaphore after the model is loaded, and a
    # non-positive count produces an empty or malformed run that still writes a report.
    for flag, value, minimum in (("--n_problems", args.n_problems, 1),
                                 ("--k_samples", args.k_samples, 1),
                                 ("--max_concurrent", args.max_concurrent, 1),
                                 ("--judge_max_tokens", args.judge_max_tokens, 1)):
        if value < minimum:
            parser.error(f"{flag} must be >= {minimum} (got {value})")

    # --assistant_prefill changes the prompt the model continues from, so generating
    # without it and extracting with it reads activations at a different position than
    # produced the verdict. Only VLLMBackend threads it through; accepting it silently
    # elsewhere makes it a no-op flag, which is worse than an error.
    if args.assistant_prefill and args.backend != "vllm":
        parser.error(
            f"--assistant_prefill is implemented only on the vllm backend, but "
            f"--backend {args.backend} was requested. It would be silently ignored at "
            f"generation time while extraction still applies it, so the activations "
            f"would be read at a different prompt than produced the verdict."
        )

    loader = loader_for(args.dataset)()
    prompts = prompts_for(args.dataset)()

    print(f"Loading {args.n_problems} problems from {args.dataset} "
          f"(split={dataset_split(args.dataset)})…")
    examples = loader.load(args.n_problems, args.seed)
    print(f"Loaded {len(examples)} problems.")

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"

    # Refuse to resume into a directory built from a different run. Checked BEFORE the
    # backend is built so a mismatch fails in a second rather than after a model load.
    check_fingerprint(output_dir / "run_meta.json", run_fingerprint(args, examples),
                      raw_exists=raw_path.exists(), adopt_existing=args.adopt_existing,
                      digest_at=lambda k: problems_digest(examples, k))

    print(f"Initialising backend: {args.backend} / {args.model}")
    backend = build_backend(args)

    # Resume: skip problems already present in raw.jsonl from a prior (killed) run.
    done_idx: set[int] = set()
    if raw_path.exists():
        existing, repair = read_raw_records(raw_path)
        # Direct evidence of a shrink, independent of what run_meta.json recorded: an
        # idx beyond this run's sample is a problem the directory covers and this
        # invocation does not, and it would still be counted in the report.
        beyond = sorted(i for i in existing if i >= len(examples))
        if beyond:
            raise SystemExit(
                f"[FATAL] {raw_path} holds {len(beyond)} record(s) with idx >= "
                f"{len(examples)} (first={beyond[0]}), i.e. it covers a larger sample "
                f"than this invocation's {len(examples)} problems.\n"
                f"        Those records would still be counted in the report, so the "
                f"reported sample size would not be the one requested. Use a fresh "
                f"--output_dir for the smaller run."
            )
        failed = {i for i, r in existing.items() if r.get("error") is not None}
        done_idx = set(existing) - (failed if args.retry_failed else set())
        if repair["malformed"] or repair["duplicate"]:
            print(f"[WARN] {raw_path}: dropping {repair['malformed']} unparseable line(s) "
                  f"and {repair['duplicate']} duplicate idx row(s) of "
                  f"{repair['lines']}; rewriting the file.", file=sys.stderr)
        if failed:
            if args.retry_failed:
                print(f"[INFO] {len(failed)} record(s) hold a generation error (API "
                      f"failure, OOM, timeout); retrying them. Pass --no-retry-failed "
                      f"to keep them as permanent skips.", file=sys.stderr)
            else:
                print(f"[WARN] {len(failed)} record(s) hold a generation error and "
                      f"--no-retry-failed was given: they stay unlabelable, so the "
                      f"evaluated sample is {len(failed)} smaller than --n_problems and "
                      f"the loss is not random with respect to the problems.",
                      file=sys.stderr)
        # A retried record must not leave its old error row behind, or the file ends up
        # with two rows for one idx. Rewriting also truncates a corrupt partial tail, so
        # the next append cannot be concatenated onto it.
        if repair["malformed"] or repair["duplicate"] or (failed and args.retry_failed):
            rewrite_raw(raw_path, {i: r for i, r in existing.items() if i in done_idx})
        if done_idx:
            print(f"Resuming: {len(done_idx)} problems already saved, skipping them.",
                  flush=True)

    pending = [(idx, ex) for idx, ex in enumerate(examples) if idx not in done_idx]

    # Gate concurrency at the problem level so each problem runs its full
    # solve->judge->majority lifecycle within a bounded set. Without this,
    # all problems' calls share one global queue and the late-submitted judge
    # call starves behind every other problem, so nothing ever completes.
    problem_sem = asyncio.Semaphore(args.max_concurrent)

    async def gated(idx: int, example: dict) -> dict:
        async with problem_sem:
            return await run_problem_safe(
                idx=idx,
                example=example,
                gold_answer=loader.parse_gold(example),
                prompts=prompts,
                backend=backend,
                k_samples=args.k_samples,
                judge_max_tokens=args.judge_max_tokens,
                dataset=args.dataset,
                solve_max_tokens=args.solve_max_tokens,
            )

    tasks = [gated(idx, example) for idx, example in pending]

    # Append-and-fsync each result as it completes so a kill never loses progress.
    from tqdm import tqdm
    bar = tqdm(total=len(examples), initial=len(done_idx), desc="Evaluating")
    with open(raw_path, "a") as raw_f:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            raw_f.write(json.dumps(r) + "\n")
            raw_f.flush()
            os.fsync(raw_f.fileno())
            bar.update(1)
    bar.close()

    # Build the report from the full raw.jsonl (covers resumed + fresh results). Parsed
    # with the same defensive reader as the resume path: an unconditional json.loads
    # here crashes on a corrupt tail *after* the whole run has been paid for, and
    # counts a duplicated idx twice. Done before backend shutdown so an engine-teardown
    # error can never cost the report.
    final_records, repair = read_raw_records(raw_path)
    if repair["malformed"] or repair["duplicate"]:
        print(f"[WARN] {raw_path}: {repair['malformed']} unparseable line(s) and "
              f"{repair['duplicate']} duplicate idx row(s) excluded from the report.",
              file=sys.stderr)
    results = [relabel_result(final_records[i]) for i in sorted(final_records)]
    n_errors = sum(1 for r in results if r.get("error") is not None)
    print(f"\nRaw results → {raw_path} ({len(results)} problems"
          f"{f', {n_errors} generation error(s)' if n_errors else ''})", flush=True)

    report = generate_report(results)
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"Report → {report_path}")

    try:
        await backend.aclose()
    except Exception as exc:
        print(f"[WARN] backend.aclose() failed (ignored): {exc}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
