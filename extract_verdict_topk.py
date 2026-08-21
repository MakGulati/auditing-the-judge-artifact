#!/usr/bin/env python3
"""Dump the judge's verdict-position top-K next-token distribution.

`hidden_rich.npz` stores `p_yes` already normalised, so the mass that the shipped
prefix rule binned as NO cannot be recovered from it: whether the judge put 0.3 on
" No" or on " not" leaves the same number behind. This script re-runs the same
forward pass and saves the raw (token id, probability) pairs, which is what makes
the binning rule auditable after the fact.

It is a strict superset of the information `yes_no_mass` consumes — same prompt,
same position, same tokenizer path — so `p_yes_shipped` here reproduces the
`p_yes` already in the activation dump, bit for bit, and
`filtering/pyes_rule_compare.py` checks exactly that before it reports anything.

No activations are stored: the output is ~1 MB per thousand records, against 2.8 GB
for the matching `hidden_rich.npz`. Runtime is one forward pass per record, the
same cost as the original extraction.

  python extract_verdict_topk.py --model google/gemma-3-12b-it \
    --input results_gemma3_test/gsm8k/raw.jsonl --out results_gemma3_test/verdict_topk.npz
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoConfig

from dataset_registry import prompts_for
from extraction_common import (encode_judge_prompt, load_tokenizer,
                               prepare_extraction, read_records, record_digest,
                               yes_no_mass)
from metrics import provenance
from metrics.correctness import DEFAULT_DATASET
from metrics.verdict_tokens import RULES, vocab_classes

DEFAULT_TOPK = 64


def load_model(model_id: str, arch: str):
    """Load the judge for a forward pass, mirroring the activation extractors.

    `arch=auto` follows the same split those scripts encode by filename: gemma3-style
    multimodal wrappers need AutoModelForImageTextToText, flat text configs need
    AutoModelForCausalLM. Loading a gemma3 checkpoint as a CausalLM does not fail
    loudly — it can produce a model whose logits are read at the wrong position — so
    the choice is recorded in the dump's meta.
    """
    cfg = AutoConfig.from_pretrained(model_id)
    if arch == "auto":
        names = " ".join(getattr(cfg, "architectures", None) or [])
        arch = ("image_text" if ("ImageTextToText" in names
                                 or "ForConditionalGeneration" in names
                                 or hasattr(cfg, "text_config")) else "causal")
    load_kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    qc = getattr(cfg, "quantization_config", None)
    qmethod = (qc.get("quant_method") if isinstance(qc, dict)
               else getattr(qc, "quant_method", None)) if qc else None
    if qmethod == "fp8":
        from transformers import FineGrainedFP8Config
        torch.float8_e8m0fnu = getattr(torch, "float8_e8m0fnu", torch.float8_e4m3fn)
        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        print("  (FP8 -> bf16 dequantize)", flush=True)
    if arch == "image_text":
        from transformers import AutoModelForImageTextToText as Auto
    else:
        from transformers import AutoModelForCausalLM as Auto
    return Auto.from_pretrained(model_id, **load_kwargs).eval(), arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"),
                    help="HF repo id of the judge checkpoint (falls back to $MODEL). "
                         "MUST match the model that generated --input, and must match "
                         "the model recorded in the matching hidden_rich.npz.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                    help=f"how many top tokens to store per record (default {DEFAULT_TOPK}). "
                         "Must be >= 40 so the shipped rule's window is fully contained; "
                         "storing more lets the analysis re-cut at an API's limit (e.g. 20) "
                         "without another GPU pass.")
    ap.add_argument("--arch", choices=("auto", "causal", "image_text"), default="auto")
    ap.add_argument("--assistant_prefill", default="",
                    help="must match the --assistant_prefill used at generation time; "
                         "otherwise the distribution is read at a different position "
                         "than produced the verdict")
    # add_common_args is not reused here: it also advertises --store_resid, which only
    # the activation extractors implement, and a flag that silently does nothing is
    # worse than an absent one.
    ap.add_argument("--dataset", default=os.environ.get("DATASET") or DEFAULT_DATASET,
                    help="dataset that produced --input (falls back to $DATASET, then "
                         f"{DEFAULT_DATASET!r}). Selects the judge prompt, which MUST be "
                         "the one generation used.")
    ap.add_argument("--tokenizer", default=None,
                    help="HF repo id of the tokenizer, when generation ran with a "
                         "--tokenizer other than --model. Defaults to the generation "
                         "run's recorded tokenizer, else --model.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=1500)
    args = ap.parse_args()
    if not args.model:
        ap.error("--model is required (or set the MODEL env var)")
    if args.topk < 40:
        # yes_no_mass reads the top 40. A smaller window here would make p_yes_shipped
        # un-reproducible and silently understate every mass this script reports.
        ap.error(f"--topk {args.topk} < 40, the window the shipped rule reads; the "
                 f"comparison would be against a truncated version of that rule")

    # Binds this dump to the generation run: a tokenizer, prompt format or prefill
    # that extraction cannot reproduce reads the distribution at a different position.
    gen_run = prepare_extraction(ap, args)

    tok = load_tokenizer(args.tokenizer or args.model)
    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    model, arch = load_model(args.model, args.arch)
    print(f"model ready {time.time() - t0:.1f}s (arch={arch})", flush=True)

    print("building vocabulary class lists ...", flush=True)
    classes = {r: vocab_classes(tok, r) for r in RULES}
    for r in RULES:
        print(f"  {r}: {len(classes[r][0])} YES tokens, {len(classes[r][1])} NO tokens",
              flush=True)
    idx_t = {r: (torch.tensor(classes[r][0], device="cuda", dtype=torch.long),
                 torch.tensor(classes[r][1], device="cuda", dtype=torch.long))
             for r in RULES}

    P = prompts_for(args.dataset)()
    records, _ = read_records(args.input, args.limit, args.dataset)
    N = len(records)
    print(f"records: {N}", flush=True)

    K = args.topk
    store = {
        "idx": np.zeros(N, np.int32),
        "top_ids": np.zeros((N, K), np.int32),
        "top_probs": np.zeros((N, K), np.float32),
        "p_yes_shipped": np.zeros(N, np.float32),
        # Content fingerprint of the record each row was read from: `idx` is a position
        # in a shuffled subset, so it cannot prove a resumed dump belongs to this file.
        "rec_sha": np.zeros(N, "<U16"),
    }
    for r in RULES:
        for side in ("yes", "no"):
            store[f"mass_{side}_{r}_full"] = np.zeros(N, np.float32)

    meta = {
        "model": args.model, "arch": arch, "dataset": args.dataset, "topk": K,
        "assistant_prefill": args.assistant_prefill, "input": os.path.abspath(args.input),
        "vocab_size": len(tok),
        "n_vocab_yes": {r: len(classes[r][0]) for r in RULES},
        "n_vocab_no": {r: len(classes[r][1]) for r in RULES},
        "gen_run": gen_run,
        # Which rule produced the rec_sha values below, so a later definition change is
        # reported as a definition change rather than as a record mismatch.
        "digest_version": provenance.DIGEST_VERSION,
    }

    def save(n):
        seen = np.unique(store["top_ids"][:n])
        out = {k: v[:n] for k, v in store.items()}
        out["seen_ids"] = seen.astype(np.int32)
        # Decoded forms of every id that ever entered a top-K window, so the analysis
        # (and anyone auditing it) needs no tokenizer and no GPU. Stored as a unicode
        # array rather than object dtype so loading needs no allow_pickle.
        out["seen_texts"] = np.array([tok.decode([int(i)]) for i in seen])
        out["meta"] = json.dumps(meta)
        tmp = args.out + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, args.out)

    # --- resume -----------------------------------------------------------------
    # A full train split is thousands of forward passes; a resume that silently
    # accepted a dump written under different settings would splice two experiments
    # into one file, so every field that changes the distribution is checked.
    done = set()
    n = 0
    if os.path.exists(args.out):
        prev = np.load(args.out)
        pmeta = json.loads(str(prev["meta"]))
        differing = [k for k in ("model", "arch", "dataset", "topk", "assistant_prefill")
                     if pmeta.get(k) != meta[k]]
        if differing:
            raise SystemExit(
                f"[FATAL] {args.out} was written with different settings "
                f"({', '.join(f'{k}={pmeta.get(k)!r} vs {meta[k]!r}' for k in differing)}). "
                f"Resuming would mix two runs in one file; delete it or pass a new --out.")
        k = len(prev["idx"])
        if k > N:
            raise SystemExit(f"[FATAL] {args.out} holds {k} records but --input yields "
                             f"only {N}; it was not written from this file.")
        for key in store:
            if key in prev.files:
                store[key][:k] = prev[key]
        by_idx = {r["idx"]: r for r in records}
        resumed = prev["idx"].tolist()
        unknown = [i for i in resumed if i not in by_idx]
        if unknown:
            raise SystemExit(
                f"[FATAL] {args.out} references {len(unknown)} idx value(s) absent from "
                f"--input (first={unknown[0]}); it was written from a different file.")
        prev_digest_version = pmeta.get("digest_version", 1)
        if "rec_sha" not in prev.files:
            print(f"[WARN] {args.out} predates per-record fingerprints; its rows are "
                  f"matched by idx alone, which cannot detect a different raw file. "
                  f"Re-extract to authenticate it.", flush=True)
        elif prev_digest_version != provenance.DIGEST_VERSION:
            # Every stored digest would mismatch, and reporting that as "different
            # records" would send you hunting for a data problem that does not exist.
            raise SystemExit(
                f"[FATAL] {args.out} stores rec_sha under digest_version="
                f"{prev_digest_version}; this build computes v{provenance.DIGEST_VERSION} "
                f"(v2 added judge_verdict). The stored fingerprints cannot be compared "
                f"against the current rule, so resuming would extend the file on an "
                f"unverified prefix.\n"
                f"        This is a definition change, not a data mismatch — re-extract "
                f"to a new --out.")
        else:
            bad = [i for j, i in enumerate(resumed)
                   if str(prev["rec_sha"][j]) != record_digest(by_idx[i])]
            if bad:
                raise SystemExit(
                    f"[FATAL] {args.out} was written from different records than --input "
                    f"holds: {len(bad)} of {k} resumed rows disagree (first idx={bad[0]}). "
                    f"Two runs can share every index while holding different problems, so "
                    f"the stored distributions were read at other prompts. Use a new --out.")
        done = set(resumed)
        n = k
        print(f"resuming: {n} records already in {args.out}", flush=True)

    t1 = time.time()
    start = n
    for rec in records:
        if rec["idx"] in done:
            continue
        prompt = P.judge_prompt(rec["problem"], rec["phase1_solution"])
        ids = encode_judge_prompt(tok, prompt, args.assistant_prefill).to("cuda")
        with torch.no_grad():
            out = model(input_ids=ids)
        probs = torch.softmax(out.logits[0, -1].float(), -1)

        topv, topi = torch.topk(probs, K)
        store["idx"][n] = rec["idx"]
        store["rec_sha"][n] = record_digest(rec)
        store["top_ids"][n] = topi.cpu().numpy()
        store["top_probs"][n] = topv.cpu().numpy()
        # The shipped statistic, computed by the shipped function on the same row.
        store["p_yes_shipped"][n] = yes_no_mass(probs, tok)
        # Full-vocabulary masses: what each rule would count with no top-K truncation.
        for r in RULES:
            yi, ni = idx_t[r]
            store[f"mass_yes_{r}_full"][n] = float(probs[yi].sum())
            store[f"mass_no_{r}_full"][n] = float(probs[ni].sum())
        n += 1

        if n % 200 == 0:
            rate = (n - start) / max(time.time() - t1, 1e-9)
            print(f"  {n}/{N}  {rate:.1f}/s  eta {(N - n) / max(rate, 1e-9) / 60:.1f}m",
                  flush=True)
        if n % args.save_every == 0:
            save(n)
    save(n)
    print(f"DONE: {n} records -> {args.out} ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
