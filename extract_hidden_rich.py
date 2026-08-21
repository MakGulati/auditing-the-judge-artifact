#!/usr/bin/env python3
"""Richer activation extraction for the self-judge probe (mistral3 / Ministral).

One forward pass per judge record, capturing enough of the internal state to test
attention-head probing without re-running the model:

  Z_head : (N, nL, Dh)   float16  o_proj INPUT at last token = concat of per-head
                                   attention outputs (reshape to n_heads x head_dim
                                   for attention-head probing / ITI direction-finding)
  X_last : (N, nL+1, H)  float32  last-token residual per layer   (--store_resid)
  X_mean : (N, nL+1, H)  float32  mean-over-tokens residual       (--store_resid)
  y, maj, p_yes, idx, meta

The tokenizer is loaded exactly the way `pipeline.backend.TransformersBackend` loads
it (AutoTokenizer, via `extraction_common.load_tokenizer`), so the judge prompt is
templated identically at generation and at extraction time. There is no fallback to a
bare `PreTrainedTokenizerFast(tokenizer_file=...)`: it has no special tokens, so a
chat template referencing bos/eos renders them empty and the activations would be read
at a different prompt than the one that produced the recorded verdict. If no properly
configured tokenizer can be loaded, extraction stops.

Run with torch + transformers on a CUDA GPU (see requirements.txt).
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch

torch.float8_e8m0fnu = getattr(torch, "float8_e8m0fnu", torch.float8_e4m3fn)

from transformers import AutoConfig, AutoModelForImageTextToText, FineGrainedFP8Config
from dataset_registry import prompts_for
from extraction_common import (Dump, HeadTap, add_common_args, encode_judge_prompt,
                               head_dims, load_tokenizer, prepare_extraction,
                               read_records, yes_no_mass)

DEFAULT_MODEL = "mistralai/Ministral-3-8B-Instruct-2512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL") or DEFAULT_MODEL,
                    help="HF repo id of the judge checkpoint (falls back to $MODEL, then "
                         f"{DEFAULT_MODEL}). MUST match the model that generated --input.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    add_common_args(ap)
    args = ap.parse_args()
    # Fails fast (before the checkpoint load) if this extraction could not reproduce
    # the prompt that generation actually sent.
    gen_run = prepare_extraction(ap, args)

    tok = load_tokenizer(args.tokenizer or args.model)
    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()

    cfg0 = AutoConfig.from_pretrained(args.model)
    qc = getattr(cfg0, "quantization_config", None)
    qmethod = (qc.get("quant_method") if isinstance(qc, dict)
               else getattr(qc, "quant_method", None)) if qc else None
    load_kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    if qmethod == "fp8":
        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        print("  (FP8 -> bf16 dequantize)", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs).eval()

    nL, H, nH, Dh = head_dims(model.config)
    print(f"model ready {time.time()-t0:.1f}s  layers={nL} hidden={H} heads={nH} "
          f"head_dim={Dh//nH} o_proj_in={Dh}", flush=True)

    tap = HeadTap(model, nL)
    P = prompts_for(args.dataset)()
    records, _ = read_records(args.input, args.limit, args.dataset)
    N = len(records)
    print(f"records: {N}", flush=True)

    dump = Dump(args.out, N, nL=nL, H=H, nH=nH, Dh=Dh,
                model=args.model, store_resid=args.store_resid,
                dataset=args.dataset, gen_run=gen_run)
    dump.resume(records)

    t1 = time.time()
    start = dump.filled
    for rec in records:
        if rec["idx"] in dump.done_ids:
            continue
        prompt = P.judge_prompt(rec["problem"], rec["phase1_solution"])
        ids = encode_judge_prompt(tok, prompt).to("cuda")
        tap.reset()
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=dump.store_resid)
        z_head = tap.stack(rec["idx"])
        x_last = x_mean = None
        if dump.store_resid:
            hs = out.hidden_states  # tuple len nL+1, each (1, seq, H)
            x_last = np.stack([hs[L][0, -1, :].float().cpu().numpy() for L in range(nL + 1)])
            x_mean = np.stack([hs[L][0].float().mean(0).cpu().numpy() for L in range(nL + 1)])
        probs = torch.softmax(out.logits[0, -1].float(), -1)
        dump.add(rec, z_head, yes_no_mass(probs, tok), x_last, x_mean)

        if dump.filled % 200 == 0:
            rate = (dump.filled - start) / max(time.time() - t1, 1e-9)
            print(f"  {dump.filled}/{N}  {rate:.1f}/s  "
                  f"eta {(N-dump.filled)/max(rate,1e-9)/60:.1f}m", flush=True)
        if dump.filled % args.save_every == 0:
            dump.flush()
    dump.flush()
    print(f"DONE: extracted {dump.filled} -> {args.out} ({time.time()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
