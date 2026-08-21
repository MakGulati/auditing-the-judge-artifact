#!/usr/bin/env python3
"""Rich activation extraction for Gemma 3-family judges.

Architecture-specific behavior:
  - AutoModelForImageTextToText with a bf16-native checkpoint (no FP8 dequantize).
  - gemma3's o_proj INPUT dim = n_heads*head_dim, which is NOT hidden_size
    (gemma-3-12b: 16 heads * 256 head_dim = 4096 vs hidden 3840) -> Z_head sized to Dh.

`--model` (or $MODEL) selects the checkpoint; layer/head geometry is read from its
config and recorded in the dump's `meta`, so the probe can verify that --head_dim
matches the model that actually produced the activations.

Outputs:
  Z_head (N,nL,Dh)  f16 o_proj input (concat per-head attn out) -> reshape nH x head_dim
  X_last/X_mean (N,nL+1,H) f32 residual stream  (--store_resid; off by default)
  y, maj, p_yes, idx, meta

Run with the GPU environment documented in requirements-vllm.txt.
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText
from dataset_registry import prompts_for
from extraction_common import (Dump, HeadTap, add_common_args, encode_judge_prompt,
                               head_dims, load_tokenizer, prepare_extraction,
                               read_records, yes_no_mass)

DEFAULT_MODEL = "google/gemma-3-12b-it"


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
    print(f"loading {args.model} (bf16)...", flush=True)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
    ).eval()

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
            hs = out.hidden_states
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
