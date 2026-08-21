#!/usr/bin/env python3
"""Rich activation extraction for the reported flat-config CausalLM judges
(Qwen2.5 and Llama 3.1), parallel to the Gemma 3 extractor.

Differences from extract_hidden_rich_gemma3.py:
  - AutoModelForCausalLM (not AutoModelForImageTextToText) — for plain text LMs
    like Qwen2ForCausalLM / LlamaForCausalLM.
  - config may be FLAT (no .text_config); we fall back to the top-level config.
  - optional FP8->bf16 dequantize if the checkpoint is FP8 (bf16 loads as-is).

o_proj INPUT dim = n_heads * head_dim (== hidden_size for Qwen2/Llama; may differ
for archs like Gemma3) -> Z_head sized to that. Outputs match the other extractors:
  Z_head (N,nL,Dh) f16 | X_last/X_mean (N,nL+1,H) f32 (--store_resid) | y maj p_yes idx meta

  python extract_hidden_rich_causal.py --model Qwen/Qwen2.5-7B-Instruct \
    --input results_qwen25_7b/gsm8k/raw.jsonl --out results_qwen25_7b/hidden_rich.npz
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from dataset_registry import prompts_for
from extraction_common import (Dump, HeadTap, add_common_args, encode_judge_prompt,
                               head_dims, load_tokenizer, prepare_extraction,
                               read_records, yes_no_mass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"),
                    help="HF repo id of the judge checkpoint (falls back to $MODEL, "
                         "so run_full.sh / smoke_test.sh can drive it via the MODEL env var). "
                         "MUST match the model that generated --input.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--assistant_prefill", default="",
                    help="text appended after the generation prompt, matching the "
                         "--assistant_prefill used at generation time, "
                         "so activations are read at the same position that emitted "
                         "the verdict")
    add_common_args(ap)
    args = ap.parse_args()
    if not args.model:
        ap.error("--model is required (or set the MODEL env var)")
    # Fails fast (before the checkpoint load) if this extraction could not reproduce
    # the prompt that generation actually sent — including a prefill mismatch.
    gen_run = prepare_extraction(ap, args)

    tok = load_tokenizer(args.tokenizer or args.model)
    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()

    # bf16 by default; dequantize if the checkpoint ships FP8 weights (no native FP8 on Ampere).
    cfg0 = AutoConfig.from_pretrained(args.model)
    qc = getattr(cfg0, "quantization_config", None)
    qmethod = (qc.get("quant_method") if isinstance(qc, dict)
               else getattr(qc, "quant_method", None)) if qc else None
    load_kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    if qmethod == "fp8":
        from transformers import FineGrainedFP8Config
        torch.float8_e8m0fnu = getattr(torch, "float8_e8m0fnu", torch.float8_e4m3fn)
        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        print("  (FP8 -> bf16 dequantize)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval()

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
    dump.meta["assistant_prefill"] = args.assistant_prefill
    dump.resume(records)

    t1 = time.time()
    start = dump.filled
    for rec in records:
        if rec["idx"] in dump.done_ids:
            continue
        prompt = P.judge_prompt(rec["problem"], rec["phase1_solution"])
        ids = encode_judge_prompt(tok, prompt, args.assistant_prefill).to("cuda")
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
