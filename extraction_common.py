"""Shared machinery for the `extract_hidden_rich*.py` activation extractors.

The three extractors differ only in how they load a model (mistral3 FP8 dequantize /
gemma3 bf16 / flat-config CausalLM). Everything after that — hooking o_proj, sizing
the dump, resuming, labelling, saving — is identical and lives here so a fix lands
once instead of three times.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from metrics.correctness import (DEFAULT_DATASET, is_labelable, label_correctness,
                                 label_policy_for, record_dataset)
# Re-exported: the extractors and the probe scripts must fingerprint records the same
# way, and the probe scripts run in an environment without torch/transformers.
# Imported as a module, not by name: `record_digest` reads DIGEST_VERSION at call
# time, so binding a snapshot here could leave the stamp and the comparison
# disagreeing — which surfaces as 'these are different records' rather than as
# the version skew it actually is.
from metrics import provenance
from metrics.provenance import read_run_meta, record_digest  # noqa: F401

# Matches a decoder layer's output projection. Deliberately anchored so a vision
# tower's `vision_tower.transformer.layers.N.self_attn.o_proj` is also matched and
# therefore caught by the hook-count assertion instead of silently overwriting a
# text layer's buffer.
_O_PROJ = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$")

# fp16 max; Z_head is stored fp16 to halve the dump, so anything at or above this
# would silently become ±inf and poison the LDA covariance in probe_models.lda_auc.
_FP16_MAX = 65504.0


def add_common_args(ap) -> None:
    ap.add_argument("--dataset", default=os.environ.get("DATASET") or DEFAULT_DATASET,
                    help="dataset that produced --input (falls back to $DATASET, then "
                         f"{DEFAULT_DATASET!r}). Selects the judge prompt, which MUST "
                         "be the one generation used.")
    ap.add_argument("--tokenizer", default=None,
                    help="HF repo id of the tokenizer, when generation ran with a "
                         "--tokenizer other than --model. Defaults to the value "
                         "recorded in the generation run's run_meta.json, else --model. "
                         "Tokenization decides where the last-token activation is read, "
                         "so it must be the tokenizer generation templated with.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=1500)
    ap.add_argument("--store_resid", action="store_true",
                    help="also store X_last/X_mean (per-layer residual stream). No "
                         "current analysis reads them and they roughly triple the dump, "
                         "so they are off by default.")


# Generation settings that decide the token sequence the judge saw, and therefore the
# position every activation is read at. Extraction must reproduce each of them.
_PROMPT_IDENTITY_KEYS = ("model", "tokenizer", "mistral_format", "assistant_prefill",
                         "dataset", "judge_max_tokens", "backend", "split", "seed",
                         "problems_sha256", "k_samples")

# Of those, the ones that must be identical across every resume of a single dump: they
# change the token sequence itself. `problems_sha256` deliberately is NOT here — it
# moves when a generation run is extended, which is a legitimate reason to re-extract.
_PROMPT_STRICT_KEYS = ("model", "tokenizer", "mistral_format", "assistant_prefill",
                       "dataset", "judge_prompt_sha256", "extraction_tokenizer",
                       "extraction_assistant_prefill")


def judge_prompt_digest(dataset: str) -> str:
    """Fingerprint of the judge template itself, with the fields left as placeholders.

    ``dataset`` already pins which prompt *class* is used, but not its wording. Editing
    the template between two extraction runs that share one --out would put activations
    read at two different prompts into a single dump, with nothing recording it.
    """
    from dataset_registry import prompts_for

    rendered = prompts_for(dataset)().judge_prompt("\0PROBLEM\0", "\0SOLUTION\0")
    return hashlib.sha256(rendered.encode()).hexdigest()[:16]


def prepare_extraction(ap, args) -> dict:
    """Validate the shared args and bind this extraction to the generation run.

    Returns the generation identity to stamp into the dump. Called before the model is
    loaded so a mismatch costs a second rather than a checkpoint load.

    The mismatches this catches are all silent: generation may have used a different
    tokenizer (``--tokenizer``), a different tokenization path entirely
    (``--mistral_format``, which templates with ``mistral_common`` and cannot be
    reproduced by ``AutoTokenizer``), or an assistant prefill. Every one of them moves
    the last token, which is exactly where the activation is read.
    """
    if args.limit < 0:
        ap.error(f"--limit must be >= 0 (got {args.limit})")
    if args.save_every < 1:
        ap.error(f"--save_every must be >= 1 (got {args.save_every})")

    prefill = getattr(args, "assistant_prefill", "")
    meta = read_run_meta(args.input)
    if meta is None:
        print(f"[WARN] no run_meta.json beside {args.input}; nothing verifies that this "
              f"extraction reproduces generation's tokenizer, prefill or prompt format. "
              f"Re-generate (or copy the run_meta.json) to enable the check.", flush=True)
        gen = {}
    else:
        gen = {k: meta.get(k) for k in _PROMPT_IDENTITY_KEYS if k in meta}
        problems = []
        if meta.get("dataset", DEFAULT_DATASET) != args.dataset:
            problems.append(
                f"generation ran --dataset {meta.get('dataset')!r}, extraction "
                f"--dataset {args.dataset!r}: a different judge prompt")
        if meta.get("model") and meta["model"] != args.model:
            problems.append(
                f"generation ran --model {meta['model']!r}, extraction --model "
                f"{args.model!r}: different weights and possibly a different template")
        if meta.get("mistral_format"):
            problems.append(
                "generation ran --mistral-format, which templates the prompt with "
                "mistral_common from params.json/tekken.json. No extractor can "
                "reproduce that: they all template with AutoTokenizer, which tokenizes "
                "the same conversation differently")
        gen_tok = meta.get("tokenizer")
        if gen_tok and args.tokenizer and gen_tok != args.tokenizer:
            problems.append(
                f"generation ran --tokenizer {gen_tok!r}, extraction was given "
                f"--tokenizer {args.tokenizer!r}")
        if (meta.get("assistant_prefill") or "") != (prefill or ""):
            problems.append(
                f"generation ran --assistant_prefill "
                f"{meta.get('assistant_prefill', '')!r}, extraction "
                f"{prefill!r}" + ("" if hasattr(args, "assistant_prefill") else
                                  f" (this extractor cannot apply one — use "
                                  f"extract_hidden_rich_causal.py)"))
        if problems:
            raise SystemExit(
                f"[FATAL] this extraction would not reproduce the prompt that produced "
                f"the verdicts in {args.input}:\n"
                + "".join(f"  - {p}\n" for p in problems) +
                "        Every one of these moves the token the activation is read at, "
                "with no visible\n        symptom: the dump fills and the probe trains "
                "on activations from the wrong position."
            )
        if gen_tok and not args.tokenizer:
            print(f"[INFO] generation used --tokenizer {gen_tok!r}; extracting with it "
                  f"rather than with --model.", flush=True)
            args.tokenizer = gen_tok
    gen["judge_prompt_sha256"] = judge_prompt_digest(args.dataset)
    gen["extraction_tokenizer"] = args.tokenizer or args.model
    gen["extraction_assistant_prefill"] = prefill
    return gen


def _validate_tokenizer(tok, source: str):
    """Reject a tokenizer that would render the chat template differently than generation.

    The failure this guards against is silent: a chat template that interpolates
    ``{{ bos_token }}`` renders it as the empty string when the tokenizer carries no
    such token, so the prompt loses a leading token and every activation is read one
    position off from the sequence that produced the recorded verdict. Only the tokens
    the template actually references are required, so models that legitimately have no
    BOS are not rejected.
    """
    template = getattr(tok, "chat_template", None)
    if not template:
        raise ValueError(
            f"tokenizer from {source} has no chat_template, so the judge prompt cannot "
            f"be templated the way generation templated it")
    for name in ("bos_token", "eos_token", "pad_token", "unk_token", "sep_token"):
        if name in template and getattr(tok, name, None) is None:
            raise ValueError(
                f"tokenizer from {source} has no {name}, but its chat_template "
                f"interpolates {{{{ {name} }}}} — it would render as an empty string and "
                f"shift every token position relative to generation")
    return tok


def find_snapshots(model: str) -> list[str]:
    """Local snapshots for `model` that could carry a full tokenizer, newest first.

    Requires tokenizer_config.json, not just tokenizer.json: the config is what carries
    bos/eos/pad and the chat template, and a snapshot with only the raw tokenizer.json
    (e.g. left by a mistral-format vLLM load) cannot reproduce generation's prompt.
    Returns every candidate rather than just the newest, because the most recently
    fetched revision can be the partial one while an older revision is complete.
    """
    base = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots"
    )
    if not os.path.isdir(base):
        raise FileNotFoundError(f"no local snapshot directory at {base}")
    snaps = [os.path.join(base, d) for d in os.listdir(base)]
    full = [s for s in snaps
            if os.path.exists(os.path.join(s, "tokenizer_config.json"))
            and os.path.exists(os.path.join(s, "tokenizer.json"))]
    if not full:
        raise FileNotFoundError(
            f"no snapshot under {base} has both tokenizer.json and tokenizer_config.json")
    # Directory names are content hashes, so alphabetical order is arbitrary; order by
    # recency of fetch.
    return sorted(full, key=os.path.getmtime, reverse=True)


def find_snapshot(model: str) -> str:
    """Newest fully-configured local snapshot for `model`."""
    return find_snapshots(model)[0]


def load_tokenizer(model: str):
    """Load a fully configured tokenizer — the same object generation templated with.

    Both generation backends use ``AutoTokenizer``, which reads tokenizer_config.json
    for bos/eos/pad and the chat template. There is deliberately **no** degraded
    fallback to ``PreTrainedTokenizerFast(tokenizer_file=...)``: that reconstructs the
    vocabulary without any of the special-token configuration, which silently
    reintroduces the prompt mismatch this whole path exists to prevent. Either a
    properly configured tokenizer loads, or extraction stops.

    The local-snapshot attempt is a *path* fallback, not a configuration fallback — it
    still goes through AutoTokenizer and still carries the full config.
    """
    attempts = []
    try:
        return _validate_tokenizer(AutoTokenizer.from_pretrained(model), repr(model))
    except Exception as exc:  # noqa: BLE001 — recorded and re-raised below
        attempts.append(f"hub id {model!r}: {type(exc).__name__}: {exc}")
    try:
        snapshots = find_snapshots(model)
    except FileNotFoundError as exc:
        attempts.append(f"local snapshot: {exc}")
    else:
        for snap in snapshots:
            try:
                return _validate_tokenizer(AutoTokenizer.from_pretrained(snap), snap)
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"local snapshot .../{os.path.basename(snap)[:12]}: "
                                f"{type(exc).__name__}: {exc}")

    import transformers
    raise SystemExit(
        "[FATAL] could not load a fully configured tokenizer for "
        f"{model!r}; tried:\n" + "".join(f"  - {a}\n" for a in attempts) +
        "\n        Extraction will NOT fall back to a bare PreTrainedTokenizerFast: it\n"
        "        drops the special tokens and chat template, so activations would be\n"
        "        read at a different prompt than the one that produced the verdict.\n"
        f"\n        Installed transformers is {transformers.__version__}. A "
        f"'does not exist or is not currently imported'\n"
        "        or KeyError above usually means this env is too old for the "
        "checkpoint\n"
        "        (Ministral/mistral3 needs transformers==5.12.1 — see requirements.txt);\n"
        "        run extraction in the env that matches the model, not the vLLM env.\n"
        "        Otherwise fetch the full checkpoint (tokenizer_config.json + chat "
        "template)."
    )


def encode_judge_prompt(tok, prompt: str, assistant_prefill: str = ""):
    """Tokenize the judge prompt exactly the way generation did.

    `VLLMBackend.chat_complete` builds the prompt as
    ``apply_chat_template(..., tokenize=False) + prefill`` and hands vLLM that single
    string, so the tokenizer sees the join. Concatenating two independently tokenized
    ID sequences instead — ``cat(template_ids, tok(prefill).ids)`` — can differ at the
    boundary, because a BPE merge spanning the join (a template ending in "\\n" and a
    prefill starting with "\\n") exists in one path and not the other. Activations must
    be read at the same token sequence that produced the recorded verdict, so template
    to text first and tokenize once.
    """
    if not assistant_prefill:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"]
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    ) + assistant_prefill
    # apply_chat_template already emitted whatever BOS the template calls for; letting
    # the plain __call__ add another would shift every position by one.
    return tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]


def yes_no_mass(probs_row, tok) -> float:
    """Judge's normalised P(YES) over the YES/NO token mass at the verdict position."""
    topv, topi = torch.topk(probs_row, 40)
    my = mn = 0.0
    for p, i in zip(topv.tolist(), topi.tolist()):
        t = tok.decode([i]).strip().strip('":*').upper()
        if not t:
            continue
        if t.startswith("YES") or t in ("Y", "YE"):
            my += p
        elif t.startswith("NO") or t == "N":
            mn += p
    d = my + mn
    return my / d if d > 0 else 0.5


def head_dims(cfg) -> tuple[int, int, int, int]:
    """(n_layers, hidden_size, n_heads, o_proj_input_dim) from a text config.

    o_proj's input is n_heads * head_dim, which is NOT hidden_size on every arch
    (gemma-3-12b: 16*256=4096 vs hidden 3840).
    """
    tc = getattr(cfg, "text_config", cfg)
    nL = tc.num_hidden_layers
    H = tc.hidden_size
    nH = tc.num_attention_heads
    head_dim = getattr(tc, "head_dim", None) or (H // nH)
    return nL, H, nH, nH * head_dim


class HeadTap:
    """Captures each layer's o_proj input (concatenated per-head attention output).

    The buffer is cleared before every forward pass and checked after, so a layer
    whose hook did not fire raises instead of silently persisting the previous
    record's activations into this record's row.
    """

    def __init__(self, model, nL: int) -> None:
        self._nL = nL
        self.buf: list[Any] = [None] * nL
        n_hooked = 0
        for name, mod in model.named_modules():
            m = _O_PROJ.search(name)
            if not m:
                continue
            li = int(m.group(1))
            if not (0 <= li < nL):
                continue

            def make_hook(li):
                def hook(module, inp, out):
                    self.buf[li] = inp[0][0, -1, :].detach().float().cpu().numpy()
                return hook

            mod.register_forward_hook(make_hook(li))
            n_hooked += 1
        print(f"hooked {n_hooked} o_proj modules (expected {nL})", flush=True)
        if n_hooked != nL:
            raise RuntimeError(
                f"o_proj hook count mismatch: hooked {n_hooked}, expected {nL}. "
                f"A non-text tower probably also matches '{_O_PROJ.pattern}'; its "
                f"layer indices would collide with the decoder's."
            )

    def reset(self) -> None:
        self.buf = [None] * self._nL

    def stack(self, idx: int) -> np.ndarray:
        missing = [i for i, v in enumerate(self.buf) if v is None]
        if missing:
            raise RuntimeError(
                f"record idx={idx}: o_proj hook did not fire for layer(s) {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''}. Refusing to write stale activations."
            )
        z = np.stack(self.buf)
        peak = float(np.abs(z).max())
        if not np.isfinite(peak) or peak >= _FP16_MAX:
            raise RuntimeError(
                f"record idx={idx}: |o_proj input| peaks at {peak:.4g}, at or beyond the "
                f"fp16 max ({_FP16_MAX:g}). Storing it as fp16 would produce inf and "
                f"corrupt the probe's covariance. Store Z_head as fp32 for this model."
            )
        return z.astype(np.float16)


def check_records_dataset(records: list[dict], dataset: str, source: str) -> None:
    """Refuse to extract with a judge prompt other than the one generation used.

    Extraction re-templates the judge prompt from scratch, so picking the wrong prompt
    class reads activations at a *different token sequence* than the one that produced
    the recorded verdict — a mismatch with no visible symptom: the dump fills, the
    probe trains, and the number it reports is meaningless. Records written before the
    ``dataset`` field existed can only be GSM8K, so they are accepted as such and
    rejected against anything else.
    """
    seen = sorted({record_dataset(r) for r in records})
    if not seen or seen == [dataset]:
        return
    raise SystemExit(
        f"[FATAL] --dataset {dataset!r} does not match {source}, whose records report "
        f"dataset {', '.join(repr(s) for s in seen)}.\n"
        f"        Extraction would template the judge prompt for {dataset!r} while the "
        f"verdicts in that file were produced by a different prompt, so every "
        f"activation would be read at the wrong token sequence.\n"
        f"        Re-run with the matching --dataset (records with no dataset field "
        f"count as {DEFAULT_DATASET!r})."
    )


def read_records(path: str, limit: int = 0,
                 dataset: str | None = None) -> tuple[list[dict], dict[str, int]]:
    """Load raw.jsonl, dropping records that cannot carry a correctness label.

    Passing ``dataset`` also verifies the file was generated by that dataset. The check
    runs before the labelability filter so a mismatched file reports the mismatch
    rather than "0 labelable records".
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = records[:limit]
    if dataset is not None:
        check_records_dataset(records, dataset, path)
    n_all = len(records)
    kept = [r for r in records if is_labelable(r)]
    dropped = [r for r in records if not is_labelable(r)]
    n_err = sum(1 for r in dropped if r.get("error") is not None)
    n_trunc = sum(1 for r in dropped
                  if r.get("error") is None and r.get("truncated") is True)
    stats = {
        "loaded": n_all,
        "kept": len(kept),
        "errors": n_err,
        "truncated": n_trunc,
        "no_gold": len(dropped) - n_err - n_trunc,
    }
    if dropped:
        print(f"skipping {len(dropped)} unlabelable record(s): "
              f"{stats['errors']} generation error(s), "
              f"{stats['truncated']} budget-truncated, "
              f"{stats['no_gold']} unparseable gold answer(s)", flush=True)
    return kept, stats


class Dump:
    """Pre-allocated activation dump with resume, metadata and atomic saves."""

    def __init__(self, out_path: str, n: int, *, nL: int, H: int, nH: int, Dh: int,
                 model: str, store_resid: bool,
                 dataset: str = DEFAULT_DATASET, gen_run: dict | None = None) -> None:
        self.path = out_path
        self.store_resid = store_resid
        # `dataset` is an identity field, not advisory: two datasets' activations in
        # one dump would be labelled under two different equivalence policies and
        # judged by two different prompts. _check_meta makes a mismatch fatal.
        #
        # `gen_run` carries the generation fingerprint (which model/tokenizer/prefill/
        # problem set produced the verdicts) into the dump, so a downstream consumer can
        # check that two dumps describe the same experiment without re-reading raw.jsonl.
        self.meta = {
            "model": model, "n_layers": nL, "hidden_size": H, "n_heads": nH,
            "o_proj_in": Dh, "head_dim": Dh // nH,
            "label_policy": label_policy_for(dataset),
            "dataset": dataset, "store_resid": bool(store_resid),
            "gen_run": dict(gen_run or {}),
            # Which rule produced the rec_sha column; see Dump.resume.
            "digest_version": provenance.DIGEST_VERSION,
        }
        # The subset of the generation identity that decides the token sequence, split
        # out because it is the part that must not change *between resumes of one dump*.
        # The rest of gen_run (the problem-set digest above all) legitimately moves when
        # a generation run is extended, and blocking a re-extract on that would be noise.
        self.meta["prompt_identity"] = {k: self.meta["gen_run"].get(k)
                                        for k in _PROMPT_STRICT_KEYS}
        self.Z_head = np.zeros((n, nL, Dh), dtype=np.float16)
        self.X_last = np.zeros((n, nL + 1, H), dtype=np.float32) if store_resid else None
        self.X_mean = np.zeros((n, nL + 1, H), dtype=np.float32) if store_resid else None
        self.y = np.zeros(n, dtype=np.int8)
        self.maj = np.zeros(n, dtype=np.int8)
        self.p_yes = np.zeros(n, dtype=np.float32)
        self.idx = np.zeros(n, dtype=np.int32)
        # Per-row content fingerprint of the record the row was extracted from. `idx`
        # alone cannot prove a resumed dump belongs to the raw file it is being resumed
        # against; this can.
        self.rec_sha = np.zeros(n, dtype="<U16")
        self.filled = 0
        self.done_ids: set[int] = set()

    # ── resume ────────────────────────────────────────────────────────────────
    def resume(self, records: list[dict]) -> None:
        if not os.path.exists(self.path):
            return
        z = np.load(self.path)
        self._check_meta(z)
        k = z["idx"].shape[0]
        if k > len(self.idx):
            raise SystemExit(
                f"[FATAL] {self.path} holds {k} rows but the current input yields only "
                f"{len(self.idx)} labelable records. The input shrank (or --limit was "
                f"lowered); refusing to resume into a smaller dump."
            )
        self.Z_head[:k] = z["Z_head"]
        self.y[:k] = z["y"]; self.maj[:k] = z["maj"]
        self.p_yes[:k] = z["p_yes"]; self.idx[:k] = z["idx"]
        # Adopt the existing file's schema so a resume never silently drops arrays
        # an earlier run wrote.
        if "X_last" in z.files and not self.store_resid:
            print("[WARN] existing dump stores X_last/X_mean; keeping them for schema "
                  "consistency (pass --store_resid to silence)", flush=True)
            self.store_resid = True
            self.meta["store_resid"] = True
            self.X_last = np.zeros((len(self.idx),) + z["X_last"].shape[1:], np.float32)
            self.X_mean = np.zeros((len(self.idx),) + z["X_mean"].shape[1:], np.float32)
        if self.store_resid and "X_last" in z.files:
            self.X_last[:k] = z["X_last"]; self.X_mean[:k] = z["X_mean"]

        self.done_ids = set(z["idx"].tolist())
        if len(self.done_ids) != k:
            raise SystemExit(f"[FATAL] {self.path} contains duplicate idx values; "
                             f"{k} rows but {len(self.done_ids)} unique indices.")
        # Repair labels of the resumed prefix under the current policy.
        by_idx = {r["idx"]: r for r in records}
        unknown = [i for i in self.idx[:k].tolist() if i not in by_idx]
        if unknown:
            raise SystemExit(
                f"[FATAL] {self.path} references {len(unknown)} record idx values absent "
                f"from the current --input (first={unknown[0]}). The dump and the raw file "
                f"come from different runs, or --limit changed. Use a fresh --out."
            )
        # Matching by idx alone accepts any raw file carrying the same indices: model
        # and geometry still agree, the labels are simply recomputed from the new text,
        # and the retained activations were read at entirely different prompts.
        stored_digest_version = (json.loads(str(z["meta"])).get("digest_version", 1)
                                 if "meta" in z.files else 1)
        if "rec_sha" not in z.files:
            print(f"[WARN] {self.path} predates per-record fingerprints; its rows are "
                  f"matched to --input by idx alone, which cannot detect that they came "
                  f"from a different raw file. Re-extract to enable the check.",
                  flush=True)
        elif stored_digest_version != provenance.DIGEST_VERSION:
            # Every stored digest would mismatch at once. Reporting that as "different
            # records" would send you looking for a data problem that does not exist.
            raise SystemExit(
                f"[FATAL] {self.path} stores rec_sha under digest_version="
                f"{stored_digest_version}; this build computes v{provenance.DIGEST_VERSION} "
                f"(v2 added judge_verdict).\n"
                f"        The stored fingerprints cannot be compared against the current "
                f"rule, so resuming would extend the dump on an unverified prefix.\n"
                f"        This is a definition change, not a data mismatch — re-extract "
                f"to a fresh --out."
            )
        else:
            self.rec_sha[:k] = z["rec_sha"]
            mismatched = [i for j, i in enumerate(self.idx[:k].tolist())
                          if self.rec_sha[j] and self.rec_sha[j] != record_digest(by_idx[i])]
            if mismatched:
                raise SystemExit(
                    f"[FATAL] {self.path} was extracted from different records than "
                    f"--input holds: {len(mismatched)} of {k} resumed rows disagree "
                    f"(first idx={mismatched[0]}).\n"
                    f"        'idx' is a position in a shuffled subset, so two runs can "
                    f"share every index while holding different problems, solutions or "
                    f"gold answers.\n"
                    f"        The activations already in the dump were read at that "
                    f"file's prompts, not this one's. Use a fresh --out."
                )
        for j, record_idx in enumerate(self.idx[:k].tolist()):
            rec = by_idx[record_idx]
            self.y[j] = _label(rec, "phase1_answer", record_idx)
            self.maj[j] = _label(rec, "majority_answer", record_idx)
            self.rec_sha[j] = record_digest(rec)
        self.filled = k
        print(f"resuming: {k} already extracted", flush=True)

    # Advisory: these may legitimately change between resumes without invalidating
    # the activations already written. `gen_run` is advisory only because the strict
    # subset of it is compared separately as `prompt_identity`.
    # `digest_version` is handled explicitly in resume() so the failure reads as a
    # definition change; leaving it to the generic drift check would report it as a
    # different model/geometry, which is both wrong and unactionable.
    _META_ADVISORY = ("label_policy", "store_resid", "gen_run", "digest_version")

    def _check_meta(self, z) -> None:
        if "meta" not in z.files:
            print("[WARN] existing dump predates metadata; cannot verify it came from "
                  "the same model/head_dim. Verify manually or re-extract.", flush=True)
            return
        old = json.loads(str(z["meta"]))
        # A key the older dump never recorded is unknown, not conflicting: treating
        # absence as a mismatch would make every dump written before a new field was
        # added unresumable. Only a key present on both sides with different values
        # proves the dumps came from different runs.
        drift, unknown = {}, []
        for k, v in self.meta.items():
            if k in self._META_ADVISORY:
                continue
            if k not in old:
                unknown.append(k)
            elif old[k] != v:
                drift[k] = (old[k], v)
        if drift:
            detail = "; ".join(f"{k}: existing={a!r} current={b!r}"
                               for k, (a, b) in sorted(drift.items()))
            raise SystemExit(
                f"[FATAL] {self.path} was extracted from a different model/geometry "
                f"({detail}). Resuming would mix activations from two models. Use a fresh --out."
            )
        if unknown:
            print(f"[WARN] {self.path} predates dump metadata field(s) "
                  f"{', '.join(sorted(unknown))}; they could not be verified against the "
                  f"rows already written.", flush=True)

    # ── writing ───────────────────────────────────────────────────────────────
    def add(self, rec: dict, z_head: np.ndarray, p_yes: float,
            x_last: np.ndarray | None = None, x_mean: np.ndarray | None = None) -> None:
        i = self.filled
        self.Z_head[i] = z_head
        if self.store_resid:
            self.X_last[i] = x_last
            self.X_mean[i] = x_mean
        self.y[i] = _label(rec, "phase1_answer", rec["idx"])
        self.maj[i] = _label(rec, "majority_answer", rec["idx"])
        self.p_yes[i] = p_yes
        self.idx[i] = rec["idx"]
        self.rec_sha[i] = record_digest(rec)
        self.filled += 1

    def flush(self) -> None:
        """Save atomically: a kill mid-write must not truncate the previous dump."""
        n = self.filled
        arrays = dict(Z_head=self.Z_head[:n], y=self.y[:n], maj=self.maj[:n],
                      p_yes=self.p_yes[:n], idx=self.idx[:n],
                      rec_sha=self.rec_sha[:n],
                      meta=np.array(json.dumps(self.meta)))
        if self.store_resid:
            arrays["X_last"] = self.X_last[:n]
            arrays["X_mean"] = self.X_mean[:n]
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, self.path)


def _label(rec: dict, answer_key: str, record_idx: int) -> int:
    lab = label_correctness(rec.get(answer_key), rec.get("gold_answer"),
                            record_dataset(rec))
    if lab is None:
        # read_records filters these out; reaching here means the caller bypassed it.
        raise RuntimeError(f"record idx={record_idx} has an unparseable gold answer "
                           f"({rec.get('gold_answer')!r}) and cannot be labelled")
    return int(lab)
