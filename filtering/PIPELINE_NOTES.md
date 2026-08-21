# White-box probe gate + filtering (clean train/test split)

A linear probe on the judge's **attention-head activations** ranks self-judge correctness
far better than the model's spoken YES/NO verdict, and gates a filter-and-route reward
pipeline (ACCEPT / REJECT / ROUTE-to-self-consistency).

This folder uses a **clean train/test split**: the probe is *fit* on the GSM8K **train**
split and *evaluated* on the held-out **test** split — no fold leakage. `train_test_probe.py`
does top-32 head selection + shallow-feature residualization, all fit on train only, then
reports held-out **test** AUC (raw + shallow-controlled), both judge baselines, and a
risk-coverage table. Logistic regression is the default baseline; `--models lr,mlp` also runs
a small sklearn MLP on the exact same split, heads, residualizer, and metrics.

"Fit on train only" includes the **operating point**: `error_detection_compare.py` picks each
detector's decision threshold on train, at the judge's train precision, and applies it
unchanged to test.

That calibrates on train. It does **not** equalise precision on test, and the results are
not "at matched precision" — each rule's *realised test precision* is printed beside its
recall exactly because those values diverge, sometimes by tens of points. Recalls measured
at different realised precisions are not directly comparable, so **AUC and risk-coverage
remain the primary comparisons** and the recall table is a secondary, clearly-conditioned
view.

Reading the best threshold off the test PR curve is a best-of-all-thresholds statistic on
the reporting split. It is still printed, labelled **test-oracle, descriptive only**, and is
deliberately absent from every figure and from the main table.

Stochastic probes are **ensembled**: the MLP is fitted at `--n_seeds` initialisations and
its predicted probabilities are averaged. That one ensemble drives every curve, threshold,
bar and table cell, so the line on a figure and the number annotating it are the same
object. The two uncertainty sources — train-resample spread and initialisation-only spread
— are reported separately and never merged into a single `mean ± sd`.

Correctness is recomputed here from `phase1_answer` and `gold_answer` in `raw.jsonl`; it does
not trust the historical `y` field in activation dumps, and each load reports the disagreement
count. Pass `--label_policy stored` only to reproduce the legacy exact-string-label
sensitivity condition.

The equivalence rule follows the `dataset` field on each record — a record without one
predates the MATH add-on and is treated as GSM8K:

- **GSM8K**: exact **numeric equivalence** after parsing, so `64`, `64.0`, and `64.00` share
  one label.
- **MATH**: LaTeX normalization then string equality, plus decimal equality when both sides
  are plain numeric literals. `\dfrac{1}{2}` and `\frac12` are one answer; `\frac{1}{2}` and
  `0.5` deliberately are not.

`load_split` refuses a `raw.jsonl` that mixes datasets, and refuses a dump whose recorded
dataset disagrees with the raw file — either would put two equivalence policies under one
probe. Swap `gsm8k` for `math` in the paths below to analyse a MATH run; figure captions read
the dataset off the records rather than being hardcoded.

> One-command version of everything below: **`./run_full.sh`** from the repo root (it runs
> generate-train → generate-test → extract both → `train_test_probe.py`). The steps here are
> the manual breakdown.

## Environment

```bash
conda activate self_judge      # from `pip install -r requirements.txt` (see top-level README)
PY=python                      # torch>=2.5, transformers==5.12.1, mistral_common==1.11.3
```

Generation uses `run_eval.py --backend transformers` (FP8→bf16 for Ministral, native bf16 for
Gemma3) — the `--backend vllm` GGUF path can't load `mistral3`. Download the full safetensors
checkpoint first (see the top-level README Setup callout).

## Per-model variables

```bash
# Ministral:
MODEL=mistralai/Ministral-3-8B-Instruct-2512 ; EXTRACT=extract_hidden_rich.py ; HEAD_DIM=128 ; TAG=ministral
# Gemma3 (gated — huggingface-cli login first):
# MODEL=google/gemma-3-12b-it ; EXTRACT=extract_hidden_rich_gemma3.py ; HEAD_DIM=256 ; TAG=gemma3
TRAIN=results_${TAG}_train ; TEST=results_${TAG}_test
```

## Pipeline

**A. Generate train-split judge records.** The probe only needs `solve_correct` + the judge
activations, *not* the k-sample majority — so use `--k_samples 1` (≈3× faster than k=5):

```bash
GSM8K_SPLIT=train $PY run_eval.py --backend transformers --model "$MODEL" --n_problems 2000 --k_samples 1 \
  --output_dir "$TRAIN"          # resumable; bump --n_problems up to 7473 for the full train split
```

**B. Generate test-split judge records:**

```bash
GSM8K_SPLIT=test $PY run_eval.py --backend transformers --model "$MODEL" --n_problems 1319 --k_samples 1 \
  --output_dir "$TEST"
```

**C. Extract attention-head activations** for each split (one forward pass per record,
captures `Z_head` = o_proj input at the verdict token; uses the model's `$EXTRACT` script):

```bash
$PY "$EXTRACT" --input "$TRAIN/gsm8k/raw.jsonl" --out "$TRAIN/hidden_rich.npz"
$PY "$EXTRACT" --input "$TEST/gsm8k/raw.jsonl"  --out "$TEST/hidden_rich.npz"
```

**D. Fit probe on TRAIN, evaluate on held-out TEST** (`--head_dim` must match the model:
128 Ministral / 256 Gemma3):

```bash
$PY filtering/train_test_probe.py \
  --train_hidden "$TRAIN/hidden_rich.npz" --train_raw "$TRAIN/gsm8k/raw.jsonl" \
  --test_hidden  "$TEST/hidden_rich.npz"  --test_raw  "$TEST/gsm8k/raw.jsonl" \
  --head_dim "$HEAD_DIM" --title "$TAG" --out "filtering/figures/${TAG}_train_test_probe.png"
```

Prints held-out TEST AUC (raw + shallow-controlled), both judge baselines (the binary
verdict's operating point and the continuous `p_yes` AUC), and a risk-coverage table;
saves the figure to `filtering/figures/`. More train errors → a more stable controlled probe.

Risk-coverage here means a filtering gate: accept the highest-`P(correct)` items first and
report the **wrong rate among accepted**, so the curve meets the no-filtering line at 100%
coverage. Note that `p_yes` is the fair black-box comparison — the verdict is binary, so
its ROC AUC is just balanced accuracy and cannot reach what a continuous score can.

To compare logistic regression against the analysis-only MLP:

```bash
$PY filtering/train_test_probe.py \
  --train_hidden "$TRAIN/hidden_rich.npz" --train_raw "$TRAIN/gsm8k/raw.jsonl" \
  --test_hidden  "$TEST/hidden_rich.npz"  --test_raw  "$TEST/gsm8k/raw.jsonl" \
  --head_dim "$HEAD_DIM" --title "$TAG" --models lr,mlp \
  --out "filtering/figures/${TAG}_train_test_probe_mlp.png"

$PY filtering/error_detection_compare.py \
  --train_hidden "$TRAIN/hidden_rich.npz" --train_raw "$TRAIN/gsm8k/raw.jsonl" \
  --test_hidden  "$TEST/hidden_rich.npz"  --test_raw  "$TEST/gsm8k/raw.jsonl" \
  --head_dim "$HEAD_DIM" --title "$TAG" --models lr,mlp \
  --out "filtering/figures/${TAG}_error_detection_mlp.png"
```

## Recreate LR-vs-MLP comparison plots

Run these from the project root after the corresponding `results_*_{train,test}` activation
dumps exist. They regenerate the saved risk-coverage plots in `filtering/figures/`.

`run_full.sh` step 4 runs `train_test_probe.py` with the **default `--models lr`**, so the
LR-vs-MLP panel is always a separate invocation — a completed `run_full.sh` does not
produce it. `$DS` is the dataset subdirectory `run_eval.py` wrote the records into
(`gsm8k` or `math`); it must match the dataset the dumps were extracted from.

```bash
PY=python ; DS=gsm8k

# Gemma3:
TAG=gemma3 ; HEAD_DIM=256 ; TRAIN=results_${TAG}_train ; TEST=results_${TAG}_test
MPLCONFIGDIR=/tmp $PY filtering/train_test_probe.py \
  --train_hidden "$TRAIN/hidden_rich.npz" --train_raw "$TRAIN/$DS/raw.jsonl" \
  --test_hidden  "$TEST/hidden_rich.npz"  --test_raw  "$TEST/$DS/raw.jsonl" \
  --head_dim "$HEAD_DIM" --title "$TAG" --models lr,mlp \
  --out "filtering/figures/${TAG}_probe_compare.png"

# Gemma3 on MATH (same command, different tag and dataset subdirectory):
# TAG=gemma3_math ; HEAD_DIM=256 ; DS=math

# Llama-3.1-8B:
TAG=llama31_8b ; HEAD_DIM=128 ; TRAIN=results_${TAG}_train ; TEST=results_${TAG}_test
MPLCONFIGDIR=/tmp $PY filtering/train_test_probe.py \
  --train_hidden "$TRAIN/hidden_rich.npz" --train_raw "$TRAIN/gsm8k/raw.jsonl" \
  --test_hidden  "$TEST/hidden_rich.npz"  --test_raw  "$TEST/gsm8k/raw.jsonl" \
  --head_dim "$HEAD_DIM" --title "$TAG" --models lr,mlp \
  --out "filtering/figures/${TAG}_probe_compare.png"

# Qwen2.5-7B:
TAG=qwen25_7b ; HEAD_DIM=128 ; TRAIN=results_${TAG}_train ; TEST=results_${TAG}_test
MPLCONFIGDIR=/tmp $PY filtering/train_test_probe.py \
  --train_hidden "$TRAIN/hidden_rich.npz" --train_raw "$TRAIN/gsm8k/raw.jsonl" \
  --test_hidden  "$TEST/hidden_rich.npz"  --test_raw  "$TEST/gsm8k/raw.jsonl" \
  --head_dim "$HEAD_DIM" --title "$TAG" --models lr,mlp \
  --out "filtering/figures/${TAG}_probe_compare.png"
```

## Ministral via the vLLM fast path

Ministral-3-8B-2512 now generates on the vLLM backend (no transformers-5 env needed for
generation): its `ministral3` text_config can't be parsed by the transformers bundled
with vLLM 0.10, but the repo ships the native mistral format (`params.json` +
`tekken.json` + `consolidated.safetensors`), loaded with the new `--mistral-format`
flag (FP8 weights, ~10 GB; prompts templated via `mistral_common`, ~4 rec/s at
`--max_concurrent 48`):

```bash
# p311_vllm env:
GSM8K_SPLIT=train $PY run_eval.py --backend vllm --model mistralai/Ministral-3-8B-Instruct-2512 \
  --mistral-format --n_problems 7473 --k_samples 1 --max_concurrent 48 --output_dir results_ministral_train
GSM8K_SPLIT=test $PY run_eval.py --backend vllm --model mistralai/Ministral-3-8B-Instruct-2512 \
  --mistral-format --n_problems 1319 --k_samples 1 --max_concurrent 48 --output_dir results_ministral_test

# extraction still needs the transformers==5.12.1 env (self_judge; ~8 rec/s):
$PY extract_hidden_rich.py --input results_ministral_train/gsm8k/raw.jsonl --out results_ministral_train/hidden_rich.npz
$PY extract_hidden_rich.py --input results_ministral_test/gsm8k/raw.jsonl  --out results_ministral_test/hidden_rich.npz
```

## Paper figures

`make_paper_figures.py` recomputes every metric from the raw artifacts (same
train-only pipeline as above), caches per-model JSON to
`figures/paper/metrics_{tag}.json`, and renders the camera-ready figures + LaTeX
table:

```bash
# slow: loads the npz dumps
MPLCONFIGDIR=/tmp $PY filtering/make_paper_figures.py --compute --runs gemma3
MPLCONFIGDIR=/tmp $PY filtering/make_paper_figures.py --runs gemma3   # re-plot from cache

# skip the train-resample variance pass (much faster, weaker error bars):
MPLCONFIGDIR=/tmp $PY filtering/make_paper_figures.py --compute --runs gemma3 \
  --n_train_resamples 0
```

`--runs` is **required**: there is no default selection, so the same command produces
the same figure on every machine. Pass `--runs all` for every registered tag (fatal if
any lacks data) or `--runs present` to opt explicitly into "whatever exists here".

### Which runs it looks at, and how to add one

There is no path argument: every run is an entry in the `RUNS` dict at the top of
`make_paper_figures.py`, keyed by tag. That dict is a registry of every run the paper
has ever included, so on any one machine most entries have no data. It holds **runs**,
not models — an entry is a checkpoint *and* a dataset *and* its result directories.

`--runs` selects which ones, and it is mandatory:

| value | meaning |
|---|---|
| `a,b` | exactly those, in that order (whitespace and duplicates are tolerated) |
| `all` | every registered tag; fatal if any lacks data, so the output set depends on the registry rather than on this machine |
| `present` | only the tags with data here — still machine-dependent, but now a choice rather than a default nobody saw |

Naming a tag that has no data is a fatal error listing the exact missing files, not a
silent skip: a request for a specific panel that quietly produced a different figure
would be worse than stopping. Requesting two tags when only one has data also stops,
rather than rendering a figure that looks finished but answers a narrower question.

Caches are bound to the configuration they were computed from (`config_sha` over the
tag, title, head_dim, dataset and paths). Reusing a tag for a different run is therefore
rejected instead of silently re-plotting the old numbers under the new name.

Runs whose dumps predate provenance recording (no `rec_sha`, `gen_run` or
`prompt_identity`) are **refused**, because nothing verifies their activations came
from the records beside them. Those fields cannot be added retroactively — re-extract,
or pass `--allow_unauthenticated` to plot them anyway.

Adding a run means adding an entry. A **new dataset needs its own entry**, not a flag —
`dataset` selects both the `raw.jsonl` subdirectory and the answer-equivalence policy
the labels are recomputed under:

```python
"gemma3_math": dict(title="Gemma3-12B (MATH)", head_dim=256, dataset="math",
                    train=ROOT / "results_gemma3_math_train",
                    test=ROOT / "results_gemma3_math_test"),
```

`head_dim` must match the checkpoint (256 Gemma3, 128 for the 8B models); the dump
records its own geometry and the loader rejects a mismatch. `dataset` defaults to
`gsm8k` when omitted.

### Compute per tag, then re-plot with every tag you want on the figure

The caches are per tag (`metrics_{tag}.json`). Each panel figure holds one panel per
tag passed in that invocation, and **the filename names that selection** —
`fig_risk_coverage__gemma3+gemma3_math.png` — so re-plotting one tag alone cannot
overwrite the comparison figure a draft already cites:

```bash
# compute once per run — slow, loads the activation dumps
MPLCONFIGDIR=/tmp $PY filtering/make_paper_figures.py --compute --runs gemma3
MPLCONFIGDIR=/tmp $PY filtering/make_paper_figures.py --compute --runs gemma3_math

# then render the side-by-side comparison from cache — fast, no --compute
MPLCONFIGDIR=/tmp $PY filtering/make_paper_figures.py --runs gemma3,gemma3_math
```

Panel order follows the order of `--runs`, and is part of the filename, so
`--runs a,b` and `--runs b,a` produce two files rather than one. Long selections fall
back to `{n}runs-{hash}`.

### What each figure actually plots

Worth knowing before writing a caption, because the three figures do not show the same
probe variants:

| figure | probes shown | baselines shown |
|---|---|---|
| `fig_risk_coverage` | LR + MLP, **raw (dashed) and controlled (solid)** | `p_yes`, no-filtering line |
| `fig_error_detection_pr` | LR + MLP, **controlled only** | `p_yes` curve, verdict as one point |
| `fig_recall_at_train_calibrated_op` | LR + MLP, **controlled only** | verdict, `p_yes` |
| `table_probe_auc.tex` | all four (LR, LR-ctrl, MLP, MLP-ctrl) | verdict bal.acc, `p_yes` AUC |

So the **raw** probe — usually the strongest number — appears in a figure only as the
dashed risk-coverage line, and otherwise lives in the table and in
`metrics_{tag}.json`. That is a deliberately conservative default, not an oversight: if
the headline claim rests on the raw probe, say so in the caption rather than letting a
reader match the table's best row to a curve that is not on the plot.

How the reported uncertainty is built:

- **Recall of a fixed rule** is a binomial proportion over the wrong examples, so its
  interval is **Wilson**, not bootstrap. A percentile bootstrap bound can land on the
  wrong side of the point estimate, and clamping the resulting negative error bar to
  zero silently redraws "the interval excludes the estimate" as "the interval touches
  it". Wilson bounds always bracket the estimate.
- **The paired probe-minus-judge difference** is not a single proportion, so it still
  uses a 2000-resample bootstrap. Resampling is **unstratified**, so uncertainty in
  the wrong-answer prevalence propagates into precision; holding prevalence fixed
  understates the interval. Thresholds are frozen before resampling, so a replicate
  re-estimates the same statistic rather than re-maximising over thresholds.
- **Probe AUC ±** comes from `--n_train_resamples` replicates that resample the train
  split and **redo head selection inside each replicate**, covering data sampling +
  head selection + initialisation. Refitting an MLP at several seeds holds the data
  and the selected heads fixed and therefore reports initialisation noise alone; with
  a few hundred minority training examples choosing 32 of several hundred candidate
  heads, selection is plausibly the larger term. The seed-only sd is still cached as
  `auc_{probe}_{variant}_sd`, and the LaTeX table states which of the two its ± is.

Where a detector's train precision target is unreachable no operating point exists;
that is reported as **N/A** (`--` in the table, an annotated empty bar in the figure),
never as recall 0 — a literal 0 would read as a detector that genuinely caught nothing.

Outputs in `figures/paper/`: `fig_risk_coverage.{pdf,png}`,
`fig_error_detection_pr.{pdf,png}`,
`fig_recall_at_train_calibrated_op.{pdf,png}`,
`table_probe_auc.tex`, and `heads_{tag}.json` — the **full** candidate-head LDA-AUC
ranking, not just the selected top-k, so the top-k cut can be checked against the rest
of the distribution and re-tuned post hoc.

## Auditing the `p_yes` baseline's YES/NO binning

`p_yes` is the continuous black-box baseline the probes are measured against, and it is
built by summing next-token mass over "YES tokens" and "NO tokens". The rule in
`extraction_common.yes_no_mass` matches by **prefix**, so on gemma-3-12b's 262k
vocabulary **765 tokens count as NO** — including ` not`, ` now`, ` note`, ` nothing`,
` normal`, ` North`, ` November` — against **29 counting as YES** (` yesterday` among
them). Mass the judge puts on ` not` is therefore recorded as a vote that the solution
is wrong. `hidden_rich.npz` stores `p_yes` already normalised, so that mass cannot be
recovered from it after the fact.

`extract_verdict_topk.py` re-runs the same forward pass and saves the raw top-K
`(token id, probability)` pairs (~1 MB per thousand records, no activations);
`filtering/pyes_rule_compare.py` recomputes `p_yes` from them under the shipped prefix
rule and under exact matching, at several top-K windows, and reports whether the
baseline's AUC and its recall at a train-calibrated operating point move.

```bash
TAG=gemma3 ; MODEL=google/gemma-3-12b-it
for S in train test; do
  python extract_verdict_topk.py --model "$MODEL" \
    --input results_${TAG}_${S}/gsm8k/raw.jsonl --out results_${TAG}_${S}/verdict_topk.npz
done

MPLCONFIGDIR=/tmp python filtering/pyes_rule_compare.py \
  --train_topk results_${TAG}_train/verdict_topk.npz --train_raw results_${TAG}_train/gsm8k/raw.jsonl \
  --test_topk  results_${TAG}_test/verdict_topk.npz  --test_raw  results_${TAG}_test/gsm8k/raw.jsonl \
  --verify_train_hidden results_${TAG}_train/hidden_rich.npz \
  --verify_test_hidden  results_${TAG}_test/hidden_rich.npz \
  --title "$TAG" --out filtering/figures/${TAG}_pyes_rule_compare.png
```

The `--verify_*_hidden` flags are the point of the exercise, not an option: they assert
that `p_yes_shipped` in the new dump equals the `p_yes` already in the activation dump,
which is what ties this audit to the numbers the paper figures were built from. Same
prompt, same position, same tokenizer path, so the two must agree to 1e-5; a mismatch is
fatal rather than a warning, because it would mean the two passes did not see the same
input and no rule comparison built on them means anything.

The `@20` rows answer a second question: whether the baseline is reproducible black-box
at all. OpenAI-style `top_logprobs` caps at 20, below the 40 the shipped rule reads.

Both directions are reported — the prefix rule invents NO votes out of ` not` and YES
votes out of ` yesterday`, and only the data says which dominates. `metrics/verdict_tokens.py`
holds both rules, with `tests/test_verdict_tokens.py` pinning `PREFIX` to reproduce
`yes_no_mass` exactly so the published rule cannot be "cleaned up" out from under the
figures.

## Notes / caveats
- Activation dumps (`hidden_rich.npz`) and `raw.jsonl` are gitignored — regenerate via A–C.
- Raw probe scores are **saturated** (pile at 0/1); ranking (AUC) is the load-bearing
  quantity. Calibrate (isotonic/Platt) before using literal accept/reject thresholds.
- The MLP is analysis-only for now; the primary filtering pipeline uses logistic regression.
  It is refit over `--n_seeds` inits and reported mean ± sd, because a single init is one
  sample, not a result.
- All three extractors take `--model` (falling back to `$MODEL`) and record the model id
  and attention geometry in the dump. `--head_dim` is checked against that record: any
  divisor of the o_proj width used to reshape without error, silently splitting real heads.
- The train and test splits are checked against **each other**, not only each against its
  own raw file: model, geometry, dataset, and the generation settings that decide the
  prompt (tokenizer, prefill, judge budget, backend) must agree, and the two problem sets
  must differ. Two splits built from the same problems are leakage that inflates the
  headline number rather than corrupting it, so nothing else would catch it.
  `--allow_split_mismatch` downgrades the check to a warning for a deliberate cross-run
  comparison; `smoke_test.sh` passes it because it feeds one tiny split to both sides.
- Each dump row carries a content fingerprint of the record it came from, so
  `load_split` fails when a `hidden_rich.npz` is paired with a `raw.jsonl` from another
  run. `idx` is a position in a shuffled subset, so index agreement alone proves nothing;
  dumps written before this check warn instead, and re-extracting enables it.
- The bf16 solver is stronger than the original Q4 GGUF, so its natural wrong-rate (and thus
  absolute filtering-error numbers) are lower; the probe-vs-verdict **AUC gap** is the result
  to compare.
- The selected heads are written to `*_heads.json` next to each figure (layer, head index,
  train LDA AUC) — for an attention-head probing result that is the interpretability output.

### Statistical caveats a reviewer will raise

- The wrong-answer class is small (roughly 6% of GSM8K at bf16), so the probe is fit
  against a few hundred minority examples over `topk * head_dim` features, with the heads
  themselves selected from all candidates on the same train split. Treat the absolute AUC
  as optimistic and the probe-vs-baseline **gap** as the claim.
- Bootstrap CIs resample the test split with every decision rule held fixed. They quantify
  test-set sampling noise only — not variance from head selection, probe refitting, or
  the choice of train split.
