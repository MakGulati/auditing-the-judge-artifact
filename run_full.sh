#!/usr/bin/env bash
# FULL run with CLEAN TRAIN/TEST SEPARATION (the rigorous setup):
#   fit the attention-head probe on the GSM8K *train* split, evaluate on the held-out *test*
#   split — no OOF fold leakage. Steps: generate train + test -> extract both -> train_test_probe.
#
# The probe only needs solve_correct + judge activations (not the k-sample majority), so this
# runs with k=1 to keep generation fast.
#
#   ./run_full.sh                  # Ministral via transformers (default)
#
#   # Gemma3-12b (gemma3 arch, head_dim 256, its own extractor):
#   MODEL=google/gemma-3-12b-it EXTRACT=extract_hidden_rich_gemma3.py HEAD_DIM=256 TAG=gemma3 ./run_full.sh
#
#   # A 14B (Qwen3 / DeepSeek-R1-Distill-Qwen) needs fp8 + a capped warmup to fit a 32 GiB card:
#   QUANT=fp8 MAX_NUM_SEQS=16 GPU_MEM=0.95 MAX_MODEL_LEN=8192 \
#     MODEL=Qwen/Qwen3-14B EXTRACT=extract_hidden_rich_causal.py HEAD_DIM=128 TAG=qwen3_14b ./run_full.sh
#
#   # Ministral-3-8B (mistral3): needs the transformers backend + a transformers-5.x env:
#   PY=<transformers-5.x python> BACKEND=transformers \
#     MODEL=mistralai/Ministral-3-8B-Instruct-2512 EXTRACT=extract_hidden_rich.py \
#     HEAD_DIM=128 TAG=ministral ./run_full.sh
#
#   # Hendrycks MATH instead of GSM8K (LaTeX answers, \boxed{} output contract):
#   DATASET=math TAG=gemma3_math BACKEND=vllm MODEL=google/gemma-3-12b-it \
#     EXTRACT=extract_hidden_rich_gemma3.py HEAD_DIM=256 ./run_full.sh
#   Use a TAG that is distinct per dataset: results_${TAG}_{train,test}/hidden_rich.npz
#   is one path per TAG, so reusing a TAG across datasets aborts at extraction (the
#   dump records which dataset it came from) rather than mixing two datasets.
#
# Backends (BACKEND=, generation only — steps 1-2):
#   vllm          default; much faster. Works for Qwen/Gemma in the 'p311_vllm' env (vllm 0.10.2,
#                 torch cu128 — runs fine on driver 555 via CUDA minor-version compat).
#                 NOT usable for Ministral there: vLLM's transformers 4.56 can't parse the
#                 'ministral3' nested config (needs transformers 5.x, untested vs this vLLM).
#   transformers  the in-process path for the mistral3 Ministral checkpoint. Needs the
#                 transformers-5.12 env (e.g. conda 'self_judge').
#
# Extractor: use extract_hidden_rich_causal.py for flat-config CausalLMs (Qwen2/Qwen3/Llama),
#   extract_hidden_rich_gemma3.py for gemma3, extract_hidden_rich.py for mistral3/Ministral.
# Pick PY to match the backend's env: the default vLLM path uses the p311_vllm python (it also
# has the sklearn/scipy/matplotlib stack, so extract + probe run there too).
#
# Knobs:  DATASET (gsm8k default | math), N_TRAIN / N_TEST (default to the full split of
#   the selected dataset: gsm8k 7473/1319, math 7500/5000). MATH also honours
#   MATH_SUBJECTS and MATH_LEVELS (comma-separated; default all).
# vLLM knobs (ignored unless BACKEND=vllm): GGUF_FILE (empty=safetensors), GPU_MEM,
#   MAX_MODEL_LEN, MAX_CONCURRENT, MAX_NUM_SEQS, QUANT, KV_DTYPE.
# SOLVE_MAX_TOKENS (default 1024): raise for datasets with long solutions; MATH
#   needs more. MAX_MODEL_LEN must exceed (longest problem + SOLVE_MAX_TOKENS).
#
set -euo pipefail
cd "$(dirname "$0")"

export MODEL=${MODEL:-mistralai/Ministral-3-8B-Instruct-2512}   # all extractors read MODEL
export DATASET=${DATASET:-gsm8k}                                # all extractors read DATASET
EXTRACT=${EXTRACT:-extract_hidden_rich.py}
HEAD_DIM=${HEAD_DIM:-128}
TAG=${TAG:-ministral}
PY=${PY:-python3}

# Per-dataset split env var and full-split sizes. Each loader owns the name of the env
# var that selects its split, so generation and the run fingerprint cannot disagree.
case "$DATASET" in
  gsm8k) SPLIT_ENV=GSM8K_SPLIT; FULL_TRAIN=7473; FULL_TEST=1319 ;;
  math)  SPLIT_ENV=MATH_SPLIT;  FULL_TRAIN=7500; FULL_TEST=5000 ;;
  *) echo "ERROR: unknown DATASET=$DATASET (expected 'gsm8k' or 'math')." >&2; exit 1 ;;
esac
N_TRAIN=${N_TRAIN:-$FULL_TRAIN}
N_TEST=${N_TEST:-$FULL_TEST}
# Forwarded to BOTH generation and extraction so activations are read at the same
# position that emitted the verdict (e.g. '\n</think>\n\n' for thinking-off R1 distills).
PREFILL=${PREFILL:-}

# HEAD_DIM must match the model's attention head_dim. The extractors record the true
# geometry in the dump and filtering/probe_models.py refuses a mismatch, so a wrong
# value fails loudly here instead of silently fragmenting real attention heads.
case "$MODEL" in
  *gemma-3*) EXPECT_HEAD_DIM=256 ;;
  *)         EXPECT_HEAD_DIM="" ;;
esac
if [ -n "$EXPECT_HEAD_DIM" ] && [ "$HEAD_DIM" != "$EXPECT_HEAD_DIM" ]; then
  echo "ERROR: MODEL=$MODEL needs HEAD_DIM=$EXPECT_HEAD_DIM (got $HEAD_DIM)." >&2
  exit 1
fi

# Reply budget for a solution. The final answer is the LAST thing written, so a
# solution cut off here has none and is recorded as a wrong answer. 1024 fits
# GSM8K; it truncated 24% of MATH. Run calibrate_solve_budget.py to size it, and
# keep MAX_MODEL_LEN above (longest problem + SOLVE_MAX_TOKENS), because the
# judge prompt embeds the whole solution.
SOLVE_MAX_TOKENS=${SOLVE_MAX_TOKENS:-1024}
BACKEND=${BACKEND:-transformers}
GGUF_FILE=${GGUF_FILE:-}                 # empty = load HF safetensors (not a .gguf quant)
GPU_MEM=${GPU_MEM:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_CONCURRENT=${MAX_CONCURRENT:-32}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-}           # empty = vLLM default; set low (e.g. 16) for a
                                         # ~14B model on a 32 GiB card (sampler-warmup OOM)
QUANT=${QUANT:-}                         # empty = bf16; 'fp8' ~halves weight VRAM -> bigger ctx
KV_DTYPE=${KV_DTYPE:-}                   # empty = auto; 'fp8' ~halves KV mem -> even bigger ctx

# Backend-specific generation args; vLLM gets its tuning knobs, others use run_eval defaults.
GEN_ARGS=(--backend "$BACKEND" --model "$MODEL" --dataset "$DATASET" \
          --solve_max_tokens "$SOLVE_MAX_TOKENS")
# --dataset selects the judge prompt at extraction; it MUST be the one generation used,
# or activations are read at a different token sequence than produced the verdict.
EXTRACT_ARGS=(--dataset "$DATASET")
if [ "$BACKEND" = "vllm" ]; then
  GEN_ARGS+=(--gguf-file "$GGUF_FILE" \
             --gpu-memory-utilization "$GPU_MEM" \
             --max-model-len "$MAX_MODEL_LEN" \
             --max_concurrent "$MAX_CONCURRENT")
  [ -n "$MAX_NUM_SEQS" ] && GEN_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
  [ -n "$QUANT" ]        && GEN_ARGS+=(--quantization "$QUANT")
  [ -n "$KV_DTYPE" ]     && GEN_ARGS+=(--kv-cache-dtype "$KV_DTYPE")
  [ -n "$PREFILL" ]      && GEN_ARGS+=(--assistant_prefill "$PREFILL")
fi
if [ -n "$PREFILL" ]; then
  # Only VLLMBackend threads the prefill into generation (run_eval.py refuses it on the
  # other backends). Forwarding it to extraction regardless of backend would apply it on
  # exactly one side of the pipeline — the token-position mismatch this forwarding
  # exists to prevent.
  if [ "$BACKEND" != "vllm" ]; then
    echo "ERROR: PREFILL is set but BACKEND=$BACKEND does not support it, so generation" >&2
    echo "       would run without the prefill while extraction applied it." >&2
    echo "       Use BACKEND=vllm, or unset PREFILL." >&2
    exit 1
  fi
  case "$EXTRACT" in
    *causal*) EXTRACT_ARGS+=(--assistant_prefill "$PREFILL") ;;
    *) echo "ERROR: PREFILL is set but $EXTRACT cannot apply it, so extraction would" >&2
       echo "       read activations at a different position than generation." >&2
       echo "       Use EXTRACT=extract_hidden_rich_causal.py." >&2
       exit 1 ;;
  esac
fi

TRAIN=results_${TAG}_train
TEST=results_${TAG}_test

echo ">>> [1/4] generate TRAIN split: $N_TRAIN $DATASET problems ($MODEL, backend=$BACKEND)  — resumable"
env "$SPLIT_ENV=train" $PY run_eval.py "${GEN_ARGS[@]}" \
  --n_problems "$N_TRAIN" --k_samples 1 --output_dir "$TRAIN"

echo ">>> [2/4] generate TEST split: $N_TEST $DATASET problems (backend=$BACKEND)  — resumable"
env "$SPLIT_ENV=test" $PY run_eval.py "${GEN_ARGS[@]}" \
  --n_problems "$N_TEST" --k_samples 1 --output_dir "$TEST"

echo ">>> [3/4] extract attention-head activations for both splits ($MODEL)"
$PY "$EXTRACT" --model "$MODEL" "${EXTRACT_ARGS[@]}" \
  --input "$TRAIN/$DATASET/raw.jsonl" --out "$TRAIN/hidden_rich.npz"
$PY "$EXTRACT" --model "$MODEL" "${EXTRACT_ARGS[@]}" \
  --input "$TEST/$DATASET/raw.jsonl"  --out "$TEST/hidden_rich.npz"

echo ">>> [4/4] fit probe on TRAIN, evaluate on held-out TEST"
$PY filtering/train_test_probe.py \
  --train_hidden "$TRAIN/hidden_rich.npz" --train_raw "$TRAIN/$DATASET/raw.jsonl" \
  --test_hidden  "$TEST/hidden_rich.npz"  --test_raw  "$TEST/$DATASET/raw.jsonl" \
  --head_dim "$HEAD_DIM" --title "$TAG" \
  --out "filtering/figures/${TAG}_train_test_probe.png"

echo ">>> FULL RUN DONE — held-out TEST AUC printed above; figure in filtering/figures/${TAG}_train_test_probe.png"
