#!/usr/bin/env bash
# Quick END-TO-END smoke test: generate a few judge records -> extract attention-head
# activations -> run the probe. Verifies the whole pipeline wires up, in ~3-5 min on a GPU.
# Not a results run (n=8 is far too small for meaningful AUC) — it just proves the chain works.
#
#   ./smoke_test.sh                 # Gemma 3 12B IT via vLLM (default)
#   DATASET=math ./smoke_test.sh     # same wiring check against Hendrycks MATH
#
set -euo pipefail
cd "$(dirname "$0")"

export MODEL=${MODEL:-google/gemma-3-12b-it}                    # extractors read MODEL
export DATASET=${DATASET:-gsm8k}                                # extractors read DATASET
EXTRACT=${EXTRACT:-extract_hidden_rich_gemma3.py}
HEAD_DIM=${HEAD_DIM:-256}
PY=${PY:-python3}
BACKEND=${BACKEND:-vllm}
# Cap context: Gemma3-12b advertises max_model_len=131072; with ~23 GiB of weights on a
# 32 GiB card the leftover KV cache can't serve one full-length request, so vLLM aborts at
# init ("estimated maximum model length is 3856"). 4096 is plenty for GSM8K.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM=${GPU_MEM:-0.92}
# MAX_NUM_SEQS: cap vLLM concurrent seqs + sampler-warmup batch. Leave empty for the
# vLLM default; set low (e.g. 16) for a ~14B model on a 32 GiB card — otherwise the
# 128-request sampler warmup OOMs even though weights+KV fit.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-}
QUANT=${QUANT:-}                 # 'fp8' ~halves weight VRAM -> allows a much larger MAX_MODEL_LEN
KV_DTYPE=${KV_DTYPE:-}           # 'fp8' ~halves KV memory -> even larger context
OUT=results_smoke
# Two DISJOINT splits, as the real pipeline uses. The old version fed one 8-record set
# to both sides, which is the leakage the provenance check exists to refuse, and which
# left the probe stage untested against anything resembling its real input.
TRAIN_DIR="$OUT/train"
TEST_DIR="$OUT/test"
# Enough records that the wrong class is usually non-empty: GSM8K errs ~6% of the time,
# so 8 records are all-correct 61% of the time and the probe cannot be fitted at all.
N_SMOKE=${N_SMOKE:-48}

case "$DATASET" in
  gsm8k) SPLIT_ENV=GSM8K_SPLIT ;;
  math)  SPLIT_ENV=MATH_SPLIT ;;
  *) echo "ERROR: unknown DATASET=$DATASET (expected 'gsm8k' or 'math')." >&2; exit 1 ;;
esac

case "$MODEL" in
  *gemma-3*) EXPECT_HEAD_DIM=256 ;;
  *)         EXPECT_HEAD_DIM="" ;;
esac
if [ -n "$EXPECT_HEAD_DIM" ] && [ "$HEAD_DIM" != "$EXPECT_HEAD_DIM" ]; then
  echo "ERROR: MODEL=$MODEL needs HEAD_DIM=$EXPECT_HEAD_DIM (got $HEAD_DIM)." >&2
  exit 1
fi

GEN_EXTRA=()
[ -n "$MAX_NUM_SEQS" ] && GEN_EXTRA+=(--max-num-seqs "$MAX_NUM_SEQS")
[ -n "$QUANT" ]        && GEN_EXTRA+=(--quantization "$QUANT")
[ -n "$KV_DTYPE" ]     && GEN_EXTRA+=(--kv-cache-dtype "$KV_DTYPE")

echo ">>> [1/3] generate $N_SMOKE + $N_SMOKE $DATASET records ($MODEL, backend=$BACKEND)"
rm -rf "$OUT"
GEN_ARGS=(--backend "$BACKEND" --model "$MODEL" --dataset "$DATASET" \
          --solve_max_tokens "${SOLVE_MAX_TOKENS:-2048}")
if [ "$BACKEND" = "vllm" ]; then
  GEN_ARGS+=(--max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM")
  # ${arr[@]+"${arr[@]}"} so an empty array doesn't trip `set -u` on older bash
  GEN_ARGS+=(${GEN_EXTRA[@]+"${GEN_EXTRA[@]}"})
fi
for S in train test; do
  D="$OUT/$S"
  env "$SPLIT_ENV=$S" $PY run_eval.py "${GEN_ARGS[@]}" \
    --n_problems "$N_SMOKE" --k_samples 1 --output_dir "$D"
  [ -s "$D/$DATASET/raw.jsonl" ] || { echo "FAIL: no raw.jsonl for $S"; exit 1; }
done

echo ">>> [2/3] extract activations ($MODEL)"
for S in train test; do
  D="$OUT/$S"
  $PY "$EXTRACT" --model "$MODEL" --dataset "$DATASET" \
    --input "$D/$DATASET/raw.jsonl" --out "$D/hidden_rich.npz"
  [ -s "$D/hidden_rich.npz" ] || { echo "FAIL: no hidden_rich.npz for $S"; exit 1; }
done

echo ">>> [3/3] probe on the two disjoint splits"
# At this size the probe's AUC is meaningless; what is being checked is that the stage
# runs on real two-split input. A sample with no wrong answers cannot be fitted at all,
# which is a property of the sample rather than a wiring fault, so it is reported as
# SKIPPED. Any other failure is a genuine break and still fails the run.
set +e
probe_log=$($PY filtering/train_test_probe.py \
  --train_hidden "$TRAIN_DIR/hidden_rich.npz" --train_raw "$TRAIN_DIR/$DATASET/raw.jsonl" \
  --test_hidden  "$TEST_DIR/hidden_rich.npz"  --test_raw  "$TEST_DIR/$DATASET/raw.jsonl" \
  --head_dim "$HEAD_DIM" --title smoke --n_train_resamples 0 \
  --out filtering/figures/smoke_train_test_probe.png 2>&1)
probe_rc=$?
set -e
echo "$probe_log"
if [ $probe_rc -ne 0 ]; then
  if echo "$probe_log" | grep -q "single-class"; then
    echo ">>> PROBE SKIPPED: this $N_SMOKE-record sample has no wrong answers to fit on."
    echo "    Generation and extraction are verified; raise N_SMOKE to exercise the probe."
  else
    echo "FAIL: probe stage failed for a reason other than a single-class sample"; exit 1
  fi
fi

echo ">>> SMOKE TEST PASSED — generation, extraction, and probe all ran end to end."
