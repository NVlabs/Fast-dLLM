#!/usr/bin/env bash
# Fast-dDrive Waymo-E2E open-loop evaluation launcher.
#
# Required env:
#   EVAL_JSON    — Waymo E2E validation JSON
#   IMAGE_ROOT   — root that EVAL_JSON's image paths are relative to
#
# Optional env:
#   MODEL_PATH   — Fast-dDrive checkpoint dir or HuggingFace id
#                  (default: Efficient-Large-Model/Fast-dDrive — paper checkpoint on the HF Hub)
#   MODE         — section_diffusion | scaffold_spec | inference_scaling
#                  (default: scaffold_spec — paper canonical SS)
#   OUTPUT_DIR   — default: fast_ddrive/eval_outputs/<ckpt_basename>_<mode>
#   NUM_GPUS     — default: auto-detect from nvidia-smi
#   MAX_SAMPLES  — default: full validation set

set -eo pipefail

FAST_DDRIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-Efficient-Large-Model/Fast-dDrive}"
: "${EVAL_JSON:?Set EVAL_JSON to the Waymo E2E val JSON path.}"
: "${IMAGE_ROOT:?Set IMAGE_ROOT to the directory referenced by EVAL_JSON image paths.}"

MODE="${MODE:-scaffold_spec}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d '[:space:]' || true)}"
[[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] && [ "${NUM_GPUS}" -ge 1 ] || NUM_GPUS=1
OUTPUT_DIR="${OUTPUT_DIR:-${FAST_DDRIVE_ROOT}/eval_outputs/$(basename "${MODEL_PATH}")_${MODE}}"

echo "=========================================="
echo "Fast-dDrive eval"
echo "  Model:    ${MODEL_PATH}"
echo "  Mode:     ${MODE}"
echo "  NUM_GPUS: ${NUM_GPUS}"
echo "  Output:   ${OUTPUT_DIR}"
echo "=========================================="

python3 "${FAST_DDRIVE_ROOT}/eval/batch_inference.py" \
    --model_path "${MODEL_PATH}" \
    --eval_json "${EVAL_JSON}" \
    --image_root "${IMAGE_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --mode "${MODE}" \
    --num_gpus "${NUM_GPUS}" \
    ${CONFIDENCE_THRESHOLD:+--confidence_threshold ${CONFIDENCE_THRESHOLD}} \
    ${MAX_SAMPLES:+--max_samples ${MAX_SAMPLES}} \
    "$@"
