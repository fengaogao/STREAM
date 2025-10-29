#!/usr/bin/env bash
# Example launch script for running STREAM's offline training on multiple GPUs.
#
# Usage:
#   MODEL_PATH="/path/to/your/hf/model" ./experiments/train_offline_multi_gpu_example.sh
#
# You can override DATA_DIR, OUT_DIR, CUDA_VISIBLE_DEVICES, and other arguments
# by exporting them before invoking the script. The defaults below mirror the
# typical MovieLens-10M layout used in the repository.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/ml-10M100K}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/causal-multi-gpu}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-2-7b-hf}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Please set MODEL_PATH to the pretrained causal LM checkpoint (local path or HF id)." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

python "${ROOT_DIR}/stream/train_offline.py" \
  --model_type causal \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --pretrained_name_or_path "${MODEL_PATH}" \
  --epochs "${EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --lr "${LR:-2e-4}" \
  --seed "${SEED:-17}" \
  --device "${DEVICE:-cuda}" \
  "$@"
