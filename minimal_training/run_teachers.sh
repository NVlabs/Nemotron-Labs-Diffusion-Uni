#!/bin/bash
# Run on the teacher node; each process owns four visible GPUs.
set -euo pipefail
: "${VLM_MODEL:?Set the VLM teacher model path or HF repository}"
: "${EDIT_MODEL:?Set the Qwen-Image-Edit-2511 model path or HF repository}"
: "${VLM_PYTHON:?Set the Python executable in the VLM teacher environment}"
: "${EDIT_PYTHON:?Set the Python executable in the image teacher environment}"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3 "$VLM_PYTHON" -u "$root/teachers.py" vlm --model "$VLM_MODEL" --port 8123 &
vlm_pid=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 "$EDIT_PYTHON" -u "$root/teachers.py" edit --model "$EDIT_MODEL" --port 8124 &
edit_pid=$!
trap 'kill "$vlm_pid" "$edit_pid" 2>/dev/null || true' EXIT
wait -n "$vlm_pid" "$edit_pid"
