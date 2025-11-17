#!/usr/bin/env bash
set -euo pipefail

# This script benchmarks four variants:
#   1) PyTorch FP32
#   2) PyTorch INT8 (converted from LSQ QAT)
#   3) ONNX FP32
#   4) ONNX INT8
#
# Prerequisites:
#   - Datasets downloaded to ./data (Set5/LRbicx4, etc.)
#   - LSQ checkpoints present in ./results/ESPCN_x4-T91-*/ directories
#   - Python environment with requirements installed (pip install -r requirements.txt)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ORT_DISABLE_CPU_AFFINITY=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
RESULTS_DIR="${ROOT_DIR}/results"
LOG_DIR="${RESULTS_DIR}/linux_benchmarks_logs"
mkdir -p "${LOG_DIR}"

FP32_CKPT="${RESULTS_DIR}/ESPCN_x4-T91/g_best.pth.tar"
LSQ_QAT_CKPT="${RESULTS_DIR}/ESPCN_x4-T91-LSQ-QAT/g_best.pth.tar"
INT8_OUT_DIR="${RESULTS_DIR}/ESPCN_x4-T91-LSQ-QAT/g_best_linux_INT8"
ONNX_DIR="${RESULTS_DIR}/ESPCN_x4-T91-LSQ-QAT/onnx_export"
ONNX_FP32="${ONNX_DIR}/espcn_lsq_fp32.onnx"
ONNX_INT8="${ONNX_DIR}/espcn_lsq_int8.onnx"

INPUT_SHAPE="1 1 64 64"
RUNS=200
WARMUP=20

echo "Logging to ${LOG_DIR}"

echo "================================================================"
echo "1/5 PyTorch FP32"
echo "================================================================"
python "${ROOT_DIR}/scripts/test_set5.py" \
    --checkpoint "${FP32_CKPT}" \
    --mode fp32 \
    --device cpu | tee "${LOG_DIR}/pytorch_fp32.txt"

echo "================================================================"
echo "2/5 PyTorch INT8 (convert + evaluate)"
echo "================================================================"
python "${ROOT_DIR}/scripts/convert_qat_to_int8.py" \
    --method lsq \
    --checkpoint "${LSQ_QAT_CKPT}" \
    --output_dir "${INT8_OUT_DIR}" | tee "${LOG_DIR}/pytorch_int8.txt"

echo "================================================================"
echo "3/5 ONNX FP32 export + benchmark"
echo "================================================================"
python "${ROOT_DIR}/scripts/export_lsq_to_onnx.py" \
    --checkpoint "${LSQ_QAT_CKPT}" \
    --output "${ONNX_FP32}" \
    --lr-size 64 \
    --opset 17 | tee "${LOG_DIR}/onnx_export.txt"

python "${ROOT_DIR}/scripts/benchmark_onnx_runtime.py" \
    --onnx "${ONNX_FP32}" \
    --device cpu \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --input-shape ${INPUT_SHAPE} \
    --threads 4 | tee "${LOG_DIR}/onnx_fp32.txt"

echo "================================================================"
echo "4/5 ONNX INT8 quantization"
echo "================================================================"
python "${ROOT_DIR}/scripts/quantize_onnx_model.py" \
    --onnx "${ONNX_FP32}" \
    --output "${ONNX_INT8}" \
    --mode static \
    --calib-dir "${ROOT_DIR}/data/Set5/LRbicx4" \
    --input-shape ${INPUT_SHAPE} \
    --samples 5 \
    --per-channel | tee "${LOG_DIR}/onnx_quantize.txt"

echo "================================================================"
echo "5/5 ONNX INT8 benchmark"
echo "================================================================"
python "${ROOT_DIR}/scripts/benchmark_onnx_runtime.py" \
    --onnx "${ONNX_INT8}" \
    --device cpu \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --input-shape ${INPUT_SHAPE} \
    --threads 4 | tee "${LOG_DIR}/onnx_int8.txt"

echo "================================================================"
echo "All benchmarks complete. Logs saved in ${LOG_DIR}"
echo "================================================================"

