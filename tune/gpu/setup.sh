#!/usr/bin/env bash
# Run on the pod from /workspace/creo (the repo, copied with rsync).
# Versions checked 2026-11-02: vllm 0.32.0 pins torch 2.13.0 and transformers <5.20;
# trl 1.17.1 supports vllm <0.33.
set -euo pipefail
cd /workspace/creo
python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
pip install -q --upgrade pip
# CUDA 12.9 builds: vllm's own wheel and torch from the PyTorch index.
VLLM=https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
pip install -q --extra-index-url https://download.pytorch.org/whl/cu129 "vllm @ $VLLM" \
  "torch==2.13.0+cu129" "transformers==5.19.0" "trl==1.17.1" "peft==0.21.2" \
  "datasets==5.1.0" "accelerate==1.15.0" "pydantic==2.14.2" "pyyaml==6.0.3" "httpx==0.28.1"
pip install -q --no-deps -e .
python -c "import torch, vllm, trl, transformers; print(torch.__version__, torch.cuda.get_device_name(), vllm.__version__, trl.__version__, transformers.__version__)"
