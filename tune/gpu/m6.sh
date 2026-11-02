#!/usr/bin/env bash
# M6 on the pod: GGUF and AWQ copies of the deployed model, their quality, and a
# vLLM serving sweep (prefix caching on/off, AWQ). Run from /workspace/creo.
set -x
M=${1:-/workspace/models/grpo-sft-0.6b/merged}
USD_PER_HOUR=${2:-1.19}
cd /workspace/creo/tune/gpu
export HF_HOME=/workspace/hf
# Scratch on the container disk (40 GB, ephemeral): the 20 GB volume holds the
# training venv and the models.
Q=/root/q
mkdir -p $Q

# Quantization tools in their own venv (llmcompressor and vllm pin different compressed-tensors).
python -m venv --system-site-packages /root/venv-q
source /root/venv-q/bin/activate
pip install -q "llmcompressor==0.15.0" "transformers==5.19.0" "datasets==5.1.0" sentencepiece
pip install -q --no-deps -e /workspace/creo
test -d /root/llama.cpp || git clone -q --depth 1 --branch b12032 https://github.com/ggml-org/llama.cpp /root/llama.cpp
pip install -q -e /root/llama.cpp/gguf-py
python /root/llama.cpp/convert_hf_to_gguf.py "$M" --outtype f16 --outfile $Q/creo-classify-0.6b-f16.gguf
python /root/llama.cpp/convert_hf_to_gguf.py "$M" --outtype q8_0 --outfile $Q/creo-classify-0.6b-q8_0.gguf
python awq.py --model "$M" --out $Q/deploy-awq
deactivate

source /workspace/venv/bin/activate
python gen.py --model $Q/deploy-awq --out /workspace/out/deploy-awq
echo QUANT_DONE

serve() {  # serve <model> <name> [extra vllm args...]
  local model=$1 name=$2; shift 2
  vllm serve "$model" --served-model-name creo-classify --port 8000 --max-model-len 4096 \
    --gpu-memory-utilization 0.85 "$@" > "/workspace/serve-$name.log" 2>&1 &
  local pid=$!
  until curl -sf localhost:8000/v1/models > /dev/null; do sleep 2; done
  python -m creo.tune.bench --url http://localhost:8000/v1 --model creo-classify \
    --name "vllm-5090-$name" --usd-per-hour "$USD_PER_HOUR" --note "vllm 0.32.0, RTX 5090, $*"
  kill $pid; wait $pid 2>/dev/null
}
cd /workspace/creo
serve "$M" bf16-prefix
serve "$M" bf16-noprefix --no-enable-prefix-caching
serve $Q/deploy-awq awq-prefix
echo BENCH_DONE
