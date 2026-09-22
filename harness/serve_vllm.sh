#!/bin/bash
# 起评测用的 vLLM 服务（vLLM 0.19.1，单卡 TP=1，前缀缓存关）。两套口径：
#   ENGINE=native  （推理外壳线）Qwen3.5 官方原生调用：qwen3_coder 工具解析器 + 思考解析器（默认挂 qwen3_toolend 插件）
#                  + 精简模板（不传 expanded_tools 时与官方模板逐 token 相同，所以原始 / 精简两组共用一台服务）
#   ENGINE=legacy  （奖励线评测）qwen3_xml 工具解析器、无思考解析器、官方模板
# 前缀缓存一律关：Qwen3.5 的 GDN 混合架构在 vLLM 0.19.1 上开缓存后，一旦出现刷屏，97–99% 的后续回复继续刷屏。
# 用法：MODEL=<Qwen3.5-4B 或 checkpoint 的 hf 目录> ENGINE=native PORT=8000 GPU=0 bash harness/serve_vllm.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${MODEL:?设 MODEL=<模型目录>}"
ENGINE=${ENGINE:-native}; PORT=${PORT:-8000}; GPU=${GPU:-0}
COMMON=(--model "$MODEL" --served-model-name basemodel --port "$PORT" --tensor-parallel-size 1 --dtype auto
        --gpu-memory-utilization 0.85 --gdn-prefill-backend triton --enable-auto-tool-choice)
case $ENGINE in
  native)
    PARSER=${PARSER:-qwen3_toolend}
    RP=(--reasoning-parser "$PARSER"); [ "$PARSER" = qwen3_toolend ] && RP=(--reasoning-parser-plugin "$HERE/qwen3_toolend_reasoning_parser.py" --reasoning-parser qwen3_toolend)
    EXTRA=(--max-model-len 65536 --max-num-seqs 64 --tool-call-parser qwen3_coder "${RP[@]}"
           --chat-template "${TEMPLATE:-$HERE/chat_template_lean.jinja}"
           --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}') ;;
  legacy)
    EXTRA=(--max-model-len 32768 --tool-call-parser qwen3_xml) ;;
  *) echo "ENGINE 必须是 native 或 legacy"; exit 2 ;;
esac
CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0 exec python -m vllm.entrypoints.openai.api_server "${COMMON[@]}" "${EXTRA[@]}"
