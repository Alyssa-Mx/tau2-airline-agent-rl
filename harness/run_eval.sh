#!/bin/bash
# 评测：tau2 官方 CLI 跑 airline，客服 = 本地 vLLM 上的被测模型，用户模拟器 = DeepSeek-v4-Flash（temp 0）。
# 每题 4 遍、最多 40 步、seed 900、并发 32；默认测 20 道 held-out 测试题（task_id % 5 ∈ {0,1}）。
#   MODE=lean_think   定版：四项精简 + 原生思考（思考额度 4096 + 回复 2048）+ 续写 / 重采兜底      [ENGINE=native]
#   MODE=orig_nothink 同引擎对照：四项精简全关、不思考（其余与定版相同）                           [ENGINE=native]
#   MODE=lean_nothink 只开精简、不思考                                                           [ENGINE=native]
#   MODE=legacy       奖励线各 checkpoint 的评测口径：temp 0.5、每轮 2048、stop </tool_call>、一轮一个工具调用 [ENGINE=legacy]
# 判分：TAU2_DATA_DIR 指 tau2-bench/data。仓库里报的主口径是"宽松"= DB × COMMUNICATE（官方 reward_basis），
#   用 analysis/ 从仿真结果离线计算；需要"严格"（再加 ACTION）时把 tasks.json 的 reward_basis 加上 ACTION。
# 用法：MODE=lean_think RUN=my_eval PORT=8000 TAU2_USER_API_BASE=... TAU2_USER_API_KEY=... bash harness/run_eval.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${MODE:?}; RUN=${RUN:?}; PORT=${PORT:-8000}
: "${TAU2_USER_API_BASE:?}"; : "${TAU2_USER_API_KEY:?}"
TASKS=${TASKS:-"0 1 5 6 10 11 15 16 20 21 25 26 30 31 35 36 40 41 45 46"}
VLLM="\"api_base\": \"http://127.0.0.1:$PORT/v1\", \"api_key\": \"EMPTY\""
REP='"repetition_detection": {"min_pattern_size": 1, "max_pattern_size": 20, "min_count": 32}'
case $MODE in
  lean_think|orig_think)
    AGENT_ARGS="{$VLLM, \"temperature\": 1.0, \"max_tokens\": 6144, \"parallel_tool_calls\": true, \"top_p\": 0.95, \"presence_penalty\": 1.5,
      \"extra_body\": {\"top_k\": 20, \"min_p\": 0.0, \"repetition_penalty\": 1.0, \"thinking_token_budget\": 4096, $REP,
                      \"chat_template_kwargs\": {\"enable_thinking\": true}}}" ;;
  lean_nothink|orig_nothink)
    AGENT_ARGS="{$VLLM, \"temperature\": 0.7, \"max_tokens\": 2048, \"parallel_tool_calls\": true, \"top_p\": 0.8, \"presence_penalty\": 1.5,
      \"extra_body\": {\"top_k\": 20, \"min_p\": 0.0, \"repetition_penalty\": 1.0, $REP, \"chat_template_kwargs\": {\"enable_thinking\": false}}}" ;;
  legacy)
    AGENT_ARGS="{$VLLM, \"temperature\": 0.5, \"max_tokens\": 2048, \"stop\": [\"</tool_call>\"], \"parallel_tool_calls\": false,
      \"extra_body\": {\"include_stop_str_in_output\": true, \"chat_template_kwargs\": {\"enable_thinking\": false}}}" ;;
  *) echo "MODE 必须是 lean_think / orig_think / lean_nothink / orig_nothink / legacy"; exit 2 ;;
esac
case $MODE in
  lean_*) export LEAN_TOOL_DEFS=1 LEAN_COMPACT_RESULTS=1 LEAN_MASK_OLDER=1 LEAN_DROP_FLOODS=1 LEAN_REGEN_FLOOD_FALLBACK=1 ;;
  *)      export LEAN_TOOL_DEFS=0 LEAN_COMPACT_RESULTS=0 LEAN_MASK_OLDER=0 LEAN_DROP_FLOODS=0 LEAN_REGEN_FLOOD_FALLBACK=0 ;;
esac
if [ "$MODE" = legacy ]; then export LEAN_AGENT_NAME=llm_agent             # 奖励线用 tau2 原版客服实现
else export LEAN_AGENT_NAME=lean_agent LEAN_THINK_CLOSE_CONTINUE=1 LEAN_EMPTY_RETRY=3; fi
export LEAN_LOG_DIR=${LEAN_LOG_DIR:-runs_eval/$RUN/lean_logs}
USER_ARGS="{\"temperature\": 0, \"api_base\": \"$TAU2_USER_API_BASE\", \"api_key\": \"$TAU2_USER_API_KEY\"}"
python "$HERE/lean_tau2_cli.py" run --domain airline --agent llm_agent --user user_simulator \
  --agent-llm openai/basemodel --agent-llm-args "$AGENT_ARGS" \
  --user-llm "${TAU2_USER_LLM:-openai/deepseek-v4-flash}" --user-llm-args "$USER_ARGS" \
  --num-trials ${TRIALS:-4} --max-concurrency ${MAXC:-32} --max-steps 40 --max-retries 5 --retry-delay 6 \
  --seed 900 --save-to "$RUN" --task-ids $TASKS
