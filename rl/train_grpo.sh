#!/bin/bash
# GRPO 训练启动：Qwen3.5-4B × τ²-bench airline，verl 0.9（FSDP2 + vLLM async rollout），单机 4 卡。
# 三个臂只差奖励 / 超长处理，其余逐项相同：
#   ARM=binary   官方 0/1（被截断记 0），adv_estimator=grpo
#   ARM=partial  completion-aware 0 / 0.5 / 1（tau2_reward.py），adv_estimator=grpo
#   ARM=mask     官方 0/1 + DAPO 式超长过滤（被截断的对话 response_mask 全 0），adv_estimator=grpo_masked
#
# 默认值 = 实验实际使用的配置：全参数、AdamW lr 1e-6 恒定、每步 30 题 × 6 条 = 180 段对话、mini-batch 2 题（15 次更新）、
#   KL 0.001（low_var_kl）、clip 0.2、token-mean（verl 默认）、每段对话生成预算 16384、每轮 2048、最多 40 步 / 30 个客服轮、
#   同一句回复 8 次即终止；用户模拟器 DeepSeek-v4-Flash temp 0。
# 实验流程是 "3 步 + 从第 3 步 checkpoint 接着训 3 步"（中间 AdamW 状态清零一次；binary / partial 两臂相同）：
#   ARM=partial RUN=partial_p1 bash rl/train_grpo.sh
#   ARM=partial RUN=partial_p2 MODEL=runs/partial_p1/ckpts/global_step_3/actor/huggingface bash rl/train_grpo.sh
# （mask 臂是后补的对照，直接连续训 6 步：STEPS=6。）
#
# 必填环境变量：MODEL（Qwen3.5-4B 目录）、TAU2_DATA_DIR（tau2-bench/data）、TAU2_USER_API_BASE / TAU2_USER_API_KEY（用户模拟器的
#   OpenAI 兼容接口）。数据：python rl/prepare_data.py --data-dir $TAU2_DATA_DIR --out data_rl/
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARM=${ARM:?必须设 ARM=binary / partial / mask}
case $ARM in
  binary)  ADV=grpo;        export TAU2_REWARD_V3=0 TAU2_MASK_OVERLONG=0 ;;
  partial) ADV=grpo;        export TAU2_REWARD_V3=1 TAU2_MASK_OVERLONG=0 TAU2_PARTIAL=${TAU2_PARTIAL:-0.5} ;;
  mask)    ADV=grpo_masked; export TAU2_REWARD_V3=0 TAU2_MASK_OVERLONG=1 ;;
  *) echo "ARM 必须是 binary / partial / mask，收到 $ARM"; exit 2 ;;
esac
export PYTHONPATH=$HERE:${PYTHONPATH:-}
: "${MODEL:?设 MODEL=<Qwen3.5-4B 目录或上一段的 hf checkpoint>}"
: "${TAU2_DATA_DIR:?设 TAU2_DATA_DIR=<tau2-bench>/data（官方判分口径 DB × COMMUNICATE）}"
: "${TAU2_USER_API_BASE:?设 TAU2_USER_API_BASE=<用户模拟器的 OpenAI 兼容接口>}"
export TAU2_USER_LLM=${TAU2_USER_LLM:-openai/deepseek-v4-flash}
export TAU2_USER_API_KEY=${TAU2_USER_API_KEY:?设 TAU2_USER_API_KEY}
export TAU2_MAX_STEPS=${TAU2_MAX_STEPS:-40}
MAX_ASSIST_TURNS=${MAX_ASSIST_TURNS:-$(( TAU2_MAX_STEPS * 3 / 4 ))}
export TAU2_TURN_MAX_TOKENS=${TAU2_TURN_MAX_TOKENS:-2048}
export TAU2_REPEAT_KILL=${TAU2_REPEAT_KILL:-8}
export TAU2_THREADS=${TAU2_THREADS:-256}           # 180 段对话全并发
export LITELLM_LOCAL_MODEL_COST_MAP=True
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

RUN=${RUN:-${ARM}_$(date +%m%d_%H%M)}
OUT=${OUT:-runs/$RUN}; mkdir -p $OUT
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_$RUN}; mkdir -p $RAY_TMPDIR
export RAY_ADDRESS=local RAY_prestart_worker_first_driver=0 PYTHONUNBUFFERED=1
export LOGURU_LEVEL=${LOGURU_LEVEL:-WARNING}
export VERL_ZMQ_TAG=${VERL_ZMQ_TAG:-$RUN-}
export TAU2_ROLLOUT_DIR=${TAU2_ROLLOUT_DIR:-$OUT/rollouts}     # 每条 rollout 全量落盘，事后可审计
DATA=${DATA:-data_rl}
GPUS=${GPUS:-0,1,2,3}; export CUDA_VISIBLE_DEVICES=$GPUS; NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
N=${N:-6}; BATCH=${BATCH:-30}; MINI=${MINI:-2}; LR=${LR:-1e-6}; STEPS=${STEPS:-3}; RESP_LEN=${RESP_LEN:-16384}
PROMPT_LEN=${PROMPT_LEN:-6144}; MAXLEN=${MAXLEN:-$((PROMPT_LEN + RESP_LEN))}
if [ "${SMOKE:-0}" = "1" ]; then BATCH=4; N=2; MINI=2; STEPS=2; fi

{ echo "date=$(date '+%F %T') ARM=$ARM ADV=$ADV RUN=$RUN GPUS=$GPUS MODEL=$MODEL"
  echo "N=$N BATCH=$BATCH MINI=$MINI LR=$LR STEPS=$STEPS RESP_LEN=$RESP_LEN PROMPT_LEN=$PROMPT_LEN"
  echo "TAU2_MAX_STEPS=$TAU2_MAX_STEPS MAX_ASSIST_TURNS=$MAX_ASSIST_TURNS TAU2_TURN_MAX_TOKENS=$TAU2_TURN_MAX_TOKENS TAU2_REPEAT_KILL=$TAU2_REPEAT_KILL"
  echo "TAU2_REWARD_V3=$TAU2_REWARD_V3 TAU2_PARTIAL=${TAU2_PARTIAL:-} TAU2_MASK_OVERLONG=$TAU2_MASK_OVERLONG TAU2_USER_LLM=$TAU2_USER_LLM"; } > $OUT/config.env

python3 -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=$ADV algorithm.use_kl_in_reward=False \
  data.train_files=$DATA/train.parquet data.val_files=$DATA/val.parquet data.return_raw_chat=True \
  data.train_batch_size=$BATCH data.max_prompt_length=$PROMPT_LEN data.max_response_length=$RESP_LEN data.truncation=error \
  actor_rollout_ref.model.path=$MODEL actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAXLEN actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  "actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}" \
  actor_rollout_ref.ref.strategy=fsdp2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=$N actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.prompt_length=$PROMPT_LEN actor_rollout_ref.rollout.response_length=$RESP_LEN \
  actor_rollout_ref.rollout.max_model_len=$MAXLEN actor_rollout_ref.rollout.max_num_batched_tokens=$MAXLEN \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSIST_TURNS \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=$HERE/agent_loop.yaml actor_rollout_ref.rollout.agent.default_agent_loop=tau2_agent \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=tau2_masked_adv.Tau2MaskedAgentLoopManager \
  actor_rollout_ref.rollout.agent.num_workers=$NGPU \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton" \
  trainer.logger=['console'] trainer.project_name=tau2_airline trainer.experiment_name=$RUN \
  trainer.n_gpus_per_node=$NGPU trainer.nnodes=1 trainer.default_local_dir=$OUT/ckpts \
  trainer.val_before_train=False trainer.test_freq=-1 \
  trainer.save_freq=${SAVE_FREQ:-1} trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-0} trainer.resume_mode=${RESUME:-auto} \
  trainer.total_training_steps=$STEPS trainer.rollout_data_dir=$OUT/rollout_dump \
  ray_kwargs.ray_init.num_cpus=${RAY_CPUS:-32} +ray_kwargs.ray_init.include_dashboard=False \
  trainer.use_v1=False "$@" 2>&1 | tee $OUT/train.log
