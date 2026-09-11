#!/usr/bin/env bash
# The run49/run50 trainer, exactly as launch_work.sh typed it into the tmux trainer window
# (recovered from the pane scrollback). Relaunch with: bash scripts/trainer49.sh
cd "$HOME/open-rl" || exit 1
export OPEN_RL_SNAPSHOT_DIR=/dev/shm/open-rl/peft OPEN_RL_CHECKPOINT_DIR=$HOME/open-rl-checkpoints OPEN_RL_TMP_DIR=$HOME/open-rl-tmp
CUDA_VISIBLE_DEVICES=0,1,2,3 FLA_TILELANG=1 REDIS_URL=redis://127.0.0.1:6379 OPEN_RL_FSDP_WORLD_SIZE=4 OPEN_RL_WORKER_PROBE_PORT=8090 \
OPEN_RL_TRAINER_BACKEND=automodel OPEN_RL_ENABLE_FFT=true OPEN_RL_AUTOMODEL_TP=1 OPEN_RL_AUTOMODEL_CP=4 OPEN_RL_AUTOMODEL_LORA_RANK=32 \
OPEN_RL_TIME_SLICING=off OPEN_RL_CONTROL_BACKEND=cpu:gloo,cuda:nccl \
SAMPLER_BASE_URLS=http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003 \
BASE_MODEL=Qwen/Qwen3.8-27B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPEN_RL_TRAIN_TOKEN_BUDGET=262144 \
OPEN_RL_ACTIVATION_CPU_OFFLOAD=1 OPEN_RL_OPTIM_CPU_STEP=1 OPEN_RL_LOG_CUDA_MEMORY=1 PYTHONPATH=$HOME/open-rl/src \
env -u NCCL_ENV_PLUGIN -u NCCL_CONF_FILE NCCL_NET=Socket \
$HOME/automodel/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m server.training_requests_processor |& tee -a $HOME/open-rl/artifacts/box-logs/trainer.log
