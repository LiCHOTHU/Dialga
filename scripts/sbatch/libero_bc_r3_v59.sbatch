#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=08:00:00 --mem-per-gpu=80G --job-name=libero_bc_r3
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
cd /storage/home/hcoda1/8/lwang831/workspace/imitation
source ~/.bashrc; conda activate porygon
export MUJOCO_GL=egl EGL_DEVICE_ID=0
CK=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/libero_v59/v5_best.pt
DP=/storage/scratch1/8/lwang831/imitation/data
OP=/storage/scratch1/8/lwang831/imitation/cold_start
EXP=libero_bc_r3_v59
echo "=== TRAIN (r3 = z_dyn only, libero_v59) ==="
python train.py --config-name=train task=libero algo=fm_policy_r3     algo.encoder.dialga_ckpt_path=$CK algo.chunk_size=16 algo.action_horizon=8     training.n_epochs=30 rollout.enabled=false logging.mode=disabled     exp_name=$EXP data_prefix=$DP output_prefix=$OP     hydra.run.dir=/storage/scratch1/8/lwang831/imitation/hydra/$EXP
CKPT=$OP/libero/libero_90/$EXP/multitask_model_latest.pth
echo "=== ROLLOUT (tasks 0..9, 5 rollouts each) ==="
for T in $(seq 0 9); do
  python eval_libero_bc.py algo=fm_policy_r3     algo.encoder.dialga_ckpt_path=$CK cold_start_checkpoint=$CKPT     task_index=$T rl.eval_rollouts_per_env=5 logging.mode=disabled     exp_name=eval_${EXP}_t$T     hydra.run.dir=/storage/scratch1/8/lwang831/eval_$EXP/hydra/t$T || echo "task $T failed"
done
echo "BC_R3_DONE $(date)"
