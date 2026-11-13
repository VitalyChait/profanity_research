#!/usr/bin/env bash
#SBATCH --job-name=minorsafe-grid
#SBATCH --output=results/slurm_%A_%a.out
#SBATCH --error=results/slurm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-4

set -euo pipefail

PROFILE="${PROFILE:-gpu_80gb}"
EXPERIMENTS=("main_grid" "recomposition" "subculture_audit" "ramp_up" "quant_grid")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
EXP="${EXPERIMENTS[${IDX}]}"

echo "[slurm] Running experiment=${EXP} with profile=${PROFILE} on $(hostname)"
bash scripts/run_phase.sh "${EXP}" "${PROFILE}"

