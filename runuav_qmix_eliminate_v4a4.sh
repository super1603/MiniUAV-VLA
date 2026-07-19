#!/usr/bin/env bash
#
# Train a new Drones QMIX expert for the eliminate/attack task.
#
# Main differences from the old sweep:
#   - Only trains QMIX.
#   - Uses view_range=4 and attack_range=4.
#   - Keeps discovered-but-not-eliminated humans in each agent's target features
#     via obs_only_undiscovered_humans=False. This is important for eliminate mode:
#     after a human is discovered, drones still need to track it until elimination.
#
# Usage:
#   bash runuav_qmix_eliminate_v4a4.sh
#
# Common overrides:
#   GPU_ID=0 SEEDS="0 1 2" T_MAX=3000000 bash runuav_qmix_eliminate_v4a4.sh
#   USE_WANDB=False bash runuav_qmix_eliminate_v4a4.sh

set -uo pipefail

cd "$(dirname "$0")"

# ---- Resources ----
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ---- W&B logging ----
USE_WANDB="${USE_WANDB:-False}"                # True | False
WANDB_TEAM="${WANDB_TEAM:-}"
WANDB_PROJECT="${WANDB_PROJECT:-miniuav-vla}"
WANDB_MODE="${WANDB_MODE:-online}"             # online | offline
WANDB_NAME="${WANDB_NAME:-qmix_eliminate_v4a4_seed{seed}}"

# ---- Core experiment config ----
ALG="qmix"
MAP_SIZE="${MAP_SIZE:-30}"
DRONE_NUM="${DRONE_NUM:-4}"
HUMAN_NUM="${HUMAN_NUM:-6}"
VIEW_RANGE="${VIEW_RANGE:-4}"
ATTACK_RANGE="${ATTACK_RANGE:-4}"
EPISODE_LIMIT="${EPISODE_LIMIT:-120}"
SEEDS_STR="${SEEDS:-0}"
read -r -a SEEDS <<< "${SEEDS_STR}"

# Use the default long training budget unless overridden.
T_MAX="${T_MAX:-2050000}"
SAVE_MODEL_INTERVAL="${SAVE_MODEL_INTERVAL:-50000}"

# ---- Eliminate-task observation/reward overrides ----
MISSION_MODE="${MISSION_MODE:-eliminate}"
OBS_ONLY_UNDISCOVERED_HUMANS="${OBS_ONLY_UNDISCOVERED_HUMANS:-False}"
OBS_NUM_CLOSEST_HUMANS="${OBS_NUM_CLOSEST_HUMANS:-6}"
OBS_NUM_CLOSEST_OBSTACLES="${OBS_NUM_CLOSEST_OBSTACLES:-6}"
OBS_INCLUDE_TEAMMATES="${OBS_INCLUDE_TEAMMATES:-True}"
OBS_OBSTACLES_WITHIN_VIEW_ONLY="${OBS_OBSTACLES_WITHIN_VIEW_ONLY:-True}"

REWARD_NEW_TARGET="${REWARD_NEW_TARGET:-2.0}"
REWARD_NEW_ELIMINATION="${REWARD_NEW_ELIMINATION:-25.0}"
REWARD_SUCCESS="${REWARD_SUCCESS:-80.0}"
REWARD_APPROACH_COEF="${REWARD_APPROACH_COEF:-0.1}"
REWARD_STEP_PENALTY="${REWARD_STEP_PENALTY:--0.02}"
REWARD_COLLISION_PENALTY="${REWARD_COLLISION_PENALTY:--0.2}"
REWARD_VIEW_OVERLAP_PENALTY="${REWARD_VIEW_OVERLAP_PENALTY:--0.02}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-0.0}"

# ---- Evaluation/logging intervals ----
TEST_NEPI="${TEST_NEPI:-20}"
TEST_INTERVAL="${TEST_INTERVAL:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
RUNNER_LOG_INTERVAL="${RUNNER_LOG_INTERVAL:-500}"
LEARNER_LOG_INTERVAL="${LEARNER_LOG_INTERVAL:-500}"
PROGRESS_LOG_INTERVAL="${PROGRESS_LOG_INTERVAL:-500}"

# ---- Log dir + auto-tee to all.log ----
LOG_DIR="logs/qmix_eliminate_v4a4/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
ALL_LOG="${LOG_DIR}/all.log"
SUMMARY_TSV="${LOG_DIR}/summary.tsv"
printf "run_idx\ttag\tstart_time\tend_time\texit_code\n" > "${SUMMARY_TSV}"

exec > >(tee -a "${ALL_LOG}") 2>&1

total=${#SEEDS[@]}

echo "=========================================="
echo "Drones QMIX eliminate training"
echo "GPU:          ${CUDA_VISIBLE_DEVICES}"
echo "alg:          ${ALG}"
echo "map/drone/human/view/attack: ${MAP_SIZE}/${DRONE_NUM}/${HUMAN_NUM}/${VIEW_RANGE}/${ATTACK_RANGE}"
echo "mission:      ${MISSION_MODE}"
echo "obs_only_undiscovered_humans: ${OBS_ONLY_UNDISCOVERED_HUMANS}"
echo "reward_new_target:       ${REWARD_NEW_TARGET}"
echo "reward_new_elimination:  ${REWARD_NEW_ELIMINATION}"
echo "reward_success:          ${REWARD_SUCCESS}"
echo "reward_approach_coef:    ${REWARD_APPROACH_COEF}"
echo "reward_step_penalty:     ${REWARD_STEP_PENALTY}"
echo "seeds:        ${SEEDS[*]}"
echo "t_max:        ${T_MAX}"
echo "save interval:${SAVE_MODEL_INTERVAL}"
echo "wandb:        use=${USE_WANDB}, team=${WANDB_TEAM}, project=${WANDB_PROJECT}, mode=${WANDB_MODE}"
echo "total runs:   ${total}"
echo "log dir:      ${LOG_DIR}"
echo "started at:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

run_idx=0

for seed in "${SEEDS[@]}"; do
  run_idx=$((run_idx + 1))

  tag="qmix_elim_map${MAP_SIZE}_d${DRONE_NUM}_h${HUMAN_NUM}_v${VIEW_RANGE}_a${ATTACK_RANGE}_seed${seed}"
  log_file="${LOG_DIR}/${tag}.log"
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"

  echo ""
  echo ">>> [${run_idx}/${total}] ${tag}"
  echo ">>> log:   ${log_file}"
  echo ">>> start: ${start_time}"

  set +e
  python src/main.py \
    --config="${ALG}" \
    --env-config=drones \
    with \
    seed="${seed}" \
    t_max="${T_MAX}" \
    save_model=True \
    save_model_interval="${SAVE_MODEL_INTERVAL}" \
    test_nepisode="${TEST_NEPI}" \
    test_interval="${TEST_INTERVAL}" \
    log_interval="${LOG_INTERVAL}" \
    runner_log_interval="${RUNNER_LOG_INTERVAL}" \
    learner_log_interval="${LEARNER_LOG_INTERVAL}" \
    progress_log_interval="${PROGRESS_LOG_INTERVAL}" \
    use_wandb="${USE_WANDB}" \
    wandb_team="${WANDB_TEAM}" \
    wandb_project="${WANDB_PROJECT}" \
    wandb_mode="${WANDB_MODE}" \
    wandb_name="${WANDB_NAME}" \
    env_args.map_size="${MAP_SIZE}" \
    env_args.drone_num="${DRONE_NUM}" \
    env_args.human_num="${HUMAN_NUM}" \
    env_args.view_range="${VIEW_RANGE}" \
    env_args.attack_range="${ATTACK_RANGE}" \
    env_args.episode_limit="${EPISODE_LIMIT}" \
    env_args.mission_mode="${MISSION_MODE}" \
    env_args.obs_only_undiscovered_humans="${OBS_ONLY_UNDISCOVERED_HUMANS}" \
    env_args.obs_num_closest_humans="${OBS_NUM_CLOSEST_HUMANS}" \
    env_args.obs_num_closest_obstacles="${OBS_NUM_CLOSEST_OBSTACLES}" \
    env_args.obs_include_teammates="${OBS_INCLUDE_TEAMMATES}" \
    env_args.obs_obstacles_within_view_only="${OBS_OBSTACLES_WITHIN_VIEW_ONLY}" \
    env_args.reward_new_target="${REWARD_NEW_TARGET}" \
    env_args.reward_new_elimination="${REWARD_NEW_ELIMINATION}" \
    env_args.reward_success="${REWARD_SUCCESS}" \
    env_args.reward_approach_coef="${REWARD_APPROACH_COEF}" \
    env_args.reward_step_penalty="${REWARD_STEP_PENALTY}" \
    env_args.reward_collision_penalty="${REWARD_COLLISION_PENALTY}" \
    env_args.reward_view_overlap_penalty="${REWARD_VIEW_OVERLAP_PENALTY}" \
    env_args.reward_timeout="${REWARD_TIMEOUT}" \
    2>&1 | tee "${log_file}"
  exit_code=${PIPESTATUS[0]}
  set -e

  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${run_idx}" "${tag}" "${start_time}" "${end_time}" "${exit_code}" \
    >> "${SUMMARY_TSV}"

  if [[ ${exit_code} -ne 0 ]]; then
    echo ">>> [WARN] run ${run_idx} (${tag}) exited with code ${exit_code}; continuing."
  fi
done

echo ""
echo "=========================================="
echo "All ${total} run(s) finished at $(date '+%Y-%m-%d %H:%M:%S')."
echo "Logs dir:     ${LOG_DIR}"
echo "Summary TSV:  ${SUMMARY_TSV}"

failed=$(awk -F'\t' 'NR>1 && $5 != 0 {print $2}' "${SUMMARY_TSV}" || true)
if [[ -n "${failed}" ]]; then
  echo "[WARN] The following runs had non-zero exit codes:"
  echo "${failed}" | sed 's/^/  - /'
else
  echo "All runs exited with code 0."
fi
echo "=========================================="
