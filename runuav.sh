#!/usr/bin/env bash
#
# Batch sweep for epymarl_battleuav / drones.
# Runs all combinations sequentially on a single GPU (GPU 1).
#
# Usage (just run it inside screen/tmux; logs are handled automatically):
#   bash runuav.sh

set -uo pipefail

cd "$(dirname "$0")"

# ---- Resources ----
export CUDA_VISIBLE_DEVICES=1

# ---- W&B logging (override src/config/default.yaml at runtime) ----
USE_WANDB="${USE_WANDB:-False}"                # True | False
WANDB_TEAM="${WANDB_TEAM:-}"
WANDB_PROJECT="${WANDB_PROJECT:-miniuav-vla}"
WANDB_MODE="${WANDB_MODE:-online}"             # online | offline

# ---- Sweep config (edit these arrays directly) ----
ALGS=(maddpg coma ia2c ippo maa2c vdn qmix ippo mappo)
SEEDS=(0)
MAP_SIZES=(30 50 70)

# Four paired scenarios (indexed together).
#   index 0: drone=4  human=6  view=6  attack=3   (baseline)
#   index 1: drone=6  human=6  view=6  attack=3   (more drones, full sensing)
#   index 2: drone=4  human=6  view=4  attack=2   (baseline, restricted sensing)
#   index 3: drone=6  human=6  view=4  attack=2   (more drones, restricted sensing)
SCENARIO_DRONES=( 4 6 4 6 )
SCENARIO_HUMANS=( 6 6 6 6 )
SCENARIO_VIEWS=(  6 6 4 4 )
SCENARIO_ATTACKS=(3 3 2 2 )

# Smoke-test first: set T_MAX small so every combination can be validated quickly.
# After the 108-run dry-run succeeds, change T_MAX to "" to use the env yaml default (2050000).
T_MAX=1000000
# T_MAX=""

# ---- Log dir + auto-tee to all.log ----
LOG_DIR="logs/sweeps/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
ALL_LOG="${LOG_DIR}/all.log"
SUMMARY_TSV="${LOG_DIR}/summary.tsv"
printf "run_idx\ttag\tstart_time\tend_time\texit_code\n" > "${SUMMARY_TSV}"

# Redirect stdout + stderr of this script to both terminal and all.log.
exec > >(tee -a "${ALL_LOG}") 2>&1

echo "Logging everything to: ${ALL_LOG}"

# ---- Sanity checks ----
scenario_len=${#SCENARIO_DRONES[@]}
if [[ ${#SCENARIO_HUMANS[@]}  -ne ${scenario_len} ]] || \
   [[ ${#SCENARIO_VIEWS[@]}   -ne ${scenario_len} ]] || \
   [[ ${#SCENARIO_ATTACKS[@]} -ne ${scenario_len} ]]; then
  echo "[ERROR] SCENARIO_* arrays must all have the same length." >&2
  exit 1
fi

total=$(( ${#MAP_SIZES[@]} * scenario_len * ${#ALGS[@]} * ${#SEEDS[@]} ))

echo "=========================================="
echo "GPU:          ${CUDA_VISIBLE_DEVICES}"
echo "Algorithms:   ${ALGS[*]}"
echo "map_sizes:    ${MAP_SIZES[*]}"
echo "Scenarios (drone/human/view/attack):"
for si in "${!SCENARIO_DRONES[@]}"; do
  printf "              #%d: d=%s  h=%s  v=%s  a=%s\n" \
    "$si" "${SCENARIO_DRONES[$si]}" "${SCENARIO_HUMANS[$si]}" \
    "${SCENARIO_VIEWS[$si]}" "${SCENARIO_ATTACKS[$si]}"
done
echo "seeds:        ${SEEDS[*]}"
echo "t_max:        ${T_MAX:-<use env yaml default>}"
echo "wandb:        use=${USE_WANDB}, team=${WANDB_TEAM}, project=${WANDB_PROJECT}, mode=${WANDB_MODE}"
echo "Total runs:   ${total}"
echo "Log dir:      ${LOG_DIR}"
echo "  - all.log       : combined log of the whole sweep"
echo "  - <tag>.log     : per-run detailed log"
echo "  - summary.tsv   : progress table with exit codes"
echo "Started at:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

run_idx=0

for map_size in "${MAP_SIZES[@]}"; do
  for si in "${!SCENARIO_DRONES[@]}"; do
    drone="${SCENARIO_DRONES[$si]}"
    human="${SCENARIO_HUMANS[$si]}"
    view="${SCENARIO_VIEWS[$si]}"
    attack="${SCENARIO_ATTACKS[$si]}"
    for alg in "${ALGS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_idx=$(( run_idx + 1 ))

        tag="${alg}_map${map_size}_d${drone}_h${human}_v${view}_a${attack}_seed${seed}"
        log_file="${LOG_DIR}/${tag}.log"
        start_time="$(date '+%Y-%m-%d %H:%M:%S')"

        echo ""
        echo ">>> [${run_idx}/${total}] ${tag}"
        echo ">>> log:   ${log_file}"
        echo ">>> start: ${start_time}"

        extra_args=()
        if [[ -n "${T_MAX}" ]]; then
          extra_args+=("t_max=${T_MAX}")
        fi

        # W&B overrides (sacred: any default.yaml key can be overridden here)
        extra_args+=(
          "use_wandb=${USE_WANDB}"
          "wandb_team=${WANDB_TEAM}"
          "wandb_project=${WANDB_PROJECT}"
          "wandb_mode=${WANDB_MODE}"
        )

        set +e
        python src/main.py \
          --config="${alg}" \
          --env-config=drones \
          with \
          seed="${seed}" \
          env_args.map_size="${map_size}" \
          env_args.drone_num="${drone}" \
          env_args.human_num="${human}" \
          env_args.view_range="${view}" \
          env_args.attack_range="${attack}" \
          "${extra_args[@]}" \
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
    done
  done
done

echo ""
echo "=========================================="
echo "All ${total} runs finished at $(date '+%Y-%m-%d %H:%M:%S')."
echo "Logs dir:     ${LOG_DIR}"
echo "Summary TSV:  ${SUMMARY_TSV}"

# Quick report
failed=$(awk -F'\t' 'NR>1 && $5 != 0 {print $2}' "${SUMMARY_TSV}" || true)
if [[ -n "${failed}" ]]; then
  echo "[WARN] The following runs had non-zero exit codes:"
  echo "${failed}" | sed 's/^/  - /'
else
  echo "All runs exited with code 0."
fi
echo "=========================================="
