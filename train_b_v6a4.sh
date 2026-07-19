#!/usr/bin/env bash
# Path B for v6a4_tuned data: SigLIP2-base-patch16-224 vision encoder + 196 real image tokens.
#
# Two-step pipeline:
#   1) SFT path B on v6a4_tuned data (checkpoint with _p16img196 suffix)
#   2) Closed-loop eval on the path B checkpoint
#
# Prerequisites: see train_b_v4a4.sh.
set -uo pipefail

DEVICE="${DEVICE:-cuda:0}"

NEW_CKPT="VLA/models/sft_vlm_drones_qmix_elim_v6a4_tuned_10k_p16img196_768.pth"
NEW_EVAL="VLA/eval_runs/drones_qmix_elim_v6a4_tuned_10k_p16img196_closed_loop_eval.jsonl"

echo "==> Step 1/2: SFT path B on v6a4_tuned (p16-224 + 196 image tokens)"
conda run --no-capture-output -n minivla python VLA/SFT/train_drones_sft.py \
  --data_path VLA/data/drones_qmix_elim_v6a4_tuned_10k/drones_qmix_elim_v6a4_tuned_train_alpaca.jsonl \
  --val_data_path VLA/data/drones_qmix_elim_v6a4_tuned_10k/drones_qmix_elim_v6a4_tuned_val_alpaca.jsonl \
  --out_dir VLA/models \
  --save_name sft_vlm_drones_qmix_elim_v6a4_tuned_10k_p16img196_768.pth \
  --pretrained_path out/pretrain_vlm_768.pth \
  --vision_model_path model/siglip2-base-p16-224 \
  --image_token_len 196 \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --action_loss_weight 1.0 \
  --device "${DEVICE}" \
  --freeze_llm 1

echo "==> Step 2/2: Closed-loop eval on path B checkpoint"
conda run --no-capture-output -n minivla python VLA/close_loop_drones_evaluation.py \
  --checkpoint_path "${NEW_CKPT}" \
  --vision_model_path model/siglip2-base-p16-224 \
  --image_token_len 196 \
  --episodes 100 \
  --mission-mode eliminate \
  --view-range 6 \
  --attack-range 4 \
  --video_dir VLA/eval_runs/videos_qmix_elim_v6a4_tuned_10k_p16img196 \
  --video_episodes 5 \
  --save_traces \
  --output_path "${NEW_EVAL}" \
  --device "${DEVICE}"

echo
echo "Done. Path B result: ${NEW_EVAL}"
