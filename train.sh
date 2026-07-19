conda run --no-capture-output -n minivla python VLA/envs/generate_drones_qmix_dataset.py \
  --checkpoint-path results/models/qmix_seed0_drones_2026-05-17_11-16-03 \
  --out-dir VLA/data/drones_qmix_elim_v4a4_10k \
  --prefix drones_qmix_elim_v4a4 \
  --num-transitions 10000 \
  --mission-mode eliminate \
  --success-only \
  --episode-log-interval 1 \
  --device cuda:0

#评估qmix专家数据集的表现
conda run --no-capture-output -n minivla python VLA/envs/render_drones_qmix_rollouts.py \
  --checkpoint-path results/models/qmix_seed0_drones_2026-05-17_11-16-03 \
  --output-dir VLA/eval_runs/qmix_elim_v4a4_20eps \
  --episodes 20 \
  --seed 2026 \
  --device cuda:0 \
  --draw-attack-range \
  --video-fps 6



conda run --no-capture-output -n minivla python VLA/SFT/train_drones_sft.py \
  --data_path VLA/data/drones_qmix_elim_v4a4_10k/drones_qmix_elim_v4a4_train_alpaca.jsonl \
  --val_data_path VLA/data/drones_qmix_elim_v4a4_10k/drones_qmix_elim_v4a4_val_alpaca.jsonl \
  --out_dir VLA/models \
  --save_name sft_vlm_drones_qmix_elim_v4a4_10k_promptpos_768.pth \
  --pretrained_path out/pretrain_vlm_768.pth \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --action_loss_weight 1.0 \
  --device cuda:0 \
  --freeze_llm 1

conda run --no-capture-output -n minivla python VLA/close_loop_drones_evaluation.py \
  --checkpoint_path VLA/models/sft_vlm_drones_qmix_elim_v4a4_10k_promptpos_768.pth \
  --episodes 100 \
  --mission-mode eliminate \
  --view-range 4 \
  --attack-range 4 \
  --video_dir VLA/eval_runs/videos_qmix_elim_v4a4_10k \
  --video_episodes 5 \
  --save_traces \
  --output_path VLA/eval_runs/drones_qmix_elim_v4a4_10k_closed_loop_eval.jsonl \
  --device cuda:0
