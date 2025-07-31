CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --task_suite_name libero_spatial


WANDB_MODE=offline torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/checkpoints/openvla-7b \
  --data_root_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/modified_libero_rlds/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/experiments/finetune/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state


python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/mobile_aloha/aloha_static_battery_raw/ \
  --out_base_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/tmp/ \
  --percent_val 0.05  > logs_aloha/aloha_static_battery_raw.log 2>&1

WANDB_MODE=offline torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/checkpoints/openvla-7b \
  --data_root_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/tensorflow_datasets/ \
  --dataset_name aloha1_static \
  --run_root_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/experiments/finetune/aloha_static0604/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 200005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "eternity" \
  --wandb_project "oft0604" \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film


# raw aloha
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_modi.py \
  --vla_path /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/checkpoints/openvla-7b \
  --data_root_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/tensorflow_datasets/ \
  --dataset_name aloha1_static \
  --run_root_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/experiments/finetune/aloha_static0604/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 200005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film


# aloha-tape lerobot
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_modi.py \
  --vla_path /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/checkpoints/openvla-7b \
  --data_root_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/tensorflow_datasets/ \
  --dataset_name lerobot_dataset \
  --run_root_dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/zcd/openvla-oft/experiments/finetune/aloha_tape_lerobot0608/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 110005 \
  --use_val_set False \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film

# agibot
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_modi.py \
  --vla_path /inspire/hdd/project/robot-decision/cengchendong-CZXS25230112/openvla-oft/checkpoints/openvla-7b \
  --data_root_dir /inspire/hdd/project/robot-decision/public/Unidomain/datasets/tmprlds/ \
  --dataset_name agibot_pull \
  --run_root_dir /inspire/hdd/project/robot-decision/cengchendong-CZXS25230112/openvla-oft/experiments/finetune/agibot_pull0729/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 60000 \
  --max_steps 150005 \
  --use_val_set True \
  --val_freq 10005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--left_right_wrist_imgs--proprio_state--film \
  --resume False \
  --resume_step 50000 \
  --merge_lora_during_training False

#--val_freq 10005 \ --save_freq 10000
# agibot test
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_modi.py \
  --vla_path experiments/finetune/agibot_pour0729/openvla-7b+agibot_pour-80000_chkpt \
  --data_root_dir /inspire/hdd/project/robot-decision/public/Unidomain/datasets/tmprlds/ \
  --dataset_name agibot_pour_ice \
  --run_root_dir /inspire/hdd/project/robot-decision/cengchendong-CZXS25230112/openvla-oft/experiments/finetune/agibot_pour_ice0731/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 30000 \
  --max_steps 60008 \
  --use_val_set True \
  --val_freq 5000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note pour_ice_from_pour80000 \
  --resume False \
  --resume_step 50000 \
  --merge_lora_during_training False

# merge
python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint ./checkpoints/openvla-7b \
        --lora_finetuned_checkpoint_dir /inspire/hdd/project/robot-decision/cengchendong-CZXS25230112/openvla-oft/experiments/finetune/agibot_pour0729/openvla-7b+agibot_pour+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--25_acts_chunk--continuous_acts--L1_regression--left_right_wrist_imgs--proprio_state--film--100000_chkpt