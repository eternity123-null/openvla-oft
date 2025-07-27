#!/bin/bash

# 数据子目录列表
subdirs=(
  "aloha_mobile_cabinet"
  "aloha_mobile_chair_raw"
  "aloha_mobile_elevator_raw"
  "aloha_mobile_shrimp_raw"
  "aloha_mobile_wash_pan"
  "aloha_mobile_wipe_wine_raw"
  "aloha_static_battery_raw"
  "aloha_static_candy_raw"
  "aloha_static_cups_open_raw"
  "aloha_static_pro_pencil_raw"
  "aloha_static_tape_raw"
  "aloha_static_thread_velcro_raw"
  "aloha_static_ziploc_slide_raw"
)

# 路径设置
dataset_root="/inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/mobile_aloha"
out_root="/inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/aloha1_preprocessed/"
script_path="experiments/robot/aloha/preprocess_split_aloha_data.py"
max_jobs=20
# 当前运行的job数量
current_jobs=0

# 日志文件存储目录
log_dir="./logs_aloha"
mkdir -p "$log_dir"



for subdir in "${subdirs[@]}"; do

    # 构建日志文件路径
  log_file="${log_dir}/${subdir}.log"

  echo "Starting processing: $subdir"
  echo "Log file: $log_file"

  python "$script_path" \
    --dataset_path "${dataset_root}/${subdir}/" \
    --out_base_dir "$out_root" \
    --percent_val 0.05 > "$log_file" 2>&1 &

  ((current_jobs+=1))

  # 如果达到并发限制，等待任一任务结束
  if (( current_jobs >= max_jobs )); then
    wait -n
    ((current_jobs-=1))
  fi
done

# 等待所有剩余任务完成
wait
echo "All preprocessing tasks completed!"
