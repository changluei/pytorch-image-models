#!/bin/bash
#SBATCH --job-name=sleep        # 任务名
#SBATCH --output=output.log        # 输出文件
#SBATCH --error=error.log          # 错误文件
#SBATCH --ntasks=1                 # 任务数
#SBATCH --cpus-per-task=4          # 每个任务的CPU核数，没卡不超过6个
#SBATCH --time=06:00:00            # 最长运行时间
#SBATCH --partition=gpu            # 分区名
#SBATCH --gres=gpu:1               # GPU资源请求

sleep infinity