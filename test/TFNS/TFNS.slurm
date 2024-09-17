#!/bin/bash
#SBATCH -J TFNS-test       # 指定作业名
#SBATCH -o LLama2-13B-16bit-fin-r4.out     # 屏幕上的输出文件重定向到 ***.out
#SBATCH -N 1                        # 作业申请 1 个节点
#SBATCH --cpus-per-task=64          # 单任务使用的 CPU 核心数为
#SBATCH --gres=gpu:1                # 单个节点使用 N 块 GPU 卡

python TFNS.py
