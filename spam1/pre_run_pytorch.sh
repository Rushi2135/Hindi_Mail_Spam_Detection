#!/bin/bash
#SBATCH --job-name=runner_job
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   
#SBATCH --output=runner_job.out 
#SBATCH --time=2-00:00:00 

python DNN_sRNN_pytorch_code_00.py
