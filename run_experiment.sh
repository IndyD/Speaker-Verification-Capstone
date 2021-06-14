#!/bin/bash
#SBATCH -J speaker_verification_id
#SBATCH --output=output/log_%j.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=250G
#SBATCH -t 5000
#SBATCH -D /work/users/idhillon/Speaker-Verification-Capstone


for config in "$@"; do
  python3.9 generate_data.py "$config"
  python3 run_model.py "$config"

done