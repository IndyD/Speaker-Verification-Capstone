#!/bin/bash
#SBATCH -J speaker_verification_id
#SBATCH --output=output/log_%j.txt
#SBATCH -p high-mem-1 --mem=100G
#SBATCH -t 800
#SBATCH -D /work/users/idhillon/Speaker-Verification-Capstone


for config in "$@"; do
  echo "<<<< RUNNING CONFIG FILE: $config";
  python3.9 generate_spectrograms.py "$config";
  python3 generate_datasets.py "$config";
  python3 run_model.py "$config";

done
