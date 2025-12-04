#!/bin/bash
#SBATCH -J mibench_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00
#SBATCH -p batch_ugrad
#SBATCH --output=logs/mibench_eval_%j.out
#SBATCH --error=logs/mibench_eval_%j.err

set -e
mkdir -p logs

echo "=========================================="
echo "MiBench 벤치마크 평가 시작"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "시작 시간: $(date)"
echo "=========================================="

source /data/sofusion20/anaconda3/etc/profile.d/conda.sh
conda activate llvm-opt

cd /data/sofusion20/mibench_eval

# Step 1
if [ ! -d "mibench" ]; then
    bash 01_setup_mibench.sh
fi

# Step 2
if [ ! -f "prompts/mibench_prompts.jsonl" ]; then
    python 02_generate_prompts.py
fi

# Step 3
MODEL_PATH="bigcode/starcoderbase-1b"
LORA_PATH=""  # LoRA 사용시 경로 지정
if [ -n "$LORA_PATH" ]; then
    python 03_run_inference.py --model "$MODEL_PATH" --lora "$LORA_PATH"
else
    python 03_run_inference.py --model "$MODEL_PATH"
fi

# Step 4
python 04_evaluate_binary_size.py

# Step 5
python 05_generate_report.py

echo "평가 완료: $(date)"
```

---

**8. requirements.txt**
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
accelerate>=0.25.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
jsonlines>=4.0.0