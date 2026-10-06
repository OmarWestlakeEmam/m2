#!/usr/bin/env bash
# Train up to 4 different members at once, one per GPU: joint model, then deep and broad fine-tunes.
# usage: bash run4.sh "2 3 4 6" a      (GPU ids, round name)
set -u
GPUS=(${1:-2 3 4 6}); TAG=${2:-a}
R=$(python -c "from core.common import load_cfg; print(load_cfg('cfg/a.yaml')['paths']['runs'])")
mkdir -p "$R"
V=("train.seed=0" "train.seed=1 norm.mode=robust" "train.seed=2 model.depth=0" "train.seed=3 loss.w_mean=0")
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}; n=$TAG$i; v=${V[$i]}
  (
    export CUDA_VISIBLE_DEVICES=$g
    python -m core.train -c cfg/a.yaml       -s train.name=$n  train.workers=12 $v &&
    python -m core.train -c cfg/ft_deep.yaml  -s train.name=${n}d train.init=$R/$n/best.pt train.workers=12 $v &&
    python -m core.train -c cfg/ft_broad.yaml -s train.name=${n}b train.init=$R/$n/best.pt train.workers=12 $v
  ) > "$R/log_$n.txt" 2>&1 &
  echo "GPU $g -> $n ($v), log: $R/log_$n.txt"
done
