#!/bin/bash
cd /data/omar/m2 && source ~/envs/m2/bin/activate
member() {
  g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python -m core.train -c cfg/a.yaml        -s train.name=$n    train.workers=12 "$@" &&
  CUDA_VISIBLE_DEVICES=$g python -m core.train -c cfg/ft_deep.yaml  -s train.name=${n}d train.workers=12 train.init=d/r/$n/best_deep.pt  "$@" &&
  CUDA_VISIBLE_DEVICES=$g python -m core.train -c cfg/ft_broad.yaml -s train.name=${n}b train.workers=12 train.init=d/r/$n/best_broad.pt "$@"
}
case $1 in
  3) member 3 v1 train.seed=1 norm.mode=robust ;;
  4) member 4 v2 train.seed=2 model.depth=0 ;;
  6) member 6 v3 train.seed=3 loss.w_mean=0 ;;
  2) while kill -0 252531 2>/dev/null; do sleep 60; done
     CUDA_VISIBLE_DEVICES=2 python -m core.train -c cfg/ft_deep.yaml  -s train.name=j1d train.workers=12 train.init=d/r/j1/best.pt &&
     CUDA_VISIBLE_DEVICES=2 python -m core.train -c cfg/ft_broad.yaml -s train.name=j1b train.workers=12 train.init=d/r/j1/best.pt
     member 2 v4 train.seed=4 ;;
esac
