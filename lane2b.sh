#!/bin/bash
cd /data/omar/m2 && source ~/envs/m2/bin/activate
export CUDA_VISIBLE_DEVICES=2
T="python -m core.train"
$T -c cfg/ft_deep.yaml  -s train.name=j1d train.workers=12 train.init=d/r/j1/best.pt &&
$T -c cfg/ft_broad.yaml -s train.name=j1b train.workers=12 train.init=d/r/j1/best.pt &&
python -m core.calib -c cfg/a.yaml --who deep  --members d/r/j1d/best.pt &&
python -m core.calib -c cfg/a.yaml --who broad --members d/r/j1b/best.pt &&
python -m core.infer -c cfg/a.yaml --track deep  --members d/r/j1d/best.pt --calib d/o/calib_deep.json  --out d/o/sub_deep_j1.csv &&
python -m core.infer -c cfg/a.yaml --track broad --members d/r/j1b/best.pt --calib d/o/calib_broad.json --out d/o/sub_broad_j1.csv &&
echo "SUBMISSION FILES READY"
M="train.seed=4 model.drop=0.2 model.droppath=0.2"
$T -c cfg/a.yaml -s train.name=r1 train.workers=12 train.epochs=12 train.subj_drop=0.3 train.lr_adapter=3.0e-4 $M &&
$T -c cfg/ft_deep.yaml  -s train.name=r1d train.workers=12 train.init=d/r/r1/best_deep.pt  $M &&
$T -c cfg/ft_broad.yaml -s train.name=r1b train.workers=12 train.init=d/r/r1/best_broad.pt $M
