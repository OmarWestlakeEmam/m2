#!/bin/bash
cd /data/omar/m2 && source ~/envs/m2/bin/activate
k=$1; export CUDA_VISIBLE_DEVICES=$2
C="-c cfg/a.yaml -s train.workers=12"
python -m core.train $C train.name=f$k train.fold=$k/2 train.epochs=12 &&
python -m core.ctx extract $C --name c$k --member d/r/f$k/best.pt --fold $k/2 &&
python -m core.ctx fit     $C --name c$k &&
python -m core.ctx val     $C --name c$k --who deep &&
python -m core.ctx val     $C --name c$k --who broad &&
python -m core.ctx predict $C --name c$k --track deep &&
python -m core.ctx predict $C --name c$k --track broad &&
echo "CTX $k DONE"
