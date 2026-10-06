#!/bin/bash
cd /data/omar/m2 && source ~/envs/m2/bin/activate
n=$1; k=$2; export CUDA_VISIBLE_DEVICES=$3; shift 3
C="-c cfg/a.yaml -s train.workers=12"
python -m core.train $C train.name=f$n train.fold=$k/2 train.epochs=12 "$@" &&
python -m core.ctx extract $C --name c$n --member d/r/f$n/best.pt --fold $k/2 &&
python -m core.ctx fit     $C --name c$n &&
for w in deep broad; do python -m core.ctx val $C --name c$n --who $w && python -m core.ctx predict $C --name c$n --track $w || exit 1; done &&
echo "CTX $n DONE"
