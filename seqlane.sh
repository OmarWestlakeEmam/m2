#!/bin/bash
# usage: bash seqlane.sh NAME GPU INIT_CKPT [extra overrides]
cd /data/omar/m2 && source ~/envs/m2/bin/activate
n=$1; export CUDA_VISIBLE_DEVICES=$2; init=$3; shift 3
C="-c cfg/a.yaml -s train.workers=12 data.pre=250"
python -m core.seq train $C train.init=$init "$@" --name $n &&
for w in deep broad; do python -m core.seq val $C --name $n --who $w && python -m core.seq predict $C --name $n --track $w || exit 1; done &&
echo "SEQ $n DONE"
