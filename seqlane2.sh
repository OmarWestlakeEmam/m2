cd /data/omar/m2 && source ~/envs/m2/bin/activate
n=$1; export CUDA_VISIBLE_DEVICES=$2; init=$3; shift 3
S="-s train.workers=12 data.pre=250 $*"
python -m core.seq train -c cfg/a.yaml $S train.init=$init --name $n &&
python -m core.seq val -c cfg/a.yaml $S --name $n --who deep &&
python -m core.seq val -c cfg/a.yaml $S --name $n --who broad &&
python -m core.seq predict -c cfg/a.yaml $S --name $n --track deep &&
python -m core.seq predict -c cfg/a.yaml $S --name $n --track broad &&
echo "SEQ $n DONE"
