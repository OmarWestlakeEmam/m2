cd /data/omar/m2 && source ~/envs/m2/bin/activate
export CUDA_VISIBLE_DEVICES=2
P="data.pre=250"
python -m core.train -c cfg/a.yaml        -s train.name=p2  train.workers=12 train.epochs=12 $P &&
python -m core.train -c cfg/ft_deep.yaml  -s train.name=p2d train.workers=12 train.init=d/r/p2/best_deep.pt  $P &&
python -m core.train -c cfg/ft_broad.yaml -s train.name=p2b train.workers=12 train.init=d/r/p2/best_broad.pt $P
