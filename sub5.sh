cd /data/omar/m2 && source ~/envs/m2/bin/activate
export CUDA_VISIBLE_DEVICES=4
until grep -q "SEQ s1 DONE" s1.log; do sleep 30; done
X=d/o; DS="cq0 cq1 cq2 cq3 cw0 cw1"
rep() { for i in $(seq $3); do printf "$X/$1_s1.npz "; done; }
ctx() { for n in $DS; do printf "$X/$1_deep_$n.npz "; done; }
for k in 3 6 12; do
  echo "=== deep k$k"
  python -m core.calib -c cfg/a.yaml --who deep --members d/r/p1d/best.pt --extra $(ctx xv) $(rep xv_deep x $k) --out $X/c5d_$k.json 2>&1 | grep -E "ensemble"
  python -m core.infer -c cfg/a.yaml --track deep --members d/r/p1d/best.pt --extra $(ctx x) $(rep x_deep x $k) --calib $X/c5d_$k.json --out $X/sub_deep_5k$k.csv 2>&1 | grep wrote
done
for k in 2 4 8; do
  echo "=== broad k$k"
  python -m core.calib -c cfg/a.yaml --who broad --members d/r/p1b/best.pt d/r/r1b/best.pt --extra $(rep xv_broad x $k) --out $X/c5b_$k.json 2>&1 | grep ensemble
  python -m core.infer -c cfg/a.yaml --track broad --members d/r/p1b/best.pt d/r/r1b/best.pt --extra $(rep x_broad x $k) --calib $X/c5b_$k.json --out $X/sub_broad_5k$k.csv 2>&1 | grep wrote
done
echo "SUBMISSION 5 READY"
