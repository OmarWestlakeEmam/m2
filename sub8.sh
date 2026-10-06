cd /data/omar/m2 && source ~/envs/m2/bin/activate
export CUDA_VISIBLE_DEVICES=4
for n in s2 s9 s10 s11; do until grep -q "SEQ $n DONE" $n.log; do grep -q Traceback $n.log && { echo "$n failed"; exit 1; }; sleep 60; done; done
X=d/o; MS="s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11"; DS="cq0 cq1 cq2 cq3 cw0 cw1"
rep() { for m in $MS; do for i in $(seq $3); do printf "$X/$1_$2_$m.npz "; done; done; }
ctx() { for n in $DS; do printf "$X/$1_deep_$n.npz "; done; }
for v in deep broad; do
  echo "=== single-model check $v"
  for m in s2 s6 s7 s8 s9 s10 s11; do python -m core.calib -c cfg/a.yaml --who $v --members d/r/p1d/best.pt --extra $X/xv_${v}_$m.npz --out /tmp/junk.json 2>&1 | grep "extra"; done
done
for k in 1 2; do
  echo "=== deep k$k"
  python -m core.calib -c cfg/a.yaml --who deep --members d/r/p1d/best.pt --extra $(ctx xv) $(rep xv deep $k) --out $X/c8d_$k.json 2>&1 | grep ensemble
  python -m core.infer -c cfg/a.yaml --track deep --members d/r/p1d/best.pt --extra $(ctx x) $(rep x deep $k) --calib $X/c8d_$k.json --out $X/sub_deep_8k$k.csv 2>&1 | grep wrote
  echo "=== broad k$k"
  python -m core.calib -c cfg/a.yaml --who broad --members d/r/p1b/best.pt d/r/r1b/best.pt --extra $(rep xv broad $k) --out $X/c8b_$k.json 2>&1 | grep ensemble
  python -m core.infer -c cfg/a.yaml --track broad --members d/r/p1b/best.pt d/r/r1b/best.pt --extra $(rep x broad $k) --calib $X/c8b_$k.json --out $X/sub_broad_8k$k.csv 2>&1 | grep wrote
done
echo "SUBMISSION 8 READY"
