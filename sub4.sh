cd /data/omar/m2 && source ~/envs/m2/bin/activate
export CUDA_VISIBLE_DEVICES=4
X=d/o
DS="cq0 cq1 cq2 cq3 cw0 cw1"; DA="c0 c1 cz0 cz1 $DS"; BA="c1 cz0 cz1 $DS"
xv() { for n in $2; do printf "$X/xv_$1_$n.npz "; done; }
xx() { for n in $2; do printf "$X/x_$1_$n.npz "; done; }
for v in all strong; do
  if [ $v = all ]; then D="$DA"; DM="d/r/j1d/best.pt d/r/p1d/best.pt"; B="$BA"; BM="d/r/j1b/best.pt d/r/p1b/best.pt d/r/r1b/best.pt";
  else D="$DS"; DM="d/r/p1d/best.pt"; B="$DS"; BM="d/r/p1b/best.pt d/r/r1b/best.pt"; fi
  echo "=== $v"
  python -m core.calib -c cfg/a.yaml --who deep  --members $DM --extra $(xv deep "$D")  --out $X/calib4_$v.json 2>&1 | grep ensemble
  python -m core.calib -c cfg/a.yaml --who broad --members $BM --extra $(xv broad "$B") --out $X/calib4_$v.json 2>&1 | grep ensemble
  python -m core.infer -c cfg/a.yaml --track deep  --members $DM --extra $(xx deep "$D")  --calib $X/calib4_$v.json --out $X/sub_deep_4$v.csv 2>&1 | grep wrote
  python -m core.infer -c cfg/a.yaml --track broad --members $BM --extra $(xx broad "$B") --calib $X/calib4_$v.json --out $X/sub_broad_4$v.csv 2>&1 | grep wrote
done
echo "SUBMISSION 4 READY"
