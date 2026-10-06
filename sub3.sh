cd /data/omar/m2 && source ~/envs/m2/bin/activate
export CUDA_VISIBLE_DEVICES=2
done_all() { grep -q "CTX q0 DONE" q0.log && grep -q "CTX q1 DONE" q1.log && grep -q "CTX z1 DONE" z.log && ! pgrep -f p1.sh >/dev/null && [ -f d/r/p1b/best.pt ]; }
until done_all; do
  pgrep -f "ctxlane2.sh|p1.sh" >/dev/null || { echo "a lane stopped early - check q0.log q1.log z.log p1.log"; exit 1; }
  sleep 120
done
X="d/o"
python -m core.calib -c cfg/a.yaml --who deep --members d/r/j1d/best.pt d/r/p1d/best.pt \
  --extra $X/xv_deep_c0.npz $X/xv_deep_c1.npz $X/xv_deep_cq0.npz $X/xv_deep_cq1.npz $X/xv_deep_cz0.npz $X/xv_deep_cz1.npz --out $X/calib3.json &&
python -m core.calib -c cfg/a.yaml --who broad --members d/r/j1b/best.pt d/r/p1b/best.pt d/r/r1b/best.pt \
  --extra $X/xv_broad_c1.npz $X/xv_broad_cq0.npz $X/xv_broad_cq1.npz $X/xv_broad_cz0.npz $X/xv_broad_cz1.npz --out $X/calib3.json &&
python -m core.infer -c cfg/a.yaml --track deep --members d/r/j1d/best.pt d/r/p1d/best.pt \
  --extra $X/x_deep_c0.npz $X/x_deep_c1.npz $X/x_deep_cq0.npz $X/x_deep_cq1.npz $X/x_deep_cz0.npz $X/x_deep_cz1.npz --calib $X/calib3.json --out $X/sub_deep_3.csv &&
python -m core.infer -c cfg/a.yaml --track broad --members d/r/j1b/best.pt d/r/p1b/best.pt d/r/r1b/best.pt \
  --extra $X/x_broad_c1.npz $X/x_broad_cq0.npz $X/x_broad_cq1.npz $X/x_broad_cz0.npz $X/x_broad_cz1.npz --calib $X/calib3.json --out $X/sub_broad_3.csv &&
echo "SUBMISSION 3 READY"
