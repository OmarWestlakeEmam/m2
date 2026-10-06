cd /data/omar/m2 && source ~/envs/m2/bin/activate
export CUDA_VISIBLE_DEVICES=4
until grep -q "CTX 0 DONE" ctx0.log && grep -q "CTX 1 DONE" ctx1.log; do
  pgrep -f ctxlane.sh >/dev/null || { echo "context lanes stopped early - check ctx0.log/ctx1.log"; exit 1; }
  sleep 60
done
python -m core.calib -c cfg/a.yaml --who deep  --members d/r/j1d/best.pt --extra d/o/xv_deep_c0.npz d/o/xv_deep_c1.npz --out d/o/calib_ctx.json &&
python -m core.calib -c cfg/a.yaml --who broad --members d/r/j1b/best.pt --extra d/o/xv_broad_c0.npz d/o/xv_broad_c1.npz --out d/o/calib_ctx.json &&
python -m core.infer -c cfg/a.yaml --track deep  --members d/r/j1d/best.pt --extra d/o/x_deep_c0.npz d/o/x_deep_c1.npz --calib d/o/calib_ctx.json --out d/o/sub_deep_ctx.csv &&
python -m core.infer -c cfg/a.yaml --track broad --members d/r/j1b/best.pt --extra d/o/x_broad_c0.npz d/o/x_broad_c1.npz --calib d/o/calib_ctx.json --out d/o/sub_broad_ctx.csv &&
echo "SUBMISSION 2 READY"
