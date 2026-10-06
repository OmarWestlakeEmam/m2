cd /data/omar/m2 && source ~/envs/m2/bin/activate
python -c "from core import ho; from core.common import load_cfg; c=load_cfg('cfg/a.yaml',[]); ho.load(c,'deep',250); ho.load(c,'broad',250)" &&
{ bash ctxlane2.sh w0 0 3 data.pre=250 > w0.log 2>&1 &
  bash ctxlane2.sh w1 1 6 data.pre=250 > w1.log 2>&1 &
  wait; }
