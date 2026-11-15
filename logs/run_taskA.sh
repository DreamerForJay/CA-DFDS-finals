#!/bin/bash
# 任務 A：抽取特徵（本機 RTX 5080 主機）
cd /d/Carla-0.10.0-Win64-Shipping/CA-DFDS-finals
PY=./.venv-cadfds/Scripts/python.exe
R=d:/Carla-0.10.0-Win64-Shipping/Carla-0.10.0-Win64-Shipping/datas
DS=Test_dataset_Focus1500cm_f1.4_f22.0
SHARED=$R/firetruck_black/${DS}_mask
W=16
for c in white gray black red green blue; do
  [ -f features/cal_$c.npz ] || $PY extract_features.py --dataset $R/training/firetruck_$c/$DS --mask_dir $R/training/Training_dataset_mask --name cal_$c --workers $W
done
for d in $R/firetruck_*/; do
  c=$(basename $d); c=${c#firetruck_}
  [ -f features/test_$c.npz ] || $PY extract_features.py --dataset $R/firetruck_$c/$DS --mask_dir $R/firetruck_$c/${DS}_mask $SHARED --name test_$c --workers $W
done
[ -f features/test_firetruck.npz ] || $PY extract_features.py --dataset $R/firetruck/$DS --mask_dir $SHARED --name test_firetruck --workers $W
echo DONE
