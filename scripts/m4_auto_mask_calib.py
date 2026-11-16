# -*- coding: utf-8 -*-
"""M4 / issue #5 (part 1, torch env): automatic masks (YOLO11m -> SAM ViT-B, deep_worker.auto) for the CALIBRATION set,
so that calibration AND test both use automatic masks (fully automatic pipeline, no hand masks anywhere).
Masks -> logs/scratch/auto_masks/calib/<color>/T<t>_<shot>.npy ; per-image log -> logs/scratch/auto_masks_calib.csv
Usage: ../.venv/Scripts/python.exe scripts/m4_auto_mask_calib.py
Then:  .venv-cadfds/Scripts/python.exe scripts/m4_auto_mask_report.py
"""
import os, sys, csv, glob, re
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'baselines'))
import deep_worker as W

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'
MASK = os.path.join(DATA, 'training', 'Training_dataset_mask')
OUT = os.path.join(ROOT, 'logs', 'scratch', 'auto_masks', 'calib')

W._lazy_imports(); W.handle({'cmd': 'warmup', 'models': ['sam', 'yolo']})
rows = []
for c in ('white', 'gray', 'black'):
    for tdir in sorted(glob.glob(os.path.join(DATA, 'training', f'firetruck_{c}', DS, 'T*'))):
        t = int(re.search(r'T(\d+)$', tdir).group(1)); mp = os.path.join(MASK, f'T{t}.npy')
        if not os.path.exists(mp):
            continue
        h = np.load(mp) > 0
        for k, f in enumerate(sorted(glob.glob(os.path.join(tdir, 'R', '*.png')))[:10], 1):
            out = os.path.join(OUT, c, f'T{t}_{k:02d}.npy'); os.makedirs(os.path.dirname(out), exist_ok=True)
            o = W.auto(f, out); s = np.load(out) > 0
            if s.shape != h.shape:
                import cv2; hh = cv2.resize(h.astype(np.uint8), (s.shape[1], s.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            else:
                hh = h
            rows.append(dict(set='calib', color=c, t=t, shot=k, dist_m=float(re.search(r'D(\d+(?:\.\d+)?)_', os.path.basename(f)).group(1)),
                             path=f, auto_source=o['source'], yolo_ms=round(o['det_ms'], 1), auto_sam_ms=round(o['sam_ms'], 1),
                             auto_iou=round(float((s & hh).sum() / (s | hh).sum()), 4), auto_mask=out))
        print(c, f'T{t:02d}', rows[-1]['auto_source'], rows[-1]['auto_iou'], flush=True)
with open(os.path.join(ROOT, 'logs', 'scratch', 'auto_masks_calib.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('done', len(rows))
