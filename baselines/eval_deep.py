# -*- coding: utf-8 -*-
"""Batch evaluation in the torch environment: UniDepth V2 / ML-Depth-Pro predictions + SAM / YOLO→SAM automatic masks.

Settings (aligned with the paper's protocol for comparison methods):
- Official pretrained weights, no retraining; the median metric depth inside the mask is the prediction. Camera intrinsics fx = 1137.8 px (CARLA 40 mm / 36 mm).
- Linear calibration (added 2026-11-16): run the same calibration set as CA-DFDS (white/gray/black, 1.0–7.0 m),
  then fit d = a·pred + b in scripts/compare_methods.py, so every method is corrected with the same calibration data.
- --auto: fully automatic mask (deep_worker.auto: YOLO11m → SAM; when YOLO misses, fall back to the near-range full-frame box). Each image is processed on its own, without video tracking.
  The automatic masks are saved to logs/scratch/auto_masks/<set>/<color>/T<n>_<shot>.npy for CA-DFDS to reuse
  (scripts/eval_auto_mask.py), and depth is also recomputed with the automatic masks.

Usage (repo root, torch environment):
  ../.venv/Scripts/python.exe baselines/eval_deep.py --set test  --colors red green blue black --sam --auto --out results/deep_baselines.csv
  ../.venv/Scripts/python.exe baselines/eval_deep.py --set calib --colors white gray black --out results/deep_calib.csv
"""
import os, sys, csv, glob, re, argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_worker as W

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'

ap = argparse.ArgumentParser()
ap.add_argument('--set', choices=['test', 'calib'], default='test')
ap.add_argument('--colors', nargs='+', default=['red', 'green', 'blue', 'black'])
ap.add_argument('--shots', type=int, default=10)
ap.add_argument('--sam', action='store_true', help='SAM box prompt using the hand-mask bounding box')
ap.add_argument('--auto', action='store_true', help='YOLO→SAM fully automatic mask')
ap.add_argument('--out', default='results/deep_baselines.csv')
a = ap.parse_args()

if a.set == 'test':
    root = lambda c: os.path.join(DATA, f'firetruck_{c}', DS); mask_dir = os.path.join(DATA, 'firetruck_black', DS + '_mask')
else:
    root = lambda c: os.path.join(DATA, 'training', f'firetruck_{c}', DS); mask_dir = os.path.join(DATA, 'training', 'Training_dataset_mask')
items = []
for c in a.colors:
    for tdir in sorted(glob.glob(os.path.join(root(c), 'T*'))):
        t = int(re.search(r'T(\d+)$', tdir).group(1)); mp = os.path.join(mask_dir, f'T{t}.npy')
        if not os.path.exists(mp):
            continue
        for k, f in enumerate(sorted(glob.glob(os.path.join(tdir, 'R', '*.png')))[:a.shots], 1):
            items.append((c, t, k, float(re.search(r'D(\d+(?:\.\d+)?)_', os.path.basename(f)).group(1)), f, mp))
print(f'{a.set}: {len(items)} images, device={W.device()}', flush=True)
W.handle({'cmd': 'warmup', 'models': ['unidepth', 'depthpro'] + (['sam'] if a.sam or a.auto else []) + (['yolo'] if a.auto else [])})

tmp = os.path.join(os.environ.get('TEMP', '.'), 'cadfds_eval_sam.npy')
auto_dir = os.path.join(ROOT, 'logs', 'scratch', 'auto_masks', a.set)
rows = []
for i, (c, t, k, d, f, mp) in enumerate(items, 1):
    h = np.load(mp) > 0
    r = dict(color=c, t=t, shot=k, dist_m=d, path=f)
    for m in ('unidepth', 'depthpro'):
        o = W.depth(m, f, mp); r[f'{m}_m'] = round(o['median_m'], 4); r[f'{m}_ms'] = round(o['ms'], 1)
    if a.sam:
        ys, xs = np.nonzero(h); box = [xs.min(), ys.min(), xs.max(), ys.max()]
        o = W.sam(f, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], tmp, box); s = np.load(tmp) > 0
        r['sam_iou'] = round(float((s & h).sum() / (s | h).sum()), 4); r['sam_ms'] = round(o['ms'], 1)
    if a.auto:
        out = os.path.join(auto_dir, c, f'T{t}_{k:02d}.npy'); os.makedirs(os.path.dirname(out), exist_ok=True)
        o = W.auto(f, out)
        r['yolo_found'] = int(bool(o['dets'])); r['auto_source'] = o['source']; r['yolo_ms'] = round(o['det_ms'], 1)
        if o['found']:
            s = np.load(out) > 0
            r['yolo_cls'] = o['dets'][0]['cls'] if o['dets'] else ''; r['yolo_conf'] = round(o['dets'][0]['conf'], 3) if o['dets'] else ''
            r['auto_sam_ms'] = round(o['sam_ms'], 1)
            r['auto_iou'] = round(float((s & h).sum() / (s | h).sum()), 4); r['auto_mask'] = out
            for m in ('unidepth', 'depthpro'):
                r[f'{m}_auto_m'] = round(W.depth(m, f, out)['median_m'], 4)
    rows.append(r)
    if i % 20 == 0 or i == len(items):
        print(f'{i}/{len(items)} {c} T{t:02d} gt={d} uni={r["unidepth_m"]} dp={r["depthpro_m"]}'
              + (f' src={r.get("auto_source")} iou={r.get("auto_iou")}' if a.auto else ''), flush=True)

keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
with open(a.out, 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print('saved', a.out)
