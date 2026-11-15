# -*- coding: utf-8 -*-
"""Batch-evaluate UniDepthV2 / ML-Depth-Pro on the CA-DFDS test set (run in the torch environment).

Setup (matches the paper's comparison-method protocol):
- Official pretrained weights, no retraining and no calibration curve: the model's metric depth is the prediction.
- Same images and region as CA-DFDS: the median depth inside the hand-drawn mask (car rear) is the predicted distance.
- Known camera intrinsics passed in (fx = 1137.8 px, from the CARLA camera's 40 mm lens / 36 mm sensor).
- Optionally SAM box segmentation (--sam): uses the hand-drawn mask's bounding box as the prompt and records SAM's IoU with the hand-drawn mask.

Usage (repo root):
  D:\\Carla-0.10.0-Win64-Shipping\\.venv\\Scripts\\python.exe baselines/eval_deep.py --colors red green blue black --out results/deep_baselines.csv
"""
import os, sys, csv, glob, re, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_worker as W

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'
MASK_DIR = os.path.join(DATA, 'firetruck_black', DS + '_mask')

ap = argparse.ArgumentParser()
ap.add_argument('--colors', nargs='+', default=['red', 'green', 'blue', 'black'])
ap.add_argument('--shots', type=int, default=10)
ap.add_argument('--sam', action='store_true')
ap.add_argument('--out', default='results/deep_baselines.csv')
a = ap.parse_args()

items = []
for c in a.colors:
    for tdir in sorted(glob.glob(os.path.join(DATA, f'firetruck_{c}', DS, 'T*'))):
        t = int(re.search(r'T(\d+)$', tdir).group(1))
        mp = os.path.join(MASK_DIR, f'T{t}.npy')
        if not os.path.exists(mp):
            continue
        for f in sorted(glob.glob(os.path.join(tdir, 'R', '*.png')))[:a.shots]:
            items.append((c, t, float(re.search(r'D(\d+(?:\.\d+)?)_', os.path.basename(f)).group(1)), f, mp))
print(f'{len(items)} images, device={W.device()}', flush=True)
for k in ('unidepth', 'depthpro') + (('sam',) if a.sam else ()):
    {'sam': W.get_sam, 'depthpro': W.get_depthpro, 'unidepth': W.get_unidepth}[k]()
W.depth('unidepth', items[0][3], items[0][4]); W.depth('depthpro', items[0][3], items[0][4])   # warm-up

tmp = os.path.join(os.environ.get('TEMP', '.'), 'cadfds_eval_sam.npy')
rows = []
for i, (c, t, d, f, mp) in enumerate(items, 1):
    r = dict(color=c, t=t, dist_m=d, path=f)
    for k in ('unidepth', 'depthpro'):
        o = W.depth(k, f, mp); r[f'{k}_m'] = round(o['median_m'], 4); r[f'{k}_ms'] = round(o['ms'], 1)
    if a.sam:
        h = np.load(mp) > 0; ys, xs = np.nonzero(h); box = [xs.min(), ys.min(), xs.max(), ys.max()]
        o = W.sam(f, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], tmp, box); s = np.load(tmp) > 0
        r['sam_iou'] = round(float((s & h).sum() / (s | h).sum()), 4); r['sam_ms'] = round(o['ms'], 1)
    rows.append(r)
    if i % 20 == 0 or i == len(items):
        print(f'{i}/{len(items)} {c} T{t:02d} gt={d} uni={r["unidepth_m"]} dp={r["depthpro_m"]}', flush=True)

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
with open(a.out, 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('saved', a.out)
