# -*- coding: utf-8 -*-
"""Deep-model outputs needed for the demo assets (torch environment).

1. Rotation 180° (DEMO_ASSETS C-F1 / S6): UniDepth V2 and ML-Depth-Pro on every test image rotated by 180°
   (red/green/blue/black × T01–T12 × 10 = 480), mask rotated with the image (= segmentation re-finds the car),
   median depth inside the mask → results/demo/rotation_deep.csv
2. Depth maps for the demo frames (default: every colour × T01–T12 × shot 01, original and rotated), saved as
   float16 maps at 512×384 → logs/scratch/demo_cache/<model>_<color>_T<t>_<shot>[_rot].npy (not committed)

Usage (repo root): ../.venv/Scripts/python.exe baselines/demo_deep.py [--no-rotation]
"""
import os, sys, csv, glob, re, time, argparse
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_worker as W

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'
MASK_DIR = os.path.join(DATA, 'firetruck_black', DS + '_mask')
CACHE = os.path.join(ROOT, 'logs', 'scratch', 'demo_cache')
MODELS = ('unidepth', 'depthpro')

ap = argparse.ArgumentParser()
ap.add_argument('--colors', nargs='+', default=['red', 'green', 'blue', 'black'])
ap.add_argument('--map-shot', type=int, default=1, help='shot whose depth maps are cached for the videos')
ap.add_argument('--no-rotation', action='store_true')
a = ap.parse_args()
os.makedirs(CACHE, exist_ok=True)
print('device', W.device(), flush=True)   # triggers deep_worker's lazy imports (cv2, torch)
W.handle({'cmd': 'warmup', 'models': list(MODELS)})
tmp_i = os.path.join(CACHE, '_rot_img.npy'); tmp_m = os.path.join(CACHE, '_rot_mask.npy')

rows = []
items = []
for c in a.colors:
    for tdir in sorted(glob.glob(os.path.join(DATA, f'firetruck_{c}', DS, 'T*'))):
        t = int(re.search(r'T(\d+)$', tdir).group(1)); mp = os.path.join(MASK_DIR, f'T{t}.npy')
        if os.path.exists(mp):
            for k, f in enumerate(sorted(glob.glob(os.path.join(tdir, 'R', '*.png')))[:10], 1):
                items.append((c, t, k, float(re.search(r'D(\d+(?:\.\d+)?)_', os.path.basename(f)).group(1)), f, mp))
t0 = time.time()
for i, (c, t, k, d, f, mp) in enumerate(items, 1):
    want_map = k == a.map_shot
    if a.no_rotation and not want_map:
        continue
    img = W.imread(f); m = np.load(mp) > 0
    np.save(tmp_i, cv2.rotate(img, cv2.ROTATE_180)); np.save(tmp_m, cv2.rotate(m.astype(np.uint8), cv2.ROTATE_180))
    r = dict(color=c, t=t, shot=k, dist_m=d, path=f)
    for mdl in MODELS:
        if want_map:
            o = W.depth(mdl, f, mp, return_map=True)
            np.save(os.path.join(CACHE, f'{mdl}_{c}_T{t:02d}_{k:02d}.npy'), cv2.resize(o['map'], (512, 384), interpolation=cv2.INTER_AREA).astype(np.float16))
        o = W.depth(mdl, tmp_i, tmp_m, return_map=want_map)
        r[f'{mdl}_rot_m'] = round(o['median_m'], 4); r[f'{mdl}_rot_ms'] = round(o['ms'], 1)
        if want_map:
            np.save(os.path.join(CACHE, f'{mdl}_{c}_T{t:02d}_{k:02d}_rot.npy'), cv2.resize(o['map'], (512, 384), interpolation=cv2.INTER_AREA).astype(np.float16))
    rows.append(r)
    if i % 40 == 0 or i == len(items):
        print(f'{i}/{len(items)} {c} T{t:02d} gt={d} uni_rot={r["unidepth_rot_m"]} dp_rot={r["depthpro_rot_m"]} ({time.time() - t0:.0f}s)', flush=True)
for p in (tmp_i, tmp_m):
    if os.path.exists(p):
        os.remove(p)
if not a.no_rotation:
    out = os.path.join(ROOT, 'results', 'demo', 'rotation_deep.csv'); os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('saved', out)
