# -*- coding: utf-8 -*-
"""Build the calibration curves for each CA-DFDS variant used in the GUI comparison → results/calibration_methods.json

Same procedure as evaluate.py: on the calibration set (white + gray + black, 1.0–7.0 m), average the 10 images at each
distance, then fit a quadratic d_cm = c2·σ² + c1·σ + c0.
  gray    : grayscale Laplacian sharpness (classic DFD baseline without colour information)
  opt     : CA-DFDS paper weights (0.8136, 0.1782, 0.0081)
  reopt5  : weights re-optimised with eq. (5) on 2026-11-15 (0.9434, 0.0566, 0)
  jonly   : lightness J only (1, 0, 0)
  unidepth_lin / depthpro_lin : deep models' metric depth → linear calibration on the same calibration set (needs results/deep_calib.csv)
Usage: python scripts/build_calibrations.py
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cadfds_core as C
from evaluate import load, feat, per_dist

reopt = json.load(open(os.path.join(ROOT, 'results', 'reopt_eq5.json'), encoding='utf-8'))['weights']
METHODS = {'gray': 'gray', 'opt': list(C.W_OPT), 'reopt5': reopt, 'jonly': [1.0, 0.0, 0.0]}
cal = [load(os.path.join(ROOT, 'features', f'cal_{c}.npz')) for c in ('white', 'gray', 'black')]
out = {}
for k, s in METHODS.items():
    X, Y = [], []
    for D in cal:
        x, d = per_dist(feat(D, s if s == 'gray' else tuple(s)), D['dist']); X += list(x); Y += list(d)
    co, r2 = C.fit_quadratic(np.array(X), np.array(Y))
    out[k] = {'weights': s, 'coef_cm': [float(c) for c in co], 'r2': round(r2, 4),
              'sigma_range': [float(min(X)), float(max(X))]}
    print(f'{k:7s} R2={r2:.4f} coef={co}')
# Deep models: linear calibration on the same calibration set (white/gray/black 1.0–7.0 m) d = a·pred + b (per-distance mean, same protocol as CA-DFDS)
dc = os.path.join(ROOT, 'results', 'deep_calib.csv')
if os.path.exists(dc):
    import csv
    rows = list(csv.DictReader(open(dc, encoding='utf-8-sig')))
    for k in ('unidepth', 'depthpro'):
        X, Y = [], []
        for c in sorted({r['color'] for r in rows}):
            for d in sorted({float(r['dist_m']) for r in rows}):
                v = [float(r[f'{k}_m']) for r in rows if r['color'] == c and float(r['dist_m']) == d]
                if v:
                    X.append(np.mean(v)); Y.append(d * 100)
        co = np.polyfit(X, Y, 1); pred = np.polyval(co, X)
        r2 = 1 - ((np.array(Y) - pred) ** 2).sum() / ((np.array(Y) - np.mean(Y)) ** 2).sum()
        out[k + '_lin'] = {'model': k, 'coef_cm': [float(c) for c in co], 'r2': round(float(r2), 4),
                           'formula': 'd_cm = c1*pred_m + c0 (linear calibration on the calibration set)'}
        print(f'{k}_lin R2={r2:.4f} coef={co}')
out['_note'] = 'calib = cal_white + cal_gray + cal_black, per-distance mean, quadratic fit (scripts/build_calibrations.py)'
json.dump(out, open(os.path.join(ROOT, 'results', 'calibration_methods.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('saved results/calibration_methods.json')
