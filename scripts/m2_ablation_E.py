# -*- coding: utf-8 -*-
"""M2-E / issue #4: ablation with a larger calibration set (white+gray+black+red+green+blue).

Weights are re-optimized ONLY on the calibration features (cal_*.npz, 1.0-7.0 m) with reoptimize.py's procedure
(SLSQP, w>=0, sum w = 1, start (0.6,0.2,0.2), alternating with the quadratic fit), then tested on the paper's 19 test
colors minus the 5 whose hue is in the calibration set (red, green, blue, black, gray) -> 14 unseen colors.
Models A-D are recomputed under the same 6-color calibration, and also under the original 3-color calibration,
so every row uses the same test split.
Usage: python scripts/m2_ablation_E.py --out results/ablation_E.csv
"""
import os, sys, csv, json, argparse
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cadfds_core as C
from evaluate import load, per_dist
from reoptimize import points

UNSEEN = ['deepGray', 'deep160Gray', 'deepRed', 'deepGreen', 'deepBlue', 'brown', 'chartreuse', 'cyan', 'magenta',
          'orange', 'rose', 'springGreen', 'violet', 'yellow']
SEEN = ['red', 'green', 'blue', 'black', 'gray']


def reopt(cal, form):
    w = np.array([0.6, 0.2, 0.2]); prev = None; hist = []
    for k in range(1, 101):
        X, Y = points(cal, w, form); co = np.polyfit(X, Y, 2)
        rm = float(np.sqrt(np.mean((np.polyval(co, X) - Y) ** 2))); hist.append((k, [round(float(v), 4) for v in w], round(rm, 3)))
        if prev is not None and abs(prev - rm) < 1e-4:
            break
        prev = rm
        obj = lambda v: np.sqrt(np.mean((np.polyval(co, points(cal, v, form)[0]) - Y) ** 2))
        w = minimize(obj, w, method='SLSQP', bounds=[(0, 1)] * 3, constraints=[{'type': 'eq', 'fun': lambda v: v.sum() - 1}]).x
    return np.clip(w, 0, None), hist


def feat(D, s):
    return D['gray'] if isinstance(s, str) else C.sigma_eq5(D['cov'], s)


def evaluate(cal, tests, s):
    X, Y = [], []
    for D in cal:
        x, d = per_dist(feat(D, s), D['dist']); X += list(x); Y += list(d)
    co, r2 = C.fit_quadratic(np.array(X), np.array(Y))
    out = {}
    for c, D in tests.items():
        x = feat(D, s); ok = np.isfinite(x) & np.isfinite(D['dist'])
        out[c] = float(np.sqrt(np.mean((np.polyval(co, x[ok]) - D['dist'][ok]) ** 2)) / 100)
    return out, r2


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/ablation_E.csv'); a = ap.parse_args()
    cal3 = [load(f'features/cal_{c}.npz') for c in ('white', 'gray', 'black')]
    cal6 = [load(f'features/cal_{c}.npz') for c in ('white', 'gray', 'black', 'red', 'green', 'blue')]
    tests = {c: load(f'features/test_{c}.npz') for c in SEEN + UNSEEN}
    w6_5, h5 = reopt(cal6, '5'); w6_7, h7 = reopt(cal6, '7')
    print('E (eq5 reopt on 6 colors):', w6_5.round(4), 'rounds', len(h5)); print('E (eq7 reopt on 6 colors):', w6_7.round(4), 'rounds', len(h7))
    json.dump({'calib_colors': ['white', 'gray', 'black', 'red', 'green', 'blue'],
               'eq5': {'weights': [round(float(v), 4) for v in w6_5], 'history': h5},
               'eq7': {'weights': [round(float(v), 4) for v in w6_7], 'history': h7}},
              open(a.out.replace('.csv', '_weights.json'), 'w'), indent=2)
    models = [('A', 'gray', 'gray'), ('B', 'CIECAM02 fixed (0.6,0.2,0.2)', C.W_FIXED['w1']),
              ('C', 'CIECAM02 paper opt (0.8136,0.1782,0.0081)', C.W_OPT), ('D', 'CIECAM02 J only', (1.0, 0.0, 0.0)),
              ('E', 'CIECAM02 reopt eq5 on 6 colors ' + str(tuple(w6_5.round(4))), tuple(w6_5)),
              ('E7', 'CIECAM02 reopt eq7-weights on 6 colors (inference eq5) ' + str(tuple(w6_7.round(4))), tuple(w6_7))]
    rows = []
    for calname, cal in (('cal3_white_gray_black', cal3), ('cal6_+red_green_blue', cal6)):
        for code, desc, s in models:
            if code.startswith('E') and calname.startswith('cal3'):
                continue
            per, r2 = evaluate(cal, tests, s)
            mu = np.mean([per[c] for c in UNSEEN]); ms = np.mean([per[c] for c in SEEN])
            r = dict(calib=calname, model=code, desc=desc, calib_r2=round(r2, 4), mean14_unseen=round(mu, 4), mean5_seen_hue=round(ms, 4))
            r.update({c: round(per[c], 4) for c in SEEN + UNSEEN}); rows.append(r)
            print(f'{calname:22s} {code:3s} R2 {r2:.4f} unseen14 {mu:.3f} seen5 {ms:.3f}  {desc}')
    with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print('saved', a.out)
