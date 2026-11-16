# -*- coding: utf-8 -*-
"""M1 / issue #3: compare the two sharpness definitions with the same weights and calibration set.

  Eq. (5): sigma = Std_M(wJ*lapJ + wa*lap_ac + wb*lap_bc) = sqrt(w' Sigma w)   (includes channel covariance)
  Eq. (7): S     = wJ*Std(lapJ) + wa*Std(lap_ac) + wb*Std(lap_bc) = w . sqrt(diag Sigma)

For each weight set and each form, a quadratic d(s) is fitted on the calibration colors (per-distance mean,
same as evaluate.py), then every test image is predicted on its own (per-image RMSE, m).
Usage: python scripts/m1_eq5_vs_eq7.py --out results/eq5_vs_eq7.csv
"""
import os, sys, csv, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cadfds_core as C
from evaluate import load, per_dist

PAPER19 = ['red', 'green', 'blue', 'black', 'gray', 'deepGray', 'deep160Gray', 'deepRed', 'deepGreen', 'deepBlue',
           'brown', 'chartreuse', 'cyan', 'magenta', 'orange', 'rose', 'springGreen', 'violet', 'yellow']


def feature(D, w, form):
    w = np.asarray(w, float)
    if form == '5':
        return C.sigma_eq5(D['cov'], w)
    return np.sqrt(np.clip(np.einsum('nii->ni', D['cov']), 0, None)) @ w


def fit(cal, w, form):
    X, Y = [], []
    for D in cal:
        x, d = per_dist(feature(D, w, form), D['dist']); X += list(x); Y += list(d)
    return C.fit_quadratic(np.array(X), np.array(Y))


def rmse(co, D, w, form):
    x = feature(D, w, form); ok = np.isfinite(x) & np.isfinite(D['dist'])
    return float(np.sqrt(np.mean((np.polyval(co, x[ok]) - D['dist'][ok]) ** 2)) / 100)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', nargs='+', default=['features/cal_white.npz', 'features/cal_gray.npz', 'features/cal_black.npz'])
    ap.add_argument('--out', default='results/eq5_vs_eq7.csv')
    a = ap.parse_args()
    cal = [load(f) for f in a.calib]
    tests = {c: load(f'features/test_{c}.npz') for c in PAPER19}
    weights = {'opt(0.8136,0.1782,0.0081)': C.W_OPT, 'w1(0.6,0.2,0.2)': C.W_FIXED['w1'], 'w2(0.8,0.1,0.1)': C.W_FIXED['w2'],
               'reopt5(0.9532,0.0468,0)': (0.9532, 0.0468, 0.0), 'Jonly(1,0,0)': (1.0, 0.0, 0.0)}
    for f in ('results/reopt_eq5.json', 'results/reopt_eq7.json'):
        import json
        j = json.load(open(f)); weights[f'reopt_eq{j["form"]}_local{tuple(j["weights"])}'.replace(' ', '')] = tuple(j['weights'])
    rows = []
    for wn, w in weights.items():
        for form in ('5', '7'):
            co, r2 = fit(cal, w, form)
            per = {c: rmse(co, D, w, form) for c, D in tests.items()}
            for c, v in per.items():
                rows.append(dict(weights=wn, form=f'eq{form}', test=c, rmse_per_image_m=round(v, 4), calib_r2=round(r2, 4),
                                 coef_cm=' '.join('%.6g' % x for x in co)))
            m19 = np.mean(list(per.values())); m17 = np.mean([v for c, v in per.items() if c not in ('black', 'gray')])
            rows.append(dict(weights=wn, form=f'eq{form}', test='MEAN19', rmse_per_image_m=round(m19, 4), calib_r2=round(r2, 4),
                             coef_cm=' '.join('%.6g' % x for x in co)))
            rows.append(dict(weights=wn, form=f'eq{form}', test='MEAN17_unseen', rmse_per_image_m=round(m17, 4), calib_r2=round(r2, 4),
                             coef_cm=' '.join('%.6g' % x for x in co)))
            print(f'{wn:40s} eq{form}  R2 {r2:.4f}  mean19 {m19:.3f}  mean17 {m17:.3f}  '
                  + ' '.join(f'{c[:5]}={per[c]:.3f}' for c in ('red', 'green', 'blue', 'black')))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print('saved', a.out)
