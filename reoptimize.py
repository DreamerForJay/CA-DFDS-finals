# -*- coding: utf-8 -*-
"""以明確設定重做通道權重之交替最佳化（論文 參、五 所述之可重現設定）。
限制 w_i >= 0、Σw = 1；SLSQP（數值差分梯度）；起始 (0.6, 0.2, 0.2)；
停止：相鄰兩輪校正 RMSE 變化 < 1e-4 cm 或 100 輪。
  --form 7：以各通道標準差之線性組合（式 (7)）為替代特徵
  --form 5：直接以式 (5) 最佳化
用法：python reoptimize.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz --form 5
"""
import argparse, json, numpy as np
from scipy.optimize import minimize
import cadfds_core as C
from evaluate import load, per_dist

def points(cal, w, form):
    X, Y = [], []
    for D in cal:
        x = C.sigma_eq5(D['cov'], w) if form == '5' else np.sqrt(np.einsum('nii->ni', D['cov'])) @ np.asarray(w)
        xd, d = per_dist(x, D['dist']); X += list(xd); Y += list(d)
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--calib', nargs='+', required=True)
    ap.add_argument('--form', choices=['5', '7'], default='5'); ap.add_argument('--out', default='results/reopt.json')
    a = ap.parse_args(); cal = [load(f) for f in a.calib]
    w = np.array([0.6, 0.2, 0.2]); prev = None; hist = []
    for k in range(1, 101):
        X, Y = points(cal, w, a.form); co = np.polyfit(X, Y, 2)
        rm = float(np.sqrt(np.mean((np.polyval(co, X) - Y) ** 2))); hist.append(rm)
        if prev is not None and abs(prev - rm) < 1e-4: break
        prev = rm
        obj = lambda v: np.sqrt(np.mean((np.polyval(co, points(cal, v, a.form)[0]) - Y) ** 2))
        w = minimize(obj, w, method='SLSQP', bounds=[(0, 1)] * 3,
                     constraints=[{'type': 'eq', 'fun': lambda v: v.sum() - 1}]).x
    res = dict(form=a.form, weights=[round(float(x), 4) for x in w], rounds=k, calib_rmse_cm=round(hist[-1], 3))
    print(res); json.dump(res, open(a.out, 'w'), indent=2)
