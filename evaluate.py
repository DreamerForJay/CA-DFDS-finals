# -*- coding: utf-8 -*-
"""以校正集擬合二次校正曲線，評估測試集 RMSE（論文表 8 之流程，可用於獨立測試距離）。

用法：
  python evaluate.py --calib features/cal_white.npz features/cal_gray.npz features/cal_black.npz \
                     --test features/test_red.npz features/test_green.npz features/test_blue.npz \
                     --extra "reopt5=0.9532,0.0468,0" --out results/table8_testdist.csv
輸出：
  results/*.csv（每種清晰度設定 × 每個測試車色之 RMSE，兩種協定）
  results/calibration_opt.json（最終權重之校正曲線，供 GUI 與 demo 使用）
兩種協定：
  per_distance：每距離先平均 10 張之清晰度再預測（同宇森校正程式）
  per_image   ：每張影像各自預測（實際使用情境）
"""
import os, argparse, csv, numpy as np
import cadfds_core as C

def load(f):
    d = np.load(f, allow_pickle=True)
    return dict(cov=d['cov'], gray=d['gray'], dist=d['dist_cm'], name=os.path.basename(f)[:-4])

def feat(D, setting):
    return D['gray'] if setting == 'gray' else C.sigma_eq5(D['cov'], setting)

def per_dist(x, dist):
    xs, ds = [], []
    for d in np.unique(dist[np.isfinite(dist)]):
        v = x[(dist == d) & np.isfinite(x)]
        if len(v): xs.append(v.mean()); ds.append(d)
    return np.array(xs), np.array(ds)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', nargs='+', required=True)
    ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--extra', nargs='*', default=[], help='name=wJ,wa,wb')
    ap.add_argument('--out', default='results/eval.csv')
    a = ap.parse_args()
    cal = [load(f) for f in a.calib]; tests = [load(f) for f in a.test]
    settings = {'gray': 'gray', 'w1(0.6,0.2,0.2)': C.W_FIXED['w1'], 'w2(0.8,0.1,0.1)': C.W_FIXED['w2'], 'opt(0.8136,0.1782,0.0081)': C.W_OPT}
    for e in a.extra:
        k, v = e.split('='); settings[k] = tuple(float(x) for x in v.split(','))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    rows = []
    for sname, s in settings.items():
        X, Y = [], []
        for D in cal:
            x, d = per_dist(feat(D, s), D['dist']); X += list(x); Y += list(d)
        co, r2 = C.fit_quadratic(np.array(X), np.array(Y))
        if sname.startswith('opt'):
            C.save_calibration(os.path.join(os.path.dirname(a.out) or '.', 'calibration_opt.json'), C.W_OPT, co,
                               [D['name'] for D in cal], r2)
        for D in tests:
            x = feat(D, s); ok = np.isfinite(x) & np.isfinite(D['dist'])
            img_rmse = np.sqrt(np.mean((np.polyval(co, x[ok]) - D['dist'][ok]) ** 2)) / 100
            xd, dd = per_dist(x, D['dist'])
            dist_rmse = np.sqrt(np.mean((np.polyval(co, xd) - dd) ** 2)) / 100
            rows.append(dict(setting=sname, test=D['name'], rmse_per_distance_m=round(dist_rmse, 4),
                             rmse_per_image_m=round(img_rmse, 4), n_images=int(ok.sum()), n_distances=len(dd),
                             calib_r2=round(r2, 4), coef_cm=' '.join('%.6g' % c for c in co)))
            print(f'{sname:32s} {D["name"]:18s} per-dist {dist_rmse:.3f} m   per-image {img_rmse:.3f} m')
    with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('saved', a.out)
