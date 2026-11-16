# -*- coding: utf-8 -*-
"""M7 report figures in the style of the lab-meeting slides (per-colour test tables, calibration curve, error-% bars).

Input : features/mv_*.npz (from scripts/m7_multivehicle_eval.py), features/cal_*.npz (fire truck)
Output: results/m7_report/*.png and results/m7_report/tables.json (numbers used by docs/M7_MULTIVEHICLE_REPORT.md)
Method: CA-DFDS sigma (Eq. 5) with the paper weights W_OPT; quadratic d(sigma) fitted on per-distance mean sharpness of the
        white/black/gray calibration set of the SAME vehicle (protocol B_own). Each test image is predicted separately;
        the table shows the per-distance mean prediction, and RMSE is over all images (same number as RESULTS 補充 R).
Usage : .venv-cadfds/Scripts/python.exe scripts/m7_report.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import cadfds_core as C
from evaluate import load, per_dist

for f in (r'C:\Windows\Fonts\msjh.ttc', r'C:\Windows\Fonts\kaiu.ttf'):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DFKai-SB', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUT = 'results/m7_report'; os.makedirs(OUT, exist_ok=True)
W = C.W_OPT
VEH = {'sedan_charger': '轎車（dodge.charger）', 'suv_patrol': 'SUV（nissan.patrol）', 'truck_ambulance': '貨車（ambulance.ford）'}
CN = {'red': '紅色', 'blue': '藍色', 'black': '黑色', 'white': '白色', 'gray': '灰色', 'green': '綠色', 'native': '原廠塗裝'}
TC = {'red': '#d62728', 'blue': '#1f77b4', 'black': '#222222'}


def sig(D):
    return C.sigma_eq5(D['cov'], W)


def fit(cal):
    X, Y = [], []
    for D in cal:
        x, d = per_dist(sig(D), D['dist']); X += list(x); Y += list(d)
    co, r2 = C.fit_quadratic(np.array(X), np.array(Y))
    return co, r2, np.array(X), np.array(Y)


def table_png(path, title, rows, rmse):
    n = len(rows) + 1
    fig = plt.figure(figsize=(9, 0.42 * n + 1.3), dpi=150)
    ax = fig.add_axes([0.02, 0.6 / (0.42 * n + 1.3), 0.96, 0.42 * n / (0.42 * n + 1.3)]); ax.axis('off')
    head = ['原始距離 (m)', '預測距離 (m)', '差距 (m)', '誤差百分比']
    cells = [[f"{r['d']:.1f}", f"{r['pred']:.2f}", f"{r['diff']:+.2f}", f"{r['pct']:.2f}%"] for r in rows]
    t = ax.table(cellText=cells, colLabels=head, cellLoc='center', bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False); t.set_fontsize(14)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('black')
        if i == 0:
            c.set_text_props(weight='bold')
    fig.text(0.5, 1 - 0.35 / (0.42 * n + 1.3), title, ha='center', va='center', fontsize=20, weight='bold')
    fig.text(0.5, 0.3 / (0.42 * n + 1.3), f'RMSE：{rmse:.4f} m（逐張，共 {rows[0]["n_all"]} 張）', ha='center', va='center', fontsize=15)
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def main():
    report = {'weights': list(W), 'vehicles': {}}
    bars = []
    for v, vname in VEH.items():
        cal = [load(f'features/mv_{v}_{c}_calib.npz') for c in ('white', 'black', 'gray')]
        co, r2, X, Y = fit(cal)
        # calibration curve (slide 2 style)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        ax.scatter(X, Y, color='blue', alpha=0.55, label='All Samples')
        xs = np.linspace(X.min(), X.max(), 200); ax.plot(xs, np.polyval(co, xs), 'r-', label='Combined Fit')
        ax.set_xlabel('Sharpness Score'); ax.set_ylabel('Distance (cm)'); ax.grid(True); ax.legend(loc='upper left')
        ax.set_title(f'Calibration for Colors: white, black, gray — {v}  (R² = {r2:.3f})')
        fig.savefig(f'{OUT}/calib_{v}.png', bbox_inches='tight'); plt.close(fig)
        vrep = {'name': vname, 'calib_r2': round(r2, 4), 'coef_cm': [float(c) for c in co], 'colors': {}}
        for c in ('red', 'blue', 'black'):
            D = load(f'features/mv_{v}_{c}_test.npz')
            pred = np.polyval(co, sig(D)) / 100; dist = D['dist'] / 100
            rows = []
            for d in np.unique(dist):
                p = pred[dist == d].mean()
                rows.append(dict(d=float(d), pred=float(p), diff=float(p - d), pct=float(abs(p - d) / d * 100), n_all=len(pred)))
            rmse = float(np.sqrt(np.mean((pred - dist) ** 2)))
            table_png(f'{OUT}/table_{v}_{c}.png', f'f/1.4、焦距 15 m 之{CN[c]}{vname.split("（")[0]}測試結果', rows, rmse)
            near = np.mean([r['pct'] for r in rows if r['d'] < 3.7]); far = np.mean([r['pct'] for r in rows if r['d'] >= 3.7])
            vrep['colors'][c] = dict(rows=rows, rmse_m=round(rmse, 4), near_pct=round(float(near), 2), far_pct=round(float(far), 2))
            bars.append((v, c, near, far))
        report['vehicles'][v] = vrep
    # error-% bars before/after 3.7 m (slide 6 style), one group per vehicle-colour
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    lab = [f'{VEH[v].split("（")[0]}\n{CN[c]}' for v, c, _, _ in bars]; x = np.arange(len(bars)); w = 0.4
    b1 = ax.bar(x - w / 2, [b[2] for b in bars], w, color='#3274a1', label='3.7 m 以前平均誤差 (%)')
    b2 = ax.bar(x + w / 2, [b[3] for b in bars], w, color='#e1812c', label='3.7 m 以後平均誤差 (%)')
    ax.bar_label(b1, fmt='%.2f', fontsize=9); ax.bar_label(b2, fmt='%.2f', fontsize=9)
    ax.set_xticks(x, lab); ax.set_ylabel('平均誤差 (%)'); ax.legend(title='距離區間'); ax.grid(axis='y', alpha=0.4)
    ax.set_title(f'各車種、各色在 3.7 m 前後的平均誤差對比（權重 {W}，各車種自己校正）')
    fig.savefig(f'{OUT}/error_pct_37m.png', bbox_inches='tight'); plt.close(fig)
    # protocol comparison (firetruck calibration vs own calibration etc.), from results/multi_vehicle_carla.csv
    import csv
    R = [r for r in csv.DictReader(open('results/multi_vehicle_carla.csv', encoding='utf-8-sig')) if r['setting'] == 'cadfds' and r['color'] == 'ALL']
    vs = ['firetruck', 'sedan_charger', 'suv_patrol', 'truck_ambulance', 'truck_carlacola']
    vn = ['消防車', '轎車', 'SUV', '救護車', 'carlacola']
    pr = [('A_firetruck', '消防車校正（論文設定）'), ('B_own', '自己車種校正'), ('C_pooled', '4 車種合併校正'), ('D_leave_one_out', '留一車種（未見過）')]
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150); x = np.arange(len(vs)); w = 0.2
    proto_tab = {}
    for k, (p, pn) in enumerate(pr):
        y = [next((float(r['rmse_m']) for r in R if r['protocol'] == p and r['vehicle'] == v), np.nan) for v in vs]
        proto_tab[p] = dict(zip(vs, y))
        b = ax.bar(x + (k - 1.5) * w, y, w, label=pn); ax.bar_label(b, fmt='%.2f', fontsize=8)
    ax.set_xticks(x, vn); ax.set_ylabel('RMSE (m)'); ax.legend(); ax.grid(axis='y', alpha=0.4)
    ax.set_title('不同校正方式下各車種的 RMSE（CA-DFDS，論文權重）')
    fig.savefig(f'{OUT}/protocol_rmse.png', bbox_inches='tight'); plt.close(fig)
    report['protocols'] = proto_tab
    json.dump(report, open(f'{OUT}/tables.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for v, r in report['vehicles'].items():
        print(v, 'R2', r['calib_r2'], {c: (x['rmse_m'], x['near_pct'], x['far_pct']) for c, x in r['colors'].items()})


if __name__ == '__main__':
    main()
