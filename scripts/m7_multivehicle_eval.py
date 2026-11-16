# -*- coding: utf-8 -*-
"""M7: evaluate CA-DFDS / grayscale DFD on the CARLA multi-vehicle data made by scripts/gen_multivehicle.py.

Features (torch env, GPU pipeline = CPU features to 1e-13), CARLA instance-segmentation masks, f/1.4 (R) images:
  features/mv_<vehicle>_<color>_<test|calib>.npz   (same format as features/test_*.npz, plus texture_f22 per image)
Vehicles: sedan_charger, suv_patrol, truck_ambulance (test red/blue/black 1.2-7.2 m, calib white/gray/black 1.0-7.0 m),
          truck_carlacola (test, native livery only), and the existing fire truck (hand masks; test red/green/blue/black 1.2-6.7 m,
          calib white/gray/black).
Calibration protocols (quadratic d(sigma), per-distance means of the calibration colors, as in evaluate.py):
  A firetruck   : fire-truck calibration only (the paper's setting) -> every vehicle
  B own         : the vehicle's own calibration set -> its own test set
  C pooled      : calibration sets of all 4 vehicles together -> every vehicle
  D leave-one-out: calibration of the other vehicles -> held-out vehicle (unseen vehicle type)
Outputs: results/multi_vehicle_carla.csv (+ _by_distance.csv), results/multi_vehicle_carla_examples.jpg
Usage: ../.venv/Scripts/python.exe scripts/m7_multivehicle_eval.py
"""
import os, sys, csv, glob
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import cadfds_core as C
from evaluate import load, per_dist

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
MV = os.path.join(DATA, 'multivehicle')
NEW = ['sedan_charger', 'suv_patrol', 'truck_ambulance']
SET = {'gray': 'gray', 'cadfds': C.W_OPT, 'jonly': (1.0, 0.0, 0.0)}
KIND = {'test': 'Test', 'calib': 'Training'}


def extract(veh, color, st, g):
    out = f'features/mv_{veh}_{color}_{st}.npz'
    if os.path.exists(out):
        return out
    ds = os.path.join(MV, f'{veh}_{color}', f'{KIND[st]}_dataset_Focus1500cm_f1.4_f22.0')
    cov, gr, dist, ts, paths, tex = [], [], [], [], [], []
    for p, t, d in C.discover(ds):
        img = C.imread(p); m = C.load_mask(os.path.join(ds + '_mask', f'T{t}.npy'), img.shape)
        if m.sum() < 100:
            continue
        ci, cm = C.crop_to_mask(img, m)
        cov.append(g.cov_from_bgr(ci, cm)); gr.append(C.gray_sigma(ci, cm)); dist.append(d); ts.append(t); paths.append(p)
        li = C.imread(p.replace(os.sep + 'R' + os.sep, os.sep + 'L' + os.sep))
        lc, lm = C.crop_to_mask(li, m); tex.append(C.gray_sigma(lc, lm))
    np.savez(out, cov=np.array(cov), gray=np.array(gr), dist_cm=np.array(dist), t=np.array(ts), path=np.array(paths), texture_f22=np.array(tex))
    print('saved', out, len(cov), flush=True)
    return out


def feat(D, s):
    return D['gray'] if isinstance(s, str) else C.sigma_eq5(D['cov'], s)


def curve(cal, s):
    X, Y = [], []
    for D in cal:
        x, d = per_dist(feat(D, s), D['dist']); ok = np.isfinite(x); X += list(x[ok]); Y += list(d[ok])
    return C.fit_quadratic(np.array(X), np.array(Y))


def errors(co, D, s):
    x = feat(D, s); ok = np.isfinite(x) & np.isfinite(D['dist'])
    return np.polyval(co, x[ok]) / 100 - D['dist'][ok] / 100, D['dist'][ok] / 100


def main():
    import cadfds_gpu as G
    g = G.CIECAM02GPU()
    cal = {'firetruck': [load(f'features/cal_{c}.npz') for c in ('white', 'gray', 'black')]}
    test = {'firetruck': {c: load(f'features/test_{c}.npz') for c in ('red', 'green', 'blue', 'black')}}
    tex = {}
    for v in NEW:
        cal[v] = [load(extract(v, c, 'calib', g)) for c in ('white', 'gray', 'black')]
        test[v] = {c: load(extract(v, c, 'test', g)) for c in ('red', 'blue', 'black')}
        tex[v] = float(np.mean(np.load(f'features/mv_{v}_red_test.npz')['texture_f22']))
    if os.path.isdir(os.path.join(MV, 'truck_carlacola_native')):
        test['truck_carlacola'] = {'native': load(extract('truck_carlacola', 'native', 'test', g))}
    vehicles_cal = list(cal)
    rows, bydist = [], []
    for sn, s in SET.items():
        curves = {'A_firetruck': {v: curve(cal['firetruck'], s) for v in test},
                  'B_own': {v: curve(cal[v], s) for v in test if v in cal},
                  'C_pooled': {v: curve(sum((cal[k] for k in vehicles_cal), []), s) for v in test},
                  'D_leave_one_out': {v: curve(sum((cal[k] for k in vehicles_cal if k != v), []), s) for v in test}}
        for proto, cv in curves.items():
            for v, (co, r2) in cv.items():
                allE, allD = [], []
                for c, D in test[v].items():
                    e, d = errors(co, D, s); allE += list(e); allD += list(d)
                    rows.append(dict(setting=sn, protocol=proto, vehicle=v, color=c, n=len(e), calib_r2=round(r2, 4),
                                     rmse_m=round(float(np.sqrt(np.mean(e ** 2))), 4), bias_m=round(float(e.mean()), 4)))
                allE, allD = np.array(allE), np.array(allD)
                rows.append(dict(setting=sn, protocol=proto, vehicle=v, color='ALL', n=len(allE), calib_r2=round(r2, 4),
                                 rmse_m=round(float(np.sqrt(np.mean(allE ** 2))), 4), bias_m=round(float(allE.mean()), 4)))
                for d in np.unique(allD):
                    e = allE[allD == d]
                    bydist.append(dict(setting=sn, protocol=proto, vehicle=v, dist_m=float(d), rmse_m=round(float(np.sqrt(np.mean(e ** 2))), 4),
                                       bias_m=round(float(e.mean()), 4)))
    for name, R in (('results/multi_vehicle_carla.csv', rows), ('results/multi_vehicle_carla_by_distance.csv', bydist)):
        with open(name, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=list(R[0])); w.writeheader(); w.writerows(R)
    print('\nRMSE (m), all colors, CA-DFDS paper weights / gray / J only:')
    for proto in ('A_firetruck', 'B_own', 'C_pooled', 'D_leave_one_out'):
        line = []
        for v in test:
            r = {s: next((x for x in rows if x['protocol'] == proto and x['vehicle'] == v and x['color'] == 'ALL' and x['setting'] == s), None) for s in SET}
            if r['cadfds']:
                line.append(f"{v}: {r['cadfds']['rmse_m']:.3f}/{r['gray']['rmse_m']:.3f}/{r['jonly']['rmse_m']:.3f} (bias {r['cadfds']['bias_m']:+.2f})")
        print(proto, ' | '.join(line))
    print('texture f/22 (red test):', {k: round(v, 3) for k, v in tex.items()})
    # example grid: each vehicle (red, or native) at 4.2 m, f/1.4, with CARLA mask
    tiles = []
    for v in NEW + (['truck_carlacola'] if 'truck_carlacola' in test else []):
        c = 'native' if v == 'truck_carlacola' else 'red'
        ds = os.path.join(MV, f'{v}_{c}', 'Test_dataset_Focus1500cm_f1.4_f22.0')
        p = sorted(glob.glob(os.path.join(ds, 'T07', 'R', '*.png')))[0]; im = C.imread(p)
        m = C.load_mask(os.path.join(ds + '_mask', 'T7.npy'), im.shape)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (0, 255, 0), 6)
        cv2.putText(im, f'{v} {c} 4.2 m', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6)
        tiles.append(cv2.resize(im, (768, 576), interpolation=cv2.INTER_AREA))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imencode('.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile('results/multi_vehicle_carla_examples.jpg')


if __name__ == '__main__':
    main()
