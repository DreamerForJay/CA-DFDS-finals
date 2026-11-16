# -*- coding: utf-8 -*-
"""M7/M8 / issue #8: other vehicle type + failure cases, using only data already on the lab host (no new CARLA renders).

Calibration: fire truck white+gray+black, 1.0-7.0 m. Two curves: hand masks (paper) and automatic masks (cal_auto_*.npz,
from scripts/m4_auto_mask_report.py); sets without hand masks are scored with the automatic-mask curve.
Features: GPU pipeline (cadfds_gpu, fp64, = CPU features to 1e-13) + grayscale sigma, cropped to the mask box.

Cases
  suv_runA / suv_runB : datas/audi.etron and datas/ford.mustang. Both folders actually contain the SAME SUV
                        (CARLA 0.10 has no audi.etron / mustang blueprints; get_dataset2.py fell back to nissan.patrol_2021),
                        two separate renders. 1.2-7.2 m, automatic masks (no hand masks exist).
  beyond_7.2m         : fire truck red/green/blue/black test T13 (7.2 m > calibration max 7.0 m), automatic masks.
  flipped             : datas/reverse/firetruck_<c>_reverse (images flipped top-bottom), reverse hand masks, T1-T12.
  exposure_xG         : fire truck red/black test, hand masks, image gain G in linear light (0.5, 0.7, 1.4, 2.0), clipped.
  texture             : 19 paper colors, hand masks: texture = grayscale Laplacian std in mask on the f/22 (L) image;
                        Spearman correlation of per-color RMSE with mean texture; per-distance RMSE vs mask area.
Outputs: results/multi_vehicle.csv, results/failure_cases/{summary.csv, per_image.csv, texture_by_color.csv, *.jpg}
Usage (torch env): ../.venv/Scripts/python.exe scripts/m78_vehicle_failure.py
"""
import os, sys, glob, csv, json
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'baselines')); os.chdir(ROOT)
import cadfds_core as C
from evaluate import load, per_dist

DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'
TEST_MASK = os.path.join(DATA, 'firetruck_black', DS + '_mask')
REV_MASK = os.path.join(DATA, 'reverse', DS + '_mask_reverse')
OUT = os.path.join('results', 'failure_cases'); os.makedirs(OUT, exist_ok=True)
SCR = os.path.join('logs', 'scratch', 'auto_masks', 'm78'); os.makedirs(SCR, exist_ok=True)
SET = {'gray': 'gray', 'cadfds': C.W_OPT, 'jonly': (1.0, 0.0, 0.0)}
PAPER19 = ['red', 'green', 'blue', 'black', 'gray', 'deepGray', 'deep160Gray', 'deepRed', 'deepGreen', 'deepBlue',
           'brown', 'chartreuse', 'cyan', 'magenta', 'orange', 'rose', 'springGreen', 'violet', 'yellow']
v = np.arange(256) / 255.0
DEC = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def gain_lut(g):
    lin = np.clip(DEC * g, 0, 1)
    return np.round(255 * np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)).astype(np.uint8)


def curves(cal):
    out = {}
    for k, s in SET.items():
        X, Y = [], []
        for D in cal:
            x = D['gray'] if k == 'gray' else C.sigma_eq5(D['cov'], s)
            xd, d = per_dist(x, D['dist']); ok = np.isfinite(xd); X += list(xd[ok]); Y += list(d[ok])
        out[k] = C.fit_quadratic(np.array(X), np.array(Y))[0]
    return out


def predict(cov, gs, cv):
    return {k: float(np.polyval(cv[k], gs if k == 'gray' else C.sigma_eq5(cov, s))) / 100 for k, s in SET.items()}


def main():
    import cadfds_gpu as G
    import deep_worker as Wk
    Wk._lazy_imports(); Wk.handle({'cmd': 'warmup', 'models': ['sam', 'yolo']})
    g = G.CIECAM02GPU()
    cv_hand = curves([load(f'features/cal_{c}.npz') for c in ('white', 'gray', 'black')])
    cv_auto = curves([load(f'features/cal_auto_{c}.npz') for c in ('white', 'gray', 'black')])
    rows = []

    def run(case, img, mask, dist_m, path, cv, curve_name, extra=None):
        ci, cm = C.crop_to_mask(img, mask)
        cov = g.cov_from_bgr(ci, cm); gs = C.gray_sigma(ci, cm)
        p = predict(cov, gs, cv)
        r = dict(case=case, path=path, dist_m=dist_m, curve=curve_name, mask_px=int(mask.sum()), sigma_cadfds=round(C.sigma_eq5(cov, C.W_OPT), 4))
        r.update({f'{k}_pred_m': round(x, 4) for k, x in p.items()}); r.update(extra or {}); rows.append(r)
        return r

    # --- SUV (two renders) and beyond-range 7.2 m: automatic masks
    jobs = []
    for case, folder in (('suv_runA', 'audi.etron'), ('suv_runB', 'ford.mustang')):
        jobs += [(case, p, t, d) for p, t, d in C.discover(os.path.join(DATA, folder, DS))]
    for c in ('red', 'green', 'blue', 'black'):
        jobs += [('beyond_7.2m', p, t, d) for p, t, d in C.discover(os.path.join(DATA, f'firetruck_{c}', DS)) if t == 13]
    for i, (case, p, t, d) in enumerate(jobs):
        mp = os.path.join(SCR, f'{case}_{i:04d}.npy'); o = Wk.auto(p, mp)
        img = C.imread(p); m = C.load_mask(mp, img.shape)
        run(case, img, m, d / 100, p, cv_auto, 'auto', dict(mask_source=o['source'], mask_file=mp))
        if i % 50 == 0:
            print(case, i, '/', len(jobs), flush=True)
    # --- flipped (reverse) with reverse hand masks
    for c in ('red', 'green', 'blue', 'black'):
        for p, t, d in C.discover(os.path.join(DATA, 'reverse', f'firetruck_{c}_reverse', DS)):
            mp = C.find_mask([REV_MASK], t)
            if mp is None:
                continue
            img = C.imread(p); run('flipped', img, C.load_mask(mp, img.shape), d / 100, p, cv_hand, 'hand', dict(color=c))
    print('flipped done', flush=True)
    # --- exposure gain, hand masks
    for gn in (0.5, 0.7, 1.4, 2.0):
        lut = gain_lut(gn)
        for c in ('red', 'black'):
            for p, t, d in C.discover(os.path.join(DATA, f'firetruck_{c}', DS)):
                mp = C.find_mask([TEST_MASK], t)
                if mp is None:
                    continue
                img = cv2.LUT(C.imread(p), lut); m = C.load_mask(mp, img.shape)
                r = run(f'exposure_x{gn}', img, m, d / 100, p, cv_hand, 'hand', dict(color=c))
                r['clipped_frac'] = round(float((img[m] == 255).any(-1).mean()), 4)
        print('exposure', gn, flush=True)

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, 'per_image.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)

    # --- summary vs reference (fire truck, same protocol)
    ref = {}
    for k, s in SET.items():
        for name, cv_, pref in (('hand', cv_hand, 'test_'), ('auto', cv_auto, 'test_auto_')):
            e = []
            for c in ('red', 'green', 'blue', 'black'):
                D = load(f'features/{pref}{c}.npz'); x = D['gray'] if k == 'gray' else C.sigma_eq5(D['cov'], s)
                ok = np.isfinite(x) & np.isfinite(D['dist']); e += list(np.polyval(cv_[k], x[ok]) / 100 - D['dist'][ok] / 100)
            ref[(k, name)] = float(np.sqrt(np.mean(np.square(e))))
    summ = []
    for name in ('reference_firetruck_hand', 'reference_firetruck_auto'):
        cur = name.split('_')[-1]
        summ.append(dict(case=name, curve=cur, n=480, dist_range='1.2-6.7', **{f'{k}_rmse_m': round(ref[(k, cur)], 4) for k in SET}))
    for case in dict.fromkeys(r['case'] for r in rows):
        R = [r for r in rows if r['case'] == case]
        for sub, RR in (('all', R), ('1.2-6.7m', [r for r in R if r['dist_m'] <= 6.7])) if case.startswith('suv') else (('all', R),):
            d = np.array([r['dist_m'] for r in RR])
            s = dict(case=case if sub == 'all' else f'{case} ({sub})', curve=RR[0]['curve'], n=len(RR), dist_range=f'{d.min()}-{d.max()}')
            for k in SET:
                e = np.array([r[f'{k}_pred_m'] for r in RR]) - d
                s[f'{k}_rmse_m'] = round(float(np.sqrt(np.mean(e ** 2))), 4); s[f'{k}_bias_m'] = round(float(e.mean()), 4)
            if 'clipped_frac' in RR[0]:
                s['clipped_frac_mean'] = round(float(np.mean([r['clipped_frac'] for r in RR])), 4)
            summ.append(s)
    keys = []
    for s in summ:
        keys += [k for k in s if k not in keys]
    with open(os.path.join(OUT, 'summary.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=''); w.writeheader(); w.writerows(summ)
    for s in summ:
        print(s, flush=True)

    # --- multi_vehicle.csv: SUV per distance
    mv = []
    for case in ('suv_runA', 'suv_runB'):
        R = [r for r in rows if r['case'] == case]
        for d in sorted({r['dist_m'] for r in R}):
            RR = [r for r in R if r['dist_m'] == d]
            m = dict(vehicle='nissan.patrol_2021 (SUV)', run=case, folder='audi.etron' if case.endswith('A') else 'ford.mustang', dist_m=d,
                     n=len(RR), fallback_masks=sum(r['mask_source'] == 'fallback' for r in RR), mask_px_mean=int(np.mean([r['mask_px'] for r in RR])))
            for k in SET:
                e = np.array([r[f'{k}_pred_m'] for r in RR]) - d
                m[f'{k}_rmse_m'] = round(float(np.sqrt(np.mean(e ** 2))), 4); m[f'{k}_bias_m'] = round(float(e.mean()), 4)
            mv.append(m)
    with open('results/multi_vehicle.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(mv[0])); w.writeheader(); w.writerows(mv)

    # --- example images: worst CA-DFDS error per case
    for case in dict.fromkeys(r['case'] for r in rows):
        R = [r for r in rows if r['case'] == case]; r = max(R, key=lambda r: abs(r['cadfds_pred_m'] - r['dist_m']))
        img = C.imread(r['path'])
        if case.startswith('exposure'):
            img = cv2.LUT(img, gain_lut(float(case.split('_x')[1])))
        mf = r.get('mask_file') or C.find_mask([REV_MASK if case == 'flipped' else TEST_MASK], int(os.path.basename(os.path.dirname(os.path.dirname(r['path'])))[1:]))
        m = C.load_mask(mf, img.shape); cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = img.copy(); cv2.drawContours(vis, cs, -1, (0, 0, 255), 6)
        cv2.putText(vis, f"{case}: true {r['dist_m']:.1f} m, CA-DFDS {r['cadfds_pred_m']:.2f} m, gray {r['gray_pred_m']:.2f} m",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 5)
        cv2.imencode('.jpg', cv2.resize(vis, (1024, 768), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(
            os.path.join(OUT, f'worst_{case}.jpg'))

    # --- texture / mask area (19 colors, hand masks, f/22 image)
    tex_rows = []
    cvh = cv_hand
    for c in PAPER19:
        D = np.load(f'features/test_{c}.npz', allow_pickle=True); x = C.sigma_eq5(D['cov'], C.W_OPT)
        tex, err = [], []
        for p, t, d, s in zip(D['path'], D['t'], D['dist_cm'], x):
            if not np.isfinite(s):
                continue
            pl = str(p).replace(os.sep + 'R' + os.sep, os.sep + 'L' + os.sep).replace('\\R\\', '\\L\\')
            im = C.imread(pl)
            if im is None:
                continue
            m = C.load_mask(C.find_mask([TEST_MASK], int(t)), im.shape); ci, cm = C.crop_to_mask(im, m)
            tex.append(C.gray_sigma(ci, cm)); err.append(np.polyval(cvh['cadfds'], s) / 100 - d / 100)
        tex_rows.append(dict(color=c, n=len(err), texture_f22_mean=round(float(np.mean(tex)), 3),
                             cadfds_rmse_m=round(float(np.sqrt(np.mean(np.square(err)))), 4)))
        print('texture', c, tex_rows[-1], flush=True)
    from scipy.stats import spearmanr
    rho, pv = spearmanr([r['texture_f22_mean'] for r in tex_rows], [r['cadfds_rmse_m'] for r in tex_rows])
    with open(os.path.join(OUT, 'texture_by_color.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(tex_rows[0])); w.writeheader(); w.writerows(tex_rows)
    json.dump({'spearman_texture_vs_rmse': {'rho': round(float(rho), 3), 'p': round(float(pv), 4), 'n_colors': len(tex_rows)}},
              open(os.path.join(OUT, 'texture_stats.json'), 'w'), indent=2)
    print('spearman texture vs rmse', rho, pv)


if __name__ == '__main__':
    main()
