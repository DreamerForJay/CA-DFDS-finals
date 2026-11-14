# -*- coding: utf-8 -*-
"""對一個資料集（單一車色）抽取每張影像之 3x3 共變異矩陣與灰階清晰度。

用法：
  python extract_features.py --dataset <.../firetruck_red/Test_dataset_Focus1500cm_f1.4_f22.0>
                             --mask_dir <遮罩資料夾，可多個> --name test_red --workers 8
輸出：features/<name>.npz（cov, gray, dist_cm, t, path）
遮罩搜尋順序：--mask_dir 依序；每個資料夾內找 T<n>.npy / T<nn>.npy / .png。
"""
import os, argparse, numpy as np
from multiprocessing import Pool
import cadfds_core as C

def work(a):
    path, t, mpath = a
    img = C.imread(path)
    if img is None or mpath is None:
        return None
    m = C.load_mask(mpath, img.shape)
    return C.laplacian_cov(img, m), C.gray_sigma(img, m)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--mask_dir', nargs='+', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--aperture', default='R', help='光圈子資料夾；R = f/1.4')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    items = C.discover(a.dataset, a.aperture)
    jobs = [(p, t, C.find_mask(a.mask_dir, t)) for p, t, _ in items]
    miss = sorted({t for _, t, m in jobs if m is None})
    if miss: print('[警告] 找不到遮罩之 T：', miss)
    print(len(jobs), 'images')
    with Pool(a.workers) as pool:
        res = pool.map(work, jobs, chunksize=2)
    cov = np.array([r[0] if r else np.full((3, 3), np.nan) for r in res])
    gray = np.array([r[1] if r else np.nan for r in res])
    bad = [jobs[i][0] for i, r in enumerate(res) if r is None]
    if bad: print('[警告] 無法讀取／無遮罩：', len(bad), bad[:5])
    os.makedirs('features', exist_ok=True)
    np.savez(os.path.join('features', a.name + '.npz'), cov=cov, gray=gray,
             dist_cm=np.array([d for _, _, d in items]), t=np.array([t for _, t, _ in items]),
             path=np.array([p for p, _, _ in items]))
    print('saved features/%s.npz' % a.name)
