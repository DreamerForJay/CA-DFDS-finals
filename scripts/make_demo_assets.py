# -*- coding: utf-8 -*-
"""Demo assets for the project page and slides (docs/DEMO_ASSETS.md) → results/demo/<code>/

Every number drawn in a video or figure comes from the result files of the same run:
  features/test_<c>.npz (CA-DFDS / grayscale σ, hand masks), features/stmap_test.npz (scripts/stmap_eval.py),
  results/deep_baselines.csv (UniDepth V2 / ML-Depth-Pro), results/demo/rotation_deep.csv (baselines/demo_deep.py),
  results/calibration_methods.json (all calibration curves), results/method_comparison.json (480-image RMSE).
Depth maps for the deep-model panels: logs/scratch/demo_cache/ (baselines/demo_deep.py, torch env).

Assets (folder per code, see DEMO_ASSETS.md §4 for the file layout):
  C-A1 teaser, three panels           C-C1 input → output steps (near / mid / far)
  C-E1 1×4 method comparison          C-F1 before/after: rotation 180°, grayscale vs CIECAM02 sharpness
  C-B1 camera setup diagram           C-D1 defocus theory curves (focus 7 m vs 15 m, f/1.4 vs f/22)
  C-H1 leaderboard table              C-I1 analysis charts          C-M1 timing        C-K1 failure cases
Usage: python scripts/make_demo_assets.py [--only C-A1 C-E1 ...]
Needs logs/bin/ffmpeg.exe (or ffmpeg on PATH) and the Microsoft JhengHei font (C:/Windows/Fonts/msjh*.ttc).
"""
import os, sys, csv, json, shutil, subprocess, argparse, re, glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import cadfds_core as C

R = lambda *p: os.path.join(ROOT, *p)
OUT = R('results', 'demo')
CACHE = R('logs', 'scratch', 'demo_cache')
DATA = os.environ.get('CADFDS_DATA', r'D:\Carla-0.10.0-Win64-Shipping\Carla-0.10.0-Win64-Shipping\datas')
DS = 'Test_dataset_Focus1500cm_f1.4_f22.0'
MASK_DIR = os.path.join(DATA, 'firetruck_black', DS + '_mask')
W, H, FPS = 1920, 1080, 30
FONT_R, FONT_B = r'C:\Windows\Fonts\msjh.ttc', r'C:\Windows\Fonts\msjhbd.ttc'
_fc = {}


def F(size, bold=False):
    k = (size, bold)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FONT_B if bold else FONT_R, size)
    return _fc[k]


# colours (RGB) — same identities as the GUI
BG, PANEL, LINE, TXT, SUB = (14, 22, 38), (24, 35, 56), (70, 88, 118), (238, 242, 250), (160, 175, 200)
COL = {'cadfds': (240, 180, 40), 'gray': (170, 170, 170), 'stmap': (70, 200, 160),
       'unidepth': (90, 150, 255), 'depthpro': (190, 120, 240), 'gt': (255, 255, 255)}
NAME = {'cadfds': 'CA-DFDS（本作品）', 'gray': '灰階 DFD', 'stmap': 'STMAP（1994）',
        'unidepth': 'UniDepthV2（2025）', 'depthpro': 'Depth Pro（2024）'}
NAME_EN = {'cadfds': 'CA-DFDS (ours)', 'gray': 'Grayscale DFD', 'stmap': 'STMAP (Subbarao & Surya, 1994)',
           'unidepth': 'UniDepthV2 (Piccinelli et al., 2025)', 'depthpro': 'Depth Pro (Bochkovskii et al., 2024)'}
INPUTS = {'cadfds': '單張 f/1.4 影像＋車輛遮罩', 'stmap': '兩張影像（f/1.4＋f/22），取樣框',
          'unidepth': '單張影像＋相機內參，遮罩內中位數', 'depthpro': '單張影像＋焦距，遮罩內中位數', 'gray': '單張影像＋遮罩'}
COLOR_ZH = {'red': '紅色車', 'green': '綠色車', 'blue': '藍色車', 'black': '黑色車'}


def err_col(e):
    a = abs(e)
    return (90, 210, 120) if a < 0.2 else (255, 170, 60) if a < 0.5 else (255, 90, 90)


# ---------------------------------------------------------------- data
CAL = json.load(open(R('results', 'calibration_methods.json'), encoding='utf-8'))
MC = json.load(open(R('results', 'method_comparison.json'), encoding='utf-8'))
key = lambda p: os.path.normcase(os.path.normpath(str(p)))
DEEP = {key(r['path']): r for r in csv.DictReader(open(R('results', 'deep_baselines.csv'), encoding='utf-8-sig'))}
ROT = {}
if os.path.exists(os.path.join(OUT, 'rotation_deep.csv')):
    ROT = {key(r['path']): r for r in csv.DictReader(open(os.path.join(OUT, 'rotation_deep.csv'), encoding='utf-8-sig'))}
_st = np.load(R('features', 'stmap_test.npz'))
STM = {key(p): (float(s), float(sr)) for p, s, sr in zip(_st['path'], _st['sigma'], _st['sigma_rot'])}
poly = lambda k, x: float(np.polyval(CAL[k]['coef_cm'], x)) / 100
FEAT = {}


def feats(c):
    if c not in FEAT:
        z = np.load(R('features', f'test_{c}.npz'), allow_pickle=True)
        FEAT[c] = {key(p): i for i, p in enumerate(z['path'])}, z
    return FEAT[c]


def frame_info(c, t, shot=1):
    """All method predictions for one test image (hand mask)."""
    f = sorted(glob.glob(os.path.join(DATA, f'firetruck_{c}', DS, f'T{t:02d}', 'R', '*.png')))[shot - 1]
    idx, z = feats(c); i = idx[key(f)]; d = DEEP[key(f)]
    gt = float(z['dist_cm'][i]) / 100
    p = {'cadfds': poly('opt', C.sigma_eq5(z['cov'][i], CAL['opt']['weights'])), 'gray': poly('gray', z['gray'][i]),
         'stmap': poly('stmap', STM[key(f)][0]),
         'unidepth': poly('unidepth_lin', float(d['unidepth_m'])), 'depthpro': poly('depthpro_lin', float(d['depthpro_m'])),
         'unidepth_raw': float(d['unidepth_m']), 'depthpro_raw': float(d['depthpro_m'])}
    r = ROT.get(key(f))
    if r:
        p.update(unidepth_rot_raw=float(r['unidepth_rot_m']), depthpro_rot_raw=float(r['depthpro_rot_m']),
                 unidepth_rot=poly('unidepth_lin', float(r['unidepth_rot_m'])), depthpro_rot=poly('depthpro_lin', float(r['depthpro_rot_m'])),
                 stmap_rot=poly('stmap', STM[key(f)][1]))
    return dict(color=c, t=t, shot=shot, path=f, gt=gt, pred=p, sigma=float(C.sigma_eq5(z['cov'][i], CAL['opt']['weights'])))


_img_cache = {}


def load(c, t, shot=1):
    k = (c, t, shot)
    if k not in _img_cache:
        fi = frame_info(c, t, shot); img = C.imread(fi['path'])
        m = C.load_mask(os.path.join(MASK_DIR, f'T{t}.npy'), img.shape)
        _img_cache[k] = (fi, img, m)
    return _img_cache[k]


_chan_cache = {}


def channels(c, t, shot=1, rot=False):
    """CIECAM02 J/a_c/b_c and Laplacians on the mask crop (identical to the full-image result), pasted back to full size."""
    k = (c, t, shot, rot)
    if k not in _chan_cache:
        fi, img, m = load(c, t, shot)
        if rot:
            img, m = cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(m.astype(np.uint8), cv2.ROTATE_180) > 0
        ys, xs = np.nonzero(m); pad = 4
        y0, x0 = max(ys.min() - pad, 0), max(xs.min() - pad, 0); y1, x1 = ys.max() + pad + 1, xs.max() + pad + 1
        ch = C.ciecam_channels(img[y0:y1, x0:x1])
        lap = [cv2.Laplacian(x, cv2.CV_64F, ksize=3) for x in ch]
        g = cv2.cvtColor(cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), 0), cv2.COLOR_BGR2GRAY).astype(np.float64)
        lg = cv2.Laplacian(g, cv2.CV_64F, ksize=3)
        full = lambda a: (lambda z: (z.__setitem__((slice(y0, y1), slice(x0, x1)), a), z)[1])(np.zeros(m.shape))
        w = np.asarray(CAL['opt']['weights'])
        _chan_cache[k] = dict(img=img, mask=m, J=full(ch[0]), ac=full(ch[1]), bc=full(ch[2]),
                              lapw=full(w[0] * lap[0] + w[1] * lap[1] + w[2] * lap[2]), lapg=full(lg),
                              lapJ=full(lap[0]), lapa=full(lap[1]), lapb=full(lap[2]))
    return _chan_cache[k]


# ---------------------------------------------------------------- drawing helpers
def bgr2pil(a):
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


def pil2bgr(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def bbox(m, grow=0.25, aspect=4 / 3, shape=None):
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w, h = (x1 - x0) * (1 + grow), (y1 - y0) * (1 + grow)
    if w / h < aspect: w = h * aspect
    else: h = w / aspect
    Hh, Ww = shape or m.shape
    w, h = min(w, Ww), min(h, Hh)
    x0 = int(np.clip(cx - w / 2, 0, Ww - w)); y0 = int(np.clip(cy - h / 2, 0, Hh - h))
    return x0, y0, int(w), int(h)


def crop(a, b):
    x, y, w, h = b
    return a[y:y + h, x:x + w]


def overlay_mask(img, m, col=(0, 220, 255), alpha=0.25, thick=3):
    out = img.copy()
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(col)).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, col, thick)
    return out


def heat(lap, m, img, vmax, dim=0.25, cmap=cv2.COLORMAP_INFERNO):
    """|Laplacian| inside the mask as a colour map (fixed vmax across frames); outside the mask the image is dimmed."""
    v = np.sqrt(np.clip(np.abs(lap) / vmax, 0, 1))   # square-root scale so blurred (near) frames stay visible
    hm = cv2.applyColorMap((v * 255).astype(np.uint8), cmap)
    out = (img * dim).astype(np.uint8)
    out[m] = hm[m]
    return out


def chan_img(x, m, img, lo, hi, cmap=cv2.COLORMAP_VIRIDIS):
    v = np.clip((x - lo) / (hi - lo), 0, 1)
    hm = cv2.applyColorMap((v * 255).astype(np.uint8), cmap)
    out = (img * 0.25).astype(np.uint8); out[m] = hm[m]
    return out


def depth_vis(dmap, shape, m, vmax=10.0):
    d = cv2.resize(dmap.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    v = np.clip(d / vmax, 0, 1)
    vis = cv2.applyColorMap(((1 - v) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)   # near = red, far = blue
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs, -1, (255, 255, 255), 3)
    return vis


def fit(a, w, h):
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


def text(d, xy, s, size=24, col=TXT, bold=False, anchor='la'):
    d.text(xy, s, font=F(size, bold), fill=col, anchor=anchor)


def rbox(d, b, fill=PANEL, outline=None, r=14, width=2):
    d.rounded_rectangle(b, r, fill=fill, outline=outline, width=width)


class Video:
    """Frames → mp4v temp → ffmpeg H.264 (-crf 23 -pix_fmt yuv420p -movflags +faststart)."""
    def __init__(self, path, fps=FPS, size=(W, H)):
        self.path, self.tmp, self.n = path, path.replace('.mp4', '_raw.mp4'), 0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vw = cv2.VideoWriter(self.tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    def add(self, fr, sec=None, n=None):
        if isinstance(fr, Image.Image):
            fr = pil2bgr(fr)
        for _ in range(n if n is not None else max(1, int(round(sec * FPS)))):
            self.vw.write(fr); self.n += 1

    def fade(self, a, b, sec=0.4):
        a = pil2bgr(a) if isinstance(a, Image.Image) else a; b = pil2bgr(b) if isinstance(b, Image.Image) else b
        k = max(1, int(sec * FPS))
        for i in range(k):
            self.vw.write(cv2.addWeighted(a, 1 - (i + 1) / k, b, (i + 1) / k, 0)); self.n += 1

    def close(self):
        self.vw.release()
        ff = shutil.which('ffmpeg') or R('logs', 'bin', 'ffmpeg.exe')
        subprocess.run([ff, '-y', '-loglevel', 'error', '-i', self.tmp, '-c:v', 'libx264', '-preset', 'slow', '-crf', '23',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', self.path], check=True)
        os.remove(self.tmp)
        mb = os.path.getsize(self.path) / 2 ** 20
        print(f'  {os.path.relpath(self.path, ROOT)}  {self.n / FPS:.1f} s  {mb:.1f} MB', flush=True)
        return dict(file=os.path.basename(self.path), seconds=round(self.n / FPS, 2), mb=round(mb, 2))


def save_jpg(im, p, q=90):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    (im if isinstance(im, Image.Image) else bgr2pil(im)).convert('RGB').save(p, quality=q)


def save_png(im, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    (im if isinstance(im, Image.Image) else bgr2pil(im)).save(p, optimize=True)


def write_csv(p, rows):
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def write_json(p, obj):
    json.dump(obj, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


def readme(folder, title, body):
    open(os.path.join(folder, 'README.md'), 'w', encoding='utf-8').write(f'# {title}\n\n' + body.strip() + '\n')


def header(d, title, sub=None):
    text(d, (48, 26), title, 38, TXT, True)
    if sub:
        text(d, (48, 78), sub, 22, SUB)


def footer(d, s):
    text(d, (48, H - 30), s, 18, SUB)


RMSE480 = MC['hand']['overall_rmse_m']
SEQ = [(t, 1) for t in range(12, 0, -1)]   # T12 (6.7 m) → T01 (1.2 m), shot 01
rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
FOOT = 'CARLA 模擬 · 40 mm · f/1.4 · 對焦 15 m · 2048×1536 · 遮罩：資料集手繪 · 數字出處 results/demo/'


# ================================================================ C-A1 teaser
def plot_curve(d, box, hist, cur, xs_all, ymax=7.5, title='估測距離 vs 真實距離', methods=('cadfds',), legend=True, show_gt=True):
    """hist: list of {method: pred, 'gt': gt}; x = frame index."""
    x0, y0, x1, y1 = box; rbox(d, box)
    text(d, (x0 + 18, y0 + 12), title, 22, TXT, True)
    px0, py0, px1, py1 = x0 + 66, y0 + 56, x1 - 20, y1 - 52
    n = len(xs_all)
    sx = lambda i: px0 + (px1 - px0) * (i + 0.5) / n
    sy = lambda v: py1 - (py1 - py0) * v / ymax
    for v in range(0, int(ymax) + 1):
        d.line([px0, sy(v), px1, sy(v)], fill=(40, 55, 80), width=1); text(d, (px0 - 10, sy(v)), f'{v}', 17, SUB, anchor='rm')
    text(d, (px0 - 52, py0 - 6), 'm', 17, SUB)
    for i, lab in enumerate(xs_all):
        if i % 2 == 0 or n <= 12:
            text(d, (sx(i), py1 + 10), lab, 15, SUB, anchor='ma')
    text(d, ((px0 + px1) / 2, y1 - 22), '真實距離（m）→ 車輛接近', 16, SUB, anchor='ma')
    gts = [h['gt'] for h in hist]
    if gts and show_gt:
        d.line([(sx(i), sy(g)) for i, g in enumerate(gts)] if len(gts) > 1 else [(sx(0), sy(gts[0]))] * 2, fill=COL['gt'], width=3)
    for mth in methods:
        pts = [(sx(i), sy(h[mth])) for i, h in enumerate(hist) if np.isfinite(h[mth])]
        if len(pts) > 1:
            d.line(pts, fill=COL[mth], width=3)
        for x, y in pts:
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=COL[mth])
    if cur is not None:
        d.line([sx(cur), py0, sx(cur), py1], fill=(255, 255, 255, 120), width=1)
    if legend:
        lx = px0 + 10
        for lab, c in ([('真實距離', COL['gt'])] if show_gt else []) + [(NAME[m], COL[m]) for m in methods]:
            d.line([lx, py0 + 8, lx + 26, py0 + 8], fill=c, width=4); text(d, (lx + 32, py0 + 8), lab, 16, TXT, anchor='lm')
            lx += 60 + d.textlength(lab, font=F(16))


def asset_A1():
    code = 'C-A1_teaser'; fo = os.path.join(OUT, code); os.makedirs(os.path.join(fo, 'keyframes'), exist_ok=True)
    print('C-A1 teaser', flush=True)
    c = 'red'
    infos = [load(c, t, s)[0] for t, s in SEQ]
    vmax = np.percentile(np.abs(channels(c, 12)['lapw'][channels(c, 12)['mask']]), 99)
    xs_all = [f"{fi['gt']:.1f}" for fi in infos]
    pw, ph = 600, 450; X = [40, 660, 1280]; Y0 = 170
    frames, hist, rows = [], [], []
    for k, (t, s) in enumerate(SEQ):
        fi, img, m = load(c, t, s); ch = channels(c, t, s); p = fi['pred']['cadfds']; e = p - fi['gt']
        hist.append({'gt': fi['gt'], 'cadfds': p})
        rows.append(dict(frame=k, color=c, T=t, shot=s, gt_m=fi['gt'], cadfds_m=round(p, 4), err_m=round(e, 4),
                         sigma=round(fi['sigma'], 4), image=os.path.relpath(fi['path'], DATA)))
        im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
        header(d, 'CA-DFDS：單張失焦影像 → 車距', '單一相機、單張影像 · 由影像模糊程度（失焦）估測前車距離 · 不需深度學習模型')
        left = overlay_mask(img, m)
        bx = bbox(m, 0.15); cx_, cy_, cw, chh = bx
        cv2.rectangle(left, (cx_, cy_), (cx_ + cw, cy_ + chh), (255, 255, 255), 3)
        mid = crop(heat(ch['lapw'], ch['mask'], img, vmax), bx)
        im.paste(bgr2pil(fit(left, pw, ph)), (X[0], Y0)); im.paste(bgr2pil(fit(mid, pw, ph)), (X[1], Y0))
        for i, lab in enumerate(['① 輸入影像＋車輛遮罩', '② 清晰度熱圖（CIECAM02 加權拉普拉斯，白框放大）']):
            text(d, (X[i], Y0 - 34), lab, 21, TXT, True)
        # label on image
        lx, ly = X[0] + 14, Y0 + 14; rbox(d, (lx, ly, lx + 250, ly + 50), fill=(0, 0, 0), r=10)
        text(d, (lx + 12, ly + 25), f'CA-DFDS {p:.2f} m', 26, COL['cadfds'], True, 'lm')
        # colour bar
        cb = np.linspace(0, 1, 300)[None, :].repeat(14, 0); cbi = cv2.applyColorMap((cb * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        im.paste(bgr2pil(cbi), (X[1] + 150, Y0 + ph + 12)); text(d, (X[1] + 140, Y0 + ph + 19), '模糊', 16, SUB, anchor='rm')
        text(d, (X[1] + 460, Y0 + ph + 19), '清晰', 16, SUB, anchor='lm')
        plot_curve(d, (X[2], Y0 - 40, X[2] + pw, Y0 + ph), hist, k, xs_all, title='③ 估測距離 vs 真實距離')
        # number cards
        cy = Y0 + ph + 60
        for i, (lab, val, col) in enumerate([('真實距離（GT）', f"{fi['gt']:.2f} m", TXT), ('CA-DFDS 估測', f'{p:.2f} m', COL['cadfds']),
                                              ('誤差', f'{e:+.2f} m（{100 * e / fi["gt"]:+.1f}%）', err_col(e)),
                                              ('清晰度 σ', f"{fi['sigma']:.3f}", SUB)]):
            bx0 = 40 + i * 465; rbox(d, (bx0, cy, bx0 + 440, cy + 150))
            text(d, (bx0 + 22, cy + 18), lab, 22, SUB); text(d, (bx0 + 22, cy + 62), val, 46, col, True)
        text(d, (40, cy + 175), '影像 ' + f"{COLOR_ZH[c]} T{t:02d} shot{s:02d}" + ' · 距離 6.7 m → 1.2 m（資料集中 12 個距離，非連續拍攝）', 19, SUB)
        footer(d, FOOT)
        frames.append(im)
    seq_rmse = rmse([r['err_m'] for r in rows])
    end = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(end)
    text(d, (W / 2, 300), 'CA-DFDS 結果', 52, TXT, True, 'ma')
    text(d, (W / 2, 420), f'本段 12 個距離（紅色車）RMSE {seq_rmse:.2f} m', 40, COL['cadfds'], True, 'ma')
    text(d, (W / 2, 500), f"測試集 480 張（紅／綠／藍／黑 × 1.2–6.7 m）RMSE {RMSE480['opt']:.3f} m", 34, TXT, False, 'ma')
    text(d, (W / 2, 580), '單張影像 · 單一相機 · CPU 可算（GPU 版 CIECAM02 約 10–14 FPS）', 28, SUB, False, 'ma')
    footer(d, FOOT)
    v = Video(os.path.join(fo, 'video.mp4'))
    for i, fr in enumerate(frames):
        v.add(fr, 1.25)
        v.fade(fr, frames[i + 1] if i + 1 < len(frames) else end, 0.2)
    v.add(end, 2.4); info = v.close()
    vs = Video(os.path.join(fo, 'video_short.mp4'))
    for i, fr in enumerate(frames):
        vs.add(fr, 0.85); vs.fade(fr, frames[i + 1] if i + 1 < len(frames) else end, 0.12)
    vs.add(end, 1.6); info_s = vs.close()
    save_jpg(frames[0], os.path.join(fo, 'poster.jpg'))
    for i in (0, 5, 9, 11):
        save_png(frames[i], os.path.join(fo, 'keyframes', f'{i:02d}_{rows[i]["gt_m"]:.1f}m.png'))
    write_csv(os.path.join(fo, 'per_frame.csv'), rows)
    write_json(os.path.join(fo, 'summary.json'), dict(asset='C-A1 teaser', frames=len(rows), color=c, distance_range_m=[1.2, 6.7],
               seq_rmse_m=round(seq_rmse, 4), seq_mean_abs_err_m=round(float(np.mean(np.abs([r['err_m'] for r in rows]))), 4),
               test480_rmse_m=RMSE480['opt'], calibration='results/calibration_methods.json opt', video=info, video_short=info_s))
    readme(fo, 'C-A1 Teaser（三聯畫面）', f"""
- 內容：紅色消防車由 6.7 m 接近到 1.2 m（資料集 T12→T01，每個距離 shot01）。
  左：輸入影像＋車輛遮罩＋估測距離；中：CIECAM02 加權拉普拉斯清晰度熱圖（車輛區域放大，所有幀共用同一色階）；右：估測 vs 真實距離曲線。
- 資料：CARLA 模擬，40 mm、f/1.4、對焦 15 m。遮罩為資料集手繪遮罩。**12 個距離是分別拍攝的靜態影像，不是連續影片**。
- 方法：CA-DFDS 論文權重 (0.8136, 0.1782, 0.0081)，校正曲線 `results/calibration_methods.json` 的 `opt`（白＋灰＋黑校正集）。
- 數字：本段 RMSE {seq_rmse:.3f} m；片尾的 480 張 RMSE {RMSE480['opt']:.3f} m 出自 `results/method_comparison.json`。逐幀數值：`per_frame.csv`。
- 檔案：`video.mp4`（{info['seconds']} s）、`video_short.mp4`（{info_s['seconds']} s）、`poster.jpg`、`keyframes/`。
- 產生：`python scripts/make_demo_assets.py --only C-A1`
""")


# ================================================================ C-C1 input → output
def asset_C1():
    code = 'C-C1_input_output'; fo = os.path.join(OUT, code); print('C-C1 input→output', flush=True)
    c = 'red'; picks = [(1, '近'), (6, '中'), (12, '遠')]
    w = CAL['opt']['weights']; co = CAL['opt']['coef_cm']
    steps_meta = [('0_input', '① 輸入影像（f/1.4，全畫面）'), ('1_mask', '② 車輛遮罩（哪些像素是車）'),
                  ('2_J', '③ CIECAM02 明度 J'), ('3_ac', '④ 色度 a_c（紅–綠）'), ('4_bc', '⑤ 色度 b_c（黃–藍）'),
                  ('5_sharpness', '⑥ 加權拉普拉斯清晰度'), ('6_result', '⑦ σ → 校正曲線 → 距離')]
    vmax = np.percentile(np.abs(channels(c, 12)['lapw'][channels(c, 12)['mask']]), 99)
    allvals, video_frames = {}, []
    for t, tag in picks:
        fi, img, m = load(c, t); ch = channels(c, t); bx = bbox(m, 0.3)
        vals = dict(distance_tag=tag, color=c, T=t, shot=1, gt_m=fi['gt'], image=os.path.relpath(fi['path'], DATA),
                    mask_pixels=int(m.sum()), mask_bbox_xywh=list(map(int, bbox(m, 0, 1e-9))),
                    J_mean=round(float(ch['J'][m].mean()), 3), ac_mean=round(float(ch['ac'][m].mean()), 3), bc_mean=round(float(ch['bc'][m].mean()), 3),
                    std_lap_J=round(float(ch['lapJ'][m].std()), 4), std_lap_ac=round(float(ch['lapa'][m].std()), 4),
                    std_lap_bc=round(float(ch['lapb'][m].std()), 4), weights=w,
                    sigma=round(float(ch['lapw'][m].std()), 4), sigma_from_features=round(fi['sigma'], 4),
                    calib_coef_cm=co, pred_m=round(fi['pred']['cadfds'], 4), err_m=round(fi['pred']['cadfds'] - fi['gt'], 4))
        allvals[f'{fi["gt"]:.1f}m'] = vals
        J = ch['J'][m]; a_ = ch['ac'][m]; b_ = ch['bc'][m]
        pan = {'0_input': img, '1_mask': overlay_mask(img, m, alpha=0.35),
               '2_J': chan_img(ch['J'], m, img, 0, 100, cv2.COLORMAP_BONE),
               '3_ac': chan_img(ch['ac'], m, img, -40, 40, cv2.COLORMAP_COOL),
               '4_bc': chan_img(ch['bc'], m, img, -40, 40, cv2.COLORMAP_SPRING),
               '5_sharpness': heat(ch['lapw'], m, img, vmax)}
        sd = os.path.join(fo, 'steps', f'{fi["gt"]:.1f}m'); os.makedirs(sd, exist_ok=True)
        imgs = {}
        for name, lab in steps_meta:
            if name == '6_result':
                cv = Image.new('RGB', (1600, 1200), BG); d = ImageDraw.Draw(cv)
                text(d, (60, 50), lab, 44, TXT, True)
                # calibration curve
                bx0, by0, bx1, by1 = 120, 170, 1000, 1000
                rbox(d, (bx0 - 80, by0 - 30, bx1 + 30, by1 + 90))
                ss = np.linspace(*CAL['opt']['sigma_range'], 100); dd = np.polyval(co, ss) / 100
                smin, smax = ss.min() * 0.9, ss.max() * 1.05
                sx = lambda s: bx0 + (bx1 - bx0) * (s - smin) / (smax - smin); sy = lambda v: by1 - (by1 - by0) * v / 7.5
                for v in range(8):
                    d.line([bx0, sy(v), bx1, sy(v)], fill=(40, 55, 80)); text(d, (bx0 - 12, sy(v)), str(v), 22, SUB, anchor='rm')
                d.line([(sx(s), sy(v)) for s, v in zip(ss, dd)], fill=COL['cadfds'], width=4)
                sg = vals['sigma']; pd = vals['pred_m']
                d.line([sx(sg), by1, sx(sg), sy(pd)], fill=(255, 255, 255), width=2); d.line([bx0, sy(pd), sx(sg), sy(pd)], fill=(255, 255, 255), width=2)
                d.ellipse([sx(sg) - 11, sy(pd) - 11, sx(sg) + 11, sy(pd) + 11], fill=COL['cadfds'])
                text(d, ((bx0 + bx1) / 2, by1 + 40), '清晰度 σ（遮罩內加權拉普拉斯標準差）', 24, SUB, anchor='ma')
                text(d, (bx0 - 70, by0 - 22), '距離 d（m）', 22, SUB)
                text(d, (1060, 190), '本張數值', 30, TXT, True)
                lines = [('σ_J', f"{vals['std_lap_J']:.3f}"), ('σ_ac', f"{vals['std_lap_ac']:.3f}"), ('σ_bc', f"{vals['std_lap_bc']:.3f}"),
                         ('權重 w', f'({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})'), ('σ = √(w^T Σ w)', f"{sg:.4f}"),
                         ('d = c2·σ² + c1·σ + c0', f'{pd:.2f} m'), ('真實距離', f"{vals['gt_m']:.2f} m"),
                         ('誤差', f"{vals['err_m']:+.2f} m")]
                for i, (a1, a2) in enumerate(lines):
                    text(d, (1060, 260 + i * 92), a1, 24, SUB); text(d, (1060, 292 + i * 92), a2, 34, COL['cadfds'] if a1.startswith('d =') else TXT, True)
                im = cv
            else:
                a = pan[name] if name == '0_input' else crop(pan[name], bx)
                im = bgr2pil(fit(a, 1600, 1200)); d = ImageDraw.Draw(im)
                rbox(d, (24, 24, 40 + d.textlength(lab, font=F(42, True)) + 20, 100), fill=(0, 0, 0), r=12)
                text(d, (44, 62), lab, 42, TXT, True, 'lm')
                sub = {'1_mask': f'{vals["mask_pixels"]:,} 像素', '2_J': f'遮罩內平均 J = {vals["J_mean"]:.1f}',
                       '3_ac': f'平均 a_c = {vals["ac_mean"]:+.1f}', '4_bc': f'平均 b_c = {vals["bc_mean"]:+.1f}',
                       '5_sharpness': f'σ = {vals["sigma"]:.3f}（越亮越清晰；三個距離共用同一色階）',
                       '0_input': f'{COLOR_ZH[c]} · 真實距離 {vals["gt_m"]:.1f} m'}.get(name)
                if sub:
                    rbox(d, (24, 1110, 40 + d.textlength(sub, font=F(34)) + 20, 1176), fill=(0, 0, 0), r=12)
                    text(d, (44, 1143), sub, 34, TXT, False, 'lm')
            save_png(im, os.path.join(sd, f'{name}.png')); imgs[name] = im
        # strip for slides: 7 panels in one row
        tw, th = 400, 300; strip = Image.new('RGB', (7 * tw + 6 * 12, th + 70), BG); d = ImageDraw.Draw(strip)
        for i, (name, lab) in enumerate(steps_meta):
            strip.paste(imgs[name].resize((tw, th), Image.LANCZOS), (i * (tw + 12), 0))
            text(d, (i * (tw + 12) + tw / 2, th + 14), lab.split('（')[0], 22, TXT, True, 'ma')
        save_png(strip, os.path.join(sd, 'strip.png'))
        for name, lab in steps_meta:
            fr = Image.new('RGB', (W, H), BG); dd = ImageDraw.Draw(fr)
            header(dd, f'CA-DFDS 處理步驟 · {tag}距離 {vals["gt_m"]:.1f} m', '同一張影像逐步處理：遮罩 → CIECAM02 色彩通道 → 拉普拉斯清晰度 → σ → 距離')
            fr.paste(imgs[name].resize((1240, 930), Image.LANCZOS), (40, 112))
            for i, (n2, l2) in enumerate(steps_meta):
                on = n2 == name; y = 130 + i * 86
                rbox(dd, (1350, y, 1880, y + 72), fill=(60, 48, 20) if on else PANEL, outline=COL['cadfds'] if on else None)
                text(dd, (1370, y + 36), l2, 22, TXT if on else SUB, on, 'lm')
            text(dd, (1350, 760), f'估測 {vals["pred_m"]:.2f} m', 44, COL['cadfds'], True)
            text(dd, (1350, 830), f'真實 {vals["gt_m"]:.2f} m · 誤差 {vals["err_m"]:+.2f} m', 26, err_col(vals['err_m']))
            footer(dd, FOOT)
            video_frames.append((tag, fr))
    write_json(os.path.join(fo, 'steps.json'), allvals)
    v = Video(os.path.join(fo, 'video.mp4'))
    for i, (tag, fr) in enumerate(video_frames):
        v.add(fr, 1.3)
        if i + 1 < len(video_frames): v.fade(fr, video_frames[i + 1][1], 0.2)
    info = v.close()
    vs = Video(os.path.join(fo, 'video_short.mp4'))
    mids = [fr for tag, fr in video_frames if tag == '中']
    for i, fr in enumerate(mids):
        vs.add(fr, 1.4)
        if i + 1 < len(mids): vs.fade(fr, mids[i + 1], 0.2)
    info_s = vs.close()
    save_jpg(video_frames[0][1], os.path.join(fo, 'poster.jpg'))
    os.makedirs(os.path.join(fo, 'keyframes'), exist_ok=True)
    for t_, tag in picks:
        dist = f'{load(c, t_)[0]["gt"]:.1f}m'; shutil.copy(os.path.join(fo, 'steps', dist, 'strip.png'), os.path.join(fo, 'keyframes', f'strip_{dist}.png'))
    rows = [dict(distance=k, **{kk: (json.dumps(vv) if isinstance(vv, list) else vv) for kk, vv in v_.items()}) for k, v_ in allvals.items()]
    write_csv(os.path.join(fo, 'per_frame.csv'), rows)
    write_json(os.path.join(fo, 'summary.json'), dict(asset='C-C1 input→output', distances_m=[v_['gt_m'] for v_ in allvals.values()],
               pred_m=[v_['pred_m'] for v_ in allvals.values()], video=info, video_short=info_s))
    readme(fo, 'C-C1 輸入 → 輸出（近／中／遠）', f"""
- 紅色車 1.2 m（近）、3.7 m（中）、6.7 m（遠），shot01。每個距離 7 步，每步一張 1600×1200 PNG：`steps/<距離>/0_input.png … 6_result.png`，可直接做成網頁的「步驟滑桿」。
- `steps/<距離>/strip.png`（= `keyframes/strip_<距離>.png`）：7 步橫排一張，給簡報／列印用。
- `steps.json`：每步的數值（遮罩像素數、J／a_c／b_c 平均、各通道拉普拉斯標準差、權重、σ、校正係數、估測距離、誤差）。
  `sigma` 是由這次重算的拉普拉斯圖直接取標準差；`sigma_from_features` 是 `features/test_red.npz` 的值，兩者一致。
- 顯示說明：③–⑥ 為車輛區域放大；⑥ 的色階三個距離共用（以 6.7 m 的 99 百分位為上限、平方根色階），越亮越清晰。對焦在 15 m，所以車越近越模糊、σ 越小。
- 注意：論文中遮罩用 SAM 2（2024）；此處用資料集手繪遮罩（與實驗數字一致）。全自動遮罩（YOLO11 → SAM）見 GUI 與 RESULTS 補充 H。
- 影片：`video.mp4`（三個距離 × 7 步，{info['seconds']} s）、`video_short.mp4`（只有 3.7 m，{info_s['seconds']} s）。
- 產生：`python scripts/make_demo_assets.py --only C-C1`
""")


# ================================================================ C-E1 1×4 method comparison
E_METHODS = ['depthpro', 'unidepth', 'stmap', 'cadfds']


def e1_cell(k, c, t, s, fi, img, m, vmax, rot=False):
    ch = channels(c, t, s, rot)
    if k in ('unidepth', 'depthpro'):
        p = os.path.join(CACHE, f'{k}_{c}_T{t:02d}_{s:02d}{"_rot" if rot else ""}.npy')
        a = depth_vis(np.load(p), img.shape, ch['mask']) if os.path.exists(p) else img.copy()
    elif k == 'stmap':
        g = cv2.cvtColor(ch['img'], cv2.COLOR_BGR2GRAY); a = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        ys, xs = np.nonzero(m)   # STMAP: sampling box from the un-rotated mask (cannot re-locate the car)
        cv2.rectangle(a, (xs.min(), ys.min()), (xs.max(), ys.max()), (160, 200, 70), 6)
    else:
        a = heat(ch['lapw'], ch['mask'], ch['img'], vmax)
        cs, _ = cv2.findContours(ch['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(a, cs, -1, (40, 180, 240), 3)
    return a


def asset_E1():
    code = 'C-E1_method_compare'; fo = os.path.join(OUT, code); os.makedirs(os.path.join(fo, 'keyframes'), exist_ok=True)
    print('C-E1 method comparison', flush=True)
    vmax = np.percentile(np.abs(channels('red', 12)['lapw'][channels('red', 12)['mask']]), 99)
    cw, chh, gap, X0, Y0 = 450, 338, 18, 42, 172
    rows, segs = [], {}
    for c in ('red', 'green', 'blue', 'black'):
        hist, frames = [], []
        xs_all = [f"{load(c, t, s)[0]['gt']:.1f}" for t, s in SEQ]
        for k, (t, s) in enumerate(SEQ):
            fi, img, m = load(c, t, s); P = fi['pred']
            r = dict(color=c, frame=k, T=t, shot=s, gt_m=fi['gt'])
            for mth in E_METHODS:
                r[f'{mth}_m'] = round(P[mth], 4); r[f'{mth}_err_m'] = round(P[mth] - fi['gt'], 4)
            r['unidepth_raw_m'] = round(P['unidepth_raw'], 4); r['depthpro_raw_m'] = round(P['depthpro_raw'], 4)
            rows.append(r); hist.append({'gt': fi['gt'], **{mth: abs(P[mth] - fi['gt']) for mth in E_METHODS}})
            im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
            header(d, f'同一張影像、四種方法 · {COLOR_ZH[c]} · 真實距離 {fi["gt"]:.1f} m',
                   '深度模型（Depth Pro、UniDepthV2）已用同一校正集做線性校正（公平比較）；STMAP 需兩張不同光圈影像')
            for j, mth in enumerate(E_METHODS):
                x = X0 + j * (cw + gap)
                a = e1_cell(mth, c, t, s, fi, img, m, vmax)
                im.paste(bgr2pil(fit(a, cw, chh)), (x, Y0))
                if mth == 'cadfds':
                    d.rectangle([x - 4, Y0 - 52, x + cw + 4, Y0 + chh + 4], outline=COL['cadfds'], width=4)
                text(d, (x, Y0 - 44), NAME[mth], 25, COL[mth], True)
                e = P[mth] - fi['gt']
                rbox(d, (x + 8, Y0 + chh - 78, x + 300, Y0 + chh - 8), fill=(0, 0, 0), r=10)
                text(d, (x + 20, Y0 + chh - 62), f'{P[mth]:.2f} m', 30, TXT, True, 'lm')
                text(d, (x + 20, Y0 + chh - 26), f'誤差 {e:+.2f} m', 22, err_col(e), True, 'lm')
                text(d, (x, Y0 + chh + 12), INPUTS[mth], 17, SUB)
                if mth in ('unidepth', 'depthpro'):
                    text(d, (x, Y0 + chh + 36), f'原始輸出 {P[mth + "_raw"]:.2f} m → 線性校正 {P[mth]:.2f} m', 17, SUB)
            plot_curve(d, (X0, 615, W - X0, 1035), hist, k, xs_all, ymax=max(2.0, np.ceil(max(max(h[m_] for m_ in E_METHODS) for h in hist) * 2) / 2),
                       title='逐幀絕對誤差（m）', methods=E_METHODS, legend=True, show_gt=False)
            footer(d, FOOT + ' · 深度圖色階 0–10 m（紅 = 近）')
            frames.append(im)
        segs[c] = frames
    # end card from the 480-image comparison
    end = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(end)
    text(d, (W / 2, 120), '測試集 480 張整體 RMSE（紅／綠／藍／黑 × 1.2–6.7 m）', 40, TXT, True, 'ma')
    tab = [('cadfds', 'opt', '單張影像', RMSE480['opt'], None), ('unidepth', 'unidepth_lin', '單張影像', RMSE480['unidepth_lin'], RMSE480['unidepth']),
           ('stmap', 'stmap', '兩張影像', RMSE480['stmap'], None), ('depthpro', 'depthpro_lin', '單張影像', RMSE480['depthpro_lin'], RMSE480['depthpro'])]
    tab.sort(key=lambda r: r[3])
    for i, (mth, kk, inp, v_, raw) in enumerate(tab):
        y = 230 + i * 130; rbox(d, (360, y, 1560, y + 108), outline=COL['cadfds'] if mth == 'cadfds' else None)
        text(d, (400, y + 54), NAME[mth], 34, COL[mth], True, 'lm'); text(d, (900, y + 54), inp, 26, SUB, False, 'lm')
        text(d, (1180, y + 54), f'{v_:.3f} m', 40, TXT, True, 'lm')
        if raw is not None:
            text(d, (1400, y + 54), f'原始 {raw:.2f} m', 22, SUB, False, 'lm')
    text(d, (W / 2, 790), '近距離（3.7 m 以內）CA-DFDS 較準；遠距離 UniDepthV2（線性校正後）較準，兩者互補。', 28, TXT, False, 'ma')
    text(d, (W / 2, 840), '深度模型原始輸出未校正時誤差較大；此處依同一校正集做線性校正後比較。', 24, SUB, False, 'ma')
    footer(d, '數字出處 results/method_comparison.json（scripts/compare_methods.py） · CARLA 模擬資料')
    v = Video(os.path.join(fo, 'video.mp4'))
    for c in ('red', 'green', 'blue', 'black'):
        for i, fr in enumerate(segs[c]):
            v.add(fr, 0.9)
            nxt = segs[c][i + 1] if i + 1 < len(segs[c]) else None
            if nxt is not None: v.fade(fr, nxt, 0.15)
    v.add(end, 3.0); info = v.close()
    vs = Video(os.path.join(fo, 'video_short.mp4'))
    for i, fr in enumerate(segs['red']):
        vs.add(fr, 0.85)
        if i + 1 < 12: vs.fade(fr, segs['red'][i + 1], 0.12)
    vs.add(end, 2.0); info_s = vs.close()
    save_jpg(segs['red'][0], os.path.join(fo, 'poster.jpg'))
    for i in (0, 5, 11):
        save_png(segs['red'][i], os.path.join(fo, 'keyframes', f'red_{i:02d}.png'))
    save_png(end, os.path.join(fo, 'keyframes', 'end_card.png'))
    # 12 distances × 4 methods panels (DEMO_PLAN D4: results/demo/compare/<dist>_<method>.png)
    pd_ = os.path.join(fo, 'panels'); os.makedirs(pd_, exist_ok=True)
    for t, s in SEQ:
        fi, img, m = load('red', t, s)
        for mth in E_METHODS:
            a = e1_cell(mth, 'red', t, s, fi, img, m, vmax); im = bgr2pil(fit(a, 1600, 1200)); d = ImageDraw.Draw(im)
            e = fi['pred'][mth] - fi['gt']
            rbox(d, (24, 1060, 900, 1176), fill=(0, 0, 0), r=12)
            text(d, (44, 1092), f'{NAME[mth]}  {fi["pred"][mth]:.2f} m', 40, COL[mth], True, 'lm')
            text(d, (44, 1146), f'真實 {fi["gt"]:.1f} m · 誤差 {e:+.2f} m', 30, err_col(e), False, 'lm')
            save_jpg(im, os.path.join(pd_, f'{fi["gt"]:.1f}m_{mth}.jpg'), 88)
    write_csv(os.path.join(fo, 'per_frame.csv'), rows)
    summ = {'asset': 'C-E1 1x4 method comparison', 'frames': len(rows), 'distance_range_m': [1.2, 6.7], 'colors': ['red', 'green', 'blue', 'black'],
            'sequence_rmse_m': {mth: round(rmse([r[f'{mth}_err_m'] for r in rows]), 4) for mth in E_METHODS},
            'sequence_rmse_m_by_color': {c: {mth: round(rmse([r[f'{mth}_err_m'] for r in rows if r['color'] == c]), 4) for mth in E_METHODS}
                                         for c in ('red', 'green', 'blue', 'black')},
            'test480_rmse_m': {k: RMSE480[k] for k in ('opt', 'stmap', 'unidepth', 'unidepth_lin', 'depthpro', 'depthpro_lin')},
            'methods': {m_: NAME_EN[m_] for m_ in E_METHODS}, 'video': info, 'video_short': info_s}
    write_json(os.path.join(fo, 'summary.json'), summ)
    readme(fo, 'C-E1 四方法同步比較（1×4）', f"""
- 同一張影像、同一幀，四格：Depth Pro（2024）、UniDepthV2（2025）、STMAP（1994）、CA-DFDS（本作品，金框）。每格左下角為該方法估測距離與誤差；下方為逐幀絕對誤差曲線。
- 影片 `video.mp4`：紅→綠→藍→黑，每色 6.7 m → 1.2 m 共 12 個距離（shot01），{info['seconds']} s；`video_short.mp4` 只有紅色車，{info_s['seconds']} s。片尾為 480 張整體 RMSE（`results/method_comparison.json`）。
- `panels/<距離>_<方法>.jpg`：紅色車 12 個距離 × 4 方法單格大圖（1600×1200），對應 DEMO_PLAN D4。
- `per_frame.csv`：48 幀 × 各方法估測、誤差，以及深度模型原始輸出（網頁畫曲線用）。`summary.json`：這 48 幀與 480 張的 RMSE。
- 各方法設定：
  - **CA-DFDS**：單張 f/1.4 影像＋遮罩，論文權重，校正 `opt`。格內為清晰度熱圖。
  - **STMAP**（Subbarao & Surya 1994，空間域 DFD）：需要 f/1.4 與 f/22 **兩張影像**；沒有分割能力，在遮罩外框內整塊計算（`scripts/stmap_eval.py`，與學長 `stmap_dfds_rotation_eval.py` 同公式），二次校正曲線（同一校正集）。格內綠框為取樣框。
  - **UniDepthV2 / Depth Pro**：官方權重、未重新訓練，相機焦距 1137.8 px，取遮罩內深度中位數，再以同一校正集做線性校正 d = a·pred + b。格內為深度圖（0–10 m，紅 = 近），白線為遮罩。原始輸出也列在 CSV。
- 480 張整體 RMSE：CA-DFDS {RMSE480['opt']:.3f} m、UniDepthV2 線性校正 {RMSE480['unidepth_lin']:.3f} m（原始 {RMSE480['unidepth']:.3f}）、STMAP {RMSE480['stmap']:.3f} m、Depth Pro 線性校正 {RMSE480['depthpro_lin']:.3f} m（原始 {RMSE480['depthpro']:.3f}）。
- 解讀口徑（AGENTS.md）：公平比較時 CA-DFDS 與 UniDepthV2 相當；近距離 CA-DFDS 較準、遠距離 UniDepthV2 較準。不要寫「大幅勝過深度模型」。
- 產生：`../.venv/Scripts/python.exe baselines/demo_deep.py`（深度圖與旋轉推論）→ `python scripts/make_demo_assets.py --only C-E1`
""")


# ================================================================ C-F1 before / after
def asset_F1():
    code = 'C-F1_before_after'; fo = os.path.join(OUT, code); os.makedirs(os.path.join(fo, 'keyframes'), exist_ok=True)
    print('C-F1 before/after', flush=True)
    # --- (1) rotation 180°: whole test set
    from multiprocessing import Pool
    tests = [(c, t, s) for c in ('red', 'green', 'blue', 'black') for t in range(1, 13) for s in range(1, 11)]
    paths = []
    for c, t, s in tests:
        f = sorted(glob.glob(os.path.join(DATA, f'firetruck_{c}', DS, f'T{t:02d}', 'R', '*.png')))[s - 1]
        paths.append((f, os.path.join(MASK_DIR, f'T{t}.npy')))
    cache = os.path.join(OUT, 'rotation_cadfds.npz')
    if not os.path.exists(cache):
        with Pool(16) as pool:
            cov = pool.map(_rot_cov, paths, chunksize=2)
        np.savez(cache, path=np.array([p for p, _ in paths]), cov_rot=np.array(cov))
    z = np.load(cache); covr = {key(p): cv for p, cv in zip(z['path'], z['cov_rot'])}
    rows = []
    for (c, t, s), (f, _) in zip(tests, paths):
        idx, zz = feats(c); i = idx[key(f)]; gt = float(zz['dist_cm'][i]) / 100; d = DEEP[key(f)]; r = ROT[key(f)]
        so = C.sigma_eq5(zz['cov'][i], CAL['opt']['weights']); sr = C.sigma_eq5(covr[key(f)], CAL['opt']['weights'])
        rows.append(dict(color=c, T=t, shot=s, gt_m=gt, cadfds_m=round(poly('opt', so), 4), cadfds_rot_m=round(poly('opt', sr), 4),
                         stmap_m=round(poly('stmap', STM[key(f)][0]), 4), stmap_rot_m=round(poly('stmap', STM[key(f)][1]), 4),
                         unidepth_m=round(poly('unidepth_lin', float(d['unidepth_m'])), 4), unidepth_rot_m=round(poly('unidepth_lin', float(r['unidepth_rot_m'])), 4),
                         depthpro_m=round(poly('depthpro_lin', float(d['depthpro_m'])), 4), depthpro_rot_m=round(poly('depthpro_lin', float(r['depthpro_rot_m'])), 4),
                         unidepth_raw_m=float(d['unidepth_m']), unidepth_rot_raw_m=float(r['unidepth_rot_m']),
                         depthpro_raw_m=float(d['depthpro_m']), depthpro_rot_raw_m=float(r['depthpro_rot_m'])))
    write_csv(os.path.join(fo, 'rotation_per_image.csv'), rows)
    rs = {}
    for mth in ('cadfds', 'stmap', 'unidepth', 'depthpro', 'unidepth_raw', 'depthpro_raw'):
        a_ = np.array([r[f'{mth}_m' if not mth.endswith('raw') else mth.replace('_raw', '_raw_m')] - r['gt_m'] for r in rows])
        b_ = np.array([r[f'{mth}_rot_m' if not mth.endswith('raw') else mth.replace('_raw', '_rot_raw_m')] - r['gt_m'] for r in rows])
        rs[mth] = dict(rmse_orig_m=round(rmse(a_), 4), rmse_rot_m=round(rmse(b_), 4), delta_m=round(rmse(b_) - rmse(a_), 4),
                       std_orig_m=round(float(np.std(a_)), 4), std_rot_m=round(float(np.std(b_)), 4))
    rs['_max_abs_cadfds_change_m'] = round(float(max(abs(r['cadfds_rot_m'] - r['cadfds_m']) for r in rows)), 6)
    # pair images (red, every distance): original | rotated, each with 4 methods
    vmax = np.percentile(np.abs(channels('red', 12)['lapw'][channels('red', 12)['mask']]), 99)
    rot_frames = []
    for t, s in SEQ:
        fi, img, m = load('red', t, s); rr = next(r for r in rows if r['color'] == 'red' and r['T'] == t and r['shot'] == s)
        im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
        header(d, f'旋轉 180° 穩健性 · 紅色車 · 真實距離 {fi["gt"]:.1f} m', '同一張影像旋轉 180°（不是鏡射）；遮罩跟著旋轉（= 分割重新找到車）；STMAP 無分割，取樣框留在原位')
        for j, (lab, rot) in enumerate((('原始影像', False), ('旋轉 180°', True))):
            x = 40 + j * 930; a = overlay_mask(channels('red', t, s, rot)['img'], channels('red', t, s, rot)['mask'])
            im.paste(bgr2pil(fit(a, 880, 660)), (x, 140)); text(d, (x, 104), lab, 26, TXT, True)
            for i, mth in enumerate(E_METHODS[::-1]):
                v_ = rr[f'{mth}_rot_m' if rot else f'{mth}_m']; e = v_ - fi['gt']; y = 820 + i * 52
                text(d, (x, y), NAME[mth], 24, COL[mth], True, 'lm'); text(d, (x + 330, y), f'{v_:.2f} m', 26, TXT, True, 'lm')
                text(d, (x + 480, y), f'誤差 {e:+.2f} m', 24, err_col(e), False, 'lm')
        footer(d, FOOT)
        rot_frames.append((fi['gt'], im))
    end = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(end)
    text(d, (W / 2, 130), '旋轉 180° 前後 RMSE（480 張）', 44, TXT, True, 'ma')
    for i, mth in enumerate(['cadfds', 'stmap', 'unidepth', 'depthpro']):
        y = 250 + i * 130; rbox(d, (300, y, 1620, y + 108), outline=COL['cadfds'] if mth == 'cadfds' else None)
        text(d, (340, y + 54), NAME[mth], 32, COL[mth], True, 'lm')
        text(d, (900, y + 54), f"{rs[mth]['rmse_orig_m']:.3f} → {rs[mth]['rmse_rot_m']:.3f} m", 36, TXT, True, 'lm')
        dd_ = rs[mth]['delta_m']; text(d, (1360, y + 54), f'{dd_:+.3f} m', 32, err_col(dd_ if abs(dd_) > 0.005 else 0), True, 'lm')
    text(d, (W / 2, 800), 'CA-DFDS 只看遮罩內的模糊程度，與構圖無關，旋轉後結果不變。', 28, TXT, False, 'ma')
    text(d, (W / 2, 850), '深度模型依賴場景構圖（地面、天空位置）；STMAP 無法重新找到車。深度模型為線性校正後數值。', 24, SUB, False, 'ma')
    footer(d, '數字出處 results/demo/C-F1_before_after/rotation_summary.json · CARLA 模擬資料')
    write_json(os.path.join(fo, 'rotation_summary.json'), rs)
    # --- (2) grayscale vs CIECAM02 sharpness maps
    gc_frames, gc_vals = [], {}
    for t, tag in ((1, '近'), (6, '中'), (12, '遠')):
        for c in ('red', 'black'):
            fi, img, m = load(c, t); ch = channels(c, t); bx = bbox(m, 0.3)
            vg = np.percentile(np.abs(ch['lapg'][m]), 99); vw = np.percentile(np.abs(ch['lapw'][m]), 99)
            ga = crop(heat(ch['lapg'], m, img, vg), bx); ca = crop(heat(ch['lapw'], m, img, vw), bx)
            pg = fi['pred']['gray']; pc = fi['pred']['cadfds']
            pa = bgr2pil(fit(ga, 1600, 1200)); pb = bgr2pil(fit(ca, 1600, 1200))
            for im_, lab, p_ in ((pa, '灰階拉普拉斯', pg), (pb, 'CIECAM02 加權拉普拉斯（CA-DFDS）', pc)):
                d = ImageDraw.Draw(im_); rbox(d, (24, 24, 60 + d.textlength(lab, font=F(42, True)), 100), fill=(0, 0, 0), r=12)
                text(d, (44, 62), lab, 42, TXT, True, 'lm')
                s_ = f'估測 {p_:.2f} m · 真實 {fi["gt"]:.1f} m · 誤差 {p_ - fi["gt"]:+.2f} m'
                rbox(d, (24, 1110, 60 + d.textlength(s_, font=F(34)), 1176), fill=(0, 0, 0), r=12); text(d, (44, 1143), s_, 34, err_col(p_ - fi['gt']), False, 'lm')
            nm = f'{c}_{fi["gt"]:.1f}m'
            save_png(pa, os.path.join(fo, 'gray_vs_ciecam', f'{nm}_gray.png')); save_png(pb, os.path.join(fo, 'gray_vs_ciecam', f'{nm}_ciecam.png'))
            gc_vals[nm] = dict(gt_m=fi['gt'], gray_m=round(pg, 4), cadfds_m=round(pc, 4))
            gc_frames.append((nm, pa, pb, fi))
    # wipe video: grayscale → CIECAM02 slider
    gcv = []
    for nm, pa, pb, fi in gc_frames:
        A, B = np.array(pa.resize((1280, 960))), np.array(pb.resize((1280, 960)))
        seq = []
        for i in range(int(2.0 * FPS)):
            u = 0.5 - 0.5 * np.cos(np.pi * i / (2.0 * FPS - 1)); xcut = int(1280 * u)
            comp = A.copy(); comp[:, :xcut] = B[:, :xcut]
            im = Image.new('RGB', (W, H), BG); im.paste(Image.fromarray(comp).resize((1240, 930)), (40, 112)); d = ImageDraw.Draw(im)
            d.line([40 + xcut * 1240 // 1280, 112, 40 + xcut * 1240 // 1280, 1042], fill=(255, 255, 255), width=4)
            header(d, f'灰階 vs CIECAM02 清晰度 · {COLOR_ZH[fi["color"]]} {fi["gt"]:.1f} m', '左側滑入：CA-DFDS 的 CIECAM02 加權拉普拉斯；右側：傳統灰階拉普拉斯（各自色階）')
            for j, (lab, k_) in enumerate((('灰階 DFD', 'gray'), ('CA-DFDS', 'cadfds'))):
                y = 200 + j * 170; rbox(d, (1360, y, 1880, y + 150))
                text(d, (1380, y + 20), lab, 26, COL[k_], True); v_ = fi['pred'][k_]
                text(d, (1380, y + 64), f'{v_:.2f} m', 44, TXT, True); text(d, (1600, y + 84), f'誤差 {v_ - fi["gt"]:+.2f}', 24, err_col(v_ - fi['gt']))
            text(d, (1360, 560), '480 張 RMSE', 24, SUB)
            text(d, (1360, 600), f"灰階 {RMSE480['gray']:.3f} m", 30, COL['gray'], True)
            text(d, (1360, 646), f"CA-DFDS {RMSE480['opt']:.3f} m", 30, COL['cadfds'], True)
            text(d, (1360, 692), f"只用 J {RMSE480['jonly']:.3f} m", 24, SUB)
            footer(d, FOOT)
            seq.append(pil2bgr(im))
        gcv.append(seq)
    v = Video(os.path.join(fo, 'video.mp4'))
    for i, (g, fr) in enumerate(rot_frames):
        v.add(fr, 1.2)
        if i + 1 < len(rot_frames): v.fade(fr, rot_frames[i + 1][1], 0.15)
    v.fade(rot_frames[-1][1], end, 0.3); v.add(end, 3.0)
    for seq in gcv:
        for fr in seq: v.add(fr, n=1)
        v.add(seq[-1], 0.8)
    info = v.close()
    vs = Video(os.path.join(fo, 'video_short.mp4'))
    for g, fr in rot_frames[::3]:
        vs.add(fr, 1.4)
    vs.add(end, 2.4)
    for seq in gcv[2:4]:
        for fr in seq: vs.add(fr, n=1)
        vs.add(seq[-1], 0.6)
    info_s = vs.close()
    save_jpg(rot_frames[5][1], os.path.join(fo, 'poster.jpg'))
    save_png(rot_frames[5][1], os.path.join(fo, 'keyframes', 'rotation_3.7m.png')); save_png(end, os.path.join(fo, 'keyframes', 'rotation_rmse.png'))
    # pair PNGs for the web drag-slider: same size, same crop
    pr = os.path.join(fo, 'rotation_pairs'); os.makedirs(pr, exist_ok=True)
    for t, s in SEQ:
        fi = load('red', t, s)[0]
        for rot in (False, True):
            a = overlay_mask(channels('red', t, s, rot)['img'], channels('red', t, s, rot)['mask'])
            save_jpg(bgr2pil(fit(a, 1600, 1200)), os.path.join(pr, f'red_{fi["gt"]:.1f}m_{"rot180" if rot else "orig"}.jpg'), 88)
    write_csv(os.path.join(fo, 'per_frame.csv'), [r for r in rows if r['color'] == 'red' and r['shot'] == 1])
    write_json(os.path.join(fo, 'summary.json'), dict(asset='C-F1 before/after', rotation_480=rs, gray_vs_ciecam=gc_vals,
               test480_rmse_m={k: RMSE480[k] for k in ('gray', 'opt', 'jonly')}, video=info, video_short=info_s))
    readme(fo, 'C-F1 前後對照（旋轉 180°；灰階 vs CIECAM02）', f"""
**① 旋轉 180°**（論文 Table 4 的設定；測試集 480 張）

| 方法 | 原始 RMSE | 旋轉後 RMSE | 變化 |
|---|---|---|---|
""" + '\n'.join(f"| {NAME[m_]} | {rs[m_]['rmse_orig_m']:.3f} m | {rs[m_]['rmse_rot_m']:.3f} m | {rs[m_]['delta_m']:+.3f} m |" for m_ in ('cadfds', 'stmap', 'unidepth', 'depthpro'))
           + f"""
| UniDepthV2 原始輸出（未校正） | {rs['unidepth_raw']['rmse_orig_m']:.3f} m | {rs['unidepth_raw']['rmse_rot_m']:.3f} m | {rs['unidepth_raw']['delta_m']:+.3f} m |
| Depth Pro 原始輸出（未校正） | {rs['depthpro_raw']['rmse_orig_m']:.3f} m | {rs['depthpro_raw']['rmse_rot_m']:.3f} m | {rs['depthpro_raw']['delta_m']:+.3f} m |

- 設定：影像以 `cv2.ROTATE_180` 旋轉（不是鏡射）。CA-DFDS 與深度模型的遮罩跟著旋轉（代表分割模型重新找到車）；深度模型用原本的線性校正。
  STMAP 沒有分割能力，取樣框留在原位置（與學長 `stmap_dfds_rotation_eval.py` 相同）。
- UniDepthV2 原始輸出旋轉後 RMSE 反而下降（{rs['unidepth_raw']['rmse_orig_m']:.2f} → {rs['unidepth_raw']['rmse_rot_m']:.2f} m），是因為原本整體高估約 1.8 m、旋轉後估測值整體變小，偏差剛好抵銷；
  誤差標準差由 {rs['unidepth_raw']['std_orig_m']:.2f} 增為 {rs['unidepth_raw']['std_rot_m']:.2f} m，穩定度其實變差。套用校正後（公平比較）RMSE 由 {rs['unidepth']['rmse_orig_m']:.3f} 增為 {rs['unidepth']['rmse_rot_m']:.3f} m。
- CA-DFDS 重新計算了 480 張旋轉影像的 CIECAM02 特徵（`rotation_cadfds.npz`），估測距離最大變化 {rs['_max_abs_cadfds_change_m']} m：3×3 高斯與拉普拉斯對 180° 旋轉對稱，遮罩內的統計量不變，所以結果不受構圖影響。
- 深度模型推論：`baselines/demo_deep.py`（`results/demo/rotation_deep.csv`）。逐張結果：`rotation_per_image.csv`；彙總：`rotation_summary.json`。
- `rotation_pairs/red_<距離>_orig.jpg`／`_rot180.jpg`：同尺寸成對圖，可做網頁拖曳對照。

**② 灰階 vs CIECAM02 清晰度**

- `gray_vs_ciecam/<車色>_<距離>_gray.png`／`_ciecam.png`：同一裁切、同尺寸（1600×1200）；紅色與黑色車 × 1.2／3.7／6.7 m。兩張各自用遮罩內 99 百分位做色階上限。
- 480 張 RMSE（手繪遮罩）：灰階 {RMSE480['gray']:.3f} m、CA-DFDS {RMSE480['opt']:.3f} m、只用 J {RMSE480['jonly']:.3f} m。
  **注意**：改善主要來自明度 J（只用 J 已達 {RMSE480['jonly']:.3f} m），色度 a_c、b_c 的貢獻很小（RESULTS 補充 E、M）；黑色車上 CA-DFDS 改善最明顯（灰階 0.541 → 0.468 m）。

- 影片 `video.mp4`：紅色車 12 個距離的旋轉對照 → RMSE 卡 → 灰階／CIECAM02 拖曳滑桿（{info['seconds']} s）。`video_short.mp4`：{info_s['seconds']} s。
- 產生：`../.venv/Scripts/python.exe baselines/demo_deep.py` → `python scripts/make_demo_assets.py --only C-F1`
""")


def _rot_cov(a):
    f, mp = a
    img = C.imread(f); m = C.load_mask(mp, img.shape)
    img, m = cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(m.astype(np.uint8), cv2.ROTATE_180) > 0
    ci, cm = C.crop_to_mask(img, m)
    return C.laplacian_cov(ci, cm)


# ================================================================ C-B1 camera setup (diagram from the generator's parameters)
CAM = dict(lens_mm=40.0, sensor_mm=36.0, width_px=2048, height_px=1536, fstops=[1.4, 22.0], focus_m=15.0,
           camera_height_m=0.8, camera_x_offset_m=4.0, distances_m='1.0–7.0 (calibration), 1.2–6.7 (test, masks T1–T12)',
           source='Carla-0.10.0-Win64-Shipping/get_dataset2.py (LENS_FOCAL_LENGTH_MM, FOCUS_DISTANCE_CM, CAMERA_HEIGHT, CAMERA_X_OFFSET)')


def asset_B1():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
    plt.rcParams['font.family'] = ['Microsoft JhengHei']; plt.rcParams['axes.unicode_minus'] = False
    code = 'C-B1_camera_setup'; fo = os.path.join(OUT, code); os.makedirs(os.path.join(fo, 'keyframes'), exist_ok=True)
    print('C-B1 camera setup', flush=True)
    fov = 2 * np.degrees(np.arctan(CAM['sensor_mm'] / (2 * CAM['lens_mm']))); vfov = 2 * np.degrees(np.arctan(CAM['sensor_mm'] * 0.75 / (2 * CAM['lens_mm'])))
    fig, axs = plt.subplots(2, 1, figsize=(16, 10), dpi=120, gridspec_kw=dict(height_ratios=[1, 1.6]))
    fig.patch.set_facecolor('#0e1626')
    for ax in axs:
        ax.set_facecolor('#0e1626'); ax.tick_params(colors='#a0afc8'); [s.set_color('#46587a') for s in ax.spines.values()]
    # side view
    ax = axs[0]; ax.set_title('側視圖（Side view）', color='white', fontsize=17, loc='left')
    ax.add_patch(Rectangle((-4.6, 0.25), 4.8, 1.2, color='#3a4a6a')); ax.text(-2.3, 0.85, '本車（ego）', color='white', ha='center', fontsize=13)
    ax.plot([0], [CAM['camera_height_m']], 'o', color='#f0b428', ms=12); ax.text(0.05, 1.35, f"相機 h = {CAM['camera_height_m']} m\n40 mm · f/1.4 ＋ f/22", color='#f0b428', fontsize=12)
    h0 = CAM['camera_height_m']; tv = np.tan(np.radians(vfov / 2))
    ax.plot([0, 9.5], [h0, h0 + 9.5 * tv], color='#f0b428', lw=1, ls='--'); ax.plot([0, h0 / tv], [h0, 0], color='#f0b428', lw=1, ls='--')
    for dd, al in ((1.2, 0.95), (3.7, 0.6), (6.7, 0.35)):   # rear face of the car (thick) + body outline
        ax.add_patch(Rectangle((dd, 0.15), 2.2, 2.8, fill=False, ec='#c84040', lw=1.5, alpha=al, ls=':'))
        ax.plot([dd, dd], [0.15, 2.95], color='#e05050', lw=6, alpha=al); ax.text(dd, 3.15, f'{dd} m', color='white', fontsize=12, ha='center')
    ax.text(4.0, 2.0, '前車（消防車）· 12 個距離擇 3 示意', color='#e8a0a0', fontsize=11, ha='center')
    ax.axhline(0, color='#8090a8', lw=2)
    for v in range(1, 8):
        ax.plot([v, v], [-0.05, -0.25], color='white'); ax.text(v, -0.55, f'{v} m', color='white', ha='center', fontsize=11)
    ax.annotate('', xy=(13.6, 1.6), xytext=(10.5, 1.6), arrowprops=dict(arrowstyle='->', color='#5af', lw=2))
    ax.text(10.6, 1.8, '對焦平面在 15 m（後對焦）', color='#5af', fontsize=13)
    ax.set_xlim(-5, 14); ax.set_ylim(-0.9, 3.8); ax.set_aspect('equal'); ax.set_yticks([]); ax.set_xticks([])
    # top view
    ax = axs[1]; ax.set_title('俯視圖（Top view）：距離 = 相機到前車車尾', color='white', fontsize=17, loc='left')
    ax.add_patch(Rectangle((-4.6, -1.0), 4.8, 2.0, color='#3a4a6a'))
    ax.add_patch(Polygon([[0, 0], [15, 15 * np.tan(np.radians(fov / 2))], [15, -15 * np.tan(np.radians(fov / 2))]], color='#f0b428', alpha=0.08))
    for s_ in (-1, 1):
        ax.plot([0, 15], [0, s_ * 15 * np.tan(np.radians(fov / 2))], color='#f0b428', lw=1, ls='--')
    ax.text(6.5, 4.0, f'水平視角 {fov:.1f}°', color='#f0b428', fontsize=12)
    for dd in np.arange(1.0, 7.01, 0.5):
        ax.plot([dd, dd], [-1.4, 1.4], color='#6a7a98', lw=0.8)
    for dd in (1.2, 6.7):
        ax.add_patch(Rectangle((dd, -1.25), 2.2, 2.5, color='#c84040', alpha=0.85 if dd < 3 else 0.35))
        ax.text(dd + 0.1, -1.9, f'{dd} m', color='white', fontsize=12)
    ax.plot([15, 15], [-6, 6], color='#5af', lw=3); ax.text(14.6, 6.3, '對焦 15 m', color='#5af', fontsize=13, ha='right')
    ax.add_patch(Rectangle((1.0, -6.6), 6.0, 0.5, color='#2d6a4f')); ax.text(4.0, -7.3, '驗證範圍 1–7 m（車越近越模糊、無前後歧義）', color='#7fd1a8', ha='center', fontsize=13)
    ax.plot([0], [0], 'o', color='#f0b428', ms=12)
    ax.set_xlim(-5, 16); ax.set_ylim(-7.8, 7.2); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('CA-DFDS 拍攝配置（CARLA 模擬，get_dataset2.py 參數）', color='white', fontsize=20, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p = os.path.join(fo, 'camera_setup.png'); fig.savefig(p, facecolor=fig.get_facecolor()); plt.close(fig)
    # camera views at three distances
    vw = Image.new('RGB', (1600, 440), BG); d = ImageDraw.Draw(vw)
    for i, t in enumerate((1, 6, 12)):
        fi, img, m = load('red', t); vw.paste(bgr2pil(fit(img, 520, 390)), (i * 540, 0)); text(d, (i * 540 + 10, 400), f'相機視角 · {fi["gt"]:.1f} m', 24, TXT, True)
    save_png(vw, os.path.join(fo, 'camera_views.png'))
    shutil.copy(p, os.path.join(fo, 'keyframes', 'camera_setup.png'))
    im = Image.new('RGB', (W, H), (14, 22, 38)); sp = Image.open(p).convert('RGB'); sp.thumbnail((W, H)); im.paste(sp, ((W - sp.width) // 2, 0)); save_jpg(im, os.path.join(fo, 'poster.jpg'))
    write_json(os.path.join(fo, 'summary.json'), dict(asset='C-B1 camera setup', camera=CAM, fov_h_deg=round(fov, 2), fov_v_deg=round(vfov, 2),
               focal_px=round(CAM['width_px'] * CAM['lens_mm'] / CAM['sensor_mm'], 1)))
    readme(fo, 'C-B1 拍攝角度／相機配置', f"""
- `camera_setup.png`：側視＋俯視示意圖（相機位置、視角、距離刻度、15 m 對焦平面）。依 `get_dataset2.py` 的參數繪製：40 mm 鏡頭、36 mm 感光元件（水平視角 {fov:.1f}°）、
  2048×1536、相機高 0.8 m、f/1.4（R 資料夾）與 f/22（L 資料夾）兩台相機同位置、對焦 15 m、前車在相機正前方，距離為相機到前車車尾。
- `camera_views.png`：實際相機視角（紅色車 1.2／3.7／6.7 m）。
- **尚缺**：CARLA 第三人稱側面追蹤視角影片與俯視全景（需開 CARLA 伺服器重新渲染），尚未產出；目前以示意圖代替。
- 示意圖中車輛大小僅示意，不按比例。
- 產生：`python scripts/make_demo_assets.py --only C-B1`
""")


# ================================================================ C-D1 defocus theory
def coc_px(d, s, N, f=40.0, pitch=36.0 / 2048):
    """Thin-lens blur-circle diameter in pixels; d, s in metres."""
    d, s = np.asarray(d, float) * 1000, s * 1000
    return (f * f / N) * np.abs(d - s) / (d * (s - f)) / pitch


def asset_D1():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei']; plt.rcParams['axes.unicode_minus'] = False
    code = 'C-D1_defocus_theory'; fo = os.path.join(OUT, code); os.makedirs(os.path.join(fo, 'keyframes'), exist_ok=True)
    print('C-D1 defocus theory', flush=True)
    ds = np.round(np.arange(0.5, 20.01, 0.05), 3)
    curves = {'f1.4_focus15m': coc_px(ds, 15, 1.4), 'f1.4_focus7m': coc_px(ds, 7, 1.4), 'f22_focus15m': coc_px(ds, 15, 22.0)}
    write_csv(os.path.join(fo, 'curves.csv'), [dict(distance_m=float(x), **{k: round(float(v[i]), 4) for k, v in curves.items()}) for i, x in enumerate(ds)])
    # measured sharpness per distance (calibration set, CA-DFDS σ) for the "theory vs data" panel
    meas = {}
    for c in ('white', 'gray', 'black'):
        z = np.load(R('features', f'cal_{c}.npz'), allow_pickle=True)
        s = C.sigma_eq5(z['cov'], CAL['opt']['weights']); dd = z['dist_cm'] / 100
        meas[c] = {f'{x:.1f}': round(float(np.mean(s[dd == x])), 4) for x in sorted(set(dd))}
    write_json(os.path.join(fo, 'summary.json'), dict(asset='C-D1 defocus theory', model='thin lens, blur-circle diameter c = f²/N · |d−s| / (d·(s−f)), in pixels (pitch 36 mm/2048)',
               lens_mm=40, pitch_mm=round(36 / 2048, 6), cases={'f1.4_focus15m': 'used (rear focus)', 'f1.4_focus7m': 'front focus: ambiguous around 7 m',
                                                                'f22_focus15m': 'small aperture: almost no blur (reference image for STMAP)'},
               coc_px_at={k: {f'{x}': round(float(coc_px(x, *(15, 1.4) if k == 'f1.4_focus15m' else (7, 1.4) if k == 'f1.4_focus7m' else (15, 22.0))), 2)
                              for x in (1.0, 1.2, 3.7, 6.7, 7.0)} for k in curves},
               measured_sigma_calibration_set=meas, note='theoretical thin-lens values; CARLA depth of field is a renderer approximation'))
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120); fig.patch.set_facecolor('#0e1626'); ax.set_facecolor('#0e1626')
    ax.tick_params(colors='#c0cce0', labelsize=14); [s.set_color('#46587a') for s in ax.spines.values()]
    ax.axvspan(1, 7, color='#2d6a4f', alpha=0.25); ax.text(4, 66, '驗證範圍 1–7 m', color='#7fd1a8', ha='center', fontsize=16)
    ax.plot(ds, curves['f1.4_focus15m'], color='#f0b428', lw=3.5, label='f/1.4，對焦 15 m（本作品：後對焦，1–7 m 單調）')
    ax.plot(ds, curves['f1.4_focus7m'], color='#ff6b6b', lw=2.5, ls='--', label='f/1.4，對焦 7 m（前對焦：7 m 前後模糊量相同 → 歧義）')
    ax.plot(ds, curves['f22_focus15m'], color='#9aa', lw=2.5, label='f/22，對焦 15 m（小光圈幾乎不模糊）')
    ax.set_xlim(0.5, 20); ax.set_ylim(0, 75); ax.set_xlabel('物體距離（m）', color='white', fontsize=16); ax.set_ylabel('模糊圈直徑（像素）', color='white', fontsize=16)
    ax.legend(fontsize=14, facecolor='#18233a', edgecolor='#46587a', labelcolor='white', loc='upper right')
    ax.set_title('薄透鏡理論：模糊量 vs 距離（40 mm 鏡頭、2048 px / 36 mm 感光元件）', color='white', fontsize=19, loc='left')
    fig.tight_layout(); fig.canvas.draw(); T = ax.transData; fh = fig.get_size_inches()[1] * fig.dpi; sc = 1280 / (fig.get_size_inches()[0] * fig.dpi)
    to_img = lambda xd, yd: (40 + T.transform((xd, yd))[0] * sc, 130 + (fh - T.transform((xd, yd))[1]) * sc)   # data → frame pixels
    p = os.path.join(fo, 'defocus_curves.png'); fig.savefig(p, facecolor=fig.get_facecolor()); plt.close(fig)
    shutil.copy(p, os.path.join(fo, 'keyframes', 'defocus_curves.png'))
    # animation: distance cursor 7 → 1 m, blur circles for focus 15 m vs 7 m and f/22
    v = Video(os.path.join(fo, 'video.mp4')); base = Image.open(p).convert('RGB').resize((1280, 720))
    first = None
    for dcur in np.linspace(7.0, 1.0, int(8 * FPS)):
        im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
        header(d, '為什麼對焦在 15 m？', '距離游標由 7 m 移到 1 m：後對焦（15 m）時車越近越模糊、一一對應；前對焦（7 m）在對焦點兩側會出現相同模糊量')
        im.paste(base, (40, 130))
        (xa, ya0), (_, ya1) = to_img(dcur, 0), to_img(dcur, 75)
        d.line([xa, ya0, xa, ya1], fill=(255, 255, 255), width=2)
        for kk, cc in (((15, 1.4), COL['cadfds']), ((7, 1.4), (255, 107, 107)), ((15, 22.0), (170, 180, 180))):
            px_, py_ = to_img(dcur, float(coc_px(dcur, *kk))); d.ellipse([px_ - 8, py_ - 8, px_ + 8, py_ + 8], fill=cc, outline=(255, 255, 255))
        cases = [('f/1.4 · 對焦 15 m', coc_px(dcur, 15, 1.4), COL['cadfds']), ('f/1.4 · 對焦 7 m', coc_px(dcur, 7, 1.4), (255, 107, 107)),
                 ('f/22 · 對焦 15 m', coc_px(dcur, 15, 22.0), (170, 180, 180))]
        text(d, (1360, 140), f'距離 {dcur:.2f} m', 44, TXT, True)
        for i, (lab, cpx, col) in enumerate(cases):
            y = 230 + i * 260; rbox(d, (1360, y, 1880, y + 240)); text(d, (1380, y + 16), lab, 24, col, True)
            text(d, (1380, y + 56), f'模糊圈 {float(cpx):.1f} px', 30, TXT, True)
            r_ = max(1.5, float(cpx) * 1.4); cx, cy = 1760, y + 140
            d.ellipse([cx - r_, cy - r_, cx + r_, cy + r_], fill=col)
        footer(d, '薄透鏡理論值（curves.csv）；CARLA 的景深是渲染器近似，實際模糊以影像量測的 σ 為準（summary.json measured_sigma_calibration_set）')
        if first is None: first = im
        v.add(im, n=1)
    v.add(im, 1.0); info = v.close()
    vs = Video(os.path.join(fo, 'video_short.mp4'))
    cap = cv2.VideoCapture(os.path.join(fo, 'video.mp4')); k = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if k % 2 == 0: vs.add(fr, n=1)
        k += 1
    cap.release(); info_s = vs.close()
    save_jpg(first, os.path.join(fo, 'poster.jpg'))
    # M3: f/1.4 (R folder) vs f/22 (L folder) of the same frame, same crop and size → drag-slider pairs
    pdir = os.path.join(fo, 'aperture_pairs'); os.makedirs(pdir, exist_ok=True)
    for t in (1, 6, 12):
        fi, img, m = load('red', t); l_img = C.imread(fi['path'].replace(os.sep + 'R' + os.sep, os.sep + 'L' + os.sep)); bx = bbox(m, 0.3)
        for lab, a_ in (('f1.4', img), ('f22', l_img)):
            im = bgr2pil(fit(crop(a_, bx), 1600, 1200)); d = ImageDraw.Draw(im)
            s_ = f'{"f/1.4（大光圈，本作品使用）" if lab == "f1.4" else "f/22（小光圈，STMAP 的參考影像）"} · 真實距離 {fi["gt"]:.1f} m'
            rbox(d, (24, 24, 60 + d.textlength(s_, font=F(40, True)), 100), fill=(0, 0, 0), r=12); text(d, (44, 62), s_, 40, TXT, True, 'lm')
            save_jpg(im, os.path.join(pdir, f'red_{fi["gt"]:.1f}m_{lab}.jpg'), 90)
    write_csv(os.path.join(fo, 'per_frame.csv'), [dict(distance_m=float(x), coc_f14_focus15m_px=round(float(coc_px(x, 15, 1.4)), 3),
                                                       coc_f14_focus7m_px=round(float(coc_px(x, 7, 1.4)), 3), coc_f22_focus15m_px=round(float(coc_px(x, 15, 22.0)), 3))
                                                  for x in np.round(np.arange(1.0, 7.01, 0.1), 2)])
    readme(fo, 'C-D1 散焦理論曲線（對焦 7 m vs 15 m；f/1.4 vs f/22）', f"""
- 薄透鏡模糊圈直徑 c = f²/N · |d − s| / (d·(s − f))，換算成像素（像素間距 36 mm / 2048）。f = 40 mm，N = 1.4 或 22，s = 15 m 或 7 m。
- `curves.csv`（0.5–20 m，每 0.05 m）可直接給網頁做互動曲線；`per_frame.csv` 為 1–7 m 每 0.1 m。
- `summary.json` 另附校正集實測的 CA-DFDS σ（每距離平均），可與理論曲線並列：理論模糊量越大 → 實測 σ 越小。
- 重點：對焦在 15 m 時，1–7 m 內模糊量隨距離單調變化，不會有「對焦點前後模糊量相同」的歧義；f/22 幾乎不模糊（STMAP 用它當參考影像）。
- 這是理論值，不是 CARLA 量測；CARLA 的景深是渲染器近似。
- 影片：距離游標 7 → 1 m，三種設定的模糊圈同步變化（`video.mp4` {info['seconds']} s，`video_short.mp4` {info_s['seconds']} s）。
- `aperture_pairs/red_<距離>_f1.4.jpg`／`_f22.jpg`：同一幀 f/1.4（R 資料夾）與 f/22（L 資料夾）實際影像，同裁切、同尺寸（1600×1200），可做拖曳對照（DEMO_PLAN M3）。
- 這份資料同時回應 CA-DFDS 網站 TODO「散焦理論曲線的相機參數」。
- 產生：`python scripts/make_demo_assets.py --only C-D1`
""")


# ================================================================ C-H1 leaderboard, C-I1 charts, C-M1 timing, C-K1 failures
def asset_H1():
    code = 'C-H1_leaderboard'; fo = os.path.join(OUT, code); os.makedirs(fo, exist_ok=True); print('C-H1 leaderboard', flush=True)
    rot = json.load(open(os.path.join(OUT, 'C-F1_before_after', 'rotation_summary.json'), encoding='utf-8')) if os.path.exists(os.path.join(OUT, 'C-F1_before_after', 'rotation_summary.json')) else {}
    meta = [  # key, name, year, venue, inputs, calibration, rotation key
        ('opt', 'CA-DFDS（本作品）', 2026, '—', '單張影像＋遮罩', '二次（校正集）', 'cadfds'),
        ('jonly', 'CA-DFDS 只用 J（消融）', 2026, '—', '單張影像＋遮罩', '二次（校正集）', None),
        ('reopt5', 'CA-DFDS 重新最佳化權重', 2026, '—', '單張影像＋遮罩', '二次（校正集）', None),
        ('gray', '灰階 DFD（拉普拉斯）', None, '傳統基準', '單張影像＋遮罩', '二次（校正集）', None),
        ('unidepth_lin', 'UniDepthV2（線性校正）', 2025, 'arXiv 2502.20110', '單張影像＋內參', '線性（校正集）', 'unidepth'),
        ('unidepth', 'UniDepthV2（原始）', 2025, 'arXiv 2502.20110', '單張影像＋內參', '無', 'unidepth_raw'),
        ('depthpro_lin', 'Depth Pro（線性校正）', 2024, 'arXiv 2410.02073 / ICLR 2025', '單張影像＋焦距', '線性（校正集）', 'depthpro'),
        ('depthpro', 'Depth Pro（原始）', 2024, 'arXiv 2410.02073 / ICLR 2025', '單張影像＋焦距', '無', 'depthpro_raw'),
        ('stmap', 'STMAP', 1994, 'IJCV 13(3)', '兩張影像（f/1.4＋f/22）', '二次（校正集）', 'stmap')]
    hb = MC['hand']; ab = MC.get('auto', {})
    rows = []
    for k, nm, yr, venue, inp, cal, rk in meta:
        rows.append(dict(method_key=k, method=nm, year=yr or '', venue=venue, inputs=inp, calibration=cal,
                         rmse_m=hb['overall_rmse_m'][k], mae_m=hb['overall_mae_m'][k], bias_m=hb['overall_bias_m'][k],
                         **{f'rmse_{c}_m': hb['per_color_rmse_m'][k][c] for c in ('red', 'green', 'blue', 'black')},
                         rmse_near_1_2_to_3_7_m=round(rmse([hb['per_distance_rmse_m'][k][g] for g in ('1.2', '1.7', '2.2', '2.7', '3.2', '3.7')]), 4),
                         rmse_far_4_2_to_6_7_m=round(rmse([hb['per_distance_rmse_m'][k][g] for g in ('4.2', '4.7', '5.2', '5.7', '6.2', '6.7')]), 4),
                         rmse_auto_mask_m=ab.get('overall_rmse_m', {}).get(k, ''),
                         rmse_rot180_m=rot.get(rk, {}).get('rmse_rot_m', '') if rk else '',
                         time_ms=MC['time_ms'].get(k, ''), ours=k == 'opt'))
    rows.sort(key=lambda r: r['rmse_m'])
    for i, r in enumerate(rows, 1):
        r['rank'] = i
    write_csv(os.path.join(fo, 'leaderboard.csv'), rows); write_json(os.path.join(fo, 'leaderboard.json'), dict(
        note='CARLA test set, 480 images (red/green/blue/black × T01–T12 × 10 shots, 1.2–6.7 m), hand masks unless stated. '
             'rmse_near/far = RMSE over the per-distance RMSEs. time_ms: CA-DFDS/gray/STMAP CPU single thread incl. image read; deep models RTX 5080 inference only.',
        source='results/method_comparison.json, results/demo/C-F1_before_after/rotation_summary.json', rows=rows))
    L = ['| 排名 | 方法 | 年份 | 出處 | 輸入 | RMSE (m) | 近 1.2–3.7 m | 遠 4.2–6.7 m | 自動遮罩 | 旋轉 180° | 時間 |', '|' + '---|' * 11]
    fmt = lambda v, f='{:.3f}': '' if v == '' else f.format(v)
    for r in rows:
        b = '**' if r['ours'] else ''
        L.append(f"| {r['rank']} | {b}{r['method']}{b} | {r['year']} | {r['venue']} | {r['inputs']} | {b}{r['rmse_m']:.3f}{b} | "
                 f"{r['rmse_near_1_2_to_3_7_m']:.3f} | {r['rmse_far_4_2_to_6_7_m']:.3f} | {fmt(r['rmse_auto_mask_m'])} | "
                 f"{fmt(r['rmse_rot180_m'])} | {fmt(r['time_ms'], '{:.0f} ms')} |")
    open(os.path.join(fo, 'leaderboard.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    readme(fo, 'C-H1 排行榜表（CARLA 480 張）', """
- `leaderboard.csv`／`leaderboard.json`（網頁可排序表格用）、`leaderboard.md`（簡報、README 用）。
- 欄位：整體 RMSE／MAE／偏差、各車色 RMSE、近距離（1.2–3.7 m）與遠距離（4.2–6.7 m）RMSE、全自動遮罩（YOLO11 → SAM）RMSE、旋轉 180° 後 RMSE、每張時間。
- 年份與出處：STMAP = Subbarao & Surya, “Depth from defocus: A spatial domain approach,” IJCV 13(3), 1994；
  Depth Pro = Bochkovskii et al., arXiv 2410.02073（ICLR 2025）；UniDepthV2 = Piccinelli et al., arXiv 2502.20110（2025）。
- 時間口徑不同：CA-DFDS／灰階／STMAP 為 CPU 單執行緒、含讀圖（`results/timing.json`）；深度模型為 RTX 5080 只計推論。CA-DFDS GPU 版見 C-M1。
- 20–110 cm 實拍結果（前期研究 DFDS 論文 Table 3/4）不在本表；本表全部是本機重新量測的 CARLA 結果。
- 數字出處：`results/method_comparison.json`（`scripts/compare_methods.py`）、`../C-F1_before_after/rotation_summary.json`。
""")
    print(open(os.path.join(fo, 'leaderboard.md'), encoding='utf-8').read())


def asset_I1():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei']; plt.rcParams['axes.unicode_minus'] = False
    code = 'C-I1_analysis'; fo = os.path.join(OUT, code); os.makedirs(fo, exist_ok=True); print('C-I1 analysis charts', flush=True)
    hb = MC['hand']['per_distance_rmse_m']; dists = sorted(hb['opt'], key=float)
    show = [('opt', 'CA-DFDS（本作品）', COL['cadfds']), ('gray', '灰階 DFD', COL['gray']), ('stmap', 'STMAP（1994）', COL['stmap']),
            ('unidepth_lin', 'UniDepthV2 線性校正（2025）', COL['unidepth']), ('depthpro_lin', 'Depth Pro 線性校正（2024）', COL['depthpro'])]
    hexc = lambda c: '#%02x%02x%02x' % c

    def style(ax, fig):
        fig.patch.set_facecolor('white'); ax.grid(alpha=0.3); ax.tick_params(labelsize=13)

    data = {'error_vs_distance': {k: {g: hb[k][g] for g in dists} for k, _, _ in show}}
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150); style(ax, fig)
    for k, nm, c in show:
        ax.plot([float(g) for g in dists], [hb[k][g] for g in dists], 'o-', color=hexc(c) if k != 'gray' else '#888', lw=3.5 if k == 'opt' else 2, label=nm)
    ax.set_xlabel('真實距離（m）', fontsize=15); ax.set_ylabel('RMSE（m）', fontsize=15); ax.legend(fontsize=12)
    ax.set_title('各距離 RMSE（CARLA 測試集 480 張，手繪遮罩）', fontsize=17, loc='left'); fig.tight_layout()
    fig.savefig(os.path.join(fo, 'error_vs_distance.png')); plt.close(fig)
    # per colour
    pc = MC['hand']['per_color_rmse_m']; cols = ['red', 'green', 'blue', 'black']
    data['error_by_color'] = {k: pc[k] for k, _, _ in show}
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150); style(ax, fig); bw = 0.16
    for i, (k, nm, c) in enumerate(show):
        ax.bar(np.arange(4) + (i - 2) * bw, [pc[k][c_] for c_ in cols], bw, color=hexc(c) if k != 'gray' else '#999', label=nm)
    ax.set_xticks(range(4)); ax.set_xticklabels(['紅', '綠', '藍', '黑'], fontsize=15); ax.set_ylabel('RMSE（m）', fontsize=15); ax.legend(fontsize=12)
    ax.set_title('各車色 RMSE（手繪遮罩）', fontsize=17, loc='left'); fig.tight_layout(); fig.savefig(os.path.join(fo, 'error_by_color.png')); plt.close(fig)
    # 19 colours
    rr = list(csv.DictReader(open(R('results', '19colors_opt.csv'), encoding='utf-8-sig')))
    pick = lambda s: {r['test'].replace('test_', ''): float(r['rmse_per_image_m']) for r in rr if r['setting'].startswith(s)}
    co, gr = pick('opt('), pick('gray')
    names = sorted(co, key=lambda c: co[c]); data['19_colors'] = {'cadfds_opt': co, 'gray': gr, 'calibration_colors': ['white', 'gray', 'black']}
    fig, ax = plt.subplots(figsize=(14, 6.75), dpi=150); style(ax, fig); x = np.arange(len(names))
    ax.bar(x - 0.2, [co[n] for n in names], 0.4, color=hexc(COL['cadfds']), label='CA-DFDS'); ax.bar(x + 0.2, [gr[n] for n in names], 0.4, color='#999', label='灰階 DFD')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11); ax.set_ylabel('RMSE（m）', fontsize=15); ax.legend(fontsize=13)
    win = sum(co[n] < gr[n] for n in names)
    ax.set_title(f'22 個車色資料夾 RMSE（CA-DFDS 較佳 {win}/{len(names)}；校正只用白／灰／黑）', fontsize=17, loc='left'); fig.tight_layout()
    fig.savefig(os.path.join(fo, 'error_19colors.png')); plt.close(fig)
    # robustness (exposure etc.) from failure_cases/summary.csv if present
    fs = R('results', 'failure_cases', 'summary.csv')
    if os.path.exists(fs):
        data['robustness_failure_cases'] = list(csv.DictReader(open(fs, encoding='utf-8-sig')))
    write_json(os.path.join(fo, 'charts.json'), data)
    readme(fo, 'C-I1 分析圖表', f"""
- `error_vs_distance.png`：各距離 RMSE（5 種方法）。近距離 CA-DFDS／灰階較準，遠距離 UniDepthV2（線性校正）較準。
- `error_by_color.png`：紅／綠／藍／黑各車色 RMSE。
- `error_19colors.png`：22 個車色資料夾（`results/19colors_opt.csv`），CA-DFDS vs 灰階；CA-DFDS 較佳 {win}/{len(names)}。
- `charts.json`：上述三張圖的原始數據＋失敗案例摘要（`results/failure_cases/summary.csv`），給網頁做互動圖（例如 Chart.js）。
- 數字出處：`results/method_comparison.json`、`results/19colors_opt.csv`。白底圖，可直接放簡報。
""")


def asset_M1():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei']; plt.rcParams['axes.unicode_minus'] = False
    code = 'C-M1_timing'; fo = os.path.join(OUT, code); os.makedirs(fo, exist_ok=True); print('C-M1 timing', flush=True)
    tf = json.load(open(R('results', 'timing_full_pipeline.json'), encoding='utf-8'))
    out = {'source': 'results/timing_full_pipeline.json (scripts/m3_timing_full_pipeline.py), RTX 5080 + Intel Core Ultra 7 265K, firetruck_red 30 images',
           'pipelines': {}}
    for grp in ('cpu', 'gpu'):
        for k, v in tf[grp].items():
            if not isinstance(v, dict) or 'total' not in v: continue
            out['pipelines'][k] = dict(total_median_ms=v['total']['median_ms'], fps=v['fps'], ciecam02_median_ms=v['ciecam02']['median_ms'], rmse_m=v.get('rmse_m'))
    out['auto_mask'] = {k: v['median_ms'] for k, v in tf['mask']['auto_mask_yolo_sam'].items() if isinstance(v, dict)}
    write_json(os.path.join(fo, 'timing.json'), out)
    lab = {'cpu_full': 'CPU 全圖', 'cpu_crop': 'CPU 裁切', 'gpu_fp64_full': 'GPU fp64 全圖', 'gpu_fp64_crop': 'GPU fp64 裁切',
           'gpu_fp32_full': 'GPU fp32 全圖', 'gpu_fp32_crop': 'GPU fp32 裁切'}
    ks = [k for k in lab if k in out['pipelines']]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150); fig.patch.set_facecolor('white')
    vals = [out['pipelines'][k]['total_median_ms'] for k in ks]
    b = ax.barh([lab[k] for k in ks], vals, color=['#999' if k.startswith('cpu') else '#f0b428' for k in ks])
    for r_, v_, k in zip(b, vals, ks):
        ax.text(v_ * 1.05, r_.get_y() + r_.get_height() / 2, f"{v_:.0f} ms · {out['pipelines'][k]['fps']} FPS", va='center', fontsize=13)
    ax.set_xscale('log'); ax.set_xlim(50, 9000); ax.set_xlabel('每張處理時間（ms，對數座標；含讀圖，不含取得遮罩）', fontsize=14); ax.invert_yaxis(); ax.tick_params(labelsize=13)
    ax.set_title('CA-DFDS 每張處理時間（RTX 5080 + Core Ultra 7 265K，RMSE 皆相同）', fontsize=16, loc='left'); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fo, 'timing.png')); plt.close(fig)
    am = out['auto_mask']
    readme(fo, 'C-M1 即時性（處理時間）', f"""
- `timing.png`、`timing.json`：CA-DFDS 整條管線每張時間（讀圖 → 遮罩 → CIECAM02 → 拉普拉斯 → 校正）。
  GPU 版 CIECAM02（`cadfds_gpu.py`）與 CPU 結果相同（RMSE 一致），fp32 全圖約 {out['pipelines']['gpu_fp32_full']['total_median_ms']:.0f} ms／張（{out['pipelines']['gpu_fp32_full']['fps']} FPS）。
- 全自動遮罩 YOLO11 → SAM：{', '.join(f'{k} {v} ms' for k, v in am.items())}（中位數）。
- 硬體：RTX 5080 + Intel Core Ultra 7 265K（實驗室主機）。量測腳本 `scripts/m3_timing_full_pipeline.py`（RESULTS 補充 K）。
- 畫面角落顯示 FPS 的錄影：GUI v3 的「計時與硬體」卡即時顯示（見 C-G1 GUI 錄影）。
""")


def asset_K1():
    code = 'C-K1_failure_cases'; fo = os.path.join(OUT, code); os.makedirs(fo, exist_ok=True); print('C-K1 failure cases', flush=True)
    src = R('results', 'failure_cases'); cap = {
        'worst_beyond_7.2m.jpg': '超出校正範圍（7.2 m）：校正曲線只涵蓋 1–7 m，外插誤差變大。',
        'worst_exposure_x0.5.jpg': '曝光 ×0.5（偏暗）：紋理對比下降，σ 變小，距離被高估或低估。',
        'worst_exposure_x0.7.jpg': '曝光 ×0.7：同上，程度較輕。',
        'worst_exposure_x1.4.jpg': '曝光 ×1.4（偏亮）：亮部飽和失去紋理。',
        'worst_exposure_x2.0.jpg': '曝光 ×2.0：大面積過曝，清晰度量測失準。',
        'worst_flipped.jpg': '上下鏡射影像（資料集 reverse）：與旋轉不同，說明見 RESULTS 補充 Q。',
        'worst_suv_runA.jpg': '換車種（SUV）未重新校正：車身紋理不同，消防車的校正曲線不適用。',
        'worst_suv_runB.jpg': '換車種（SUV，第二次渲染）：同上。'}
    items = []
    for f, c in cap.items():
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(fo, f)); items.append(dict(file=f, caption=c))
    ex = R('results', 'gui_new_v3_4_box_bad.png')
    if os.path.exists(ex):
        shutil.copy(ex, os.path.join(fo, 'mask_error_box_iou0.33.png'))
        items.append(dict(file='mask_error_box_iou0.33.png', caption='遮罩錯誤：框沒框好（與手繪遮罩 IoU 0.33），清晰度混入背景，誤差變大。'))
    write_json(os.path.join(fo, 'captions.json'), items)
    readme(fo, 'C-K1 失敗案例', 'Each image has a one-line cause (`captions.json`, for the web gallery).\n\n'
           + '\n'.join(f"- `{i['file']}`：{i['caption']}" for i in items) +
           '\n\n- 原始資料與數值：`results/failure_cases/`（`summary.csv`、`per_image.csv`）、RESULTS 補充 Q。逆光情境尚未量測。')


ASSETS = {'C-A1': asset_A1, 'C-C1': asset_C1, 'C-E1': asset_E1, 'C-F1': asset_F1, 'C-B1': asset_B1, 'C-D1': asset_D1,
          'C-H1': asset_H1, 'C-I1': asset_I1, 'C-M1': asset_M1, 'C-K1': asset_K1}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*', default=None); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for k, fn in ASSETS.items():
        if a.only is None or k in a.only:
            fn()
