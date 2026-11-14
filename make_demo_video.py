# -*- coding: utf-8 -*-
"""決賽 demo 影片：車輛由遠到近，逐張顯示遮罩、真實距離、CA-DFDS 估測距離與誤差。

用法：
  python make_demo_video.py --calibration results/calibration_opt.json \
      --features features/test_red.npz features/test_green.npz features/test_blue.npz \
      --labels 紅色車 綠色車 藍色車 --mask_dir <遮罩資料夾…> --out results/CA-DFDS_demo_testdist.mp4
--shots all：每距離播放全部張數（較長、較像連續影片）；預設每距離 1 張。
需 Windows 字型 msjhbd.ttc，或以 --font 指定（Linux 可用 NotoSansCJK）。
"""
import os, argparse, subprocess, shutil, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import cadfds_core as C

ap = argparse.ArgumentParser()
ap.add_argument('--calibration', required=True); ap.add_argument('--features', nargs='+', required=True)
ap.add_argument('--labels', nargs='+', required=True); ap.add_argument('--mask_dir', nargs='+', required=True)
ap.add_argument('--out', default='results/CA-DFDS_demo.mp4'); ap.add_argument('--shots', default='1')
ap.add_argument('--font', default=r'C:\Windows\Fonts\msjhbd.ttc'); ap.add_argument('--hold', type=float, default=1.1)
a = ap.parse_args()
w, co = C.load_calibration(a.calibration)
F = lambda s: ImageFont.truetype(a.font, s)
fb, fm, fs, fx = F(46), F(32), F(24), F(20)
W, H, FPS = 1920, 1080, 30
tmp = a.out.replace('.mp4', '_raw.mp4'); os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
PAL = [(220, 60, 60), (40, 170, 90), (60, 110, 230), (230, 160, 40), (160, 90, 200)]

def card(lines, sec):
    im = Image.new('RGB', (W, H), (18, 28, 48)); d = ImageDraw.Draw(im); y = 380
    for t, f, c in lines:
        d.text(((W - d.textlength(t, font=f)) / 2, y), t, font=f, fill=c); y += f.size + 26
    fr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    for _ in range(int(FPS * sec)): vw.write(fr)

card([('CA-DFDS 單眼失焦車距估測　系統展示', fb, (255, 255, 255)),
      ('單一相機 · f/1.4 · 對焦 15 m · 單張影像估測距離', fm, (190, 210, 240))], 3)
summary = []
for k, (ff, lab) in enumerate(zip(a.features, a.labels)):
    D = np.load(ff, allow_pickle=True); col = PAL[k % len(PAL)]
    pred = np.polyval(co, C.sigma_eq5(D['cov'], w)) / 100; gt = D['dist_cm'] / 100
    order = np.argsort(-gt, kind='stable'); hist = []; seen = {}
    for i in order:
        if not np.isfinite(pred[i]): continue
        seen[gt[i]] = seen.get(gt[i], 0) + 1
        if a.shots != 'all' and seen[gt[i]] > int(a.shots): continue
        img = C.imread(str(D['path'][i])); mp = C.find_mask(a.mask_dir, int(D['t'][i]))
        m = C.load_mask(mp, img.shape).astype(np.uint8); hist.append((gt[i], pred[i]))
        disp = cv2.resize(img, (1280, 960)); mk = cv2.resize(m, (1280, 960), interpolation=cv2.INTER_NEAREST)
        disp[mk > 0] = (0.75 * disp[mk > 0] + 0.25 * np.array([0, 215, 255])).astype(np.uint8)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(disp, cs, -1, (0, 220, 255), 3)
        cv = Image.new('RGB', (W, H), (18, 28, 48)); cv.paste(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)), (40, 80))
        d = ImageDraw.Draw(cv); x0 = 1360
        d.text((40, 18), f'CA-DFDS 系統展示 · {lab}', font=fm, fill=(255, 255, 255))
        for j, (cap, val, c) in enumerate([('真實距離', f'{gt[i]:.2f} m', (255, 255, 255)), ('CA-DFDS 估測距離', f'{pred[i]:.2f} m', col),
                                           ('誤差', f'{pred[i] - gt[i]:+.2f} m', (255, 210, 90))]):
            d.text((x0, 90 + j * 120), cap, font=fs, fill=(170, 185, 210)); d.text((x0, 125 + j * 120), val, font=fb, fill=c)
        px, py, pw, ph = x0, 520, 500, 440; d.rectangle([px, py, px + pw, py + ph], outline=(90, 110, 140), width=2)
        sx = lambda v: px + 30 + (v - 0.5) / 7.0 * (pw - 50); sy = lambda v: py + ph - 30 - (v - 0.5) / 7.0 * (ph - 50)
        d.line([sx(.5), sy(.5), sx(7.5), sy(7.5)], fill=(120, 130, 150), width=2)
        for g, p_ in hist: d.ellipse([sx(g) - 7, sy(p_) - 7, sx(g) + 7, sy(p_) + 7], fill=col)
        d.text((px + 8, py + 6), '估測 vs 真實（m）', font=fx, fill=(200, 210, 230))
        fr = cv2.cvtColor(np.array(cv), cv2.COLOR_RGB2BGR)
        for _ in range(int(FPS * (a.hold if a.shots != 'all' else 0.35))): vw.write(fr)
    g = np.array([h[0] for h in hist]); p_ = np.array([h[1] for h in hist])
    summary.append((lab, float(np.sqrt(np.mean((p_ - g) ** 2))), col, len(hist)))
card([('展示結果（單張影像估測）', fb, (255, 255, 255))] + [(f'{l}：RMSE {r:.2f} m（{n} 張）', fm, c) for l, r, c, n in summary], 4)
vw.release()
ff = shutil.which('ffmpeg')
if ff:
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', a.out], check=True)
    os.remove(tmp)
else:
    os.replace(tmp, a.out)
print('saved', a.out, summary)
