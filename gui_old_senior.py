# -*- coding: utf-8 -*-
"""【學長舊版】CA-DFDS 車距估測 GUI（學長方法：論文權重、單一方法，tkinter，無需網路）。

此為 repo 最初的 gui_demo.py（commit 782066e），2026-11-16 改名保留作為對照，程式邏輯未改，
只改了視窗標題，並讓校正檔路徑以本檔所在資料夾為基準。
新版（6 種方法比較、自動遮罩、影片）請用 gui_new_v3.py。


操作：
  1. 「開啟影像」：選 f/1.4、對焦 15 m 拍攝之影像（CARLA 或實拍）。
  2. 「載入遮罩」：選對應之 T<n>.npy／.png；或直接在影像上按住左鍵塗抹車輛範圍（右鍵清除）。
  3. 「估測距離」：顯示清晰度 σ 與估測距離；若檔名含 D<距離>_，同時顯示真實距離與誤差。
校正曲線：預設讀取 results/calibration_opt.json；無此檔時使用論文之曲線
  d = −6.562σ² + 128.83σ + 68.39（cm），權重 (0.8136, 0.1782, 0.0081)。
執行：python gui_old_senior.py [--calibration results/calibration_opt.json]（或點兩下 run_gui_old_senior.bat）
"""
import os, re, sys, argparse, threading, numpy as np, cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cadfds_core as C

ap = argparse.ArgumentParser(); ap.add_argument('--calibration', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'calibration_opt.json')); a = ap.parse_args()
if os.path.exists(a.calibration):
    W_, CO = C.load_calibration(a.calibration); SRC = a.calibration
else:
    W_, CO = np.array(C.W_OPT), np.array([-6.562, 128.83, 68.39]); SRC = '論文校正曲線'
VW, VH = 960, 720

class App:
    def __init__(s, root):
        s.root = root; root.title('CA-DFDS 學長舊版 GUI（論文權重・單一方法）'); root.configure(bg='#1a2438')
        bar = tk.Frame(root, bg='#1a2438'); bar.pack(fill='x', padx=10, pady=8)
        for t, cmd in [('開啟影像', s.open_img), ('載入遮罩', s.open_mask), ('清除遮罩', s.clear), ('估測距離', s.run)]:
            tk.Button(bar, text=t, command=cmd, font=('Microsoft JhengHei', 13), width=10).pack(side='left', padx=4)
        s.brush = tk.IntVar(value=25); tk.Label(bar, text='筆刷', fg='white', bg='#1a2438').pack(side='left', padx=(16, 2))
        tk.Scale(bar, from_=5, to=80, orient='horizontal', variable=s.brush, length=120).pack(side='left')
        body = tk.Frame(root, bg='#1a2438'); body.pack(padx=10, pady=4)
        s.cv = tk.Canvas(body, width=VW, height=VH, bg='black', highlightthickness=0); s.cv.pack(side='left')
        s.info = tk.Label(body, text=f'校正：{SRC}\n請開啟影像', font=('Microsoft JhengHei', 18), fg='white', bg='#1a2438', justify='left', width=22, anchor='nw')
        s.info.pack(side='left', fill='y', padx=16)
        s.cv.bind('<B1-Motion>', s.paint); s.cv.bind('<Button-1>', s.paint); s.cv.bind('<Button-3>', lambda e: s.clear())
        s.img = None; s.mask = None; s.path = None

    def open_img(s):
        p = filedialog.askopenfilename(filetypes=[('Image', '*.png *.jpg *.jpeg *.bmp')])
        if not p: return
        s.img = C.imread(p); s.path = p; s.mask = np.zeros(s.img.shape[:2], np.uint8)
        s.sc = min(VW / s.img.shape[1], VH / s.img.shape[0]); s.show(); s.info.config(text=f'校正：{SRC}\n{os.path.basename(p)}\n請載入或塗抹遮罩')

    def open_mask(s):
        if s.img is None: return
        p = filedialog.askopenfilename(filetypes=[('Mask', '*.npy *.png')])
        if p: s.mask = C.load_mask(p, s.img.shape).astype(np.uint8); s.show()

    def clear(s):
        if s.img is not None: s.mask[:] = 0; s.show()

    def paint(s, e):
        if s.img is None: return
        x, y = int(e.x / s.sc), int(e.y / s.sc); cv2.circle(s.mask, (x, y), int(s.brush.get() / s.sc), 1, -1); s.show()

    def show(s):
        d = s.img.copy(); m = s.mask > 0
        d[m] = (0.7 * d[m] + 0.3 * np.array([0, 215, 255])).astype(np.uint8)
        d = cv2.resize(d, (int(d.shape[1] * s.sc), int(d.shape[0] * s.sc)))
        s.tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))); s.cv.delete('all'); s.cv.create_image(0, 0, anchor='nw', image=s.tk)

    def run(s):
        if s.img is None or s.mask is None or s.mask.sum() < 100:
            messagebox.showinfo('CA-DFDS', '請先開啟影像並標示車輛遮罩'); return
        s.info.config(text='計算中…')
        def job():
            import time; t0 = time.perf_counter()
            sig = C.sharpness(s.img, s.mask > 0, W_); dist = np.polyval(CO, sig) / 100; ms = 1000 * (time.perf_counter() - t0)
            txt = f'清晰度 σ = {sig:.2f}\n\n估測距離\n{dist:.2f} m\n\n計算時間 {ms:.0f} ms'
            m = re.search(r'D(\d+(?:\.\d+)?)_', os.path.basename(s.path))
            if m: g = float(m.group(1)); txt += f'\n\n真實距離 {g:.2f} m\n誤差 {dist - g:+.2f} m'
            s.root.after(0, lambda: s.info.config(text=txt))
        threading.Thread(target=job, daemon=True).start()

root = tk.Tk(); App(root); root.mainloop()
