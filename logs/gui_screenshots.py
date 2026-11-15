# -*- coding: utf-8 -*-
"""任務 G 輔助：實際開啟 gui_demo.py 的視窗，依序載入測試影像與遮罩、按「估測距離」，並擷取視窗畫面。
與手動操作相同的流程（開啟影像 → 載入遮罩 → 估測距離），只是由程式代替滑鼠點選檔案。
用法：python logs/gui_screenshots.py <輸出資料夾> <影像1> <遮罩1> [<影像2> <遮罩2> ...]
"""
import os, sys, runpy, time
import tkinter as tk
from tkinter import filedialog
from PIL import ImageGrab

out_dir, pairs = sys.argv[1], sys.argv[2:]
jobs = [(pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2)]
os.makedirs(out_dir, exist_ok=True)
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(here); sys.path.insert(0, here)

queue = []  # 下一次檔案對話框要回傳的路徑
filedialog.askopenfilename = lambda **k: queue.pop(0)

apps = {}
real_mainloop = tk.Tk.mainloop


def driver(root):
    app = apps['app']
    def step(k):
        if k >= len(jobs):
            root.destroy(); return
        img, mask = jobs[k]
        queue[:] = [img, mask]
        app.open_img(); app.open_mask(); app.run()
        def wait(t=0):
            txt = app.info.cget('text')
            if '估測距離' in txt or t > 600:
                root.update(); time.sleep(0.3)
                x, y = root.winfo_rootx(), root.winfo_rooty()
                w, h = root.winfo_width(), root.winfo_height()
                p = os.path.join(out_dir, f'gui_{k + 1}.png')
                ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(p)
                print('saved', p, '|', txt.replace('\n', ' '))
                root.after(200, lambda: step(k + 1))
            else:
                root.after(100, lambda: wait(t + 1))
        root.after(500, wait)
    root.after(800, lambda: step(0))


def patched_mainloop(self, n=0):
    import gc
    apps['app'] = next(o for o in gc.get_objects() if type(o).__name__ == 'App' and hasattr(o, 'info'))
    self.attributes('-topmost', True); self.lift(); self.focus_force()
    driver(self)
    real_mainloop(self, n)


tk.Tk.mainloop = patched_mainloop
sys.argv = ['gui_demo.py', '--calibration', 'results/calibration_opt.json']
runpy.run_path(os.path.join(here, 'gui_demo.py'), run_name='__main__')
