# -*- coding: utf-8 -*-
"""Export web-ready copies of results/demo/ for the project page repo (DreamerForJay/CA-DFDS, GitHub Pages /docs).

→ <web_repo>/docs/assets/demo/<code>/ : video.mp4 (H.264, already ≤ 25 MB), poster.jpg, data files (csv/json),
  PNG figures re-encoded as JPG (max width 1600, quality 85), and docs/assets/demo/manifest.json describing every asset
  (title, caption, files, headline numbers) so the page can be built from data.
Slide versions (video_short.mp4, keyframes/ PNG) stay in CA-DFDS-finals/results/demo and on Google Drive.
Usage: python scripts/export_web_assets.py [--web ../CA-DFDS]
"""
import os, sys, json, glob, shutil, argparse
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'results', 'demo')

ap = argparse.ArgumentParser(); ap.add_argument('--web', default=os.path.join(os.path.dirname(ROOT), 'CA-DFDS')); a = ap.parse_args()
DST = os.path.join(a.web, 'docs', 'assets', 'demo')
if not os.path.isdir(os.path.join(a.web, 'docs')):
    sys.exit(f'web repo not found: {a.web}')
J = lambda *p: json.load(open(os.path.join(SRC, *p), encoding='utf-8'))
mc = json.load(open(os.path.join(ROOT, 'results', 'method_comparison.json'), encoding='utf-8'))['hand']['overall_rmse_m']
rot = J('C-F1_before_after', 'rotation_summary.json')

# (code, title, caption, files to copy as-is, image globs to convert to JPG, headline numbers)
ASSETS = [
    ('C-A1_teaser', 'Teaser：單張失焦影像 → 車距',
     '紅色消防車由 6.7 m 接近到 1.2 m（資料集 12 個距離，非連續拍攝）。左：輸入＋遮罩；中：CIECAM02 清晰度熱圖；右：估測 vs 真實距離。',
     ['video.mp4', 'poster.jpg', 'per_frame.csv', 'summary.json'], [], {'sequence_rmse_m': J('C-A1_teaser', 'summary.json')['seq_rmse_m'], 'test480_rmse_m': mc['opt']}),
    ('C-B1_camera_setup', '拍攝配置', '40 mm、f/1.4＋f/22、對焦 15 m、相機高 0.8 m；距離 = 相機到前車車尾（CARLA get_dataset2.py 參數）。',
     ['poster.jpg', 'summary.json'], ['camera_setup.png', 'camera_views.png'], {}),
    ('C-C1_input_output', '輸入 → 輸出（逐步）',
     '同一張影像：輸入 → 遮罩 → J／a_c／b_c → 加權拉普拉斯清晰度 → σ → 校正曲線 → 距離。近／中／遠三個距離，每步一張圖，可做步驟滑桿。',
     ['video.mp4', 'poster.jpg', 'steps.json', 'per_frame.csv', 'summary.json'], ['steps/*/*.png'], {}),
    ('C-D1_defocus_theory', '為什麼對焦在 15 m', '薄透鏡模糊圈 vs 距離：後對焦（15 m）在 1–7 m 單調、無歧義；f/22 幾乎不模糊。',
     ['video.mp4', 'poster.jpg', 'curves.csv', 'per_frame.csv', 'summary.json'], ['defocus_curves.png', 'aperture_pairs/*.jpg'], {}),
    ('C-E1_method_compare', '四方法同步比較',
     '同一張影像、同一幀：Depth Pro（2024）、UniDepthV2（2025）、STMAP（1994）、CA-DFDS。深度模型已用同一校正集線性校正；STMAP 需兩張不同光圈影像。',
     ['video.mp4', 'poster.jpg', 'per_frame.csv', 'summary.json'], ['keyframes/end_card.png', 'panels/*.jpg'],
     {'test480_rmse_m': {k: mc[k] for k in ('opt', 'unidepth_lin', 'unidepth', 'stmap', 'depthpro_lin', 'depthpro')}}),
    ('C-F1_before_after', '前後對照：旋轉 180°、灰階 vs CIECAM02',
     '旋轉 180° 後 CA-DFDS 結果不變；深度模型與 STMAP 變差。另附灰階與 CIECAM02 清晰度熱圖成對圖（可做拖曳對照）。',
     ['video.mp4', 'poster.jpg', 'rotation_summary.json', 'per_frame.csv', 'summary.json'],
     ['rotation_pairs/*.jpg', 'gray_vs_ciecam/*.png', 'keyframes/rotation_rmse.png'],
     {'rotation_rmse_m': {k: [rot[k]['rmse_orig_m'], rot[k]['rmse_rot_m']] for k in ('cadfds', 'stmap', 'unidepth', 'depthpro')}}),
    ('C-H1_leaderboard', '排行榜（CARLA 480 張）', '各方法 RMSE、近／遠距離、自動遮罩、旋轉 180°、時間；方法附年份與出處。',
     ['leaderboard.json', 'leaderboard.csv', 'leaderboard.md'], [], {}),
    ('C-I1_analysis', '分析圖表', '誤差 vs 距離、各車色、22 個車色資料夾；charts.json 為原始數據，可做互動圖。',
     ['charts.json'], ['*.png'], {}),
    ('C-K1_failure_cases', '失敗案例', '每張圖一句原因（captions.json）。', ['captions.json'], ['*.jpg', '*.png'], {}),
    ('C-M1_timing', '即時性', 'CA-DFDS 每張處理時間：CPU 與 GPU 版 CIECAM02（RMSE 相同）。', ['timing.json'], ['timing.png'], {}),
]

if os.path.isdir(DST):
    shutil.rmtree(DST)
manifest = {'generated_by': 'CA-DFDS-finals/scripts/export_web_assets.py', 'source': 'CA-DFDS-finals/results/demo (see each README.md there)',
            'note': 'All numbers come from CA-DFDS-finals results/ (CARLA simulation, 1.2–6.7 m test set). Do not edit by hand; re-export.',
            'assets': []}
total = 0
for code, title, cap, files, globs, nums in ASSETS:
    s, d = os.path.join(SRC, code), os.path.join(DST, code); os.makedirs(d, exist_ok=True); out = []
    for f in files:
        if os.path.exists(os.path.join(s, f)):
            shutil.copy(os.path.join(s, f), os.path.join(d, f)); out.append(f)
    for g in globs:
        for p in sorted(glob.glob(os.path.join(s, g))):
            rel = os.path.relpath(p, s).replace('\\', '/'); base = os.path.splitext(rel)[0] + '.jpg'
            os.makedirs(os.path.dirname(os.path.join(d, base)), exist_ok=True)
            im = Image.open(p).convert('RGB'); im.thumbnail((1600, 1600))
            im.save(os.path.join(d, base), quality=85, optimize=True, progressive=True); out.append(base)
    sz = sum(os.path.getsize(os.path.join(d, f)) for f in out); total += sz
    manifest['assets'].append(dict(code=code.split('_')[0], folder=f'assets/demo/{code}', title=title, caption=cap, files=out, numbers=nums))
    print(f'{code:24s} {len(out):3d} files {sz / 2 ** 20:6.1f} MB')
json.dump(manifest, open(os.path.join(DST, 'manifest.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(f'total {total / 2 ** 20:.1f} MB → {DST}')
