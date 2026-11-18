"""Loopback-only research demo and live inference on local CARLA samples."""

import argparse
import csv
import errno
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "web_demo"
RESULTS = ROOT / "results"
DEMO = RESULTS / "demo"
DATA = Path(os.environ.get("CADFDS_DATA", ROOT.parent / "Carla-0.10.0-Win64-Shipping" / "datas"))
DATASET = "Test_dataset_Focus1500cm_f1.4_f22.0"
INFERENCE_LOCK = threading.Lock()
PAPER = ROOT.parent / "paper" / "CA-DFDS_附件三_作品說明書.pdf"
MOUNTS = {"/demo/": DEMO, "/research/": RESULTS / "failure_cases", "/site/": SITE}


def read_json(directory, filename):
    return json.loads((DEMO / directory / filename).read_text(encoding="utf-8-sig"))


def read_csv(directory, filename):
    with (DEMO / directory / filename).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def benchmark_data():
    return {
        "frames": read_csv("C-E1_method_compare", "per_frame.csv"),
        "rotation": read_csv("C-F1_before_after", "rotation_per_image.csv"),
        "rotationSummary": read_json("C-F1_before_after", "rotation_summary.json"),
        "steps": read_json("C-C1_input_output", "steps.json"),
        "leaders": read_json("C-H1_leaderboard", "leaderboard.json"),
        "analysis": read_json("C-I1_analysis", "charts.json"),
        "timing": read_json("C-M1_timing", "timing.json"),
        "failures": read_json("C-K1_failure_cases", "captions.json"),
        "optics": read_csv("C-D1_defocus_theory", "curves.csv"),
        "robustness": read_csv("../failure_cases", "summary.csv"),
        "texture": read_csv("../failure_cases", "texture_by_color.csv"),
        "boxes": json.loads((SITE / "figures" / "vehicle_boxes.json").read_text(encoding="utf-8")),
    }


def infer_sample(frame, gpu=False):
    import numpy as np
    import cadfds_core as core

    color, index, distance = frame["color"], int(frame["T"]), frame["gt_m"]
    source = DATA / f"firetruck_{color}"
    image_path = source / DATASET / f"T{index:02}" / "R" / f"D{distance}_shot01.png"
    mask_path = core.find_mask([str(source / f"{DATASET}_mask")], index)
    if not image_path.is_file() or not mask_path:
        raise FileNotFoundError("CARLA 影像或資料集遮罩不存在，請檢查 CADFDS_DATA。")
    calibration = json.loads((RESULTS / "calibration_methods.json").read_text(encoding="utf-8"))

    covariance_from_image = core.laplacian_cov
    if gpu:
        import torch
        from cadfds_gpu import CIECAM02GPU

        if not torch.cuda.is_available():
            raise RuntimeError("Torch 找不到 CUDA 裝置。")
        covariance_from_image = CIECAM02GPU(dtype=torch.float32).cov_from_bgr
    start = time.perf_counter()
    image = core.imread(image_path)
    if image is None:
        raise ValueError("無法解碼 CARLA 影像。")
    mask = core.load_mask(mask_path, image.shape)
    if mask.sum() < 2:
        raise ValueError("遮罩像素不足。")
    image, mask = core.crop_to_mask(image, mask)
    covariance = covariance_from_image(image, mask)
    sigmas = {key: core.sigma_eq5(covariance, calibration[key]["weights"])
              for key in ("opt", "reopt5", "jonly")}
    sigmas["gray"] = core.gray_sigma(image, mask)
    rows = []
    for key, sigma in sigmas.items():
        model = calibration[key]
        predicted = float(np.polyval(model["coef_cm"], sigma) / 100)
        if not np.isfinite(predicted):
            raise ValueError("測距結果超出可計算範圍。")
        low, high = model["sigma_range"]
        rows.append({"key": key, "sigma": sigma, "predicted": predicted,
                     "error": predicted - float(distance), "extrapolated": not low <= sigma <= high})
    elapsed = time.perf_counter() - start
    return {"gt": float(distance), "rows": rows, "ms": elapsed * 1000, "fps": 1 / elapsed}


class Handler(BaseHTTPRequestHandler):
    def valid_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def json_response(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def do_POST(self):
        if (not self.valid_host() or self.headers.get("Origin") not in
                {None, f"http://{self.headers.get('Host')}"}):
            self.send_error(403)
            return
        match = re.fullmatch(r"/api/infer/(\d{1,3})(?:/(gpu))?", self.path)
        frames = self.server.data["frames"]
        if not match or int(match[1]) >= len(frames):
            self.send_error(404)
            return
        if not INFERENCE_LOCK.acquire(blocking=False):
            self.json_response({"error": "已有測距正在執行。"}, 409)
            return
        try:
            if match[2]:
                torch_python = os.environ.get("CADFDS_TORCH_PY", ROOT.parent / ".venv" / "Scripts" / "python.exe")
                worker = subprocess.run(
                    [str(torch_python), str(Path(__file__).resolve()), "--gpu-frame", match[1]],
                    capture_output=True, encoding="utf-8", errors="replace", timeout=45,
                )
                result = json.loads(worker.stdout) if worker.stdout else {"error": "無法啟動 CUDA 測距程序。"}
                if worker.returncode:
                    raise RuntimeError(result["error"])
            else:
                result = infer_sample(frames[int(match[1])])
        except (OSError, ValueError, ImportError, RuntimeError, subprocess.TimeoutExpired) as error:
            self.json_response({"error": str(error)}, 422)
        else:
            self.json_response(result)
        finally:
            INFERENCE_LOCK.release()

    def do_GET(self):
        if not self.valid_host():
            self.send_error(403)
            return
        path = unquote(urlsplit(self.path).path)
        if path == "/api/data":
            self.json_response(self.server.data)
            return
        if path == "/":
            self.send_file(SITE / "index.html")
            return
        if path == "/paper.pdf" and PAPER.is_file():
            self.send_file(PAPER)
            return
        for prefix, directory in MOUNTS.items():
            if path.startswith(prefix):
                root = directory.resolve()
                file = (root / path[len(prefix):]).resolve()
                if file.is_relative_to(root) and file.is_file():
                    self.send_file(file)
                    return
        self.send_error(404)

    do_HEAD = do_GET

    def send_file(self, file):
        size = file.stat().st_size
        start, stop = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if match and match[1]:
                start = int(match[1])
                stop = min(stop, int(match[2])) if match[2] else stop
            elif match and match[2]:
                start = max(0, size - int(match[2]))
            if not match or not any(match.groups()) or start > stop:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", mimetypes.guess_type(file)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(stop - start + 1))
        self.end_headers()
        if self.command == "HEAD":
            return
        with file.open("rb") as stream:
            stream.seek(start)
            remaining = stop - start + 1
            try:
                while remaining:
                    chunk = stream.read(min(remaining, 128 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--gpu-frame", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu_frame is not None:
        sys.stdout.reconfigure(encoding="utf-8")
        frames = read_csv("C-E1_method_compare", "per_frame.csv")
        try:
            if not 0 <= args.gpu_frame < len(frames):
                raise ValueError("樣本索引超出範圍。")
            print(json.dumps(infer_sample(frames[args.gpu_frame], gpu=True), ensure_ascii=False, allow_nan=False))
            return
        except (OSError, ValueError, ImportError, RuntimeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False))
            raise SystemExit(1) from None
    data = benchmark_data()
    port = 8765 if args.port is None else args.port
    try:
        server = LocalServer(("127.0.0.1", port), Handler)
    except OSError as error:
        if args.port is not None or (error.errno != errno.EADDRINUSE and
                                     getattr(error, "winerror", None) not in (10013, 10048)):
            raise
        server = LocalServer(("127.0.0.1", 0), Handler)
    with server:
        server.data = data
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"CA-DFDS web demo: {url}  (Ctrl+C to stop)", flush=True)
        if not args.no_browser:
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
