# -*- coding: utf-8 -*-
"""GPU (PyTorch) version of the CA-DFDS sharpness pipeline (M3 / issue #1).

Re-implements cadfds_core.laplacian_cov in torch so it runs on CUDA:
  3x3 Gaussian (cv2, on the uint8 image, identical to the CPU path) -> sRGB EOTF -> XYZ -> CIECAM02 forward
  (colour-science 0.4.6 XYZ_to_CIECAM02 formulas, incl. its signed power `spow` and zero-safe division)
  -> J, a_c = C cos h, b_c = C sin h -> 3x3 Laplacian (cv2 ksize=3 kernel, BORDER_REFLECT_101) -> 3x3 covariance in mask.
Viewing conditions as in the paper: D65 white, L_A = 2000/pi*0.2, Y_b = 20, Average surround.
Only needs torch + numpy + cv2 (runs in the ../.venv torch environment, which has no colour-science).
Accuracy vs the CPU features is checked by scripts/m3_timing_full_pipeline.py --validate.
"""
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F

M_SRGB = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]   # colour RGB_COLOURSPACES['sRGB']
XYZ_W = [95.04559270516715, 100.0, 108.90577507598785]                                     # D65 (0.3127, 0.3290) * 100
CAT02 = [[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.003, 0.0136, 0.9834]]
HPE = [[0.38971, 0.68898, -0.07868], [-0.22981, 1.18340, 0.04641], [0.0, 0.0, 1.0]]
L_A = 2000 / math.pi * 0.2
Y_B = 20.0
F_SUR, C_SUR, NC_SUR = 1.0, 0.69, 1.0            # Average surround


def _spow(x, p):
    return torch.sign(x) * torch.abs(x) ** p


def _sdiv(a, b):
    return torch.nan_to_num(a / b, nan=0.0, posinf=0.0, neginf=0.0)


class CIECAM02GPU:
    def __init__(self, device='cuda', dtype=torch.float64):
        self.dev, self.dt = torch.device(device), dtype
        t = lambda m: torch.tensor(m, dtype=torch.float64)
        cat, hpe = t(CAT02), t(HPE)
        Yw = XYZ_W[1]
        n = Y_B / Yw
        k = 1 / (5 * L_A + 1); k4 = k ** 4
        FL = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * (5 * L_A) ** (1 / 3)
        Nbb = 0.725 * (1 / n) ** 0.2
        z = 1.48 + math.sqrt(n)
        D = F_SUR * (1 - (1 / 3.6) * math.exp((-L_A - 42) / 92))
        RGBw = cat @ t(XYZ_W)
        DRGB = Yw * D / RGBw + 1 - D
        # one 3x3 matrix: XYZ(0..100) -> HPE rgb_p, folding sRGB->XYZ and the von Kries step
        M = hpe @ torch.linalg.inv(cat) @ torch.diag(DRGB) @ cat @ t(M_SRGB) * 100.0
        rgb_pw = hpe @ torch.linalg.inv(cat) @ (DRGB * RGBw)
        rgb_aw = self._compress(rgb_pw, FL)
        Aw = float((2 * rgb_aw[0] + rgb_aw[1] + rgb_aw[2] / 20 - 0.305) * Nbb)
        self.M = M.to(self.dev, self.dt)
        self.FL, self.Nbb, self.Ncb, self.cz, self.Aw, self.n = FL, Nbb, Nbb, C_SUR * z, Aw, n
        self.Cfac = (1.64 - 0.29 ** n) ** 0.73
        lap = torch.tensor([[2., 0., 2.], [0., -8., 0.], [2., 0., 2.]], dtype=self.dt)
        self.lap = lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(self.dev)

    @staticmethod
    def _compress(rgb, FL):
        f = _spow(FL * torch.abs(rgb) / 100, 0.42)
        return 400 * torch.sign(rgb) * f / (27.13 + f) + 0.1

    def channels(self, blur_rgb_u8):
        """blur_rgb_u8: torch uint8 (H, W, 3) RGB already Gaussian-blurred, on device. Returns (3, H, W) J, a_c, b_c."""
        v = blur_rgb_u8.to(self.dt) / 255.0
        lin = torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
        rgb_p = lin @ self.M.T                                          # (H, W, 3)
        R, G, B = self._compress(rgb_p, self.FL).unbind(-1)
        a = R - 12 * G / 11 + B / 11
        b = (R + G - 2 * B) / 9
        hr = torch.atan2(b, a)                                         # radians; colour's (deg % 360) gives same cos/sin
        h_deg = torch.rad2deg(hr) % 360
        e_t = 0.25 * (torch.cos(2 + h_deg * math.pi / 180) + 3.8)
        A = (2 * R + G + B / 20 - 0.305) * self.Nbb
        J = 100 * _spow(_sdiv(A, torch.full_like(A, self.Aw)), self.cz)
        t = ((50000 / 13) * NC_SUR * self.Ncb) * _sdiv(e_t * torch.sqrt(a * a + b * b), R + G + 21 * B / 20)
        C = _spow(t, 0.9) * _spow(J / 100, 0.5) * self.Cfac
        J, C, hr = (torch.nan_to_num(x) for x in (J, C, hr))
        return torch.stack([J, C * torch.cos(hr), C * torch.sin(hr)])

    def laplacian_cov(self, channels, mask):
        x = F.pad(channels.unsqueeze(0), (1, 1, 1, 1), mode='reflect')     # torch 'reflect' == cv2 BORDER_REFLECT_101
        L = F.conv2d(x, self.lap, groups=3)[0]                           # (3, H, W)
        V = L[:, mask]                                                   # (3, N)
        V = V - V.mean(dim=1, keepdim=True)
        return (V @ V.T) / V.shape[1]

    def cov_from_bgr(self, img_bgr, mask_np):
        """Same input/output as cadfds_core.laplacian_cov (numpy BGR uint8 + bool mask -> numpy 3x3)."""
        blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)
        rgb = torch.from_numpy(np.ascontiguousarray(blur[..., ::-1])).to(self.dev)
        m = torch.from_numpy(mask_np).to(self.dev)
        return self.laplacian_cov(self.channels(rgb), m).double().cpu().numpy()
