### Hand-drawn masks

| Method | red | green | blue | black | Overall RMSE | MAE | Mean bias | Time per image |
|---|---|---|---|---|---|---|---|---|
| CA-DFDS (paper weights) | 0.276 | 0.196 | 0.348 | 0.468 | **0.337** | 0.270 | +0.056 | 2264 ms |
| CA-DFDS re-optimized eq. 5 | 0.243 | 0.184 | 0.304 | 0.456 | **0.314** | 0.246 | +0.044 | 2264 ms |
| CA-DFDS J only | 0.227 | 0.191 | 0.307 | 0.458 | **0.313** | 0.246 | +0.043 | 2264 ms |
| Grayscale DFD | 0.216 | 0.174 | 0.393 | 0.541 | **0.362** | 0.263 | +0.102 | 83 ms |
| UniDepth V2 (raw) | 1.713 | 2.096 | 1.705 | 1.974 | **1.879** | 1.832 | +1.832 | 27 ms |
| UniDepth V2 (linear calib.) | 0.303 | 0.392 | 0.317 | 0.405 | **0.357** | 0.288 | -0.120 | 27 ms |
| ML-Depth-Pro (raw) | 0.603 | 0.924 | 1.068 | 0.821 | **0.871** | 0.678 | -0.380 | 260 ms |
| ML-Depth-Pro (linear calib.) | 0.648 | 0.827 | 0.978 | 0.439 | **0.751** | 0.476 | +0.118 | 260 ms |

### Fully automatic masks (YOLO11→SAM)

| Method | red | green | blue | black | Overall RMSE | MAE | Mean bias | Time per image |
|---|---|---|---|---|---|---|---|---|
| CA-DFDS (paper weights) | 0.307 | 0.228 | 0.308 | 0.378 | **0.310** | 0.257 | -0.025 | 2264 ms |
| CA-DFDS re-optimized eq. 5 | 0.278 | 0.231 | 0.263 | 0.368 | **0.289** | 0.238 | -0.042 | 2264 ms |
| CA-DFDS J only | 0.263 | 0.241 | 0.269 | 0.370 | **0.290** | 0.240 | -0.044 | 2264 ms |
| Grayscale DFD | 0.251 | 0.199 | 0.335 | 0.457 | **0.325** | 0.252 | +0.018 | 83 ms |
| UniDepth V2 (raw) | 1.716 | 2.098 | 1.708 | 1.980 | **1.883** | 1.835 | +1.835 | 27 ms |
| UniDepth V2 (linear calib.) | 0.301 | 0.391 | 0.315 | 0.402 | **0.355** | 0.286 | -0.117 | 27 ms |
| ML-Depth-Pro (raw) | 0.602 | 0.923 | 1.068 | 0.820 | **0.870** | 0.677 | -0.379 | 260 ms |
| ML-Depth-Pro (linear calib.) | 0.649 | 0.827 | 0.978 | 0.438 | **0.750** | 0.476 | +0.119 | 260 ms |
