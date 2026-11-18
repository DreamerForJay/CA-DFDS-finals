"use strict";
const get = id => document.getElementById(id);
const root = "/demo/";
const methods = [
  ["cadfds", "CA-DFDS", "#245a78", "2026"], ["unidepth", "UniDepth V2", "#246951", "2025"],
  ["depthpro", "Depth Pro", "#76538f", "2024"], ["stmap", "STMAP", "#a3592b", "1994"]
];
const labels = {opt:"CA-DFDS", reopt5:"CA-DFDS 重新最佳化", jonly:"CA-DFDS 只用 J", gray:"灰階 DFD",
  unidepth_lin:"UniDepth V2 · 線性校正", depthpro_lin:"Depth Pro · 線性校正", stmap:"STMAP",
  unidepth_raw:"UniDepth V2 · 原始", depthpro_raw:"Depth Pro · 原始"};
const format = (value, digits=3) => value === "" || value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
const signed = value => (value >= 0 ? "+" : "") + format(value);
const stamp = (tag, value) => Object.assign(document.createElement(tag), {textContent:value});
let data;
let playback;
let frameVisible = false;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
let autoFrames = !reducedMotion.matches;
let plots = new Map();

function table(el, headings, rows, highlight = () => false) {
  el.replaceChildren();
  const heading = el.createTHead().insertRow();
  headings.forEach(value => {
    const th = stamp("th", value);
    th.scope = "col";
    heading.append(th);
  });
  const body = el.createTBody();
  rows.forEach(row => {
    const tr = body.insertRow();
    if (highlight(row)) tr.className = "ours";
    row.forEach(value => tr.insertCell().textContent = value);
  });
}

function plot(canvas, series, selected, unit, diagonal = false, xLabel = "GT (m)", scatter = false) {
  plots.set(canvas, [series, selected, unit, diagonal, xLabel, scatter]);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const points = series.flatMap(item => item.points);
  const minX = Math.min(...points.map(point => point[0]));
  const maxX = Math.max(...points.map(point => point[0]));
  const minY = Math.min(0, Math.floor(Math.min(...points.map(point => point[1]))));
  const maxY = Math.max(0.1, Math.ceil(Math.max(...points.map(point => point[1])) * 10) / 10);
  const bounds = {left:51, right:width-15, top:27, bottom:height-32};
  const x = value => bounds.left + (value - minX) / (maxX - minX) * (bounds.right - bounds.left);
  const y = value => bounds.bottom - (value - minY) / (maxY - minY) * (bounds.bottom - bounds.top);
  function line(x1,y1,x2,y2,color) {
    ctx.strokeStyle=color; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  }
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillStyle = "#546477";
  ctx.fillText(unit, 5, 14);
  for (let tick=0;tick<=4;tick++) {
    const value = minY + (maxY-minY)*tick/4;
    line(bounds.left,y(value),bounds.right,y(value),"#dae1e8");
    ctx.fillText(format(value, maxY <= 1 ? 2 : 1), 5, y(value)+4);
  }
  for (let tick=0;tick<=4;tick++) {
    const value = minX + (maxX-minX)*tick/4;
    ctx.fillText(format(value,1), x(value)-10, bounds.bottom+19);
  }
  ctx.fillText(xLabel, bounds.right-ctx.measureText(xLabel).width, height-2);
  if (diagonal) {
    ctx.setLineDash([4,4]);
    line(x(minX),y(minX),x(maxX),y(maxX),"#7b8693");
    ctx.setLineDash([]);
  }
  if (selected !== null) line(x(selected), bounds.top, x(selected), bounds.bottom, "#6b7784");
  for (const item of series) {
    ctx.strokeStyle = item.color;
    ctx.fillStyle = item.color;
    ctx.lineWidth = 2;
    if (!scatter) {
      ctx.beginPath();
      item.points.forEach(([px,py], index) => index ? ctx.lineTo(x(px),y(py)) : ctx.moveTo(x(px),y(py)));
      ctx.stroke();
    }
    if (scatter || selected === null) {
      item.points.forEach(([px,py]) => {
        ctx.beginPath(); ctx.arc(x(px),y(py),3,0,Math.PI*2); ctx.fill();
      });
    }
    const point = item.points.find(([px]) => Math.abs(px - selected) < 1e-5);
    if (point) {
      ctx.beginPath(); ctx.arc(x(point[0]), y(point[1]), 4, 0, Math.PI*2); ctx.fill();
    }
  }
}

function frames() {
  return data.frames.filter(row => row.color === get("color").value).sort((a,b) => Number(a.gt_m)-Number(b.gt_m));
}
const panels = methods.map(([key, name, tint, year]) => {
  const panel = stamp("article", "");
  panel.className = "method-panel";
  panel.style.setProperty("--method", tint);
  const value = stamp("strong", "");
  const error = stamp("span", "");
  const detail = stamp("small", "");
  const link = stamp("a", "");
  const image = document.createElement("img");
  image.alt = name + " 輸出圖";
  link.target = "_blank";
  link.rel = "noopener";
  link.append(image);
  const figure = document.createElement("figure");
  figure.append(link,stamp("figcaption",name));
  get("comparison-outputs").append(figure);
  panel.append(stamp("h3",name), stamp("small",year), value, error, detail);
  get("method-cards").append(panel);
  return {key, value, error, detail, link, image};
});

function updateComparison() {
  const rows = frames();
  const row = rows[Number(get("frame").value)];
  const {color, gt_m: distance} = row;
  get("frame-value").value = distance + " m";
  const vehicle = get("vehicle-input");
  vehicle.src = root + "scenario_replays/" + color + "_" + distance + "m.jpg";
  const tint = color === "red" ? "#ff5c54" : "#5ef287";
  const [left,top,right,bottom] = data.boxes[color + "_" + distance];
  Object.assign(get("vehicle-box").style,{
    left:left*100+"%", top:top*100+"%", width:(right-left)*100+"%",
    height:(bottom-top)*100+"%", borderColor:tint
  });
  const colors = {red:"紅",green:"綠",blue:"藍",black:"黑"};
  get("vehicle-caption").textContent = colors[color] + "色消防車 · GT " + distance + " m";
  get("comparison-outputs").hidden = color !== "red";
  for (const {key, value, error, detail, link, image} of panels) {
    const raw = row[key + "_raw_m"];
    if (color === "red") {
      link.href = root + "C-E1_method_compare/panels/" + distance + "m_" + key + ".jpg";
      image.src = link.href;
    }
    value.textContent = format(row[key + "_m"]) + " m";
    error.textContent = "誤差 " + signed(Number(row[key + "_err_m"])) + " m";
    detail.hidden = !raw;
    detail.textContent = raw ? "原始 " + format(raw) + " m" : "";
  }
  plot(get("compare-chart"), methods.map(([key,,color]) => ({
    color,points:rows.map(item => [Number(item.gt_m),Number(item[key+"_m"])])
  })), Number(distance), "估測 (m)", true);
}

function updatePlayback() {
  const playing = autoFrames && frameVisible && !document.hidden;
  get("play").textContent = autoFrames ? "暫停" : "播放";
  if (playing && !playback) {
    playback = setInterval(() => {
      const slider = get("frame");
      slider.value = Number(slider.value) === 0 ? slider.max : Number(slider.value)-1;
      updateComparison();
    }, 1100);
  } else if (!playing && playback) {
    clearInterval(playback);
    playback = null;
  }
}

function stopPlayback() {
  autoFrames = false;
  updatePlayback();
}

function configureMotion() {
  if (!("IntersectionObserver" in window)) return;
  const seenVideos = new Set();
  const startedVideos = new WeakSet();
  function updateVideos() {
    for (const video of document.querySelectorAll("video[data-autoplay]")) {
      if (!seenVideos.has(video) || document.hidden || reducedMotion.matches) {
        if (startedVideos.has(video)) video.pause();
      } else if (!startedVideos.has(video)) {
        startedVideos.add(video);
        video.play().catch(() => {});
      }
    }
  }
  const videos = new IntersectionObserver(entries => {
    for (const {target,isIntersecting} of entries) {
      if (isIntersecting) seenVideos.add(target);
      else {
        seenVideos.delete(target);
        if (startedVideos.has(target)) target.pause();
        startedVideos.delete(target);
      }
    }
    updateVideos();
  }, {threshold:.25});
  document.querySelectorAll("video[data-autoplay]").forEach(video => videos.observe(video));
  const comparison = new IntersectionObserver(([entry]) => {
    frameVisible = entry.isIntersecting;
    updatePlayback();
  }, {threshold:.35});
  comparison.observe(get("method-cards"));
  document.addEventListener("visibilitychange",() => {
    if (!document.hidden) for (const video of seenVideos) startedVideos.delete(video);
    updatePlayback();
    updateVideos();
  });
  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) stopPlayback();
    else for (const video of seenVideos) startedVideos.delete(video);
    updateVideos();
  });
}
async function runInference() {
  stopPlayback();
  const frame = frames()[get("frame").value];
  const index = data.frames.indexOf(frame);
  const gpu = get("engine").value === "gpu";
  const target = get("live-result");
  get("live").disabled = true;
  target.textContent = "測距中…";
  try {
    const response = await fetch("/api/infer/" + index + (gpu ? "/gpu" : ""), {method:"POST"});
    const result = await response.json();
    if (!response.ok) throw Error(result.error);
    const grid = document.createElement("table");
    table(grid, ["方法","σ","估測 (m)","誤差 (m)","校正範圍"], result.rows.map(row =>
      [labels[row.key],format(row.sigma,4),format(row.predicted),signed(row.error),row.extrapolated ? "外插" : "範圍內"]));
    const engine = gpu ? "CUDA FP32 色彩轉換＋CPU 灰階" : "CPU";
    const summary = stamp("p", engine + " · " + frame.color + "／GT " + result.gt + " m · " +
      format(result.ms,1) + " ms／張 · " + format(result.fps,2) + " FPS（單次延遲換算，含讀圖與既有遮罩）");
    const scroll = stamp("div","");
    scroll.className = "table-scroll";
    scroll.append(grid);
    target.replaceChildren(summary,scroll);
  } catch (error) {
    target.textContent = "本機測距失敗 · " + error.message;
  } finally {
    get("live").disabled = false;
  }
}

const steps = ["0_input","1_mask","2_J","3_ac","4_bc","5_sharpness","6_result"];
function updateSteps() {
  const index = Number(get("step").value);
  const distance = get("step-distance").value;
  const row = data.steps[distance];
  const text = ["原始輸入 · f/1.4", "資料集手繪遮罩 · "+row.mask_pixels.toLocaleString()+" 像素",
    "J 明度 · 平均 "+row.J_mean, "a₍c₎ 色彩分量 · 平均 "+row.ac_mean,
    "b₍c₎ 色彩分量 · 平均 "+row.bc_mean, "加權拉普拉斯標準差 σ = "+row.sigma,
    "距離 "+format(row.pred_m)+" m · GT "+row.gt_m+" m · 誤差 "+signed(row.err_m)+" m"];
  get("step-text").textContent = (index+1) + " / 7 · " + text[index];
  get("step-image").src = root + "C-C1_input_output/steps/" + distance + "/" + steps[index] + ".png";
}
function updateOptics() {
  const selected = Number(get("optics").value);
  const spec = [["f1.4_focus15m","f/1.4 · 對焦 15 m","#245a78"],["f1.4_focus7m","f/1.4 · 對焦 7 m","#246951"],["f22_focus15m","f/22 · 對焦 15 m","#a3592b"]];
  const rows = data.optics.filter(row => Number(row.distance_m) >= 1 && Number(row.distance_m) <= 7);
  const current = rows.find(row => Math.abs(Number(row.distance_m)-selected)<1e-5);
  get("optics-value").value = format(selected,2) + " m";
  const readout = get("optics-readout");
  readout.replaceChildren(...spec.map(([key,name,color]) => {
    const span = stamp("span", name + " = " + format(current[key],2) + " px");
    span.style.color = color;
    span.style.marginInlineEnd = "18px";
    return span;
  }));
  plot(get("optics-chart"),spec.map(([key,,color]) => ({color,points:rows.map(row => [Number(row.distance_m),Number(row[key])])})),selected,"模糊直徑 (px)",false,"距離 (m)");
}
function updatePair() {
  const distance = get("pair-distance").value;
  const stem = "red_" + distance;
  const pairs = {
    aperture:["C-D1_defocus_theory/aperture_pairs/",stem+"_f1.4.jpg",stem+"_f22.jpg","f/1.4","f/22"],
    rotation:["C-F1_before_after/rotation_pairs/",stem+"_orig.jpg",stem+"_rot180.jpg","原圖","旋轉 180°"],
    color:["C-F1_before_after/gray_vs_ciecam/",stem+"_gray.png",stem+"_ciecam.png","灰階","CIECAM02"]
  };
  const [folder,left,right,leftName,rightName] = pairs[get("pair-kind").value];
  get("pair-left").src = root+folder+left;
  get("pair-right").src = root+folder+right;
  get("left-label").textContent = leftName;
  get("right-label").textContent = rightName;
  const row = data.rotation.find(item => item.color === "red" && Number(item.shot) === 1 && Number(item.gt_m) === parseFloat(distance));
  table(get("rotation-table"), ["方法","原圖 → 旋轉估測 (m)","原圖 → 旋轉 RMSE (m)"], methods.map(([key,name]) => {
    const summary = data.rotationSummary[key];
    return [name, format(row[key+"_m"])+" → "+format(row[key+"_rot_m"]),format(summary.rmse_orig_m)+" → "+format(summary.rmse_rot_m)];
  }));
}
function updateSplit() {
  const percent = get("split").value + "%";
  get("pair-left").style.clipPath = "inset(0 calc(100% - " + percent + ") 0 0)";
  get("wipe-line").style.left = percent;
}

function updateLeaderboard() {
  const key = get("sort").value;
  const cols = ["rmse_m","mae_m","rmse_near_1_2_to_3_7_m","rmse_far_4_2_to_6_7_m","rmse_red_m","rmse_green_m","rmse_blue_m","rmse_black_m","rmse_rot180_m"];
  const score = row => format(row[key]) === "—" ? Infinity : Number(row[key]);
  const rows = [...data.leaders.rows].sort((a,b) => score(a)-score(b)).map((row,index) =>
    [index+1, row.method, row.year || "—", ...cols.map(col => format(row[col])), row.inputs, row.calibration]);
  table(get("leaders"),["名次","方法","年份","RMSE (m)","MAE (m)","近","遠","紅","綠","藍","黑","旋轉","輸入","校正"],rows,row => row[1] === data.leaders.rows.find(item => item.ours)?.method);
}
function updateError() {
  const points = Object.entries(data.analysis.error_vs_distance[get("error-method").value])
    .map(([distance,rmse]) => [Number(distance),rmse]).sort((a,b) => a[0]-b[0]);
  plot(get("error-chart"),[{color:"#245a78",points}],null,"RMSE (m)");
}
function updateAnalysis() {
  const url = root+"C-I1_analysis/"+get("analysis-select").value;
  get("analysis-image").src = url;
  get("analysis-link").href = url;
}
function gallery() {
  const videos = [
    ["車輛接近","scenario_replays/approach.mp4","scenario_replays/red_6.7m.jpg","scenario_replays/approach.csv"],
    ["前車往返","scenario_replays/roundtrip.mp4","scenario_replays/red_3.7m.jpg","scenario_replays/roundtrip.csv"],
    ["四車色切換","scenario_replays/colors.mp4","scenario_replays/blue_3.7m.jpg","scenario_replays/colors.csv"],
    ["輸入到輸出","C-C1_input_output/video_short.mp4","C-C1_input_output/poster.jpg","C-C1_input_output/per_frame.csv"],
    ["四方法同步跑分","C-E1_method_compare/video_short.mp4","C-E1_method_compare/poster.jpg","C-E1_method_compare/per_frame.csv"],
    ["原圖與旋轉 180°","C-F1_before_after/video_short.mp4","C-F1_before_after/poster.jpg","C-F1_before_after/per_frame.csv"],
    ["網頁操作與 CUDA 測距","scenario_replays/walkthrough.mp4","scenario_replays/walkthrough.jpg"]
  ];
  for (const [title, path, poster, csv] of videos) {
    const figure = stamp("figure","");
    figure.className = "card media-card";
    const video = Object.assign(document.createElement("video"),{src:root+path,poster:root+poster,preload:"none",controls:true,playsInline:true});
    video.setAttribute("aria-label",title);
    if (path.endsWith("approach.mp4") || path.startsWith("C-E1")) {
      video.muted = true;
      video.loop = true;
      video.dataset.autoplay = "";
    }
    const caption = stamp("figcaption",title+(csv ? " · " : ""));
    if (csv) caption.append(Object.assign(stamp("a","逐幀 CSV"),{href:root+csv,download:""}));
    figure.append(video,caption);
    get("videos").append(figure);
  }
  for (const {file,caption} of data.failures) {
    const figure = stamp("figure","");
    figure.className = "card media-card";
    const image = Object.assign(document.createElement("img"),{src:root+"C-K1_failure_cases/"+file,alt:caption,loading:"lazy"});
    figure.append(image,stamp("figcaption",caption));
    get("failures").append(figure);
  }
}

async function initialize() {
  const response = await fetch("/api/data");
  if (!response.ok) throw Error("讀取資料失敗");
  data = await response.json();
  get("error-method").replaceChildren(...Object.keys(data.analysis.error_vs_distance).map(key => new Option(labels[key] || key,key)));
  get("error-method").value = "opt";
  gallery();
  updateComparison();
  updateSteps();
  updateOptics();
  updatePair();
  updateSplit();
  updateLeaderboard();
  updateError();
  updateAnalysis();
  const exposure = data.robustness.filter(row => row.case.startsWith("exposure_x")).map(row =>
    [Number(row.case.split("x").at(-1)),Number(row.cadfds_rmse_m)]);
  const texture = data.texture.map(row => [Number(row.texture_f22_mean),Number(row.cadfds_rmse_m)]);
  plot(get("exposure-chart"),[{color:"#a3592b",points:exposure}],null,"RMSE (m)",false,"曝光乘數");
  plot(get("texture-chart"),[{color:"#246951",points:texture}],null,"RMSE (m)",false,"紋理量",true);
  table(get("timing"),["管線","延遲中位數 (ms)","FPS","RMSE (m)"],Object.entries(data.timing.pipelines).map(([key,item]) =>
    [key,format(item.total_median_ms,2),format(item.fps,2),format(item.rmse_m,4)]));
  get("mask-timing").textContent = "YOLO + SAM 自動遮罩含 I/O 耗時 " + data.timing.auto_mask.wall_incl_io + " ms。此為獨立遮罩計時。";
  const handlers = {color:() => {stopPlayback(); updateComparison();},frame:() => {stopPlayback(); updateComparison();},step:updateSteps,"step-distance":updateSteps,
    optics:updateOptics,"pair-kind":updatePair,"pair-distance":updatePair,split:updateSplit,sort:updateLeaderboard,
    "error-method":updateError,"analysis-select":updateAnalysis,
    view:() => {get("camera-image").src = root + "C-B1_camera_setup/" + get("view").value;}};
  for (const [id,handler] of Object.entries(handlers)) get(id).addEventListener("input",handler);
  get("play").addEventListener("click",() => {autoFrames = !autoFrames; updatePlayback();});
  configureMotion();
  get("live").addEventListener("click",runInference);
  const wipe = get("wipe");
  function drag(event) {
    const rect = wipe.getBoundingClientRect();
    get("split").value = (event.clientX-rect.left)/rect.width*100;
    updateSplit();
  }
  wipe.addEventListener("pointerdown",event => {wipe.setPointerCapture(event.pointerId); drag(event);});
  wipe.addEventListener("pointermove",event => {if (wipe.hasPointerCapture(event.pointerId)) drag(event);});
  window.addEventListener("resize",() => plots.forEach((args,canvas) => plot(canvas,...args)));
}
initialize().catch(error => {get("load-status").textContent = "展示資料載入失敗 · " + error.message;});
