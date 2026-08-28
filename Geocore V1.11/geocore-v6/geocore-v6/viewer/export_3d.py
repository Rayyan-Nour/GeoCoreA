"""
Export pipeline outputs to a single self-contained interactive 3-D HTML viewer.

Design goals (rebuilt from scratch):

* CORRECT: geographic (lat/lon) DEMs produce terrain extents in METERS
  (the black-screen regression in tests/test_viewer3d.py), projected DEMs
  keep native extents exactly.
* WORKING interaction: raycast hover shows lat/lon + elevation continuously;
  click drops a query pin and reads every layer value at that pixel, plus
  nearest target / known deposit. Targets and deposits are hoverable.
* NO FLICKER by construction: one opaque terrain mesh with ONE composited
  texture (draped layer multiplied by hillshade on an offscreen canvas) -
  no coplanar overlays to z-fight; capped devicePixelRatio; a single
  requestAnimationFrame loop; textures never swapped per-frame.
* Self-contained: one .html, three.js from CDN with an offline notice,
  no other dependencies. Custom orbit controls (drag orbits, right-drag
  pans, scroll zooms) - no OrbitControls import.

Usage:
    from viewer.export_3d import export_viewer_3d
    export_viewer_3d(run_dir, out_html, deposits_csv=optional)
"""
from __future__ import annotations

import csv as _csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyproj import CRS, Transformer

from .export_web import (_read, _colormap_png, _value_grid, _hillshade,
                         LAYER_SPECS)


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------

def _metric_extents(shape: Tuple[int, int], transform, crs: str
                    ) -> Tuple[float, float, bool]:
    """(extentX_m, extentY_m, is_geographic). Geographic degrees -> meters
    at the DEM's central latitude; projected grids keep native units."""
    H, W = shape
    px = abs(transform.a)
    py = abs(transform.e)
    geographic = False
    try:
        geographic = CRS.from_user_input(crs).is_geographic
    except Exception:
        geographic = False
    if geographic:
        y_top = transform.f
        lat_c = y_top + (H / 2.0) * transform.e   # transform.e < 0
        m_lat = 111_320.0
        m_lon = 111_320.0 * max(math.cos(math.radians(lat_c)), 1e-6)
        return W * px * m_lon, H * py * m_lat, True
    return W * px, H * py, False


def _heightfield(dem: np.ndarray, max_px: int = 220) -> Dict:
    """Downsampled height grid for the mesh; null where invalid."""
    H, W = dem.shape
    step = max(1, int(np.ceil(max(H, W) / max_px)))
    sub = dem[::step, ::step].astype(np.float64)
    out = [[(None if not np.isfinite(v) else round(float(v), 2))
            for v in row] for row in sub]
    return {"rows": sub.shape[0], "cols": sub.shape[1], "z": out}


def _wgs84_bounds(shape, transform, crs) -> Optional[List[List[float]]]:
    try:
        H, W = shape
        xs = [transform.c, transform.c + W * transform.a]
        ys = [transform.f, transform.f + H * transform.e]
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lons, lats = [], []
        for x in xs:
            for y in ys:
                lon, lat = tr.transform(x, y)
                if not (np.isfinite(lon) and np.isfinite(lat)):
                    return None
                lons.append(lon); lats.append(lat)
        return [[min(lats), min(lons)], [max(lats), max(lons)]]
    except Exception:
        return None


def _frac_from_xy(x: float, y: float, shape, transform
                  ) -> Optional[Tuple[float, float]]:
    """Grid-CRS coords -> (fx east 0..1, fy north-to-south 0..1)."""
    H, W = shape
    x0 = transform.c
    x1 = transform.c + W * transform.a
    y0 = transform.f                     # top (north)
    y1 = transform.f + H * transform.e   # bottom
    fx = (x - x0) / (x1 - x0)
    fy = (y - y0) / (y1 - y0)
    if not (np.isfinite(fx) and np.isfinite(fy)):
        return None
    if fx < -0.02 or fx > 1.02 or fy < -0.02 or fy > 1.02:
        return None
    return float(np.clip(fx, 0, 1)), float(np.clip(fy, 0, 1))


def _latlon_to_frac(lat: float, lon: float, shape, transform, crs,
                    geographic: bool) -> Optional[Tuple[float, float]]:
    if geographic:
        return _frac_from_xy(lon, lat, shape, transform)
    try:
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = tr.transform(lon, lat)
        return _frac_from_xy(x, y, shape, transform)
    except Exception:
        return None


# ----------------------------------------------------------------------
# data readers
# ----------------------------------------------------------------------

def _load_targets(run_dir: Path, shape, transform, crs, geographic
                  ) -> List[Dict]:
    p = run_dir / "geocore_targets.csv"
    if not p.exists():
        return []
    out: List[Dict] = []
    with open(p, newline="", encoding="utf-8") as fh:
        rd = _csv.DictReader(fh)
        if not rd.fieldnames:
            return []
        for row in rd:
            try:
                lat = float(row["latitude"]); lon = float(row["longitude"])
                prob = float(row["probability"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(lat) <= 90 and abs(lon) <= 360:
                fr = _latlon_to_frac(lat, lon, shape, transform, crs,
                                     geographic)
            else:  # grid coords written straight through (no-CRS runs)
                fr = _frac_from_xy(lon, lat, shape, transform)
            if fr is None:
                continue

            def _f(key):
                v = (row.get(key) or "").strip()
                try:
                    return float(v)
                except ValueError:
                    return None

            out.append({
                "fx": round(fr[0], 5), "fy": round(fr[1], 5),
                "rank": int(float(row.get("rank") or len(out) + 1)),
                "prob": round(prob, 3),
                "unc": _f("uncertainty"),
                "ds": _f("depth_spectral_m"),
                "de": _f("depth_euler_m"),
                "status": (row.get("depth_status") or "").strip() or None,
                "near_km": _f("nearest_deposit_km"),
                "near_name": (row.get("nearest_deposit_name") or "").strip()
                             or None,
                "driver": (row.get("driver_1") or "").strip() or None,
                "driver_pct": _f("driver_1_percentile"),
            })
    return out


def _load_deposits(csv_path, shape, transform, crs, geographic) -> List[Dict]:
    if not csv_path or not Path(csv_path).exists():
        return []
    out: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = _csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        xc = next((cols[k] for k in ("longitude", "lon", "x", "east",
                                     "easting", "dec_long") if k in cols),
                  None)
        yc = next((cols[k] for k in ("latitude", "lat", "y", "north",
                                     "northing", "dec_lat") if k in cols),
                  None)
        nc = next((cols[k] for k in ("site_name", "name", "dep_name")
                   if k in cols), None)
        if not (xc and yc):
            return []
        for row in rd:
            try:
                x = float(row[xc]); y = float(row[yc])
            except (TypeError, ValueError):
                continue
            if abs(x) <= 360 and abs(y) <= 90:      # geographic-looking
                fr = _latlon_to_frac(y, x, shape, transform, crs, geographic)
            else:                                    # grid coords
                fr = _frac_from_xy(x, y, shape, transform)
            if fr is None:
                continue
            out.append({"fx": round(fr[0], 5), "fy": round(fr[1], 5),
                        "name": (row.get(nc) or "deposit") if nc
                                else "deposit"})
            if len(out) >= 4000:
                break
    return out


# ----------------------------------------------------------------------
# main exporter
# ----------------------------------------------------------------------

def export_viewer_3d(run_dir: Path | str, out_html: Path | str,
                     deposits_csv: Optional[Path | str] = None,
                     title: str = "GeoCore Analytics — 3D Target Review"
                     ) -> Path:
    run_dir = Path(run_dir)
    out_html = Path(out_html)

    dem_path = run_dir / "geocore_dem.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"missing {dem_path}; run the pipeline first")
    dem, transform, crs = _read(dem_path)
    shape = dem.shape

    extent_x, extent_y, geographic = _metric_extents(shape, transform, crs)
    finite = dem[np.isfinite(dem)]
    zmin = float(finite.min()) if finite.size else 0.0
    zmax = float(finite.max()) if finite.size else 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    hs_uri, _, _ = _colormap_png(_hillshade(dem, transform), "gray", 0, 1)

    layers, legends, grads, grids = [], {}, {}, {}
    for fname, lid, name, cmap, vmin, vmax, fmt, unit in LAYER_SPECS:
        p = run_dir / fname
        if not p.exists():
            continue
        data, _, _ = _read(p)
        if not np.isfinite(data).any():
            continue
        uri, lo, hi = _colormap_png(data, cmap, vmin, vmax)
        layers.append({"id": lid, "name": name, "png": uri,
                       "fmt": fmt, "unit": unit})
        legends[lid] = {"lo": round(lo, 4), "hi": round(hi, 4)}
        grids[lid] = _value_grid(data)
        try:
            import matplotlib
            ramp = (matplotlib.colormaps[cmap](np.linspace(0, 1, 64)) * 255
                    ).astype(np.uint8)[None, :, :]
            from PIL import Image
            import io as _io, base64 as _b64
            img = Image.fromarray(ramp, "RGBA").resize((256, 10),
                                                       Image.NEAREST)
            b = _io.BytesIO(); img.save(b, format="PNG")
            grads[lid] = ("data:image/png;base64,"
                          + _b64.b64encode(b.getvalue()).decode())
        except Exception:
            grads[lid] = ""

    payload = {
        "title": title,
        "extentX": extent_x, "extentY": extent_y,
        "zmin": zmin, "zmax": zmax,
        "geographic": geographic,
        "bounds": _wgs84_bounds(shape, transform, crs),
        "hf": _heightfield(dem),
        "hillshade": hs_uri,
        "layers": layers, "legends": legends, "gradients": grads,
        "grids": grids,
        "targets": _load_targets(run_dir, shape, transform, crs, geographic),
        "deposits": _load_deposits(deposits_csv, shape, transform, crs,
                                   geographic),
    }

    boot = ("const GC = " + json.dumps(payload)
            + ";\nif (typeof THREE === \"undefined\") {\n"
              "  document.getElementById(\"offline\").style.display = \"grid\";\n"
              "} else { init(); }")
    html = _HTML_TEMPLATE.replace("/*__BOOT__*/", boot)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


# ----------------------------------------------------------------------
# template (brand: deep slate-teal ground, bone text, ochre signal,
# sage validated; mono instrument type; no external fonts required)
# ----------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoCore Analytics — 3D Target Review</title>
<style>
  :root{ --ink:#0d1418; --ink2:#13201f; --line:rgba(233,227,213,.12);
         --bone:#e9e3d5; --dim:#8d978f; --signal:#d98a3d; --true:#69a892; }
  html,body{height:100%;margin:0;background:var(--ink);color:var(--bone);
    font-family:ui-monospace,'IBM Plex Mono','Cascadia Mono',Consolas,monospace;
    font-size:13px;overflow:hidden}
  #stage{position:absolute;inset:0}
  canvas{display:block;outline:none}
  #panel{position:absolute;top:14px;left:14px;z-index:10;width:270px;
    background:color-mix(in srgb,var(--ink2) 92%,transparent);
    border:1px solid var(--line);border-radius:6px;padding:14px 16px;
    backdrop-filter:blur(8px);max-height:calc(100% - 28px);overflow:auto}
  #panel h1{font-size:14px;margin:0 0 2px;color:var(--signal);
    letter-spacing:.04em}
  #panel .sub{color:var(--dim);font-size:10.5px;letter-spacing:.06em;
    margin-bottom:12px;text-transform:uppercase}
  .row{margin:10px 0}
  .row label{display:block;color:var(--dim);font-size:10px;
    letter-spacing:.14em;text-transform:uppercase;margin-bottom:5px}
  select,input[type=range]{width:100%;box-sizing:border-box;
    accent-color:var(--signal)}
  select{background:var(--ink);color:var(--bone);border:1px solid var(--line);
    border-radius:3px;padding:7px 8px;font:inherit}
  .legend{display:flex;align-items:center;gap:6px;margin-top:5px;
    font-size:10px;color:var(--dim)}
  .legend img{height:8px;flex:1;border-radius:2px}
  .views{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:12px}
  .views button{background:var(--ink);color:var(--bone);
    border:1px solid var(--line);border-radius:3px;padding:8px 0;font:inherit;
    font-size:11px;letter-spacing:.06em;cursor:pointer}
  .views button:hover{border-color:var(--signal);color:var(--signal)}
  .key{border-top:1px solid var(--line);margin-top:12px;padding-top:10px;
    font-size:11px;color:var(--dim);line-height:1.9}
  .key .sw{display:inline-block;width:9px;height:9px;border-radius:2px;
    margin-right:7px;vertical-align:-1px}
  #readout{position:absolute;left:50%;bottom:16px;transform:translateX(-50%);
    z-index:10;background:color-mix(in srgb,var(--ink2) 94%,transparent);
    border:1px solid var(--line);border-radius:4px;padding:8px 16px;
    font-size:12px;letter-spacing:.03em;white-space:nowrap;color:#bfe7d8}
  #query{position:absolute;top:14px;right:14px;z-index:10;width:250px;
    background:color-mix(in srgb,var(--ink2) 94%,transparent);
    border:1px solid var(--line);border-radius:6px;padding:12px 14px;
    display:none;font-size:12px;line-height:1.85}
  #query h2{font-size:11px;margin:0 0 6px;color:var(--signal);
    letter-spacing:.14em;text-transform:uppercase}
  #query .v{color:#bfe7d8}
  #query .t{color:var(--signal)}
  #hint{position:absolute;left:14px;bottom:14px;z-index:10;color:var(--dim);
    font-size:10.5px;letter-spacing:.04em}
  #offline{position:absolute;inset:0;display:none;place-items:center;
    z-index:50;background:var(--ink);text-align:center;padding:40px}
</style>
</head>
<body>
<div id="stage"></div>
<div id="offline"><div><h2>3-D library unavailable</h2>
  <p>This viewer loads three.js from a CDN. Connect to the internet once,
     then reopen this file.</p></div></div>

<div id="panel">
  <h1 id="ttl">GeoCore 3D Target Review</h1>
  <div class="sub">hover reads ground · click queries a pixel</div>
  <div class="row"><label>Draped layer</label>
    <select id="layerSel"></select>
    <div class="legend"><span id="lgLo"></span><img id="lgImg">
      <span id="lgHi"></span></div></div>
  <div class="row"><label>Layer opacity <span id="opV">80%</span></label>
    <input type="range" id="op" min="0" max="100" value="80"></div>
  <div class="row"><label>Vertical exaggeration <span id="exV">2.0×</span></label>
    <input type="range" id="ex" min="5" max="60" value="20"></div>
  <div class="views">
    <button data-v="reset">Reset</button><button data-v="top">Top</button>
    <button data-v="oblique">Oblique</button><button data-v="side">Side</button>
  </div>
  <div class="key">
    <span class="sw" style="background:var(--bone)"></span>Known deposit (training)<br>
    <span class="sw" style="background:var(--signal)"></span>Predicted target (ranked)<br>
    <span class="sw" style="background:var(--true)"></span>Your query point
  </div>
</div>

<div id="query"></div>
<div id="readout">move over the terrain…</div>
<div id="hint">drag orbits · right-drag pans · scroll zooms · click queries</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
function init(){
  const D = GC, stage = document.getElementById("stage");
  document.getElementById("ttl").textContent = D.title;

  // ---------- renderer / scene ----------
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  stage.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1418);
  const span = Math.max(D.extentX, D.extentY);
  scene.fog = new THREE.Fog(0x0d1418, span*1.6, span*4.5);
  scene.add(new THREE.HemisphereLight(0xbcd8d2, 0x1a2320, 0.85));
  const sun = new THREE.DirectionalLight(0xf2e2c8, 0.9);
  sun.position.set(-0.6*span, 0.9*span, 0.4*span); scene.add(sun);

  const camera = new THREE.PerspectiveCamera(
      50, innerWidth/innerHeight, span/2000, span*10);

  // ---------- heightfield helpers ----------
  const hf = D.hf, R = hf.rows, C = hf.cols, zr = (D.zmax - D.zmin) || 1;
  function hAt(fx, fy){                       // bilinear sample, meters>zmin
    const gx = fx*(C-1), gy = fy*(R-1);
    const x0 = Math.floor(gx), y0 = Math.floor(gy);
    const x1 = Math.min(C-1, x0+1), y1 = Math.min(R-1, y0+1);
    const tx = gx-x0, ty = gy-y0;
    const g = (r,c)=>{ const v = hf.z[r][c]; return v===null? D.zmin : v; };
    const a = g(y0,x0)*(1-tx)+g(y0,x1)*tx, b = g(y1,x0)*(1-tx)+g(y1,x1)*tx;
    return (a*(1-ty)+b*ty) - D.zmin;
  }

  // ---------- terrain mesh (ONE mesh, ONE texture: no z-fighting) ----------
  let exag = 2.0;
  const geo = new THREE.PlaneGeometry(D.extentX, D.extentY, C-1, R-1);
  geo.rotateX(-Math.PI/2);                    // +x east, +z south, +y up
  const pos = geo.attributes.position;
  for (let r=0; r<R; r++) for (let c=0; c<C; c++){
    const i = r*C + c, v = hf.z[r][c];
    pos.setY(i, (v===null? 0 : v - D.zmin));
  }
  geo.computeVertexNormals();
  const mat = new THREE.MeshLambertMaterial({color:0xffffff});
  const terrain = new THREE.Mesh(geo, mat);
  terrain.scale.y = exag;
  scene.add(terrain);

  // composited draped texture: layer (with opacity) x hillshade
  const texCanvas = document.createElement("canvas");
  texCanvas.width = texCanvas.height = 1024;
  const tctx = texCanvas.getContext("2d");
  const tex = new THREE.CanvasTexture(texCanvas);
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  mat.map = tex;
  const imgs = {};                             // preloaded Images
  function preload(id, uri){ return new Promise(res=>{
    const im = new Image(); im.onload = ()=>{ imgs[id]=im; res(); };
    im.onerror = ()=>res(); im.src = uri; }); }
  let curLayer = D.layers.length ? D.layers[0].id : null, opacity = 0.8;
  function compose(){
    tctx.globalCompositeOperation = "source-over";
    tctx.fillStyle = "#8f948f";
    tctx.fillRect(0,0,1024,1024);
    if (curLayer && imgs[curLayer]){
      tctx.globalAlpha = opacity;
      tctx.drawImage(imgs[curLayer], 0,0,1024,1024);
      tctx.globalAlpha = 1;
    }
    if (imgs.__hs){
      tctx.globalCompositeOperation = "multiply";
      tctx.drawImage(imgs.__hs, 0,0,1024,1024);
    }
    tex.needsUpdate = true;
  }

  // ---------- markers ----------
  const markers = new THREE.Group(); scene.add(markers);
  const pickables = [terrain];
  const mScale = span/110;
  function place(fx, fy){ return new THREE.Vector3(
      (fx-0.5)*D.extentX, hAt(fx,fy)*exag, (fy-0.5)*D.extentY); }

  const depGeo = new THREE.ConeGeometry(mScale*0.30, mScale*0.9, 6);
  const depMat = new THREE.MeshLambertMaterial({color:0xe9e3d5});
  D.deposits.forEach(d=>{
    const m = new THREE.Mesh(depGeo, depMat);
    m.userData = {kind:"deposit", d};
    markers.add(m); pickables.push(m);
  });

  const tGeo = new THREE.OctahedronGeometry(mScale*0.42);
  const tMat = new THREE.MeshLambertMaterial(
      {color:0xd98a3d, emissive:0x7a4413});
  const rings = [];
  D.targets.forEach(t=>{
    const m = new THREE.Mesh(tGeo, tMat.clone());
    m.userData = {kind:"target", t};
    markers.add(m); pickables.push(m);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(mScale*0.7, mScale*0.85, 40),
      new THREE.MeshBasicMaterial({color:0xd98a3d, transparent:true,
        opacity:0.55, side:THREE.DoubleSide}));
    ring.rotation.x = -Math.PI/2; ring.userData = {t};
    markers.add(ring); rings.push(ring);
  });
  const pin = new THREE.Mesh(new THREE.SphereGeometry(mScale*0.3, 18, 14),
      new THREE.MeshLambertMaterial({color:0x69a892, emissive:0x1d4436}));
  pin.visible = false; scene.add(pin);

  function layoutMarkers(){
    let i = 0;
    markers.children.forEach(m=>{
      const u = m.userData;
      if (u.kind === "deposit"){
        const p = place(u.d.fx, u.d.fy); m.position.set(p.x, p.y+mScale*0.45, p.z);
      } else if (u.kind === "target"){
        const p = place(u.t.fx, u.t.fy); m.position.set(p.x, p.y+mScale*0.55, p.z);
      } else if (u.t){                       // ring
        const p = place(u.t.fx, u.t.fy); m.position.set(p.x, p.y+mScale*0.06, p.z);
      }
      i++;
    });
  }

  // ---------- custom controls ----------
  const center = new THREE.Vector3(0, zr*exag*0.35, 0);
  let radius = span*1.15, theta = Math.PI*0.35, phi = Math.PI*0.32;
  function applyCam(){
    phi = Math.max(0.05, Math.min(Math.PI/2 - 0.02, phi));
    radius = Math.max(span*0.03, Math.min(span*5, radius));
    camera.position.set(
      center.x + radius*Math.sin(phi)*Math.sin(theta),
      center.y + radius*Math.cos(phi),
      center.z + radius*Math.sin(phi)*Math.cos(theta));
    camera.lookAt(center);
  }
  const views = {
    oblique(){ theta = Math.PI*0.35; phi = Math.PI*0.32; radius = span*1.15;
               center.set(0, zr*exag*0.35, 0); },
    top(){ theta = 0; phi = 0.06; radius = span*1.5;
           center.set(0, 0, 0); },
    side(){ theta = Math.PI/2; phi = Math.PI*0.46; radius = span*1.2;
            center.set(0, zr*exag*0.4, 0); },
    reset(){ views.oblique(); }
  };
  views.oblique(); applyCam();

  let drag = null;
  const el = renderer.domElement;
  el.addEventListener("contextmenu", e=>e.preventDefault());
  el.addEventListener("pointerdown", e=>{
    drag = {b:e.button, x:e.clientX, y:e.clientY, moved:false};
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointerup", e=>{
    if (drag && !drag.moved) onClick(e);
    drag = null;
  });
  el.addEventListener("pointermove", e=>{
    if (drag){
      const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
      if (Math.abs(dx)+Math.abs(dy) > 3) drag.moved = true;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.b === 2 || e.shiftKey){       // pan
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix,0);
        const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix,1);
        center.addScaledVector(right, -dx*radius*0.0016);
        center.addScaledVector(up,  dy*radius*0.0016);
      } else {                                // orbit
        theta -= dx*0.006; phi -= dy*0.006;
      }
      applyCam();
    } else { hoverPtr = {x:e.clientX, y:e.clientY}; }
  });
  el.addEventListener("wheel", e=>{
    e.preventDefault();
    radius *= (1 + Math.sign(e.deltaY)*0.09); applyCam();
  }, {passive:false});

  // ---------- raycast hover + click ----------
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let hoverPtr = null;
  const readout = document.getElementById("readout");
  const [[S,Wd],[N,E]] = D.bounds || [[0,0],[0,0]];
  function coordText(fx, fy){
    if (D.bounds){
      const lat = N - fy*(N-S), lon = Wd + fx*(E-Wd);
      return "Lat " + lat.toFixed(5) + "\u00B0  Lon " + lon.toFixed(5) + "\u00B0";
    }
    return "E " + (fx*D.extentX).toFixed(0) + " m  N "
         + ((1-fy)*D.extentY).toFixed(0) + " m";
  }
  function gridVal(id, fx, fy){
    const g = D.grids[id]; if (!g) return null;
    const r = Math.min(g.rows-1, Math.floor(fy*g.rows));
    const c = Math.min(g.cols-1, Math.floor(fx*g.cols));
    return g.values[r][c];
  }
  function castAt(px, py){
    ndc.x = (px/innerWidth)*2-1; ndc.y = -(py/innerHeight)*2+1;
    ray.setFromCamera(ndc, camera);
    return ray.intersectObjects(pickables, false)[0] || null;
  }
  function doHover(){
    if (!hoverPtr) return;
    const hit = castAt(hoverPtr.x, hoverPtr.y);
    if (!hit){ readout.textContent = "off terrain"; return; }
    const u = hit.object.userData || {};
    if (u.kind === "target"){
      const t = u.t;
      readout.textContent = "T-" + String(t.rank).padStart(2,"0")
        + " \u00B7 p=" + t.prob.toFixed(2)
        + (t.status ? " \u00B7 depth " + t.status : "");
      return;
    }
    if (u.kind === "deposit"){
      readout.textContent = "Known deposit \u00B7 " + u.d.name; return;
    }
    const fx = (hit.point.x + D.extentX/2)/D.extentX;
    const fy = (hit.point.z + D.extentY/2)/D.extentY;
    const elev = hit.point.y/exag + D.zmin;
    readout.textContent = coordText(fx, fy)
      + "  Elev " + elev.toFixed(0) + " m";
  }
  const query = document.getElementById("query");
  function onClick(e){
    const hit = castAt(e.clientX, e.clientY);
    if (!hit){ query.style.display = "none"; pin.visible = false; return; }
    const u = hit.object.userData || {};
    if (u.kind === "target"){ showTarget(u.t); return; }
    if (u.kind === "deposit"){
      query.innerHTML = "<h2>Known deposit</h2><span class='v'>"
        + u.d.name + "</span>";
      query.style.display = "block"; return;
    }
    const fx = (hit.point.x + D.extentX/2)/D.extentX;
    const fy = (hit.point.z + D.extentY/2)/D.extentY;
    pin.position.copy(hit.point); pin.position.y += mScale*0.2;
    pin.visible = true;
    let h = "<h2>Pixel query</h2><span class='v'>" + coordText(fx,fy)
      + "</span><br>Elevation <span class='v'>"
      + (hit.point.y/exag + D.zmin).toFixed(0) + " m</span><br>";
    D.layers.forEach(ly=>{
      const v = gridVal(ly.id, fx, fy);
      h += ly.name + ": <span class='v'>"
        + (v===null ? "no data"
           : v.toFixed(ly.fmt === ".0f" ? 0 : 3) + ly.unit) + "</span><br>";
    });
    let best = null, bd = 1e9;
    D.targets.forEach(t=>{
      const dx = (t.fx-fx)*D.extentX, dz = (t.fy-fy)*D.extentY;
      const d = Math.hypot(dx,dz); if (d < bd){ bd = d; best = t; }
    });
    if (best) h += "Nearest target: <span class='t'>T-"
      + String(best.rank).padStart(2,"0") + "</span> \u00B7 "
      + (bd/1000).toFixed(2) + " km";
    query.innerHTML = h; query.style.display = "block";
  }
  function showTarget(t){
    let h = "<h2>Target T-" + String(t.rank).padStart(2,"0") + "</h2>"
      + "Probability <span class='t'>" + t.prob.toFixed(3) + "</span><br>"
      + (t.unc!==null && t.unc!==undefined
         ? "Uncertainty <span class='v'>\u00B1" + t.unc.toFixed(2) + "</span><br>" : "")
      + (t.driver ? "Top driver <span class='v'>" + t.driver
         + (t.driver_pct ? " (P" + t.driver_pct.toFixed(0) + ")" : "")
         + "</span><br>" : "")
      + (t.ds ? "Depth (spectral) <span class='v'>" + t.ds.toFixed(0)
        + " m</span><br>" : "")
      + (t.de ? "Depth (Euler) <span class='v'>" + t.de.toFixed(0)
        + " m</span><br>" : "")
      + (t.status ? "Depth status <span class='"
        + (t.status === "corroborated" ? "v" : "t") + "'>" + t.status
        + "</span><br>" : "")
      + (t.near_name ? "Nearest known deposit <span class='v'>"
        + t.near_name + (t.near_km !== null ? " \u00B7 "
        + t.near_km.toFixed(1) + " km" : "") + "</span><br>" : "")
      + "<span style='color:var(--dim);font-size:10px'>Ranking for "
      + "follow-up \u2014 not a discovery claim. Depths are magnetic "
      + "source depths, not ore depths.</span>";
    query.innerHTML = h; query.style.display = "block";
  }

  // ---------- UI wiring ----------
  const sel = document.getElementById("layerSel");
  D.layers.forEach((ly,i)=>{
    const o = document.createElement("option");
    o.value = ly.id; o.textContent = ly.name; sel.appendChild(o);
  });
  const noneOpt = document.createElement("option");
  noneOpt.value = ""; noneOpt.textContent = "Hillshade only";
  sel.appendChild(noneOpt);
  function legend(){
    const lg = curLayer ? D.legends[curLayer] : null;
    document.getElementById("lgLo").textContent = lg ? lg.lo.toFixed(2) : "";
    document.getElementById("lgHi").textContent = lg ? lg.hi.toFixed(2) : "";
    document.getElementById("lgImg").src =
      curLayer && D.gradients[curLayer] ? D.gradients[curLayer] : "";
  }
  sel.addEventListener("change", ()=>{ curLayer = sel.value || null;
    legend(); compose(); });
  document.getElementById("op").addEventListener("input", e=>{
    opacity = e.target.value/100;
    document.getElementById("opV").textContent = e.target.value + "%";
    compose();
  });
  document.getElementById("ex").addEventListener("input", e=>{
    exag = e.target.value/10;
    document.getElementById("exV").textContent = exag.toFixed(1) + "\u00D7";
    terrain.scale.y = exag; layoutMarkers();
  });
  document.querySelectorAll(".views button").forEach(b=>
    b.addEventListener("click", ()=>{ views[b.dataset.v](); applyCam(); }));

  addEventListener("resize", ()=>{
    camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  // ---------- boot ----------
  const loads = [preload("__hs", D.hillshade)];
  D.layers.forEach(ly=>loads.push(preload(ly.id, ly.png)));
  Promise.all(loads).then(()=>{ compose(); legend(); });
  layoutMarkers();

  let t0 = performance.now();
  (function loop(now){
    requestAnimationFrame(loop);
    const t = (now - t0)/1000;
    rings.forEach((r,i)=>{
      const s = 1 + 0.14*Math.sin(t*2.6 + i*0.9);
      r.scale.set(s, s, s);
      r.material.opacity = 0.38 + 0.2*Math.sin(t*2.6 + i*0.9);
    });
    doHover();
    renderer.render(scene, camera);
  })(t0);
}
/*__BOOT__*/
</script>
</body>
</html>
"""
