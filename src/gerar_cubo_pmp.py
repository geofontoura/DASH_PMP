"""Modelo 3D interativo do PMP ("cubão") -- mesmas funcionalidades do
viewer 3D do Taió (gerar_visualizador_3d.py), com os dados do PMP:

  - topografia real (curvas3.shp) + 7 formações sedimentares (planos paralelos
    ajustados nos furos/afloramentos, erosão em cascata) + 3 sills reais +
    FUROS de sondagem em 3D (colunas coloridas pelas unidades atravessadas);
  - modos de cor da topografia: Hipsometria / Geologia real (CPRM) / Satélite;
  - Sólido ON/OFF, Topografia ON/OFF, Furos ON/OFF, opacidade, exagero vertical;
  - ferramenta de CORTE em 4 direções (Leste→Oeste, Oeste→Leste, Norte→Sul,
    Sul→Norte) com slider e animação, expondo a estratigrafia no bloco;
  - tema escuro/claro e legenda clicável (nativa do Plotly).

Diferença técnica pro Taió: lá cada posição de corte é um "frame" Plotly com o
estado inteiro (por isso o HTML de 95 MB e a grade grossa de 63x63). Aqui o
Python só entrega as GRADES (terreno, contatos, cores) e o navegador monta as
malhas e faz o recorte em JavaScript -- o arquivo fica com poucos MB, com grade
de 130x130 e o corte é contínuo (interpola o plano exatamente onde você para).

Fontes: ../2_Banco_de_Dados (area2/curvas3/Litologia_PMP2.shp + banco de poços)
via _comum_pmp.py; satélite Esri World Imagery (baixado uma vez, em cache).
Este script só LÊ essas fontes.

Uso:
    python gerar_cubo_pmp.py
Gera:
    cubo_pmp.html
"""
import base64
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from _comum_pmp import (
    MARCA_ROXO, MARCA_ROXO_ESCURO, MARCA_NAVY, MARCA_CINZA_CLARO, MARCA_FONTE,
    UNIDADES_ESTILIZADO, NOMES_ESTILIZADO, CORES_ESTILIZADO, ESPESSURA_SILL_M,
    ORDEM_PROFUNDIDADE_FURO, CORES_HIPSOMETRICAS, COR_POR_SIGLA, COR_LITOLOGIA_PADRAO,
    logo_base64, carregar_area_pmp, carregar_vertices_curvas_pmp, carregar_litologia_pmp,
    carregar_sills_individualizados, construir_interpolador, avaliar_interpolador,
    pontos_dentro_poligono, calcular_planos_estilizados, calcular_contatos_estilizados,
    obter_satelite_utm, amostrar_satelite_rgb, preparar_furos, _intervalos_corpo, _hex_rgb,
    LEAFLET_LINKS,
)

BASE = Path(__file__).resolve().parent
OUT_HTML = BASE / "cubo_pmp.html"
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{__import__('plotly').offline.get_plotlyjs_version()}.min.js"

N = 130                 # grade do terreno/contatos (nós por eixo)
PASSO_SILL_M = 100.0    # espaçamento da grade de cada sill (m)
MAX_NOS_SILL = 110
OFFSET_TERRENO_M = 2.0  # terreno "flutua" 2 m sobre o topo das formações expostas (evita z-fighting)
OFFSET_SILL_M = 4.0     # sill sobe 4 m sobre o terreno (aparece como corpo, não como manchinha)


def b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def rampa_hipsometrica(z):
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    t = np.clip((z - zmin) / (zmax - zmin), 0, 1)
    paleta = np.array([_hex_rgb(c) for c in CORES_HIPSOMETRICAS], dtype=float)
    n = len(paleta) - 1
    pos = t * n
    idx = np.clip(pos.astype(int), 0, n - 1)
    return np.clip(paleta[idx] + (paleta[idx + 1] - paleta[idx]) * (pos - idx)[..., None], 0, 255).astype(np.uint8)


def cores_geologia(grid_e, grid_n, litologia):
    pts = gpd.GeoDataFrame({"idx": np.arange(grid_e.size)},
                           geometry=gpd.points_from_xy(grid_e.ravel(), grid_n.ravel()), crs="EPSG:31982")
    j = gpd.sjoin(pts, litologia[["SIGLA_UNID", "geometry"]], how="left", predicate="within")
    j = j.drop_duplicates(subset="idx").set_index("idx").reindex(range(grid_e.size))
    cor = j["SIGLA_UNID"].map(COR_POR_SIGLA).fillna(COR_LITOLOGIA_PADRAO)
    rgb = np.array([_hex_rgb(c) for c in cor], dtype=np.uint8)
    return rgb.reshape(grid_e.shape + (3,))


def montar_dados():
    poligono, (e0, n0, e1, n1) = carregar_area_pmp()
    xt, yt, zt = carregar_vertices_curvas_pmp()
    interp_terreno = construir_interpolador(xt, yt, zt)
    litologia = carregar_litologia_pmp()

    xs = np.linspace(e0, e1, N)
    ys = np.linspace(n0, n1, N)
    grid_e, grid_n = np.meshgrid(xs, ys)          # linha 0 = sul
    # raio grande: preenche as bordas sem curva (vizinho mais próximo) pro bloco ficar fechado
    terreno = avaliar_interpolador(interp_terreno, grid_e.ravel(), grid_n.ravel(), raio_mascara_km=60).reshape(grid_e.shape)
    assert not np.isnan(terreno).any()
    print(f"[info] terreno {np.nanmin(terreno):.0f}-{np.nanmax(terreno):.0f} m, grade {N}x{N}")

    planos = calcular_planos_estilizados(litologia, interp_terreno)
    contatos = calcular_contatos_estilizados(planos, terreno, grid_e, grid_n)
    a, b, _ = planos[UNIDADES_ESTILIZADO[0]]
    print(f"[info] mergulho comum das formações: {np.degrees(np.arctan(np.hypot(a, b))):.2f}°")
    z_piso = float(np.floor((min(np.nanmin(c) for c in contatos.values()) - 200) / 50) * 50)

    # --- cores do terreno nos 3 modos
    print("[info] cores: hipsometria / geologia / satélite")
    sat, sat_bounds = obter_satelite_utm()
    cores = {
        "hips": rampa_hipsometrica(terreno),
        "geo": cores_geologia(grid_e, grid_n, litologia),
        "sat": amostrar_satelite_rgb(sat, sat_bounds, grid_e.ravel(), grid_n.ravel()).reshape(grid_e.shape + (3,)),
    }

    camadas = [dict(nome=NOMES_ESTILIZADO[u], cor=CORES_ESTILIZADO[u], top=b64(contatos[u], np.float32))
               for u in UNIDADES_ESTILIZADO]

    # --- sills: grade própria em cima do bbox do polígono; topo = terreno, base = terreno - espessura
    terr_bilin = RegularGridInterpolator((ys, xs), terreno, bounds_error=False, fill_value=None)
    sills = []
    for nome, geom in carregar_sills_individualizados():
        g0, h0, g1, h1 = geom.bounds
        nx = int(min(MAX_NOS_SILL, max(8, np.ceil((g1 - g0) / PASSO_SILL_M))))
        ny = int(min(MAX_NOS_SILL, max(8, np.ceil((h1 - h0) / PASSO_SILL_M))))
        sx, sy = np.linspace(g0, g1, nx), np.linspace(h0, h1, ny)
        se, sn = np.meshgrid(sx, sy)
        dentro = pontos_dentro_poligono(se, sn, geom.buffer(PASSO_SILL_M / 2))
        zt_ = terr_bilin(np.column_stack([sn.ravel(), se.ravel()])).reshape(se.shape)
        top = np.where(dentro, zt_ + OFFSET_SILL_M, np.nan)
        base = np.where(dentro, zt_ - ESPESSURA_SILL_M, np.nan)
        sills.append(dict(nome=f"Sill — {nome}", cor=CORES_ESTILIZADO["Gp_SerraGeral"], xs=[round(float(v), 1) for v in sx],
                          ys=[round(float(v), 1) for v in sy], top=b64(top, np.float32), base=b64(base, np.float32)))

    # --- furos em 3D: colunas ancoradas no terreno do modelo
    furos = []
    tab = preparar_furos()
    tab = tab[tab["tipo"] == "Furo"]
    idx_unid = {col: k for k, (col, _, _) in enumerate(ORDEM_PROFUNDIDADE_FURO)}
    for _, r in tab.iterrows():
        tops = []
        for col, k in idx_unid.items():
            v = r.get(f"Prof_topo_{col}")
            if pd.notna(v):
                tops.append([round(float(v), 2), k])
        tops.sort()
        prof = None if pd.isna(r["Profundidade"]) else round(float(r["Profundidade"]), 2)
        corpos = [[round(a_, 2), round(b_, 2)] for a_, b_ in _intervalos_corpo(r.get("Prof_corpos_intrusivos_SG"))]
        if prof is None and not tops and not corpos:
            continue
        z0 = float(terr_bilin([[r["N"], r["E"]]])[0])
        furos.append(dict(n=r["nome"], x=round(float(r["E"]), 1), y=round(float(r["N"]), 1), z0=round(z0, 1),
                          cota=None if pd.isna(r["Cota_boca"]) else round(float(r["Cota_boca"]), 1), prof=prof,
                          mun=None if pd.isna(r.get("Municipio")) else str(r["Municipio"]), fundo=r["unidade_fundo"],
                          tops=tops, corpos=corpos))
    print(f"[info] {len(furos)} furos em 3D, {len(sills)} sills, {len(camadas)} formações")

    return dict(
        N=N, xs=[round(float(v), 1) for v in xs], ys=[round(float(v), 1) for v in ys],
        zmin=float(z_piso), zmax=float(np.nanmax(terreno)) + 40, z_piso=z_piso,
        terreno=b64(terreno, np.float32),
        cores={k: b64(v, np.uint8) for k, v in cores.items()},
        camadas=camadas, sills=sills, furos=furos,
        unidades_furo=[dict(n=n_, c=c_) for _, n_, c_ in ORDEM_PROFUNDIDADE_FURO],
        cor_sill=CORES_ESTILIZADO["Gp_SerraGeral"], esp_sill=ESPESSURA_SILL_M,
        offsets=dict(terreno=OFFSET_TERRENO_M),
    )


TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" href="assets/favicon.png">
<link rel="shortcut icon" href="assets/favicon.ico">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
<title>Modelo 3D — PMP</title>
<script src="@@PLOTLY@@"></script>
<style>
  * { box-sizing: border-box; }
  :root { --bg: @@NAVY@@; --painel: #262B3D; --texto: @@CINZA@@; --borda: #3a3f52; }
  body.tema-claro { --bg: @@CINZA@@; --painel: #FFFFFF; --texto: @@NAVY@@; --borda: #D8D8E2; }
  html, body { height: 100%; }
  body { margin: 0; background: var(--bg); color: var(--texto); font-family: @@FONTE@@; display: flex; flex-direction: column; transition: background .2s, color .2s; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 10px 24px; border-bottom: 1px solid @@ROXO@@; gap: 16px; flex-shrink: 0; }
  header h1 { font-size: 19px; margin: 0; }
  header h1 b { color: @@ROXO@@; }
  header img { width: 46px; height: 46px; border-radius: 50%; border: 2px solid @@ROXO@@; box-shadow: 0 0 10px rgba(123,47,255,.6); }
  #barra { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; padding: 8px 24px; background: var(--painel); border-bottom: 1px solid var(--borda); font-size: 12px; flex-shrink: 0; }
  .grupo { display: flex; align-items: center; gap: 6px; }
  .grupo > span.rot { opacity: .65; text-transform: uppercase; letter-spacing: .04em; font-size: 10.5px; }
  button, select { font-family: @@FONTE@@; font-size: 12px; }
  .btn { background: transparent; color: var(--texto); border: 1px solid @@ROXO@@; border-radius: 6px; padding: 5px 10px; cursor: pointer; opacity: .65; transition: opacity .12s, background .12s; }
  .btn:hover { opacity: 1; }
  .btn.ativo { opacity: 1; background: @@ROXO@@; color: #fff; }
  select { background: var(--bg); color: var(--texto); border: 1px solid @@ROXO@@; border-radius: 6px; padding: 4px 6px; }
  input[type=range] { accent-color: @@ROXO@@; width: 130px; }
  #corte-pos { width: 190px; }
  .desab { opacity: .35; pointer-events: none; }
  #val-corte, #val-opac, #val-exag { min-width: 38px; font-variant-numeric: tabular-nums; }
  #cubo { flex: 1; min-height: 0; }
  footer { text-align: center; padding: 4px; opacity: .5; font-size: 11px; flex-shrink: 0; }
  #nota-sat { display: none; opacity: .7; font-size: 10.5px; }
</style>
</head>
<body>
<header>
  <h1><b>Modelo 3D interativo</b> — PMP · Criciúma</h1>
  @@LOGO@@
</header>
<div id="barra">
  <div class="grupo"><span class="rot">Cor</span>
    <button class="btn ativo" data-cor="hips">Hipsometria</button>
    <button class="btn" data-cor="geo">Geologia</button>
    <button class="btn" data-cor="sat">Satélite</button>
    <span id="nota-sat">© Esri World Imagery</span></div>
  <div class="grupo"><span class="rot">Camadas</span>
    <button class="btn" id="b-solido">Sólido: OFF</button>
    <button class="btn ativo" id="b-topo">Topografia: ON</button>
    <button class="btn ativo" id="b-furos">Furos: ON</button></div>
  <div class="grupo"><span class="rot">Corte</span>
    <button class="btn" id="b-corte">Corte: OFF</button>
    <select id="corte-dir">
      <option value="0">Leste → Oeste</option><option value="1">Oeste → Leste</option>
      <option value="2">Norte → Sul</option><option value="3">Sul → Norte</option>
    </select>
    <input type="range" id="corte-pos" min="0" max="95" value="35" step="0.5"><span id="val-corte">35%</span>
    <button class="btn" id="b-anim">Animar: OFF</button></div>
  <div class="grupo"><span class="rot">Opacidade</span><input type="range" id="opac" min="0.3" max="1" step="0.05" value="1"><span id="val-opac">100%</span></div>
  <div class="grupo"><span class="rot">Exagero vertical</span><input type="range" id="exag" min="1" max="8" step="0.5" value="@@EXAG@@"><span id="val-exag">@@EXAG@@×</span></div>
  <div class="grupo"><span class="rot">Tema</span>
    <button class="btn ativo" data-tema="escuro">Escuro</button><button class="btn" data-tema="claro">Claro</button></div>
</div>
<div id="cubo"></div>
<footer>GS Tech · PMP</footer>
<script>
(function() {
    'use strict';
    var D = @@DATA@@;
    function dec(s, Tipo) { var b = atob(s), u = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Tipo(u.buffer); }
    var N = D.N, XS = Float64Array.from(D.xs), YS = Float64Array.from(D.ys);
    var TERR = dec(D.terreno, Float32Array);
    var CORES = { hips: dec(D.cores.hips, Uint8Array), geo: dec(D.cores.geo, Uint8Array), sat: dec(D.cores.sat, Uint8Array) };
    var CAMADAS = D.camadas.map(function(c) { return { nome: c.nome, cor: c.cor, top: dec(c.top, Float32Array) }; });
    var SILLS = D.sills.map(function(s) { return { nome: s.nome, cor: D.cor_sill, xs: Float64Array.from(s.xs), ys: Float64Array.from(s.ys),
                                                 top: dec(s.top, Float32Array), base: dec(s.base, Float32Array) }; });
    var FUROS = D.furos, UNID = D.unidades_furo;
    var E0 = XS[0], E1 = XS[N - 1], N0 = YS[0], N1 = YS[N - 1];
    var PISO = new Float32Array(N * N).fill(D.z_piso);

    var DIRS = [{ eixo: 'x', inv: false }, { eixo: 'x', inv: true }, { eixo: 'y', inv: false }, { eixo: 'y', inv: true }];
    var st = { cor: 'hips', solido: false, topo: true, furos: true, corte: false, dir: 0, pos: 35, opac: 1, exag: @@EXAG@@, tema: 'escuro', anim: false };

    // ------------------------------------------------ recorte de grade (plano de corte)
    function lerp(a, b, t) { return t === 0 ? a : (t === 1 ? b : a + (b - a) * t); }
    function nosEixo(vec, inv, v) {
        var n = vec.length, out = [], i, k;
        if (!inv) {
            for (i = 0; i < n; i++) { if (vec[i] <= v) out.push({ a: i, b: i, t: 0, c: vec[i] }); else break; }
            if (out.length && out.length < n && v > vec[out.length - 1]) {
                k = out.length - 1; out.push({ a: k, b: k + 1, t: (v - vec[k]) / (vec[k + 1] - vec[k]), c: v });
            }
        } else {
            var ini = n;
            for (i = n - 1; i >= 0; i--) { if (vec[i] >= v) ini = i; else break; }
            if (ini < n) {
                if (ini > 0 && v < vec[ini]) { k = ini - 1; out.push({ a: k, b: k + 1, t: (v - vec[k]) / (vec[k + 1] - vec[k]), c: v }); }
                for (i = ini; i < n; i++) out.push({ a: i, b: i, t: 0, c: vec[i] });
            }
        }
        return out;
    }
    function identidade(vec) { var o = []; for (var i = 0; i < vec.length; i++) o.push({ a: i, b: i, t: 0, c: vec[i] }); return o; }
    function subgrade(xs, ys, spec) {
        var cn = (spec && spec.eixo === 'x') ? nosEixo(xs, spec.inv, spec.v) : identidade(xs);
        var rn = (spec && spec.eixo === 'y') ? nosEixo(ys, spec.inv, spec.v) : identidade(ys);
        return { cn: cn, rn: rn, nxO: xs.length, X: cn.map(function(o) { return o.c; }), Y: rn.map(function(o) { return o.c; }) };
    }
    function amostrar(arr, sg, ch) {
        var nx = sg.cn.length, ny = sg.rn.length, out = new Float32Array(nx * ny * ch), r, c, q, rn, cn, o = 0, w = sg.nxO;
        for (r = 0; r < ny; r++) { rn = sg.rn[r];
            for (c = 0; c < nx; c++) { cn = sg.cn[c];
                for (q = 0; q < ch; q++) {
                    var v00 = arr[(rn.a * w + cn.a) * ch + q], v01 = arr[(rn.a * w + cn.b) * ch + q];
                    var v10 = arr[(rn.b * w + cn.a) * ch + q], v11 = arr[(rn.b * w + cn.b) * ch + q];
                    out[o++] = lerp(lerp(v00, v01, cn.t), lerp(v10, v11, cn.t), rn.t);
                }
            }
        }
        return out;
    }

    // ------------------------------------------------ malha (topo + paredes opcionais)
    function malha(X, Y, top, base, o) {
        var nx = X.length, ny = Y.length, vx = [], vy = [], vz = [], I = [], J = [], K = [], r, c, k;
        var dz = o.dz || 0;
        for (r = 0; r < ny; r++) for (c = 0; c < nx; c++) { vx.push(X[c]); vy.push(Y[r]); vz.push(top[r * nx + c] + dz); }
        var mapaBase = {};
        function idxBase(k) {
            var v = mapaBase[k];
            if (v === undefined) { v = vx.length; vx.push(vx[k]); vy.push(vy[k]); vz.push(base[k]); mapaBase[k] = v; }
            return v;
        }
        var w = nx - 1, inc = new Uint8Array(Math.max(0, (ny - 1) * w));
        for (r = 0; r < ny - 1; r++) for (c = 0; c < w; c++) {
            var a = r * nx + c, b = a + 1, d = a + nx, e = d + 1;
            var ok = isFinite(top[a]) && isFinite(top[b]) && isFinite(top[d]) && isFinite(top[e]);
            if (ok && base) {
                ok = isFinite(base[a]) && isFinite(base[b]) && isFinite(base[d]) && isFinite(base[e]) &&
                     Math.max(top[a] - base[a], top[b] - base[b], top[d] - base[d], top[e] - base[e]) > (o.minEsp || 0);
            }
            inc[r * w + c] = ok ? 1 : 0;
            if (ok) { I.push(a, b); J.push(b, e); K.push(d, d); }
        }
        if (base && o.paredes) {
            function quad(p, q) { var bp = idxBase(p), bq = idxBase(q); I.push(p, p); J.push(q, bq); K.push(bq, bp); }
            for (r = 0; r < ny - 1; r++) for (c = 0; c < w; c++) {
                if (!inc[r * w + c]) continue;
                var a2 = r * nx + c, b2 = a2 + 1, d2 = a2 + nx, e2 = d2 + 1;
                if (r === 0 || !inc[(r - 1) * w + c]) quad(a2, b2);
                if (r === ny - 2 || !inc[(r + 1) * w + c]) quad(d2, e2);
                if (c === 0 || !inc[r * w + c - 1]) quad(a2, d2);
                if (c === w - 1 || !inc[r * w + c + 1]) quad(b2, e2);
            }
        }
        return { x: Float32Array.from(vx), y: Float32Array.from(vy), z: Float32Array.from(vz),
                 i: Int32Array.from(I), j: Int32Array.from(J), k: Int32Array.from(K) };
    }

    // ------------------------------------------------ figura
    function especCorte() {
        if (!st.corte) return null;
        var dd = DIRS[st.dir], f = st.pos / 100, e = dd.eixo === 'x' ? (E1 - E0) : (N1 - N0);
        var v = dd.eixo === 'x' ? (dd.inv ? E0 + f * e : E1 - f * e) : (dd.inv ? N0 + f * e : N1 - f * e);
        return { eixo: dd.eixo, inv: dd.inv, v: v };
    }
    var TEMAS = {
        escuro: { paper: '@@NAVY@@', grid: '#3a3f52', texto: '@@CINZA@@' },
        claro:  { paper: '@@CINZA@@', grid: '#C9C9D6', texto: '@@NAVY@@' },
    };
    var gd = document.getElementById('cubo');
    var IDX = { terreno: 0, camadas: 1, sills: 1 + CAMADAS.length, furos: 1 + CAMADAS.length + SILLS.length };
    var N_FUROS_TR = UNID.length + 2;   // 1 por unidade + "sem topos" + "corpo intrusivo"

    function traceMalha(nome, cor, extra) {
        var t = { type: 'mesh3d', x: [], y: [], z: [], i: [], j: [], k: [], color: cor, name: nome, showlegend: true,
                  hoverinfo: 'name', flatshading: false, opacity: 1,
                  lighting: { ambient: 0.62, diffuse: 0.85, specular: 0.08, roughness: 0.9, fresnel: 0.1 } };
        for (var q in (extra || {})) t[q] = extra[q];
        return t;
    }
    function tracesIniciais() {
        var tr = [traceMalha('Topografia', '#C6924A', { showlegend: false })];
        CAMADAS.forEach(function(c) { tr.push(traceMalha(c.nome, c.cor)); });
        SILLS.forEach(function(s) { tr.push(traceMalha(s.nome, s.cor)); });
        UNID.forEach(function(u, k) {
            tr.push({ type: 'scatter3d', mode: 'lines', x: [], y: [], z: [], line: { color: u.c, width: 7 }, name: 'Furo · ' + u.n,
                      legendgroup: 'furos', showlegend: false, hoverinfo: 'skip' });
        });
        tr.push({ type: 'scatter3d', mode: 'lines', x: [], y: [], z: [], line: { color: '#9AA0B4', width: 7 }, name: 'Furo · sem topos medidos',
                  legendgroup: 'furos', showlegend: false, hoverinfo: 'skip' });
        tr.push({ type: 'scatter3d', mode: 'lines', x: [], y: [], z: [], line: { color: '#E63946', width: 11 }, name: 'Corpo intrusivo (furo)',
                  legendgroup: 'furos', showlegend: false, hoverinfo: 'skip' });
        tr.push({ type: 'scatter3d', mode: 'markers', x: [], y: [], z: [], text: [], hoverinfo: 'text', name: 'Furos de sondagem',
                  legendgroup: 'furos', showlegend: true, marker: { size: 3.5, color: '#FFFFFF', line: { color: '@@NAVY@@', width: 1 } } });
        return tr;
    }
    function fmt0(v) { return Math.round(v).toString(); }

    function mantido(x, y, spec) {
        if (!spec) return true;
        var v = spec.eixo === 'x' ? x : y;
        return spec.inv ? v >= spec.v : v <= spec.v;
    }
    function dadosFuros(spec) {
        var seg = []; for (var q = 0; q < N_FUROS_TR; q++) seg.push({ x: [], y: [], z: [] });
        var mx = [], my = [], mz = [], mt = [];
        function push(t, w, a, b) { t.x.push(w.x, w.x, null); t.y.push(w.y, w.y, null); t.z.push(w.z0 - a, w.z0 - b, null); }
        FUROS.forEach(function(w) {
            if (!mantido(w.x, w.y, spec)) return;
            for (var t = 0; t < w.tops.length; t++) {
                var ini = w.tops[t][0], fim = (t + 1 < w.tops.length) ? w.tops[t + 1][0] : ((w.prof !== null && w.prof > ini) ? w.prof : null);
                if (fim !== null && fim > ini) push(seg[w.tops[t][1]], w, ini, fim);
            }
            if (!w.tops.length && w.prof !== null) push(seg[UNID.length], w, 0, w.prof);
            w.corpos.forEach(function(cp) { push(seg[UNID.length + 1], w, cp[0], cp[1]); });
            mx.push(w.x); my.push(w.y); mz.push(w.z0);
            mt.push('<b>' + w.n + '</b>' + (w.mun ? '<br>' + w.mun : '') + (w.prof !== null ? '<br>prof. ' + fmt0(w.prof) + ' m' : '')
                    + (w.cota !== null ? '<br>cota da boca ' + fmt0(w.cota) + ' m' : '') + '<br>unidade mais profunda: ' + w.fundo);
        });
        return { seg: seg, m: { x: mx, y: my, z: mz, t: mt } };
    }

    var pend = false;
    function redesenhar() { if (pend) return; pend = true; setTimeout(function() { pend = false; desenhar(); }, 16); }   // setTimeout (não rAF): rAF pausa com a aba em segundo plano

    function desenhar() {
        var spec = especCorte();
        var sg = subgrade(XS, YS, spec);
        var topA = amostrar(TERR, sg, 1);
        var cor = amostrar(CORES[st.cor], sg, 3), vc = new Array(sg.cn.length * sg.rn.length);
        for (var q = 0; q < vc.length; q++) vc[q] = [Math.round(cor[q * 3]), Math.round(cor[q * 3 + 1]), Math.round(cor[q * 3 + 2])];
        var mt = malha(sg.X, sg.Y, topA, null, { dz: D.offsets.terreno });

        // camadas: base da camada i = topo da i+1 (a última vai até o piso)
        var tops = CAMADAS.map(function(c) { return amostrar(c.top, sg, 1); });
        var piso = amostrar(PISO, sg, 1);
        var ms = [], idx = [];
        CAMADAS.forEach(function(c, i) {
            var base = (i + 1 < tops.length) ? tops[i + 1] : piso;
            ms.push(st.solido ? malha(sg.X, sg.Y, tops[i], base, { paredes: true, minEsp: 0.3 }) : malha(sg.X, sg.Y, tops[i], null, {}));
            idx.push(IDX.camadas + i);
        });
        SILLS.forEach(function(s, i) {
            var sg2 = subgrade(s.xs, s.ys, spec);
            var t2 = amostrar(s.top, sg2, 1), b2 = amostrar(s.base, sg2, 1);
            ms.push(st.solido ? malha(sg2.X, sg2.Y, t2, b2, { paredes: true, minEsp: 0.3 }) : malha(sg2.X, sg2.Y, t2, null, {}));
            idx.push(IDX.sills + i);
        });
        var upd = { x: [], y: [], z: [], i: [], j: [], k: [] };
        ms.forEach(function(m) { ['x', 'y', 'z', 'i', 'j', 'k'].forEach(function(q) { upd[q].push(m[q]); }); });
        Plotly.restyle(gd, { x: [mt.x], y: [mt.y], z: [mt.z], i: [mt.i], j: [mt.j], k: [mt.k], vertexcolor: [vc] }, [IDX.terreno]);
        Plotly.restyle(gd, upd, idx);

        var fu = dadosFuros(spec), fx = [], fy = [], fz = [], fidx = [];
        for (var u = 0; u < N_FUROS_TR; u++) { fx.push(fu.seg[u].x); fy.push(fu.seg[u].y); fz.push(fu.seg[u].z); fidx.push(IDX.furos + u); }
        Plotly.restyle(gd, { x: fx, y: fy, z: fz }, fidx);
        Plotly.restyle(gd, { x: [fu.m.x], y: [fu.m.y], z: [fu.m.z], text: [fu.m.t] }, [IDX.furos + N_FUROS_TR]);
    }

    function visibilidades() {
        Plotly.restyle(gd, { visible: st.topo ? true : 'legendonly' }, [IDX.terreno]);
        var fi = []; for (var u = 0; u <= N_FUROS_TR; u++) fi.push(IDX.furos + u);
        Plotly.restyle(gd, { visible: st.furos ? true : 'legendonly' }, fi);
    }
    function opacidade() {
        var idx = [IDX.terreno]; CAMADAS.forEach(function(c, i) { idx.push(IDX.camadas + i); }); SILLS.forEach(function(s, i) { idx.push(IDX.sills + i); });
        Plotly.restyle(gd, { opacity: st.opac }, idx);
    }
    function layoutTema() {
        var t = TEMAS[st.tema];
        var ax = function(tit, rng) { return { title: tit, range: rng, color: t.texto, gridcolor: t.grid, zerolinecolor: t.grid, showbackground: false }; };
        var ext = E1 - E0, R = (D.zmax - D.zmin) / ext * st.exag;
        return {
            paper_bgcolor: t.paper, font: { family: '@@FONTE@@', color: t.texto, size: 11 },
            'scene.xaxis': ax('E (UTM 22S)', [E0, E1]), 'scene.yaxis': ax('N (UTM 22S)', [N0, N1]), 'scene.zaxis': ax('Altitude (m)', [D.zmin, D.zmax]),
            'scene.aspectratio': { x: 1, y: (N1 - N0) / ext, z: Math.min(R, 1.2) },
            'legend.bgcolor': 'rgba(0,0,0,0)', 'legend.font.color': t.texto,
        };
    }

    Plotly.newPlot(gd, tracesIniciais(), {
        margin: { l: 0, r: 0, t: 6, b: 0 }, paper_bgcolor: '@@NAVY@@',
        scene: { aspectmode: 'manual', camera: { eye: { x: 1.25, y: -1.45, z: 0.85 } }, xaxis: {}, yaxis: {}, zaxis: {} },
        legend: { x: 0.005, y: 0.99, itemsizing: 'constant', font: { size: 11 } },
    }, { responsive: true, displaylogo: false }).then(function() {
        Plotly.relayout(gd, layoutTema());
        desenhar(); visibilidades(); opacidade();
    });

    // ------------------------------------------------ controles
    function marcar(sel, ativo) { document.querySelectorAll(sel).forEach(function(b) { b.classList.toggle('ativo', b === ativo); }); }
    document.querySelectorAll('[data-cor]').forEach(function(b) {
        b.addEventListener('click', function() {
            st.cor = b.getAttribute('data-cor'); marcar('[data-cor]', b);
            document.getElementById('nota-sat').style.display = st.cor === 'sat' ? 'inline' : 'none';
            redesenhar();
        });
    });
    function alternar(id, chave, rotulo, apos) {
        var b = document.getElementById(id);
        b.addEventListener('click', function() {
            st[chave] = !st[chave]; b.classList.toggle('ativo', st[chave]);
            b.textContent = rotulo + (st[chave] ? ': ON' : ': OFF');
            apos();
        });
    }
    alternar('b-solido', 'solido', 'Sólido', redesenhar);
    alternar('b-topo', 'topo', 'Topografia', visibilidades);
    alternar('b-furos', 'furos', 'Furos', visibilidades);

    var bCorte = document.getElementById('b-corte');
    function atualizarCorteUI() {
        ['corte-dir', 'corte-pos', 'b-anim'].forEach(function(id) { document.getElementById(id).classList.toggle('desab', !st.corte); });
    }
    atualizarCorteUI();
    bCorte.addEventListener('click', function() {
        st.corte = !st.corte; bCorte.classList.toggle('ativo', st.corte); bCorte.textContent = 'Corte: ' + (st.corte ? 'ON' : 'OFF');
        if (st.corte && !st.solido) { st.solido = true; var bs = document.getElementById('b-solido'); bs.classList.add('ativo'); bs.textContent = 'Sólido: ON'; }  // o corte precisa do sólido pra mostrar a face
        if (!st.corte) pararAnim();
        atualizarCorteUI(); redesenhar();
    });
    document.getElementById('corte-dir').addEventListener('change', function(ev) { st.dir = +ev.target.value; redesenhar(); });
    var sliderCorte = document.getElementById('corte-pos');
    sliderCorte.addEventListener('input', function() { st.pos = +sliderCorte.value; document.getElementById('val-corte').textContent = Math.round(st.pos) + '%'; redesenhar(); });

    var animTimer = null, animSentido = 1;
    function pararAnim() { st.anim = false; clearInterval(animTimer); animTimer = null; var b = document.getElementById('b-anim'); b.classList.remove('ativo'); b.textContent = 'Animar: OFF'; }
    document.getElementById('b-anim').addEventListener('click', function() {
        if (st.anim) { pararAnim(); return; }
        st.anim = true; this.classList.add('ativo'); this.textContent = 'Animar: ON';
        animTimer = setInterval(function() {
            st.pos += animSentido * 1.5;
            if (st.pos >= 95) { st.pos = 95; animSentido = -1; } else if (st.pos <= 0) { st.pos = 0; animSentido = 1; }
            sliderCorte.value = st.pos; document.getElementById('val-corte').textContent = Math.round(st.pos) + '%'; redesenhar();
        }, 90);
    });

    document.getElementById('opac').addEventListener('input', function(ev) {
        st.opac = +ev.target.value; document.getElementById('val-opac').textContent = Math.round(st.opac * 100) + '%'; opacidade();
    });
    document.getElementById('exag').addEventListener('input', function(ev) {
        st.exag = +ev.target.value; document.getElementById('val-exag').textContent = st.exag + '×'; Plotly.relayout(gd, layoutTema());
    });
    document.querySelectorAll('[data-tema]').forEach(function(b) {
        b.addEventListener('click', function() {
            st.tema = b.getAttribute('data-tema'); marcar('[data-tema]', b);
            document.body.classList.toggle('tema-claro', st.tema === 'claro'); Plotly.relayout(gd, layoutTema());
        });
    });
    window.addEventListener('resize', function() { Plotly.Plots.resize(gd); });
})();
</script>
</body>
</html>
"""


def main():
    dados = montar_dados()
    ext = dados["xs"][-1] - dados["xs"][0]
    exag = 2.5
    logo = logo_base64()
    html = TEMPLATE
    for token, valor in {
        "@@PLOTLY@@": PLOTLY_CDN, "@@FONTE@@": MARCA_FONTE, "@@NAVY@@": MARCA_NAVY, "@@ROXO@@": MARCA_ROXO,
        "@@CINZA@@": MARCA_CINZA_CLARO, "@@EXAG@@": str(exag),
        "@@LOGO@@": f'<img src="data:image/jpeg;base64,{logo}" alt="GS Tech">' if logo else "",
        "@@DATA@@": json.dumps(dados, ensure_ascii=False, separators=(",", ":")),
    }.items():
        html = html.replace(token, valor)
    assert "@@" not in html, "token sem substituir"
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"-> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
