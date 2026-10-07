"""Seção 2D interativa do PMP -- mesmas funcionalidades do app de seção do
Taió (gerar_secao_interativa.py), com os dados do PMP:

  - mapa em planta clicável + perfil sincronizado, com linha de corte
    rotacionável (4 ângulos), posição por slider ou clique no mapa;
  - modos de mapa: Hipsometria / Geologia real (CPRM + sills) / Satélite;
  - FUROS: pontos no mapa e colunas projetadas no perfil (furos a até 1,5 km
    da linha de corte, coloridos pela unidade atravessada, corpo intrusivo em
    vermelho) -- dá pra conferir o modelo contra o dado de sondagem;
  - gráfico de espessura das formações na linha atual, tema escuro/claro.

Mesmos planos por formação do cubo 3D (_comum_pmp.py::calcular_planos_estilizados).
Fontes: ../2_Banco_de_Dados (area2/curvas3/Litologia_PMP2.shp + banco de poços)
via _comum_pmp.py; satélite Esri World Imagery (cache). Só LÊ essas fontes.

Uso:
    python gerar_secao_pmp.py
Gera:
    secao_pmp.html
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import shapely
from PIL import Image
from plotly.subplots import make_subplots

from _comum_pmp import (
    MARCA_ROXO, MARCA_NAVY, MARCA_CINZA_CLARO, MARCA_FONTE,
    UNIDADES_ESTILIZADO, NOMES_ESTILIZADO, CORES_ESTILIZADO, ESPESSURA_SILL_M,
    COR_POR_SIGLA, SIGLAS_SILL_INDIVIDUALIZADO, ORDEM_PROFUNDIDADE_FURO,
    logo_base64, carregar_area_pmp, carregar_vertices_curvas_pmp, carregar_litologia_pmp,
    carregar_sills_individualizados, construir_interpolador, avaliar_interpolador, avaliar_plano,
    calcular_planos_estilizados, poligono_para_scatter_xy, pontos_dentro_poligono,
    tema_claro, tema_escuro, adicionar_escala_e_norte, quantizar,
    obter_satelite_utm, preparar_furos, _intervalos_corpo, carregar_linhas_secao,
)

BASE = Path(__file__).resolve().parent
OUT_HTML = BASE / "secao_pmp.html"

ANGULOS = [
    ("Leste-Oeste", 0.0), ("Diagonal (SO-NE)", 45.0),
    ("Sul-Norte", 90.0), ("Diagonal (SE-NO)", 135.0),
]
PASSO_POSICAO_M = 1_500.0  # area nova e bem menor (~26x47km) -- passo mais fino que a versao regional
N_AMOSTRAS_LINHA = 150
RESOLUCAO_MAPA = 140
RAIO_MASCARA_KM = 3.0
N_AMOSTRAS_SECAO_FIXA = 260   # pontos ao longo de cada linha A-D (polilinha)
BUFFER_FUROS_M = 1_500.0   # furos até essa distância da linha de corte aparecem no perfil
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{__import__('plotly').offline.get_plotlyjs_version()}.min.js"
NOMES_CLASSES_FURO = [n for _, n, _ in ORDEM_PROFUNDIDADE_FURO] + ["sem topos medidos", "corpo intrusivo"]
CORES_CLASSES_FURO = [c for _, _, c in ORDEM_PROFUNDIDADE_FURO] + ["#9AA0B4", "#E63946"]


def cobertura_bbox(vx, vy, cx, cy, e_min, e_max, n_min, n_max):
    cantos = [(e_min, n_min), (e_min, n_max), (e_max, n_min), (e_max, n_max)]
    projs = [(x - cx) * vx + (y - cy) * vy for x, y in cantos]
    return min(projs), max(projs)


def faixas_contiguas(mask):
    """Lista de (i0,i1) pra cada trecho contíguo de True em mask -- um sill
    pode cruzar a linha de corte em vários pontos separados."""
    faixas = []
    inicio = None
    for i, v in enumerate(mask):
        if v and inicio is None:
            inicio = i
        elif not v and inicio is not None:
            faixas.append((inicio, i))
            inicio = None
    if inicio is not None:
        faixas.append((inicio, len(mask)))
    return faixas


def banda_mascarada(dists, topo, base, mask):
    faixas = faixas_contiguas(mask)
    if not faixas:
        return np.array([np.nan]), np.array([np.nan])
    xs, ys = [], []
    for k, (i0, i1) in enumerate(faixas):
        if k > 0:
            xs.append(np.nan)
            ys.append(np.nan)
        d, t, b = dists[i0:i1], topo[i0:i1], base[i0:i1]
        xs.extend(np.concatenate([d, d[::-1]]))
        ys.extend(np.concatenate([t, b[::-1]]))
    return np.array(xs), np.array(ys)


def satelite_jpeg_uri(largura=1100):
    """Satélite Esri do retângulo como data-URI JPEG (fundo do mapa em modo Satélite)."""
    raster, _ = obter_satelite_utm()
    img = Image.fromarray(np.moveaxis(raster, 0, -1))
    h = int(round(img.height * largura / img.width))
    img = img.resize((largura, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def preparar_furos_secao(interp_terreno):
    """Furos (só sondagens) prontos pro mapa/perfil: posição, cota do terreno do
    modelo na boca (âncora, pra coluna encostar na linha do terreno), segmentos
    por unidade (ini, fim, classe) e intervalos de corpo intrusivo."""
    tab = preparar_furos()
    tab = tab[tab["tipo"] == "Furo"].reset_index(drop=True)
    x, y = tab["E"].to_numpy(float), tab["N"].to_numpy(float)
    z0 = avaliar_interpolador(interp_terreno, x, y, raio_mascara_km=60)
    idx_unid = {col: k for k, (col, _, _) in enumerate(ORDEM_PROFUNDIDADE_FURO)}
    F = dict(x=[], y=[], z0=[], nome=[], segs=[], corpos=[], hover=[], cor=[])
    for i, r in tab.iterrows():
        if np.isnan(z0[i]):
            continue
        tops = sorted((float(r[f"Prof_topo_{c}"]), k) for c, k in idx_unid.items() if not np.isnan(r.get(f"Prof_topo_{c}", np.nan)))
        prof = None if np.isnan(r["Profundidade"]) else float(r["Profundidade"])
        segs = []
        for j, (ini, k) in enumerate(tops):
            fim = tops[j + 1][0] if j + 1 < len(tops) else (prof if prof is not None and prof > ini else None)
            if fim is not None and fim > ini:
                segs.append((ini, fim, k))
        if not tops and prof is not None:
            segs.append((0.0, prof, len(ORDEM_PROFUNDIDADE_FURO)))   # "sem topos medidos"
        F["x"].append(float(x[i])); F["y"].append(float(y[i])); F["z0"].append(float(z0[i])); F["nome"].append(r["nome"])
        F["segs"].append(segs); F["corpos"].append(_intervalos_corpo(r.get("Prof_corpos_intrusivos_SG"))); F["cor"].append(r["cor"])
        mun = r.get("Municipio")
        F["hover"].append(f"<b>{r['nome']}</b>" + (f"<br>{mun}" if isinstance(mun, str) else "")
                          + (f"<br>prof. {prof:.0f} m" if prof is not None else "") + f"<br>{r['unidade_fundo']}")
    for k in ("x", "y", "z0"):
        F[k] = np.array(F[k])
    return F


def furos_no_perfil(F, dx, dy, px, py, t, cx, cy, s0):
    """Colunas de furo (±BUFFER_FUROS_M da linha) projetadas no perfil: 1 conjunto
    de segmentos por classe (unidade / sem topos / corpo intrusivo) + a base
    preta (contorno) com todos juntos. Cada segmento: x=[d,d,None], y=[z0-ini, z0-fim, None]."""
    n_cls = len(NOMES_CLASSES_FURO)
    cls = [([], [], []) for _ in range(n_cls)]
    lab = ([], [], [])   # rótulo (código do furo) na base de cada coluna
    perp = (F["x"] - cx) * px + (F["y"] - cy) * py - t
    for i in np.where(np.abs(perp) <= BUFFER_FUROS_M)[0]:
        d = round(float(((F["x"][i] - cx - t * px) * dx + (F["y"][i] - cy - t * py) * dy - s0) / 1000), 3)
        z0, nome = F["z0"][i], F["nome"][i]
        for ini, fim, k in F["segs"][i]:
            cls[k][0].extend([d, d, None]); cls[k][1].extend([round(z0 - ini, 1), round(z0 - fim, 1), None])
            cls[k][2].extend([f"<b>{nome}</b><br>{NOMES_CLASSES_FURO[k]}: {ini:.0f}–{fim:.0f} m", None, None])
        for a, b in F["corpos"][i]:
            k = n_cls - 1
            cls[k][0].extend([d, d, None]); cls[k][1].extend([round(z0 - a, 1), round(z0 - b, 1), None])
            cls[k][2].extend([f"<b>{nome}</b><br>corpo intrusivo: {a:.1f}–{b:.1f} m ({b - a:.1f} m)", None, None])
        fundo = max([f for _, f, _ in F["segs"][i]] + [b for _, b in F["corpos"][i]], default=None)
        if fundo is not None:
            lab[0].append(d); lab[1].append(round(z0 - fundo, 1)); lab[2].append(nome)
    base = ([v for c in cls for v in c[0]], [v for c in cls for v in c[1]], [None] * sum(len(c[0]) for c in cls))
    return dict(base=base, cls=cls, lab=lab)


def furos_na_polilinha(F, linha):
    """Como furos_no_perfil, mas pra uma polilinha (seção A-D): a distância no
    perfil é o comprimento ao longo da linha até a projeção do furo, e o
    critério de inclusão é a distância perpendicular à polilinha."""
    n_cls = len(NOMES_CLASSES_FURO)
    cls = [([], [], []) for _ in range(n_cls)]
    lab = ([], [], [])   # rótulo (código do furo) na base de cada coluna
    for i in range(len(F["x"])):
        pt = shapely.Point(F["x"][i], F["y"][i])
        if linha.distance(pt) > BUFFER_FUROS_M:
            continue
        d = round(float(linha.project(pt)) / 1000, 3)
        z0, nome = F["z0"][i], F["nome"][i]
        for ini, fim, k in F["segs"][i]:
            cls[k][0].extend([d, d, None]); cls[k][1].extend([round(z0 - ini, 1), round(z0 - fim, 1), None])
            cls[k][2].extend([f"<b>{nome}</b><br>{NOMES_CLASSES_FURO[k]}: {ini:.0f}–{fim:.0f} m", None, None])
        for a, b in F["corpos"][i]:
            k = n_cls - 1
            cls[k][0].extend([d, d, None]); cls[k][1].extend([round(z0 - a, 1), round(z0 - b, 1), None])
            cls[k][2].extend([f"<b>{nome}</b><br>corpo intrusivo: {a:.1f}–{b:.1f} m ({b - a:.1f} m)", None, None])
        fundo = max([f for _, f, _ in F["segs"][i]] + [b for _, b in F["corpos"][i]], default=None)
        if fundo is not None:
            lab[0].append(d); lab[1].append(round(z0 - fundo, 1)); lab[2].append(nome)
    base = ([v for c in cls for v in c[0]], [v for c in cls for v in c[1]], [None] * sum(len(c[0]) for c in cls))
    return dict(base=base, cls=cls, lab=lab)


def secao_na_polilinha(linha, interp_terreno, planos, sills, furos):
    """Perfil ao longo de uma polilinha (seção A-D): mesmas camadas da seção
    rotacionável (terreno real, formações em cascata de erosão, sill onde a
    linha cruza o polígono mapeado, furos projetados)."""
    comp = float(linha.length)
    d_m = np.linspace(0.0, comp, N_AMOSTRAS_SECAO_FIXA)
    pts = shapely.line_interpolate_point(linha, d_m)
    xs, ys = shapely.get_x(pts), shapely.get_y(pts)
    dists = quantizar(d_m / 1000, 3)
    terreno = avaliar_interpolador(interp_terreno, xs, ys, raio_mascara_km=RAIO_MASCARA_KM)
    contatos, corte = {}, terreno.copy()
    for unidade in UNIDADES_ESTILIZADO:
        corte = np.minimum(corte, avaliar_plano(planos[unidade], xs, ys))
        contatos[unidade] = corte.copy()
    bandas = {}
    for i, unidade in enumerate(UNIDADES_ESTILIZADO):
        topo = contatos[unidade]
        base = contatos[UNIDADES_ESTILIZADO[i + 1]] if i + 1 < len(UNIDADES_ESTILIZADO) else topo - 300.0
        bandas[unidade] = (np.concatenate([dists, dists[::-1]]), np.concatenate([quantizar(topo, 1), quantizar(base, 1)[::-1]]))
    dentro = np.zeros_like(xs, dtype=bool)
    for _, geom_sill in sills:
        dentro |= pontos_dentro_poligono(xs, ys, geom_sill)
    sx, sy = banda_mascarada(dists, quantizar(terreno, 1), quantizar(terreno - ESPESSURA_SILL_M, 1), dentro)
    vx, vy = np.array(linha.coords)[:, 0], np.array(linha.coords)[:, 1]
    return dict(
        terreno=(dists, quantizar(terreno, 1)), linha_mapa=(quantizar(vx, 0), quantizar(vy, 0)),
        bandas=bandas, sill=(sx, sy), furos=furos_na_polilinha(furos, linha),
    ), comp / 1000


def main():
    poligono, bounds = carregar_area_pmp()
    e_min, n_min, e_max, n_max = bounds
    cx, cy = (e_min + e_max) / 2, (n_min + n_max) / 2

    xt, yt, zt = carregar_vertices_curvas_pmp()
    interp_terreno = construir_interpolador(xt, yt, zt)
    litologia = carregar_litologia_pmp()
    planos = calcular_planos_estilizados(litologia, interp_terreno)
    sills = carregar_sills_individualizados()
    furos = preparar_furos_secao(interp_terreno)
    sat_uri = satelite_jpeg_uri()
    print(f"[info] {len(furos['x'])} furos (buffer {BUFFER_FUROS_M:.0f} m no perfil)")

    # --- mapa em planta: hipsometria + geologia real ---
    mx, my = np.meshgrid(np.linspace(e_min, e_max, RESOLUCAO_MAPA), np.linspace(n_min, n_max, RESOLUCAO_MAPA))
    dentro_grade = pontos_dentro_poligono(mx, my, poligono)
    mz = avaliar_interpolador(interp_terreno, mx.ravel(), my.ravel(), raio_mascara_km=RAIO_MASCARA_KM).reshape(mx.shape)
    mz = quantizar(np.where(dentro_grade, mz, np.nan), 1)
    mx, my = quantizar(mx, 0), quantizar(my, 0)

    # --- angulos: amostras (s_vals) + posicoes (t_vals) cobrindo o bbox da area ---
    angulos_info = []
    for nome, theta_deg in ANGULOS:
        rad = np.radians(theta_deg)
        dx, dy = np.cos(rad), np.sin(rad)
        px, py = -np.sin(rad), np.cos(rad)
        s_min, s_max = cobertura_bbox(dx, dy, cx, cy, e_min, e_max, n_min, n_max)
        t_min, t_max = cobertura_bbox(px, py, cx, cy, e_min, e_max, n_min, n_max)
        n_pos = int(np.floor((t_max - t_min) / PASSO_POSICAO_M)) + 1
        t_vals = t_min + np.arange(n_pos) * PASSO_POSICAO_M
        if t_vals[-1] < t_max:
            t_vals = np.append(t_vals, t_max)
        angulos_info.append(dict(nome=nome, dx=dx, dy=dy, px=px, py=py,
                                  s_vals=np.linspace(s_min, s_max, N_AMOSTRAS_LINHA), t_vals=t_vals))

    # --- pre-computa uma secao (bandas de formacao + sill + furos) por
    # combinacao angulo x posicao ---
    todas_secoes = []
    for info in angulos_info:
        dists = quantizar((info["s_vals"] - info["s_vals"][0]) / 1000, 3)
        secoes_angulo = []
        for t in info["t_vals"]:
            xs = cx + t * info["px"] + info["s_vals"] * info["dx"]
            ys = cy + t * info["py"] + info["s_vals"] * info["dy"]
            terreno = avaliar_interpolador(interp_terreno, xs, ys, raio_mascara_km=RAIO_MASCARA_KM)

            contatos = {}
            corte_atual = terreno.copy()
            for unidade in UNIDADES_ESTILIZADO:
                plano = avaliar_plano(planos[unidade], xs, ys)
                corte_atual = np.minimum(corte_atual, plano)
                contatos[unidade] = corte_atual.copy()

            bandas = {}
            for i, unidade in enumerate(UNIDADES_ESTILIZADO):
                topo = contatos[unidade]
                base = contatos[UNIDADES_ESTILIZADO[i + 1]] if i + 1 < len(UNIDADES_ESTILIZADO) else topo - 300.0
                bandas[unidade] = (
                    np.concatenate([dists, dists[::-1]]),
                    np.concatenate([quantizar(topo, 1), quantizar(base, 1)[::-1]]),
                )

            # sill: poligono de afloramento = "aqui o sill está exposto na
            # superfície", ou seja terreno real = topo do sill ali -- usar os
            # contatos erodidos de Fm_SerraAlta/Fm_Teresina dava espessura
            # ZERO (os dois colapsam pro mesmo terreno onde o sill aflora,
            # exatamente o ponto que o polígono marca) -- topo = terreno,
            # base = terreno - ESPESSURA_SILL_M, só onde a linha passa DENTRO
            # do polígono real do corpo
            topo_sill = terreno
            base_sill = terreno - ESPESSURA_SILL_M
            dentro_algum_sill = np.zeros_like(xs, dtype=bool)
            for _, geom_sill in sills:
                dentro_algum_sill |= pontos_dentro_poligono(xs, ys, geom_sill)
            sx, sy = banda_mascarada(dists, quantizar(topo_sill, 1), quantizar(base_sill, 1), dentro_algum_sill)

            secoes_angulo.append(dict(
                terreno=(dists, quantizar(terreno, 1)),
                linha_mapa=(quantizar(np.array([xs[0], xs[-1]]), 0), quantizar(np.array([ys[0], ys[-1]]), 0)),
                bandas=bandas, sill=(sx, sy),
                furos=furos_no_perfil(furos, info["dx"], info["dy"], info["px"], info["py"], t, cx, cy, info["s_vals"][0]),
            ))
        todas_secoes.append(secoes_angulo)

    # --- seções fixas A-D (linhas desenhadas pelo usuário em secao*.shp)
    linhas_secao = carregar_linhas_secao()
    secoes_fixas = []
    for nome_l, geom_l in linhas_secao:
        # o modelo só existe dentro do retângulo (area2.shp): recorta a linha nele; se ela sai e volta,
        # fica o trecho mais longo (fora da área não há terreno/planos confiáveis e o perfil quebraria)
        recorte = shapely.intersection(geom_l, poligono)
        pedacos = [g for g in getattr(recorte, "geoms", [recorte]) if g.geom_type == "LineString" and g.length > 0]
        if not pedacos:
            print(f"[aviso] linha {nome_l} não cruza a área -- ignorada"); continue
        if len(pedacos) > 1:
            pedacos = [shapely.line_merge(shapely.MultiLineString(pedacos))] if shapely.line_merge(shapely.MultiLineString(pedacos)).geom_type == "LineString" else [max(pedacos, key=lambda g: g.length)]
        if recorte.length < geom_l.length - 1:
            print(f"[info] linha {nome_l}: {geom_l.length / 1000:.1f} km -> {pedacos[0].length / 1000:.1f} km dentro da área")
        geom_l = pedacos[0]
        sec, comp_km = secao_na_polilinha(geom_l, interp_terreno, planos, sills, furos)
        secoes_fixas.append(dict(nome=nome_l, compKm=round(comp_km, 3), secao=sec, vertices=[(float(x), float(y)) for x, y in np.array(geom_l.coords)[:, :2]]))
    print(f"[info] seções fixas: " + ", ".join(f"{d['nome']} ({d['compKm']:.1f} km)" for d in secoes_fixas))

    # --- figura: mapa (row1,col1) + perfil (row1,col2) ---
    fig = make_subplots(rows=1, cols=2, column_widths=[0.34, 0.66], horizontal_spacing=0.05)

    fig.add_trace(go.Heatmap(x=mx[0, :], y=my[:, 0], z=mz, colorscale="YlOrBr_r", showscale=False, hoverinfo="none"), row=1, col=1)

    idx_geo_inicio = len(fig.data)
    for row in litologia.itertuples():
        cor = COR_POR_SIGLA.get(row.SIGLA_UNID, "#CCCCCC")
        gx, gy = poligono_para_scatter_xy(row.geometry)
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="lines", line=dict(width=0), fill="toself", fillcolor=cor,
            showlegend=False, visible=False, hoverinfo="none",
        ), row=1, col=1)
    idx_geo_fim = len(fig.data) - 1
    n_geo = idx_geo_fim - idx_geo_inicio + 1

    adicionar_escala_e_norte(fig, e_min, e_max, n_min, n_max, row=1, col=1, xref="x", yref="y", comprimento_km=5.0)
    fig.update_xaxes(showticklabels=False, row=1, col=1, range=[e_min, e_max], autorange=False, scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(showticklabels=False, row=1, col=1, range=[n_min, n_max], autorange=False)

    idx_furos_mapa = len(fig.data)
    fig.add_trace(go.Scatter(
        x=furos["x"], y=furos["y"], mode="markers", name="Furos", showlegend=False, text=furos["hover"],
        hovertemplate="%{text}<extra></extra>", marker=dict(size=6, color=furos["cor"], line=dict(width=1, color=MARCA_NAVY)),
    ), row=1, col=1)

    # as 4 linhas A-D sempre visíveis no mapa (tracejado claro, rótulo na ponta inicial e final)
    lx, ly, lt = [], [], []
    for d in secoes_fixas:
        for (x, y) in d["vertices"]:
            lx.append(x); ly.append(y); lt.append("")
        lt[-len(d["vertices"])] = d["nome"]
        lt[-1] = d["nome"] + "'"
        lx.append(None); ly.append(None); lt.append("")
    fig.add_trace(go.Scatter(x=lx, y=ly, text=lt, mode="lines+text", line=dict(color="rgba(255,255,255,.75)", width=1.6, dash="dot"),
                              textposition="top center", textfont=dict(size=13, color=MARCA_NAVY), showlegend=False, hoverinfo="skip"), row=1, col=1)

    n_pos_inicial = len(angulos_info[0]["t_vals"])
    p0 = n_pos_inicial // 2
    inicial = todas_secoes[0][p0]

    idx_linha_mapa = len(fig.data)
    fig.add_trace(go.Scatter(x=inicial["linha_mapa"][0], y=inicial["linha_mapa"][1], mode="lines",
                              line=dict(color="#FF3B6B", width=2.5, dash="dash"), showlegend=False, hoverinfo="skip"), row=1, col=1)

    idx_bandas = {}
    for unidade in UNIDADES_ESTILIZADO:
        x, y = inicial["bandas"][unidade]
        idx_bandas[unidade] = len(fig.data)
        fig.add_trace(go.Scatter(
            x=x, y=y, fill="toself", fillcolor=CORES_ESTILIZADO[unidade], line=dict(width=0), mode="lines",
            name=NOMES_ESTILIZADO[unidade], showlegend=False,
            hovertemplate=f"{NOMES_ESTILIZADO[unidade]}<extra></extra>",
        ), row=1, col=2)

    idx_sill = len(fig.data)
    sx, sy = inicial["sill"]
    fig.add_trace(go.Scatter(
        x=sx, y=sy, fill="toself", fillcolor=CORES_ESTILIZADO["Gp_SerraGeral"], line=dict(width=0.5, color=MARCA_NAVY), mode="lines",
        name="Sill (Gp. Serra Geral)", showlegend=False, hovertemplate="Sill<extra></extra>",
    ), row=1, col=2)

    idx_terreno_perfil = len(fig.data)
    fig.add_trace(go.Scatter(x=inicial["terreno"][0], y=inicial["terreno"][1], mode="lines",
                              line=dict(color="#E8A33D", width=2), name="Terreno real", showlegend=False,
                              hovertemplate="Terreno real<extra></extra>"), row=1, col=2)

    # furos no perfil: contorno preto + 1 traço colorido por classe (unidade / sem topos / corpo intrusivo)
    fb = inicial["furos"]["base"]
    idx_furos_base = len(fig.data)
    fig.add_trace(go.Scatter(x=fb[0], y=fb[1], mode="lines", line=dict(color="black", width=9), name="Furos",
                              showlegend=False, hoverinfo="skip"), row=1, col=2)
    idx_furos_cls = []
    for k, (nome_k, cor_k) in enumerate(zip(NOMES_CLASSES_FURO, CORES_CLASSES_FURO)):
        cx_, cy_, ct_ = inicial["furos"]["cls"][k]
        idx_furos_cls.append(len(fig.data))
        fig.add_trace(go.Scatter(x=cx_, y=cy_, text=ct_, mode="lines", line=dict(color=cor_k, width=5 if k < len(NOMES_CLASSES_FURO) - 1 else 7),
                                  name=f"Furo · {nome_k}", showlegend=False, hovertemplate="%{text}<extra></extra>"), row=1, col=2)

    # código do furo na base de cada coluna, com buffer branco (halo via CSS: .textpoint text { stroke: white })
    lb = inicial["furos"]["lab"]
    idx_furos_rot = len(fig.data)
    fig.add_trace(go.Scatter(x=lb[0], y=lb[1], text=lb[2], mode="text", textposition="bottom center",
                              textfont=dict(size=9, color=MARCA_NAVY, family=MARCA_FONTE), name="Códigos dos furos",
                              showlegend=False, hoverinfo="skip", cliponaxis=True), row=1, col=2)
    idx_traces_frame = ([idx_linha_mapa] + [idx_bandas[u] for u in UNIDADES_ESTILIZADO]
                        + [idx_sill, idx_terreno_perfil, idx_furos_base] + idx_furos_cls + [idx_furos_rot])

    comprimento0_km = (angulos_info[0]["s_vals"][-1] - angulos_info[0]["s_vals"][0]) / 1000
    fig.update_xaxes(title_text="Distância ao longo da seção (km)", row=1, col=2, range=[0, comprimento0_km], autorange=False)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=2)

    ordem_frame = (["linha_mapa"] + UNIDADES_ESTILIZADO + ["sill", "terreno", "furos_base"]
                   + [f"furos_{k}" for k in range(len(NOMES_CLASSES_FURO))] + ["rotulos"])
    def montar_dados_frame(secao):
        dados = []
        for chave in ordem_frame:
            text = None
            if chave == "linha_mapa":
                x, y = secao["linha_mapa"]
            elif chave == "terreno":
                x, y = secao["terreno"]
            elif chave == "sill":
                x, y = secao["sill"]
            elif chave == "furos_base":
                x, y, _ = secao["furos"]["base"]
            elif chave == "rotulos":
                x, y, text = secao["furos"]["lab"]
                dados.append(go.Scatter(x=x, y=y, text=text)); continue
            elif chave.startswith("furos_"):
                x, y, text = secao["furos"]["cls"][int(chave.split("_")[1])]
            else:
                x, y = secao["bandas"][chave]
            dados.append(go.Scatter(x=x, y=y, text=text) if text is not None else go.Scatter(x=x, y=y))
        return dados

    frames = []
    for a, secoes_angulo in enumerate(todas_secoes):
        for p, secao in enumerate(secoes_angulo):
            frames.append(go.Frame(data=montar_dados_frame(secao), name=f"{a}_{p}", traces=idx_traces_frame))
    for i, d in enumerate(secoes_fixas):
        frames.append(go.Frame(data=montar_dados_frame(d["secao"]), name=f"L_{i}", traces=idx_traces_frame))
    fig.frames = frames

    # --- tudo que é controle fica FORA do plot (barra HTML no topo); o plot só tem mapa + perfil
    tema = tema_escuro()
    fig.update_layout(
        paper_bgcolor=tema["paper_bgcolor"], plot_bgcolor=tema["plot_bgcolor"],
        font=dict(family=MARCA_FONTE, color=tema["font_color"]), showlegend=False,
        margin=dict(l=62, r=14, t=10, b=48),
        images=[dict(source=sat_uri, xref="x", yref="y", x=e_min, y=n_max, sizex=e_max - e_min, sizey=n_max - n_min,
                     xanchor="left", yanchor="top", sizing="stretch", layer="below", opacity=1, visible=False)],
    )
    for eixo in ("xaxis", "yaxis", "xaxis2", "yaxis2"):
        fig.layout[eixo].update(color=tema["axis_color"], gridcolor=tema["grid_color"])

    idx_mapa_cor = [0] + list(range(idx_geo_inicio, idx_geo_fim + 1))
    # o heatmap fica sempre visível (opacidade 0 nos outros modos) -- é ele que recebe o clique que move a linha de corte
    modos_mapa = [
        dict(nome="Hipsometria", visible=[True] + [False] * n_geo, opacity=[1] + [1] * n_geo, img=False),
        dict(nome="Geologia (CPRM + sills)", visible=[True] + [True] * n_geo, opacity=[0] + [1] * n_geo, img=False),
        dict(nome="Satélite", visible=[True] + [False] * n_geo, opacity=[0] + [1] * n_geo, img=True),
    ]
    legenda = (
        [dict(nome=NOMES_ESTILIZADO[u], cor=CORES_ESTILIZADO[u], idx=[idx_bandas[u]]) for u in UNIDADES_ESTILIZADO]
        + [dict(nome="Sill (Gp. Serra Geral)", cor=CORES_ESTILIZADO["Gp_SerraGeral"], idx=[idx_sill]),
           dict(nome="Terreno real", cor="#E8A33D", idx=[idx_terreno_perfil])]
    )
    furos_legenda = [dict(nome=n, cor=c) for n, c in zip(NOMES_CLASSES_FURO, CORES_CLASSES_FURO)]
    cfg = dict(
        cx=cx, cy=cy, p0=p0, idxMapaMax=idx_geo_fim, idxMapaCor=idx_mapa_cor, modos=modos_mapa,
        idxFuros=[idx_furos_mapa, idx_furos_base] + idx_furos_cls + [idx_furos_rot],
        angulos=[dict(nome=info["nome"], px=info["px"], py=info["py"], compKm=(info["s_vals"][-1] - info["s_vals"][0]) / 1000,
                      t=[round(float(t), 2) for t in info["t_vals"]]) for info in angulos_info],
        temas=dict(escuro=dict(paper=tema_escuro()["paper_bgcolor"], plot=tema_escuro()["plot_bgcolor"], font=tema_escuro()["font_color"],
                               grid=tema_escuro()["grid_color"], axis=tema_escuro()["axis_color"]),
                   claro=dict(paper=tema_claro()["paper_bgcolor"], plot=tema_claro()["plot_bgcolor"], font=tema_claro()["font_color"],
                              grid=tema_claro()["grid_color"], axis=tema_claro()["axis_color"])),
        legenda=legenda,
        secoes=[dict(nome=d['nome'], compKm=d['compKm']) for d in secoes_fixas],
    )

    def opcoes(itens):
        return "".join(f'<option value="{i}">{n}</option>' for i, n in enumerate(itens))

    html_botoes_secao = "".join(f'<button class="btn btn-secao" data-i="{i}" title="Seção {d["nome"]} — {d["compKm"]:.1f} km">{d["nome"]}</button>'
                                for i, d in enumerate(secoes_fixas))
    html_legenda = "".join(
        f'<span class="leg-item" data-i="{i}"><i style="background:{it["cor"]}"></i>{it["nome"]}</span>' for i, it in enumerate(legenda))
    html_furos_leg = "".join(f'<span class="leg-fixo"><i style="background:{it["cor"]}"></i>{it["nome"]}</span>' for it in furos_legenda)

    logo = logo_base64()
    grafico = pio.to_html(fig, full_html=False, include_plotlyjs=False, post_script=POST_JS.replace("@@CFG@@", json.dumps(cfg, ensure_ascii=False)),
                          div_id="secao", auto_play=False, config={"responsive": True, "displaylogo": False})   # auto_play=False: senão o Plotly sai tocando todos os frames
    html = TEMPLATE
    for token, valor in {
        "@@PLOTLY@@": PLOTLY_CDN, "@@FONTE@@": MARCA_FONTE, "@@NAVY@@": MARCA_NAVY, "@@ROXO@@": MARCA_ROXO,
        "@@CINZA@@": MARCA_CINZA_CLARO,
        "@@LOGO@@": f'<img src="data:image/jpeg;base64,{logo}" alt="GS Tech">' if logo else "",
        "@@OPC_ANGULO@@": opcoes([a["nome"] for a in cfg["angulos"]]),
        "@@OPC_MAPA@@": opcoes([m["nome"] for m in modos_mapa]),
        "@@NPOS@@": str(len(angulos_info[0]["t_vals"]) - 1), "@@P0@@": str(p0),
        "@@BOTOES_SECAO@@": html_botoes_secao, "@@LEGENDA@@": html_legenda, "@@FUROS_LEG@@": html_furos_leg,
        "@@GRAFICO@@": grafico,
    }.items():
        html = html.replace(token, valor)
    assert "@@" not in html.replace("@@CFG@@", ""), "token sem substituir"
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[info] {len(angulos_info)} ângulos, {n_pos_inicial} posições iniciais, {len(sills)} sills, {len(frames)} frames")
    print(f"-> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


POST_JS = r"""
(function() {
    var C = @@CFG@@;
    var gd = document.getElementById('secao');
    var anguloAtual = 0, posAtual = C.p0, furosOn = true, secaoFixa = null;
    var sl = document.getElementById('corte-pos'), lab = document.getElementById('corte-val');
    var botoesSecao = document.querySelectorAll('.btn-secao');
    var OPT = {mode: 'immediate', frame: {duration: 0, redraw: true}, transition: {duration: 0}};

    function rotulo(a, p) { var v = C.angulos[a].t[p]; return (v >= 0 ? '+' : '') + v.toFixed(0) + ' m'; }
    function marcarSecao(i) {
        secaoFixa = i;
        botoesSecao.forEach(function(b) { b.classList.toggle('ativo', +b.getAttribute('data-i') === i); });
        sl.classList.toggle('apagado', i !== null);
    }
    // volta pra linha de corte rotacionável (sai da seção A-D)
    function sairSecao() {
        if (secaoFixa === null) return;
        marcarSecao(null);
        Plotly.relayout(gd, {'xaxis2.range': [0, C.angulos[anguloAtual].compKm]});
    }
    function irPara(a, p) {
        anguloAtual = a; posAtual = p; sl.value = p; lab.textContent = rotulo(a, p);
        Plotly.animate(gd, [a + '_' + p], OPT).catch(function() {});
    }
    function irSecao(i) {
        marcarSecao(i);
        var s = C.secoes[i];
        lab.textContent = 'Seção ' + s.nome + ' · ' + s.compKm.toFixed(1) + ' km';
        Plotly.relayout(gd, {'xaxis2.range': [0, s.compKm]});
        Plotly.animate(gd, ['L_' + i], OPT).catch(function() {});
    }
    botoesSecao.forEach(function(b) {
        b.addEventListener('click', function() {
            var i = +b.getAttribute('data-i');
            if (secaoFixa === i) { sairSecao(); irPara(anguloAtual, posAtual); } else { irSecao(i); }
        });
    });

    document.getElementById('sel-angulo').addEventListener('change', function(ev) {
        var a = +ev.target.value, n = C.angulos[a].t.length;
        marcarSecao(null);
        sl.max = n - 1;
        Plotly.relayout(gd, {'xaxis2.range': [0, C.angulos[a].compKm]});
        irPara(a, Math.floor(n / 2));
    });
    sl.addEventListener('input', function() {
        if (secaoFixa !== null) { marcarSecao(null); Plotly.relayout(gd, {'xaxis2.range': [0, C.angulos[anguloAtual].compKm]}); }
        irPara(anguloAtual, +sl.value);
    });

    // clique no mapa move a linha de corte (e sai da seção A-D)
    gd.on('plotly_click', function(data) {
        var pt = data.points[0];
        if (!(pt.curveNumber >= 0 && pt.curveNumber <= C.idxMapaMax)) return;
        if (secaoFixa !== null) { marcarSecao(null); Plotly.relayout(gd, {'xaxis2.range': [0, C.angulos[anguloAtual].compKm]}); }
        var info = C.angulos[anguloAtual];
        var t = (pt.x - C.cx) * info.px + (pt.y - C.cy) * info.py, melhor = 0, dmin = Infinity;
        for (var p = 0; p < info.t.length; p++) { var d = Math.abs(info.t[p] - t); if (d < dmin) { dmin = d; melhor = p; } }
        irPara(anguloAtual, melhor);
    });

    // modo do mapa
    document.getElementById('sel-mapa').addEventListener('change', function(ev) {
        var m = C.modos[+ev.target.value];
        Plotly.update(gd, {visible: m.visible, opacity: m.opacity}, {'images[0].visible': m.img}, C.idxMapaCor);
    });

    // furos ON/OFF
    var bF = document.getElementById('b-furos');
    bF.addEventListener('click', function() {
        furosOn = !furosOn; bF.classList.toggle('ativo', furosOn); bF.textContent = 'Furos: ' + (furosOn ? 'ON' : 'OFF');
        Plotly.restyle(gd, {visible: furosOn}, C.idxFuros);
    });

    // legenda clicável (liga/desliga cada camada do perfil)
    document.querySelectorAll('.leg-item').forEach(function(el) {
        el.addEventListener('click', function() {
            var it = C.legenda[+el.getAttribute('data-i')], off = el.classList.toggle('off');
            Plotly.restyle(gd, {visible: off ? 'legendonly' : true}, it.idx);
        });
    });

    // tema: página + gráfico
    function aplicarTema(nome) {
        var t = C.temas[nome];
        document.body.classList.toggle('tema-claro', nome === 'claro');
        document.querySelectorAll('[data-tema]').forEach(function(b) { b.classList.toggle('ativo', b.getAttribute('data-tema') === nome); });
        var up = {paper_bgcolor: t.paper, plot_bgcolor: t.plot, 'font.color': t.font};
        ['xaxis', 'yaxis', 'xaxis2', 'yaxis2'].forEach(function(ax) { up[ax + '.color'] = t.axis; up[ax + '.gridcolor'] = t.grid; });
        Plotly.relayout(gd, up);
    }
    document.querySelectorAll('[data-tema]').forEach(function(b) { b.addEventListener('click', function() { aplicarTema(b.getAttribute('data-tema')); }); });

    lab.textContent = rotulo(0, C.p0);
})();
"""


TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" href="assets/favicon.png">
<link rel="shortcut icon" href="assets/favicon.ico">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
<title>Seção 2D interativa — PMP</title>
<script src="@@PLOTLY@@"></script>
<style>
  * { box-sizing: border-box; }
  :root { --bg: @@NAVY@@; --painel: #262B3D; --texto: @@CINZA@@; --borda: #3a3f52; }
  body.tema-claro { --bg: #FFFFFF; --painel: #F2F2F7; --texto: @@NAVY@@; --borda: #D8D8E2; }
  html, body { height: 100%; }
  body { margin: 0; background: var(--bg); color: var(--texto); font-family: @@FONTE@@; display: flex; flex-direction: column; transition: background .2s, color .2s; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 6px 20px; border-bottom: 1px solid @@ROXO@@; gap: 16px; flex-shrink: 0; }
  header h1 { font-size: 17px; margin: 0; }
  header h1 b { color: @@ROXO@@; }
  header img { width: 38px; height: 38px; border-radius: 50%; border: 2px solid @@ROXO@@; box-shadow: 0 0 10px rgba(123,47,255,.6); }
  #barra { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 22px; padding: 10px 20px; background: var(--painel); border-bottom: 1px solid var(--borda); font-size: 12px; flex-shrink: 0; }
  .grupo { display: flex; align-items: center; gap: 7px; }
  .grupo > .rot { opacity: .65; text-transform: uppercase; letter-spacing: .04em; font-size: 10.5px; }
  select, .btn { font-family: @@FONTE@@; font-size: 12px; background: transparent; color: var(--texto); border: 1px solid @@ROXO@@; border-radius: 6px; padding: 5px 9px; }
  select { background: var(--bg); padding: 4px 6px; }
  .btn { cursor: pointer; opacity: .65; }
  .btn:hover { opacity: 1; }
  .btn.ativo { opacity: 1; background: @@ROXO@@; color: #fff; }
  input[type=range] { accent-color: @@ROXO@@; width: 280px; }
  .btn-secao { min-width: 34px; font-weight: 700; padding: 5px 11px; }
  input[type=range].apagado { opacity: .4; }
  #corte-val { min-width: 62px; font-variant-numeric: tabular-nums; font-weight: 600; }
  #legenda { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 16px; padding: 8px 20px 10px; font-size: 11.5px; flex-shrink: 0; border-bottom: 1px solid var(--borda); }
  .leg-item, .leg-fixo { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
  .leg-item { cursor: pointer; }
  .leg-item.off { opacity: .35; text-decoration: line-through; }
  #legenda i { width: 14px; height: 10px; border-radius: 2px; display: inline-block; border: 1px solid rgba(255,255,255,.35); }
  .leg-sep { opacity: .5; font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em; margin-left: 10px; }
  #secao .textpoint text { paint-order: stroke fill; stroke: #FFFFFF; stroke-width: 3.2px; stroke-linejoin: round; font-weight: 600; }
  #wrap { flex: 1; min-height: 0; padding: 14px 8px 0 8px; }
  #wrap .plotly-graph-div, #wrap > div { height: 100% !important; }
  footer { text-align: center; padding: 3px; opacity: .5; font-size: 10.5px; flex-shrink: 0; }
</style>
</head>
<body>
<header>
  <h1><b>Seção 2D interativa</b> — PMP · Criciúma</h1>
  @@LOGO@@
</header>
<div id="barra">
  <div class="grupo"><span class="rot">Linha de corte</span><select id="sel-angulo">@@OPC_ANGULO@@</select></div>
  <div class="grupo"><span class="rot">Mapa</span><select id="sel-mapa">@@OPC_MAPA@@</select></div>
  <div class="grupo"><span class="rot">Seções</span>@@BOTOES_SECAO@@</div>
  <div class="grupo"><span class="rot">Furos</span><button class="btn ativo" id="b-furos">Furos: ON</button></div>
  <div class="grupo"><span class="rot">Posição do corte</span><input type="range" id="corte-pos" min="0" max="@@NPOS@@" step="1" value="@@P0@@"><span id="corte-val"></span><span style="opacity:.55">ou clique no mapa</span></div>
  <div class="grupo"><span class="rot">Tema</span><button class="btn ativo" data-tema="escuro">Escuro</button><button class="btn" data-tema="claro">Claro</button></div>
</div>
<div id="legenda">@@LEGENDA@@<span class="leg-sep">Furos:</span>@@FUROS_LEG@@</div>
<div id="wrap">@@GRAFICO@@</div>
<footer>GS Tech · PMP</footer>
</body>
</html>
"""


if __name__ == "__main__":
    main()
