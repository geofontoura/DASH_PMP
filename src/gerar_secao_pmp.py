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
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from plotly.subplots import make_subplots

from _comum_pmp import (
    MARCA_ROXO, MARCA_NAVY, MARCA_CINZA_CLARO, MARCA_FONTE,
    UNIDADES_ESTILIZADO, NOMES_ESTILIZADO, CORES_ESTILIZADO, ESPESSURA_SILL_M,
    COR_POR_SIGLA, SIGLAS_SILL_INDIVIDUALIZADO, ORDEM_PROFUNDIDADE_FURO,
    logo_base64, carregar_area_pmp, carregar_vertices_curvas_pmp, carregar_litologia_pmp,
    carregar_sills_individualizados, construir_interpolador, avaliar_interpolador, avaliar_plano,
    calcular_planos_estilizados, poligono_para_scatter_xy, pontos_dentro_poligono,
    botoes_tema, tema_escuro, adicionar_escala_e_norte, quantizar,
    obter_satelite_utm, preparar_furos, _intervalos_corpo,
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
    base = ([v for c in cls for v in c[0]], [v for c in cls for v in c[1]], [None] * sum(len(c[0]) for c in cls))
    return dict(base=base, cls=cls)


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

    # --- pre-computa uma secao (bandas de formacao + sill + espessuras) por
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
            espessuras = []
            for i, unidade in enumerate(UNIDADES_ESTILIZADO):
                topo = contatos[unidade]
                base = contatos[UNIDADES_ESTILIZADO[i + 1]] if i + 1 < len(UNIDADES_ESTILIZADO) else topo - 300.0
                bandas[unidade] = (
                    np.concatenate([dists, dists[::-1]]),
                    np.concatenate([quantizar(topo, 1), quantizar(base, 1)[::-1]]),
                )
                espessuras.append(float(np.nanmean(topo - base)))

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
                bandas=bandas, sill=(sx, sy), espessuras=espessuras,
                furos=furos_no_perfil(furos, info["dx"], info["dy"], info["px"], info["py"], t, cx, cy, info["s_vals"][0]),
            ))
        todas_secoes.append(secoes_angulo)

    # --- figura: mapa (row1,col1) + perfil (row1,col2) + barras (row2) ---
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.32, 0.68], row_heights=[0.78, 0.22],
        horizontal_spacing=0.06, vertical_spacing=0.17,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("Mapa (clique p/ mover o corte)", "Perfil", "Espessura na linha atual"),
    )

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

    n_pos_inicial = len(angulos_info[0]["t_vals"])
    p0 = n_pos_inicial // 2
    inicial = todas_secoes[0][p0]

    idx_linha_mapa = len(fig.data)
    fig.add_trace(go.Scatter(x=inicial["linha_mapa"][0], y=inicial["linha_mapa"][1], mode="lines",
                              line=dict(color="#FF3B6B", width=2.5, dash="dash"), showlegend=False), row=1, col=1)

    idx_bandas = {}
    for unidade in UNIDADES_ESTILIZADO:
        x, y = inicial["bandas"][unidade]
        idx_bandas[unidade] = len(fig.data)
        fig.add_trace(go.Scatter(
            x=x, y=y, fill="toself", fillcolor=CORES_ESTILIZADO[unidade], line=dict(width=0), mode="lines",
            name=NOMES_ESTILIZADO[unidade], showlegend=True,
            hovertemplate=f"{NOMES_ESTILIZADO[unidade]}<extra></extra>",
        ), row=1, col=2)

    idx_sill = len(fig.data)
    sx, sy = inicial["sill"]
    fig.add_trace(go.Scatter(
        x=sx, y=sy, fill="toself", fillcolor=CORES_ESTILIZADO["Gp_SerraGeral"], line=dict(width=0.5, color=MARCA_NAVY), mode="lines",
        name="Sill (Gp. Serra Geral)", showlegend=True,
        hovertemplate="Sill<extra></extra>",
    ), row=1, col=2)

    idx_terreno_perfil = len(fig.data)
    fig.add_trace(go.Scatter(x=inicial["terreno"][0], y=inicial["terreno"][1], mode="lines",
                              line=dict(color="#E8A33D", width=2), name="Terreno real", showlegend=True), row=1, col=2)

    # furos no perfil: contorno preto (1 legenda) + 1 traço colorido por classe (unidade / sem topos / corpo intrusivo)
    fb = inicial["furos"]["base"]
    idx_furos_base = len(fig.data)
    fig.add_trace(go.Scatter(x=fb[0], y=fb[1], mode="lines", line=dict(color="black", width=9), name=f"Furos (±{BUFFER_FUROS_M / 1000:.1f} km)",
                              legendgroup="furos", showlegend=True, hoverinfo="skip"), row=1, col=2)
    idx_furos_cls = []
    for k, (nome_k, cor_k) in enumerate(zip(NOMES_CLASSES_FURO, CORES_CLASSES_FURO)):
        cx_, cy_, ct_ = inicial["furos"]["cls"][k]
        idx_furos_cls.append(len(fig.data))
        fig.add_trace(go.Scatter(x=cx_, y=cy_, text=ct_, mode="lines", line=dict(color=cor_k, width=5 if k < len(NOMES_CLASSES_FURO) - 1 else 7),
                                  name=f"Furo · {nome_k}", legendgroup="furos", showlegend=False,
                                  hovertemplate="%{text}<extra></extra>"), row=1, col=2)

    idx_traces_frame = ([idx_linha_mapa] + [idx_bandas[u] for u in UNIDADES_ESTILIZADO]
                        + [idx_sill, idx_terreno_perfil, idx_furos_base] + idx_furos_cls)

    comprimento0_km = (angulos_info[0]["s_vals"][-1] - angulos_info[0]["s_vals"][0]) / 1000
    fig.update_xaxes(title_text="Distância ao longo da seção (km)", row=1, col=2, range=[0, comprimento0_km], autorange=False)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=2)

    ordem_frame = (["linha_mapa"] + UNIDADES_ESTILIZADO + ["sill", "terreno", "furos_base"]
                   + [f"furos_{k}" for k in range(len(NOMES_CLASSES_FURO))])
    frames = []
    for a, secoes_angulo in enumerate(todas_secoes):
        for p, secao in enumerate(secoes_angulo):
            dados_frame = []
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
                elif chave.startswith("furos_"):
                    x, y, text = secao["furos"]["cls"][int(chave.split("_")[1])]
                else:
                    x, y = secao["bandas"][chave]
                dados_frame.append(go.Scatter(x=x, y=y, text=text) if text is not None else go.Scatter(x=x, y=y))
            frames.append(go.Frame(data=dados_frame, name=f"{a}_{p}", traces=idx_traces_frame))
    fig.frames = frames

    idx_barra = len(fig.data)
    fig.add_trace(go.Bar(
        x=[NOMES_ESTILIZADO[u] for u in UNIDADES_ESTILIZADO], y=inicial["espessuras"],
        marker_color=[CORES_ESTILIZADO[u] for u in UNIDADES_ESTILIZADO], marker_line=dict(color=MARCA_ROXO, width=1),
        text=[f"{e:.0f}m" for e in inicial["espessuras"]], textposition="outside",
        showlegend=False, hoverinfo="none",
    ), row=2, col=1)
    fig.update_yaxes(title_text="Espessura (m)", row=2, col=1)

    # --- menus: ângulo (skip -- tratado em JS), modo do mapa (update: visibilidade + fundo satélite), furos, tema
    idx_mapa_cor = [0] + list(range(idx_geo_inicio, idx_geo_fim + 1))
    botoes_angulo = [dict(label=nome, method="skip") for nome, _ in ANGULOS]
    botoes_mapa_cor = [
        dict(label="Mapa: Hipsometria", method="update",
             args=[{"visible": [True] + [False] * n_geo, "opacity": [1] + [1] * n_geo}, {"images[0].visible": False}, idx_mapa_cor]),
        dict(label="Mapa: Geologia (CPRM + sills)", method="update",
             args=[{"visible": [True] + [True] * n_geo, "opacity": [0] + [1] * n_geo}, {"images[0].visible": False}, idx_mapa_cor]),
        dict(label="Mapa: Satélite", method="update",
             args=[{"visible": [True] + [False] * n_geo, "opacity": [0] + [1] * n_geo}, {"images[0].visible": True}, idx_mapa_cor]),
    ]   # o heatmap fica sempre visível (opacidade 0 nos outros modos) -- é ele que recebe o clique que move a linha de corte
    idx_furos_todos = [idx_furos_mapa, idx_furos_base] + idx_furos_cls
    botoes_furos = [
        dict(label="Furos: ON", method="restyle", args=[{"visible": True}, idx_furos_todos]),
        dict(label="Furos: OFF", method="restyle", args=[{"visible": False}, idx_furos_todos]),
    ]

    tema = tema_escuro()
    estilo_menu = dict(direction="down", xanchor="left", y=1.34, yanchor="top", bgcolor="#4A4A4A", font=dict(color="white"))
    fig.update_layout(
        paper_bgcolor=tema["paper_bgcolor"], plot_bgcolor=tema["plot_bgcolor"],
        font=dict(family=MARCA_FONTE, color=tema["font_color"]),
        legend=dict(x=1.01, y=0.9, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=170, t=175, b=60),
        updatemenus=[
            dict(buttons=botoes_angulo, x=0.0, **estilo_menu),
            dict(buttons=botoes_mapa_cor, x=0.16, **estilo_menu),
            dict(buttons=botoes_furos, x=0.40, **estilo_menu),
            dict(buttons=botoes_tema(eixos_2d=["xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"]), x=0.50, active=1, **estilo_menu),
        ],
        sliders=[dict(
            active=p0, x=0.0, len=0.66, xanchor="left", y=1.19, yanchor="top",
            currentvalue=dict(prefix="Posição do corte: ", font=dict(size=12, color=tema["font_color"])),
            font=dict(size=1, color="rgba(0,0,0,0)"),   # esconde os rótulos das 23 posições (só o valor atual aparece)
            steps=[
                dict(method="animate", args=[[f"0_{p}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                     label=f"{angulos_info[0]['t_vals'][p]:+.0f} m")
                for p in range(n_pos_inicial)
            ],
        )],
        images=[dict(source=sat_uri, xref="x", yref="y", x=e_min, y=n_max, sizex=e_max - e_min, sizey=n_max - n_min,
                     xanchor="left", yanchor="top", sizing="stretch", layer="below", opacity=1, visible=False)],
    )
    for eixo in ("xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"):
        fig.layout[eixo].update(color=tema["axis_color"], gridcolor=tema["grid_color"])

    angulos_js = ",\n        ".join(
        "{dx:%.6f, dy:%.6f, px:%.6f, py:%.6f, s0:%.3f, compKm:%.3f, t:[%s]}" % (
            info["dx"], info["dy"], info["px"], info["py"], info["s_vals"][0],
            (info["s_vals"][-1] - info["s_vals"][0]) / 1000,
            ",".join(f"{t:.2f}" for t in info["t_vals"]),
        )
        for info in angulos_info
    )

    def fmt(e):
        return "NaN" if np.isnan(e) else f"{e:.1f}"

    espessuras_js = ",\n        ".join(
        "[" + ",\n         ".join("[" + ",".join(fmt(e) for e in secao["espessuras"]) + "]" for secao in secoes_angulo) + "]"
        for secoes_angulo in todas_secoes
    )

    post_script = f"""
    (function() {{
        var CX = {cx}, CY = {cy};
        var ANGULOS = [{angulos_js}];
        var ESPESSURAS = [{espessuras_js}];
        var IDX_BARRA = {idx_barra};
        var IDX_MAPA_MIN = 0, IDX_MAPA_MAX = {idx_geo_fim};
        var anguloAtual = 0;
        var gd = document.getElementsByClassName('plotly-graph-div')[0];

        function atualizarBarra(a, p) {{
            var vals = ESPESSURAS[a][p];
            var rotulos = vals.map(function(v) {{ return Math.round(v) + 'm'; }});
            Plotly.restyle(gd, {{y: [vals], text: [rotulos]}}, [IDX_BARRA]);
        }}
        function stepsParaAngulo(a) {{
            var info = ANGULOS[a];
            var steps = [];
            for (var p = 0; p < info.t.length; p++) {{
                var rotulo = (info.t[p] >= 0 ? '+' : '') + info.t[p].toFixed(0) + ' m';
                steps.push({{method: 'animate', args: [[a + '_' + p], {{mode: 'immediate', frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}}}], label: rotulo}});
            }}
            return steps;
        }}
        function irParaAngulo(a) {{
            anguloAtual = a;
            var posMeio = Math.floor(ANGULOS[a].t.length / 2);
            Plotly.relayout(gd, {{'sliders[0].steps': stepsParaAngulo(a), 'sliders[0].active': posMeio, 'xaxis2.range': [0, ANGULOS[a].compKm]}});
            Plotly.animate(gd, [a + '_' + posMeio], {{mode: 'immediate', frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}}}).catch(function() {{}});
            atualizarBarra(a, posMeio);
        }}
        gd.on('plotly_buttonclicked', function(ev) {{
            if (typeof ev.active !== 'number') return;
            if (Math.abs(ev.menu.y - 1.34) <= 0.01 && ev.menu.x < 0.1) {{ irParaAngulo(ev.active); }}
        }});
        gd.on('plotly_sliderchange', function() {{
            setTimeout(function() {{ atualizarBarra(anguloAtual, gd.layout.sliders[0].active); }}, 0);
        }});
        gd.on('plotly_click', function(data) {{
            var ponto = data.points[0];
            if (!(ponto.curveNumber >= IDX_MAPA_MIN && ponto.curveNumber <= IDX_MAPA_MAX)) return;
            var info = ANGULOS[anguloAtual];
            var t = (ponto.x - CX) * info.px + (ponto.y - CY) * info.py;
            var melhorIdx = 0, melhorDist = Infinity;
            for (var p = 0; p < info.t.length; p++) {{
                var d = Math.abs(info.t[p] - t);
                if (d < melhorDist) {{ melhorDist = d; melhorIdx = p; }}
            }}
            Plotly.relayout(gd, {{'sliders[0].active': melhorIdx}});
            Plotly.animate(gd, [anguloAtual + '_' + melhorIdx], {{mode: 'immediate', frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}}}).catch(function() {{}});
            atualizarBarra(anguloAtual, melhorIdx);
        }});
        // abre com o slider na posição central (o Plotly ignora sliders.active no carregamento com frames)
        Plotly.relayout(gd, {{'sliders[0].active': {p0}}});
        atualizarBarra(0, {p0});
    }})();
    """

    logo = logo_base64()
    grafico = pio.to_html(fig, full_html=False, include_plotlyjs=False, post_script=post_script, div_id="secao",
                          config={"responsive": True, "displaylogo": False})
    html = TEMPLATE
    for token, valor in {
        "@@PLOTLY@@": PLOTLY_CDN, "@@FONTE@@": MARCA_FONTE, "@@NAVY@@": MARCA_NAVY, "@@ROXO@@": MARCA_ROXO,
        "@@CINZA@@": MARCA_CINZA_CLARO,
        "@@LOGO@@": f'<img src="data:image/jpeg;base64,{logo}" alt="GS Tech">' if logo else "",
        "@@GRAFICO@@": grafico,
    }.items():
        html = html.replace(token, valor)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[info] {len(angulos_info)} ângulos, {n_pos_inicial} posições iniciais, {len(sills)} sills, {len(frames)} frames")
    print(f"-> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


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
  html, body { height: 100%; }
  body { margin: 0; background: @@NAVY@@; color: @@CINZA@@; font-family: @@FONTE@@; display: flex; flex-direction: column; transition: background .2s; }
  body.tema-claro { background: #FFFFFF; color: @@NAVY@@; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 10px 24px; border-bottom: 1px solid @@ROXO@@; gap: 16px; flex-shrink: 0; }
  header h1 { font-size: 19px; margin: 0; }
  header h1 b { color: @@ROXO@@; }
  header img { width: 46px; height: 46px; border-radius: 50%; border: 2px solid @@ROXO@@; box-shadow: 0 0 10px rgba(123,47,255,.6); }
  #wrap { flex: 1; min-height: 0; }
  #wrap .plotly-graph-div, #wrap > div { height: 100% !important; }
  footer { text-align: center; padding: 4px; opacity: .5; font-size: 11px; flex-shrink: 0; }
</style>
</head>
<body>
<header>
  <h1><b>Seção 2D interativa</b> — PMP · Criciúma</h1>
  @@LOGO@@
</header>
<div id="wrap">@@GRAFICO@@</div>
<footer>GS Tech · PMP</footer>
<script>
  (function() {
    var gd = document.getElementById('secao');
    // o dropdown de tema do Plotly só recolore o gráfico; aqui a página inteira acompanha
    gd.on('plotly_relayout', function(ev) {
      if (ev && ev['paper_bgcolor'] !== undefined) document.body.classList.toggle('tema-claro', ev['paper_bgcolor'] === '#FFFFFF');
    });
  })();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
