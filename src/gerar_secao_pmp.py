"""Gera a seção interativa do caso PMP -- reconstruída do zero, SEM NENHUM
dado de poço (nem banco de dados de poço, nem interpolação de poço) -- só
area.shp/curvas.shp/Litologia_PMP.shp, igual ao cubo estilizado
(gerar_cubo_estilizado_pmp.py, mesmos planos por formação, ver
_comum_pmp.py::calcular_planos_estilizados).

Um modo só (mais simples que a versão anterior, que tinha 3 modos e
dependia de poço): mapa em planta (hipsometria ou geologia real) + perfil,
com 4 ângulos de corte e posição por slider ou clique direto no mapa --
mesma técnica do gerar_secao_interativa.py do Taió.

Fontes: ../2_Banco_de_Dados/{area,curvas,Litologia_PMP}.shp -- via
_comum_pmp.py. Este script só LÊ essas fontes.

Uso:
    python gerar_secao_estilizada_pmp.py

Gera:
    secao_estilizada_pmp.html
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _comum_pmp import (
    MARCA_ROXO, MARCA_NAVY, MARCA_CINZA_CLARO, MARCA_FONTE,
    UNIDADES_ESTILIZADO, NOMES_ESTILIZADO, CORES_ESTILIZADO, ESPESSURA_SILL_M,
    logo_base64, carregar_area_pmp, carregar_vertices_curvas_pmp, carregar_litologia_pmp,
    carregar_sills_individualizados, construir_interpolador, avaliar_interpolador, avaliar_plano,
    calcular_planos_estilizados, poligono_para_scatter_xy, pontos_dentro_poligono,
    botoes_tema, adicionar_escala_e_norte, quantizar,
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


def main():
    poligono, bounds = carregar_area_pmp()
    e_min, n_min, e_max, n_max = bounds
    cx, cy = (e_min + e_max) / 2, (n_min + n_max) / 2

    xt, yt, zt = carregar_vertices_curvas_pmp()
    interp_terreno = construir_interpolador(xt, yt, zt)
    litologia = carregar_litologia_pmp()
    planos = calcular_planos_estilizados(litologia, interp_terreno)
    sills = carregar_sills_individualizados()

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
            ))
        todas_secoes.append(secoes_angulo)

    # --- figura: mapa (row1,col1) + perfil (row1,col2) + barras (row2) ---
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.32, 0.68], row_heights=[0.78, 0.22],
        horizontal_spacing=0.06, vertical_spacing=0.12,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("Mapa (clique p/ mover o corte)", "Perfil", "Espessura na linha atual"),
    )

    fig.add_trace(go.Heatmap(x=mx[0, :], y=my[:, 0], z=mz, colorscale="YlOrBr_r", showscale=False, hoverinfo="none"), row=1, col=1)

    idx_geo_inicio = len(fig.data)
    for row in litologia.itertuples():
        unidade = row.unidade_padrao
        eh_sill = row.HIERARQUIA == "Corpo"
        cor = CORES_ESTILIZADO.get("Gp_SerraGeral" if eh_sill else unidade, "#CCCCCC")
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

    n_pos_inicial = len(angulos_info[0]["t_vals"])
    p0 = n_pos_inicial // 2
    inicial = todas_secoes[0][p0]

    idx_linha_mapa = len(fig.data)
    fig.add_trace(go.Scatter(x=inicial["linha_mapa"][0], y=inicial["linha_mapa"][1], mode="lines",
                              line=dict(color="black", width=2, dash="dash"), showlegend=False), row=1, col=1)

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
                              line=dict(color=MARCA_NAVY, width=2), name="Terreno real", showlegend=True), row=1, col=2)

    idx_traces_frame = [idx_linha_mapa] + [idx_bandas[u] for u in UNIDADES_ESTILIZADO] + [idx_sill, idx_terreno_perfil]

    comprimento0_km = (angulos_info[0]["s_vals"][-1] - angulos_info[0]["s_vals"][0]) / 1000
    fig.update_xaxes(title_text="Distância ao longo da seção (km)", row=1, col=2, range=[0, comprimento0_km], autorange=False)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=2)

    ordem_frame = ["linha_mapa"] + UNIDADES_ESTILIZADO + ["sill", "terreno"]
    frames = []
    for a, secoes_angulo in enumerate(todas_secoes):
        for p, secao in enumerate(secoes_angulo):
            dados_frame = []
            for chave in ordem_frame:
                if chave == "linha_mapa":
                    x, y = secao["linha_mapa"]
                elif chave == "terreno":
                    x, y = secao["terreno"]
                elif chave == "sill":
                    x, y = secao["sill"]
                else:
                    x, y = secao["bandas"][chave]
                dados_frame.append(go.Scatter(x=x, y=y))
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

    botoes_angulo = [dict(label=nome, method="skip") for nome, _ in ANGULOS]
    botoes_mapa_cor = [
        dict(label="Mapa: Hipsometria", method="restyle", args=[{"visible": [True] + [False] * n_geo}, [0] + list(range(idx_geo_inicio, idx_geo_fim + 1))]),
        dict(label="Mapa: Geologia (real + sills)", method="restyle", args=[{"visible": [False] + [True] * n_geo}, [0] + list(range(idx_geo_inicio, idx_geo_fim + 1))]),
    ]

    fig.update_layout(
        title=dict(text="PMP — Seção estilizada (mapa interativo, sem poço)", font=dict(family=MARCA_FONTE, color=MARCA_NAVY, size=18)),
        paper_bgcolor=MARCA_CINZA_CLARO, font=dict(family=MARCA_FONTE),
        legend=dict(x=1.01, y=0.9),
        margin=dict(l=60, r=140, t=150, b=60),
        updatemenus=[
            dict(buttons=botoes_angulo, direction="down", x=0.0, xanchor="left", y=1.2, yanchor="top", bgcolor="#4A4A4A", font=dict(color="white")),
            dict(buttons=botoes_mapa_cor, direction="down", x=0.20, xanchor="left", y=1.2, yanchor="top", bgcolor="#4A4A4A", font=dict(color="white")),
            dict(buttons=botoes_tema(eixos_2d=["xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"]),
                 direction="down", x=0.46, xanchor="left", y=1.2, yanchor="top", bgcolor="#4A4A4A", font=dict(color="white")),
        ],
        sliders=[dict(
            active=p0, x=0.0, len=0.66, xanchor="left", y=1.10, yanchor="top",
            currentvalue=dict(prefix="Posição do corte: ", font=dict(color=MARCA_NAVY)),
            steps=[
                dict(method="animate", args=[[f"0_{p}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                     label=f"{angulos_info[0]['t_vals'][p]:+.0f} m")
                for p in range(n_pos_inicial)
            ],
        )],
    )

    logo = logo_base64()
    if logo:
        fig.add_layout_image(dict(source=f"data:image/jpeg;base64,{logo}", xref="paper", yref="paper", x=1.0, y=1.24,
                                   sizex=0.09, sizey=0.09, xanchor="right", yanchor="top"))

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
            Plotly.animate(gd, [a + '_' + posMeio], {{mode: 'immediate', frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}}});
            atualizarBarra(a, posMeio);
        }}
        gd.on('plotly_buttonclicked', function(ev) {{
            if (typeof ev.active !== 'number') return;
            if (Math.abs(ev.menu.y - 1.2) <= 0.01 && ev.menu.x < 0.1) {{ irParaAngulo(ev.active); }}
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
            Plotly.animate(gd, [anguloAtual + '_' + melhorIdx], {{mode: 'immediate', frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}}});
            atualizarBarra(anguloAtual, melhorIdx);
        }});
    }})();
    """

    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn", post_script=post_script)
    print(f"[info] {len(angulos_info)} ângulos, {n_pos_inicial} posições iniciais, {len(sills)} sills")
    print(f"-> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
