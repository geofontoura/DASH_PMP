"""Gera o mapa web 2D do caso PMP (retângulo area2.shp/curvas3.shp/
Litologia_PMP2.shp) -- SEM NENHUM dado de poço -- hipsometria real
(curvas3.shp) ou mapa geológico real (Litologia_PMP2.shp, com os 3 sills
mapeados em destaque) + contorno da área + escala/norte. Produto
independente, mais simples que o mapa embutido no modo interativo da seção
(sem linha de corte).

Fontes: ../2_Banco_de_Dados/{area2,curvas3,Litologia_PMP2}.shp -- via
_comum_pmp.py. Este script só LÊ essas fontes.

Uso:
    python gerar_mapa_web_pmp.py

Gera:
    mapa_web_pmp.html
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from _comum_pmp import (
    MARCA_ROXO, MARCA_NAVY, MARCA_CINZA_CLARO, MARCA_FONTE,
    NOMES_ESTILIZADO, CORES_ESTILIZADO, SIGLAS_SILL_INDIVIDUALIZADO,
    logo_base64, carregar_area_pmp, carregar_vertices_curvas_pmp, carregar_litologia_pmp,
    construir_interpolador, avaliar_interpolador, poligono_para_scatter_xy,
    pontos_dentro_poligono, botoes_tema, adicionar_escala_e_norte,
)

BASE = Path(__file__).resolve().parent
OUT_HTML = BASE / "mapa_web_pmp.html"

RESOLUCAO_MAPA = 160
RAIO_MASCARA_KM = 3.0


def main():
    poligono, bounds = carregar_area_pmp()
    e_min, n_min, e_max, n_max = bounds

    xt, yt, zt = carregar_vertices_curvas_pmp()
    interp_terreno = construir_interpolador(xt, yt, zt)

    import numpy as np
    mx, my = np.meshgrid(np.linspace(e_min, e_max, RESOLUCAO_MAPA), np.linspace(n_min, n_max, RESOLUCAO_MAPA))
    dentro_grade = pontos_dentro_poligono(mx, my, poligono)
    mz = avaliar_interpolador(interp_terreno, mx.ravel(), my.ravel(), raio_mascara_km=RAIO_MASCARA_KM).reshape(mx.shape)
    mz = np.where(dentro_grade, mz, np.nan)

    litologia = carregar_litologia_pmp()

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=mx[0, :], y=my[:, 0], z=mz, colorscale="YlOrBr_r", showscale=True,
        colorbar=dict(title="Altitude (m)", x=1.08), hoverinfo="none", name="Hipsometria",
    ))

    idx_geo_inicio = len(fig.data)
    cores_todas = dict(CORES_ESTILIZADO)
    nomes_todas = dict(NOMES_ESTILIZADO)
    for row in litologia.itertuples():
        unidade = row.unidade_padrao
        nome_legenda = nomes_todas.get(unidade, row.SIGLA_UNID) if pd.notna(unidade) else row.NOME_UNIDA
        cor = cores_todas.get(unidade, "#CCCCCC") if pd.notna(unidade) else "#CCCCCC"
        eh_sill = row.SIGLA_UNID in SIGLAS_SILL_INDIVIDUALIZADO
        if eh_sill:
            nome_legenda = f"Sill — {row.NOME_UNIDA}"
            cor = CORES_ESTILIZADO["Gp_SerraGeral"]
        gx, gy = poligono_para_scatter_xy(row.geometry)
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="lines", line=dict(width=1.5 if eh_sill else 0, color=MARCA_NAVY),
            fill="toself", fillcolor=cor, opacity=0.9 if eh_sill else 0.75,
            name=nome_legenda, showlegend=eh_sill, visible=False,
            hovertemplate=f"{nome_legenda}<br>{row.NOME_UNIDA}<extra></extra>",
        ))
    idx_geo_fim = len(fig.data) - 1
    n_geo = idx_geo_fim - idx_geo_inicio + 1

    adicionar_escala_e_norte(fig, e_min, e_max, n_min, n_max, comprimento_km=5.0)

    botoes_cor = [
        dict(label="Hipsometria", method="restyle", args=[{"visible": [True] + [False] * n_geo}, [0] + list(range(idx_geo_inicio, idx_geo_fim + 1))]),
        dict(label="Geologia real (com sills)", method="restyle", args=[{"visible": [False] + [True] * n_geo}, [0] + list(range(idx_geo_inicio, idx_geo_fim + 1))]),
    ]

    fig.update_layout(
        title=dict(text="PMP — Mapa web (hipsometria / geologia real)", font=dict(family=MARCA_FONTE, color=MARCA_NAVY, size=18)),
        xaxis=dict(range=[e_min, e_max], scaleanchor="y", scaleratio=1, showticklabels=False, constrain="domain"),
        yaxis=dict(range=[n_min, n_max], showticklabels=False),
        paper_bgcolor=MARCA_CINZA_CLARO, plot_bgcolor="white", font=dict(family=MARCA_FONTE),
        legend=dict(x=1.12, y=0.9),
        margin=dict(l=20, r=160, t=90, b=20), height=800,
        updatemenus=[
            dict(buttons=botoes_cor, direction="down", x=0.0, xanchor="left", y=1.08, yanchor="top",
                 bgcolor=MARCA_ROXO, font=dict(color="white")),
            dict(buttons=botoes_tema(eixos_2d=["xaxis", "yaxis"]), direction="down", x=0.22, xanchor="left", y=1.08, yanchor="top",
                 bgcolor="#4A4A4A", font=dict(color="white")),
        ],
    )

    logo = logo_base64()
    if logo:
        fig.add_layout_image(dict(
            source=f"data:image/jpeg;base64,{logo}", xref="paper", yref="paper", x=1.0, y=1.12,
            sizex=0.08, sizey=0.08, xanchor="right", yanchor="top",
        ))

    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"[info] {len(litologia)} polígonos geológicos ({litologia['SIGLA_UNID'].isin(SIGLAS_SILL_INDIVIDUALIZADO).sum()} sills)")
    print(f"-> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
