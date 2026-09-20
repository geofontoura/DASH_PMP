"""Gera o cubo 3D SIMPLES do caso PMP -- rodada do retângulo
(area2.shp/curvas3.shp/Litologia_PMP2.shp, colocados pelo usuário direto em
2_Banco_de_Dados/), SEM NENHUM dado de poço.

Versão enxuta pedida pelo usuário ("gere apenas o modelo 3d simples, a
topografia, tire as funções por hora, só o modelo rodado"): topografia real
(curvas3.shp) + as 7 formações sedimentares + os 3 sills reais, TUDO sempre
visível, SEM nenhum controle interativo (sem toggle Superfícies/Sólidos, sem
toggle Hipsometria/Geologia, sem slider de opacidade, sem tema claro/escuro)
-- só o gráfico 3D pronto (o usuário ainda pode ligar/desligar cada camada
pela legenda nativa do Plotly, isso não é um "controle" customizado).

Mesma técnica de mergulho/azimute por mínimos quadrados ancorada no contato
real de afloramento (`ajustar_plano_formacao`/`calcular_planos_estilizados`
em _comum_pmp.py) -- agora as 7 formações têm contato REAL mapeado em
Litologia_PMP2.shp (a v1, Litologia_PMP.shp, só tinha 5; Fm_Teresina/P2t e
Fm_RioDoRasto/P23rr agora também aparecem no retângulo).

Cores: paleta CPRM oficial onde há bate exato e inequívoco de SIGLA_UNID na
biblioteca de estilo local (Fm_Taciba, Fm_Palermo, Fm_RioBonito) -- as outras
4 (Fm_Irati, Fm_SerraAlta, Fm_Teresina, Fm_RioDoRasto) mantêm a paleta de
terra anterior, porque a biblioteca só tem essas siglas com prefixo de
folha/região ambíguo (várias cores bem diferentes pro mesmo código, sem
metadado que diga qual é a nossa) -- ver nota em _comum_pmp.py.

Este é o script "irmão simples" de gerar_cubo_estilizado_pmp.py (que continua
no disco, com todos os controles, caso o usuário queira essa versão de
volta) -- os dois compartilham toda a lógica de dado via _comum_pmp.py.

Fontes: ../2_Banco_de_Dados/{area2,curvas3,Litologia_PMP2}.shp -- via
_comum_pmp.py. Este script só LÊ essas fontes.

Uso:
    python gerar_cubo_simples_pmp.py

Gera:
    cubo_simples_pmp.html
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from _comum_pmp import (
    MARCA_NAVY, MARCA_CINZA_CLARO, MARCA_FONTE,
    UNIDADES_ESTILIZADO, NOMES_ESTILIZADO, CORES_ESTILIZADO, ESPESSURA_SILL_M,
    logo_base64, carregar_area_pmp, carregar_vertices_curvas_pmp, carregar_litologia_pmp,
    carregar_sills_individualizados, construir_interpolador, avaliar_interpolador,
    construir_solido_poligono,
    pontos_dentro_poligono, calcular_planos_estilizados, calcular_contatos_estilizados,
    SIGLAS_POR_UNIDADE,
)

BASE = Path(__file__).resolve().parent
OUT_HTML = BASE / "cubo_pmp.html"

RESOLUCAO_GRID = 130
RAIO_MASCARA_TERRENO_KM = 3.0  # curvas3.shp ja e local/recortada -- mascara bem apertada


def montar_figura() -> go.Figure:
    poligono, bounds = carregar_area_pmp()
    e_min, n_min, e_max, n_max = bounds

    xt, yt, zt = carregar_vertices_curvas_pmp()
    interp_terreno = construir_interpolador(xt, yt, zt)

    grid_e, grid_n = np.meshgrid(
        np.linspace(e_min, e_max, RESOLUCAO_GRID), np.linspace(n_min, n_max, RESOLUCAO_GRID)
    )
    dentro_grade = pontos_dentro_poligono(grid_e, grid_n, poligono)
    terreno = avaliar_interpolador(interp_terreno, grid_e.ravel(), grid_n.ravel(),
                                    raio_mascara_km=RAIO_MASCARA_TERRENO_KM).reshape(grid_e.shape)
    terreno = np.where(dentro_grade, terreno, np.nan)
    terreno_medio = float(np.nanmean(terreno))
    print(f"[info] terreno: {int(dentro_grade.sum())}/{terreno.size} células dentro da área, "
          f"altitude média {terreno_medio:.0f}m")

    litologia = carregar_litologia_pmp()

    planos = calcular_planos_estilizados(litologia, interp_terreno)
    contatos = calcular_contatos_estilizados(planos, terreno, grid_e, grid_n)
    for unidade in UNIDADES_ESTILIZADO:
        a, b, _ = planos[unidade]
        mergulho = np.degrees(np.arctan(np.hypot(a, b)))
        real = "real (Litologia_PMP2)" if SIGLAS_POR_UNIDADE.get(unidade) else "estimado (offset de Fm_SerraAlta)"
        print(f"[info] {unidade}: mergulho {mergulho:.2f}° -- {real}")

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=grid_e, y=grid_n, z=terreno, colorscale="YlOrBr_r", showscale=False, opacity=1.0,
        name="Terreno real (hipsometria)", showlegend=True,
        hovertemplate="Terreno real<br>E=%{x:.0f} N=%{y:.0f} Alt=%{z:.0f}m<extra></extra>",
    ))

    for unidade in UNIDADES_ESTILIZADO:
        z = contatos[unidade]
        cor = CORES_ESTILIZADO[unidade]
        fig.add_trace(go.Surface(
            x=grid_e, y=grid_n, z=z, colorscale=[[0, cor], [1, cor]], showscale=False, opacity=1.0,
            name=NOMES_ESTILIZADO[unidade], showlegend=True,
            hovertemplate=f"{NOMES_ESTILIZADO[unidade]}<br>E=%{{x:.0f}} N=%{{y:.0f}} Alt=%{{z:.0f}}m<extra></extra>",
        ))

    # --- sills individualizados: poligono REAL (Litologia_PMP2) -- o
    # poligono de afloramento SIGNIFICA "o sill está exposto na superfície
    # aqui" -- ou seja, dentro dele o terreno real JÁ É a elevação do topo
    # do sill (por isso usar os contatos erodidos de Fm_SerraAlta/Teresina
    # dava espessura ZERO ali: os dois colapsavam pro mesmo terreno,
    # confirmado numericamente -- o sill sempre aflora exatamente onde a
    # erosão regional já cortou até esse nível). Desenha um SÓLIDO de
    # verdade (não só uma superfície fina): topo = terreno real, base =
    # terreno - ESPESSURA_SILL_M -- fica saliente/colorido em vermelho
    # exatamente na área de cobertura do polígono.
    sills = carregar_sills_individualizados()

    def topo_sill_fn(xs, ys):
        return avaliar_interpolador(interp_terreno, xs.ravel(), ys.ravel(), raio_mascara_km=RAIO_MASCARA_TERRENO_KM).reshape(xs.shape)

    def base_sill_fn(xs, ys):
        return topo_sill_fn(xs, ys) - ESPESSURA_SILL_M

    cor_sill = CORES_ESTILIZADO["Gp_SerraGeral"]
    for nome_sill, geom_sill in sills:
        trace_sill = construir_solido_poligono(
            geom_sill, topo_sill_fn, base_sill_fn, cor_sill, f"Sill — {nome_sill}", f"Sill {nome_sill}",
            resolucao=30,
        )
        trace_sill.visible = True  # construir_solido() vem com visible=False por padrão (pensado pro toggle Sólidos/Superfícies do cubo completo, que este script não tem)
        fig.add_trace(trace_sill)

    # exagero vertical calculado a partir do dado real (nao mais um fator fixo
    # 0.5 arbitrario) -- a area tem bastante relevo de verdade (terreno varia
    # ~30-1480m dentro do retangulo), entao um z=0.5 fixo distorcia MUITO
    # (~9x de exagero real); aqui calculo a razao real z/xy do proprio dado
    # (terreno + pilha de formacoes) e aplico so um exagero controlado
    # (EXAGERO_Z) por cima, pra continuar dando pra ver a separacao das
    # camadas sem parecer picos de montanha irreais.
    EXAGERO_Z = 2.5
    z_min = min(np.nanmin(terreno), min(np.nanmin(c) for c in contatos.values()))
    z_max = max(np.nanmax(terreno), max(np.nanmax(c) for c in contatos.values()))
    xy_range = max(e_max - e_min, n_max - n_min)
    aspecto_z = min((z_max - z_min) / xy_range * EXAGERO_Z, 1.0)
    print(f"[info] relevo real: {z_min:.0f}-{z_max:.0f}m -- aspectratio z={aspecto_z:.3f} (exagero {EXAGERO_Z:.1f}x)")

    fig.update_layout(
        title=dict(
            text="PMP — Cubo simples (topografia real + 7 formações + sills)<br>"
            f"<sub>Retângulo area2.shp ({(e_max-e_min)/1000:.0f}×{(n_max-n_min)/1000:.0f}km) · "
            f"mergulho ajustado por formação (contato real, Litologia_PMP2.shp)</sub>",
            font=dict(family=MARCA_FONTE, color=MARCA_NAVY, size=17),
        ),
        scene=dict(
            xaxis_title="E (UTM 22S)", yaxis_title="N (UTM 22S)", zaxis_title="Altitude (m)",
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=aspecto_z),
        ),
        paper_bgcolor=MARCA_CINZA_CLARO, font=dict(family=MARCA_FONTE),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=90, b=0),
    )

    logo = logo_base64()
    if logo:
        fig.add_layout_image(dict(
            source=f"data:image/jpeg;base64,{logo}", xref="paper", yref="paper", x=1.0, y=1.08,
            sizex=0.1, sizey=0.1, xanchor="right", yanchor="top",
        ))
    return fig


def main():
    fig = montar_figura()
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"-> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
