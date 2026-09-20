"""Dashboard de FUROS do PMP -- mesma estrutura do dashboard do Taió
(gerar_dashboard_geoquimico.py: lista pesquisável + tabelas-resumo + gráficos
Plotly sincronizados + mapa Leaflet, tema claro/escuro), só que com os dados
do PMP e SEM geoquímica (o PMP ainda não tem): é uma ferramenta de
investigação dos furos de sondagem que caem dentro do retângulo (area2.shp).

O que mostra:
  - Lista dos furos (busca por nome/município/agência/fonte/unidade); clicar
    numa linha voa até o furo no mapa e o destaca nos gráficos (e vice-versa:
    clicar no mapa ou num ponto do gráfico seleciona a linha).
  - Gráficos no meio: profundidade x cota da boca, distribuição de
    profundidade e de cota, furos por fonte, unidade mais profunda atingida,
    profundidade do topo de cada unidade, espessura das unidades medida em
    furo (com a literatura por cima) e espessura do corpo intrusivo.
  - Resumo dos corpos intrusivos: área (polígono CPRM), furos dentro e
    espessura MEDIDA em furo (a espessura de 50 m usada no cubo é assunção).

Fontes: ../2_Banco_de_Dados (area2/curvas3/Litologia_PMP2.shp + banco de
poços) via _comum_pmp.py. Só LÊ essas fontes.

Uso:
    python gerar_dashboard_pmp.py
Gera:
    dashboard_pmp.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shapely

from _comum_pmp import (
    MARCA_ROXO, MARCA_ROXO_ESCURO, MARCA_NAVY, MARCA_CINZA_CLARO, MARCA_FONTE,
    CORES_ESTILIZADO, ESPESSURA_ESTILIZADO_M, ESPESSURA_SILL_M, ORDEM_PROFUNDIDADE_FURO,
    COR_FURO_SEM_TOPO, ROTULO_FURO_SEM_TOPO, LEAFLET_LINKS, JS_BASEMAPS, logo_base64,
    gerar_hipsometria_leaflet, preparar_geologia_leaflet, preparar_contorno_area_leaflet, preparar_furos,
    carregar_litologia_pmp, SIGLAS_SILL_INDIVIDUALIZADO, _intervalos_corpo,
)

BASE = Path(__file__).resolve().parent
OUT_HTML = BASE / "dashboard_pmp.html"
COR_PAINEL = "#2A2F45"
COR_DESTAQUE_CORPO = "#E63946"
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{__import__('plotly').offline.get_plotlyjs_version()}.min.js"


def tema_grafico(fig, titulo, altura=320, nota=None):
    fig.update_layout(
        paper_bgcolor=MARCA_NAVY, plot_bgcolor=COR_PAINEL, height=altura + (24 if nota else 0),
        font=dict(family=MARCA_FONTE, color=MARCA_CINZA_CLARO, size=11),
        margin=dict(l=55, r=20, t=45, b=45 + (24 if nota else 0)),
        title=dict(text=f"<b>{titulo}</b>", x=0.02, font=dict(size=13, color=MARCA_CINZA_CLARO)),
        legend=dict(bgcolor="rgba(45,10,74,0.75)", bordercolor=MARCA_ROXO, borderwidth=1, font=dict(size=10)),
    )
    fig.update_xaxes(color=MARCA_CINZA_CLARO, gridcolor="#3a3f52", zerolinecolor="#3a3f52")
    fig.update_yaxes(color=MARCA_CINZA_CLARO, gridcolor="#3a3f52", zerolinecolor="#3a3f52")
    if nota:
        fig.add_annotation(text=nota, xref="paper", yref="paper", x=1, y=0, yshift=-52, showarrow=False,
                           font=dict(size=9, color=MARCA_CINZA_CLARO), opacity=0.6, xanchor="right", yanchor="top")
    return fig


def num(v, casas=0):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{casas}f}"


# ---------------------------------------------------------------- dados
def calcular_espessuras(f):
    """Espessura (m) de cada unidade medida no furo = diferença de Prof_topo
    entre a unidade e a IMEDIATAMENTE abaixo na coluna (só onde ambos os
    topos foram medidos). Rio Bonito termina no Itararé, ou no embasamento."""
    def dif(topo, base):
        d = f[base] - f[topo]
        return d[d > 0].dropna()
    rb = pd.concat([dif("Prof_topo_Fm_RioBonito", "Prof_topo_Gp_Itarare"),
                    dif("Prof_topo_Fm_RioBonito", "Prof_topo_Embasamento")]).groupby(level=0).first()
    return {
        "Estrada Nova": (dif("Prof_topo_sbGp_EstradaNova", "Prof_topo_Fm_Irati"),
                         ESPESSURA_ESTILIZADO_M["Fm_Teresina"] + ESPESSURA_ESTILIZADO_M["Fm_SerraAlta"], CORES_ESTILIZADO["Fm_Teresina"]),
        "Irati": (dif("Prof_topo_Fm_Irati", "Prof_topo_Fm_Palermo"), ESPESSURA_ESTILIZADO_M["Fm_Irati"], CORES_ESTILIZADO["Fm_Irati"]),
        "Palermo": (dif("Prof_topo_Fm_Palermo", "Prof_topo_Fm_RioBonito"), ESPESSURA_ESTILIZADO_M["Fm_Palermo"], "#9FB4C4"),
        "Rio Bonito": (rb, ESPESSURA_ESTILIZADO_M["Fm_RioBonito"], "#8A9AA3"),
    }


def linhas_estatistica(itens, casas=0):
    """itens: lista de (rotulo, Series) -> <tr> no estilo do Taió (faixa min-max
    com marcas de média e mediana)."""
    out = []
    for rot, s in itens:
        s = s.dropna()
        if len(s) == 0:
            continue
        mn, mx, me, md = float(s.min()), float(s.max()), float(s.mean()), float(s.median())
        faixa = mx - mn
        pm = (me - mn) / faixa * 100 if faixa else 50
        pd_ = (md - mn) / faixa * 100 if faixa else 50
        out.append(f"""<tr><td>{rot}</td><td>{len(s)}</td><td>{mn:.{casas}f}</td>
            <td><div class="faixa-elemento"><div class="marca-mediana" style="left:{pd_:.1f}%" title="Mediana: {md:.{casas}f}"></div>
            <div class="marca-media" style="left:{pm:.1f}%" title="Média: {me:.{casas}f}"></div></div></td>
            <td>{me:.{casas}f}</td><td>{md:.{casas}f}</td><td>{mx:.{casas}f}</td></tr>""")
    return "".join(out)


def resumo_corpos(f):
    """Uma linha por sill mapeado (polígono CPRM): área, furos dentro, furos
    dentro que interceptaram corpo, espessura medida (mediana/máx)."""
    lit = carregar_litologia_pmp()
    sills = lit[lit["SIGLA_UNID"].isin(SIGLAS_SILL_INDIVIDUALIZADO)]
    linhas = []
    dentro_algum = np.zeros(len(f), dtype=bool)
    for nome, g in sills.groupby("NOME_UNIDA"):
        geom = g.geometry.union_all()
        dentro = shapely.contains_xy(geom, f["E"].to_numpy(), f["N"].to_numpy())
        dentro_algum |= dentro
        sub = f[dentro]
        esp = sub["esp_corpo_m"].dropna()
        linhas.append((nome, geom.area / 1e6, len(g), int(dentro.sum()), len(esp),
                       float(esp.median()) if len(esp) else np.nan, float(esp.max()) if len(esp) else np.nan))
    fora = f[~dentro_algum]["esp_corpo_m"].dropna()
    html = "".join(
        f"<tr><td>{n}</td><td>{a:.1f} km²</td><td>{npol}</td><td>{nf}</td><td>{ne}</td><td>{num(md, 1)} m</td><td>{num(mx, 0)} m</td></tr>"
        for n, a, npol, nf, ne, md, mx in linhas)
    html += (f"<tr><td><i>Fora dos sills mapeados</i></td><td>—</td><td>—</td><td>{int((~dentro_algum).sum())}</td>"
             f"<td>{len(fora)}</td><td>{num(float(fora.median()) if len(fora) else np.nan, 1)} m</td>"
             f"<td>{num(float(fora.max()) if len(fora) else np.nan, 0)} m</td></tr>")
    return html


# ------------------------------------------------------------- gráficos
def montar_graficos(f):
    graficos = {}
    fh = f.copy()
    fh["hover"] = fh.apply(lambda r: f"<b>{r['nome']}</b><br>{r.get('Municipio') if pd.notna(r.get('Municipio')) else '—'}<br>{r['unidade_fundo']}", axis=1)

    # 0. coluna estratigráfica do(s) furo(s) selecionado(s) -- desenhada no navegador
    # (JS: desenharColuna); aqui só o quadro vazio com a mensagem inicial
    fig = go.Figure()
    fig.add_annotation(text="Selecione um furo (lista, mapa ou gráfico) ou clique numa barra de<br>"
                            "\"Furos por fonte\" para ver a coluna estratigráfica", x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color=MARCA_CINZA_CLARO), opacity=0.7)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    tema_grafico(fig, "Coluna estratigráfica", 480)
    fig.update_layout(height=None, autosize=True)
    graficos["coluna"] = fig

    # 1. profundidade x cota da boca (clicável -> sincroniza com lista/mapa)
    d = fh.dropna(subset=["Profundidade", "Cota_boca"])
    fig = go.Figure(go.Scatter(
        x=d["Cota_boca"], y=d["Profundidade"], mode="markers", name="Furos PMP", customdata=d["id"], text=d["hover"],
        hovertemplate="%{text}<br>Cota %{x:.0f} m · Prof. %{y:.0f} m<extra></extra>",
        marker=dict(size=8, color=d["cor"], line=dict(width=1, color=MARCA_NAVY)),
    ))
    fig.update_yaxes(title_text="Profundidade total (m)", autorange="reversed")
    fig.update_xaxes(title_text="Cota da boca (m)")
    graficos["prof-cota"] = tema_grafico(fig, "Profundidade × cota da boca", 340,
                                          f"{len(d)} furos com os dois valores · cor = unidade mais profunda atingida · clique p/ selecionar")

    # 2. distribuição da profundidade
    p = fh["Profundidade"].dropna()
    fig = go.Figure(go.Histogram(x=p, xbins=dict(size=25), marker=dict(color=MARCA_ROXO, line=dict(width=1, color=MARCA_NAVY)),
                                 hovertemplate="%{x} m: %{y} furos<extra></extra>", name="Profundidade"))
    fig.add_vline(x=float(p.median()), line=dict(color="#E8A33D", dash="dash"),
                  annotation_text=f"mediana {p.median():.0f} m", annotation_font=dict(color="#E8A33D", size=10))
    fig.update_xaxes(title_text="Profundidade total (m)"); fig.update_yaxes(title_text="Nº de furos")
    graficos["dist-prof"] = tema_grafico(fig, "Distribuição da profundidade dos furos", 280, f"{len(p)} furos com profundidade · faixas de 25 m")

    # 3. distribuição da cota da boca
    c = fh["Cota_boca"].dropna()
    fig = go.Figure(go.Histogram(x=c, xbins=dict(size=25), marker=dict(color=MARCA_AZUL_HEX, line=dict(width=1, color=MARCA_NAVY)),
                                 hovertemplate="%{x} m: %{y} furos<extra></extra>", name="Cota"))
    fig.update_xaxes(title_text="Cota da boca (m)"); fig.update_yaxes(title_text="Nº de furos")
    graficos["dist-cota"] = tema_grafico(fig, "Distribuição da cota da boca (altitude dos furos)", 280, f"{len(c)} furos · faixas de 25 m")

    # 4. furos por fonte
    fonte = fh["fonte_planilha"].value_counts().sort_values()
    fig = go.Figure(go.Bar(x=fonte.values, y=fonte.index, orientation="h", marker=dict(color=MARCA_ROXO),
                           hovertemplate="%{y}: %{x} furos<extra></extra>"))
    fig.update_xaxes(title_text="Nº de furos")
    graficos["fontes"] = tema_grafico(fig, "Furos por fonte (planilha/camada de origem)", max(280, 90 + 18 * len(fonte)),
                                      "clique numa barra: mostra todos os furos da fonte lado a lado na coluna")

    # 5. unidade mais profunda atingida
    ordem = [r for _, r, _ in ORDEM_PROFUNDIDADE_FURO] + [ROTULO_FURO_SEM_TOPO]
    cores = {r: c for _, r, c in ORDEM_PROFUNDIDADE_FURO}
    cores[ROTULO_FURO_SEM_TOPO] = COR_FURO_SEM_TOPO
    cont = fh["unidade_fundo"].value_counts()
    rot = [r for r in ordem if cont.get(r, 0) > 0]
    fig = go.Figure(go.Bar(x=rot, y=[int(cont[r]) for r in rot], marker=dict(color=[cores[r] for r in rot], line=dict(width=1, color=MARCA_NAVY)),
                           text=[int(cont[r]) for r in rot], textposition="outside", hovertemplate="%{x}: %{y} furos<extra></extra>"))
    fig.update_yaxes(title_text="Nº de furos")
    graficos["unidade-fundo"] = tema_grafico(fig, "Unidade mais profunda atingida pelo furo", 300,
                                             "topos medidos (Prof_topo_*) · clique numa barra p/ ver esses furos na coluna")

    # 6. profundidade do topo de cada unidade
    fig = go.Figure()
    n_topos = []
    for col, rotulo, cor in ORDEM_PROFUNDIDADE_FURO:
        s = fh[f"Prof_topo_{col}"].dropna()
        if len(s) < 3:
            continue
        n_topos.append((rotulo, s))
        fig.add_trace(go.Box(y=s, name=f"{rotulo.split(' (')[0]}<br>n={len(s)}", marker=dict(color=cor), line=dict(color=cor),
                             boxmean=True, boxpoints="outliers"))
    fig.update_yaxes(title_text="Profundidade do topo (m)", autorange="reversed")
    fig.update_layout(showlegend=False)
    graficos["topos"] = tema_grafico(fig, "Profundidade do topo de cada unidade nos furos", 340, "caixa = quartis · linha tracejada = média")

    # 7. espessuras medidas x literatura
    esp = calcular_espessuras(fh)
    fig = go.Figure()
    for rotulo, (s, lit, cor) in esp.items():
        if len(s) == 0:
            continue
        fig.add_trace(go.Box(y=s, name=f"{rotulo}<br>n={len(s)}", marker=dict(color=cor), line=dict(color=cor), boxmean=True, showlegend=False))
    fig.add_trace(go.Scatter(x=[f"{r}<br>n={len(s)}" for r, (s, _, _) in esp.items() if len(s)],
                             y=[l for _, (s, l, _) in esp.items() if len(s)], mode="markers", name="Literatura (ponto médio)",
                             marker=dict(symbol="diamond", size=12, color="#E8A33D", line=dict(width=1, color=MARCA_NAVY)),
                             hovertemplate="Literatura: %{y} m<extra></extra>"))
    fig.update_yaxes(title_text="Espessura (m)")
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    graficos["espessuras"] = tema_grafico(fig, "Espessura das unidades: furo × literatura", 340,
                                           "topos consecutivos · ◆ = literatura usada no modelo · furo mede ≤ real onde a unidade foi truncada pela erosão")

    # 8. espessura do corpo intrusivo (intervalo mais espesso por furo)
    e = fh["esp_corpo_m"].dropna()
    fig = go.Figure(go.Histogram(x=e, xbins=dict(start=0, size=5), marker=dict(color="#A63D2F", line=dict(width=1, color=MARCA_NAVY)),
                                 hovertemplate="%{x} m: %{y} furos<extra></extra>", name="Corpo intrusivo"))
    fig.add_vline(x=float(e.median()), line=dict(color="#E8A33D", dash="dash"),
                  annotation_text=f"mediana {e.median():.1f} m", annotation_font=dict(color="#E8A33D", size=10), annotation_position="top right")
    fig.add_vline(x=ESPESSURA_SILL_M, line=dict(color="#7B2FFF", dash="dot"),
                  annotation_text=f"assumido no cubo: {ESPESSURA_SILL_M:.0f} m", annotation_font=dict(color="#B98CFF", size=10), annotation_position="bottom right")
    fig.update_xaxes(title_text="Espessura do corpo intrusivo (m)"); fig.update_yaxes(title_text="Nº de furos")
    graficos["corpo"] = tema_grafico(fig, "Espessura do corpo intrusivo interceptado nos furos", 300,
                                      f"{len(e)} furos · intervalo mais espesso de cada furo (Prof_corpos_intrusivos_SG)")
    return graficos, n_topos, esp


MARCA_AZUL_HEX = "#2E6F95"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" href="assets/favicon.png">
<link rel="shortcut icon" href="assets/favicon.ico">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
<title>Dashboard de Furos — PMP</title>
@@LEAFLET@@
<script src="@@PLOTLY@@"></script>
<style>
  * { box-sizing: border-box; }
  :root { --bg: @@NAVY@@; --painel: @@PAINEL@@; --texto: @@CINZA@@; --borda-fraca: #3a3f52; --input-bg: @@NAVY@@; }
  body.tema-claro { --bg: @@CINZA@@; --painel: #FFFFFF; --texto: @@NAVY@@; --borda-fraca: #D8D8E2; --input-bg: #FFFFFF; }
  body { margin: 0; background: var(--bg); color: var(--texto); font-family: @@FONTE@@; transition: background 0.2s, color 0.2s; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 14px 24px; border-bottom: 1px solid @@ROXO@@; gap: 16px; }
  header h1 { font-size: 20px; margin: 0; }
  header h1 b { color: @@ROXO@@; }
  header .lado-direito { display: flex; align-items: center; gap: 12px; }
  header img { width: 52px; height: 52px; border-radius: 50%; border: 2px solid @@ROXO@@; box-shadow: 0 0 10px rgba(123,47,255,0.6); }
  .btn-tema { background: @@ROXO_ESCURO@@; color: @@CINZA@@; border: 1.5px solid @@ROXO@@; border-radius: 6px; padding: 7px 12px; font-family: @@FONTE@@; font-size: 12px; cursor: pointer; }
  body.tema-claro .btn-tema { background: #EDE3FF; color: @@NAVY@@; }
  .btn-tema.ativo { opacity: 1; font-weight: 700; }
  .btn-tema:not(.ativo) { opacity: 0.55; }
  .layout { display: grid; grid-template-columns: 320px 1fr 1fr; gap: 12px; padding: 12px; height: calc(100vh - 90px); min-height: 600px; }
  .col-esquerda { display: flex; flex-direction: column; gap: 12px; min-height: 0; height: 100%; }
  .painel { background: var(--painel); border: 1px solid @@ROXO@@; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }
  .col-esquerda > .painel:first-child { flex: 1; min-height: 0; }
  .painel-estatico { flex: 0 0 auto; }
  .tabela-resumo { width: calc(100% - 28px); border-collapse: collapse; font-size: 11px; margin: 8px 14px; }
  .tabela-resumo th { text-align: left; padding: 4px 6px; border-bottom: 1px solid @@ROXO@@; opacity: 0.75; font-weight: 500; }
  .tabela-resumo td { padding: 5px 6px; border-bottom: 1px solid var(--borda-fraca); vertical-align: middle; }
  .tabela-resumo tr:last-child td { border-bottom: none; }
  .nota-resumo { font-size: 10px; opacity: 0.55; margin: 0 14px 10px 14px; line-height: 1.4; }
  .subtitulo-estatistica { font-size: 11px; font-weight: 600; opacity: 0.75; margin: 10px 14px 0 14px; text-transform: uppercase; letter-spacing: 0.04em; }
  .faixa-elemento { position: relative; width: 100%; min-width: 70px; height: 10px; }
  .faixa-elemento::before { content: ''; position: absolute; top: 4px; left: 0; right: 0; height: 2px; background: var(--borda-fraca); border-radius: 1px; }
  .marca-media { position: absolute; top: 1px; width: 8px; height: 8px; border-radius: 50%; background: @@ROXO@@; transform: translateX(-50%); }
  .marca-mediana { position: absolute; top: -1px; width: 2px; height: 12px; background: var(--texto); opacity: 0.55; transform: translateX(-50%); }
  .painel h2 { font-size: 13px; margin: 0; padding: 10px 14px; border-bottom: 1px solid var(--borda-fraca); color: var(--texto); opacity: 0.85; text-transform: uppercase; letter-spacing: 0.05em; }
  #busca { margin: 10px 14px 0 14px; padding: 7px 10px; border-radius: 6px; border: 1px solid @@ROXO@@; background: var(--input-bg); color: var(--texto); font-family: @@FONTE@@; }
  .contagem { font-size: 11px; opacity: 0.6; padding: 6px 14px 0 14px; }
  .lista-scroll { overflow-y: auto; flex: 1; padding: 0 8px 8px 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { position: sticky; top: 0; background: var(--painel); text-align: left; padding: 6px 6px; border-bottom: 1px solid @@ROXO@@; opacity: 0.75; font-weight: 500; }
  td { padding: 6px 6px; border-bottom: 1px solid var(--borda-fraca); }
  .linha-dado { cursor: pointer; }
  .linha-dado:hover { background: rgba(123,47,255,0.15); }
  .linha-dado.selecionada { background: rgba(123,47,255,0.35); }
  .nome-ponto { display: inline-block; padding: 2px 7px; border-radius: 5px; border: 1px solid rgba(123,47,255,0.35); background: rgba(123,47,255,0.08); }
  .linha-dado.selecionada .nome-ponto { color: @@ROXO@@; font-weight: 700; border-color: @@ROXO@@; background: rgba(123,47,255,0.18); }
  body.tema-claro .linha-dado.selecionada .nome-ponto { color: #5A1FBF; border-color: #5A1FBF; }
  .dot-cor { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; border: 1px solid rgba(255,255,255,0.4); }
  .linha-corpo { border-left: 3px solid @@DESTAQUE@@; }
  footer { text-align: center; padding: 8px; opacity: 0.55; font-size: 11px; }
  #mapa-leaflet { flex: 1; }
  .col-graficos { overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 12px; }
  .grafico-card { background: var(--painel); border: 1px solid var(--borda-fraca); border-left: 4px solid @@ROXO@@; border-radius: 8px; flex-shrink: 0; }
  .card-coluna { display: flex; flex-direction: column; height: calc(100vh - 134px); min-height: 480px; }
  .coluna-ctrl { display: flex; align-items: center; gap: 8px; padding: 8px 12px 0 14px; font-size: 12px; }
  .btn-mini { background: transparent; color: var(--texto); border: 1px solid @@ROXO@@; border-radius: 5px; padding: 3px 9px; font-family: @@FONTE@@; font-size: 11px; cursor: pointer; opacity: 0.6; }
  .btn-mini.ativo { opacity: 1; background: @@ROXO@@; color: #fff; }
  #grupo-info { margin-left: auto; font-size: 11px; opacity: 0.85; }
  #grupo-info button { margin-left: 6px; }
  .col-plot { flex: 1; min-height: 0; overflow-x: auto; overflow-y: hidden; }
  .col-plot > div, .col-plot .plotly-graph-div { height: 100% !important; }
  .leaflet-popup-content-wrapper { background: @@ROXO_ESCURO@@; color: @@CINZA@@; border: 1px solid @@ROXO@@; }
  .leaflet-popup-tip { background: @@ROXO_ESCURO@@; }
  .leaflet-popup-content { font-family: @@FONTE@@; font-size: 12px; }
  .leaflet-control-layers { background: @@ROXO_ESCURO@@ !important; color: @@CINZA@@; border: 1px solid @@ROXO@@ !important; font-size: 12px; }
  .leaflet-control-layers-toggle { filter: invert(1); }
  .leaflet-bar a { background: @@ROXO_ESCURO@@; color: @@CINZA@@; border-bottom-color: @@ROXO@@ !important; }
  .leaflet-bar a:hover { background: @@ROXO@@; }
  .leg-furos { background: rgba(45,10,74,0.92); color: @@CINZA@@; border: 1px solid @@ROXO@@; border-radius: 6px; padding: 6px 9px; font-size: 11px; line-height: 1.5; }
  .leg-furos h4 { margin: 0 0 3px 0; font-size: 10px; color: @@ROXO@@; text-transform: uppercase; }
  .leg-furos i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; border: 1px solid rgba(255,255,255,0.5); }
  #popup-info { position: fixed; z-index: 2000; display: none; max-width: 330px; background: @@ROXO_ESCURO@@; color: @@CINZA@@; border: 1px solid @@ROXO@@; border-radius: 8px; padding: 10px 12px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.5); pointer-events: none; }
  #popup-info .linha-popup { margin: 2px 0; opacity: 0.92; }
  #popup-info .titulo-popup { font-size: 13px; font-weight: 700; color: @@ROXO@@; margin-bottom: 4px; }
</style>
</head>
<body>
<header>
  <h1><b>Dashboard de Furos</b> — PMP · Criciúma</h1>
  <div class="lado-direito">
    <button class="btn-tema ativo" id="btn-tema-escuro" onclick="aplicarTemaGeral('escuro')">Tema: Escuro</button>
    <button class="btn-tema" id="btn-tema-claro" onclick="aplicarTemaGeral('claro')">Tema: Claro</button>
    @@LOGO@@
  </div>
</header>
<div class="layout">
  <div class="col-esquerda">
    <div class="painel">
      <h2>Lista de furos</h2>
      <input id="busca" type="text" placeholder="Buscar por nome, município, agência, fonte...">
      <div class="contagem" id="contagem"></div>
      <div class="lista-scroll">
        <table>
          <thead><tr><th>Furo</th><th>Município</th><th>Prof.</th></tr></thead>
          <tbody id="corpo-tabela">@@TABELA@@</tbody>
        </table>
      </div>
    </div>
    <div class="painel painel-estatico">
      <h2>Corpos intrusivos — área &amp; espessura medida</h2>
      <table class="tabela-resumo">
        <thead><tr><th>Corpo</th><th>Área</th><th>Pol.</th><th>Furos</th><th>c/ corpo</th><th>Esp. med.</th><th>Máx</th></tr></thead>
        <tbody>@@RESUMO@@</tbody>
      </table>
      <p class="nota-resumo">Área do polígono CPRM · "Furos" = furos dentro do polígono · "c/ corpo" = os que interceptaram corpo intrusivo · espessura = intervalo mais espesso por furo (mediana e máx). O cubo assume @@ESP_SILL@@ m.</p>
    </div>
  </div>
  <div class="painel col-graficos">
@@CARDS@@
    <div class="grafico-card painel-estatico">
      <h2>Apanhado estatístico — topos e espessuras (furos)</h2>
      <p class="subtitulo-estatistica">Profundidade do topo da unidade (m)</p>
      <table class="tabela-resumo">
        <thead><tr><th>Unidade</th><th>n</th><th>Mín</th><th>Distribuição</th><th>Média</th><th>Mediana</th><th>Máx</th></tr></thead>
        <tbody>@@STAT_TOPOS@@</tbody>
      </table>
      <p class="subtitulo-estatistica">Espessura medida em furo (m)</p>
      <table class="tabela-resumo">
        <thead><tr><th>Unidade</th><th>n</th><th>Mín</th><th>Distribuição</th><th>Média</th><th>Mediana</th><th>Máx</th></tr></thead>
        <tbody>@@STAT_ESP@@</tbody>
      </table>
      <p class="nota-resumo">Faixa do mínimo ao máximo · <span style="color:@@ROXO@@">●</span> média · <span style="opacity:0.6">▏</span> mediana</p>
    </div>
  </div>
  <div class="painel">
    <h2>Mapa</h2>
    <div id="mapa-leaflet"></div>
  </div>
</div>
<div id="popup-info"></div>
<footer>GS Tech · PMP</footer>
<script>
(function() {
    var DADOS = @@DADOS@@;
    var DADOS_POR_ID = {};
    DADOS.forEach(function(r) { DADOS_POR_ID[r.id] = r; });

    var TODOS_GD = [@@GD_IDS@@].map(function(id) { return document.getElementById(id); });

    // clique num ponto do gráfico Profundidade x Cota sincroniza com lista/mapa
    TODOS_GD.forEach(function(gd) {
        if (!gd) return;
        gd.on('plotly_click', function(ev) {
            var pt = ev.points && ev.points[0];
            if (!pt) return;
            // barras de "Furos por fonte" / "Unidade mais profunda" -> todos os furos do grupo, lado a lado
            if (gd.id === 'grafico-fontes') { selecionarGrupo('fonte', String(pt.y)); return; }
            if (gd.id === 'grafico-unidade-fundo') { selecionarGrupo('unidade', String(pt.x)); return; }
            if (pt.customdata === undefined || pt.customdata === null) return;
            selecionarPorId(pt.customdata, gd.id === 'grafico-coluna' ? 'coluna' : 'grafico');
        });
    });

    // gráficos que têm a trace "Furos PMP" -- destaque do furo selecionado
    var GRAFICOS_PONTOS = [];
    TODOS_GD.forEach(function(gd) {
        if (!gd || !gd.data) return;
        var idx = gd.data.findIndex(function(tr) { return tr.name === 'Furos PMP'; });
        if (idx === -1) return;
        GRAFICOS_PONTOS.push({ gd: gd, idx: idx, customdata: (gd.data[idx].customdata || []).slice() });
    });
    function atualizarDestaqueGraficos(id) {
        GRAFICOS_PONTOS.forEach(function(g) {
            var n = g.customdata.length;
            var tam = new Array(n).fill(8), larg = new Array(n).fill(1), cor = new Array(n).fill('@@NAVY@@');
            var pos = g.customdata.indexOf(id);
            if (pos !== -1) { tam[pos] = 15; larg[pos] = 3; cor[pos] = '@@ROXO@@'; }
            Plotly.restyle(g.gd, { 'marker.size': [tam], 'marker.line.width': [larg], 'marker.line.color': [cor] }, [g.idx]);
        });
    }

    // tema claro/escuro -- moldura muda, cores dos dados ficam fixas
    var TEMA = {
        escuro: { paper: '@@NAVY@@', painel: '@@PAINEL@@', texto: '@@CINZA@@', grid: '#3a3f52', legendBg: 'rgba(45,10,74,0.75)' },
        claro:  { paper: '@@CINZA@@', painel: '#FFFFFF', texto: '@@NAVY@@', grid: '#D8D8E2', legendBg: 'rgba(255,255,255,0.85)' },
    };
    var temaAtual = 'escuro';
    function temaPlotly(gd, nome) {
        if (!gd || !gd.layout) return;
        var t = TEMA[nome];
        Plotly.relayout(gd, {
            paper_bgcolor: t.paper, plot_bgcolor: t.painel, 'font.color': t.texto,
            'xaxis.color': t.texto, 'xaxis.gridcolor': t.grid, 'xaxis.zerolinecolor': t.grid,
            'yaxis.color': t.texto, 'yaxis.gridcolor': t.grid, 'yaxis.zerolinecolor': t.grid,
            'legend.bgcolor': t.legendBg, 'legend.font.color': t.texto, 'title.font.color': t.texto,
        });
    }
    window.aplicarTemaGeral = function(nome) {
        document.body.classList.toggle('tema-claro', nome === 'claro');
        document.getElementById('btn-tema-escuro').classList.toggle('ativo', nome === 'escuro');
        document.getElementById('btn-tema-claro').classList.toggle('ativo', nome === 'claro');
        TODOS_GD.forEach(function(gd) { temaPlotly(gd, nome); });
        temaAtual = nome;
        desenharColuna();
    };

    // ---- coluna estratigráfica (1ª janela): 1 furo, ou todos os furos de um grupo lado a lado ----
    var modoColuna = { tipo: 'vazio' };     // {tipo:'furo', id} | {tipo:'grupo', ids:[...], titulo}
    var eixoColuna = 'prof';                // 'prof' (profundidade) | 'elev' (cota da boca - profundidade)
    var selecionadoId = null;
    var LARG_LITO = 0.62, DESL_CORPO = 0.42, LARG_CORPO = 0.2, COR_CORPO = '#A63D2F';

    function fmt0(v) { return Math.round(v).toString(); }
    window.mudarEixoColuna = function(eixo) {
        eixoColuna = eixo;
        document.getElementById('eixo-prof').classList.toggle('ativo', eixo === 'prof');
        document.getElementById('eixo-elev').classList.toggle('ativo', eixo === 'elev');
        desenharColuna();
    };

    function desenharColuna() {
        var gd = document.getElementById('grafico-coluna');
        if (!gd || modoColuna.tipo === 'vazio') return;
        var t = TEMA[temaAtual];
        var ids = modoColuna.tipo === 'furo' ? [modoColuna.id] : modoColuna.ids;
        var regs = ids.map(function(id) { return DADOS_POR_ID[id]; });
        var usaElev = eixoColuna === 'elev';
        var omitidos = 0;
        if (usaElev) {
            var comCota = regs.filter(function(r) { return r.cota !== null; });
            omitidos = regs.length - comCota.length;
            regs = comCota;
        }
        var traces = [], vistos = {}, legendaGrupos = {};
        var multi = regs.length > 1;
        var comTexto = regs.length <= 6;

        regs.forEach(function(r, i) {
            var c = r.coluna;
            var conv = usaElev ? function(d) { return r.cota - d; } : function(d) { return d; };
            var nUnid = 0;
            c.tops.forEach(function(tp, k) {
                var ini = tp.t;
                var fim = (k + 1 < c.tops.length) ? c.tops[k + 1].t : ((r.prof !== null && r.prof > ini) ? r.prof : null);
                if (fim === null || fim <= ini) return;
                nUnid++;
                var a = conv(ini), b = conv(fim);
                traces.push({
                    type: 'bar', x: [i], y: [b - a], base: [a], width: [LARG_LITO], name: tp.n,
                    legendgroup: tp.n, showlegend: !vistos[tp.n], customdata: [r.id],
                    marker: { color: tp.c, line: { color: t.texto, width: 0.6 } },
                    text: comTexto ? [tp.s + '<br>' + fmt0(ini) + '–' + fmt0(fim) + ' m'] : [''],
                    textposition: 'inside', insidetextanchor: 'middle', textfont: { color: tp.tc, size: 10 },
                    hovertemplate: '<b>' + r.nome + '</b><br>' + tp.n + '<br>' + fmt0(ini) + '–' + fmt0(fim) + ' m de profundidade'
                                   + (r.cota !== null ? ' (cota ' + fmt0(r.cota - ini) + ' → ' + fmt0(r.cota - fim) + ' m)' : '')
                                   + '<extra></extra>',
                });
                vistos[tp.n] = true;
            });
            if (nUnid === 0 && r.prof !== null) {   // furo sem topos medidos: coluna cinza "sem topos"
                var a0 = conv(0), b0 = conv(r.prof);
                traces.push({
                    type: 'bar', x: [i], y: [b0 - a0], base: [a0], width: [LARG_LITO], name: 'Sem topos medidos',
                    legendgroup: 'sem', showlegend: !vistos['sem'], customdata: [r.id],
                    marker: { color: '#6b6f80', line: { color: t.texto, width: 0.6 }, pattern: { shape: '/', fgcolor: '#999' } },
                    hovertemplate: '<b>' + r.nome + '</b><br>sem topos estratigráficos medidos<br>0–' + fmt0(r.prof) + ' m<extra></extra>',
                });
                vistos['sem'] = true;
            }
            c.corpos.forEach(function(cp) {
                var a = conv(cp[0]), b = conv(cp[1]);
                traces.push({
                    type: 'bar', x: [i + DESL_CORPO], y: [b - a], base: [a], width: [LARG_CORPO], name: 'Corpo intrusivo',
                    legendgroup: 'corpo', showlegend: !vistos['corpo'], customdata: [r.id],
                    marker: { color: COR_CORPO, line: { color: '#000', width: 0.6 } },
                    hovertemplate: '<b>' + r.nome + '</b><br>Corpo intrusivo ' + cp[0].toFixed(1) + '–' + cp[1].toFixed(1)
                                   + ' m (' + (cp[1] - cp[0]).toFixed(1) + ' m)<extra></extra>',
                });
                vistos['corpo'] = true;
            });
        });

        var n = regs.length;
        var shapes = [];
        regs.forEach(function(r, i) {   // moldura no furo selecionado
            if (r.id === selecionadoId && multi) {
                shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: i - 0.44, x1: i + 0.6, y0: 0, y1: 1, layer: 'below',
                              line: { color: '@@ROXO@@', width: 2 }, fillcolor: 'rgba(123,47,255,0.10)' });
            }
        });
        var titulo, sub;
        if (modoColuna.tipo === 'furo') {
            var r0 = DADOS_POR_ID[modoColuna.id];
            titulo = 'Coluna do furo ' + r0.nome;
            sub = [r0.municipio, r0.cota !== null ? 'cota ' + fmt0(r0.cota) + ' m' : null,
                   r0.prof !== null ? 'prof. ' + fmt0(r0.prof) + ' m' : null, 'fonte ' + r0.fonte].filter(Boolean).join(' · ');
        } else {
            titulo = modoColuna.titulo;
            sub = ids.length + ' furos · clique numa coluna para destacar' + (usaElev && omitidos ? ' · ' + omitidos + ' sem cota omitidos' : '');
        }
        var layout = {
            barmode: 'overlay', bargap: 0, paper_bgcolor: t.paper, plot_bgcolor: t.painel,
            font: { family: '@@FONTE@@', color: t.texto, size: 11 }, autosize: true,
            title: { text: '<b>' + titulo + '</b><br><span style="font-size:11px;opacity:.7">' + sub + '</span>', x: 0.02, font: { size: 13, color: t.texto } },
            margin: { l: 62, r: 20, t: 135, b: multi ? 95 : 50 },
            xaxis: { tickmode: 'array', tickvals: regs.map(function(r, i) { return i + 0.1; }), ticktext: regs.map(function(r) { return r.nome; }),
                     tickangle: multi ? -60 : 0, tickfont: { size: n > 14 ? 8 : 10 }, range: [-0.6, n - 0.25],
                     showgrid: false, zeroline: false, color: t.texto },
            yaxis: usaElev ? { title: 'Elevação (m)', gridcolor: t.grid, zerolinecolor: t.grid, color: t.texto }
                           : { title: 'Profundidade (m)', autorange: 'reversed', gridcolor: t.grid, zerolinecolor: t.grid, color: t.texto },
            legend: { orientation: 'h', y: 1.01, yanchor: 'bottom', x: 0, xanchor: 'left', bgcolor: 'rgba(0,0,0,0)', font: { size: 10, color: t.texto } },
            shapes: shapes,
        };
        if (n > 12) layout.width = n * 44 + 140;   // muitas colunas: largura fixa + rolagem horizontal
        Plotly.react(gd, traces, layout, { responsive: true });
    }

    // ---- grupos (clique numa barra de fonte/unidade) ----
    var filtroIds = null;   // Set de ids visíveis na lista, ou null
    var grupoLayer = null;
    function atualizarInfoGrupo() {
        var el = document.getElementById('grupo-info');
        el.innerHTML = filtroIds ? (modoColuna.titulo + ' <button class="btn-mini ativo" onclick="limparGrupo()">limpar ✕</button>') : '';
    }
    function selecionarGrupo(tipo, chave) {
        var pred = tipo === 'fonte' ? function(r) { return r.fonte === chave; } : function(r) { return r.unidade === chave; };
        var regs = DADOS.filter(pred);
        if (!regs.length) return;
        regs.sort(function(a, b) { return a.nome.localeCompare(b.nome, undefined, { numeric: true }); });
        filtroIds = new Set(regs.map(function(r) { return r.id; }));
        modoColuna = { tipo: 'grupo', ids: regs.map(function(r) { return r.id; }),
                       titulo: (tipo === 'fonte' ? 'Fonte ' : 'Unidade mais profunda: ') + chave };
        if (selecionadoId && !filtroIds.has(selecionadoId)) selecionadoId = null;
        desenharColuna();
        atualizarInfoGrupo();
        aplicarFiltro();
        if (grupoLayer) mapa.removeLayer(grupoLayer);
        grupoLayer = L.layerGroup(regs.map(function(r) {
            return L.circleMarker([r.lat, r.lon], { radius: 10, color: '@@ROXO@@', weight: 2.5, fillOpacity: 0, interactive: false });
        })).addTo(mapa);
        mapa.flyToBounds(L.latLngBounds(regs.map(function(r) { return [r.lat, r.lon]; })), { padding: [40, 40], maxZoom: 14, duration: 0.6 });
    }
    window.limparGrupo = function() {
        filtroIds = null;
        if (grupoLayer) { mapa.removeLayer(grupoLayer); grupoLayer = null; }
        modoColuna = selecionadoId ? { tipo: 'furo', id: selecionadoId } : { tipo: 'vazio' };
        if (modoColuna.tipo === 'furo') desenharColuna();
        atualizarInfoGrupo();
        aplicarFiltro();
    };

    // ---- mapa Leaflet ----
    var mapa = L.map('mapa-leaflet', { zoomControl: true });
@@BASEMAPS@@
    var hipsometria = L.imageOverlay('data:image/png;base64,@@HIPSO@@', @@BOUNDS@@, { opacity: 1 });
    satelite.addTo(mapa);
    mapa.fitBounds(@@BOUNDS@@);

    function pontoEstilo(cor, raio) {
        return { radius: raio, fillColor: cor, color: '@@NAVY@@', weight: 1.2, fillOpacity: 0.95 };
    }
    var furosLayer = L.geoJSON(@@FUROS_GEO@@, {
        pointToLayer: function(f, latlng) { return L.circleMarker(latlng, pontoEstilo(f.properties.cor, 6)); },
        onEachFeature: function(f, layer) {
            layer.bindPopup(f.properties.popup);
            layer.on('click', function() { selecionarPorId(f.properties.id, 'mapa'); });
        },
    }).addTo(mapa);
    var corpoLayer = L.geoJSON(@@FUROS_GEO@@, {
        filter: function(f) { return f.properties.corpo; },
        pointToLayer: function(f, latlng) { return L.circleMarker(latlng, { radius: 10, color: '@@DESTAQUE@@', weight: 2, fillOpacity: 0, interactive: false }); },
    });
    var formacoesLayer = L.geoJSON(@@GEO_FORMACOES@@, {
        style: function(f) { return { color: '#000', weight: 0.5, fillColor: f.properties.cor, fillOpacity: 0.5 }; },
        onEachFeature: function(f, layer) { layer.bindPopup(f.properties.popup); },
    });
    var sillsLayer = L.geoJSON(@@GEO_SILLS@@, {
        style: function(f) { return { color: '#000', weight: 1.2, fillColor: f.properties.cor, fillOpacity: 0.75 }; },
        onEachFeature: function(f, layer) { layer.bindPopup(f.properties.popup); },
    });
    var areaLayer = L.geoJSON(@@AREA@@, { style: function() { return { color: '@@ROXO@@', weight: 2.5, fill: false, dashArray: '8,5' }; } }).addTo(mapa);

    L.control.layers(
        { "Satélite (Esri)": satelite, "Escuro (CartoDB Dark)": escuro, "Rico (CartoDB Voyager)": rico, "Relevo/Topográfico": relevo,
          "OSM Padrão": osmPadrao, "Hipsometria (curvas3.shp)": hipsometria },
        { "Furos de sondagem": furosLayer, "Furos com corpo intrusivo": corpoLayer, "Mapa geológico (CPRM)": formacoesLayer,
          "Sills (corpos intrusivos)": sillsLayer, "Área de estudo": areaLayer },
        { collapsed: true }
    ).addTo(mapa);
    L.control.scale({ metric: true, imperial: false }).addTo(mapa);

    var legFuros = L.control({ position: 'bottomleft' });
    legFuros.onAdd = function() {
        var div = L.DomUtil.create('div', 'leg-furos');
        div.innerHTML = '<h4>Unidade mais profunda</h4>' + @@LEG_FUROS@@.map(function(it) {
            return '<div><i style="background:' + it[1] + '"></i>' + it[0] + '</div>';
        }).join('') + '<div><i style="background:transparent;border:2px solid @@DESTAQUE@@"></i>Interceptou corpo intrusivo</div>';
        return div;
    };
    legFuros.addTo(mapa);

    var destaqueMapa = L.circleMarker([0, 0], { radius: 14, color: '@@ROXO@@', weight: 3, fillOpacity: 0, opacity: 0 }).addTo(mapa);

    function selecionarPorId(id, origem) {
        document.querySelectorAll('.linha-dado.selecionada').forEach(function(el) { el.classList.remove('selecionada'); });
        var linha = document.querySelector('.linha-dado[data-id="' + id + '"]');
        if (linha) {
            linha.classList.add('selecionada');
            if (origem !== 'lista') linha.scrollIntoView({ block: 'nearest' });
        }
        atualizarDestaqueGraficos(id);
        var r = DADOS_POR_ID[id];
        if (!r) return;
        selecionadoId = id;
        // se o furo pertence ao grupo aberto, mantém as colunas lado a lado e só destaca; senão mostra o furo sozinho
        if (!(modoColuna.tipo === 'grupo' && filtroIds && filtroIds.has(id))) modoColuna = { tipo: 'furo', id: id };
        desenharColuna();
        atualizarInfoGrupo();
        destaqueMapa.setLatLng([r.lat, r.lon]);
        destaqueMapa.setStyle({ opacity: 1 });
        if (origem === 'init') return;
        if (origem !== 'mapa') mapa.flyTo([r.lat, r.lon], Math.max(mapa.getZoom(), 14), { duration: 0.6 });
    }
    document.getElementById('corpo-tabela').addEventListener('click', function(ev) {
        var linha = ev.target.closest('.linha-dado');
        if (linha) selecionarPorId(linha.getAttribute('data-id'), 'lista');
    });

    // popup com os detalhes do furo ao passar o mouse na linha da lista
    var popup = document.getElementById('popup-info');
    function montarPopup(r) {
        var l = ['<div class="titulo-popup">' + r.nome + '</div>'];
        if (r.municipio) l.push('<div class="linha-popup"><b>Município:</b> ' + r.municipio + (r.localidade ? ' · ' + r.localidade : '') + '</div>');
        if (r.agencia) l.push('<div class="linha-popup"><b>Agência:</b> ' + r.agencia + '</div>');
        if (r.projeto) l.push('<div class="linha-popup"><b>Projeto:</b> ' + r.projeto + '</div>');
        if (r.cota !== null) l.push('<div class="linha-popup"><b>Cota da boca:</b> ' + Math.round(r.cota) + ' m</div>');
        if (r.prof !== null) l.push('<div class="linha-popup"><b>Profundidade:</b> ' + Math.round(r.prof) + ' m</div>');
        l.push('<div class="linha-popup"><b>Unidade mais profunda:</b> ' + r.unidade + '</div>');
        if (r.topos) l.push('<div class="linha-popup"><b>Topos (prof.):</b> ' + r.topos + '</div>');
        if (r.corpo) l.push('<div class="linha-popup" style="color:#FF8A94"><b>Corpo intrusivo:</b> ' + r.corpo + '</div>');
        l.push('<div class="linha-popup" style="opacity:.6">UTM ' + Math.round(r.x) + ', ' + Math.round(r.y) + ' · fonte: ' + r.fonte + '</div>');
        return l.join('');
    }
    var tb = document.getElementById('corpo-tabela');
    tb.addEventListener('mouseover', function(ev) {
        var linha = ev.target.closest('.linha-dado'); if (!linha) return;
        var r = DADOS_POR_ID[linha.getAttribute('data-id')]; if (!r) return;
        popup.innerHTML = montarPopup(r); popup.style.display = 'block';
    });
    tb.addEventListener('mousemove', function(ev) {
        if (popup.style.display !== 'block') return;
        var x = ev.clientX + 16, y = ev.clientY + 12;
        if (x + 340 > window.innerWidth) x = ev.clientX - 346;
        if (y + 240 > window.innerHeight) y = ev.clientY - 230;
        popup.style.left = x + 'px'; popup.style.top = y + 'px';
    });
    tb.addEventListener('mouseout', function(ev) { if (ev.target.closest('.linha-dado')) popup.style.display = 'none'; });

    function aplicarFiltro() {
        var termo = document.getElementById('busca').value.toLowerCase(), vis = 0;
        document.querySelectorAll('.linha-dado').forEach(function(el) {
            var mostra = (!termo || el.getAttribute('data-busca').indexOf(termo) !== -1)
                         && (!filtroIds || filtroIds.has(el.getAttribute('data-id')));
            el.style.display = mostra ? '' : 'none';
            if (mostra) vis++;
        });
        document.getElementById('contagem').textContent = vis + ' de ' + DADOS.length + ' furos';
    }
    document.getElementById('busca').addEventListener('input', aplicarFiltro);
    aplicarFiltro();

    // abre já com um furo de exemplo (o com mais unidades medidas) pra coluna não começar vazia
    var inicial = DADOS.slice().sort(function(a, b) { return b.coluna.tops.length - a.coluna.tops.length; })[0];
    if (inicial) selecionarPorId(inicial.id, 'init');
})();
</script>
</body>
</html>
"""


def esc(s):
    return "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def main():
    print("Carregando dados...")
    f = preparar_furos()
    f = f[f["tipo"] == "Furo"].reset_index(drop=True)  # afloramentos (sem código de furo) ficam só no webmap
    print(f"  {len(f)} furos ({int((f['n_corpos'] > 0).sum())} com corpo intrusivo)")
    graficos, n_topos, esp = montar_graficos(f)

    # ---- registros por furo (lista/popup) e GeoJSON do mapa
    def topos_txt(r):
        partes = []
        for col, rot, _ in ORDEM_PROFUNDIDADE_FURO:
            v = r.get(f"Prof_topo_{col}")
            if pd.notna(v):
                partes.append(f"{rot.split(' (')[0]} {v:.0f}")
        return " · ".join(partes)

    def luminancia_txt(hexcor):
        h = hexcor.lstrip("#")
        r_, g_, b_ = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return "#1B1F2E" if (0.299 * r_ + 0.587 * g_ + 0.114 * b_) > 150 else "#F2F2F2"

    def coluna_furo(r):
        tops = []
        for col, rot, cor in ORDEM_PROFUNDIDADE_FURO:
            v = r.get(f"Prof_topo_{col}")
            if pd.notna(v):
                tops.append(dict(n=rot, s=rot.split(" (")[0], t=float(v), c=cor, tc=luminancia_txt(cor)))
        tops.sort(key=lambda x: x["t"])
        return dict(tops=tops, corpos=[[a, b] for a, b in _intervalos_corpo(r.get("Prof_corpos_intrusivos_SG"))])

    dados, feats, linhas = [], [], []
    for _, r in f.iterrows():
        nf = lambda v: None if pd.isna(v) else float(v)
        reg = dict(
            id=r["id"], nome=r["nome"], municipio=None if pd.isna(r.get("Municipio")) else str(r["Municipio"]),
            localidade=None if pd.isna(r.get("Localidade")) else str(r["Localidade"]),
            agencia=None if pd.isna(r.get("Agencia")) else str(r["Agencia"]),
            projeto=None if pd.isna(r.get("Projeto")) else str(r["Projeto"]),
            cota=nf(r["Cota_boca"]), prof=nf(r["Profundidade"]), unidade=r["unidade_fundo"], cor=r["cor"],
            topos=topos_txt(r), corpo=r["corpo_intervalos"] or None, fonte=str(r["fonte_planilha"]),
            x=float(r["E"]), y=float(r["N"]), lat=float(r["lat"]), lon=float(r["lon"]),
            coluna=coluna_furo(r),
        )
        dados.append(reg)
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [reg["lon"], reg["lat"]]},
                      "properties": {"id": reg["id"], "cor": reg["cor"], "corpo": bool(r["n_corpos"]),
                                     "popup": f"<b>{esc(reg['nome'])}</b><br>{esc(reg['unidade'])}"
                                              + (f"<br>Prof. {reg['prof']:.0f} m" if reg["prof"] is not None else "")}})
        busca = " ".join(esc(x) for x in [reg["nome"], reg["municipio"], reg["localidade"], reg["agencia"], reg["projeto"], reg["fonte"], reg["unidade"]] if x).lower()
        linhas.append(f"""<tr class="linha-dado{' linha-corpo' if r['n_corpos'] else ''}" data-id="{reg['id']}" data-busca="{busca}">
            <td><span class="dot-cor" style="background:{reg['cor']}"></span><span class="nome-ponto">{esc(reg['nome'])}</span></td>
            <td>{esc(reg['municipio']) or '—'}</td><td>{num(reg['prof'])}</td></tr>""")

    formacoes, sills, _ = preparar_geologia_leaflet()
    hipso_b64, bounds, _ = gerar_hipsometria_leaflet(resolucao=550)
    area = preparar_contorno_area_leaflet()

    ordem_leg = [(r, c) for _, r, c in ORDEM_PROFUNDIDADE_FURO] + [(ROTULO_FURO_SEM_TOPO, COR_FURO_SEM_TOPO)]
    presentes = set(f["unidade_fundo"])
    leg_furos = [[r, c] for r, c in ordem_leg if r in presentes]

    ids = list(graficos.keys())
    def card(k, fig):
        plot = pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id="grafico-" + k, config={"responsive": True})
        if k == "coluna":
            return f'''    <div class="grafico-card card-coluna">
      <div class="coluna-ctrl">
        <span>Eixo:</span>
        <button class="btn-mini ativo" id="eixo-prof" onclick="mudarEixoColuna('prof')">Profundidade</button>
        <button class="btn-mini" id="eixo-elev" onclick="mudarEixoColuna('elev')">Elevação</button>
        <span id="grupo-info"></span>
      </div>
      <div class="col-plot">{plot}</div>
    </div>'''
        return f'    <div class="grafico-card">{plot}</div>'
    cards = "\n".join(card(k, fig) for k, fig in graficos.items())
    j = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
    logo = logo_base64()

    html = TEMPLATE
    for token, valor in {
        "@@LEAFLET@@": LEAFLET_LINKS, "@@PLOTLY@@": PLOTLY_CDN, "@@BASEMAPS@@": JS_BASEMAPS,
        "@@FONTE@@": MARCA_FONTE, "@@NAVY@@": MARCA_NAVY, "@@ROXO@@": MARCA_ROXO, "@@ROXO_ESCURO@@": MARCA_ROXO_ESCURO,
        "@@CINZA@@": MARCA_CINZA_CLARO, "@@PAINEL@@": COR_PAINEL, "@@DESTAQUE@@": COR_DESTAQUE_CORPO,
        "@@LOGO@@": f'<img src="data:image/jpeg;base64,{logo}">' if logo else "",
        "@@TABELA@@": "".join(linhas), "@@RESUMO@@": resumo_corpos(f), "@@ESP_SILL@@": f"{ESPESSURA_SILL_M:.0f}",
        "@@CARDS@@": cards, "@@GD_IDS@@": ",".join(f'"grafico-{k}"' for k in ids),
        "@@STAT_TOPOS@@": linhas_estatistica(n_topos, 0),
        "@@STAT_ESP@@": linhas_estatistica([(k, v[0]) for k, v in esp.items()], 0),
        "@@DADOS@@": j(dados), "@@BOUNDS@@": j(bounds), "@@HIPSO@@": hipso_b64,
        "@@FUROS_GEO@@": j({"type": "FeatureCollection", "features": feats}),
        "@@GEO_FORMACOES@@": j(formacoes), "@@GEO_SILLS@@": j(sills), "@@AREA@@": j(area), "@@LEG_FUROS@@": j(leg_furos),
    }.items():
        html = html.replace(token, valor)
    assert "@@" not in html, "token sem substituir"
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"-> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
