"""Módulo compartilhado pelos visualizadores web do caso PMP
(gerar_visualizador_3d_pmp.py, gerar_secao_interativa_pmp.py) -- paleta/
identidade visual, unidades estratigráficas, e as funções de carregar/
interpolar terreno real (curva de nível 10m, CURVAS.shp) e classificar mapa
geológico real (litoestratigrafia_500_sgb.shp), pra não duplicar entre os
scripts. Pensado pra ser reaproveitável em futuros casos de estudo: só
precisa trocar CURVAS_SHP/GEOLOGIA_SHP/MAPA_SGB_PARA_UNIDADE e as constantes
de unidade pro novo projeto.

Este módulo só LÊ as fontes de dado, nunca escreve nelas.
"""
import base64
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import Point

BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / ".cache"
PONTOS_UNIFICADOS_DIR = BASE.parent / "2_Banco_de_Dados" / "pontos_unificados"
CSV_PONTOS = PONTOS_UNIFICADOS_DIR / "pontos_unificados_pmp.csv"
CURVAS_SHP = PONTOS_UNIFICADOS_DIR / "CURVAS.shp"
GEOLOGIA_SHP = BASE.parent / "My Project_PMP" / "campo_criciuma" / "Dados" / "litoestratigrafia_500_sgb.shp"

# segunda rodada de dados, "limpos", colocados pelo usuário direto em
# 2_Banco_de_Dados/ (nao em pontos_unificados/) -- area MENOR (~26x47km,
# poligonos nomeados: Montanhao, Nova Veneza, Criciuma sul, Maracaja, Rio
# maior, Uru1, Uru2), curva de nivel ja recortada pra essa area, e um mapa
# litologico real (CPRM/SGB) com os 2 corpos de sill JA MAPEADOS como
# poligonos individuais (K1sg_delta_m "Montanhao", K1sg_delta_nv "Nova
# Veneza") -- nao precisa mais interpolar o sill de poço esparso aqui.
# Substitui AREA RECORTE.shp (apagado pelo usuário -- essa é a área de
# trabalho atual do projeto agora, menor e mais limpa que a antiga).
# terceira rodada -- usuário trouxe um retângulo simples (area2.shp) pra
# substituir a área irregular anterior (area.shp, os 7 polígonos nomeados),
# com litologia (Litologia_PMP2.shp) e curva de nível (curvas3.shp) já
# recortadas pro retângulo -- mais fácil de garantir cobertura de contato
# real das 7 formações (agora todas afloram dentro do retângulo, inclusive
# Fm_Teresina/P2t e Fm_RioDoRasto/P23rr, que não apareciam em Litologia_PMP.shp)
# e 3 sills mapeados (Montanhão, Nova Veneza, Urussanga). area.shp/curvas.shp/
# Litologia_PMP.shp ficam no disco só de referência, não usados mais.
AREA_PMP_SHP = PONTOS_UNIFICADOS_DIR.parent / "area2.shp"
AREA_RECORTE_SHP = AREA_PMP_SHP  # alias -- carregar_area_recorte() usada pelos scripts antigos
CURVAS_PMP_SHP = PONTOS_UNIFICADOS_DIR.parent / "curvas3.shp"
LITOLOGIA_PMP_SHP = PONTOS_UNIFICADOS_DIR.parent / "Litologia_PMP2.shp"

# bbox real de cobertura da curva de nível 10m (CURVAS.shp, EPSG:31982)
CURVAS_E_MIN, CURVAS_E_MAX = 437_944, 799_189
CURVAS_N_MIN, CURVAS_N_MAX = 6_788_168, 7_079_463
DESBASTE_CURVAS_M = 400.0  # grade de desbaste dos vértices antes de triangular (~62M -> ~570k)
RAIO_MASCARA_TERRENO_KM = 6.0  # curva é densa -- raio de mascara bem mais apertado que o dos poços

# bbox valido pra area regional do PMP (poços) -- ver unificar_pocos_pmp.py/NOTAS_UNIFICACAO.md
E_MIN, E_MAX = 400_000, 780_000
N_MIN, N_MAX = 6_600_000, 7_060_000

# identidade visual GS Tech (mesma paleta usada nos produtos do Taió)
MARCA_ROXO_ESCURO = "#2D0A4A"
MARCA_ROXO = "#7B2FFF"
MARCA_AZUL = "#2E6F95"
MARCA_NAVY = "#1B1F2E"
MARCA_CINZA_CLARO = "#F2F2F2"
MARCA_FONTE = "Montserrat, Arial, sans-serif"

LOGO_PATH = BASE / "assets" / "logo_gstech.jpg"

# unidades estratigraficas, da mais nova (topo) pra mais antiga (base).
# Gp_SerraGeral (corpo intrusivo/sill) NAO fica logo abaixo do Cenozoico --
# inspecionando Prof_corpos_intrusivos_SG (407 poços, 895 interseccoes de
# sill) contra os topos das outras unidades: o sill fica SEMPRE abaixo do
# Cenozoico (0% acima), majoritariamente ACIMA de Palermo/Rio Bonito
# (70%/64%) e por volta de Irati/Estrada Nova (48%/15% acima) -- ou seja,
# concentra perto do contato Estrada Nova/Irati, nao no topo da coluna.
# Reposicionado aqui pra bater com o dado real (ver enriquecer_sill abaixo).
UNIDADES = [
    "Cenozoico", "Fm_Botucatu", "Fm_Piramboia",
    "Fm_RosarioDoSul", "Fm_RioDoRasto", "sbGp_EstradaNova", "Gp_SerraGeral",
    "Fm_Irati", "Fm_Palermo", "Fm_RioBonito", "Gp_Itarare", "Gp_Parana", "Gp_RioIvai",
    "Embasamento",
]
NOMES_LEGIVEIS = {
    "Cenozoico": "Cenozoico", "Gp_SerraGeral": "Gp. Serra Geral (sill/diabásio)",
    "Fm_Botucatu": "Fm. Botucatu", "Fm_Piramboia": "Fm. Piramboia",
    "Fm_RosarioDoSul": "Fm. Rosário do Sul", "Fm_RioDoRasto": "Fm. Rio do Rasto",
    "sbGp_EstradaNova": "Subgp. Estrada Nova", "Fm_Irati": "Fm. Irati",
    "Fm_Palermo": "Fm. Palermo", "Fm_RioBonito": "Fm. Rio Bonito",
    "Gp_Itarare": "Gp. Itararé", "Gp_Parana": "Gp. Paraná",
    "Gp_RioIvai": "Gp. Rio Ivaí", "Embasamento": "Embasamento",
}
# tons de terra, jovem -> antigo; Serra Geral (igneo) destacado em
# vermelho-marrom (mesmo espirito da cor do sill no Taio, #A63D2F)
CORES_UNIDADES = {
    "Cenozoico": "#D9CB82", "Gp_SerraGeral": "#A63D2F", "Fm_Botucatu": "#E0B074",
    "Fm_Piramboia": "#C9915B", "Fm_RosarioDoSul": "#B9865F", "Fm_RioDoRasto": "#A87355",
    "sbGp_EstradaNova": "#8C8C86", "Fm_Irati": "#3E362C", "Fm_Palermo": "#B5AE93",
    "Fm_RioBonito": "#C9A66B", "Gp_Itarare": "#6E6455", "Gp_Parana": "#554B3D",
    "Gp_RioIvai": "#413A2F", "Embasamento": "#2B2620",
}

# mapa geologico real (CPRM/SGB, litoestratigrafia_500_sgb.shp) -> nossas
# unidades (algumas unidades do SGB nao existem separadas na planilha de
# pocos e caem na unidade combinada mais proxima -- Taciba e membro do
# Itarare; Serra Alta e Teresina sao os 2 membros do subgrupo Estrada Nova)
MAPA_SGB_PARA_UNIDADE = {
    "Fm. Taciba": "Gp_Itarare",
    "Granitoides neoproterozoicos": "Embasamento",
    "Fm. Rio Bonito": "Fm_RioBonito",
    "Fm. Palermo": "Fm_Palermo",
    "Fm. Serra Alta": "sbGp_EstradaNova",
    "Fm. Irati": "Fm_Irati",
    "Fm. Botucatu": "Fm_Botucatu",
    "Gp. Serra Geral (vulcânicas)": "Gp_SerraGeral",
    "Fm. Teresina": "sbGp_EstradaNova",
    "Fm. Rio do Rasto": "Fm_RioDoRasto",
    "Depósitos cenozoicos": "Cenozoico",
}


def logo_base64():
    if not LOGO_PATH.exists():
        return None
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")


def enriquecer_sill(df):
    """`Alt_topo_Gp_SerraGeral` (do xlsx) vem quase vazia (31 poços
    regionais, 1 só dentro do recorte) -- mas existe uma fonte MELHOR:
    `Prof_corpos_intrusivos_SG`, intervalos de profundidade "(topo,base)"
    (às vezes vários por poço) onde o poço atravessou corpo intrusivo do
    Grupo Serra Geral -- 407 poços regionais, 172 dentro do recorte. Usa o
    intervalo MAIS ESPESSO de cada poço (o "corpo principal" quando há
    mais de um) pra calcular topo/base reais (elevação = Cota_boca -
    profundidade), sobrescrevendo Alt_topo_Gp_SerraGeral onde disponível
    (mantém o valor original só nos poços sem essa coluna) e criando
    `Alt_base_Gp_SerraGeral` -- a base REAL medida do corpo, pra dar
    espessura verdadeira ao sólido/banda do sill em vez de usar o topo da
    próxima unidade (que não é a base real de um corpo intrusivo)."""
    import re

    def parse_intervalos(s):
        if pd.isna(s):
            return []
        pares = re.findall(r"\(([\d.]+)\s*,\s*([\d.]+)\)", str(s))
        return [(float(a), float(b)) for a, b in pares]

    def mais_espesso(intervalos):
        if not intervalos:
            return None
        return max(intervalos, key=lambda par: par[1] - par[0])

    melhor = df["Prof_corpos_intrusivos_SG"].apply(parse_intervalos).apply(mais_espesso)
    tem_sill = melhor.notna()

    df = df.copy()
    alt_topo_sill = pd.Series(np.nan, index=df.index)
    alt_base_sill = pd.Series(np.nan, index=df.index)
    alt_topo_sill[tem_sill] = df.loc[tem_sill, "Cota_boca"] - melhor[tem_sill].apply(lambda par: par[0])
    alt_base_sill[tem_sill] = df.loc[tem_sill, "Cota_boca"] - melhor[tem_sill].apply(lambda par: par[1])

    if "Alt_topo_Gp_SerraGeral" in df.columns:
        df["Alt_topo_Gp_SerraGeral"] = alt_topo_sill.combine_first(df["Alt_topo_Gp_SerraGeral"])
    else:
        df["Alt_topo_Gp_SerraGeral"] = alt_topo_sill
    df["Alt_base_Gp_SerraGeral"] = alt_base_sill
    return df


# --- pilha (young->old) pra area.shp/curvas.shp/Litologia_PMP.shp ---
# 5 das 7 formações sedimentares JÁ TÊM contato real mapeado em
# Litologia_PMP.shp (onde afloram, a elevação real do terreno NAQUELE
# polígono É a elevação real do topo da formação ali) -- ajustar_plano_
# formacao() usa esses pontos (vértices do próprio polígono de cada
# formação + elevação real via curvas.shp) pra achar o plano de melhor
# ajuste por MÍNIMOS QUADRADOS, dando mergulho/azimute REAIS (não mais
# estimados) por formação. Conferido: mergulho saiu 0,5-1,07°, batendo
# com a faixa "~0,5-1°" que o usuário passou da literatura -- valida a
# técnica. RMSE 5-44m (bem mais apertado que o antigo ajuste regional só
# com poço, que dava RMSE ~250-300m).
# Fm_Teresina e Fm_RioDoRasto NÃO aparecem em Litologia_PMP.shp nessa área
# (não afloram aqui) -- sem contato real, usam o MESMO mergulho/azimute do
# plano de Fm_SerraAlta (a formação real mais próxima/jovem disponível),
# só deslocados pela espessura típica (literatura) -- ver
# planos_estilizados() em gerar_cubo_estilizado_pmp.py.
UNIDADES_ESTILIZADO = [
    "Fm_RioDoRasto", "Fm_Teresina", "Fm_SerraAlta",
    "Fm_Irati", "Fm_Palermo", "Fm_RioBonito", "Fm_Taciba",
]
NOMES_ESTILIZADO = {
    "Fm_RioDoRasto": "Fm. Rio do Rasto", "Fm_Teresina": "Fm. Teresina",
    "Gp_SerraGeral": "Gp. Serra Geral (sill/diabásio)", "Fm_SerraAlta": "Fm. Serra Alta",
    "Fm_Irati": "Fm. Irati", "Fm_Palermo": "Fm. Palermo",
    "Fm_RioBonito": "Fm. Rio Bonito", "Fm_Taciba": "Fm. Taciba",
}
# espessura tipica (m, literatura -- so usada pra deslocar Teresina/RioDoRasto,
# que nao tem contato real mapeado aqui) e SIGLA_UNID (Litologia_PMP.shp) de
# quem TEM contato real -- None = sem poligono nessa area, usa so espessura
ESPESSURA_ESTILIZADO_M = {
    "Fm_RioDoRasto": 215.0, "Fm_Teresina": 325.0, "Fm_SerraAlta": 85.0,
    "Fm_Irati": 55.0, "Fm_Palermo": 95.0, "Fm_RioBonito": 175.0, "Fm_Taciba": 225.0,
}
SIGLAS_POR_UNIDADE = {
    "Fm_SerraAlta": ["P2sa"], "Fm_Irati": ["P1i"], "Fm_Palermo": ["P1p"],
    "Fm_RioBonito": ["P1rb"], "Fm_Taciba": ["C2P1t"],
    "Fm_Teresina": ["P2t"], "Fm_RioDoRasto": ["P23rr"],
}
# cores CPRM oficiais -- extraídas do arquivo de estilo ArcGIS Pro local
# (My Project_PMP/campo_criciuma/Projeto/CPRM_unidades-litoestratigraficas.stylx,
# um SQLite com os símbolos do SGB) buscando por SIGLA_UNID EXATA na tabela
# ITEMS (CLASS=5 = símbolo de preenchimento poligonal). Só 3 das 7 formações
# tiveram bate exato e inequívoco (Fm_Taciba/C2P1t, Fm_Palermo/P1p,
# Fm_RioBonito/P1rb) -- as outras 4 (P1i, P2sa, P2t, P23rr) só aparecem na
# biblioteca com prefixos de folha/região (ex.: "C2P1i", "NP1i", "A4PP2sa"...)
# e cada prefixo tem uma cor BEM diferente, sem metadado (CATEGORY/TAGS
# vazios) que diga qual corresponde à Bacia do Paraná/SC -- mantém a paleta
# de terra anterior pra essas 4 pra não arriscar cor errada.
CORES_ESTILIZADO = {
    "Fm_RioDoRasto": "#A87355", "Fm_Teresina": "#D6C79A", "Gp_SerraGeral": "#A63D2F",
    "Fm_SerraAlta": "#8C8C86", "Fm_Irati": "#3E362C",
    "Fm_Palermo": "#D6E0E6",   # CPRM P1p exato (214,224,230)
    "Fm_RioBonito": "#ADB6BB",  # CPRM P1rb exato (173,182,187)
    "Fm_Taciba": "#E9E2DE",     # CPRM C2P1t exato (233,226,222)
}
ESPESSURA_SILL_M = 50.0  # corpos K1sg_delta_* nao tem espessura no shp (so poligono em planta) -- ASSUNCAO

# mapa SIGLA_UNID (Litologia_PMP2.shp) -> nossa unidade estilizada -- agora
# as 7 formações têm polígono real (P2t/Teresina e P23rr/Rio do Rasto não
# apareciam em Litologia_PMP.shp v1, aparecem aqui) + 3 sills mapeados
MAPA_LITOLOGIA_PMP_PARA_UNIDADE = {
    "C2P1t": "Fm_Taciba", "P1p": "Fm_Palermo", "P1i": "Fm_Irati",
    "P2sa": "Fm_SerraAlta", "P1rb": "Fm_RioBonito",
    "P2t": "Fm_Teresina", "P23rr": "Fm_RioDoRasto",
    "K1sg": "Gp_SerraGeral", "K1sg_delta_m": "Gp_SerraGeral",
    "K1sg_delta_nv": "Gp_SerraGeral", "K1sg_delta_u": "Gp_SerraGeral",
}
SIGLAS_SILL_INDIVIDUALIZADO = ["K1sg_delta_m", "K1sg_delta_nv", "K1sg_delta_u"]  # "Corpo" na HIERARQUIA -- sills mapeados (Montanhão, Nova Veneza, Urussanga)


def carregar_furos_recorte():
    """Le pontos_unificados_pmp.csv (banco de poços/sondagens regional) e
    filtra só os que caem DENTRO do retângulo area2.shp -- 304 furos, bem
    mais denso e mais direto (mede a elevação do topo de verdade, sem
    depender da suposição "terreno no afloramento = topo da formação") do
    que os poucos polígonos de afloramento pra Fm_Teresina/Fm_RioDoRasto.
    Usado por ajustar_plano_formacao pra estabilizar o ajuste."""
    poligono, _ = carregar_area_pmp()
    df = pd.read_csv(CSV_PONTOS)
    # algumas colunas Alt_topo_* vêm como string (pandas infere isso quando a
    # coluna tem NaN misturado com valor -- não é erro de dado, só precisa
    # forçar numérico antes de usar em lstsq)
    for col in df.columns:
        if col.startswith("Alt_topo_") or col in ("E", "N"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col.startswith("Prof_topo_") or col in ("Cota_boca", "Profundidade"):
            # texto tipo "165m"/"299,76" -- extrai o número (mesmo tratamento do dashboard)
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False).str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
    dentro = pontos_dentro_poligono(df["E"].to_numpy(), df["N"].to_numpy(), poligono)
    return df[dentro].reset_index(drop=True)


# coluna Alt_topo_* do banco de poços regional (schema mais antigo/mais
# genérico, unidades agrupadas em subgrupo pra Estrada Nova/Itararé) que
# corresponde ao TOPO de cada formação estilizada -- None = sem coluna
# equivalente no banco de poços, só outcrop mesmo. sbGp_EstradaNova e
# Gp_Itarare são exatos (não aproximados): o topo do subgrupo/grupo É o topo
# da sua unidade mais jovem, que na coluna estratigráfica local é
# exatamente Fm_Teresina (topo de sbGp_EstradaNova) e Fm_Taciba (topo de
# Gp_Itarare) -- ver DICIONARIO_DE_DADOS_PMP.md / NOTAS_UNIFICACAO.md.
MAPA_ALT_TOPO_POR_UNIDADE = {
    "Fm_Irati": "Alt_topo_Fm_Irati",
    "Fm_Palermo": "Alt_topo_Fm_Palermo",
    "Fm_RioBonito": "Alt_topo_Fm_RioBonito",
    "Fm_Teresina": "Alt_topo_sbGp_EstradaNova",
    "Fm_Taciba": "Alt_topo_Gp_Itarare",
    "Fm_SerraAlta": None,
    "Fm_RioDoRasto": None,  # coluna existe no schema mas 0 furos com valor dentro do recorte
}


MIN_FUROS_PRIORIDADE = 5  # com >= isso, usa SO furo (mais direto/confiavel que outcrop)


def coletar_pontos_formacao(siglas, litologia_gdf, interp_terreno, furos=None, coluna_alt_topo=None,
                            raio_mascara_km=3.0):
    """Devolve (xs, ys, zs, fonte, n) -- os pontos reais disponíveis pra uma
    formação, da MELHOR fonte disponível:
    1) furos (carregar_furos_recorte(), dentro do retângulo) com valor
       não-nulo em `coluna_alt_topo` -- medida DIRETA do topo da formação no
       poço, sem depender da suposição "terreno no afloramento = topo da
       formação" -- se houver >= MIN_FUROS_PRIORIDADE, usa só isso: um
       polígono de afloramento pequeno/fragmentado (Teresina: 3 polígonos,
       Rio do Rasto: 4) dá MILHARES de vértices de contorno (um ponto a cada
       poucos metros do polígono) que, simplesmente concatenados com os
       poucos furos, afogam o furo no ajuste.
    2) se não houver furo suficiente (Fm_SerraAlta, Fm_RioDoRasto -- sem
       coluna equivalente no banco de poços regional), cai pros vértices de
       polígono de afloramento (Litologia_PMP2.shp) com elevação real do
       terreno (curvas3.shp) em cada vértice."""
    n_furo = 0
    pts_furo = None
    if furos is not None and coluna_alt_topo is not None and coluna_alt_topo in furos.columns:
        pts_furo = furos.dropna(subset=[coluna_alt_topo, "E", "N"])
        n_furo = len(pts_furo)

    if n_furo >= MIN_FUROS_PRIORIDADE:
        xs = pts_furo["E"].to_numpy(dtype=float)
        ys = pts_furo["N"].to_numpy(dtype=float)
        zs = pts_furo[coluna_alt_topo].to_numpy(dtype=float)
        return xs, ys, zs, "furo", n_furo

    xs_l, ys_l = [], []
    sub = litologia_gdf[litologia_gdf["SIGLA_UNID"].isin(siglas)] if siglas else litologia_gdf.iloc[0:0]
    for geom in sub.geometry:
        partes = geom.geoms if hasattr(geom, "geoms") else [geom]
        for parte in partes:
            coords = np.array(parte.exterior.coords)
            xs_l.append(coords[:, 0])
            ys_l.append(coords[:, 1])
    if not xs_l:
        return np.array([]), np.array([]), np.array([]), "nenhum", 0
    xs = np.concatenate(xs_l)
    ys = np.concatenate(ys_l)
    zs = avaliar_interpolador(interp_terreno, xs, ys, raio_mascara_km=raio_mascara_km)
    ok = ~np.isnan(zs)
    return xs[ok], ys[ok], zs[ok], "outcrop", int(ok.sum())


def avaliar_plano(coef, grid_e, grid_n):
    a, b, c = coef
    return a * grid_e + b * grid_n + c


def calcular_planos_estilizados(litologia_gdf, interp_terreno, furos=None):
    """Monta o dict {unidade: (a,b,c)} pras 7 formações de UNIDADES_ESTILIZADO
    -- ajuste CONJUNTO: todas as formações compartilham o MESMO mergulho/
    azimute (a,b), só a altura (c) é livre por formação (regressão com uma
    variável indicadora por formação, todos os pontos de todas juntos numa
    única lstsq) -- planos paralelos por construção NUNCA se cruzam.

    Isso corrige um bug real que achamos investigando por que os sills
    sumiam: antes, cada formação tinha seu (a,b) próprio e independente: as
    bem constrangidas por furo (Irati/Palermo/RioBonito, 78-199 pontos
    espalhados) saíam com um azimute; as fracas (Fm_RioDoRasto, só outcrop
    fragmentado) saíam com azimute BEM diferente. A cascata de erosão
    (calcular_contatos_estilizados, min sequencial jovem->velho) não
    consegue distinguir "isso é erosão real" de "o plano de uma formação
    jovem mal ajustada cruzou o de uma mais velha" -- quando cruzava, o
    plano ruim arrastava pra baixo TODAS as formações mais antigas também,
    e Fm_SerraAlta/Fm_Teresina (base/topo do sill) colapsavam pro MESMO
    valor errado -- espessura do sill zerada em 100% da área dos 3
    polígonos (confirmado numericamente). Com planos paralelos isso não
    pode mais acontecer: a ordem estratigráfica (c decrescente) é garantida
    pela própria regressão, não por um min() que pode ser enganado.

    Usa a mesma priorização de fonte por formação (furo > outcrop) de
    coletar_pontos_formacao -- as bem constrangidas (mais furos, mais
    espalhados) dominam o ajuste do mergulho/azimute compartilhado, que é
    o que queremos: confiar mais no dado mais denso e mais direto.
    Compartilhado entre o cubo e a seção -- os dois precisam dos MESMOS
    planos pra ficarem consistentes entre si."""
    if furos is None:
        furos = carregar_furos_recorte()

    dados = {}
    for unidade in UNIDADES_ESTILIZADO:
        siglas = SIGLAS_POR_UNIDADE.get(unidade)
        coluna_furo = MAPA_ALT_TOPO_POR_UNIDADE.get(unidade)
        xs, ys, zs, fonte, n = coletar_pontos_formacao(siglas, litologia_gdf, interp_terreno, furos, coluna_furo)
        dados[unidade] = (xs, ys, zs)
        print(f"[info] {unidade}: ajuste usando {n} pontos ({fonte})")

    unidades_com_dado = [u for u in UNIDADES_ESTILIZADO if len(dados[u][2]) > 0]
    if not unidades_com_dado:
        raise RuntimeError("Nenhuma formação com dado suficiente pra ajustar plano -- confira Litologia_PMP2.shp/furos")

    linhas_A, linhas_z = [], []
    for i, unidade in enumerate(unidades_com_dado):
        xs, ys, zs = dados[unidade]
        onehot = np.zeros((len(zs), len(unidades_com_dado)))
        onehot[:, i] = 1.0
        linhas_A.append(np.column_stack([xs, ys, onehot]))
        linhas_z.append(zs)
    A = np.vstack(linhas_A)
    z = np.concatenate(linhas_z)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    planos = {u: (a, b, float(coef[2 + i])) for i, u in enumerate(unidades_com_dado)}

    for u in UNIDADES_ESTILIZADO:
        if u not in planos:
            print(f"[aviso] {u}: sem dado nenhum, sem plano ajustado")
            planos[u] = None
    return planos


def calcular_contatos_estilizados(planos, terreno, grid_e, grid_n):
    """Aplica os planos (calcular_planos_estilizados) numa grade, com
    erosão em cascata (min sequencial a partir do terreno real) -- devolve
    {unidade: grade de elevação do topo}."""
    contatos = {}
    corte_atual = terreno.copy()
    for unidade in UNIDADES_ESTILIZADO:
        plano = avaliar_plano(planos[unidade], grid_e, grid_n)
        corte_atual = np.minimum(corte_atual, plano)
        contatos[unidade] = corte_atual.copy()
    return contatos


def carregar_area_pmp():
    """Lê area.shp (7 polígonos nomeados: Montanhão, Nova Veneza, Criciuma
    sul, Maracajá, Rio maior, Uru1, Uru2 -- a nova área de trabalho, menor
    que AREA RECORTE.shp) -- devolve a união deles (1 polígono) + bounds."""
    gdf = gpd.read_file(AREA_PMP_SHP).to_crs("EPSG:31982")
    poligono = gdf.geometry.union_all()
    return poligono, poligono.bounds


def carregar_vertices_curvas_pmp():
    """Le curvas3.shp (curva de nível já recortada pro retângulo area2.shp,
    coluna 'ELEV') -- bem menor que CURVAS.shp regional, não precisa de
    desbaste/cache pesado (só cache do custo de leitura/reprojeção)."""
    coluna_z = "ELEV" if "ELEV" in gpd.read_file(CURVAS_PMP_SHP, rows=1).columns else "Contour"
    chave = f"curvaspmp_{CURVAS_PMP_SHP.stat().st_size}_{int(CURVAS_PMP_SHP.stat().st_mtime)}"
    cache_arq = CACHE_DIR / f"{chave}.npz"
    if cache_arq.exists():
        dados = np.load(cache_arq)
        return dados["xs"], dados["ys"], dados["zs"]

    gdf = gpd.read_file(CURVAS_PMP_SHP).to_crs("EPSG:31982")
    coords = gdf.get_coordinates()
    coords[coluna_z] = gdf[coluna_z].reindex(coords.index).to_numpy()
    xs, ys, zs = coords["x"].to_numpy(), coords["y"].to_numpy(), coords[coluna_z].to_numpy()

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache_arq, xs=xs, ys=ys, zs=zs)
    return xs, ys, zs


def carregar_litologia_pmp():
    """Lê Litologia_PMP.shp (CPRM/SGB, EPSG:4674), reprojeta e classifica
    cada polígono na unidade estilizada (MAPA_LITOLOGIA_PMP_PARA_UNIDADE)."""
    gdf = gpd.read_file(LITOLOGIA_PMP_SHP).to_crs("EPSG:31982")
    gdf["unidade_padrao"] = gdf["SIGLA_UNID"].map(MAPA_LITOLOGIA_PMP_PARA_UNIDADE)
    return gdf


def carregar_sills_individualizados():
    """Os 2 corpos de sill JÁ MAPEADOS como polígono real em Litologia_PMP.shp
    (HIERARQUIA='Corpo': K1sg_delta_m "Montanhão", K1sg_delta_nv "Nova
    Veneza") -- devolve lista de (nome, geometria) em EPSG:31982."""
    gdf = carregar_litologia_pmp()
    sills = gdf[gdf["SIGLA_UNID"].isin(SIGLAS_SILL_INDIVIDUALIZADO)]
    return [(row.NOME_UNIDA, row.geometry) for row in sills.itertuples()]


def carregar_area_recorte():
    """Lê AREA RECORTE.shp (polígono único, já em EPSG:31982, colocado pelo
    usuário pra focar/refinar a análise numa área específica -- bounds bem
    parecidos com o antigo DEM local de Criciúma, ~42x64km) -- devolve a
    geometria shapely já reprojetada (garantida) e seus bounds."""
    gdf = gpd.read_file(AREA_RECORTE_SHP).to_crs("EPSG:31982")
    poligono = gdf.geometry.union_all()
    return poligono, poligono.bounds  # bounds = (xmin, ymin, xmax, ymax)


def pontos_dentro_poligono(xs, ys, poligono) -> np.ndarray:
    """Máscara booleana (mesmo shape de xs/ys) -- True onde o ponto cai
    dentro do polígono de recorte. Usa shapely.contains_xy (vetorizado em
    C, shapely>=2) quando disponível -- muito mais rápido que montar uma
    GeoSeries de Point em loop Python (a antiga abordagem com geopandas
    sindex ainda fica como fallback pra shapely 1.x)."""
    xs_f = np.ravel(np.asarray(xs, dtype=float))
    ys_f = np.ravel(np.asarray(ys, dtype=float))
    if hasattr(shapely, "contains_xy"):
        dentro = shapely.contains_xy(poligono, xs_f, ys_f)
    else:
        pontos = gpd.GeoSeries([Point(x, y) for x, y in zip(xs_f, ys_f)], crs="EPSG:31982")
        dentro = pontos.within(poligono).to_numpy()
    return dentro.reshape(np.shape(xs))


def carregar_vertices_curvas():
    """Le CURVAS.shp (curva de nível 10m), extrai todos os vértices com sua
    elevação (ELEV) e desbasta numa grade de DESBASTE_CURVAS_M -- o arquivo
    tem ~62 milhões de vértices, triangular isso direto é inviável; o
    desbaste mantém 1 vértice por célula (ordem estável), preservando a
    distribuição espacial da curva.

    O resultado (desbastado + reprojetado pra EPSG:31982) é cacheado num
    .npz em CACHE_DIR, com chave derivada de mtime/tamanho do .shp e do
    passo de desbaste -- ler/reprojetar os 62M vértices custa ~40s a cada
    execução, e os dois visualizadores usam os MESMOS vértices; o cache
    zera esse custo nas re-execuções (recarregado automaticamente se o
    .shp mudar)."""
    chave = f"curvas_{DESBASTE_CURVAS_M:.0f}_{CURVAS_SHP.stat().st_size}_{int(CURVAS_SHP.stat().st_mtime)}"
    cache_arq = CACHE_DIR / f"{chave}.npz"
    if cache_arq.exists():
        dados = np.load(cache_arq)
        print(f"[info] curvas de nível: cache carregado ({len(dados['xs']):,} vértices desbastados)")
        return dados["xs"], dados["ys"], dados["zs"]

    gdf = gpd.read_file(CURVAS_SHP).to_crs("EPSG:31982")
    coords = gdf.get_coordinates()
    coords["ELEV"] = gdf["ELEV"].reindex(coords.index).to_numpy()
    xs, ys, zs = coords["x"].to_numpy(), coords["y"].to_numpy(), coords["ELEV"].to_numpy()

    cel_x = np.floor(xs / DESBASTE_CURVAS_M).astype(np.int64)
    cel_y = np.floor(ys / DESBASTE_CURVAS_M).astype(np.int64)
    cel_id = cel_x * 10_000_000 + cel_y
    ordem = np.argsort(cel_id, kind="stable")
    id_ordenado = cel_id[ordem]
    manter = np.ones(len(id_ordenado), dtype=bool)
    manter[1:] = id_ordenado[1:] != id_ordenado[:-1]
    idx = ordem[manter]
    print(f"[info] curvas de nível: {len(xs):,} vértices brutos -> {len(idx):,} após desbaste ({DESBASTE_CURVAS_M:.0f}m)")

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache_arq, xs=xs[idx], ys=ys[idx], zs=zs[idx])
    print(f"[info] curvas de nível: cache gravado em {cache_arq}")
    return xs[idx], ys[idx], zs[idx]


def construir_interpolador(xt, yt, zt):
    """Monta os 3 objetos de interpolação (linear, nearest, árvore de
    distância) UMA VEZ -- reusar via avaliar_interpolador em vez de
    reconstruir a cada chamada é essencial quando se avalia muitos pontos/
    linhas diferentes (ex.: dezenas de posições de um perfil rotacionável),
    já que construir o LinearNDInterpolator de ~570k vértices da curva de
    nível sozinho leva alguns segundos."""
    pontos = np.column_stack([xt, yt])
    interp_linear = LinearNDInterpolator(pontos, zt)
    interp_nearest = NearestNDInterpolator(pontos, zt)
    arvore = cKDTree(pontos)
    return interp_linear, interp_nearest, arvore


def avaliar_interpolador(interpoladores, xs, ys, raio_mascara_km=RAIO_MASCARA_TERRENO_KM):
    interp_linear, interp_nearest, arvore = interpoladores
    z_lin = interp_linear(xs, ys)
    z_nn = interp_nearest(xs, ys)
    z = np.where(np.isnan(z_lin), z_nn, z_lin)
    dist, _ = arvore.query(np.column_stack([xs, ys]))
    return np.where(dist <= raio_mascara_km * 1000, z, np.nan)


def interpolar_terreno(xt, yt, zt, xs, ys):
    """Atalho de conveniência pra uso único (não repetido em loop) --
    constrói o interpolador e avalia numa só chamada. Se for avaliar várias
    vezes (ex.: um frame por posição de corte), use construir_interpolador
    uma vez + avaliar_interpolador em cada avaliação."""
    return avaliar_interpolador(construir_interpolador(xt, yt, zt), xs, ys)


def poligono_para_scatter_xy(geom):
    """Contorno de um polígono/multipolígono (com buracos) como x,y pra um
    Scatter com fill='toself' -- cada anel (exterior + buracos) vira um
    sub-caminho separado por NaN, forçando orientação via shapely.orient()
    pra buraco virar buraco de verdade (regra "nonzero" do Plotly, ver
    notas do Taió em gerar_secao_interativa.py)."""
    from shapely.geometry.polygon import orient
    partes = geom.geoms if hasattr(geom, "geoms") else [geom]
    xs, ys = [], []
    primeiro = True
    for parte in partes:
        parte = orient(parte, sign=1.0)
        for anel in [parte.exterior] + list(parte.interiors):
            if not primeiro:
                xs.append(np.nan)
                ys.append(np.nan)
            primeiro = False
            coords = np.array(anel.coords)
            xs.extend(coords[:, 0])
            ys.extend(coords[:, 1])
    return np.array(xs), np.array(ys)


def carregar_geologia_polygons():
    """Le litoestratigrafia_500_sgb.shp (CPRM/SGB), reprojeta e classifica
    cada polígono na nossa unidade padrão -- usado tanto pela classificação
    em grade (classificar_geologia_real) quanto pelo drape vetorial no mapa
    em planta (poligono_para_scatter_xy)."""
    geologia = gpd.read_file(GEOLOGIA_SHP).to_crs("EPSG:31982")
    geologia["unidade_padrao"] = geologia["unidade"].map(MAPA_SGB_PARA_UNIDADE)
    return geologia


def classificar_geologia_real(grid_e: np.ndarray, grid_n: np.ndarray):
    """Classifica cada célula de uma grade pela unidade geológica real
    (litoestratigrafia_500_sgb.shp, CPRM/SGB) -- retorna um índice inteiro
    por célula (-1 se fora de qualquer polígono, cobertura é só local) e a
    lista de unidades na ordem desses índices."""
    geologia = carregar_geologia_polygons()

    pontos = gpd.GeoDataFrame(
        {"idx": np.arange(grid_e.size)},
        geometry=[Point(e, n) for e, n in zip(grid_e.ravel(), grid_n.ravel())],
        crs="EPSG:31982",
    )
    juncao = gpd.sjoin(pontos, geologia[["unidade_padrao", "geometry"]], how="left", predicate="within")
    juncao = juncao.drop_duplicates(subset="idx")  # sjoin pode duplicar se polígonos se sobrepõem
    juncao = juncao.set_index("idx").reindex(range(grid_e.size))

    unidades_presentes = [u for u in UNIDADES if u in juncao["unidade_padrao"].values]
    indice_por_unidade = {u: i for i, u in enumerate(unidades_presentes)}
    classe = juncao["unidade_padrao"].map(indice_por_unidade).to_numpy(dtype=float)
    classe = np.where(np.isnan(classe), -1, classe).reshape(grid_e.shape)
    return classe, unidades_presentes


def colorscale_discreta(cores: list):
    """Colorscale Plotly 'em degraus' pra usar com surfacecolor categórico
    (um índice inteiro por célula) -- cada cor ocupa uma faixa igual,
    truque de repetir [i/N,cor] e [(i+1)/N,cor] (ver notas do Taió)."""
    n = max(len(cores), 1)
    escala = []
    for i, cor in enumerate(cores):
        escala.append([i / n, cor])
        escala.append([(i + 1) / n, cor])
    return escala, n


def tema_claro():
    return dict(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F7F7F9",
        font_color="#1B1F2E", grid_color="#D8D8DE", axis_color="#1B1F2E",
    )


def tema_escuro():
    return dict(
        paper_bgcolor=MARCA_NAVY, plot_bgcolor="#3A3A46",
        font_color=MARCA_CINZA_CLARO, grid_color="#54545f", axis_color=MARCA_CINZA_CLARO,
    )


def botoes_tema(eixos_2d: list | None = None, cena_3d: bool = False):
    """Monta os 2 botões (Tema: Claro/Escuro) de um updatemenu -- cada um
    faz um Plotly.relayout com paper/plot bgcolor + cor de fonte, e (se
    eixos_2d for passado, ex.: ["xaxis","yaxis"] ou ["xaxis2","yaxis2"]) a
    cor de grade/eixo desses eixos 2D, ou (se cena_3d) da cena 3D inteira."""
    botoes = []
    for label, tema in (("Tema: Claro", tema_claro()), ("Tema: Escuro", tema_escuro())):
        args = {
            "paper_bgcolor": tema["paper_bgcolor"],
            "font.color": tema["font_color"],
        }
        if eixos_2d:
            args["plot_bgcolor"] = tema["plot_bgcolor"]
            for eixo in eixos_2d:
                args[f"{eixo}.gridcolor"] = tema["grid_color"]
                args[f"{eixo}.color"] = tema["axis_color"]
        if cena_3d:
            args["scene.xaxis.color"] = tema["axis_color"]
            args["scene.yaxis.color"] = tema["axis_color"]
            args["scene.zaxis.color"] = tema["axis_color"]
            args["scene.xaxis.gridcolor"] = tema["grid_color"]
            args["scene.yaxis.gridcolor"] = tema["grid_color"]
            args["scene.zaxis.gridcolor"] = tema["grid_color"]
        botoes.append(dict(label=label, method="relayout", args=[args]))
    return botoes


def quantizar(arr, casas=1):
    """Arredonda um array pra N casas decimais -- usado ANTES de passar as
    coordenadas pro Plotly pra encurtar o JSON do HTML (a precisão nativa
    float64 grava ~17 dígitos por número; coords UTM a 1m e altitudes a
    0.1m não mudam nada visualmente, mas cortam bastante o tamanho do
    arquivo). Mantém NaN/None intactos."""
    arr = np.asarray(arr, dtype=float)
    return np.round(arr, casas)


def construir_solido(grid_e, grid_n, z_topo, z_base, cor, nome, hover):
    """Extruda uma superficie (z_topo) ate uma base (z_base, mesma grade) --
    topo + base + paredes do perimetro retangular da grade, como um
    go.Mesh3d. Células onde topo ou base são NaN ficam de fora (buraco na
    malha, aceitável -- só ocorre perto da borda da máscara de distância/
    polígono). Compartilhado entre gerar_visualizador_3d_pmp.py e
    gerar_cubo_estilizado_pmp.py."""
    import plotly.graph_objects as go

    ny, nx = grid_e.shape
    idx_topo = np.arange(ny * nx).reshape(ny, nx)
    idx_base = idx_topo + ny * nx

    z_topo = quantizar(z_topo, 1)
    z_base = quantizar(z_base, 1)
    grid_e = quantizar(grid_e, 0)
    grid_n = quantizar(grid_n, 0)

    vx = np.concatenate([grid_e.ravel(), grid_e.ravel()])
    vy = np.concatenate([grid_n.ravel(), grid_n.ravel()])
    vz = np.concatenate([z_topo.ravel(), z_base.ravel()])

    valid = ~np.isnan(z_topo) & ~np.isnan(z_base)
    fi, fj, fk = [], [], []

    def add_quad(a, b, c, d):
        fi.extend([a, a]); fj.extend([b, c]); fk.extend([c, d])

    for r in range(ny - 1):
        for c in range(nx - 1):
            if not (valid[r, c] and valid[r + 1, c] and valid[r, c + 1] and valid[r + 1, c + 1]):
                continue
            a, b, cc, d = idx_topo[r, c], idx_topo[r, c + 1], idx_topo[r + 1, c + 1], idx_topo[r + 1, c]
            add_quad(a, b, cc, d)  # topo
            a2, b2, c2, d2 = idx_base[r, c], idx_base[r, c + 1], idx_base[r + 1, c + 1], idx_base[r + 1, c]
            add_quad(d2, c2, b2, a2)  # base (ordem invertida)

    for c in range(nx - 1):
        for r in (0, ny - 1):
            if valid[r, c] and valid[r, c + 1]:
                t1, t2 = idx_topo[r, c], idx_topo[r, c + 1]
                b1, b2 = idx_base[r, c], idx_base[r, c + 1]
                add_quad(t1, t2, b2, b1)
    for r in range(ny - 1):
        for c in (0, nx - 1):
            if valid[r, c] and valid[r + 1, c]:
                t1, t2 = idx_topo[r, c], idx_topo[r + 1, c]
                b1, b2 = idx_base[r, c], idx_base[r + 1, c]
                add_quad(t1, t2, b2, b1)

    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=fi, j=fj, k=fk,
        color=cor, opacity=0.9, name=nome, flatshading=True, showlegend=True,
        hovertemplate=f"{hover}<extra></extra>", visible=False,
    )


def construir_solido_poligono(poligono, z_topo_fn, z_base_fn, cor, nome, hover, resolucao=40):
    """Como construir_solido, mas o footprint EM PLANTA é o polígono real
    (não um retângulo) -- usado pros sills individualizados (Litologia_PMP,
    HIERARQUIA='Corpo'), que têm forma mapeada de verdade, não uma grade
    regional. Amostra o bbox do polígono numa grade `resolucao`x`resolucao`,
    mascara pelos pontos DENTRO do polígono, e chama z_topo_fn/z_base_fn
    (funções x,y -> z) só nesses pontos."""
    e_min, n_min, e_max, n_max = poligono.bounds
    grid_e, grid_n = np.meshgrid(
        np.linspace(e_min, e_max, resolucao), np.linspace(n_min, n_max, resolucao)
    )
    dentro = pontos_dentro_poligono(grid_e, grid_n, poligono)
    z_topo = np.where(dentro, z_topo_fn(grid_e, grid_n), np.nan)
    z_base = np.where(dentro, z_base_fn(grid_e, grid_n), np.nan)
    return construir_solido(grid_e, grid_n, z_topo, z_base, cor, nome, hover)


def adicionar_escala_e_norte(fig, x_min, x_max, y_min, y_max, row=None, col=None,
                              xref="x", yref="y", cor="black", comprimento_km=10.0):
    """Escala gráfica + seta norte num mapa em planta (anotações/traces
    fixas, mesma técnica do Taió). Passe xref/yref explícitos (ex.: "x",
    "y") quando usar dentro de um make_subplots -- o mapeamento row/col ->
    xN/yN do Plotly não é 1:1 com o número da coluna, então não dá pra
    derivar automaticamente."""
    kwargs = {}
    if row is not None:
        kwargs = dict(row=row, col=col)
    comprimento = comprimento_km * 1000
    esc_x0 = x_min + (x_max - x_min) * 0.05
    esc_x1 = esc_x0 + comprimento
    esc_y0 = y_min + (y_max - y_min) * 0.04
    tick = (y_max - y_min) * 0.012
    esc_x = [esc_x0, esc_x0, None, esc_x0, esc_x1, None, esc_x1, esc_x1]
    esc_y = [esc_y0 - tick, esc_y0 + tick, None, esc_y0, esc_y0, None, esc_y0 - tick, esc_y0 + tick]
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=esc_x, y=esc_y, mode="lines", line=dict(color="white", width=5),
                              showlegend=False, hoverinfo="none"), **kwargs)
    fig.add_trace(go.Scatter(x=esc_x, y=esc_y, mode="lines", line=dict(color=cor, width=2),
                              showlegend=False, hoverinfo="none"), **kwargs)
    fig.add_annotation(
        x=(esc_x0 + esc_x1) / 2, y=esc_y0 + (y_max - y_min) * 0.03, xref=xref, yref=yref,
        text=f"{comprimento_km:.0f} km", showarrow=False, font=dict(size=11, color=cor), xanchor="center",
    )
    norte_x = x_max - (x_max - x_min) * 0.10
    norte_y0 = y_max - (y_max - y_min) * 0.24
    norte_y1 = y_max - (y_max - y_min) * 0.13
    fig.add_annotation(x=norte_x, y=norte_y1, ax=norte_x, ay=norte_y0, xref=xref, yref=yref, axref=xref, ayref=yref,
                        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=cor, text="")
    fig.add_annotation(x=norte_x, y=norte_y1 + (y_max - y_min) * 0.025, xref=xref, yref=yref,
                        text="N", showarrow=False, font=dict(size=13, color=cor))


# =====================================================================
# Camadas compartilhadas pelo webmap e pelo dashboard (Leaflet) -- mesma
# estrutura do Taió (gerar_webmap_taio.py / gerar_dashboard_geoquimico.py),
# só que alimentada com os dados do PMP: retângulo area2.shp, curvas3.shp,
# Litologia_PMP2.shp e os furos de sondagem dentro do retângulo.
# =====================================================================
import io
import json

# cor por SIGLA_UNID (Litologia_PMP2.shp). CPRM oficial só onde há bate exato
# na biblioteca de estilo (ver nota em CORES_ESTILIZADO + NP3_gamma_pc, que
# também bateu exato); as demais são cores PROVISÓRIAS, sem símbolo CPRM
# equivalente encontrado.
COR_POR_SIGLA = {
    "P23rr": CORES_ESTILIZADO["Fm_RioDoRasto"], "P2t": CORES_ESTILIZADO["Fm_Teresina"],
    "P2sa": CORES_ESTILIZADO["Fm_SerraAlta"], "P1i": CORES_ESTILIZADO["Fm_Irati"],
    "P1p": CORES_ESTILIZADO["Fm_Palermo"], "P1rb": CORES_ESTILIZADO["Fm_RioBonito"],
    "C2P1t": CORES_ESTILIZADO["Fm_Taciba"],
    "K1sg": "#6F8F72", "K1bt": "#E6B86A", "K1_beta_vs": "#9A7FA6",   # provisórias
    "NP3_gamma_pc": "#FF2014",                                          # CPRM exato (255,32,20)
    "Q2apa": "#F2E6A0", "Q2ca": "#D9CB82",                              # provisórias
    "K1sg_delta_m": CORES_ESTILIZADO["Gp_SerraGeral"],
    "K1sg_delta_nv": CORES_ESTILIZADO["Gp_SerraGeral"],
    "K1sg_delta_u": CORES_ESTILIZADO["Gp_SerraGeral"],
}
COR_LITOLOGIA_PADRAO = "#CCCCCC"

# unidade mais profunda atingida pelo furo (jovem -> antigo), a partir de Prof_topo_*
ORDEM_PROFUNDIDADE_FURO = [
    ("Cenozoico", "Cenozoico", "#D9CB82"),
    ("sbGp_EstradaNova", "Estrada Nova (Teresina/Serra Alta)", CORES_ESTILIZADO["Fm_Teresina"]),
    ("Fm_Irati", "Irati", CORES_ESTILIZADO["Fm_Irati"]),
    ("Fm_Palermo", "Palermo", "#9FB4C4"),
    ("Fm_RioBonito", "Rio Bonito", "#8A9AA3"),
    ("Gp_Itarare", "Itararé (Taciba)", "#C9BDB4"),
    ("Embasamento", "Embasamento", "#4A4A4A"),
]
COR_FURO_SEM_TOPO = "#999999"
ROTULO_FURO_SEM_TOPO = "Sem topo estratigráfico"

CORES_HIPSOMETRICAS = ["#4F9AA8", "#9FC1A3", "#D8C88C", "#C6924A", "#A66A2C"]  # mesma rampa do Taió

LEAFLET_LINKS = """<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>"""

# basemaps (tiles reais, precisam de internet) -- MESMOS 5 do webmap do Taió.
# String simples (não f-string): as chaves {z}/{x}/{y} são do Leaflet.
JS_BASEMAPS = """
    var satelite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Esri World Imagery', maxZoom: 19,
    });
    var rico = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 20, subdomains: 'abcd',
    });
    var escuro = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 20, subdomains: 'abcd',
    });
    var osmPadrao = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
    });
    var relevo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors, SRTM &copy; OpenTopoMap (CC-BY-SA)',
        maxZoom: 17, subdomains: 'abc',
    });
"""


def _hex_rgb(hex_cor):
    h = hex_cor.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _para_wgs84(gdf):
    return gdf.to_crs(4326) if gdf.crs is not None else gdf.set_crs(31982).to_crs(4326)


def _geojson_leve(gdf):
    """GeoJSON dict com coordenadas arredondadas a ~1 m (1e-5 grau) -- corta
    bastante o peso do HTML sem perda visível."""
    gdf = gdf.copy()
    gdf["geometry"] = [shapely.set_precision(g, 1e-5) if g is not None and not g.is_empty else g for g in gdf.geometry]
    return json.loads(gdf.to_json())


def gerar_hipsometria_leaflet(resolucao=700):
    """PNG RGBA (base64) hipsométrico + hillshade da topografia real
    (curvas3.shp) pra L.imageOverlay -- mesma técnica/rampa do webmap do
    Taió. A grade é amostrada DIRETO em lon/lat (transformada pra UTM pra
    consultar o terreno) e recortada pelo retângulo, então cai certo sobre os
    tiles (o retângulo em UTM não é alinhado a lat/lon, o que entortaria uma
    imagem gerada em UTM). Devolve (png_b64, [[lat_min, lon_min], [lat_max, lon_max]], (zmin, zmax))."""
    from PIL import Image
    from pyproj import Transformer

    poligono, (e0, n0, e1, n1) = carregar_area_pmp()
    xt, yt, zt = carregar_vertices_curvas_pmp()
    interp = construir_interpolador(xt, yt, zt)
    to_wgs = Transformer.from_crs("EPSG:31982", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:31982", always_xy=True)
    cantos = [to_wgs.transform(x, y) for x, y in [(e0, n0), (e0, n1), (e1, n0), (e1, n1)]]
    lon_min, lon_max = min(c[0] for c in cantos), max(c[0] for c in cantos)
    lat_min, lat_max = min(c[1] for c in cantos), max(c[1] for c in cantos)

    lons = np.linspace(lon_min, lon_max, resolucao)
    lats = np.linspace(lat_max, lat_min, resolucao)  # linha 0 = norte
    glon, glat = np.meshgrid(lons, lats)
    gx, gy = to_utm.transform(glon.ravel(), glat.ravel())
    gx, gy = np.asarray(gx).reshape(glon.shape), np.asarray(gy).reshape(glon.shape)
    dentro = pontos_dentro_poligono(gx, gy, poligono)
    gz = avaliar_interpolador(interp, gx.ravel(), gy.ravel(), raio_mascara_km=3.0).reshape(gx.shape)
    dentro &= ~np.isnan(gz)

    zmin, zmax = float(np.nanmin(gz[dentro])), float(np.nanmax(gz[dentro]))
    t = np.clip((np.nan_to_num(gz, nan=zmin) - zmin) / (zmax - zmin), 0, 1)
    paleta = np.array([_hex_rgb(c) for c in CORES_HIPSOMETRICAS], dtype=float)
    n_trechos = len(paleta) - 1
    pos = t * n_trechos
    idx = np.clip(pos.astype(int), 0, n_trechos - 1)
    rgb = paleta[idx] + (paleta[idx + 1] - paleta[idx]) * (pos - idx)[..., None]

    # hillshade (sol NO 315°, 45°) -- espaçamento real em metros
    lat_c = (lat_min + lat_max) / 2
    dy_m = (lat_max - lat_min) / (resolucao - 1) * 110_574.0
    dx_m = (lon_max - lon_min) / (resolucao - 1) * 111_320.0 * np.cos(np.radians(lat_c))
    dzdy, dzdx = np.gradient(np.nan_to_num(gz, nan=zmin), -dy_m, dx_m)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)
    az, alt = np.radians(360.0 - 315.0 + 90.0), np.radians(45.0)
    sombra = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    rgb = np.clip(rgb * (0.45 + 0.65 * sombra)[..., None], 0, 255).astype(np.uint8)

    rgba = np.dstack([rgb, np.where(dentro, 255, 0).astype(np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), [[lat_min, lon_min], [lat_max, lon_max]], (zmin, zmax)


def preparar_geologia_leaflet():
    """Devolve (geojson_formacoes, geojson_sills, itens_legenda_formacoes) em
    WGS84 -- 'categoria' = NOME_UNIDA (chave da legenda item a item)."""
    gdf = carregar_litologia_pmp()
    gdf["cor"] = gdf["SIGLA_UNID"].map(COR_POR_SIGLA).fillna(COR_LITOLOGIA_PADRAO)
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf["categoria"] = gdf["NOME_UNIDA"]
    gdf["popup"] = gdf.apply(lambda r: (
        f"<b>{r['NOME_UNIDA']}</b><br>{r['SIGLA_UNID']} · {r['HIERARQUIA']}<br>"
        f"Área do polígono: {r['area_km2']:.2f} km²"), axis=1)
    eh_sill = gdf["SIGLA_UNID"].isin(SIGLAS_SILL_INDIVIDUALIZADO)
    cols = ["SIGLA_UNID", "categoria", "cor", "popup", "geometry"]
    formacoes = _para_wgs84(gdf.loc[~eh_sill, cols])
    sills = gdf.loc[eh_sill, cols].copy()
    sills["categoria"] = "Sill — " + sills["categoria"]
    sills["popup"] = ("<b>Sill — " + gdf.loc[eh_sill, "NOME_UNIDA"] + "</b><br>Corpo intrusivo mapeado (Gp. Serra Geral)<br>Área: "
                      + gdf.loc[eh_sill, "area_km2"].map("{:.2f} km²".format))
    sills = _para_wgs84(sills)
    itens = (gdf.loc[~eh_sill].drop_duplicates("NOME_UNIDA").sort_values("area_km2", ascending=False)
             [["NOME_UNIDA", "cor"]].rename(columns={"NOME_UNIDA": "label"}).to_dict("records"))
    return _geojson_leve(formacoes), _geojson_leve(sills), itens


def preparar_curvas_leaflet(passo_m=50, tolerancia_m=8.0):
    """Curvas de nível mestras (múltiplos de `passo_m`) em WGS84, simplificadas."""
    gdf = gpd.read_file(CURVAS_PMP_SHP).to_crs("EPSG:31982")
    gdf = gdf[(gdf["ELEV"] % passo_m) == 0].copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerancia_m)
    gdf["categoria"] = "Curva de nível (%dm)" % passo_m
    gdf["popup"] = gdf["ELEV"].map(lambda v: f"Cota {v:.0f} m")
    return _geojson_leve(_para_wgs84(gdf[["ELEV", "categoria", "popup", "geometry"]]))


def preparar_contorno_area_leaflet():
    poligono, _ = carregar_area_pmp()
    gdf = gpd.GeoDataFrame({"popup": ["Área de estudo (area2.shp)"], "categoria": ["Área de estudo"]},
                           geometry=[poligono], crs=31982)
    return _geojson_leve(_para_wgs84(gdf))


def _intervalos_corpo(s):
    import re
    if pd.isna(s):
        return []
    return [(float(a), float(b)) for a, b in re.findall(r"\(([\d.]+)\s*,\s*([\d.]+)\)", str(s))]


def preparar_furos():
    """Furos dentro do retângulo como DataFrame pronto pra tabela/mapa/gráficos:
    id, nome, unidade mais profunda atingida (rótulo/cor), corpo intrusivo
    (intervalo mais espesso -> espessura medida em furo) e lat/lon."""
    from pyproj import Transformer
    f = carregar_furos_recorte().copy()
    f["id"] = ["F%03d" % i for i in range(len(f))]

    def nome_tipo(r):
        for c in ("Cod_poco", "Poco", "Codigo"):
            v = r.get(c)
            if pd.notna(v) and str(v).strip():
                return str(v).strip(), "Furo"
        # sem código de furo: são pontos de campo (ex.: "Soleira / corte de estrada")
        v = r.get("Afloramento")
        txt = str(v).strip()[:40] if pd.notna(v) and str(v).strip() else "Afloramento s/ nome"
        return txt, "Afloramento"
    nt = f.apply(nome_tipo, axis=1)
    f["nome"] = [x[0] for x in nt]
    f["tipo"] = [x[1] for x in nt]
    # ordem: furos primeiro (nome em ordem natural), depois afloramentos
    f = f.sort_values(["tipo", "nome"], key=lambda c: c.str.lower() if c.name == "nome" else c).reset_index(drop=True)
    f["id"] = ["F%03d" % i for i in range(len(f))]

    def mais_profunda(r):
        atual = (ROTULO_FURO_SEM_TOPO, COR_FURO_SEM_TOPO)
        for col, rot, cor in ORDEM_PROFUNDIDADE_FURO:
            if pd.notna(r.get(f"Prof_topo_{col}")):
                atual = (rot, cor)
        return atual
    par = f.apply(mais_profunda, axis=1)
    f["unidade_fundo"] = [p[0] for p in par]
    f["cor"] = [p[1] for p in par]

    f["esp_corpo_m"] = f["Prof_corpos_intrusivos_SG"].map(
        lambda s: max((b - a for a, b in _intervalos_corpo(s)), default=np.nan))
    f["n_corpos"] = f["Prof_corpos_intrusivos_SG"].map(lambda s: len(_intervalos_corpo(s)))
    f["corpo_intervalos"] = f["Prof_corpos_intrusivos_SG"].map(
        lambda s: "; ".join(f"{a:.1f}–{b:.1f} m" for a, b in _intervalos_corpo(s)))

    to_wgs = Transformer.from_crs("EPSG:31982", "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(f["E"].to_numpy(), f["N"].to_numpy())
    f["lon"], f["lat"] = np.round(lon, 6), np.round(lat, 6)
    return f


# =====================================================================
# Satélite (Esri World Imagery) reprojetado pro retângulo -- usado como
# modo de cor do 3D e como fundo do mapa da seção, igual aos modos
# "Satélite" dos apps do Taió (mesma técnica: contextily + reprojeção).
# Baixa os tiles UMA vez (rede) e cacheia em .cache/ -- apagar o .npy força
# baixar de novo. IMAGEM DE TERCEIROS: atribuição "Esri World Imagery";
# confira os termos da Esri antes de uso comercial/redistribuição.
# =====================================================================
SATELITE_ZOOM = 14
SATELITE_RES_M = 23.0  # metros por pixel do raster UTM cacheado


def obter_satelite_utm():
    """Raster RGB uint8 (3, H, W) do satélite no retângulo, em UTM 22S
    (linha 0 = norte), + (e_min, n_min, e_max, n_max)."""
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject
    from shapely.geometry import box

    _, (e0, n0, e1, n1) = carregar_area_pmp()
    w_px, h_px = int(round((e1 - e0) / SATELITE_RES_M)), int(round((n1 - n0) / SATELITE_RES_M))
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"satelite_z{SATELITE_ZOOM}_{w_px}x{h_px}.npy"
    if cache.exists():
        return np.load(cache), (e0, n0, e1, n1)

    import contextily as ctx
    print("Baixando satélite Esri World Imagery (uma vez, fica em cache)...")
    w, s, e, n = gpd.GeoDataFrame(geometry=[box(e0, n0, e1, n1)], crs="EPSG:31982").to_crs("EPSG:4326").total_bounds
    img, ext = ctx.bounds2img(w, s, e, n, ll=True, zoom=SATELITE_ZOOM, source=ctx.providers.Esri.WorldImagery, n_connections=8)
    src_t = from_bounds(ext[0], ext[2], ext[1], ext[3], img.shape[1], img.shape[0])
    dst_t = from_bounds(e0, n0, e1, n1, w_px, h_px)
    dst = np.zeros((3, h_px, w_px), dtype=np.uint8)
    reproject(source=np.moveaxis(img[:, :, :3], -1, 0), destination=dst, src_transform=src_t, src_crs="EPSG:3857",
              dst_transform=dst_t, dst_crs="EPSG:31982", resampling=Resampling.bilinear)
    np.save(cache, dst)
    return dst, (e0, n0, e1, n1)


def amostrar_satelite_rgb(raster, bounds, xs, ys):
    """Cor RGB (N,3 uint8) do raster de satélite nos pontos (xs, ys) em UTM."""
    e0, n0, e1, n1 = bounds
    _, h, w = raster.shape
    col = np.clip(((np.asarray(xs) - e0) / (e1 - e0) * (w - 1)).round().astype(int), 0, w - 1)
    row = np.clip(((n1 - np.asarray(ys)) / (n1 - n0) * (h - 1)).round().astype(int), 0, h - 1)
    return np.stack([raster[0, row, col], raster[1, row, col], raster[2, row, col]], axis=-1)
