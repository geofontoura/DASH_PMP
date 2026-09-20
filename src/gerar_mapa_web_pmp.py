"""Webmap do PMP -- mesmo estilo do webmap do Taió (gerar_webmap_taio.py):
mapa em tela cheia com Leaflet (zoom/pan contínuo sobre tiles reais), vários
basemaps (Rico/Satélite/Relevo/Escuro/OSM + a HIPSOMETRIA própria do projeto
como "base"), overlays ligáveis e legenda dinâmica clicável item a item.

Camadas (todas dos dados novos do PMP, retângulo area2.shp):
  - Mapa geológico CPRM (Litologia_PMP2.shp, 13 unidades) + 3 sills em destaque
  - Furos de sondagem dentro do retângulo (281, coloridos pela unidade mais
    profunda atingida) + anel nos que interceptaram corpo intrusivo
  - Curvas de nível mestras (50 m, de curvas3.shp) e contorno da área

Precisa de internet ao abrir (tiles sob demanda). A hipsometria é uma imagem
gerada aqui a partir de curvas3.shp (não é tile), então funciona sem rede.

Fontes: ../2_Banco_de_Dados/{area2,curvas3,Litologia_PMP2}.shp e
pontos_unificados_pmp.csv -- via _comum_pmp.py. Só LÊ essas fontes.

Uso:
    python gerar_mapa_web_pmp.py
Gera:
    mapa_web_pmp.html
"""
import json
from pathlib import Path

import pandas as pd

from _comum_pmp import (
    MARCA_ROXO, MARCA_ROXO_ESCURO, MARCA_NAVY, MARCA_CINZA_CLARO, MARCA_FONTE,
    CORES_ESTILIZADO, ORDEM_PROFUNDIDADE_FURO, COR_FURO_SEM_TOPO, ROTULO_FURO_SEM_TOPO,
    LEAFLET_LINKS, JS_BASEMAPS, logo_base64, gerar_hipsometria_leaflet,
    preparar_geologia_leaflet, preparar_curvas_leaflet, preparar_contorno_area_leaflet, preparar_furos,
)

BASE = Path(__file__).resolve().parent
OUT_HTML = BASE / "mapa_web_pmp.html"


def geojson_afloramentos(f):
    """Pontos de campo sem código de furo -> GeoJSON (categoria única)."""
    feats = []
    for r in f.itertuples():
        linhas = [f"<b>{r.nome}</b>"]
        if pd.notna(r.Cidade):
            linhas.append(f"{r.Cidade}")
        if pd.notna(r.Cota_boca):
            linhas.append(f"Cota: {r.Cota_boca:.0f} m")
        linhas.append(f"<span style='opacity:.6'>fonte: {r.fonte_planilha}</span>")
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                      "properties": {"categoria": "Afloramento", "popup": "<br>".join(linhas)}})
    return {"type": "FeatureCollection", "features": feats}


def geojson_furos(f):
    """Furos -> GeoJSON de pontos (WGS84) com popup pronto."""
    feats = []
    for r in f.itertuples():
        linhas = [f"<b>{r.nome}</b>"]
        if pd.notna(r.Municipio):
            linhas.append(f"{r.Municipio}" + (f" · {r.Localidade}" if pd.notna(r.Localidade) else ""))
        if pd.notna(r.Agencia):
            linhas.append(f"Agência: {r.Agencia}")
        if pd.notna(r.Cota_boca):
            linhas.append(f"Cota da boca: {r.Cota_boca:.0f} m")
        if pd.notna(r.Profundidade):
            linhas.append(f"Profundidade: {r.Profundidade:.0f} m")
        linhas.append(f"Unidade mais profunda: {r.unidade_fundo}")
        if r.n_corpos:
            linhas.append(f"<b>Corpo intrusivo</b>: {r.corpo_intervalos}")
        linhas.append(f"<span style='opacity:.6'>fonte: {r.fonte_planilha}</span>")
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
            "properties": {"id": r.id, "cor": r.cor, "categoria": r.unidade_fundo,
                           "corpo": bool(r.n_corpos), "popup": "<br>".join(linhas)},
        })
    return {"type": "FeatureCollection", "features": feats}


TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" href="assets/favicon.png">
<link rel="shortcut icon" href="assets/favicon.ico">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
<title>Webmap PMP — Criciúma</title>
@@LEAFLET@@
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: @@FONTE@@; background: @@NAVY@@; }
  #map { position: absolute; top: 0; bottom: 0; left: 0; right: 0; background: @@NAVY@@; }
  #cabecalho {
    position: absolute; top: 12px; left: 50%; transform: translateX(-50%); z-index: 1000;
    display: flex; align-items: center; gap: 10px; background: rgba(27,31,46,0.88);
    border: 1px solid @@ROXO@@; border-radius: 10px; padding: 8px 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  }
  #cabecalho img { width: 34px; height: 34px; border-radius: 50%; border: 1.5px solid @@ROXO@@; }
  #cabecalho h1 { font-size: 15px; margin: 0; color: @@CINZA@@; white-space: nowrap; }
  #cabecalho h1 b { color: @@ROXO@@; }
  .leaflet-popup-content-wrapper { background: @@ROXO_ESCURO@@; color: @@CINZA@@; border: 1px solid @@ROXO@@; }
  .leaflet-popup-tip { background: @@ROXO_ESCURO@@; }
  .leaflet-popup-content { font-family: @@FONTE@@; font-size: 12px; }
  .leaflet-control-layers { background: @@ROXO_ESCURO@@ !important; color: @@CINZA@@; border: 1px solid @@ROXO@@ !important; }
  .leaflet-control-layers-toggle { filter: invert(1); }
  #legenda {
    background: rgba(45,10,74,0.92); color: @@CINZA@@; border: 1px solid @@ROXO@@;
    border-radius: 6px; padding: 8px 10px; font-size: 11px; max-width: 220px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4); display: flex; flex-direction: column;
  }
  #legenda h4 { margin: 0 0 4px 0; font-size: 11px; color: @@ROXO@@; text-transform: uppercase; letter-spacing: 0.03em; flex-shrink: 0; }
  #legenda .legenda-corpo { overflow-y: auto; max-height: 48vh; padding-right: 4px; }
  #legenda .legenda-corpo::-webkit-scrollbar { width: 6px; }
  #legenda .legenda-corpo::-webkit-scrollbar-track { background: transparent; }
  #legenda .legenda-corpo::-webkit-scrollbar-thumb { background: @@ROXO@@; border-radius: 3px; }
  #legenda .secao { margin-bottom: 6px; }
  #legenda .secao:last-child { margin-bottom: 0; }
  #legenda .secao-titulo {
    font-weight: 600; opacity: 0.85; margin-bottom: 2px; display: flex; align-items: center; gap: 5px;
    cursor: pointer; border-radius: 4px; padding: 2px 3px; transition: background 0.12s;
  }
  #legenda .secao-titulo:hover { background: rgba(123,47,255,0.25); }
  #legenda .secao-titulo .marca { font-size: 10px; width: 11px; flex-shrink: 0; text-align: center; }
  #legenda .secao.inativa .secao-titulo { opacity: 0.5; text-decoration: line-through; }
  #legenda .item {
    display: flex; align-items: center; gap: 6px; line-height: 1.5; padding: 1px 3px 1px 16px;
    cursor: pointer; border-radius: 4px; transition: background 0.12s;
  }
  #legenda .item:hover { background: rgba(123,47,255,0.25); }
  #legenda .item.item-inativa { opacity: 0.4; text-decoration: line-through; }
  #legenda .item .marca-item { font-size: 9px; width: 10px; flex-shrink: 0; text-align: center; }
  #legenda .amostra { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; border: 1px solid rgba(255,255,255,0.5); }
  #legenda .amostra.linha { width: 14px; height: 3px; border-radius: 0; border: none; }
  #legenda .amostra.area { border-radius: 2px; }
  #legenda .amostra.anel { background: transparent !important; border: 2px solid #E63946; }
  footer {
    position: absolute; bottom: 4px; left: 50%; transform: translateX(-50%); z-index: 1000;
    font-size: 10px; color: @@CINZA@@; opacity: 0.6; pointer-events: none;
  }
</style>
</head>
<body>
<div id="map"></div>
<div id="cabecalho">
  @@LOGO@@
  <h1><b>Webmap PMP</b> — Criciúma · geologia, furos e topografia</h1>
</div>
<footer>GS Tech · PMP</footer>
<script>
(function() {
    var map = L.map('map', { zoomControl: true });

    // ---- basemaps (tiles reais, precisam de internet) ----
@@BASEMAPS@@
    var hipsometria = L.imageOverlay('data:image/png;base64,@@HIPSO@@', @@BOUNDS@@, { opacity: 1 });
    satelite.addTo(map);
    map.fitBounds(@@BOUNDS@@);  // depois do 1º basemap (sem camada o Leaflet não aplica o zoom)

    function pontoEstilo(cor, raio) {
        return { radius: raio, fillColor: cor, color: '@@NAVY@@', weight: 1.2, fillOpacity: 0.95 };
    }

    // rastreia sub-camadas por (nome da camada, categoria) pra ligar/desligar item a item na legenda
    var subCamadas = {};
    function registrarSub(nome, f, layer) {
        var cat = f.properties.categoria || 'Outro';
        if (!subCamadas[nome]) subCamadas[nome] = {};
        if (!subCamadas[nome][cat]) subCamadas[nome][cat] = [];
        subCamadas[nome][cat].push(layer);
    }

    var formacoesLayer = L.geoJSON(@@GEO_FORMACOES@@, {
        style: function(f) { return { color: '#000', weight: 0.5, fillColor: f.properties.cor, fillOpacity: 0.6 }; },
        onEachFeature: function(f, layer) { layer.bindPopup(f.properties.popup); registrarSub('Mapa geológico (CPRM)', f, layer); },
    });
    var sillsLayer = L.geoJSON(@@GEO_SILLS@@, {
        style: function(f) { return { color: '#000', weight: 1.2, fillColor: f.properties.cor, fillOpacity: 0.8 }; },
        onEachFeature: function(f, layer) { layer.bindPopup(f.properties.popup); registrarSub('Sills (corpos intrusivos)', f, layer); },
    });
    var furosLayer = L.geoJSON(@@FUROS@@, {
        pointToLayer: function(f, latlng) { return L.circleMarker(latlng, pontoEstilo(f.properties.cor, 5)); },
        onEachFeature: function(f, layer) { layer.bindPopup(f.properties.popup); registrarSub('Furos de sondagem', f, layer); },
    });
    var furosCorpoLayer = L.geoJSON(@@FUROS@@, {
        filter: function(f) { return f.properties.corpo; },
        pointToLayer: function(f, latlng) {
            return L.circleMarker(latlng, { radius: 9, color: '#E63946', weight: 2, fillOpacity: 0, interactive: false });
        },
        onEachFeature: function(f, layer) { registrarSub('Furos com corpo intrusivo', { properties: { categoria: 'Furo com corpo intrusivo' } }, layer); },
    });
    var aflorLayer = L.geoJSON(@@AFLOR@@, {
        pointToLayer: function(f, latlng) { return L.circleMarker(latlng, { radius: 5, fillColor: '#FFD166', color: '@@NAVY@@', weight: 1.2, fillOpacity: 0.95 }); },
        onEachFeature: function(f, layer) { layer.bindPopup(f.properties.popup); registrarSub('Afloramentos (campo)', f, layer); },
    });
    var curvasLayer = L.geoJSON(@@CURVAS@@, {
        style: function() { return { color: '#5A3E1B', weight: 0.8, opacity: 0.7 }; },
        onEachFeature: function(f, layer) { layer.bindPopup(f.properties.popup); registrarSub('Curvas de nível (50 m)', f, layer); },
    });
    var areaLayer = L.geoJSON(@@AREA@@, {
        style: function() { return { color: '@@ROXO@@', weight: 2.5, fill: false, dashArray: '8,5' }; },
        onEachFeature: function(f, layer) { registrarSub('Área de estudo', f, layer); },
    });

    formacoesLayer.addTo(map);
    sillsLayer.addTo(map);
    furosLayer.addTo(map);
    areaLayer.addTo(map);

    var basemaps = {
        "Satélite (Esri)": satelite,
        "Rico (CartoDB Voyager)": rico,
        "Relevo/Topográfico": relevo,
        "Escuro (CartoDB Dark)": escuro,
        "OSM Padrão": osmPadrao,
        "Hipsometria (curvas3.shp)": hipsometria,
    };
    var overlays = {
        "Mapa geológico (CPRM)": formacoesLayer,
        "Sills (corpos intrusivos)": sillsLayer,
        "Furos de sondagem": furosLayer,
        "Furos com corpo intrusivo": furosCorpoLayer,
        "Afloramentos (campo)": aflorLayer,
        "Curvas de nível (50 m)": curvasLayer,
        "Área de estudo": areaLayer,
    };
    L.control.layers(basemaps, overlays, { collapsed: false }).addTo(map);
    L.control.scale({ metric: true, imperial: false }).addTo(map);

    // ---- legenda dinâmica: só as seções das camadas ativas, clicável item a item ----
    var legendaDados = @@LEGENDA@@;
    var LegendaControl = L.Control.extend({
        options: { position: 'topright' },
        onAdd: function() {
            var div = L.DomUtil.create('div', 'leaflet-control');
            div.id = 'legenda';
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            this._div = div;
            return div;
        },
    });
    var legendaControl = new LegendaControl();
    legendaControl.addTo(map);

    function amostraHtml(tipo, cor) {
        if (tipo === 'linha') return '<span class="amostra linha" style="background:' + cor + '"></span>';
        if (tipo === 'area') return '<span class="amostra area" style="background:' + cor + '"></span>';
        if (tipo === 'anel') return '<span class="amostra anel"></span>';
        return '<span class="amostra" style="background:' + cor + '"></span>';
    }
    function itemVisivel(nome, categoria) {
        var grupo = overlays[nome];
        var lista = (subCamadas[nome] && subCamadas[nome][categoria]) || [];
        if (!lista.length) return true;
        return lista.some(function(l) { return grupo.hasLayer(l); });
    }
    function atualizarLegenda() {
        var corpo = '';
        Object.keys(overlays).forEach(function(nome) {
            if (!legendaDados[nome]) return;
            var ativa = map.hasLayer(overlays[nome]);
            var info = legendaDados[nome];
            corpo += '<div class="secao' + (ativa ? '' : ' inativa') + '">';
            corpo += '<div class="secao-titulo" data-camada="' + nome + '" title="Ligar/desligar toda a camada">'
                   + '<span class="marca">' + (ativa ? '&#9745;' : '&#9744;') + '</span>' + nome + '</div>';
            info.itens.forEach(function(it) {
                var vis = itemVisivel(nome, it.label);
                corpo += '<div class="item' + (vis ? '' : ' item-inativa') + '" data-camada="' + nome + '" data-categoria="'
                       + it.label.replace(/"/g, '&quot;') + '" title="Ligar/desligar só este item">'
                       + '<span class="marca-item">' + (vis ? '&#9745;' : '&#9744;') + '</span>'
                       + amostraHtml(info.tipo, it.cor) + '<span>' + it.label + '</span></div>';
            });
            corpo += '</div>';
        });
        legendaControl._div.innerHTML = '<h4>Legenda (clique p/ ligar/desligar)</h4><div class="legenda-corpo">' + corpo + '</div>';

        legendaControl._div.querySelectorAll('.secao-titulo').forEach(function(el) {
            el.addEventListener('click', function() {
                var layer = overlays[el.getAttribute('data-camada')];
                if (map.hasLayer(layer)) { map.removeLayer(layer); } else { map.addLayer(layer); }
                atualizarLegenda();
            });
        });
        legendaControl._div.querySelectorAll('.item').forEach(function(el) {
            el.addEventListener('click', function() {
                var nome = el.getAttribute('data-camada');
                var grupo = overlays[nome];
                var lista = (subCamadas[nome] && subCamadas[nome][el.getAttribute('data-categoria')]) || [];
                if (!lista.length) return;
                var vis = lista.some(function(l) { return grupo.hasLayer(l); });
                lista.forEach(function(l) { if (vis) grupo.removeLayer(l); else grupo.addLayer(l); });
                atualizarLegenda();
            });
        });
    }
    map.on('overlayadd', atualizarLegenda);
    map.on('overlayremove', atualizarLegenda);
    map.on('baselayerchange', atualizarLegenda);
    atualizarLegenda();
})();
</script>
</body>
</html>
"""


def main():
    print("Carregando camadas...")
    formacoes, sills, itens_formacoes = preparar_geologia_leaflet()
    print(f"  geologia: {len(formacoes['features'])} polígonos, {len(sills['features'])} sills")
    todos = preparar_furos()
    furos = todos[todos["tipo"] == "Furo"].reset_index(drop=True)
    aflor = todos[todos["tipo"] == "Afloramento"].reset_index(drop=True)
    print(f"  furos: {len(furos)} ({int((furos['n_corpos'] > 0).sum())} com corpo intrusivo) · afloramentos: {len(aflor)}")
    curvas = preparar_curvas_leaflet()
    print(f"  curvas de nível: {len(curvas['features'])} linhas")
    area = preparar_contorno_area_leaflet()
    print("Gerando hipsometria...")
    hipso_b64, bounds, (zmin, zmax) = gerar_hipsometria_leaflet()
    print(f"  hipsometria {zmin:.0f}-{zmax:.0f} m")

    cor_sill = CORES_ESTILIZADO["Gp_SerraGeral"]
    contagem = furos["unidade_fundo"].value_counts()
    ordem_rotulos = [r for _, r, _ in ORDEM_PROFUNDIDADE_FURO] + [ROTULO_FURO_SEM_TOPO]
    cores_rotulo = {r: c for _, r, c in ORDEM_PROFUNDIDADE_FURO}
    cores_rotulo[ROTULO_FURO_SEM_TOPO] = COR_FURO_SEM_TOPO
    legenda = {
        "Mapa geológico (CPRM)": {"tipo": "area", "itens": itens_formacoes},
        "Sills (corpos intrusivos)": {"tipo": "area", "itens": [
            {"cor": cor_sill, "label": f["properties"]["categoria"]} for f in sills["features"]]},
        "Furos de sondagem": {"tipo": "ponto", "itens": [
            {"cor": cores_rotulo[r], "label": r} for r in ordem_rotulos if contagem.get(r, 0) > 0]},
        "Furos com corpo intrusivo": {"tipo": "anel", "itens": [{"cor": "#E63946", "label": "Furo com corpo intrusivo"}]},
        "Afloramentos (campo)": {"tipo": "ponto", "itens": [{"cor": "#FFD166", "label": "Afloramento"}]},
        "Curvas de nível (50 m)": {"tipo": "linha", "itens": [{"cor": "#5A3E1B", "label": "Curva de nível (50m)"}]},
        "Área de estudo": {"tipo": "linha", "itens": [{"cor": MARCA_ROXO, "label": "Área de estudo"}]},
    }

    logo = logo_base64()
    j = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE
    for token, valor in {
        "@@LEAFLET@@": LEAFLET_LINKS, "@@BASEMAPS@@": JS_BASEMAPS, "@@FONTE@@": MARCA_FONTE,
        "@@NAVY@@": MARCA_NAVY, "@@ROXO@@": MARCA_ROXO, "@@ROXO_ESCURO@@": MARCA_ROXO_ESCURO,
        "@@CINZA@@": MARCA_CINZA_CLARO,
        "@@LOGO@@": f'<img src="data:image/jpeg;base64,{logo}">' if logo else "",
        "@@BOUNDS@@": j(bounds), "@@HIPSO@@": hipso_b64,
        "@@GEO_FORMACOES@@": j(formacoes), "@@GEO_SILLS@@": j(sills),
        "@@FUROS@@": j(geojson_furos(furos)), "@@AFLOR@@": j(geojson_afloramentos(aflor)), "@@CURVAS@@": j(curvas), "@@AREA@@": j(area),
        "@@LEGENDA@@": j(legenda),
    }.items():
        html = html.replace(token, valor)
    assert "@@" not in html, "token sem substituir"
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"-> {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
