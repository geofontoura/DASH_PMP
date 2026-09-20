# DASH_PMP

Dashboard de furos do PMP (Criciúma/SC): lista pesquisável dos furos de sondagem dentro da área, distribuição de
profundidade/cota/fonte, topos e espessuras das unidades (furo × literatura), espessura medida dos corpos
intrusivos, tema claro/escuro e mapa Leaflet sincronizado com a lista e os gráficos.

Página: https://geofontoura.github.io/DASH_PMP/ (`index.html`, gerado por `src/gerar_dashboard_pmp.py`).
Hub: [HUB_PMP](https://github.com/geofontoura/HUB_PMP).

## Código gerador (`src/`)

Este repo também guarda o código que gera os produtos web do PMP a partir de `area2.shp`, `curvas3.shp`,
`Litologia_PMP2.shp` e do banco de poços (só furos dentro do retângulo). Os dados de entrada **não** são versionados
(repo público): ficam em `2_Banco_de_Dados/` localmente.

```
pip install -r requirements.txt
cd src
python gerar_cubo_pmp.py       # cubo 3D   -> VIEW3D_PMP
python gerar_secao_pmp.py      # seção     -> VIEW2D_PMP
python gerar_mapa_web_pmp.py   # webmap    -> MAPS_PMP
python gerar_dashboard_pmp.py  # dashboard -> este repo (index.html)
```
