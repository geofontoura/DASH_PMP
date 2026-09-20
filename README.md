# MODEL_PMP

Modelo geológico estilizado do PMP (Criciúma/SC): gera os 3 produtos web a partir de
`area2.shp`, `curvas3.shp`, `Litologia_PMP2.shp` e do banco de poços (só os furos dentro do retângulo).

- **Planos das 7 formações** (Rio do Rasto, Teresina, Serra Alta, Irati, Palermo, Rio Bonito, Taciba): ajuste conjunto
  por mínimos quadrados, mesmo mergulho/azimute para todas (planos paralelos, nunca se cruzam), altura livre por formação.
  Fonte de pontos: furos (>=5) ou vértices de afloramento sobre o terreno real.
- **Sills** (Montanhão, Nova Veneza, Urussanga): polígono de afloramento real, topo = terreno, base = terreno − 50 m (espessura assumida).
- Erosão em cascata sobre a topografia real (curvas de nível).

## Uso
```
pip install -r requirements.txt
# colocar os dados em 2_Banco_de_Dados/ (não versionados):
#   area2.shp, curvas3.shp, Litologia_PMP2.shp, pontos_unificados/pontos_unificados_pmp.csv
cd src
python gerar_cubo_pmp.py       # -> cubo_pmp.html    (publicar em VIEW3D_PMP)
python gerar_secao_pmp.py      # -> secao_pmp.html   (publicar em VIEW2D_PMP)
python gerar_mapa_web_pmp.py   # -> mapa_web_pmp.html (publicar em MAPS_PMP)
```
Publicação: copiar cada HTML como `index.html` no repo correspondente (GitHub Pages).
