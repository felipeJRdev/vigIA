"""
vigIA — Frontend: Gera mapa interativo HTML com Leaflet + satélite Esri.
Lê resultados/previsao_grade_*.csv e gera frontend/mapa_vigia.html.
Abrir o HTML diretamente no navegador — sem servidor necessário.
"""

import os, json, glob
import pandas as pd

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = os.path.dirname(_HERE)
RESULTADOS = os.path.join(PBL, "resultados")
FRONTEND   = _HERE

arqs = sorted([f for f in glob.glob(os.path.join(RESULTADOS, "previsao_grade_*.csv"))])
if not arqs:
    raise FileNotFoundError("Nenhum previsao_grade_*.csv em resultados/. Rode E2 Fase 4 primeiro.")

df = pd.read_csv(arqs[-1], parse_dates=["Data"])
print(f"Carregado: {os.path.basename(arqs[-1])} ({len(df):,} linhas)")

dias = sorted(df["Data"].dt.strftime("%Y-%m-%d").unique())
data_js = {}
for dia in dias:
    sub = df[df["Data"].dt.strftime("%Y-%m-%d") == dia]
    data_js[dia] = [
        {"lat": round(float(r.Cell_Lat),1), "lon": round(float(r.Cell_Lon),1),
         "prob": round(float(r.prob_fogo),4), "risco": r.risco,
         "seco": int(r.DiaSemChuva), "prec": round(float(r.Precipitacao),1),
         "mun": r.Nearest_Municipio}
        for r in sub.itertuples()
    ]

data_json = json.dumps(data_js, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>vigIA — Risco de Queimadas em Goiás</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI', sans-serif; background:#1a1a2e; color:#eee; }}
  #header {{
    background:#16213e; padding:12px 20px;
    display:flex; align-items:center; gap:16px; flex-wrap:wrap;
    border-bottom:2px solid #e74c3c;
  }}
  #header h1 {{ font-size:18px; color:#e74c3c; font-weight:700; }}
  #header span {{ font-size:13px; color:#aaa; }}
  #controls {{
    background:#16213e; padding:10px 20px;
    display:flex; align-items:center; gap:12px; flex-wrap:wrap;
    border-bottom:1px solid #333;
  }}
  .day-btn {{
    background:#0f3460; border:1px solid #e74c3c; color:#eee;
    padding:6px 14px; border-radius:6px; cursor:pointer; font-size:13px;
    transition:all .2s;
  }}
  .day-btn:hover, .day-btn.active {{ background:#e74c3c; color:#fff; font-weight:600; }}
  #filter-wrap {{ display:flex; align-items:center; gap:8px; margin-left:auto; }}
  #filter-wrap label {{ font-size:13px; color:#aaa; }}
  #prob-slider {{ width:120px; accent-color:#e74c3c; }}
  #prob-val {{ font-size:13px; color:#e74c3c; font-weight:600; min-width:40px; }}
  #map {{ height: calc(100vh - 100px); }}
  #legend {{
    position:absolute; bottom:30px; right:10px; z-index:1000;
    background:rgba(22,33,62,0.92); padding:12px 16px; border-radius:8px;
    border:1px solid #333; font-size:12px; min-width:140px;
  }}
  #legend h4 {{ margin-bottom:8px; font-size:13px; color:#e74c3c; }}
  .leg-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
  .leg-dot {{ width:14px; height:14px; border-radius:3px; flex-shrink:0; }}
  #stats {{
    position:absolute; bottom:60px; left:10px; z-index:1000;
    background:rgba(22,33,62,0.92); padding:10px 14px; border-radius:8px;
    border:1px solid #333; font-size:12px; min-width:160px;
  }}
  #stats h4 {{ font-size:13px; color:#e74c3c; margin-bottom:6px; }}
  #stats p {{ margin:3px 0; }}
</style>
</head>
<body>
<div id="header">
  <h1>🔥 vigIA — Risco de Queimadas</h1>
  <span>Grade espacial 0.1° × 0.1° (~11km) | Goiás | Estágio 2</span>
</div>
<div id="controls">
  <span style="font-size:13px;color:#aaa;">Dia:</span>
  <div id="day-buttons"></div>
  <div id="filter-wrap">
    <label>Prob. mínima:</label>
    <input type="range" id="prob-slider" min="0" max="90" value="0" step="5">
    <span id="prob-val">0%</span>
  </div>
</div>
<div id="map"></div>
<div id="legend">
  <h4>Risco de Fogo</h4>
  <div class="leg-row"><div class="leg-dot" style="background:#e74c3c"></div> ALTO (≥70%)</div>
  <div class="leg-row"><div class="leg-dot" style="background:#f39c12"></div> MÉDIO (40-70%)</div>
  <div class="leg-row"><div class="leg-dot" style="background:#2ecc71"></div> BAIXO (&lt;40%)</div>
</div>
<div id="stats">
  <h4>Resumo do dia</h4>
  <p id="st-alto">🔴 ALTO: —</p>
  <p id="st-medio">🟠 MÉDIO: —</p>
  <p id="st-baixo">🟢 BAIXO: —</p>
  <p id="st-total" style="margin-top:6px;color:#aaa;">Total: —</p>
</div>

<script>
const DADOS = {data_json};
const DIAS  = {json.dumps(dias)};
const HALF  = 0.05;

const map = L.map('map', {{ center: [-15.5, -49.5], zoom: 7 }});
L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: 'Tiles &copy; Esri', maxZoom: 18 }}
).addTo(map);
L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: '', maxZoom: 18, opacity: 0.7 }}
).addTo(map);

let diaAtivo = DIAS[0]; let minProb = 0;
let camada = L.layerGroup().addTo(map);

function cor(p) {{
  return p>=0.70 ? '#e74c3c' : p>=0.40 ? '#f39c12' : '#2ecc71';
}}

function renderizar() {{
  camada.clearLayers();
  const pontos = (DADOS[diaAtivo]||[]).filter(p => p.prob >= minProb/100);
  let alto=0, medio=0, baixo=0;
  pontos.forEach(p => {{
    const bounds = [[p.lat-HALF,p.lon-HALF],[p.lat+HALF,p.lon+HALF]];
    const c = L.rectangle(bounds, {{
      fillColor: cor(p.prob), color: cor(p.prob),
      weight: 0.3, opacity: 0.6, fillOpacity: 0.55
    }});
    c.bindTooltip(
      `<b style="color:${{cor(p.prob)}}">${{(p.prob*100).toFixed(1)}}%</b> · ${{p.risco}}<br>` +
      `${{p.seco}}d sem chuva · ${{p.mun}}`,
      {{ sticky: true, opacity: 0.95 }}
    );
    c.bindPopup(`
      <b style="color:${{cor(p.prob)}};font-size:15px">${{(p.prob*100).toFixed(1)}}% — ${{p.risco}}</b><br><br>
      <b>Célula:</b> (${{p.lat}}, ${{p.lon}})<br>
      <b>Mun. proxy:</b> ${{p.mun}}<br>
      <b>Dias sem chuva:</b> ${{p.seco}}d<br>
      <b>Precipitação:</b> ${{p.prec}} mm
    `, {{ maxWidth: 240 }});
    camada.addLayer(c);
    if (p.risco==='ALTO') alto++;
    else if (p.risco==='MÉDIO') medio++;
    else baixo++;
  }});
  document.getElementById('st-alto').textContent  = '🔴 ALTO: '  + alto;
  document.getElementById('st-medio').textContent = '🟠 MÉDIO: ' + medio;
  document.getElementById('st-baixo').textContent = '🟢 BAIXO: ' + baixo;
  document.getElementById('st-total').textContent = 'Visíveis: ' + pontos.length + ' / ' + (DADOS[diaAtivo]||[]).length;
}}

const btnWrap = document.getElementById('day-buttons');
DIAS.forEach(dia => {{
  const btn = document.createElement('button');
  btn.className = 'day-btn' + (dia===diaAtivo ? ' active' : '');
  const d = new Date(dia + 'T12:00:00');
  btn.textContent = d.toLocaleDateString('pt-BR', {{weekday:'short',day:'2-digit',month:'2-digit'}});
  btn.onclick = () => {{
    diaAtivo = dia;
    document.querySelectorAll('.day-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderizar();
  }};
  btnWrap.appendChild(btn);
}});

document.getElementById('prob-slider').addEventListener('input', function() {{
  minProb = +this.value;
  document.getElementById('prob-val').textContent = minProb + '%';
  renderizar();
}});

renderizar();
</script>
</body>
</html>"""

saida = os.path.join(FRONTEND, "mapa_vigia.html")
with open(saida, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Gerado: {saida}")
print("Abra no navegador — não precisa de servidor.")
