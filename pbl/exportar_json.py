"""
exportar_json.py — Converte previsão CSV → forecast.json para o frontend vigIA.
Lê:    resultados/previsao_municipio_<data>.csv  (E1)
       resultados/previsao_grade_<data>.csv      (E2)
Salva: pbl/forecast.json
       vigIA/frontend/forecast.json
"""

import os, glob, json
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = _HERE
RAIZ       = os.path.dirname(PBL)
RESULTADOS = os.path.join(PBL, "resultados")
DADOS      = os.path.join(PBL, "dados")

# ── Localiza CSVs mais recentes ──────────────────────────────────────────────
mun_files   = sorted(glob.glob(os.path.join(RESULTADOS, "previsao_municipio_*.csv")))
grade_files = sorted(glob.glob(os.path.join(RESULTADOS, "previsao_grade_*.csv")))

if not mun_files:
    raise FileNotFoundError("Nenhum previsao_municipio_*.csv em resultados/")
if not grade_files:
    raise FileNotFoundError("Nenhum previsao_grade_*.csv em resultados/")

print(f"Municípios : {os.path.basename(mun_files[-1])}")
print(f"Grade      : {os.path.basename(grade_files[-1])}")

mun_df   = pd.read_csv(mun_files[-1],   parse_dates=["Data"])
grade_df = pd.read_csv(grade_files[-1], parse_dates=["Data"])

dias = sorted(mun_df["Data"].dt.strftime("%Y-%m-%d").unique().tolist())

# ── Municípios ───────────────────────────────────────────────────────────────
import unicodedata

def _norm(s):
    nfd = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in nfd if not unicodedata.category(c).startswith("M")).lower().strip()

municipios = {}
for dia in dias:
    df = mun_df[mun_df["Data"].dt.strftime("%Y-%m-%d") == dia] \
         .sort_values("prob_fogo", ascending=False)
    seen = {}
    for _, row in df.iterrows():
        key = _norm(row["Municipio"])
        if key in seen:
            continue  # mantém só a entrada com maior prob (já ordenado desc)
        seen[key] = True
        if not municipios.get(dia):
            municipios[dia] = []
        municipios[dia].append({
            "nome": row["Municipio"],
            "lat":  round(float(row["Latitude"]),  4),
            "lon":  round(float(row["Longitude"]), 4),
            "prob": round(float(row["prob_fogo"]), 4),
            "risco": row["risco"],
            "seco": int(row["DiaSemChuva"]),
        })

# ── Células da grade ─────────────────────────────────────────────────────────
celulas = {}
for dia in dias:
    df = grade_df[grade_df["Data"].dt.strftime("%Y-%m-%d") == dia] \
         .sort_values("prob_fogo", ascending=False)
    celulas[dia] = [
        {
            "lat":  round(float(row["Cell_Lat"]), 1),
            "lon":  round(float(row["Cell_Lon"]), 1),
            "prob": round(float(row["prob_fogo"]), 4),
            "risco": row["risco"],
            "mun":  row["Nearest_Municipio"],
            "seco": int(row["DiaSemChuva"]),
        }
        for _, row in df.iterrows()
    ]

out = {
    "gerado_em": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(timespec="minutes"),
    "dias": dias,
    "municipios": municipios,
    "celulas": celulas,
}

# ── Salva ────────────────────────────────────────────────────────────────────
dest = os.path.join(PBL, "forecast.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print(f"Salvo: {dest}  ({os.path.getsize(dest)/1024:.0f} KB)")
print("Para atualizar o frontend: copie forecast.json para vigIA/frontend/forecast.json")

print(f"\ngerado_em : {out['gerado_em']}  (horário de Brasília)")
print(f"dias      : {dias[0]} → {dias[-1]}")
print(f"municípios: {len(municipios[dias[0]])} por dia")
print(f"células   : {len(celulas[dias[0]])} por dia")
