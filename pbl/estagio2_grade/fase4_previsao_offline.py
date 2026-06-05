"""
vigIA — Estágio 2 | Fase 4 (offline): Previsão por célula sem novas chamadas API.
Reutiliza resultados/previsao_municipio_*.csv como proxy climático.
Entrada:  ../modelos/grade_full.pkl
          ../dados/mapeamento_grade.csv
          ../dados/dataset_grade.csv
          ../resultados/previsao_municipio_<data>.csv
Saída:    ../resultados/previsao_grade_<data>.csv
          ../graficos/e2_previsao_<data>.png
"""

import os, warnings, glob
from datetime import date
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = os.path.dirname(_HERE)
DADOS      = os.path.join(PBL, "dados")
MODELOS    = os.path.join(PBL, "modelos")
RESULTADOS = os.path.join(PBL, "resultados")
GRAFICOS   = os.path.join(PBL, "graficos")

FEATURES = [
    "Mes","DiaSemana","Estacao_Seca",
    "Cell_Lat","Cell_Lon","Cell_Freq",
    "DiaSemChuva","Precipitacao","media_focos_mes_hist",
]

print("=" * 65)
print("  vigIA E2 — Previsão Grade (offline) — Próximos 5 Dias")
print("=" * 65)

print("\n[1/4] Carregando modelo grade e referências...")
artefato = joblib.load(os.path.join(MODELOS, "grade_full.pkl"))
modelo   = artefato["modelo"]
grade = pd.read_csv(os.path.join(DADOS, "mapeamento_grade.csv"))
print(f"  Modelo: {artefato['nome']} | Células: {len(grade):,}")

hist = (pd.read_csv(os.path.join(DADOS, "dataset_grade.csv"),
                    usecols=["Cell_Lat","Cell_Lon","Mes","fogo","media_focos_mes_hist"])
        .query("fogo == 1")
        .groupby(["Cell_Lat","Cell_Lon","Mes"])["media_focos_mes_hist"]
        .mean().reset_index())

print("\n[2/4] Carregando previsão municipal disponível...")
prev_files = sorted([f for f in glob.glob(os.path.join(RESULTADOS,"previsao_municipio_*.csv"))])
if not prev_files:
    raise FileNotFoundError("Nenhum previsao_municipio_*.csv em resultados/. Rode E1 Fase 4 primeiro.")
prev_path = prev_files[-1]
print(f"  Usando: {os.path.basename(prev_path)}")
prev_mun = pd.read_csv(prev_path, parse_dates=["Data"])
clima_prev = prev_mun[["Municipio","Data","DiaSemChuva","Precipitacao"]].copy()

print("\n[3/4] Expandindo para células e gerando previsão...")
cells_clima = grade[["Cell_Lat","Cell_Lon","Cell_Freq","Nearest_Municipio"]].merge(
    clima_prev.rename(columns={"Municipio":"Nearest_Municipio"}),
    on="Nearest_Municipio", how="inner")
cells_clima["Mes"]          = cells_clima["Data"].dt.month
cells_clima["DiaSemana"]    = cells_clima["Data"].dt.dayofweek
cells_clima["Estacao_Seca"] = cells_clima["Mes"].between(6,10).astype(int)
cells_clima = cells_clima.merge(hist, on=["Cell_Lat","Cell_Lon","Mes"], how="left")
cells_clima["media_focos_mes_hist"] = cells_clima["media_focos_mes_hist"].fillna(0)

cells_clima["prob_fogo"] = modelo.predict_proba(cells_clima[FEATURES].values)[:, 1]
cells_clima["risco"] = cells_clima["prob_fogo"].apply(
    lambda p: "ALTO" if p>=0.70 else "MÉDIO" if p>=0.40 else "BAIXO")

hoje = date.today()
nome_csv = f"previsao_grade_{hoje}.csv"
cells_clima.sort_values(["Data","prob_fogo"],ascending=[True,False]) \
           .to_csv(os.path.join(RESULTADOS, nome_csv), index=False)

print("\n[4/4] Gerando mapa geográfico de Goiás...")
dias_prev = sorted(cells_clima["Data"].dt.date.unique())
fig, axes = plt.subplots(1, len(dias_prev), figsize=(5*len(dias_prev),8), sharex=True, sharey=True)
if len(dias_prev) == 1: axes = [axes]
cmap = mcolors.LinearSegmentedColormap.from_list("risco",["#2ecc71","#f39c12","#e74c3c"])

for ax, dia in zip(axes, dias_prev):
    day_df = cells_clima[cells_clima["Data"].dt.date == dia]
    sc = ax.scatter(day_df["Cell_Lon"], day_df["Cell_Lat"],
                    c=day_df["prob_fogo"], cmap=cmap, vmin=0, vmax=1, s=18, alpha=0.85, linewidths=0)
    alto=(day_df["risco"]=="ALTO").sum(); medio=(day_df["risco"]=="MÉDIO").sum()
    ax.set_title(f"{pd.Timestamp(dia).strftime('%d/%m (%a)')}\nALTO:{alto} MED:{medio}", fontsize=9)
    ax.set_xlabel("Longitude",fontsize=8); ax.tick_params(labelsize=7)

axes[0].set_ylabel("Latitude")
plt.colorbar(sc, ax=axes[-1], label="P(fogo)", shrink=0.8)
fig.suptitle(f"vigIA E2 — Risco por Célula 0.1° | Goiás\n"
             f"{dias_prev[0].strftime('%d/%m')} → {dias_prev[-1].strftime('%d/%m/%Y')}",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93])
nome_png = f"e2_previsao_{hoje}.png"
fig.savefig(os.path.join(GRAFICOS, nome_png), dpi=150, bbox_inches="tight"); plt.close()

print()
for dia in dias_prev:
    day_df = cells_clima[cells_clima["Data"].dt.date==dia].sort_values("prob_fogo",ascending=False)
    alto=(day_df["risco"]=="ALTO").sum(); medio=(day_df["risco"]=="MÉDIO").sum()
    print(f"  {'─'*58}")
    print(f"  {pd.Timestamp(dia).strftime('%d/%m/%Y (%A)')}  |  ALTO: {alto}  MÉDIO: {medio}")
    print(f"  {'─'*58}")
    print(f"  {'#':>3} {'Lat':>6} {'Lon':>7}  {'Prob':>6}  {'Seco':>6}")
    for j,(_,r) in enumerate(day_df.head(10).iterrows(),1):
        print(f"  {j:>3} {r.Cell_Lat:>6.1f} {r.Cell_Lon:>7.1f}  {r.prob_fogo:>5.1%}  {r.DiaSemChuva:>5.0f}d")
    print()

print(f"{'='*65}")
print(f"  resultados/{nome_csv}")
print(f"  graficos/{nome_png}")
print(f"{'='*65}")
print("\n[OK] E2 Fase 4 (offline) concluída!")
