"""
vigIA — Estágio 1 | Fase 4: Previsão de risco dos próximos 5 dias por município
Entrada:  ../modelos/municipio_full.pkl
          ../dados/mapeamento_municipio.csv
          ../dados/dataset_municipio.csv
Saída:    ../resultados/previsao_municipio_<data>.csv
          ../graficos/e1_previsao_<data>.png
"""

import os, time, requests, warnings
from datetime import date, timedelta
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
LIMIAR_CHUVA = 0.1

FEATURES = [
    "Mes","DiaSemana","Estacao_Seca",
    "Latitude","Longitude","Municipio_Freq",
    "DiaSemChuva","Precipitacao","media_focos_mes_hist",
]

print("=" * 65)
print("  vigIA E1 — Previsão de Risco por Município — Próximos 5 Dias")
print("=" * 65)

print("\n[1/4] Carregando modelo e dados de referência...")
artefato = joblib.load(os.path.join(MODELOS, "municipio_full.pkl"))
modelo   = artefato["modelo"]
mapa = pd.read_csv(os.path.join(DADOS, "mapeamento_municipio.csv"))
hist = (pd.read_csv(os.path.join(DADOS, "dataset_municipio.csv"),
                    usecols=["Municipio","Mes","media_focos_mes_hist"])
        .drop_duplicates())

hoje      = date.today()
dias_prev = [hoje + timedelta(days=i) for i in range(1, 6)]
print(f"  Modelo: {artefato['nome']}")
print(f"  Hoje: {hoje} | Previsão: {dias_prev[0]} → {dias_prev[-1]}")

print("\n[2/4] Buscando dados climáticos (Open-Meteo)...")

def buscar_clima(nome, lat, lon, retries=4):
    for tentativa in range(retries):
        try:
            r_arc = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": round(lat,4), "longitude": round(lon,4),
                        "start_date": str(hoje-timedelta(days=30)),
                        "end_date": str(hoje-timedelta(days=1)),
                        "daily": "precipitation_sum", "timezone": "America/Sao_Paulo"},
                timeout=20)
            if r_arc.status_code == 429:
                espera = int(r_arc.headers.get("Retry-After", 60))
                print(f"\n    Rate limit — aguardando {espera}s...", end=" ", flush=True)
                time.sleep(espera); continue
            r_arc.raise_for_status()

            r_frc = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": round(lat,4), "longitude": round(lon,4),
                        "daily": "precipitation_sum", "forecast_days": 6,
                        "timezone": "America/Sao_Paulo"},
                timeout=20)
            r_frc.raise_for_status()

            prec_hist = r_arc.json()["daily"]["precipitation_sum"]
            dias_secos = 0
            for p in prec_hist:
                if p is None or pd.isna(p) or p < LIMIAR_CHUVA: dias_secos += 1
                else: dias_secos = 0

            frc_data = r_frc.json()["daily"]
            prec_prev = frc_data["precipitation_sum"][1:6]
            datas_prev = frc_data["time"][1:6]

            resultado = []
            contador = dias_secos
            for data_str, precip in zip(datas_prev, prec_prev):
                if precip is None or pd.isna(precip) or precip < LIMIAR_CHUVA: contador += 1
                else: contador = 0
                resultado.append({"Municipio": nome, "Data": pd.to_datetime(data_str).date(),
                                   "Precipitacao": precip if precip else 0.0,
                                   "DiaSemChuva": contador})
            return resultado
        except Exception:
            if tentativa < retries - 1: time.sleep(10 * (2**tentativa))
            else: return None

todos_registros = []
erros = []
for i, row in mapa.iterrows():
    print(f"  [{i+1:3d}/{len(mapa)}] {row['Municipio']:<35}", end=" ", flush=True)
    resultado = buscar_clima(row["Municipio"], row["Latitude"], row["Longitude"])
    if resultado:
        todos_registros.extend(resultado)
        print(f"✓ DiaSemChuva={resultado[0]['DiaSemChuva']}")
    else:
        erros.append(row["Municipio"]); print("✗ ERRO")
    time.sleep(1.2)

print("\n[3/4] Calculando features e gerando ranking...")
clima_df = pd.DataFrame(todos_registros)
clima_df["Data"] = pd.to_datetime(clima_df["Data"])
clima_df["Mes"]          = clima_df["Data"].dt.month
clima_df["DiaSemana"]    = clima_df["Data"].dt.dayofweek
clima_df["Estacao_Seca"] = clima_df["Mes"].between(6,10).astype(int)
clima_df = clima_df.merge(mapa[["Municipio","Municipio_Freq","Latitude","Longitude"]],
                          on="Municipio", how="left")
clima_df = clima_df.merge(hist, on=["Municipio","Mes"], how="left")
clima_df["media_focos_mes_hist"] = clima_df["media_focos_mes_hist"].fillna(0)
clima_df["prob_fogo"] = modelo.predict_proba(clima_df[FEATURES].values)[:, 1]
clima_df["risco"] = clima_df["prob_fogo"].apply(
    lambda p: "ALTO" if p >= 0.70 else "MÉDIO" if p >= 0.40 else "BAIXO")

nome_csv = f"previsao_municipio_{hoje}.csv"
clima_df.sort_values(["Data","prob_fogo"], ascending=[True,False]) \
        .to_csv(os.path.join(RESULTADOS, nome_csv), index=False)

print("\n[4/4] Ranking e gráfico...\n")
for dia in dias_prev:
    dia_df = clima_df[clima_df["Data"].dt.date == dia].sort_values("prob_fogo", ascending=False)
    alto = (dia_df["risco"]=="ALTO").sum(); medio = (dia_df["risco"]=="MÉDIO").sum()
    print(f"  {'─'*58}")
    print(f"  {dia.strftime('%d/%m/%Y (%A)')}  |  ALTO: {alto}  MÉDIO: {medio}")
    print(f"  {'─'*58}")
    print(f"  {'#':>3} {'Município':<32} {'Prob':>6}  {'Risco':<8} {'Seco(dias)':>10}")
    for j, (_, r) in enumerate(dia_df.head(10).iterrows(), 1):
        emoji = "🔴" if r.risco=="ALTO" else "🟠" if r.risco=="MÉDIO" else "🟢"
        print(f"  {j:>3} {r.Municipio:<32} {r.prob_fogo:>5.1%}  {emoji} {r.risco:<6} {r.DiaSemChuva:>8.0f}d")
    print()

top_mun = (clima_df.groupby("Municipio")["prob_fogo"].mean()
           .sort_values(ascending=False).head(20).index.tolist())
pivot = (clima_df[clima_df["Municipio"].isin(top_mun)]
         .pivot(index="Municipio", columns="Data", values="prob_fogo"))
pivot = pivot.loc[top_mun]
pivot.columns = [d.strftime("%d/%m") for d in pivot.columns]

fig, ax = plt.subplots(figsize=(12,8))
cmap = mcolors.LinearSegmentedColormap.from_list("risco",["#2ecc71","#f39c12","#e74c3c"])
im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i,j]
        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                fontsize=8, color="white" if val>0.5 else "black", fontweight="bold")
plt.colorbar(im, ax=ax, label="Probabilidade de fogo")
ax.set_title(f"vigIA E1 — Risco por Município | {dias_prev[0].strftime('%d/%m')} → {dias_prev[-1].strftime('%d/%m/%Y')}",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, f"e1_previsao_{hoje}.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"{'='*65}")
print(f"  resultados/{nome_csv}")
print(f"  graficos/e1_previsao_{hoje}.png")
print(f"{'='*65}")
print("\n[OK] E1 Fase 4 concluída!")
