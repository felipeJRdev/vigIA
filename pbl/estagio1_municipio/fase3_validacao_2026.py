"""
vigIA — Estágio 1 | Fase 3: Retreino 2015-2025 + Validação 2026
Entrada:  ../dados/dataset_municipio.csv
          ../dados/mapeamento_municipio.csv
          ../dados/bdqueimadas_2026-01-01_2026-06-03.csv
Saída:    ../modelos/municipio_full.pkl   (produção)
          ../dados/clima_2026.csv
          ../resultados/dataset_validacao_2026.csv
          ../resultados/validacao_municipio_2026.csv
"""

import os, time, requests, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_curve)
import lightgbm as lgbm
from itertools import product

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = os.path.dirname(_HERE)
DADOS      = os.path.join(PBL, "dados")
MODELOS    = os.path.join(PBL, "modelos")
RESULTADOS = os.path.join(PBL, "resultados")
GRAFICOS   = os.path.join(PBL, "graficos")
os.makedirs(GRAFICOS, exist_ok=True)

FEATURES = [
    "Mes","DiaSemana","Estacao_Seca",
    "Latitude","Longitude","Municipio_Freq",
    "DiaSemChuva","Precipitacao","media_focos_mes_hist",
]

print("=" * 65)
print("  vigIA E1 — Fase 3: Retreino 2015-2025 + Validação 2026")
print("=" * 65)

# 1. Retreinar
print("\n[1/6] Retreinando LightGBM completo (2015-2025)...")
artefato = joblib.load(os.path.join(MODELOS, "municipio_avaliacao.pkl"))
params   = artefato.get("params", {})
ds = pd.read_csv(os.path.join(DADOS, "dataset_municipio.csv"))
X_full, y_full = ds[FEATURES].values, ds["fogo"].values
print(f"  Amostras: {len(ds):,} | Positivos: {y_full.sum():,} ({100*y_full.mean():.1f}%)")

modelo_full = lgbm.LGBMClassifier(
    **{k: v for k, v in params.items() if k in [
        "n_estimators","max_depth","learning_rate","num_leaves","subsample","colsample_bytree"]},
    class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
)
if not params:
    modelo_full = lgbm.LGBMClassifier(
        subsample=1.0, num_leaves=50, n_estimators=200, max_depth=-1,
        learning_rate=0.05, colsample_bytree=0.8,
        class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)

t0 = time.time()
modelo_full.fit(X_full, y_full)
print(f"  Concluído em {time.time()-t0:.0f}s")
joblib.dump({"modelo": modelo_full, "features": FEATURES, "nome": "LightGBM Município Full 2015-2025"},
            os.path.join(MODELOS, "municipio_full.pkl"))
print("  Salvo: modelos/municipio_full.pkl")

# 2. Carregar dados 2026
print("\n[2/6] Carregando focos reais 2026...")
df26 = pd.read_csv(os.path.join(DADOS, "bdqueimadas_2026-01-01_2026-06-03.csv"), parse_dates=["DataHora"])
df26["Data"] = df26["DataHora"].dt.date
positivos_2026 = set(zip(df26["Municipio"].str.upper(), df26["Data"]))
data_fim = df26["Data"].max()
print(f"  {len(df26):,} focos | {len(positivos_2026):,} pares únicos (município, dia)")

# 3. Grid 2026
print("\n[3/6] Gerando grid de validação 2026...")
mapa = pd.read_csv(os.path.join(DADOS, "mapeamento_municipio.csv"))
data_inicio = pd.to_datetime("2026-01-01").date()
todos_dias  = pd.date_range(data_inicio, data_fim, freq="D").date
grid = pd.DataFrame(list(product(mapa["Municipio"].values, todos_dias)), columns=["Municipio","Data"])
grid["fogo"] = grid.apply(lambda r: 1 if (r["Municipio"], r["Data"]) in positivos_2026 else 0, axis=1)
grid["Data"] = pd.to_datetime(grid["Data"])
grid["Mes"] = grid["Data"].dt.month; grid["DiaSemana"] = grid["Data"].dt.dayofweek
grid["Estacao_Seca"] = grid["Mes"].between(6,10).astype(int)
grid = grid.merge(mapa[["Municipio","Municipio_Freq","Latitude","Longitude"]], on="Municipio", how="left")
hist = pd.read_csv(os.path.join(DADOS, "dataset_municipio.csv"),
                   usecols=["Municipio","Mes","media_focos_mes_hist"]).drop_duplicates()
grid = grid.merge(hist, on=["Municipio","Mes"], how="left")
grid["media_focos_mes_hist"] = grid["media_focos_mes_hist"].fillna(0)
print(f"  {len(grid):,} linhas | Positivos: {grid['fogo'].sum():,} ({100*grid['fogo'].mean():.1f}%)")

# 4. Clima 2026
print("\n[4/6] Baixando clima 2026 (Open-Meteo)...")
CLIMA_2026 = os.path.join(DADOS, "clima_2026.csv")
LIMIAR = 0.1

def calc_dias_sem_chuva(series, limiar=LIMIAR):
    dias, cont = [], 0
    for p in series:
        if pd.isna(p) or p < limiar: cont += 1
        else: cont = 0
        dias.append(cont)
    return dias

def baixar_clima_2026(nome, lat, lon, retries=5):
    for t in range(retries):
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": round(lat,4), "longitude": round(lon,4),
                        "start_date": "2025-12-01", "end_date": str(data_fim),
                        "daily": "precipitation_sum", "timezone": "America/Sao_Paulo"},
                timeout=30)
            if r.status_code == 429:
                espera = int(r.headers.get("Retry-After", 60))
                print(f"\n    Rate limit — aguardando {espera}s...", end=" ", flush=True)
                time.sleep(espera); continue
            r.raise_for_status()
            dados = r.json()["daily"]
            df = pd.DataFrame({"Municipio": nome, "Data": pd.to_datetime(dados["time"]),
                                "Precipitacao": dados["precipitation_sum"]})
            df["DiaSemChuva"] = calc_dias_sem_chuva(df["Precipitacao"])
            return df[df["Data"].dt.year == 2026]
        except Exception as e:
            if t < retries - 1: time.sleep(15 * (2**t))
            else: print(f"ERRO: {e}"); return None

municipios_feitos = set()
if os.path.exists(CLIMA_2026):
    feitos = pd.read_csv(CLIMA_2026, usecols=["Municipio"])["Municipio"].unique()
    municipios_feitos = set(feitos)
    print(f"  Retomando: {len(municipios_feitos)} municípios já baixados.")

for i, row in mapa.iterrows():
    nome = row["Municipio"]
    if nome in municipios_feitos: continue
    print(f"  [{i+1:3d}/{len(mapa)}] {nome:<35}", end=" ", flush=True)
    df_m = baixar_clima_2026(nome, row["Latitude"], row["Longitude"])
    if df_m is not None:
        modo = "a" if os.path.exists(CLIMA_2026) else "w"
        df_m.to_csv(CLIMA_2026, mode=modo, header=not os.path.exists(CLIMA_2026), index=False)
        print(f"✓ max_seco={df_m['DiaSemChuva'].max()}")
    else:
        print("✗")
    time.sleep(1.5)

print("\n  Aplicando clima no grid...")
clima26 = pd.read_csv(CLIMA_2026, parse_dates=["Data"])
grid = grid.merge(clima26[["Municipio","Data","Precipitacao","DiaSemChuva"]],
                  on=["Municipio","Data"], how="left", suffixes=("_drop",""))
for col in ["Precipitacao_drop","DiaSemChuva_drop"]:
    if col in grid.columns: grid.drop(columns=[col], inplace=True)
grid["DiaSemChuva"] = grid["DiaSemChuva"].fillna(0)
grid["Precipitacao"] = grid["Precipitacao"].fillna(0)
grid.to_csv(os.path.join(RESULTADOS, "dataset_validacao_2026.csv"), index=False)

# 5. Avaliar
print("\n[5/6] Prevendo e avaliando...")
X_2026 = grid[FEATURES].values; y_2026 = grid["fogo"].values
prob = modelo_full.predict_proba(X_2026)[:, 1]
pred = (prob >= 0.5).astype(int)
auc = roc_auc_score(y_2026, prob); rec = recall_score(y_2026, pred)
prec = precision_score(y_2026, pred); f1 = f1_score(y_2026, pred)
cm = confusion_matrix(y_2026, pred); rec_03 = recall_score(y_2026, (prob>=0.3).astype(int))

print(f"\n  AUC-ROC: {auc:.4f} | Recall@0.5: {rec:.4f} | Recall@0.3: {rec_03:.4f}")
print(f"  Precisão: {prec:.4f} | F1: {f1:.4f}")
print(f"  TN={cm[0,0]:,} FP={cm[0,1]:,} | FN={cm[1,0]:,} TP={cm[1,1]:,}")

pd.DataFrame([{"Modelo":"LightGBM Município Full 2015-2025",
               "AUC":auc,"F1":f1,"Precisao":prec,"Recall_05":rec,"Recall_03":rec_03,
               "TP":cm[1,1],"FP":cm[0,1],"TN":cm[0,0],"FN":cm[1,0]}
]).to_csv(os.path.join(RESULTADOS, "validacao_municipio_2026.csv"), index=False)

# 6. Gráficos
print("\n[6/6] Gerando gráficos...")
fpr, tpr, _ = roc_curve(y_2026, prob)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(fpr, tpr, color="#e74c3c", linewidth=2, label=f"LightGBM (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--",linewidth=0.8,label="Aleatório")
ax.set_xlabel("Taxa de Falso Positivo"); ax.set_ylabel("Taxa de Verdadeiro Positivo")
ax.set_title("Curva ROC — Validação Município 2026 | Goiás")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e1_validacao_2026_roc.png"), dpi=150); plt.close()

grid["prob_fogo"] = prob
top_mun = grid.groupby("Municipio")["prob_fogo"].mean().sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10,7))
top_mun[::-1].plot(kind="barh", ax=ax, color="#e74c3c")
ax.set_title("Top 20 Municípios — Prob. Média de Fogo (Jan-Jun 2026)")
fig.tight_layout(); fig.savefig(os.path.join(GRAFICOS, "e1_validacao_2026_top_municipios.png"), dpi=150); plt.close()

print(f"\n{'='*65}")
print(f"  AUC-ROC: {auc:.4f} | Recall@0.5: {rec:.4f} | Recall@0.3: {rec_03:.4f}")
print(f"  Salvo: modelos/municipio_full.pkl")
print(f"{'='*65}")
print("\n[OK] E1 Fase 3 concluída!")
