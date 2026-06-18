"""
vigIA — Estágio 2 | Fase 3: Retreino 2015-2025 + Validação 2026 por célula
Entrada:  ../dados/dataset_grade.csv
          ../dados/mapeamento_grade.csv
          ../dados/bdqueimadas_2026-01-01_2026-06-03.csv
          ../dados/clima_2026.csv
Saída:    ../modelos/grade_full_corrigido.pkl   (produção)
          ../resultados/validacao_grade_2026.csv
"""

import os, warnings, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
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
    "Cell_Lat","Cell_Lon","Cell_Freq",
    "DiaSemChuva","Precipitacao","media_focos_mes_hist",
]

print("=" * 65)
print("  vigIA E2 — Fase 3: Retreino 2015-2025 + Validação 2026 Grade")
print("=" * 65)

# 1. Retreinar com LightGBM (produção: suporte nativo a NaN e inferência rápida)
print("\n[1/5] Retreinando LightGBM grade completo (2015-2025)...")
artefato = joblib.load(os.path.join(MODELOS, "grade_avaliacao.pkl"))
lgbm_params = artefato.get("lgbm_params", {})
ds = pd.read_csv(os.path.join(DADOS, "dataset_grade.csv"))
X_full, y_full = ds[FEATURES].values, ds["fogo"].values
print(f"  Amostras: {len(ds):,} | Positivos: {y_full.sum():,} ({100*y_full.mean():.1f}%)")
print(f"  Params LightGBM: {lgbm_params}")

modelo_full = lgbm.LGBMClassifier(
    **lgbm_params, class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
) if lgbm_params else lgbm.LGBMClassifier(
    subsample=0.8, num_leaves=31, n_estimators=200, max_depth=10,
    learning_rate=0.01, colsample_bytree=1.0,
    class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)

t0 = time.time()
modelo_full.fit(X_full, y_full)
print(f"  Treino concluído em {time.time()-t0:.0f}s")
joblib.dump({"modelo": modelo_full, "features": FEATURES, "nome": "LightGBM Grade Full 2015-2025"},
            os.path.join(MODELOS, "grade_full_corrigido.pkl"))
print("  Salvo: modelos/grade_full_corrigido.pkl")

# 2. Carregar dados 2026
print("\n[2/5] Carregando dados de 2026...")
grade = pd.read_csv(os.path.join(DADOS, "mapeamento_grade.csv"))
df26  = pd.read_csv(os.path.join(DADOS, "bdqueimadas_2026-01-01_2026-06-03.csv"), parse_dates=["DataHora"])
df26["Data"]     = df26["DataHora"].dt.date
df26["Cell_Lat"] = df26["Latitude"].round(1)
df26["Cell_Lon"] = df26["Longitude"].round(1)
celulas_validas  = set(zip(grade["Cell_Lat"], grade["Cell_Lon"]))
df26_valid = df26[df26.apply(lambda r: (r["Cell_Lat"],r["Cell_Lon"]) in celulas_validas, axis=1)]
positivos_2026 = set(zip(df26_valid["Cell_Lat"], df26_valid["Cell_Lon"],
                         pd.to_datetime(df26_valid["Data"])))
data_fim = pd.to_datetime(df26["Data"].max())
print(f"  {len(df26):,} focos | {len(positivos_2026):,} pares únicos (célula, dia)")

# 3. Grid validação 2026
print("\n[3/5] Gerando grid validação 2026...")
todos_dias = pd.date_range("2026-01-01", data_fim, freq="D")
print(f"  {len(grade):,} células × {len(todos_dias)} dias = {len(grade)*len(todos_dias):,}")

grid = pd.DataFrame(
    list(product(zip(grade["Cell_Lat"],grade["Cell_Lon"],grade["Cell_Freq"],grade["Nearest_Municipio"]), todos_dias)),
    columns=["cell_tuple","Data"])
grid[["Cell_Lat","Cell_Lon","Cell_Freq","Nearest_Municipio"]] = pd.DataFrame(grid["cell_tuple"].tolist(), index=grid.index)
grid = grid.drop(columns=["cell_tuple"])
grid["Mes"] = grid["Data"].dt.month; grid["DiaSemana"] = grid["Data"].dt.dayofweek
grid["Estacao_Seca"] = grid["Mes"].between(6,10).astype(int); grid["Ano"] = grid["Data"].dt.year
grid["fogo"] = grid.apply(
    lambda r: 1 if (r["Cell_Lat"],r["Cell_Lon"],r["Data"]) in positivos_2026 else 0, axis=1)
print(f"  Grid: {len(grid):,} linhas | Positivos: {grid['fogo'].sum():,} ({100*grid['fogo'].mean():.1f}%)")

hist_grade = (ds[ds["fogo"]==1].groupby(["Cell_Lat","Cell_Lon","Mes"]).size()
              .div(ds["Ano"].nunique()).reset_index(name="media_focos_mes_hist"))
grid = grid.merge(hist_grade, on=["Cell_Lat","Cell_Lon","Mes"], how="left")
grid["media_focos_mes_hist"] = grid["media_focos_mes_hist"].fillna(0)

# 4. Clima 2026
print("\n[4/5] Aplicando clima 2026 via município proxy...")
clima26 = pd.read_csv(os.path.join(DADOS, "clima_2026.csv"), parse_dates=["Data"])
clima26 = clima26.rename(columns={"Municipio": "Nearest_Municipio"})
grid = grid.merge(clima26[["Nearest_Municipio","Data","Precipitacao","DiaSemChuva"]],
                  on=["Nearest_Municipio","Data"], how="left")
n_sem_clima = grid["DiaSemChuva"].isna().sum()
if n_sem_clima > 0:
    print(f"  Aviso: {n_sem_clima:,} linhas sem dado climático — mantendo NaN (LightGBM trata nativamente).")

# 5. Prever e avaliar
print("\n[5/5] Prevendo e avaliando...")
prob = modelo_full.predict_proba(grid[FEATURES].values)[:, 1]
pred = (prob >= 0.5).astype(int)
y_2026 = grid["fogo"].values
auc    = roc_auc_score(y_2026, prob)
pr_auc = average_precision_score(y_2026, prob)
brier  = brier_score_loss(y_2026, prob)
rec    = recall_score(y_2026, pred)
prec   = precision_score(y_2026, pred)
f1     = f1_score(y_2026, pred)
cm     = confusion_matrix(y_2026, pred)
rec_03 = recall_score(y_2026, (prob>=0.3).astype(int))

print(f"\n  AUC-ROC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
print(f"  Recall@0.5: {rec:.4f} | Recall@0.3: {rec_03:.4f} | Precisão: {prec:.4f}")
print(f"  TN={cm[0,0]:,} FP={cm[0,1]:,} | FN={cm[1,0]:,} TP={cm[1,1]:,}")

pd.DataFrame([{"Modelo":"LightGBM Grade Full 2015-2025","AUC":auc,"PR_AUC":pr_auc,"Brier":brier,
               "F1":f1,"Precisao":prec,"Recall_05":rec,"Recall_03":rec_03,
               "TP":cm[1,1],"FP":cm[0,1],"TN":cm[0,0],"FN":cm[1,0]}
]).to_csv(os.path.join(RESULTADOS,"validacao_grade_2026.csv"),index=False)

fpr,tpr,_=roc_curve(y_2026,prob)
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(fpr,tpr,color="#e74c3c",linewidth=2,label=f"LightGBM Grade (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--",linewidth=0.8,label="Aleatório")
ax.set_xlabel("Taxa de Falso Positivo"); ax.set_ylabel("Taxa de Verdadeiro Positivo")
ax.set_title("Curva ROC — Grade 0.1° | Validação 2026 Goiás"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(GRAFICOS,"e2_validacao_2026_roc.png"),dpi=150); plt.close()

grid["prob_fogo"] = prob
prob_celula = grid.groupby(["Cell_Lat","Cell_Lon"])["prob_fogo"].mean().reset_index()
fig,ax=plt.subplots(figsize=(10,8))
sc=ax.scatter(prob_celula["Cell_Lon"],prob_celula["Cell_Lat"],c=prob_celula["prob_fogo"],
              cmap="RdYlGn_r",vmin=0,vmax=1,s=30,alpha=0.8)
plt.colorbar(sc,ax=ax,label="Prob. média de fogo (Jan-Jun 2026)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("vigIA E2 — Prob. Média por Célula 0.1° | Validação 2026")
fig.tight_layout(); fig.savefig(os.path.join(GRAFICOS,"e2_validacao_2026_mapa.png"),dpi=150); plt.close()

print(f"\n{'='*65}")
print(f"  AUC: {auc:.4f} | Recall@0.5: {rec:.4f} | Recall@0.3: {rec_03:.4f}")
print(f"  Salvo: modelos/grade_full_corrigido.pkl")
print(f"{'='*65}")
print("\n[OK] E2 Fase 3 concluída!")
