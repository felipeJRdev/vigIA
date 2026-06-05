"""
vigIA — Estágio 2 | Fase 2: Modelagem por célula 0.1° × 0.1°
Entrada:  ../dados/dataset_grade.csv
Saída:    ../modelos/grade_avaliacao.pkl
          ../resultados/resultados_grade_fase2.csv

Split temporal: Treino 2015-2022 | Val 2023 | Teste 2024-2025
"""

import os, time, warnings
import numpy as np
import pandas as pd
import joblib
import mlflow, mlflow.lightgbm, mlflow.sklearn, mlflow.xgboost
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
import lightgbm as lgbm

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = os.path.dirname(_HERE)
DADOS      = os.path.join(PBL, "dados")
MODELOS    = os.path.join(PBL, "modelos")
RESULTADOS = os.path.join(PBL, "resultados")
GRAFICOS   = os.path.join(PBL, "graficos")
MLRUNS     = os.path.join(PBL, "mlruns")
os.makedirs(GRAFICOS, exist_ok=True)

FEATURES = [
    "Mes","DiaSemana","Estacao_Seca",
    "Cell_Lat","Cell_Lon","Cell_Freq",
    "DiaSemChuva","Precipitacao","media_focos_mes_hist",
]
TARGET        = "fogo"
SEED          = 42
SEARCH_SAMPLE = 200_000

print("=" * 65)
print("  vigIA E2 — Fase 2: Modelagem por Célula 0.1°")
print("=" * 65)

print("\n[1/5] Carregando dataset_grade.csv...")
ds = pd.read_csv(os.path.join(DADOS, "dataset_grade.csv"), parse_dates=["Data"])
print(f"  Total: {len(ds):,} | Positivos: {(ds[TARGET]==1).sum():,} ({100*(ds[TARGET]==1).mean():.1f}%)")
print(f"  Células: {ds[['Cell_Lat','Cell_Lon']].drop_duplicates().shape[0]:,}")

print("\n[2/5] Split temporal...")
treino = ds[ds["Ano"] <= 2022]; val = ds[ds["Ano"]==2023]; teste = ds[ds["Ano"]>=2024]
X_tr, y_tr   = treino[FEATURES].values, treino[TARGET].values
X_val, y_val = val[FEATURES].values,    val[TARGET].values
X_te,  y_te  = teste[FEATURES].values,  teste[TARGET].values
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_tr), size=min(SEARCH_SAMPLE, len(X_tr)), replace=False)
X_search, y_search = X_tr[idx], y_tr[idx]
print(f"  Treino: {len(treino):,} | Val: {len(val):,} | Teste: {len(teste):,} | Busca: {len(X_search):,}")

print("\n[3/5] Treinamento e busca de hiperparâmetros...")
mlflow.set_tracking_uri("file://" + MLRUNS)
mlflow.set_experiment("vigIA_E2_grade")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def avaliar(nome, modelo, params, elapsed):
    prob_val = modelo.predict_proba(X_val)[:,1]; prob_te = modelo.predict_proba(X_te)[:,1]
    pred_te  = (prob_te >= 0.5).astype(int)
    m = {"val_auc": roc_auc_score(y_val, prob_val), "te_auc": roc_auc_score(y_te, prob_te),
         "te_f1": f1_score(y_te, pred_te), "te_prec": precision_score(y_te, pred_te),
         "te_rec": recall_score(y_te, pred_te)}
    with mlflow.start_run(run_name=nome):
        mlflow.log_params(params); mlflow.log_metric("tempo_s", elapsed)
        for k, v in m.items(): mlflow.log_metric(k, v)
    print(f"\n  {nome}")
    print(f"    Val → AUC={m['val_auc']:.4f}")
    print(f"    Teste→ AUC={m['te_auc']:.4f} | F1={m['te_f1']:.4f} | Rec={m['te_rec']:.4f} | Prec={m['te_prec']:.4f}")
    return m

resultados = []

print("\n  [1/3] Random Forest...")
param_rf = {"n_estimators":[100,200,300],"max_depth":[10,20,30],"min_samples_split":[2,5,10],
            "min_samples_leaf":[1,2,4],"max_features":["sqrt","log2"]}
t0 = time.time()
search_rf = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=SEED),
    param_rf, n_iter=15, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=SEED, verbose=1)
search_rf.fit(X_search, y_search)
rf_best = RandomForestClassifier(**search_rf.best_params_, class_weight="balanced", n_jobs=-1, random_state=SEED)
rf_best.fit(X_tr, y_tr)
print(f"    Melhor AUC CV: {search_rf.best_score_:.4f} | {search_rf.best_params_}")
m_rf = avaliar("Random Forest", rf_best, search_rf.best_params_, time.time()-t0)
resultados.append({"Modelo":"Random Forest",**m_rf})

print("\n  [2/3] XGBoost...")
scale_pos = int((y_tr==0).sum()/max((y_tr==1).sum(),1))
param_xgb = {"n_estimators":[100,200,300],"max_depth":[3,5,7],"learning_rate":[0.01,0.05,0.1],
             "subsample":[0.8,1.0],"colsample_bytree":[0.8,1.0]}
try:
    base_xgb = xgb.XGBClassifier(scale_pos_weight=scale_pos,device="cuda",tree_method="hist",
                                   random_state=SEED,verbosity=0,eval_metric="auc")
    base_xgb.fit(X_tr[:100],y_tr[:100]); gpu_ok=True; print("    GPU disponível.")
except Exception:
    base_xgb = xgb.XGBClassifier(scale_pos_weight=scale_pos,tree_method="hist",
                                   random_state=SEED,verbosity=0,eval_metric="auc"); gpu_ok=False
t0 = time.time()
search_xgb = RandomizedSearchCV(base_xgb,param_xgb,n_iter=20,cv=cv,scoring="roc_auc",
                                 n_jobs=1,random_state=SEED,verbose=1)
search_xgb.fit(X_search,y_search)
best_p = search_xgb.best_params_.copy()
if gpu_ok: best_p.update({"device":"cuda","tree_method":"hist"})
xgb_best = xgb.XGBClassifier(**best_p,scale_pos_weight=scale_pos,random_state=SEED,verbosity=0,eval_metric="auc")
xgb_best.fit(X_tr,y_tr)
print(f"    Melhor AUC CV: {search_xgb.best_score_:.4f} | {search_xgb.best_params_}")
m_xgb = avaliar("XGBoost",xgb_best,search_xgb.best_params_,time.time()-t0)
resultados.append({"Modelo":"XGBoost",**m_xgb})

print("\n  [3/3] LightGBM...")
param_lgbm = {"n_estimators":[100,200,300],"max_depth":[-1,10,20],"learning_rate":[0.01,0.05,0.1],
              "num_leaves":[31,50,100],"subsample":[0.8,1.0],"colsample_bytree":[0.8,1.0]}
t0 = time.time()
search_lgbm = RandomizedSearchCV(
    lgbm.LGBMClassifier(class_weight="balanced",n_jobs=1,random_state=SEED,verbose=-1),
    param_lgbm,n_iter=20,cv=cv,scoring="roc_auc",n_jobs=-1,random_state=SEED,verbose=1)
search_lgbm.fit(X_search,y_search)
lgbm_best = lgbm.LGBMClassifier(**search_lgbm.best_params_,class_weight="balanced",
                                   n_jobs=-1,random_state=SEED,verbose=-1)
lgbm_best.fit(X_tr,y_tr)
print(f"    Melhor AUC CV: {search_lgbm.best_score_:.4f} | {search_lgbm.best_params_}")
m_lgbm = avaliar("LightGBM",lgbm_best,search_lgbm.best_params_,time.time()-t0)
resultados.append({"Modelo":"LightGBM",**m_lgbm})

print("\n[4/5] Salvando melhor modelo...")
df_res = pd.DataFrame(resultados).sort_values("te_auc",ascending=False)
df_res.to_csv(os.path.join(RESULTADOS,"resultados_grade_fase2.csv"),index=False)
melhor_nome = df_res.iloc[0]["Modelo"]
melhor_mod  = {"Random Forest":rf_best,"XGBoost":xgb_best,"LightGBM":lgbm_best}[melhor_nome]
melhor_params = {"Random Forest":search_rf.best_params_,"XGBoost":search_xgb.best_params_,
                 "LightGBM":search_lgbm.best_params_}[melhor_nome]
joblib.dump({"modelo":melhor_mod,"features":FEATURES,"nome":melhor_nome,"params":melhor_params},
            os.path.join(MODELOS,"grade_avaliacao.pkl"))
print(f"  Melhor: {melhor_nome} (AUC Teste={df_res.iloc[0]['te_auc']:.4f})")

print("\n[5/5] Gráfico comparativo...")
fig,ax = plt.subplots(figsize=(10,5)); x=np.arange(len(df_res)); w=0.35
ax.bar(x-w/2,df_res["val_auc"],w,label="Val (2023)",color="#3498db")
ax.bar(x+w/2,df_res["te_auc"], w,label="Teste (2024-25)",color="#e74c3c")
ax.set_xticks(x); ax.set_xticklabels(df_res["Modelo"])
ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.5,1.0)
ax.set_title("vigIA E2 — AUC-ROC por Modelo | Grade 0.1°"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(GRAFICOS,"e2_comparacao_modelos.png"),dpi=150); plt.close()

print(f"\n{'='*65}")
print(df_res[["Modelo","val_auc","te_auc","te_f1","te_prec","te_rec"]].to_string(index=False))
print(f"{'='*65}")
print("\n[OK] E2 Fase 2 concluída!")
