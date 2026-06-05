"""
vigIA — Estágio 1 | Fase 3b: Gráficos de comparação previsão × realidade (2026)
Entrada:  ../modelos/municipio_full.pkl
          ../dados/mapeamento_municipio.csv
          ../dados/dataset_municipio.csv
          ../dados/bdqueimadas_2026-01-01_2026-06-03.csv
          ../dados/clima_2026.csv
Saída:    ../graficos/e1_comparacao_*.png  (4 gráficos)
"""

import os, warnings
from itertools import product
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = os.path.dirname(_HERE)
DADOS      = os.path.join(PBL, "dados")
MODELOS    = os.path.join(PBL, "modelos")
GRAFICOS   = os.path.join(PBL, "graficos")
os.makedirs(GRAFICOS, exist_ok=True)

FEATURES = [
    "Mes", "DiaSemana", "Estacao_Seca",
    "Latitude", "Longitude", "Municipio_Freq",
    "DiaSemChuva", "Precipitacao", "media_focos_mes_hist",
]

print("=" * 65)
print("  vigIA E1 — Fase 3b: Gráficos Previsão × Realidade 2026")
print("=" * 65)

# ── 1. Reconstruir grid 2026 ──────────────────────────────────────────
print("\n[1/5] Carregando modelo e reconstruindo grid 2026...")
artefato = joblib.load(os.path.join(MODELOS, "municipio_full.pkl"))
modelo   = artefato["modelo"]

mapa = pd.read_csv(os.path.join(DADOS, "mapeamento_municipio.csv"))
df26 = pd.read_csv(os.path.join(DADOS, "bdqueimadas_2026-01-01_2026-06-03.csv"),
                   parse_dates=["DataHora"])
df26["Data"] = df26["DataHora"].dt.date
data_fim = pd.to_datetime(df26["Data"].max())

positivos_2026 = set(zip(df26["Municipio"].str.upper(), df26["Data"]))

todos_dias = pd.date_range("2026-01-01", data_fim, freq="D")
grid = pd.DataFrame(list(product(mapa["Municipio"].values, todos_dias.date)),
                    columns=["Municipio", "Data"])
grid["fogo"] = grid.apply(
    lambda r: 1 if (r["Municipio"], r["Data"]) in positivos_2026 else 0, axis=1)
grid["Data"]         = pd.to_datetime(grid["Data"])
grid["Mes"]          = grid["Data"].dt.month
grid["DiaSemana"]    = grid["Data"].dt.dayofweek
grid["Estacao_Seca"] = grid["Mes"].between(6, 10).astype(int)

grid = grid.merge(mapa[["Municipio","Municipio_Freq","Latitude","Longitude"]],
                  on="Municipio", how="left")

hist = (pd.read_csv(os.path.join(DADOS, "dataset_municipio.csv"),
                    usecols=["Municipio","Mes","media_focos_mes_hist"])
        .drop_duplicates())
grid = grid.merge(hist, on=["Municipio","Mes"], how="left")
grid["media_focos_mes_hist"] = grid["media_focos_mes_hist"].fillna(0)

clima26 = pd.read_csv(os.path.join(DADOS, "clima_2026.csv"), parse_dates=["Data"])
grid = grid.merge(clima26[["Municipio","Data","Precipitacao","DiaSemChuva"]],
                  on=["Municipio","Data"], how="left")
grid["DiaSemChuva"]  = grid["DiaSemChuva"].fillna(0)
grid["Precipitacao"] = grid["Precipitacao"].fillna(0)

print(f"  Grid: {len(grid):,} linhas | Positivos: {grid['fogo'].sum():,} ({100*grid['fogo'].mean():.2f}%)")

grid["prob_fogo"] = modelo.predict_proba(grid[FEATURES].values)[:, 1]
y_true = grid["fogo"].values
y_prob = grid["prob_fogo"].values
auc = roc_auc_score(y_true, y_prob)
print(f"  AUC-ROC: {auc:.4f}")

# ── 2. Gráfico 1: Mapa previsão × focos reais ────────────────────────
print("\n[2/5] Gráfico 1: Mapa previsão × focos reais...")

prob_mun = (grid.groupby(["Municipio","Latitude","Longitude"])["prob_fogo"]
            .mean().reset_index(name="prob_media"))
LIMIAR = 0.3  # mesmo limiar usado na avaliação principal (recall 73.8%)

focos_set = set(df26["Municipio"].str.upper().unique())
prob_mun["teve_fogo"] = prob_mun["Municipio"].isin(focos_set)

# Município "alertado" = teve ao menos 1 dia com prob >= LIMIAR
alertados = (grid[grid["prob_fogo"] >= LIMIAR]["Municipio"].unique())
prob_mun["alto_risco"] = prob_mun["Municipio"].isin(alertados)

sem_fogo_baixo = prob_mun[~prob_mun["alto_risco"] & ~prob_mun["teve_fogo"]]
sem_fogo_alto  = prob_mun[ prob_mun["alto_risco"] & ~prob_mun["teve_fogo"]]
com_fogo_baixo = prob_mun[~prob_mun["alto_risco"] &  prob_mun["teve_fogo"]]
com_fogo_alto  = prob_mun[ prob_mun["alto_risco"] &  prob_mun["teve_fogo"]]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

cmap = mcolors.LinearSegmentedColormap.from_list("risco", ["#eaf4fb","#f39c12","#e74c3c"])
sc = axes[0].scatter(prob_mun["Longitude"], prob_mun["Latitude"],
                     c=prob_mun["prob_media"], cmap=cmap, vmin=0, vmax=0.25,
                     s=80, alpha=0.85, linewidths=0, zorder=1)
focos_reais_mun = prob_mun[prob_mun["teve_fogo"]]
axes[0].scatter(focos_reais_mun["Longitude"], focos_reais_mun["Latitude"],
                c="none", edgecolors="#1a1a2e", s=120, linewidths=1.5,
                zorder=2, label=f"Município com fogo real ({len(focos_reais_mun):,})")
plt.colorbar(sc, ax=axes[0], label="Probabilidade média prevista", shrink=0.8)
axes[0].set_title("Probabilidade prevista + municípios com fogo real\n(círculos = onde realmente houve fogo)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].legend(fontsize=9, loc="lower right")

axes[1].scatter(sem_fogo_baixo["Longitude"], sem_fogo_baixo["Latitude"],
                c="#dee2e6", s=60, alpha=0.6, linewidths=0,
                label=f"Baixo risco, sem fogo ({len(sem_fogo_baixo):,})")
axes[1].scatter(sem_fogo_alto["Longitude"], sem_fogo_alto["Latitude"],
                c="#f39c12", s=80, alpha=0.85, linewidths=0,
                label=f"Alto risco, sem fogo ({len(sem_fogo_alto):,}) — falso positivo")
axes[1].scatter(com_fogo_baixo["Longitude"], com_fogo_baixo["Latitude"],
                c="#3498db", s=90, alpha=0.85, linewidths=0,
                label=f"Baixo risco, teve fogo ({len(com_fogo_baixo):,}) — falso negativo")
axes[1].scatter(com_fogo_alto["Longitude"], com_fogo_alto["Latitude"],
                c="#e74c3c", s=110, alpha=0.95, linewidths=0,
                label=f"Alto risco + fogo real ({len(com_fogo_alto):,}) — acerto")
recall_mapa   = len(com_fogo_alto) / max(len(com_fogo_alto)+len(com_fogo_baixo), 1) * 100
precisao_mapa = len(com_fogo_alto) / max(len(com_fogo_alto)+len(sem_fogo_alto), 1) * 100
axes[1].set_title(
    f"Acertos e erros (prob ≥ {LIMIAR} em ao menos 1 dia)\n"
    f"Recall: {recall_mapa:.0f}%  |  Precisão: {precisao_mapa:.0f}%\n"
    f"Acertos: {len(com_fogo_alto)}  |  F.Neg.: {len(com_fogo_baixo)}  |  F.Pos.: {len(sem_fogo_alto)}",
    fontsize=10, fontweight="bold")
axes[1].set_xlabel("Longitude")
axes[1].legend(fontsize=8, loc="lower right")

fig.suptitle("vigIA E1 — Risco Previsto × Focos Reais por Município | Goiás Jan–Jun 2026",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e1_comparacao_mapa.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Salvo: e1_comparacao_mapa.png")

# ── 3. Gráfico 2: Por mês ─────────────────────────────────────────────
print("\n[3/5] Gráfico 2: Probabilidade prevista vs focos reais por mês...")

por_mes = grid.groupby("Mes").agg(
    prob_media   = ("prob_fogo", "mean"),
    focos_reais  = ("fogo",      "sum"),
    total        = ("fogo",      "count"),
).reset_index()
meses_nome = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
              7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}

fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(por_mes))
ax1.bar(x, por_mes["focos_reais"], color="#e74c3c", alpha=0.7, label="Focos reais (município-dia)")
ax1.set_ylabel("Pares (município, dia) com fogo real", color="#e74c3c")
ax1.tick_params(axis="y", colors="#e74c3c")
ax1.set_xticks(x)
ax1.set_xticklabels([meses_nome[m] for m in por_mes["Mes"]])

ax2 = ax1.twinx()
ax2.plot(x, por_mes["prob_media"] * 100, "o-", color="#3498db",
         linewidth=2.5, markersize=7, label="Prob. média prevista (%)")
ax2.set_ylabel("Probabilidade média prevista (%)", color="#3498db")
ax2.tick_params(axis="y", colors="#3498db")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=9)
ax1.set_title("vigIA E1 — Focos Reais vs Probabilidade Prevista por Mês | Goiás 2026",
              fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e1_comparacao_por_mes.png"), dpi=150)
plt.close()
print("  Salvo: e1_comparacao_por_mes.png")

# ── 4. Gráfico 3: Curva de captura ────────────────────────────────────
print("\n[4/5] Gráfico 3: Curva de captura...")

df_sorted = grid.sort_values("prob_fogo", ascending=False).reset_index(drop=True)
total_focos = df_sorted["fogo"].sum()
df_sorted["focos_acum"]  = df_sorted["fogo"].cumsum()
df_sorted["pct_pares"]   = (np.arange(1, len(df_sorted)+1)) / len(df_sorted) * 100
df_sorted["pct_focos"]   = df_sorted["focos_acum"] / total_focos * 100

sample = df_sorted.iloc[::20]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(sample["pct_pares"], sample["pct_focos"],
        color="#e74c3c", linewidth=2.5, label=f"Modelo município (AUC={auc:.3f})")
ax.plot([0,100],[0,100], "k--", linewidth=1, label="Aleatório")
ax.fill_between(sample["pct_pares"], sample["pct_pares"],
                sample["pct_focos"], alpha=0.08, color="#e74c3c")

for pct in [5, 10, 20]:
    idx = (sample["pct_pares"] - pct).abs().idxmin()
    cap = sample.loc[idx, "pct_focos"]
    ax.annotate(f"Top {pct}%\ncaptura {cap:.0f}%\ndos fogos",
                xy=(pct, cap), xytext=(pct+3, cap-10),
                fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    ax.scatter([pct], [cap], color="#e74c3c", zorder=5, s=60)

ax.set_xlabel("% de pares (município, dia) monitorados (ordenados por risco previsto)")
ax.set_ylabel("% de focos reais capturados")
ax.set_title("vigIA E1 — Curva de Captura | Goiás Jan–Jun 2026\n"
             "Quanto dos fogos reais o modelo encontra priorizando os municípios mais arriscados?",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e1_comparacao_captura.png"), dpi=150)
plt.close()
print("  Salvo: e1_comparacao_captura.png")

# ── 5. Gráfico 4: Distribuição de probabilidades ──────────────────────
print("\n[5/5] Gráfico 4: Distribuição de probabilidades...")

pos_probs = y_prob[y_true == 1]
neg_probs = y_prob[y_true == 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(neg_probs, bins=50, color="#3498db", alpha=0.6,
             label=f"Sem fogo ({len(neg_probs):,})", density=True)
axes[0].hist(pos_probs, bins=50, color="#e74c3c", alpha=0.7,
             label=f"Com fogo ({len(pos_probs):,})", density=True)
axes[0].axvline(0.3, color="#f39c12", linestyle="--", linewidth=1.5, label="Limiar 0.3")
axes[0].axvline(0.5, color="#c0392b", linestyle="--", linewidth=1.5, label="Limiar 0.5")
axes[0].set_xlabel("Probabilidade prevista")
axes[0].set_ylabel("Densidade")
axes[0].set_title("Distribuição de probabilidades\npor classe real", fontweight="bold")
axes[0].legend(fontsize=9)

limiares = np.linspace(0.01, 0.99, 200)
recalls = []; precisoes = []
for t in limiares:
    pred = (y_prob >= t).astype(int)
    tp = ((pred==1) & (y_true==1)).sum()
    fp = ((pred==1) & (y_true==0)).sum()
    fn = ((pred==0) & (y_true==1)).sum()
    recalls.append(tp / (tp+fn) if (tp+fn) > 0 else 0)
    precisoes.append(tp / (tp+fp) if (tp+fp) > 0 else 0)

axes[1].plot(limiares, recalls,   color="#e74c3c", linewidth=2, label="Recall")
axes[1].plot(limiares, precisoes, color="#3498db", linewidth=2, label="Precisão")
axes[1].axvline(0.3, color="#f39c12", linestyle="--", linewidth=1.5, label="Limiar 0.3")
axes[1].axvline(0.5, color="#c0392b", linestyle="--", linewidth=1.5, label="Limiar 0.5")
axes[1].set_xlabel("Limiar de decisão")
axes[1].set_ylabel("Valor da métrica")
axes[1].set_title("Recall vs Precisão por limiar\n(Jan–Jun 2026)", fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

fig.suptitle("vigIA E1 — Análise de Probabilidades por Município | Goiás Jan–Jun 2026",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e1_comparacao_probabilidades.png"), dpi=150)
plt.close()
print("  Salvo: e1_comparacao_probabilidades.png")

# ── Resumo ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  AUC-ROC: {auc:.4f}")
top5  = df_sorted[df_sorted["pct_pares"] <= 5]["fogo"].sum()  / total_focos * 100
top10 = df_sorted[df_sorted["pct_pares"] <= 10]["fogo"].sum() / total_focos * 100
top20 = df_sorted[df_sorted["pct_pares"] <= 20]["fogo"].sum() / total_focos * 100
print(f"  Curva de captura (pares município × dia):")
print(f"    Top  5% → captura {top5:.1f}% dos fogos reais")
print(f"    Top 10% → captura {top10:.1f}% dos fogos reais")
print(f"    Top 20% → captura {top20:.1f}% dos fogos reais")
recall_mapa   = len(com_fogo_alto) / max(len(com_fogo_alto)+len(com_fogo_baixo), 1) * 100
precisao_mapa = len(com_fogo_alto) / max(len(com_fogo_alto)+len(sem_fogo_alto), 1) * 100
print(f"\n  Mapa (limiar prob ≥ {LIMIAR} em ao menos 1 dia):")
print(f"    Municípios alertados : {len(alertados)}")
print(f"    Recall               : {recall_mapa:.1f}%  ({len(com_fogo_alto)} de {len(com_fogo_alto)+len(com_fogo_baixo)} com fogo)")
print(f"    Precisão             : {precisao_mapa:.1f}%  ({len(com_fogo_alto)} de {len(com_fogo_alto)+len(sem_fogo_alto)} alertados)")
print(f"\n  Gráficos em graficos/: e1_comparacao_*.png")
print(f"{'='*65}")
print("\n[OK] E1 Fase 3b concluída!")
