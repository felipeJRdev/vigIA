"""
vigIA — Estágio 2 | Fase 3b: Gráficos de comparação previsão × realidade (2026)
Gera visualizações para avaliar onde o modelo acertou e errou.

Entrada:  ../modelos/grade_full.pkl
          ../dados/mapeamento_grade.csv
          ../dados/dataset_grade.csv
          ../dados/bdqueimadas_2026-01-01_2026-06-03.csv
          ../dados/clima_2026.csv
Saída:    ../graficos/e2_comparacao_*.png  (4 gráficos)
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
import matplotlib.gridspec as gridspec
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
    "Cell_Lat", "Cell_Lon", "Cell_Freq",
    "DiaSemChuva", "Precipitacao", "media_focos_mes_hist",
]

print("=" * 65)
print("  vigIA E2 — Fase 3b: Gráficos Previsão × Realidade 2026")
print("=" * 65)

# ── 1. Carregar modelo e reconstruir grid 2026 ────────────────────────
print("\n[1/5] Carregando modelo e reconstruindo grid 2026...")
artefato = joblib.load(os.path.join(MODELOS, "grade_full.pkl"))
modelo   = artefato["modelo"]

grade = pd.read_csv(os.path.join(DADOS, "mapeamento_grade.csv"))
df26  = pd.read_csv(os.path.join(DADOS, "bdqueimadas_2026-01-01_2026-06-03.csv"),
                    parse_dates=["DataHora"])
df26["Data"]     = df26["DataHora"].dt.date
df26["Cell_Lat"] = df26["Latitude"].round(1)
df26["Cell_Lon"] = df26["Longitude"].round(1)
data_fim = pd.to_datetime(df26["Data"].max())

celulas_validas = set(zip(grade["Cell_Lat"], grade["Cell_Lon"]))
df26_valid = df26[df26.apply(
    lambda r: (r["Cell_Lat"], r["Cell_Lon"]) in celulas_validas, axis=1)]
positivos_2026 = set(zip(df26_valid["Cell_Lat"], df26_valid["Cell_Lon"],
                         pd.to_datetime(df26_valid["Data"])))

todos_dias = pd.date_range("2026-01-01", data_fim, freq="D")
grid = pd.DataFrame(
    list(product(
        zip(grade["Cell_Lat"], grade["Cell_Lon"],
            grade["Cell_Freq"], grade["Nearest_Municipio"]),
        todos_dias)),
    columns=["cell_tuple", "Data"])
grid[["Cell_Lat","Cell_Lon","Cell_Freq","Nearest_Municipio"]] = pd.DataFrame(
    grid["cell_tuple"].tolist(), index=grid.index)
grid = grid.drop(columns=["cell_tuple"])
grid["Mes"]          = grid["Data"].dt.month
grid["DiaSemana"]    = grid["Data"].dt.dayofweek
grid["Estacao_Seca"] = grid["Mes"].between(6,10).astype(int)
grid["Ano"]          = grid["Data"].dt.year
grid["fogo"] = grid.apply(
    lambda r: 1 if (r["Cell_Lat"], r["Cell_Lon"], r["Data"]) in positivos_2026 else 0,
    axis=1)

ds = pd.read_csv(os.path.join(DADOS, "dataset_grade.csv"))
hist = (ds[ds["fogo"]==1].groupby(["Cell_Lat","Cell_Lon","Mes"]).size()
        .div(ds["Ano"].nunique()).reset_index(name="media_focos_mes_hist"))
grid = grid.merge(hist, on=["Cell_Lat","Cell_Lon","Mes"], how="left")
grid["media_focos_mes_hist"] = grid["media_focos_mes_hist"].fillna(0)

clima26 = pd.read_csv(os.path.join(DADOS, "clima_2026.csv"), parse_dates=["Data"])
clima26 = clima26.rename(columns={"Municipio": "Nearest_Municipio"})
grid = grid.merge(clima26[["Nearest_Municipio","Data","Precipitacao","DiaSemChuva"]],
                  on=["Nearest_Municipio","Data"], how="left")
grid["DiaSemChuva"]  = grid["DiaSemChuva"].fillna(0)
grid["Precipitacao"] = grid["Precipitacao"].fillna(0)

print(f"  Grid: {len(grid):,} linhas | Positivos: {grid['fogo'].sum():,} ({100*grid['fogo'].mean():.2f}%)")

# Prever
grid["prob_fogo"] = modelo.predict_proba(grid[FEATURES].values)[:, 1]
y_true = grid["fogo"].values
y_prob = grid["prob_fogo"].values
auc = roc_auc_score(y_true, y_prob)
print(f"  AUC-ROC: {auc:.4f}")

# ── 2. Gráfico 1: Mapa sobreposição previsão × focos reais ───────────
print("\n[2/5] Gráfico 1: Mapa previsão × focos reais...")

prob_celula = (grid.groupby(["Cell_Lat","Cell_Lon"])["prob_fogo"]
               .mean().reset_index(name="prob_media"))

# Classificar cada célula: acerto, erro, não prevista
LIMIAR = 0.6  # célula alertada se tiver ao menos 1 dia com prob >= 0.6

focos_set = set(zip(df26_valid["Cell_Lat"].round(1), df26_valid["Cell_Lon"].round(1)))
prob_celula["teve_fogo"] = prob_celula.apply(
    lambda r: (r["Cell_Lat"], r["Cell_Lon"]) in focos_set, axis=1)

# Célula alertada = ao menos 1 dia com prob >= LIMIAR
alertadas = set(zip(
    grid[grid["prob_fogo"] >= LIMIAR]["Cell_Lat"],
    grid[grid["prob_fogo"] >= LIMIAR]["Cell_Lon"]
))
prob_celula["alto_risco"] = prob_celula.apply(
    lambda r: (r["Cell_Lat"], r["Cell_Lon"]) in alertadas, axis=1)

# Categorias para o mapa único
sem_fogo_baixo = prob_celula[~prob_celula["alto_risco"] & ~prob_celula["teve_fogo"]]
sem_fogo_alto  = prob_celula[ prob_celula["alto_risco"] & ~prob_celula["teve_fogo"]]
com_fogo_baixo = prob_celula[~prob_celula["alto_risco"] &  prob_celula["teve_fogo"]]
com_fogo_alto  = prob_celula[ prob_celula["alto_risco"] &  prob_celula["teve_fogo"]]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# ── Esquerda: mapa de probabilidade contínua com focos reais sobrepostos
cmap = mcolors.LinearSegmentedColormap.from_list("risco", ["#eaf4fb","#f39c12","#e74c3c"])
sc = axes[0].scatter(prob_celula["Cell_Lon"], prob_celula["Cell_Lat"],
                     c=prob_celula["prob_media"], cmap=cmap, vmin=0, vmax=0.08,
                     s=18, alpha=0.8, linewidths=0, zorder=1)
# Focos reais por cima
focos_reais = df26_valid[["Cell_Lat","Cell_Lon"]].drop_duplicates()
axes[0].scatter(focos_reais["Cell_Lon"], focos_reais["Cell_Lat"],
                c="none", edgecolors="#1a1a2e", s=45, linewidths=1.2,
                zorder=2, label=f"Foco real ({len(focos_reais):,} células)")
plt.colorbar(sc, ax=axes[0], label="Probabilidade média prevista", shrink=0.8)
axes[0].set_title("Probabilidade prevista + focos reais\n(círculos = onde realmente houve fogo)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].legend(fontsize=9, loc="lower right")

# ── Direita: mapa de acertos e erros (top 20% como "alerta")
axes[1].scatter(sem_fogo_baixo["Cell_Lon"], sem_fogo_baixo["Cell_Lat"],
                c="#dee2e6", s=12, alpha=0.5, linewidths=0,
                label=f"Baixo risco, sem fogo ({len(sem_fogo_baixo):,})")
axes[1].scatter(sem_fogo_alto["Cell_Lon"], sem_fogo_alto["Cell_Lat"],
                c="#f39c12", s=18, alpha=0.75, linewidths=0,
                label=f"Alto risco previsto, sem fogo ({len(sem_fogo_alto):,}) — falso positivo")
axes[1].scatter(com_fogo_baixo["Cell_Lon"], com_fogo_baixo["Cell_Lat"],
                c="#3498db", s=25, alpha=0.85, linewidths=0,
                label=f"Baixo risco previsto, teve fogo ({len(com_fogo_baixo):,}) — falso negativo")
axes[1].scatter(com_fogo_alto["Cell_Lon"], com_fogo_alto["Cell_Lat"],
                c="#e74c3c", s=35, alpha=0.95, linewidths=0,
                label=f"Alto risco previsto + fogo real ({len(com_fogo_alto):,}) — acerto")
recall_mapa   = len(com_fogo_alto) / max(len(com_fogo_alto)+len(com_fogo_baixo), 1) * 100
precisao_mapa = len(com_fogo_alto) / max(len(com_fogo_alto)+len(sem_fogo_alto), 1) * 100
axes[1].set_title(
    f"Acertos e erros (prob ≥ {LIMIAR} em ao menos 1 dia)\n"
    f"Recall: {recall_mapa:.0f}%  |  Precisão: {precisao_mapa:.0f}%\n"
    f"Acertos: {len(com_fogo_alto):,}  |  F.Neg.: {len(com_fogo_baixo):,}  |  F.Pos.: {len(sem_fogo_alto):,}",
                  fontsize=10, fontweight="bold")
axes[1].set_xlabel("Longitude")
axes[1].legend(fontsize=8, loc="lower right")

fig.suptitle("vigIA E2 — Risco Previsto × Focos Reais | Goiás Jan–Jun 2026",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e2_comparacao_mapa.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Salvo: e2_comparacao_mapa.png")

# ── 3. Gráfico 2: Por mês — prob prevista vs focos reais ──────────────
print("\n[3/5] Gráfico 2: Probabilidade prevista vs focos reais por mês...")

por_mes = grid.groupby("Mes").agg(
    prob_media   = ("prob_fogo", "mean"),
    focos_reais  = ("fogo",      "sum"),
    total_celulas= ("fogo",      "count"),
).reset_index()
por_mes["taxa_real"] = por_mes["focos_reais"] / por_mes["total_celulas"]
meses_nome = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
              7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}

fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(por_mes))
bars = ax1.bar(x, por_mes["focos_reais"], color="#e74c3c", alpha=0.7,
               label="Focos reais (BDqueimadas)")
ax1.set_ylabel("Focos reais (pares célula×dia)", color="#e74c3c")
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
ax1.set_title("vigIA E2 — Focos Reais vs Probabilidade Prevista por Mês | Goiás 2026",
              fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e2_comparacao_por_mes.png"), dpi=150)
plt.close()
print("  Salvo: e2_comparacao_por_mes.png")

# ── 4. Gráfico 3: Curva de captura ────────────────────────────────────
print("\n[4/5] Gráfico 3: Curva de captura (top-N cells)...")

# Ordenar células por prob e calcular % de fogos capturados
df_sorted = grid.sort_values("prob_fogo", ascending=False).reset_index(drop=True)
total_focos = df_sorted["fogo"].sum()
df_sorted["focos_acum"] = df_sorted["fogo"].cumsum()
df_sorted["pct_celulas"] = (np.arange(1, len(df_sorted)+1)) / len(df_sorted) * 100
df_sorted["pct_focos"]   = df_sorted["focos_acum"] / total_focos * 100

# Amostrar para não plotar 458k pontos
sample = df_sorted.iloc[::100]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(sample["pct_celulas"], sample["pct_focos"],
        color="#e74c3c", linewidth=2.5, label=f"Modelo grade (AUC={auc:.3f})")
ax.plot([0,100],[0,100], "k--", linewidth=1, label="Aleatório")
ax.fill_between(sample["pct_celulas"], sample["pct_celulas"],
                sample["pct_focos"], alpha=0.08, color="#e74c3c")

# Marcar pontos de referência
for pct in [5, 10, 20]:
    idx = (sample["pct_celulas"] - pct).abs().idxmin()
    cap = sample.loc[idx, "pct_focos"]
    ax.annotate(f"Top {pct}%\ncaptura {cap:.0f}%\ndos fogos",
                xy=(pct, cap), xytext=(pct+3, cap-8),
                fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    ax.scatter([pct], [cap], color="#e74c3c", zorder=5, s=60)

ax.set_xlabel("% de células monitoradas (ordenadas por risco previsto)")
ax.set_ylabel("% de focos reais capturados")
ax.set_title("vigIA E2 — Curva de Captura | Goiás Jan–Jun 2026\n"
             "Quanto dos fogos reais o modelo encontra monitorando o top-N de células?",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e2_comparacao_captura.png"), dpi=150)
plt.close()
print("  Salvo: e2_comparacao_captura.png")

# ── 5. Gráfico 4: Distribuição de probabilidades ──────────────────────
print("\n[5/5] Gráfico 4: Distribuição de probabilidades por classe...")

pos_probs = y_prob[y_true == 1]
neg_probs = y_prob[y_true == 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma
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

# Recall e Precisão por limiar
limiares = np.linspace(0.01, 0.99, 200)
recalls = []; precisoes = []
for t in limiares:
    pred = (y_prob >= t).astype(int)
    tp = ((pred==1) & (y_true==1)).sum()
    fp = ((pred==1) & (y_true==0)).sum()
    fn = ((pred==0) & (y_true==1)).sum()
    recalls.append(tp / (tp+fn) if (tp+fn) > 0 else 0)
    precisoes.append(tp / (tp+fp) if (tp+fp) > 0 else 0)

axes[1].plot(limiares, recalls,    color="#e74c3c", linewidth=2, label="Recall")
axes[1].plot(limiares, precisoes,  color="#3498db", linewidth=2, label="Precisão")
axes[1].axvline(0.3, color="#f39c12", linestyle="--", linewidth=1.5, label="Limiar 0.3")
axes[1].axvline(0.5, color="#c0392b", linestyle="--", linewidth=1.5, label="Limiar 0.5")
axes[1].set_xlabel("Limiar de decisão")
axes[1].set_ylabel("Valor da métrica")
axes[1].set_title("Recall vs Precisão por limiar\n(Jan–Jun 2026)", fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

fig.suptitle("vigIA E2 — Análise de Probabilidades | Goiás Jan–Jun 2026",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(GRAFICOS, "e2_comparacao_probabilidades.png"), dpi=150)
plt.close()
print("  Salvo: e2_comparacao_probabilidades.png")

# ── Resumo ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  AUC-ROC: {auc:.4f}")
top5  = df_sorted[df_sorted["pct_celulas"] <= 5]["fogo"].sum()  / total_focos * 100
top10 = df_sorted[df_sorted["pct_celulas"] <= 10]["fogo"].sum() / total_focos * 100
top20 = df_sorted[df_sorted["pct_celulas"] <= 20]["fogo"].sum() / total_focos * 100
recall_mapa   = len(com_fogo_alto) / max(len(com_fogo_alto)+len(com_fogo_baixo), 1) * 100
precisao_mapa = len(com_fogo_alto) / max(len(com_fogo_alto)+len(sem_fogo_alto), 1) * 100
print(f"  Curva de captura:")
print(f"    Top  5% de células → captura {top5:.1f}% dos fogos reais")
print(f"    Top 10% de células → captura {top10:.1f}% dos fogos reais")
print(f"    Top 20% de células → captura {top20:.1f}% dos fogos reais")
print(f"\n  Mapa (limiar prob ≥ {LIMIAR} em ao menos 1 dia):")
print(f"    Células alertadas : {len(alertadas):,}")
print(f"    Recall            : {recall_mapa:.1f}%  ({len(com_fogo_alto):,} de {len(com_fogo_alto)+len(com_fogo_baixo):,} com fogo)")
print(f"    Precisão          : {precisao_mapa:.1f}%  ({len(com_fogo_alto):,} de {len(com_fogo_alto)+len(sem_fogo_alto):,} alertadas)")
print(f"\n  Gráficos gerados em graficos/:")
print(f"    e2_comparacao_mapa.png")
print(f"    e2_comparacao_por_mes.png")
print(f"    e2_comparacao_captura.png")
print(f"    e2_comparacao_probabilidades.png")
print(f"{'='*65}")
print("\n[OK] Fase 3b concluída!")
