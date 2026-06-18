"""
vigIA — Estágio 1 | Fase 3c: Baselines de climatologia vs modelo
Entrada:  ../resultados/dataset_validacao_2026.csv  (gerado por fase3 — deve conter prob_fogo)
          ../resultados/validacao_municipio_2026.csv
Saída:    ../resultados/baseline_municipio_2026.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = os.path.dirname(_HERE)
RESULTADOS = os.path.join(PBL, "resultados")

print("=" * 65)
print("  vigIA E1 — Fase 3c: Baselines de Climatologia")
print("=" * 65)

df = pd.read_csv(os.path.join(RESULTADOS, "dataset_validacao_2026.csv"), parse_dates=["Data"])
metricas_modelo = pd.read_csv(os.path.join(RESULTADOS, "validacao_municipio_2026.csv"))

if "prob_fogo" not in df.columns:
    raise ValueError("prob_fogo ausente — execute fase3_validacao_2026.py antes desta fase.")

y = df["fogo"].values
n_pos = y.sum()
print(f"\n  {len(df):,} linhas | {n_pos:,} positivos ({100*n_pos/len(df):.1f}%)")

# --- Scores dos baselines ---
mfmh = df["media_focos_mes_hist"].fillna(0)
freq = df["Municipio_Freq"].fillna(0)
dsc  = df["DiaSemChuva"].fillna(0)
# baseline 3: media histórica + desempate por dias secos (peso pequeno para não dominar)
dsc_norm = dsc / (dsc.max() + 1e-9)
score3 = mfmh + 0.01 * dsc_norm * (mfmh.max() + 1e-9)

scores = {
    "media_focos_mes_hist":             mfmh.values,
    "Municipio_Freq":                   freq.values,
    "media_focos_mes_hist+DiaSemChuva": score3.values,
    "Modelo LightGBM":                  df["prob_fogo"].values,
}

# --- Captura top-N por dia ---
def captura_top_n(df, score_vals, n):
    tmp = df.assign(_s=score_vals)
    def _por_dia(g):
        fires = g["fogo"].sum()
        return g.nlargest(n, "_s")["fogo"].sum() / fires if fires > 0 else np.nan
    return tmp.groupby("Data").apply(_por_dia, include_groups=False).mean(skipna=True)

print(f"\n  {'Ranqueador':<40} {'AUC':>6}  {'Cap10%':>7}  {'Cap20%':>7}")
print("  " + "-" * 65)

resultados = []
for nome, sv in scores.items():
    auc  = roc_auc_score(y, sv)
    c10  = captura_top_n(df, sv, 10)
    c20  = captura_top_n(df, sv, 20)
    resultados.append({"Ranqueador": nome, "AUC": round(auc, 4),
                       "Cap_Top10": round(c10, 4), "Cap_Top20": round(c20, 4)})
    print(f"  {nome:<40} {auc:>6.4f}  {c10:>7.1%}  {c20:>7.1%}")

df_res = pd.DataFrame(resultados)
df_res.to_csv(os.path.join(RESULTADOS, "baseline_municipio_2026.csv"), index=False)

# Delta modelo vs melhor baseline
melhor_base = df_res[df_res["Ranqueador"] != "Modelo LightGBM"]["AUC"].max()
auc_modelo  = df_res[df_res["Ranqueador"] == "Modelo LightGBM"]["AUC"].iloc[0]
delta = auc_modelo - melhor_base

print(f"\n  Delta modelo − melhor baseline: {delta:+.4f}", end="")
if delta < 0.02:
    print("  ⚠ Delta pequeno — registrar como limitação na documentação.")
else:
    print()

print(f"\n{'='*65}")
print(f"  Salvo: resultados/baseline_municipio_2026.csv")
print(f"{'='*65}")
print("\n[OK] E1 Fase 3c concluída!")
