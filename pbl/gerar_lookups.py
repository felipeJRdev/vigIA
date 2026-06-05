"""
gerar_lookups.py — Extrai tabelas de climatologia dos datasets pesados.
Roda UMA VEZ. Gera arquivos pequenos usados por previsao_leve.py.

Entrada:
  dados/dataset_municipio.csv  (992 k linhas)
  dados/dataset_grade.csv      (18.4 M linhas)

Saída:
  dados/lookup_municipio.csv   (≤ 244 × 12 = 2.928 linhas)
  dados/lookup_grade.csv       (≤ 2.976 × 12 = 35.712 linhas)
"""

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.join(_HERE, "dados")

# ── Lookup E1: (Municipio, Mes) → media_focos_mes_hist ───────────────────────
print("[1/2] Gerando lookup_municipio.csv ...")
mun = pd.read_csv(
    os.path.join(DADOS, "dataset_municipio.csv"),
    usecols=["Municipio", "Mes", "media_focos_mes_hist"],
)
lookup_mun = mun.drop_duplicates(subset=["Municipio", "Mes"])
dest_mun = os.path.join(DADOS, "lookup_municipio.csv")
lookup_mun.to_csv(dest_mun, index=False)
print(f"  {len(lookup_mun):,} linhas  →  {dest_mun}")

# ── Lookup E2: (Cell_Lat, Cell_Lon, Mes) → media_focos_mes_hist ──────────────
print("[2/2] Gerando lookup_grade.csv  (carrega 18 M linhas, aguarde) ...")
grade = pd.read_csv(
    os.path.join(DADOS, "dataset_grade.csv"),
    usecols=["Cell_Lat", "Cell_Lon", "Mes", "fogo", "media_focos_mes_hist"],
)
lookup_grade = (
    grade.query("fogo == 1")
    .groupby(["Cell_Lat", "Cell_Lon", "Mes"])["media_focos_mes_hist"]
    .mean()
    .reset_index()
)
dest_grade = os.path.join(DADOS, "lookup_grade.csv")
lookup_grade.to_csv(dest_grade, index=False)
print(f"  {len(lookup_grade):,} linhas  →  {dest_grade}")

print("\n[OK] Lookups prontos. Execute previsao_leve.py para previsão sem datasets pesados.")
