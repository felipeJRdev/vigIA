"""
vigIA — Estágio 2 | Fase 1c: Aplica clima exato (0.5°) no dataset_grade.csv
Substitui DiaSemChuva e Precipitacao do proxy de município pelo clima 0.5° correto.
Entrada:  ../dados/dataset_grade.csv
          ../dados/clima_grade.csv
Saída:    ../dados/dataset_grade.csv  (atualizado no lugar)
"""

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PBL   = os.path.dirname(_HERE)
DADOS = os.path.join(PBL, "dados")

print("=" * 65)
print("  vigIA E2 — Fase 1c: Aplicando Clima Exato por Célula")
print("=" * 65)

print("\n[1/3] Carregando datasets...")
ds    = pd.read_csv(os.path.join(DADOS, "dataset_grade.csv"), parse_dates=["Data"])
clima = pd.read_csv(os.path.join(DADOS, "clima_grade.csv"),   parse_dates=["Data"])
print(f"  dataset_grade: {len(ds):,} linhas")
print(f"  clima_grade:   {len(clima):,} linhas | células: {clima[['Cell_Lat','Cell_Lon']].drop_duplicates().shape[0]:,}")

print("\n[2/3] Substituindo DiaSemChuva e Precipitacao...")
ds = ds.drop(columns=["DiaSemChuva","Precipitacao"], errors="ignore")
ds = ds.merge(clima[["Cell_Lat","Cell_Lon","Data","Precipitacao","DiaSemChuva"]],
              on=["Cell_Lat","Cell_Lon","Data"], how="left")

n_nan = ds["DiaSemChuva"].isna().sum()
if n_nan > 0:
    print(f"  Aviso: {n_nan:,} linhas sem dado — usando 0 como fallback.")
ds["DiaSemChuva"]  = ds["DiaSemChuva"].fillna(0)
ds["Precipitacao"] = ds["Precipitacao"].fillna(0)
print(f"  DiaSemChuva máx: {ds['DiaSemChuva'].max():.0f}d | Precipitacao máx: {ds['Precipitacao'].max():.1f}mm")

print("\n[3/3] Salvando dataset_grade.csv atualizado...")
cols = ["Cell_Lat","Cell_Lon","Nearest_Municipio","Data","Ano","Mes","DiaSemana",
        "Estacao_Seca","Cell_Freq","DiaSemChuva","Precipitacao","media_focos_mes_hist","fogo"]
ds[cols].to_csv(os.path.join(DADOS, "dataset_grade.csv"), index=False)

print(f"\n{'='*65}")
print(f"  dataset_grade.csv atualizado com clima exato 0.5°.")
print(f"  Total: {len(ds):,} | Positivos: {ds['fogo'].sum():,} ({100*ds['fogo'].mean():.2f}%)")
print(f"{'='*65}")
print("\n[OK] E2 Fase 1c concluída!")
print("  Próximo: python3 estagio2_grade/fase2_modelagem.py")
