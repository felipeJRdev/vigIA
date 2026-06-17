"""
vigIA — Estágio 2 | Fase 1: Dataset por célula 0.1° × 0.1° (negativos naturais completos)
Entrada:  ../../dataset_queimadas_completo.csv
          ../dados/clima_historico.csv
          ../dados/mapeamento_municipio.csv
Saída:    ../dados/dataset_grade.csv     (~11.9M linhas)
          ../dados/mapeamento_grade.csv  (2.976 células)
"""

import os, time
import numpy as np
import pandas as pd

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = os.path.dirname(_HERE)
BASE       = os.path.dirname(PBL)
DADOS      = os.path.join(PBL, "dados")
RAW        = os.path.join(BASE, "dataset_queimadas_completo.csv")

MIN_OCORRENCIAS = 5

print("=" * 65)
print("  vigIA E2 — Fase 1: Dataset por Célula 0.1° (negativos naturais)")
print("=" * 65)

print("\n[1/6] Carregando dados brutos...")
df = pd.read_csv(RAW, parse_dates=["DataHora"])
go = df[(df["Estado"].str.upper()=="GOIÁS") & (df["Bioma"]!="Mata Atlântica")].copy()
print(f"  Após filtro: {len(go):,} linhas")

go["Data"]      = go["DataHora"].dt.date
go["Ano"]       = go["DataHora"].dt.year
go["Mes"]       = go["DataHora"].dt.month
go["DiaSemana"] = go["DataHora"].dt.dayofweek
go["Cell_Lat"]  = go["Latitude"].round(1)
go["Cell_Lon"]  = go["Longitude"].round(1)

print("\n[2/6] Construindo mapeamento de células...")
total_focos = len(go)
contagem = go.groupby(["Cell_Lat","Cell_Lon"]).size().reset_index(name="Contagem")
contagem = contagem[contagem["Contagem"] >= MIN_OCORRENCIAS].reset_index(drop=True)
contagem["Cell_Freq"] = contagem["Contagem"] / total_focos
print(f"  Células com ≥{MIN_OCORRENCIAS} ocorrências: {len(contagem):,}")

mapa_mun = pd.read_csv(os.path.join(DADOS, "mapeamento_municipio.csv"))
cell_lats = contagem["Cell_Lat"].values; cell_lons = contagem["Cell_Lon"].values
mun_lats  = mapa_mun["Latitude"].values; mun_lons  = mapa_mun["Longitude"].values
dists = (cell_lats[:,None]-mun_lats[None,:])**2 + (cell_lons[:,None]-mun_lons[None,:])**2
contagem["Nearest_Municipio"] = mapa_mun["Municipio"].values[np.argmin(dists, axis=1)]
contagem.to_csv(os.path.join(DADOS, "mapeamento_grade.csv"), index=False)
print(f"  Municípios proxy únicos: {contagem['Nearest_Municipio'].nunique()}")

print("\n[3/6] Gerando exemplos positivos...")
go_valid = go.merge(contagem[["Cell_Lat","Cell_Lon"]], on=["Cell_Lat","Cell_Lon"], how="inner")
positivos = (go_valid.groupby(["Cell_Lat","Cell_Lon","Data"])
             .agg(Ano=("Ano","first"),Mes=("Mes","first"),DiaSemana=("DiaSemana","first"))
             .reset_index())
positivos["Data"] = pd.to_datetime(positivos["Data"])
positivos["fogo"] = 1
print(f"  {len(positivos):,} pares (célula, dia) com fogo")

print("\n[4/6] Carregando clima histórico em memória...")
clima = pd.read_csv(os.path.join(DADOS, "clima_historico.csv"), parse_dates=["Data"])
clima = clima.rename(columns={"Municipio": "Nearest_Municipio"})
print(f"  {len(clima):,} registros climáticos")

n_anos = go["Ano"].nunique()
climatologia = (positivos.groupby(["Cell_Lat","Cell_Lon","Mes"]).size()
                .div(n_anos).reset_index(name="media_focos_mes_hist"))

print("\n[5/6] Gerando grid completo por ano (chunks ~1M linhas)...")
print(f"  Estimativa: {len(contagem):,} células × 3652 dias ≈ {len(contagem)*3652/1e6:.1f}M linhas")

SAIDA = os.path.join(DADOS, "dataset_grade.csv")
cols  = ["Cell_Lat","Cell_Lon","Nearest_Municipio","Data","Ano","Mes","DiaSemana",
         "Estacao_Seca","Cell_Freq","DiaSemChuva","Precipitacao","media_focos_mes_hist","fogo"]

cells_ref = contagem[["Cell_Lat","Cell_Lon","Cell_Freq","Nearest_Municipio"]].copy()
cells_ref["_key"] = 1
total_linhas = 0; total_pos = 0; t_total = time.time(); primeiro = True

for year in range(2015, 2026):
    t0 = time.time()
    year_days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    days_df = pd.DataFrame({"Data": year_days, "_key": 1})
    chunk = cells_ref.merge(days_df, on="_key").drop(columns="_key")
    chunk["Ano"] = chunk["Data"].dt.year; chunk["Mes"] = chunk["Data"].dt.month
    chunk["DiaSemana"] = chunk["Data"].dt.dayofweek
    chunk["Estacao_Seca"] = chunk["Mes"].between(6,10).astype(int)

    pos_year = positivos[positivos["Ano"]==year][["Cell_Lat","Cell_Lon","Data","fogo"]]
    chunk = chunk.merge(pos_year, on=["Cell_Lat","Cell_Lon","Data"], how="left")
    chunk["fogo"] = chunk["fogo"].fillna(0).astype(int)

    chunk = chunk.merge(climatologia, on=["Cell_Lat","Cell_Lon","Mes"], how="left")
    chunk["media_focos_mes_hist"] = chunk["media_focos_mes_hist"].fillna(0)
    chunk = chunk.merge(clima[["Nearest_Municipio","Data","Precipitacao","DiaSemChuva"]],
                        on=["Nearest_Municipio","Data"], how="left")
    n_sem = chunk["DiaSemChuva"].isna().sum()
    if n_sem > 0:
        print(f"    Aviso: {n_sem:,} linhas sem dado climático em {year} — mantendo NaN.")

    n_pos = chunk["fogo"].sum(); total_linhas += len(chunk); total_pos += n_pos
    chunk[cols].to_csv(SAIDA, mode="w" if primeiro else "a", header=primeiro, index=False)
    primeiro = False
    print(f"  {year}: {len(chunk):,} linhas | {n_pos:,} positivos ({100*n_pos/len(chunk):.2f}%) | {time.time()-t0:.0f}s")

print(f"\n{'='*65}")
print(f"  Total: {total_linhas:,} | Positivos: {total_pos:,} ({100*total_pos/total_linhas:.2f}%)")
print(f"  Células: {len(contagem):,} | Tempo: {(time.time()-t_total)/60:.1f} min")
print(f"  Salvo: dados/dataset_grade.csv | dados/mapeamento_grade.csv")
print(f"{'='*65}")
print("\n[OK] E2 Fase 1 concluída!")
