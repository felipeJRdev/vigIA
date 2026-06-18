"""
vigIA — Estágio 1 | Fase 1: Dataset de classificação por município
Entrada:  ../../dataset_queimadas_completo.csv
          ../dados/clima_historico.csv
Saída:    ../dados/dataset_municipio.csv
          ../dados/mapeamento_municipio.csv
"""

import os, time
import pandas as pd
from itertools import product

_HERE      = os.path.dirname(os.path.abspath(__file__))
PBL        = os.path.dirname(_HERE)
BASE       = os.path.dirname(PBL)
DADOS      = os.path.join(PBL, "dados")
RAW        = os.path.join(DADOS, "dataset_queimadas_completo.csv")

print("=" * 60)
print("  vigIA E1 — Fase 1: Dataset de Classificação por Município")
print("=" * 60)

print("\n[1/6] Carregando dataset bruto...")
df = pd.read_csv(RAW, parse_dates=["DataHora"])
print(f"  Total bruto: {len(df):,} linhas")

go = df[
    (df["Estado"].str.upper() == "GOIÁS") &
    (df["Bioma"] != "Mata Atlântica")
].copy()
print(f"  Após filtro Goiás + Bioma: {len(go):,} linhas | {go['Municipio'].nunique()} municípios")

go["Data"]      = go["DataHora"].dt.date
go["Mes"]       = go["DataHora"].dt.month
go["DiaSemana"] = go["DataHora"].dt.dayofweek
go["Ano"]       = go["DataHora"].dt.year

import unicodedata as _ud
def _norm(s): return "".join(c for c in _ud.normalize("NFD", str(s)) if not _ud.category(c).startswith("M")).lower().strip()

# Normaliza grafias brutas para a canônica (maior contagem) antes de qualquer groupby.
# Evita NaN em Municipio_Freq/Lat/Lon no merge posterior.
_raw_counts = go["Municipio"].value_counts()
_norm_to_canonical: dict = {}
for _nome in _raw_counts.index:  # ordenado por contagem decrescente
    _n = _norm(_nome)
    if _n not in _norm_to_canonical:
        _norm_to_canonical[_n] = _nome
go["Municipio"] = go["Municipio"].map({_nome: _norm_to_canonical[_norm(_nome)] for _nome in _raw_counts.index})

print("\n[2/6] Construindo mapeamento de municípios...")
total = len(go)
mapa = (
    go.groupby("Municipio")
    .agg(
        Contagem  = ("Municipio", "count"),
        Latitude  = ("Latitude",  "mean"),
        Longitude = ("Longitude", "mean"),
    )
    .reset_index()
)
mapa["Municipio_Freq"] = mapa["Contagem"] / total
mapa = mapa.sort_values("Contagem", ascending=False).reset_index(drop=True)
# Remove entradas com mesmo nome normalizado (ex: "NIQUELÂNDIA" vs "Niquelândia"),
# mantendo a de maior Contagem (já ordenado desc). Causa: registros com grafia
# inconsistente no BDqueimadas geram municípios duplicados com freq quase zero.
mapa["_norm"] = mapa["Municipio"].apply(_norm)
mapa = mapa.drop_duplicates("_norm").drop(columns="_norm").reset_index(drop=True)
mapa.to_csv(os.path.join(DADOS, "mapeamento_municipio.csv"), index=False)
print(f"  {len(mapa)} municípios | Top 3: {mapa[['Municipio','Municipio_Freq']].head(3).to_string(index=False)}")

# B2: versão _train (≤ 2022) para avaliação sem vazamento temporal
_go_train = go[go["Ano"] <= 2022]
_freq_train = (_go_train.groupby("Municipio")["Municipio"]
               .count().div(len(_go_train))
               .rename("Municipio_Freq_train").reset_index())
mapa = mapa.merge(_freq_train, on="Municipio", how="left")
mapa["Municipio_Freq_train"] = mapa["Municipio_Freq_train"].fillna(0)

print("\n[3/6] Gerando exemplos positivos...")
positivos = (
    go.groupby(["Municipio", "Data"])
    .agg(
        Ano         = ("Ano",       "first"),
        Mes         = ("Mes",       "first"),
        DiaSemana   = ("DiaSemana", "first"),
        n_focos_dia = ("Municipio", "count"),
    )
    .reset_index()
)
positivos["fogo"] = 1
print(f"  {len(positivos):,} pares (município, dia) com fogo")

print("\n[4/6] Gerando exemplos negativos...")
todos_municipios = go["Municipio"].unique()
datas_min = go["Data"].min()
datas_max = go["Data"].max()
todos_os_dias = pd.date_range(datas_min, datas_max, freq="D").date
print(f"  Período: {datas_min} → {datas_max} ({len(todos_os_dias)} dias)")

positivos_set = set(zip(positivos["Municipio"], positivos["Data"]))
t0 = time.time()
negativos_raw = [
    (mun, dia)
    for mun, dia in product(todos_municipios, todos_os_dias)
    if (mun, dia) not in positivos_set
]
print(f"  {len(negativos_raw):,} negativos em {time.time()-t0:.1f}s")

negativos = pd.DataFrame(negativos_raw, columns=["Municipio", "Data"])
negativos["Ano"]         = pd.to_datetime(negativos["Data"]).dt.year
negativos["Mes"]         = pd.to_datetime(negativos["Data"]).dt.month
negativos["DiaSemana"]   = pd.to_datetime(negativos["Data"]).dt.dayofweek
negativos["n_focos_dia"] = 0
negativos["fogo"]        = 0

print("\n[5/6] Adicionando features de contexto...")
# B2: climatologia em duas versões — _full (todos os anos) e _train (≤ 2022)
climatologia_full = (
    positivos.groupby(["Municipio", "Mes"])["n_focos_dia"]
    .mean().reset_index()
    .rename(columns={"n_focos_dia": "media_focos_mes_hist"})
)
climatologia_train = (
    positivos[positivos["Ano"] <= 2022].groupby(["Municipio", "Mes"])["n_focos_dia"]
    .mean().reset_index()
    .rename(columns={"n_focos_dia": "media_focos_mes_hist_train"})
)
climatologia = climatologia_full.merge(climatologia_train, on=["Municipio","Mes"], how="left")
climatologia["media_focos_mes_hist_train"] = climatologia["media_focos_mes_hist_train"].fillna(0)

dataset = pd.concat([positivos, negativos], ignore_index=True)
dataset["Estacao_Seca"] = dataset["Mes"].apply(lambda m: 1 if m in [6,7,8,9,10] else 0)
dataset = dataset.merge(
    mapa[["Municipio","Municipio_Freq","Municipio_Freq_train","Latitude","Longitude"]],
    on="Municipio", how="left"
)
assert dataset["Municipio_Freq"].notna().all(), "Merge falhou: Municipio_Freq tem NaN — verificar normalização de grafias"
# B1: clima Open-Meteo para positivos e negativos (elimina vazamento de fonte)
dataset["Data"] = pd.to_datetime(dataset["Data"])
clima_hist = pd.read_csv(os.path.join(DADOS, "clima_historico.csv"), parse_dates=["Data"])
dataset = dataset.merge(
    clima_hist[["Municipio","Data","Precipitacao","DiaSemChuva"]],
    on=["Municipio","Data"], how="left"
)
n_sem = dataset["DiaSemChuva"].isna().sum()
if n_sem > 0:
    print(f"  Aviso: {n_sem:,} linhas sem dado climático — mantendo NaN.")
dataset = dataset.merge(climatologia, on=["Municipio","Mes"], how="left")
dataset["media_focos_mes_hist"]       = dataset["media_focos_mes_hist"].fillna(0)
dataset["media_focos_mes_hist_train"] = dataset["media_focos_mes_hist_train"].fillna(0)

print("\n[6/6] Salvando...")
dataset = dataset.sort_values(["Data","Municipio"]).reset_index(drop=True)
cols = [
    "Municipio","Data","Ano","Mes","DiaSemana","Estacao_Seca",
    "Latitude","Longitude","Municipio_Freq","Municipio_Freq_train",
    "DiaSemChuva","Precipitacao","n_focos_dia",
    "media_focos_mes_hist","media_focos_mes_hist_train","fogo"
]
dataset[cols].to_csv(os.path.join(DADOS, "dataset_municipio.csv"), index=False)

n_pos = (dataset["fogo"]==1).sum()
n_neg = (dataset["fogo"]==0).sum()
print(f"\n{'='*60}")
print(f"  Total: {len(dataset):,} | Positivos: {n_pos:,} ({100*n_pos/len(dataset):.1f}%) | Negativos: {n_neg:,}")
print(f"  Salvo: dados/dataset_municipio.csv | dados/mapeamento_municipio.csv")
print(f"{'='*60}")
print("\n[OK] E1 Fase 1 concluída!")
