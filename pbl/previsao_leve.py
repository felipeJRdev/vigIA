"""
previsao_leve.py — Pipeline E1 + E2 sem datasets pesados.
Requer apenas lookup_municipio.csv e lookup_grade.csv (gerados por gerar_lookups.py).

Entrada:
  modelos/municipio_full.pkl
  modelos/grade_full.pkl
  dados/mapeamento_municipio.csv
  dados/mapeamento_grade.csv
  dados/lookup_municipio.csv      (≤ 3 k linhas)
  dados/lookup_grade.csv          (≤ 36 k linhas)
  Open-Meteo API (244 chamadas paralelas, ~1 min)

Saída:
  resultados/previsao_municipio_<hoje>.csv
  resultados/previsao_grade_<hoje>.csv
"""

import os, sys, time, requests, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
DADOS      = os.path.join(_HERE, "dados")
MODELOS    = os.path.join(_HERE, "modelos")
RESULTADOS = os.path.join(_HERE, "resultados")
LIMIAR_CHUVA = 0.1

FEATURES_E1 = [
    "Mes", "DiaSemana", "Estacao_Seca",
    "Latitude", "Longitude", "Municipio_Freq",
    "DiaSemChuva", "Precipitacao", "media_focos_mes_hist",
]
FEATURES_E2 = [
    "Mes", "DiaSemana", "Estacao_Seca",
    "Cell_Lat", "Cell_Lon", "Cell_Freq",
    "DiaSemChuva", "Precipitacao", "media_focos_mes_hist",
]

def _risco(p):
    return "ALTO" if p >= 0.70 else "MÉDIO" if p >= 0.40 else "BAIXO"

hoje      = date.today()
dias_prev = [hoje + timedelta(days=i) for i in range(1, 6)]

print("=" * 65)
print("  vigIA — Previsão Leve E1+E2 — Próximos 5 Dias")
print("=" * 65)
print(f"  Hoje: {hoje}  |  Previsão: {dias_prev[0]} → {dias_prev[-1]}")

# ═══════════════════════════════════════════════════════════════
# ESTÁGIO 1 — Município
# ═══════════════════════════════════════════════════════════════
print("\n─── Estágio 1: Município ───────────────────────────────────")

print("[E1-1] Carregando modelo e lookups...")
arte1    = joblib.load(os.path.join(MODELOS, "municipio_full.pkl"))
modelo1  = arte1["modelo"]
mapa     = pd.read_csv(os.path.join(DADOS, "mapeamento_municipio.csv"))
lookup1  = pd.read_csv(os.path.join(DADOS, "lookup_municipio.csv"))
print(f"  Modelo: {arte1['nome']} | Municípios: {len(mapa)} | Lookup: {len(lookup1):,} linhas")

print("[E1-2] Buscando dados climáticos (Open-Meteo)...")

def buscar_clima(nome, lat, lon, retries=4):
    # Uma única chamada com past_days=30 substitui archive + forecast
    for tentativa in range(retries):
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": round(lat, 4), "longitude": round(lon, 4),
                    "daily": "precipitation_sum",
                    "past_days": 30,
                    "forecast_days": 6,
                    "timezone": "America/Sao_Paulo",
                },
                timeout=20,
            )
            if r.status_code == 429:
                espera = int(r.headers.get("Retry-After", 60))
                time.sleep(espera)
                continue
            r.raise_for_status()

            data      = r.json()["daily"]
            prec_all  = data["precipitation_sum"]  # 30 passado + 1 hoje + 5 previsão = 36
            prec_hist = prec_all[:31]               # últimos 30 dias + hoje (índice 30 é valor parcial do dia, aceitável)
            prec_prev = prec_all[31:36]             # amanhã → hoje+5
            datas_prev = data["time"][31:36]

            dias_secos = 0
            for p in prec_hist:
                if p is None or pd.isna(p) or p < LIMIAR_CHUVA:
                    dias_secos += 1
                else:
                    dias_secos = 0

            registros = []
            contador = dias_secos
            for data_str, precip in zip(datas_prev, prec_prev):
                if precip is None or pd.isna(precip) or precip < LIMIAR_CHUVA:
                    contador += 1
                else:
                    contador = 0
                registros.append({
                    "Municipio":    nome,
                    "Data":         pd.to_datetime(data_str).date(),
                    "Precipitacao": precip if precip else 0.0,
                    "DiaSemChuva":  contador,
                })
            return registros
        except Exception:
            if tentativa < retries - 1:
                time.sleep(5 * (2 ** tentativa))
            else:
                return None

todos, erros = [], []

def _worker(args):
    _, row = args
    return row["Municipio"], buscar_clima(row["Municipio"], row["Latitude"], row["Longitude"])

concluidos = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(_worker, item): item[1]["Municipio"]
               for item in mapa.iterrows()}
    for future in as_completed(futures):
        nome, resultado = future.result()
        concluidos += 1
        if resultado:
            todos.extend(resultado)
            print(f"  [{concluidos:3d}/244] {nome:<35} ✓ seco={resultado[0]['DiaSemChuva']}d", flush=True)
        else:
            erros.append(nome)
            print(f"  [{concluidos:3d}/244] {nome:<35} ✗ ERRO", flush=True)

if erros:
    print(f"\n  Retentando {len(erros)} municípios com erro (sequencial, 3 tentativas, 30s entre chamadas)...")
    ainda_erro = list(erros)
    for tentativa_global in range(3):
        if not ainda_erro:
            break
        print(f"  [tentativa {tentativa_global + 1}/3] {len(ainda_erro)} pendentes...")
        time.sleep(30)
        restantes = []
        for nome_err in ainda_erro:
            row = mapa[mapa["Municipio"] == nome_err].iloc[0]
            resultado = buscar_clima(nome_err, row["Latitude"], row["Longitude"])
            if resultado:
                todos.extend(resultado)
                print(f"  {nome_err:<35} ✓ (retry ok)", flush=True)
            else:
                restantes.append(nome_err)
                print(f"  {nome_err:<35} ✗ ainda falhou", flush=True)
            time.sleep(5)
        ainda_erro = restantes
    erros = ainda_erro

if erros:
    print(f"\n[ERRO FATAL] {len(erros)} municípios sem dados após todas as tentativas:")
    for m in erros:
        print(f"  - {m}")
    sys.exit(1)

print("[E1-3] Calculando features e predizendo...")
clima1 = pd.DataFrame(todos)
clima1["Data"]         = pd.to_datetime(clima1["Data"])
clima1["Mes"]          = clima1["Data"].dt.month
clima1["DiaSemana"]    = clima1["Data"].dt.dayofweek
clima1["Estacao_Seca"] = clima1["Mes"].between(6, 10).astype(int)

clima1 = clima1.merge(
    mapa[["Municipio", "Municipio_Freq", "Latitude", "Longitude"]],
    on="Municipio", how="left",
)
clima1 = clima1.merge(lookup1, on=["Municipio", "Mes"], how="left")
clima1["media_focos_mes_hist"] = clima1["media_focos_mes_hist"].fillna(0)

clima1["prob_fogo"] = modelo1.predict_proba(clima1[FEATURES_E1].values)[:, 1]
clima1["risco"]     = clima1["prob_fogo"].apply(_risco)

nome_mun = f"previsao_municipio_{hoje}.csv"
(clima1.sort_values(["Data", "prob_fogo"], ascending=[True, False])
       .to_csv(os.path.join(RESULTADOS, nome_mun), index=False))
print(f"  Salvo: resultados/{nome_mun}")

alto_total  = (clima1["risco"] == "ALTO").sum()
medio_total = (clima1["risco"] == "MÉDIO").sum()
print(f"  ALTO: {alto_total}  MÉDIO: {medio_total}  (5 dias, {len(mapa)} municípios)")

# ═══════════════════════════════════════════════════════════════
# ESTÁGIO 2 — Grade
# ═══════════════════════════════════════════════════════════════
print("\n─── Estágio 2: Grade Espacial ──────────────────────────────")

print("[E2-1] Carregando modelo grade e lookups...")
arte2   = joblib.load(os.path.join(MODELOS, "grade_full.pkl"))
modelo2 = arte2["modelo"]
grade   = pd.read_csv(os.path.join(DADOS, "mapeamento_grade.csv"))
lookup2 = pd.read_csv(os.path.join(DADOS, "lookup_grade.csv"))
print(f"  Modelo: {arte2['nome']} | Células: {len(grade):,} | Lookup: {len(lookup2):,} linhas")

print("[E2-2] Expandindo células e predizendo...")
clima_prev = clima1[["Municipio", "Data", "DiaSemChuva", "Precipitacao"]].copy()

cells = grade[["Cell_Lat", "Cell_Lon", "Cell_Freq", "Nearest_Municipio"]].merge(
    clima_prev.rename(columns={"Municipio": "Nearest_Municipio"}),
    on="Nearest_Municipio", how="inner",
)
cells["Mes"]          = cells["Data"].dt.month
cells["DiaSemana"]    = cells["Data"].dt.dayofweek
cells["Estacao_Seca"] = cells["Mes"].between(6, 10).astype(int)
cells = cells.merge(lookup2, on=["Cell_Lat", "Cell_Lon", "Mes"], how="left")
cells["media_focos_mes_hist"] = cells["media_focos_mes_hist"].fillna(0)

cells["prob_fogo"] = modelo2.predict_proba(cells[FEATURES_E2].values)[:, 1]
cells["risco"]     = cells["prob_fogo"].apply(_risco)

nome_grade = f"previsao_grade_{hoje}.csv"
(cells.sort_values(["Data", "prob_fogo"], ascending=[True, False])
      .to_csv(os.path.join(RESULTADOS, nome_grade), index=False))
print(f"  Salvo: resultados/{nome_grade}")

alto_c  = (cells["risco"] == "ALTO").sum()
medio_c = (cells["risco"] == "MÉDIO").sum()
print(f"  ALTO: {alto_c}  MÉDIO: {medio_c}  (5 dias, {len(grade):,} células)")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Próximo passo: python3 exportar_json.py")
print("=" * 65)
print("[OK] previsao_leve.py concluída!")
