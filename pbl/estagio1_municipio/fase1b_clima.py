"""
vigIA — Estágio 1 | Fase 1b: Clima histórico por município (Open-Meteo)
Entrada:  ../dados/mapeamento_municipio.csv
Saída:    ../dados/clima_historico.csv
"""

import os, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

_HERE        = os.path.dirname(os.path.abspath(__file__))
PBL          = os.path.dirname(_HERE)
DADOS        = os.path.join(PBL, "dados")
MAPA         = os.path.join(DADOS, "mapeamento_municipio.csv")
SAIDA        = os.path.join(DADOS, "clima_historico.csv")
LIMIAR_CHUVA = 0.1
MAX_WORKERS  = 3 

print("=" * 60)
print("  vigIA E1 — Fase 1b: Clima Histórico (Open-Meteo)")
print("=" * 60)

mapa = pd.read_csv(MAPA)
print(f"\n  {len(mapa)} municípios | 2015-01-01 → 2025-12-31")

def calc_dias_sem_chuva(precip_series, limiar=LIMIAR_CHUVA):
    dias, cont = [], 0
    for p in precip_series:
        if p is None or pd.isna(p) or p < limiar:
            cont += 1
        else:
            cont = 0
        dias.append(cont)
    return dias

def baixar_municipio(nome, lat, lon, retries=5):
    for tentativa in range(retries):
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude":   round(lat, 4),
                    "longitude":  round(lon, 4),
                    "start_date": "2015-01-01",
                    "end_date":   "2025-12-31",
                    "daily":      "precipitation_sum",
                    "timezone":   "America/Sao_Paulo",
                },
                timeout=30
            )
            if r.status_code == 429:
                espera = int(r.headers.get("Retry-After", 60))
                print(f"\n    Rate limit (429) — aguardando {espera}s...", flush=True)
                time.sleep(espera)
                continue
            r.raise_for_status()
            dados = r.json()["daily"]
            df = pd.DataFrame({
                "Municipio":    nome,
                "Data":         pd.to_datetime(dados["time"]),
                "Precipitacao": dados["precipitation_sum"],
            })
            df["DiaSemChuva"] = calc_dias_sem_chuva(df["Precipitacao"])
            return df
        except Exception as e:
            if tentativa < retries - 1:
                time.sleep(15 * (2 ** tentativa))
            else:
                print(f"    ERRO {nome}: {e}")
                return None

# Cache: pula municípios já baixados em execuções anteriores
municipios_feitos = set()
if os.path.exists(SAIDA):
    feitos = pd.read_csv(SAIDA, usecols=["Municipio"])["Municipio"].unique()
    municipios_feitos = set(feitos)
    print(f"  Retomando: {len(municipios_feitos)} municípios já no cache.")

pendentes = mapa[~mapa["Municipio"].isin(municipios_feitos)].reset_index(drop=True)
n_erros = 0
novos_count = 0

if len(pendentes) == 0:
    print("  Todos os municípios já baixados — usando cache.")
else:
    print(f"  Baixando {len(pendentes)} municípios em paralelo (workers={MAX_WORKERS})...\n")

    def _worker(row):
        return row["Municipio"], baixar_municipio(row["Municipio"], row["Latitude"], row["Longitude"])

    erros, concluidos = [], 0
    file_exists = os.path.exists(SAIDA)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_worker, row): row["Municipio"]
                   for _, row in pendentes.iterrows()}
        for future in as_completed(futures):
            nome, df_m = future.result()
            concluidos += 1
            if df_m is not None:
                # Grava imediatamente — Ctrl+C não perde o que já foi baixado
                df_m.to_csv(SAIDA, mode="a" if file_exists else "w",
                            header=not file_exists, index=False)
                file_exists = True
                novos_count += 1
                print(f"  [{concluidos:3d}/{len(pendentes)}] {nome:<35} ✓ max_seco={df_m['DiaSemChuva'].max()}", flush=True)
            else:
                erros.append(nome)
                print(f"  [{concluidos:3d}/{len(pendentes)}] {nome:<35} ✗ FALHOU", flush=True)

    if novos_count:
        print(f"\n  Salvos: {novos_count} municípios em dados/clima_historico.csv")

    n_erros = len(erros)
    if n_erros:
        print(f"  [AVISO] {n_erros} falhas: {erros}")

print(f"\n{'='*60}")
print(f"  Total no cache: {len(municipios_feitos) + novos_count}/{len(mapa)}")
print(f"  Salvo: dados/clima_historico.csv")
print(f"{'='*60}")
print("\n[OK] E1 Fase 1b concluída!")
