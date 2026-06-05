"""
vigIA — Estágio 1 | Fase 1b: Clima histórico por município (Open-Meteo)
Entrada:  ../dados/mapeamento_municipio.csv
Saída:    ../dados/clima_historico.csv
"""

import os, time, requests
import pandas as pd

_HERE        = os.path.dirname(os.path.abspath(__file__))
PBL          = os.path.dirname(_HERE)
DADOS        = os.path.join(PBL, "dados")
MAPA         = os.path.join(DADOS, "mapeamento_municipio.csv")
SAIDA        = os.path.join(DADOS, "clima_historico.csv")
LIMIAR_CHUVA = 0.1

print("=" * 60)
print("  vigIA E1 — Fase 1b: Clima Histórico (Open-Meteo)")
print("=" * 60)

mapa = pd.read_csv(MAPA)
print(f"\n  {len(mapa)} municípios | 2015-01-01 → 2025-12-31")
print(f"  Estimativa: ~{len(mapa)*1.2/60:.0f} min\n")

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
                print(f"\n    Rate limit (429) — aguardando {espera}s...", end=" ", flush=True)
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
                print(f"    ERRO: {e}")
                return None

municipios_feitos = set()
if os.path.exists(SAIDA):
    feitos = pd.read_csv(SAIDA, usecols=["Municipio"])["Municipio"].unique()
    municipios_feitos = set(feitos)
    print(f"  Retomando: {len(municipios_feitos)} municípios já baixados.\n")

erros = []
for i, row in mapa.iterrows():
    nome = row["Municipio"]
    if nome in municipios_feitos:
        continue
    print(f"  [{i+1:3d}/{len(mapa)}] {nome:<35}", end=" ", flush=True)
    t0 = time.time()
    df_mun = baixar_municipio(nome, row["Latitude"], row["Longitude"])
    if df_mun is not None:
        modo   = "a" if os.path.exists(SAIDA) else "w"
        header = not os.path.exists(SAIDA)
        df_mun.to_csv(SAIDA, mode=modo, header=header, index=False)
        print(f"✓ {time.time()-t0:.1f}s | max_seco={df_mun['DiaSemChuva'].max()}")
    else:
        erros.append(nome)
        print("✗ FALHOU")
    time.sleep(1.5)

print(f"\n{'='*60}")
print(f"  Municípios baixados: {len(mapa)-len(erros)}/{len(mapa)}")
if erros:
    print(f"  Falhas: {erros}")
print(f"  Salvo: dados/clima_historico.csv")
print(f"{'='*60}")
print("\n[OK] E1 Fase 1b concluída!")
