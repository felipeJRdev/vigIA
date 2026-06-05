"""
vigIA — Estágio 2 | Fase 1b: Clima histórico por grade 0.5° (148 pontos → 2.976 células)
Estratégia: 148 pontos 0.5° cobrem todas as células 0.1° (erro máx ~35km, sem rate limit).
Entrada:  ../dados/mapeamento_grade.csv
Saída:    ../dados/clima_pontos_05.csv   (dados brutos por ponto 0.5°)
          ../dados/clima_grade.csv       (expandido para células 0.1°)
"""

import os, time, requests
import pandas as pd
import numpy as np

_HERE        = os.path.dirname(os.path.abspath(__file__))
PBL          = os.path.dirname(_HERE)
DADOS        = os.path.join(PBL, "dados")
MAPA         = os.path.join(DADOS, "mapeamento_grade.csv")
PONTOS_CSV   = os.path.join(DADOS, "clima_pontos_05.csv")
SAIDA        = os.path.join(DADOS, "clima_grade.csv")
LIMIAR_CHUVA = 0.1

print("=" * 65)
print("  vigIA E2 — Fase 1b: Clima 0.5° → Células 0.1°")
print("=" * 65)

grade = pd.read_csv(MAPA)
grade["Clima_Lat"] = (grade["Cell_Lat"] / 0.5).round() * 0.5
grade["Clima_Lon"] = (grade["Cell_Lon"] / 0.5).round() * 0.5
pontos = grade[["Clima_Lat","Clima_Lon"]].drop_duplicates().reset_index(drop=True)
print(f"\n  Pontos 0.5°: {len(pontos):,} (vs {len(grade):,} células 0.1°)")
print(f"  Tempo estimado: ~{len(pontos)*4/60:.0f} min\n")

def calc_dias_sem_chuva(series, limiar=LIMIAR_CHUVA):
    dias, cont = [], 0
    for p in series:
        if p is None or pd.isna(p) or p < limiar: cont += 1
        else: cont = 0
        dias.append(cont)
    return dias

def baixar_ponto(lat, lon, retries=5):
    for tentativa in range(retries):
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": round(lat,2), "longitude": round(lon,2),
                        "start_date": "2015-01-01", "end_date": "2025-12-31",
                        "daily": "precipitation_sum", "timezone": "America/Sao_Paulo"},
                timeout=30)
            if r.status_code == 429:
                espera = int(r.headers.get("Retry-After", 60))
                print(f"\n    Rate limit — aguardando {espera}s...", end=" ", flush=True)
                time.sleep(espera); continue
            r.raise_for_status()
            dados = r.json()["daily"]
            df = pd.DataFrame({"Clima_Lat": lat, "Clima_Lon": lon,
                                "Data": pd.to_datetime(dados["time"]),
                                "Precipitacao": dados["precipitation_sum"]})
            df["DiaSemChuva"] = calc_dias_sem_chuva(df["Precipitacao"])
            return df
        except Exception as e:
            if tentativa < retries-1: time.sleep(15*(2**tentativa))
            else: print(f"    ERRO: {e}"); return None

pontos_feitos = set()
if os.path.exists(PONTOS_CSV):
    feitos = pd.read_csv(PONTOS_CSV, usecols=["Clima_Lat","Clima_Lon"])
    pontos_feitos = set(zip(feitos["Clima_Lat"].round(2), feitos["Clima_Lon"].round(2)))
    print(f"  Retomando: {len(pontos_feitos)} pontos já baixados.\n")

erros = []; t_inicio = time.time()
for i, row in pontos.iterrows():
    lat = round(row["Clima_Lat"], 2); lon = round(row["Clima_Lon"], 2)
    if (lat, lon) in pontos_feitos: continue
    print(f"  [{i+1:3d}/{len(pontos)}] ({lat:.1f}, {lon:.1f})", end=" ", flush=True)
    df_p = baixar_ponto(lat, lon)
    if df_p is not None:
        modo = "a" if os.path.exists(PONTOS_CSV) else "w"
        df_p.to_csv(PONTOS_CSV, mode=modo, header=not os.path.exists(PONTOS_CSV), index=False)
        pontos_feitos.add((lat, lon))
        print(f"✓ max_seco={df_p['DiaSemChuva'].max()}")
    else:
        erros.append((lat, lon)); print("✗ FALHOU")
    time.sleep(4.0)

print(f"\n  Download: {len(pontos_feitos)}/{len(pontos)} pontos em {(time.time()-t_inicio)/60:.1f} min")

print("\n  Expandindo para células 0.1°...")
clima_pts = pd.read_csv(PONTOS_CSV, parse_dates=["Data"])
mapa_clima = grade[["Cell_Lat","Cell_Lon","Clima_Lat","Clima_Lon"]].drop_duplicates()
clima_grade = mapa_clima.merge(clima_pts, on=["Clima_Lat","Clima_Lon"], how="left")
clima_grade[["Cell_Lat","Cell_Lon","Data","Precipitacao","DiaSemChuva"]].to_csv(SAIDA, index=False)

print(f"\n{'='*65}")
print(f"  clima_grade.csv: {len(clima_grade):,} linhas ({grade['Cell_Lat'].nunique()} células × ~3652 dias)")
if erros: print(f"  Falhas ({len(erros)}): {erros}")
print(f"  Salvo: dados/clima_grade.csv | dados/clima_pontos_05.csv")
print(f"{'='*65}")
print("\n[OK] E2 Fase 1b concluída!")
print("  Próximo: python3 estagio2_grade/fase1c_aplicar_clima.py")
