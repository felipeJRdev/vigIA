# vigIA
Repositório da disciplina FGA0083 — Aprendizado de Máquina | UnB 2026-1 | Turma 01 | Grupo 3

> Sistema de previsão de risco de queimadas no estado de Goiás usando classificação binária com LightGBM, dados históricos do BDqueimadas/INPE e clima via Open-Meteo.

---

## Grupo
- Felipe de Jesus Rodrigues — 211062867
- João Paulo Barros de Cristo — 202023805
- Guilherme Aguera de la Fuente Vilela — 190088168
- Luiz Guilherme Morais da Costa Faria — 231011696

---

## O que foi feito

### Mini Trabalhos
| MT | Descrição | Resultado |
|---|---|---|
| MT4 | Preparação de dados, pipeline de regressão FRP_log | — |
| MT5 | Seleção de modelos (8 algoritmos) | RF R²=0.6449 |
| MT6 | Otimização de hiperparâmetros (RF, XGBoost, LightGBM) | RF R²=0.7282 |

### PBL — vigIA
Redefinição do problema: em vez de prever FRP (já fornecido pelo BDqueimadas), o sistema prevê **onde haverá focos antes de eles ocorrerem**, classificando o risco por unidade geográfica e dia.

**Arquitetura em dois estágios independentes:**

```
Estágio 1 — Município (247 unidades)
  modelo: LightGBM | AUC 0.816 (validação Jan-Jun 2026)
  pergunta: "Quais municípios estão em risco nos próximos 5 dias?"

Estágio 2 — Grade Espacial 0.1° × 0.1° (2.976 células, ~11km)
  modelo: LightGBM | AUC 0.715 (validação Jan-Jun 2026)
  pergunta: "Onde dentro do município?"
```

Os dois modelos rodam em paralelo com as mesmas features climáticas. A separação em estágios é de **apresentação** — o modelo 2 não recebe saída do modelo 1.

**Frontend:** mapa interativo com Leaflet.js + satélite Esri, sem servidor necessário.

---

## Estrutura do projeto

```
pbl/
├── estagio1_municipio/       scripts do Estágio 1
│   ├── fase1_dataset.py
│   ├── fase1b_clima.py
│   ├── fase2_modelagem.py
│   ├── fase3_validacao_2026.py
│   └── fase4_previsao_5dias.py
├── estagio2_grade/           scripts do Estágio 2
│   ├── fase1_dataset_grade.py
│   ├── fase1b_clima_05graus.py
│   ├── fase1c_aplicar_clima.py
│   ├── fase2_modelagem.py
│   ├── fase3_validacao_2026.py
│   └── fase4_previsao_offline.py
├── frontend/
│   ├── gerar_mapa.py         gera o mapa HTML
│   └── mapa_vigia.html       mapa interativo (abrir no navegador)
├── dados/                    CSVs gerados pelos scripts
├── modelos/                  arquivos .pkl gerados pelo treino
├── resultados/               métricas e previsões em CSV
├── graficos/                 PNGs gerados
└── docs/
    ├── DOCUMENTACAO_PBL.md   documentação técnica completa

```

> **Nota:** `dados/`, `modelos/` e `*.pkl`/`*.csv` estão no `.gitignore`. Os arquivos são gerados localmente rodando os scripts na ordem abaixo.

---

## Como rodar

### 1. Requisitos

```bash
python3 -m venv env
source env/bin/activate
pip install scikit-learn xgboost lightgbm mlflow pandas numpy matplotlib requests
```

### 2. Dataset bruto

Baixar o dataset completo do BDqueimadas/INPE e salvar em:
```
/home/<usuario>/AprendizadoMaquina/dataset_queimadas_completo.csv
```

Fonte: [BDqueimadas](https://bdqueimadas.inpe.br/) — filtrar por Estado = Goiás, período 2015–2025.

Para validação com 2026, exportar também o período 01/01/2026–hoje filtrado por Goiás + Cerrado e salvar em `pbl/dados/bdqueimadas_2026-01-01_<data>.csv`.

### 3. Rodar o pipeline — Estágio 1 (município)

```bash
cd pbl

# Gera dataset_municipio.csv e mapeamento_municipio.csv
python3 estagio1_municipio/fase1_dataset.py

# Baixa precipitação histórica 2015-2025 para 247 municípios (~6 min)
python3 estagio1_municipio/fase1b_clima.py

# Treina RF, XGBoost e LightGBM com busca de hiperparâmetros (~20 min)
python3 estagio1_municipio/fase2_modelagem.py

# Retreina no dataset completo + valida com dados reais 2026
python3 estagio1_municipio/fase3_validacao_2026.py

# Previsão dos próximos 5 dias (requer conexão com Open-Meteo)
python3 estagio1_municipio/fase4_previsao_5dias.py
```

### 4. Rodar o pipeline — Estágio 2 (grade espacial)

```bash
# Gera dataset_grade.csv com 11.9M linhas (~1 min)
python3 estagio2_grade/fase1_dataset_grade.py

# Baixa clima para 148 pontos 0.5° (~10 min, requer API liberada)
python3 estagio2_grade/fase1b_clima_05graus.py

# Aplica clima exato no dataset
python3 estagio2_grade/fase1c_aplicar_clima.py

# Treina modelos na grade (~30 min)
python3 estagio2_grade/fase2_modelagem.py

# Retreino completo + validação 2026
python3 estagio2_grade/fase3_validacao_2026.py

# Previsão dos próximos 5 dias (offline, sem chamadas extras à API)
python3 estagio2_grade/fase4_previsao_offline.py
```

### 5. Gerar o mapa interativo

```bash
python3 frontend/gerar_mapa.py
xdg-open frontend/mapa_vigia.html   # Linux
# ou abrir o arquivo diretamente no navegador
```

---

## Dados climáticos

Todos os dados climáticos são obtidos gratuitamente via [Open-Meteo](https://open-meteo.com/) sem necessidade de autenticação.

- **Archive API** — precipitação histórica por coordenada geográfica
- **Forecast API** — previsão de precipitação para os próximos 7 dias

A API tem rate limit de ~100 req/min no tier gratuito. Os scripts já incluem delays e retomada automática em caso de interrupção.

---

## Resultados principais

| | Estágio 1 — Município | Estágio 2 — Grade 0.1° |
|---|---|---|
| Dataset treino | 992k amostras | 18.4M amostras |
| AUC Teste (2024-25) | **0.835** | 0.831 |
| AUC Validação 2026 | **0.816** | 0.710 |
| Limiar operacional | 0.3 | 0.6 |
| Recall | **79.2%** | 70.3% |
| Precisão | **95.3%** | 42.7% |
| Top 20% captura | **62.7%** dos fogos | 48.5% dos fogos |

Validação com dados reais BDqueimadas Jan–Jun 2026 (nunca vistos no treino).  
Clima via Open-Meteo grade 0.5° (148 pontos, erro máx ~35km).
