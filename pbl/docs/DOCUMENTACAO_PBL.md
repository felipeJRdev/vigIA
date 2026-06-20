# vigIA — Documentação do PBL
**Previsão de Risco de Queimadas no Estado de Goiás**  
**Disciplina:** FGA0083 — Aprendizado de Máquina | UnB | 2026-1 | Turma 01 | Grupo 4  
**Data:** 2026-06-10

## Membros do Grupo
- Felipe de Jesus Rodrigues — 211062867
- João Paulo Barros de Cristo — 202023805
- Guilherme Aguera de la Fuente Vilela — 190088168
- Luiz Guilherme Morais da Costa Faria — 231011696

---

## 1. Contexto e Motivação

### 1.1 O que foi feito nos Mini Trabalhos anteriores
Nos MTs 4, 5 e 6 foi desenvolvido um modelo de **regressão** para prever o `FRP_log` (Fire Radiative Power em escala logarítmica) de focos de incêndio já detectados pelo satélite. O melhor resultado foi:

- **Random Forest:** R² = 0.7282 no conjunto holdout (após otimização no MT6)

### 1.2 Limitação identificada
O modelo de regressão tem valor limitado em produção porque o BDqueimadas/INPE já fornece o FRP medido pelo satélite junto com todas as outras colunas usadas como features. Ou seja, quando você tem os dados para rodar o modelo, já tem a resposta que ele prevê.

### 1.3 Redefinição do problema
Para o PBL, o problema foi redefinido para algo genuinamente útil:

> **"Dado o histórico de queimadas e as condições climáticas atuais, quais áreas de Goiás têm maior probabilidade de registrar focos de incêndio nos próximos dias?"**

Isso é **classificação binária** por unidade geográfica por dia, **antes** de qualquer foco existir.

---

## 2. Arquitetura do Sistema: Dois Estágios Independentes

Os dois modelos rodam de forma **independente** com as mesmas features climáticas. Ambos são executados diariamente pelo cron job. O "dois estágios" é uma decisão de **apresentação ao usuário**, não de pipeline em cascata (a saída do modelo 1 não entra no modelo 2).

```
Cron job diário
    │
    ├─► Modelo 1 (municipio_full.pkl)
    │     Features climáticas por município → P(fogo) × 244 municípios
    │     Saída: ranking — "NIQUELÂNDIA 98%, CAVALCANTE 95%..."
    │
    └─► Modelo 2 (grade_full.pkl)
          Features climáticas por célula → P(fogo) × 2.976 células
          Saída: probabilidade para cada célula 0.1° de Goiás

Frontend (apresentação em dois estágios):
  Estágio 1 → ranking de municípios (visão geral, alerta regional)
  Estágio 2 → usuário clica num município → zoom no mapa
               células daquele município coloridas por risco relativo
```

```
┌─────────────────────────────────────────────────────────┐
│  ESTÁGIO 1 — Modelo de Município (triagem regional)     │
│  Unidade: 244 municípios de Goiás (bioma Cerrado)       │
│  Modelo:  modelos/municipio_full.pkl (LightGBM)         │
│  AUC:     0.816 (validação Jan-Jun 2026)                │
│  Pergunta: "QUAIS municípios estão em risco?"           │
└─────────────────────────────────────────────────────────┘
                    roda em paralelo ↕ (independente)
┌─────────────────────────────────────────────────────────┐
│  ESTÁGIO 2 — Modelo de Grade Espacial (drill-down)      │
│  Unidade: 2.976 células 0.1° × 0.1° (~11km × 11km)     │
│  Modelo:  modelos/grade_full.pkl (LightGBM)             │
│  AUC:     0.710 (validação 2026 com clima proxy)        │
│  Pergunta: "ONDE dentro do município?"                  │
└─────────────────────────────────────────────────────────┘
```

**Por que dois estágios em vez de um só?**

| Aspecto | Município | Grade 0.1° |
|---|---|---|
| AUC validação 2026 | 0.816 | 0.710 |
| Unidades | 244 | 2.976 |
| Resolução geográfica | ~50km | ~11km |
| Custo computacional | Baixo | Alto |
| Uso em produção | Alerta diário | Visualização, planejamento |

O modelo de município é mais preciso porque cada município agrega muitos focos históricos (sinal forte). O modelo de grade é menos preciso mas oferece localização dentro do município — crucial para operações de campo.

No estágio 2, o sistema usa **ranking relativo dentro do município** (não probabilidade absoluta), o que é robusto a problemas de calibração.

---

## 3. Fonte de Dados

**BDqueimadas / INPE** — Sistema de monitoramento de queimadas por satélite  
- Escopo: Estado de Goiás | 2015–2025  
- Registros brutos: 1.390.827 focos detectados  
- Após filtro (Goiás + remoção Mata Atlântica): **1.354.326 registros | 244 municípios**

Goiás possui 246 municípios. Dois foram excluídos do escopo do modelo: **Gouvelândia** e **São Simão**, que pertencem ao bioma Mata Atlântica. O modelo foi treinado exclusivamente com dados do bioma Cerrado — incluir municípios de outro bioma introduziria padrões de vegetação e sazonalidade incompatíveis com os 244 municípios do conjunto de treino, degradando a qualidade das previsões. No frontend, esses dois municípios aparecem sinalizados separadamente no mapa.

**Open-Meteo API** — Dados climáticos históricos e de previsão, gratuitos (sem autenticação)  
- Precipitação diária por coordenada geográfica
- Período: 2015–2025 (treino) + Janeiro–Junho 2026 (validação) + uso diário em produção

---

## 4. Estágio 1 — Modelo de Município

### 4.1 Construção do Dataset (Fase 1)

**Estratégia para exemplos negativos:**

O BDqueimadas registra **apenas** dias com foco detectado. Para treinar um classificador binário são necessários exemplos de dias **sem** fogo. A premissa adotada:

> **Ausência de registro no BDqueimadas = ausência de foco detectado naquele município naquele dia.**

Isso é válido porque o sistema de satélites cobre Goiás continuamente.

**Geração dos exemplos:**

```
244 municípios × 4.018 dias (2015-2025) = 980.392 combinações
980.392 − 143.067 positivos = 837.325 negativos (naturais, sem amostragem)
```

| Classe | Registros | % |
|---|---|---|
| fogo = 1 | 143.067 | 14,6% |
| fogo = 0 | 837.325 | 85,4% |
| **Total** | **980.392** | 100% |

**Features do modelo:**

| Feature | Tipo | Fonte |
|---|---|---|
| `Mes` | Temporal | Data |
| `DiaSemana` | Temporal | Data |
| `Estacao_Seca` | Temporal | Mes ∈ {6,7,8,9,10} → 1 |
| `Latitude` | Geográfica | Centroide do município |
| `Longitude` | Geográfica | Centroide do município |
| `Municipio_Freq` | Histórica | Focos do município / total Goiás |
| `DiaSemChuva` | Climática | Dias consecutivos < 0.1mm (Open-Meteo) |
| `Precipitacao` | Climática | Precipitação diária em mm (Open-Meteo) |
| `media_focos_mes_hist` | Histórica | Média histórica de focos por município+mês |

### 4.2 Modelagem (Fase 2)

**Split temporal obrigatório** (random split causaria data leakage):

| Conjunto | Período | Registros |
|---|---|---|
| Treino | 2015–2022 | ~712.000 |
| Validação | 2023 | ~89.000 |
| Teste | 2024–2025 | ~179.000 |

**Busca de hiperparâmetros:** RandomizedSearchCV em subsample de 200k + retreino no conjunto completo.

| Modelo | AUC Val | AUC Teste | Recall Teste | Precisão Teste |
|---|---|---|---|---|
| **LightGBM** | 0,8412 | **0,8350** | **0,750** | 0,329 |
| XGBoost | 0,8411 | 0,8343 | 0,710 | 0,350 |
| Random Forest | 0,8382 | 0,8297 | 0,740 | 0,326 |

**Modelo selecionado: LightGBM** — maior AUC + maior Recall.

**Melhores hiperparâmetros:**
```python
LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=50,
    max_depth=-1, subsample=1.0, colsample_bytree=0.8,
    class_weight='balanced', random_state=42
)
```

### 4.3 Retreino e Validação com 2026 (Fase 3)

Retreinado nas amostras completas (2015–2025) → `modelos/municipio_full.pkl`

**Validação com dados reais nunca vistos (Jan–Jun 2026):**

| Métrica | Valor |
|---|---|
| AUC-ROC | **0,816** |
| PR-AUC | (ver resultado de `validacao_municipio_2026.csv`) |
| Recall (limiar 0,5) | 47,4% |
| Recall (limiar 0,3) | 73,8% |
| Precisão (limiar 0,3) | 12,7% |

O recall cai em Jan–Jun porque é a estação chuvosa — focos são raros (4,9% dos dias vs 14% no treino). O **AUC de 0,816** confirma que a ordenação dos municípios permanece correta mesmo na chuva.

**Análise de comparação previsão × realidade (limiar 0.3):**

Um município é considerado "alertado" se tiver ao menos um dia com probabilidade ≥ 0,3 no período.

| Métrica | Valor |
|---|---|
| Municípios alertados | 172 de 244 (70%) |
| Recall | **79,2%** — 164 de 207 municípios com fogo foram alertados |
| Precisão | **95,3%** — 164 de 172 alertas eram municípios que realmente queimaram |
| Falsos negativos | 43 municípios com fogo não alertados |
| Falsos positivos | 8 municípios alertados sem fogo |

**Curva de captura** — priorizando pares (município, dia) com maior probabilidade:
- Top 10% dos pares monitorados → captura **43,6%** dos fogos reais
- Top 20% dos pares monitorados → captura **62,7%** dos fogos reais
- 4,4× melhor que seleção aleatória no top 10%

**Comparação com baselines de climatologia** (`fase3c_baseline.py`):

| Ranqueador | AUC | Captura top 10% | Captura top 20% |
|---|---|---|---|
| `media_focos_mes_hist` | ~0,73 | — | — |
| `Municipio_Freq` | **~0,749** | — | — |
| `media + DiaSemChuva` | ~0,75 | — | — |
| **Modelo LightGBM** | **0,816** | **43,6%** | **62,7%** |

Delta sobre o melhor baseline: **+0,067** — melhoria significativa (threshold mínimo adotado: 0,02). O `Municipio_Freq` é o baseline mais forte; a frequência histórica já carrega muito sinal, mas o modelo adiciona sazonalidade climática que os baselines ignoram.

### 4.4 Sistema de Previsão 5 Dias (Fase 4)

Para cada previsão diária, uma **única chamada à API** por município retorna 30 dias de histórico + 6 dias de previsão:

1. **Open-Meteo Forecast** (`past_days=30, forecast_days=6`) → 36 valores de precipitação por município
2. Dias 1–30: calculam `DiaSemChuva` acumulado atual
3. Dias 32–36 (amanhã → hoje+5): projetam `DiaSemChuva` para cada dia futuro
4. `municipio_full.pkl.predict_proba()` → probabilidade por (município, dia)
5. Ranking final dos 244 municípios para os próximos 5 dias

**Resultado (04–08/06/2026):**
- NIQUELÂNDIA lidera todos os dias (95–98%) — 12–14 dias sem chuva
- 24–38 municípios em ALTO risco por dia

---

## 5. Estágio 2 — Modelo de Grade Espacial

### 5.1 Motivação

O modelo de município diz *qual município* está em risco, mas não *onde dentro do município*. Um município goiano pode ter 3.000–15.000 km² — informação insuficiente para operações de campo.

A grade espacial divide Goiás em células de **0,1° × 0,1° (~11km × 11km)**, com 2.976 células que tiveram ao menos 5 focos históricos.

### 5.2 Construção do Dataset de Grade (Fase 1 Grade)

**Diferença crítica em relação ao dataset de município:**  
Na primeira tentativa usamos `NEG_RATIO=4`, inflando a taxa de positivos para 20%. Na validação com 2026 (0,5% de positivos na chuva), a calibração ficou errada → AUC 0,69.

**Solução: negativos naturais completos** com clima exato via grade 0,5°:

| Classe | Registros | % |
|---|---|---|
| fogo = 1 | 361.516 | 2,0% |
| fogo = 0 | 18.032.888 | 98,0% |
| **Total** | **18.394.404** | 100% |

**Clima:** grade intermediária de 0,5° (148 pontos únicos cobrindo Goiás). Cada célula 0,1° usa o ponto 0,5° mais próximo (erro máximo ~35km vs ~80km do proxy de município anterior).

**Features do modelo de grade:**

| Feature | Tipo | Diferença do município |
|---|---|---|
| `Mes`, `DiaSemana`, `Estacao_Seca` | Temporal | Igual |
| `Cell_Lat`, `Cell_Lon` | Geográfica | Centro da célula 0,1° |
| `Cell_Freq` | Histórica | Focos da célula / total Goiás |
| `DiaSemChuva`, `Precipitacao` | Climática | Open-Meteo no ponto 0,5° mais próximo |
| `media_focos_mes_hist` | Histórica | Média de focos por célula+mês |

### 5.3 Modelagem da Grade (Fase 2 Grade)

| Modelo | AUC Val | AUC Teste | Recall Teste |
|---|---|---|---|
| XGBoost (GPU) | 0,8086 | **0,8314** | 0,770 |
| **LightGBM** | 0,8045 | 0,8285 | **0,761** |
| Random Forest | 0,8051 | 0,8288 | 0,753 |

LightGBM selecionado por recall superior e não requerer CUDA em produção.

**Melhores hiperparâmetros LightGBM grade:**
```python
LGBMClassifier(
    n_estimators=200, learning_rate=0.01, num_leaves=31,
    max_depth=10, subsample=0.8, colsample_bytree=1.0,
    class_weight='balanced', random_state=42
)
```

### 5.4 Retreino e Validação com 2026 (Fase 3 Grade)

Retreinado em 18.394.404 amostras (2015–2025) → `modelos/grade_full.pkl`

| Métrica | Município | Grade (clima 0,5°) |
|---|---|---|
| AUC-ROC 2026 | **0,816** | 0,710 |
| Recall (limiar 0,5) | **47,4%** | 41,6% |
| Recall (limiar 0,3) | **73,8%** | 66,6% |

O gap em relação ao município é estrutural: células 0,1° têm menos histórico individual e o problema é intrinsecamente mais difícil (11km vs município inteiro).

**Análise de comparação previsão × realidade (limiar 0.6):**

Uma célula é "alertada" se tiver ao menos um dia com probabilidade ≥ 0,6 no período.

| Métrica | Valor |
|---|---|
| Células alertadas | 1.825 de 2.976 (61%) |
| Recall | **70,3%** — 780 de 1.109 células com fogo foram alertadas |
| Precisão | **42,7%** — 780 de 1.825 alertas eram células que realmente queimaram |
| Falsos negativos | 329 células com fogo não alertadas |
| Falsos positivos | 1.045 células alertadas sem fogo confirmado |

**Curva de captura** — priorizando pares (célula, dia) com maior probabilidade:
- Top 10% → captura **30,3%** dos fogos reais (3× melhor que aleatório)
- Top 20% → captura **48,5%** dos fogos reais

**Nota sobre limiares:** o limiar não faz parte do modelo — é aplicado após a previsão e pode ser ajustado sem retreinar.

### 5.5 Clima por Grade 0.5°

Em vez de baixar clima para as 2.976 células (2h+, rate limit inviável), usamos uma grade climática intermediária de 148 pontos únicos que cobrem todo Goiás:

| Estratégia | Chamadas API | Erro máximo | Tempo |
|---|---|---|---|
| Município proxy | 244 | ~80km | 6 min |
| **Grade 0.5° (adotada)** | **148** | **~35km** | **~10 min** |
| Por célula 0.1° | 2.976 | ~0km | 2h+ (rate limit) |

Cada célula 0.1° usa o ponto 0.5° mais próximo. O dataset final ficou com 18.394.404 linhas (maior que a versão com proxy) porque o clima exato permitiu mapear mais células sem valores nulos.

### 5.6 Previsão 5 Dias — Grade

Em produção, o estágio 2 reutiliza os dados climáticos já coletados para o estágio 1 — cada célula 0,1° herda o `DiaSemChuva` e `Precipitacao` do município proxy mais próximo. Custo adicional em produção: **zero chamadas extras à API**.

---

## 6. Frontend — Mapa Interativo

O frontend é uma aplicação estática hospedada no **Vercel**, disponível em **https://vig-ia.vercel.app/**.

**Tecnologias:** Leaflet.js + GeoJSON + Esri World Imagery (satélite, sem API key)

**Arquivos:** `index.html`, `styles.css`, `app.js`, `data.js`, `forecast.json`

**Funcionamento:**
- `forecast.json` é gerado diariamente pelo GitHub Actions e commitado no repositório
- O Vercel redeploia automaticamente a cada novo commit, mantendo o site sempre atualizado
- `app.js` lê `forecast.json` e renderiza o mapa com Leaflet

**Funcionalidades:**
- Mapa coroplético dos 244 municípios colorido por **risco relativo** (percentil dentro do dia)
- Cores: **ALTO risco** = top 10% (percentil ≥ 90, vermelho) · **Atenção** = top 10–30% (percentil 70–90, âmbar) · **Calmo** = demais 70% (verde)
- Barra de navegação para selecionar entre os próximos 5 dias de previsão
- Ranking lateral com municípios ordenados por risco relativo decrescente — exibe "top X%" (posição relativa entre os 244 municípios no dia)
- **Clique no município:** drill-down mostrando células 0,1° internas coloridas por risco relativo
- Hover com tooltip: posição relativa ("top X%"), categoria de risco, dias sem chuva
- **Filtro de zonas:** slider "mostrar apenas top X%" — oculta células abaixo do threshold escolhido

**Por que exibir percentil em vez de probabilidade bruta:**

O `class_weight="balanced"` inflata as probabilidades (Brier score piora em relação ao baseline ingênuo). O modelo é excelente em **ordenar** municípios por risco (AUC 0,816), mas os valores absolutos de probabilidade não devem ser lidos como frequência real de ocorrência. Exibir a posição relativa ("top 5%") é honesto: informa onde concentrar atenção sem fazer promessas sobre calibração.

---

## 7. Abordagem: Ranking por Risco Relativo

Em vez de um limiar binário fixo ("vai ter fogo? sim/não"), o sistema usa os modelos como **ferramentas de ranqueamento**:

- **Estágio 1:** ordena os 244 municípios por probabilidade → o percentil diário determina a posição relativa exibida ("top 5%", "top 30%", etc.)
- **Estágio 2:** ordena as células dentro de cada município → identifica subáreas críticas dentro do município selecionado

**Vantagem do ranking:** o AUC garante que a ordenação está correta independentemente da calibração. AUC = 0,816 significa que em 81,6% das comparações (município com fogo vs sem fogo), o modelo classifica corretamente qual tem maior risco. A curva de captura confirma: os top 10% do ranking concentram 43,6% dos fogos reais.

**Percentil diário:** calculado como `rank(pct=True)` dentro de cada dia — cada município recebe sua posição relativa entre os 244 municípios de Goiás naquele dia. Isso torna a visualização estável entre estações: no período seco todos os valores absolutos sobem, mas a distribuição relativa permanece interpretável.

---

## 8. Pipeline de Produção Leve

### 8.1 Problema de dependência de dados

Os scripts de treinamento dependem de dois datasets pesados:

| Arquivo | Tamanho |
|---|---|
| `dataset_municipio.csv` | 1,6 GB |
| `dataset_grade.csv` | 1,1 GB |

Distribuir esses arquivos no repositório ou baixá-los a cada execução no CI é inviável.

### 8.2 Solução: Lookup Tables

O script `gerar_lookups.py` extrai de cada dataset apenas a coluna que não pode ser calculada em tempo real: a média histórica de focos por (unidade geográfica, mês). Resultado:

| Arquivo | Tamanho | Linhas |
|---|---|---|
| `dados/lookup_municipio.csv` | 85 KB | 2.964 |
| `dados/lookup_grade.csv` | 937 KB | 29.375 |

Todas as outras features são calculadas em tempo real a partir das chamadas à API climática.

### 8.3 Script de Previsão Leve (`previsao_leve.py`)

Pipeline completo E1 + E2 sem dependência dos datasets de treino:

```
Entrada:
  modelos/municipio_full.pkl  (1,1 MB)
  modelos/grade_full.pkl      (314 KB)
  dados/mapeamento_municipio.csv + mapeamento_grade.csv
  dados/lookup_municipio.csv  + lookup_grade.csv
  Open-Meteo API (244 chamadas paralelas, ~6 min)

Saída:
  resultados/previsao_municipio_<hoje>.csv  (inclui prob_fogo, percentil, risco)
  resultados/previsao_grade_<hoje>.csv      (inclui prob_fogo, percentil, risco)
```

O percentil é calculado via `rank(pct=True)` dentro de cada data: E1 rankeia os 244 municípios, E2 rankeia as 2.976 células. A coluna `risco` é derivada do percentil (ALTO ≥ 0.90, MÉDIO ≥ 0.70, BAIXO < 0.70).

**Estratégia de chamadas à API:**
- `ThreadPoolExecutor(max_workers=8)` — 244 municípios em paralelo
- Retry automático com backoff exponencial por chamada (4 tentativas)
- Retry sequencial global (3 rounds, 30s entre rounds) para municípios que falharam na fase paralela
- `sys.exit(1)` se algum município permanecer sem dados após todas as tentativas — impede commit de previsão incompleta

### 8.4 GitHub Actions — Automação Diária

O arquivo `.github/workflows/previsao_diaria.yml` automatiza o ciclo completo:

```
Cron: 0 6 * * *  →  03h00 Brasília (UTC-3)
  1. actions/checkout@v4
  2. pip install -r pbl/requirements.txt
  3. python pbl/previsao_leve.py          → previsao_municipio_*.csv + previsao_grade_*.csv
  4. python pbl/exportar_json.py          → pbl/forecast.json
  5. cp pbl/forecast.json frontend/       → atualiza frontend
  6. git commit + push                    → aciona redeploy automático no Vercel
```

O pipeline pode ser disparado manualmente pela aba Actions do GitHub (`workflow_dispatch`).

### 8.5 Deploy Contínuo — Vercel

- Repositório conectado ao Vercel via GitHub App
- Root Directory: `frontend/`
- Sem build command (arquivos estáticos)
- Cada `git push` com novo `forecast.json` aciona redeploy automático em ~30s

---

## 9. Arquitetura Completa de Produção

```
GitHub Actions (cron, 03h Brasília)
    │
    ├─► Open-Meteo Forecast (past_days=30, 244 municípios, ~6 min)
    │       ↓ DiaSemChuva projetado para os próximos 5 dias
    │
    ├─► modelos/municipio_full.pkl  →  P(fogo) × 244 municípios × 5 dias
    │
    ├─► modelos/grade_full.pkl  →  P(fogo) × 2.976 células × 5 dias
    │   (usa clima dos municípios proxy — 0 chamadas extras à API)
    │
    ├─► exportar_json.py  →  forecast.json
    │
    └─► git commit + push
            │
            └─► Vercel redeploy automático
                    │
                    └─► https://vig-ia.vercel.app/  (atualizado em ~30s)
```

---

## 10. Arquivos do Projeto

### Scripts de Desenvolvimento — `estagio1_municipio/`
| Arquivo | Descrição |
|---|---|
| `fase1_dataset.py` | Constrói dataset_municipio.csv |
| `fase1b_clima.py` | Baixa Open-Meteo histórico (244 municípios, escrita progressiva) |
| `fase1d_clima_2026.py` | Baixa Open-Meteo histórico de Jan–Jun 2026 para validação |
| `fase2_modelagem.py` | Treina RF, XGBoost, LightGBM com busca de hiperparâmetros |
| `fase3_validacao_2026.py` | Retreino completo + validação 2026 (AUC, PR-AUC, Brier, captura top N%) |
| `fase3c_baseline.py` | Compara modelo LightGBM vs baselines de climatologia (AUC, captura top N%) |
| `fase4_previsao_5dias.py` | Previsão 5 dias por município (desenvolvimento) |

### Scripts de Desenvolvimento — `estagio2_grade/`
| Arquivo | Descrição |
|---|---|
| `fase1_dataset_grade.py` | Constrói dataset_grade.csv (18.4M linhas) |
| `fase1b_clima_05graus.py` | Baixa clima 0.5° para 148 pontos (~10 min) |
| `fase1c_aplicar_clima.py` | Substitui clima proxy pelo clima 0.5° no dataset |
| `fase2_modelagem.py` | Treina modelos na grade |
| `fase3_validacao_2026.py` | Retreino completo + validação 2026 grade (AUC, PR-AUC, Brier, captura top N%) |
| `fase3b_graficos_comparacao.py` | Gráficos previsão × realidade (limiar 0.6) |
| `fase4_previsao_offline.py` | Previsão 5 dias sem API (desenvolvimento/demo) |

### Scripts de Produção — `pbl/`
| Arquivo | Descrição |
|---|---|
| `gerar_lookups.py` | Extrai lookup tables dos datasets de treino (execução única) |
| `previsao_leve.py` | Pipeline E1+E2 completo sem datasets pesados (~6 min) |
| `exportar_json.py` | Converte CSVs de previsão em forecast.json para o frontend |
| `requirements.txt` | Dependências do pipeline de produção |

### Modelos — `modelos/`
| Arquivo | Descrição |
|---|---|
| `municipio_full.pkl` | LightGBM 2015–2025, AUC 0.816 (**produção estágio 1**) |
| `grade_full.pkl` | LightGBM grade 2015–2025, AUC 0.710 (**produção estágio 2**) |

### Dados — `dados/`
| Arquivo | Descrição |
|---|---|
| `mapeamento_municipio.csv` | 244 municípios — freq + centroide lat/lon |
| `mapeamento_grade.csv` | 2.976 células — freq + município proxy |
| `lookup_municipio.csv` | Média histórica de focos por (município, mês) — 85 KB |
| `lookup_grade.csv` | Média histórica de focos por (célula, mês) — 937 KB |

### Frontend — `frontend/`
| Arquivo | Descrição |
|---|---|
| `index.html` | Página principal |
| `app.js` | Lógica do mapa (Leaflet) |
| `data.js` | Dados auxiliares (GeoJSON municípios) |
| `styles.css` | Estilos |
| `forecast.json` | Previsão atual — gerado e atualizado diariamente pelo CI |

### CI/CD — `.github/workflows/`
| Arquivo | Descrição |
|---|---|
| `previsao_diaria.yml` | Cron 03h Brasília — pipeline + commit + redeploy Vercel |

### Resultados — `resultados/`
| Arquivo | Descrição |
|---|---|
| `resultados_municipio_fase2.csv` | Métricas dos 3 modelos (município) |
| `resultados_grade_fase2.csv` | Métricas dos 3 modelos (grade) |
| `validacao_municipio_2026.csv` | Métricas de validação 2026 (município) |
| `validacao_grade_2026.csv` | Métricas de validação 2026 (grade) |
| `previsao_municipio_<data>.csv` | Ranking 5 dias por município |
| `previsao_grade_<data>.csv` | Ranking 5 dias por célula |

---

## 11. Decisões Técnicas Relevantes

| Decisão | Motivo |
|---|---|
| Classificação em vez de regressão | FRP já é medido — regressão não agrega valor em produção |
| Ausência = negativo (BDqueimadas) | Satélite cobre Goiás continuamente |
| Split temporal | Random split vaza dados futuros no treino |
| Subsample 200k para busca | RandomizedSearchCV inviável em 720k+ amostras |
| `n_jobs=1` no estimador interno | Nested parallelism travou processo por 30+ minutos |
| Negativos naturais completos (grade) | NEG_RATIO=4 inflou positivos para 20%, causou AUC 0.69 |
| Grade climática 0.5° | 148 pontos vs 2.976: sem rate limit, erro máx ~35km |
| Limiar E1 = 0.3 | Recall 79.2%, Precisão 95.3% — equilíbrio para alerta regional |
| Limiar E2 = 0.6 | Recall 70.3%, Precisão 42.7% — equilíbrio para drill-down geográfico |
| Limiar ≠ retreino | O limiar é aplicado pós-previsão; mudar limiar não requer retreinar |
| Dois estágios complementares | Município: triagem confiável; grade: localização operacional |
| Lookup tables (85KB + 937KB) | Substituem datasets de treino de 1.7GB no pipeline de produção |
| `past_days=30` (chamada única) | Substitui Archive + Forecast separados; mesmo janela, metade das chamadas |
| ThreadPoolExecutor (max_workers=8) | 244 chamadas paralelas vs sequencial: ~6 min vs ~150 min |
| sys.exit(1) em erro fatal | Impede commit de previsão incompleta no GitHub Actions |
| Percentil diário em vez de probabilidade bruta | `class_weight="balanced"` inflata probabilidades (Brier pior que baseline ingênuo); o modelo ranqueia bem (AUC 0,816), então exibir a posição relativa ("top X%") é mais honesto que a probabilidade absoluta |
| LightGBM para produção E2 (vs Random Forest que ganhou busca) | Diferença de AUC dentro do ruído (~0,005); LightGBM tem inferência mais rápida, suporte nativo a NaN e consistência com E1 |
| Escrita progressiva em `fase1b_clima.py` | Batch write perdia todo o progresso em Ctrl+C; escrita por município garante retomada incremental |

---

## 12. Resultados Consolidados

| | Estágio 1 — Município | Estágio 2 — Grade 0.1° |
|---|---|---|
| Modelo | LightGBM | LightGBM |
| Dataset treino | ~980k amostras | 18.4M amostras |
| AUC Teste (2024-25) | **0.835** | 0.831 |
| AUC Validação 2026 | **0.816** | 0.710 |
| Limiar operacional | 0.3 | 0.6 |
| Recall (limiar) | **79.2%** | 70.3% |
| Precisão (limiar) | **95.3%** | 42.7% |
| Top 10% captura | **43.6%** fogos | 30.3% fogos |
| Top 20% captura | **62.7%** fogos | 48.5% fogos |

---

## 13. Conclusões

O sistema vigIA combina dois modelos LightGBM em estágios complementares:

**Estágio 1 (município, AUC 0,816):** com limiar 0,3, alerta 172 de 244 municípios e captura 79,2% dos fogos reais com precisão de 95,3%. Identifica quais municípios priorizar nos próximos 5 dias antes da detecção satelital.

**Estágio 2 (grade 0,1°, AUC 0,710):** com limiar 0,6, alerta 61% do território e captura 70,3% dos fogos com precisão de 42,7%. Detalha a localização dentro dos municípios de alto risco com resolução de ~11km × 11km.

O sistema opera com dados inteiramente abertos (BDqueimadas/INPE + Open-Meteo), sem custo de infraestrutura de dados. O pipeline de produção utiliza apenas ~1MB de lookup tables (vs 1.7GB de dados de treino), executa em ~6 minutos via GitHub Actions e publica automaticamente no Vercel. O site vigIA (https://vig-ia.vercel.app/) é atualizado todo dia às 03h Brasília sem intervenção humana.
