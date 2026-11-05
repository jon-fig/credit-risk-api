# Credit Risk Prediction API

Serviço REST que expõe o modelo de Machine Learning `models/credit_model.pkl`
(pipeline XGBoost para risco de inadimplência, dataset *Give Me Some Credit*)
com validação de entrada via Pydantic, persistência das predições em SQLite e
documentação interativa automática.

## Estrutura

```
.
├── models/credit_model.pkl      # Pipeline treinado (cleaner → imputer → XGBoost)
├── data/predictions.db          # Banco SQLite de histórico (criado na 1ª execução)
├── src/
│   ├── main.py                  # Aplicação FastAPI (UI, /health, /predict, /predictions)
│   ├── static/index.html        # Interface web em português (Tailwind via CDN)
│   ├── schemas.py               # Modelos Pydantic de entrada e saída
│   ├── database.py              # Conexão, escrita e leitura do SQLite
│   ├── preprocessing.py         # CreditDataCleaner usado pelo Pipeline
│   └── pipeline.py              # Script de treinamento (gera o .pkl)
├── Dockerfile                   # Imagem de produção (python:3.13-slim)
├── .dockerignore                # Arquivos ignorados no build
├── requirements.txt
├── requirements-dev.txt         # Dependências de teste (pytest, httpx)
├── tests/                       # Testes da API (conftest.py + test_api.py)
└── README.md
```

## Requisitos

- Python 3.10+ (a imagem Docker usa 3.13)
- Dependências de execução em `requirements.txt`
- Dependências de teste em `requirements-dev.txt` (inclui as de execução)
- Docker, caso queira rodar via container

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> O modelo depende de `xgboost`, `pandas` e `numpy` além do stack do FastAPI,
> pois o `.pkl` é um `Pipeline` do scikit-learn com um `XGBClassifier`.

## Como rodar localmente

Na raiz do projeto:

```bash
uvicorn src.main:app --reload
```

A API sobe em `http://127.0.0.1:8000`.

- Interface web: <http://127.0.0.1:8000/>
- Documentação interativa (Swagger UI): <http://127.0.0.1:8000/docs>
- Documentação alternativa (ReDoc): <http://127.0.0.1:8000/redoc>
- Status do serviço: <http://127.0.0.1:8000/health>

## Endpoints

### `GET /`

Serve a interface web (`src/static/index.html`): formulário em português com os 10
campos de crédito, classificação (adimplente/inadimplente), barra com a
probabilidade exata, ID do registro salvo no SQLite, timestamp, limiar de
decisão, nome do modelo e a lista das últimas avaliações. Os arquivos estáticos
ficam disponíveis em `/static`.

### `GET /health`

Retorna o status da API e indica se o modelo já foi carregado.

```json
{
  "status": "ok",
  "service": "credit-risk-prediction",
  "model_loaded": true,
  "model": "credit_model.pkl",
  "decision_threshold": 0.5,
  "docs": "/docs"
}
```

### `POST /predict`

Recebe os dados do proponente, executa a inferência, grava o registro no SQLite
e devolve a probabilidade de inadimplência.

Exemplo com `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.77,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.8,
    "MonthlyIncome": 6800.0,
    "NumberOfOpenCreditLinesAndLoans": 9,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
  }'
```

Resposta:

```json
{
  "id": 1,
  "created_at": "2026-11-02T14:03:11.512345Z",
  "probability_default": 0.037642,
  "prediction": 0,
  "classification": "adimplente",
  "threshold": 0.5,
  "model_name": "credit_model.pkl"
}
```

Campos opcionais: `MonthlyIncome` e `NumberOfDependents` podem ser omitidos (ou
`null`); nesse caso o pipeline aplica a imputação por mediana aprendida no
treino.

### `GET /predictions`

Lista as predições mais recentes registradas no banco, da mais nova para a mais
antiga. O parâmetro `limit` (1–100, padrão 10) controla quantos registros são
retornados.

```bash
curl "http://127.0.0.1:8000/predictions?limit=5"
```

```json
[
  {
    "id": 7,
    "created_at": "2026-11-02T03:16:33.309760Z",
    "probability_default": 0.532153,
    "prediction": 1,
    "classification": "inadimplente",
    "threshold": 0.5,
    "model_name": "credit_model.pkl"
  }
]
```

## Validação de entrada

O `schemas.py` valida tipos e faixas. Exemplos de requisições recusadas com
HTTP 422:

- `age` fora de 18–110;
- contadores de atraso, linhas de crédito ou dependentes negativos;
- `DebtRatio`, `MonthlyIncome` ou `RevolvingUtilizationOfUnsecuredLines` negativos;
- campo obrigatório ausente.

## Docker

A imagem usa `python:3.13-slim`, instala `libgomp1` (runtime OpenMP exigido pelo
XGBoost), copia `src/` (com a interface web), `models/` e roda como usuário
não-root. O SQLite é gravado em `/app/data/predictions.db` e o `HEALTHCHECK`
consulta `/health`.

### 1. Build

```bash
docker build -t credit-risk-api .
```

### 2. Run

```bash
docker run -d -p 8000:8000 --name credit-risk-service credit-risk-api
```

### 3. Verificar

```bash
curl http://localhost:8000/health
docker ps --filter name=credit-risk-service
```

Depois abra <http://localhost:8000/> para usar a interface web, ou
<http://localhost:8000/docs> para o Swagger.

Acompanhar logs e encerrar:

```bash
docker logs -f credit-risk-service
docker stop credit-risk-service
docker rm credit-risk-service
```

### Persistir o histórico

Por padrão o banco vive dentro do container e some ao removê-lo. Para manter o
histórico entre execuções, use um volume nomeado em `/app/data`:

```bash
docker volume create credit-risk-data
docker run -d -p 8000:8000 --name credit-risk-service \
  -v credit-risk-data:/app/data credit-risk-api
```

Para usar outro caminho, defina `PREDICTIONS_DB_PATH` (ex.:
`-e PREDICTIONS_DB_PATH=/app/data/outro.db`).

### O que não entra na imagem

`.git`, `.vscode`, `.venv`, `tests/`, caches (`__pycache__`, `.pytest_cache`),
bancos SQLite locais, `reports/`, arquivos de agentes e lixo de sistema — tudo
listado no `.dockerignore`.

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Os testes usam `TestClient` e apontam o SQLite para um diretório temporário via
`PREDICTIONS_DB_PATH`, então não tocam em `data/predictions.db`. Cobrem a página
inicial, `/health`, `/predict` (sucesso, campos opcionais, erros de validação),
a persistência no banco e `/predictions`.

## Banco de dados

Cada predição é registrada em `data/predictions.db`, tabela `predictions`:

| Coluna | Conteúdo |
| --- | --- |
| `id` | chave primária (ID da requisição) |
| `created_at` | timestamp UTC do registro |
| `input_data` | dados de entrada em JSON |
| `probability_default` | probabilidade estimada de inadimplência |
| `prediction` | classe prevista (1 = inadimplente, 0 = adimplente) |
| `threshold` | limiar de decisão aplicado |
| `model_name` | modelo utilizado |

Para inspecionar o histórico:

```bash
python - <<'EOF'
import sqlite3
with sqlite3.connect("data/predictions.db") as conn:
    for row in conn.execute("SELECT id, created_at, prediction, probability_default FROM predictions ORDER BY id DESC LIMIT 10"):
        print(row)
EOF
```

O caminho do banco pode ser sobrescrito com a variável de ambiente
`PREDICTIONS_DB_PATH`. A interface web e o endpoint `GET /predictions` leem
desta mesma tabela para exibir as últimas avaliações.

## Regerar o modelo (opcional)

O script de treinamento lê `data/cs-training.csv` e reescreve
`models/credit_model.pkl` e os relatórios em `reports/`:

```bash
python -m src.pipeline
```


### O artefato de Machine Learning e o pipeline de treinamento para esta API foram construídos e serializados no repositório [credit-risk-model](https://github.com/jon-fig/credit-risk-model).