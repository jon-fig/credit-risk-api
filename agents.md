# Objetivo do Projeto
Transformar o modelo de Machine Learning (credit_model.pkl) em um serviço de produção funcional, permitindo receber requisições HTTP via REST, validar dados de entrada com Pydantic, registrar as predições em um banco de dados SQLite local e expor documentação interativa.

## Plano de Execução - FASE 1: Desenvolvimento e Teste Local (PARAR AQUI)
O agente deve executar os passos abaixo sequencialmente na pasta src/ e na raiz, parando a execução ao término para que o usuário possa testar localmente e commitar no GitHub.

## Passo 1: Criar o arquivo requirements.txt na raiz
Listar as dependências essenciais do projeto:

`fastapi` 

`uvicorn`

`pydantic`

`scikit-learn`

`joblib`

## Passo 2: Criar o arquivo src/schemas.py
Definir os modelos de validação de dados de entrada usando Pydantic. 
Nota: Ajuste os campos da classe conforme as features reais que o seu `credit_model.pkl` espera (ex: renda, score de crédito, valor do empréstimo).

## Passo 3: Criar o arquivo src/database.py
Configurar a conexão com o banco de dados SQLite (`data/predictions.db`) e criar uma tabela de histórico para registrar:

ID da requisição / Timestamp

Dados de entrada enviados pelo cliente

Resultado da predição (probabilidade de default / classe)

## Passo 4: Criar o arquivo src/main.py
O arquivo principal da aplicação FastAPI:

Carregar o modelo .pkl utilizando joblib na inicialização do servidor.

Criar a rota raiz (GET /) retornando o status da API.

Criar a rota de predição (POST /predict) que:

Recebe e valida os dados via Pydantic (schemas.py).

Executa a inferência com o modelo carregado.

Salva o log da predição no SQLite (database.py).

Retorna a resposta JSON com a probabilidade/resultado.

## Passo 5: Criar um README.md inicial
Escrever uma documentação clara explicando como rodar o projeto localmente com Uvicorn.

## 🛑 CHECKPOINT DE PARADA DA FASE 1
Assim que os passos 1 a 5 forem concluídos, o agente deve parar a execução.
O usuário irá testar a API localmente no terminal do Dev Container.

## FASE 2: Containerização (Executada após o feedback do usuário)
Somente após a aprovação e testes locais bem-sucedidos, criaremos:

`Dockerfile` na raiz do projeto para o empacotamento final da aplicação.