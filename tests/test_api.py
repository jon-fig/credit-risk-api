"""Testes dos endpoints da API de risco de crédito."""

from __future__ import annotations

import sqlite3

BASE_PAYLOAD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.77,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.8,
    "MonthlyIncome": 6800.0,
    "NumberOfOpenCreditLinesAndLoans": 9,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}


def test_index_serve_interface_em_portugues(client):
    resposta = client.get("/")

    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/html")
    assert 'lang="pt-BR"' in resposta.text
    assert "Avaliação de Risco de Crédito" in resposta.text
    assert "Uso do Limite de Crédito Rotativo" in resposta.text


def test_health_indica_modelo_carregado(client):
    resposta = client.get("/health")

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["model_loaded"] is True
    assert corpo["decision_threshold"] == 0.5


def test_predict_retorna_probabilidade_e_classificacao(client):
    resposta = client.post("/predict", json=BASE_PAYLOAD)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert isinstance(corpo["id"], int) and corpo["id"] > 0
    assert 0.0 <= corpo["probability_default"] <= 1.0
    assert corpo["prediction"] in (0, 1)
    assert corpo["classification"] == ("inadimplente" if corpo["prediction"] == 1 else "adimplente")
    assert corpo["threshold"] == 0.5
    assert corpo["model_name"] == "credit_model.pkl"
    assert corpo["created_at"]


def test_predict_aceita_campos_opcionais_ausentes(client):
    payload = {k: v for k, v in BASE_PAYLOAD.items() if k not in {"MonthlyIncome", "NumberOfDependents"}}

    resposta = client.post("/predict", json=payload)

    assert resposta.status_code == 200
    assert 0.0 <= resposta.json()["probability_default"] <= 1.0


def test_predict_rejeita_idade_fora_da_faixa(client):
    resposta = client.post("/predict", json={**BASE_PAYLOAD, "age": 10})

    assert resposta.status_code == 422
    erros = resposta.json()["detail"]
    assert erros[0]["loc"] == ["body", "age"]
    assert erros[0]["type"] == "greater_than_equal"


def test_predict_rejeita_campo_obrigatorio_ausente(client):
    payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "DebtRatio"}

    resposta = client.post("/predict", json=payload)

    assert resposta.status_code == 422
    assert {erro["type"] for erro in resposta.json()["detail"]} == {"missing"}


def test_predict_rejeita_valor_negativo(client):
    resposta = client.post("/predict", json={**BASE_PAYLOAD, "NumberOfTimes90DaysLate": -1})

    assert resposta.status_code == 422
    assert resposta.json()["detail"][0]["loc"] == ["body", "NumberOfTimes90DaysLate"]


def test_predicao_e_persistida_no_banco(client, db_path):
    resposta = client.post("/predict", json=BASE_PAYLOAD)
    corpo = resposta.json()

    with sqlite3.connect(db_path) as conexao:
        linha = conexao.execute(
            "SELECT probability_default, prediction, threshold, model_name FROM predictions WHERE id = ?",
            (corpo["id"],),
        ).fetchone()

    assert linha is not None
    assert round(linha[0], 6) == corpo["probability_default"]
    assert linha[1] == corpo["prediction"]
    assert linha[2] == 0.5
    assert linha[3] == "credit_model.pkl"


def test_historico_retorna_predicoes_mais_recentes_primeiro(client):
    primeiro = client.post("/predict", json=BASE_PAYLOAD).json()
    segundo = client.post("/predict", json={**BASE_PAYLOAD, "age": 30}).json()

    resposta = client.get("/predictions", params={"limit": 2})

    assert resposta.status_code == 200
    registros = resposta.json()
    assert len(registros) == 2
    assert [registro["id"] for registro in registros] == [segundo["id"], primeiro["id"]]
    assert registros[0]["classification"] == (
        "inadimplente" if segundo["prediction"] == 1 else "adimplente"
    )
    assert all(0.0 <= registro["probability_default"] <= 1.0 for registro in registros)


def test_historico_valida_limite(client):
    assert client.get("/predictions", params={"limit": 0}).status_code == 422
    assert client.get("/predictions", params={"limit": 101}).status_code == 422
