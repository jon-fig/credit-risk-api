"""Modelos Pydantic para validação dos dados de entrada e saída da API.

Os nomes dos campos expostos na API são exatamente as colunas usadas no
treinamento (dataset "Give Me Some Credit"), garantindo que o Pipeline
receba o DataFrame na ordem esperada pelo imputer.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# Ordem esperada pelo Pipeline (models/credit_model.pkl -> imputer.feature_names_in_)
FEATURES: list[str] = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


class CreditApplication(BaseModel):
    """Dados de um proponente a crédito.

    ``MonthlyIncome`` e ``NumberOfDependents`` são opcionais porque a ausência
    desses dados é comum; quando omitidos, o Pipeline aplica a imputação por
    mediana aprendida no treino.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )

    revolving_utilization: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines",
        ge=0,
        description="Uso do limite de crédito rotativo (0 a 1+; pode exceder 1).",
    )
    age: int = Field(alias="age", ge=18, le=110, description="Idade do proponente em anos.")
    times_30_59_past_due: int = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        ge=0,
        description="Quantidade de atrasos entre 30 e 59 dias.",
    )
    debt_ratio: float = Field(
        alias="DebtRatio",
        ge=0,
        description="Dívida mensal (aluguel, crédito, etc.) dividida pela renda mensal.",
    )
    monthly_income: float | None = Field(
        default=None,
        alias="MonthlyIncome",
        ge=0,
        description="Renda mensal. Opcional: se omitida, é imputada pela mediana do treino.",
    )
    open_credit_lines: int = Field(
        alias="NumberOfOpenCreditLinesAndLoans",
        ge=0,
        description="Quantidade de linhas de crédito e empréstimos abertos.",
    )
    times_90_late: int = Field(
        alias="NumberOfTimes90DaysLate",
        ge=0,
        description="Quantidade de atrasos de 90 dias ou mais.",
    )
    real_estate_loans: int = Field(
        alias="NumberRealEstateLoansOrLines",
        ge=0,
        description="Quantidade de empréstimos ou linhas de crédito imobiliário.",
    )
    times_60_89_past_due: int = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        ge=0,
        description="Quantidade de atrasos entre 60 e 89 dias.",
    )
    dependents: int | None = Field(
        default=None,
        alias="NumberOfDependents",
        ge=0,
        description="Número de dependentes. Opcional: se omitido, é imputado pela mediana do treino.",
    )


class PredictionResponse(BaseModel):
    """Resultado da inferência registrado no banco de dados."""

    id: int = Field(description="ID do registro na tabela predictions.")
    created_at: datetime = Field(description="Timestamp UTC do registro.")
    probability_default: float = Field(
        ge=0, le=1, description="Probabilidade estimada de inadimplência."
    )
    prediction: int = Field(description="1 = inadimplente, 0 = adimplente.")
    classification: str = Field(description="Classificação legível da predição.")
    threshold: float = Field(description="Limiar de decisão aplicado à probabilidade.")
    model_name: str = Field(description="Modelo utilizado na inferência.")
