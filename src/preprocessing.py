"""Transformações de limpeza reutilizáveis pelo pipeline e na inferência."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# 96/98 são códigos sentinela usados no dataset para "informação indisponível".
PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
SENTINEL_THRESHOLD = 90


class CreditDataCleaner(BaseEstimator, TransformerMixin):
    """Converte códigos sentinela e idades inválidas em valores ausentes.

    A limpeza é determinística e não usa informação da variável alvo, portanto
    não causa vazamento de dados. A imputação posterior é ajustada apenas no
    conjunto de treino, dentro do Pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "CreditDataCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditDataCleaner espera um pandas.DataFrame.")
        X = X.copy()
        for column in PAST_DUE_COLUMNS:
            X[column] = X[column].mask(X[column] >= SENTINEL_THRESHOLD, np.nan)
        X["age"] = X["age"].mask(X["age"] <= 0, np.nan)
        return X
