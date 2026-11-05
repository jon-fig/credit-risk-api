"""Persistência do histórico de predições em SQLite.

O banco fica em ``data/predictions.db`` (ignorado pelo git). Cada requisição de
predição gera uma linha com o timestamp, os dados de entrada recebidos e o
resultado da inferência.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "predictions.db"
DB_PATH = Path(os.getenv("PREDICTIONS_DB_PATH", DEFAULT_DB_PATH))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    input_data TEXT NOT NULL,
    probability_default REAL NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    threshold REAL NOT NULL,
    model_name TEXT NOT NULL
);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Abre uma conexão por operação (seguro para uso com múltiplas threads)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Cria a tabela de histórico caso ainda não exista."""
    with _connect() as connection:
        connection.execute(CREATE_TABLE_SQL)


def save_prediction(
    input_data: dict,
    probability_default: float,
    prediction: int,
    threshold: float,
    model_name: str,
    created_at: datetime,
) -> int:
    """Grava uma predição e devolve o ID gerado."""
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at, input_data, probability_default,
                prediction, threshold, model_name
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at.isoformat(),
                json.dumps(input_data, ensure_ascii=False),
                probability_default,
                prediction,
                threshold,
                model_name,
            ),
        )
        return int(cursor.lastrowid)


def fetch_recent_predictions(limit: int = 10) -> list[dict]:
    """Retorna as predições mais recentes, da mais nova para a mais antiga."""
    with _connect() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, created_at, probability_default, prediction, threshold, model_name
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def count_predictions() -> int:
    """Retorna quantos registros existem na tabela (útil para conferência)."""
    with _connect() as connection:
        row = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return int(row[0])
