"""Configuração compartilhada dos testes.

O banco SQLite é redirecionado para um diretório temporário antes de importar a
aplicação, para que os testes nunca escrevam em `data/predictions.db`.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEMP_DIR = Path(tempfile.mkdtemp(prefix="credit-api-tests-"))
os.environ["PREDICTIONS_DB_PATH"] = str(TEMP_DIR / "predictions.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.main import app  # noqa: E402


@pytest.fixture(scope="session")
def db_path() -> Path:
    """Caminho do banco temporário usado durante os testes."""
    return TEMP_DIR / "predictions.db"


@pytest.fixture(scope="session")
def client():
    """Cliente HTTP com o ciclo de vida da aplicação (carrega o modelo uma vez)."""
    with TestClient(app) as test_client:
        yield test_client
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
