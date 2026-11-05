"""Pipeline de classificação binária para risco de inadimplência de crédito.

Dataset: "Give Me Some Credit" (data/cs-training.csv).
Execução: python -m src.pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (compatível com CLI/CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import PAST_DUE_COLUMNS, SENTINEL_THRESHOLD, CreditDataCleaner

RANDOM_STATE = 42
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.5

TARGET = "SeriousDlqin2yrs"
ID_COLUMN = "Unnamed: 0"

FEATURES = [
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cs-training.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def build_model(estimator: BaseEstimator) -> Pipeline:
    """Monta o Pipeline: limpeza -> imputação por mediana -> classificador."""
    return Pipeline(
        steps=[
            ("cleaner", CreditDataCleaner()),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega o CSV, remove a coluna de índice e duplicatas exatas."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")

    df = pd.read_csv(path)
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    missing = {TARGET, *FEATURES} - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    if df[TARGET].isna().any():
        raise ValueError("A variável alvo contém valores ausentes.")

    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df.attrs["duplicate_rows_removed"] = duplicates
    return df


def summarize_data(df: pd.DataFrame) -> dict:
    """Resume o dataset antes da modelagem (para auditoria em reports/)."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "features": len(FEATURES),
        "target": TARGET,
        "target_rate": round(float(df[TARGET].mean()), 6),
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "missing_values": {c: int(n) for c, n in df[FEATURES].isna().sum().items()},
        "missing_percentage": {
            c: round(float(p), 4) for c, p in (df[FEATURES].isna().mean() * 100).items()
        },
        "sentinel_values": {
            c: int((df[c] >= SENTINEL_THRESHOLD).sum()) for c in PAST_DUE_COLUMNS
        },
        "invalid_age_rows": int((df["age"] <= 0).sum()),
        "duplicate_rows_removed": int(df.attrs.get("duplicate_rows_removed", 0)),
    }


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa X/y e faz a divisão estratificada treino/teste."""
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula ROC-AUC, precisão, revocação e métricas auxiliares."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }


def plot_confusion_matrix(model_name: str, cm: list[list[int]], path: Path) -> None:
    """Salva a matriz de confusão como heatmap."""
    cm_array = np.asarray(cm)
    labels = [f"TN\n{cm_array[0, 0]}", f"FP\n{cm_array[0, 1]}", f"FN\n{cm_array[1, 0]}", f"TP\n{cm_array[1, 1]}"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_array,
        annot=np.array(labels).reshape(2, 2),
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Previsto: adimplente", "Previsto: inadimplente"],
        yticklabels=["Real: adimplente", "Real: inadimplente"],
        ax=ax,
    )
    ax.set_title(f"Matriz de confusão — {model_name} (threshold={DECISION_THRESHOLD})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    model_name: str, importances: np.ndarray, feature_names: list[str], path: Path
) -> pd.DataFrame:
    """Salva o ranking de importância das variáveis e o retorna como DataFrame."""
    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=ranking, x="importance", y="feature", color="#2a6f97", ax=ax)
    ax.set_title(f"Importância das variáveis — {model_name}")
    ax.set_xlabel("Importância relativa")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return ranking


def format_metrics(name: str, metrics: dict) -> str:
    return (
        f"{name}: ROC-AUC={metrics['roc_auc']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"PR-AUC={metrics['average_precision']:.4f}"
    )


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    print("[1/6] Carregando e auditando os dados...", flush=True)
    df = load_data()
    summary = summarize_data(df)
    (REPORTS_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"      {summary['rows']} linhas | {summary['target_rate']:.2%} de inadimplentes | "
        f"MonthlyIncome ausente em {summary['missing_percentage']['MonthlyIncome']:.2f}%",
        flush=True,
    )

    print("[2/6] Separando X/y e dividindo treino/teste (random_state=42)...", flush=True)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      Treino: {len(X_train)} | Teste: {len(X_test)}", flush=True)

    print("[3/6] Treinando RandomForestClassifier (baseline)...", flush=True)
    random_forest = build_model(
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    rf_started = time.perf_counter()
    random_forest.fit(X_train, y_train)
    rf_seconds = time.perf_counter() - rf_started

    print("[4/6] Treinando XGBClassifier (avançado)...", flush=True)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    xgboost = build_model(
        XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    xgb_started = time.perf_counter()
    xgboost.fit(X_train, y_train)
    xgb_seconds = time.perf_counter() - xgb_started

    print("[5/6] Avaliando modelos e gerando visualizações...", flush=True)
    models = {
        "RandomForest": (random_forest, rf_seconds),
        "XGBoost": (xgboost, xgb_seconds),
    }
    results: dict[str, dict] = {}
    importance_rows: list[pd.DataFrame] = []

    for name, (model, seconds) in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["train_seconds"] = round(seconds, 2)
        results[name] = metrics
        print("      " + format_metrics(name, metrics), flush=True)

        prefix = name.lower()
        plot_confusion_matrix(
            name,
            metrics["confusion_matrix"],
            FIGURES_DIR / f"{prefix}_confusion_matrix.png",
        )
        ranking = plot_feature_importance(
            name,
            model.named_steps["model"].feature_importances_,
            FEATURES,
            FIGURES_DIR / f"{prefix}_feature_importance.png",
        )
        ranking.insert(0, "model", name)
        importance_rows.append(ranking)

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "train_seconds": metrics["train_seconds"],
            }
            for name, metrics in results.items()
        ]
    ).sort_values("roc_auc", ascending=False)
    comparison.round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    pd.concat(importance_rows, ignore_index=True).round(6).to_csv(
        REPORTS_DIR / "feature_importance.csv", index=False
    )

    best_name = comparison.iloc[0]["model"]
    best_model = models[best_name][0]

    print("[6/6] Salvando artefatos...", flush=True)
    model_path = MODELS_DIR / "credit_model.pkl"
    joblib.dump(best_model, model_path)

    metrics_payload = {
        "best_model": best_name,
        "selection_metric": "roc_auc",
        "decision_threshold": DECISION_THRESHOLD,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features": FEATURES,
        "models": results,
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.perf_counter() - started
    print(f"\nMelhor modelo: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.4f})")
    print(f"Modelo salvo em: {model_path}")
    print(f"Relatórios em: {REPORTS_DIR}")
    print(f"Tempo total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
