from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
for p in (DATA_DIR, FIG_DIR, OUT_DIR):
    p.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SAMPLES = 100_000
FAILURE_RATE = 0.005
TEST_SIZE = 0.15
VALID_SIZE_FROM_TEMP = 0.1764705882  # yields ~15% of total after train/test temporary split
FN_COST = 50_000.0  # missed critical failure: downtime, maintenance, possible safety impact
FP_COST = 5000     # false alarm: inspection / controlled stop / maintenance call

FEATURES = ["vibracao_rms", "temperatura", "pressao", "corrente", "aceleracao_pico", "ruido_db"]
TARGET = "falha"


def generate_sensor_data(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1], size=n, p=[1 - FAILURE_RATE, FAILURE_RATE])

    # Baseline operational regime with correlated sensor effects.
    common = rng.normal(0, 1, n)
    vib = 2.2 + 0.35 * common + rng.normal(0, 0.45, n)
    temp = 68 + 3.0 * common + rng.normal(0, 4.0, n)
    press = 7.2 + 0.18 * common + rng.normal(0, 0.28, n)
    curr = 31 + 2.0 * common + rng.normal(0, 3.0, n)
    accel = 0.65 + 0.10 * common + rng.normal(0, 0.12, n)
    noise = 63 + 1.5 * common + rng.normal(0, 2.5, n)

    # Failure regime: higher vibration, temperature, current and acceleration,
    # with pressure/noise shifts. This creates signal but keeps overlap.
    idx = y == 1
    vib[idx] += rng.normal(2.6, 0.65, idx.sum())
    temp[idx] += rng.normal(16, 5.0, idx.sum())
    press[idx] += rng.normal(0.65, 0.25, idx.sum())
    curr[idx] += rng.normal(9, 3.0, idx.sum())
    accel[idx] += rng.normal(0.42, 0.13, idx.sum())
    noise[idx] += rng.normal(8, 2.5, idx.sum())

    df = pd.DataFrame({
        "vibracao_rms": vib,
        "temperatura": temp,
        "pressao": press,
        "corrente": curr,
        "aceleracao_pico": accel,
        "ruido_db": noise,
        TARGET: y.astype(int),
    })
    return df


def cost_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, int, int, int, int]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    total = fn * FN_COST + fp * FP_COST
    return float(total), int(tn), int(fp), int(fn), int(tp)


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "f0_5": float(fbeta_score(y_true, pred, beta=0.5, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": fn * FN_COST + fp * FP_COST,
    }


def main() -> None:
    df = generate_sensor_data()
    df.to_csv(DATA_DIR / "sensor_data.csv", index=False)

    # IMPORTANT: split before fitting preprocessing/model => no test leakage.
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), FEATURES)],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    scores_test = pipeline.predict_proba(X_test)[:, 1]

    scores_val = pipeline.predict_proba(X_val)[:, 1]

    standard = evaluate(y_test.to_numpy(), scores_test, 0.5)

    thresholds = np.linspace(0.01, 0.99, 197)
    rows = []
    for t in thresholds:
        total, tn, fp, fn, tp = cost_at_threshold(y_val.to_numpy(), scores_val, float(t))
        rows.append({"threshold": float(t), "total_cost": total, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    cost_df = pd.DataFrame(rows)
    opt = cost_df.loc[cost_df["total_cost"].idxmin()]
    optimal_threshold = float(opt["threshold"])
    optimized = evaluate(y_test.to_numpy(), scores_test, optimal_threshold)

    # Metrics comparison figure.
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "F2", "F0.5", "AUC-ROC"]
    standard_vals = [standard["accuracy"], standard["precision"], standard["recall"], standard["f1"], standard["f2"], standard["f0_5"], standard["auc_roc"]]
    optimized_vals = [optimized["accuracy"], optimized["precision"], optimized["recall"], optimized["f1"], optimized["f2"], optimized["f0_5"], optimized["auc_roc"]]
    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width/2, standard_vals, width, label="Threshold 0,5")
    ax.bar(x + width/2, optimized_vals, width, label=f"Threshold ótimo {optimal_threshold:.3f}")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, metric_names)
    ax.set_title("Comparação de métricas no conjunto de teste")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metric_comparison.png", dpi=180)
    plt.close(fig)

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, scores_test)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(fpr, tpr, label=f"AUC = {standard['auc_roc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva AUC-ROC")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)

    # Financial threshold curve.
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(cost_df["threshold"], cost_df["total_cost"], linewidth=2)
    std_cost = standard["total_cost"]
    opt_cost = optimized["total_cost"]
    ax.axvline(0.5, linestyle="--", label=f"Padrão = 0,5 | custo R$ {std_cost:,.0f}")
    ax.axvline(optimal_threshold, linestyle="--", label=f"Ótimo = {optimal_threshold:.3f} | custo R$ {opt_cost:,.0f}")
    ax.scatter([optimal_threshold], [opt_cost], s=70, zorder=3)
    ax.set_xlabel("Threshold de decisão")
    ax.set_ylabel("Custo total estimado (R$)")
    ax.set_title("Custo total vs. threshold")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_cost.png", dpi=180)
    plt.close(fig)

    metrics_df = pd.DataFrame([standard, optimized], index=["threshold_0.5", "threshold_otimo"])
    metrics_df.to_csv(OUT_DIR / "metrics_comparison.csv")
    cost_df.to_csv(OUT_DIR / "threshold_cost_curve.csv", index=False)

    summary = {
        "n_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "failure_rate_total": float(y.mean()),
        "failure_rate_train": float(y_train.mean()),
        "failure_rate_validation": float(y_val.mean()),
        "failure_rate_test": float(y_test.mean()),
        "random_state": RANDOM_STATE,
        "model": "LogisticRegression(class_weight='balanced') inside sklearn Pipeline(StandardScaler + classifier)",
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "standard": standard,
        "optimized": optimized,
        "threshold_selection_set": "validation",
        "savings": std_cost - opt_cost,
        "savings_pct": ((std_cost - opt_cost) / std_cost * 100) if std_cost else 0,
    }
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
