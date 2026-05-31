"""
07_roc_curve.py

Purpose:
    SCI-grade ROC analysis for CR-POPF prediction
    based on multivariable logistic regression model.

Outputs:
    tables/Table4_roc.csv
    figures/Figure6_roc_curve.png
    logs/07_roc_curve.log
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.config import config
from src.utils import setup_logger

# ==========================================================
# Load data
# ==========================================================


def load_data(logger):
    df = pd.read_pickle(config.ANALYSIS_FILE)
    logger.info(f"Dataset loaded: {df.shape}")
    return df


# ==========================================================
# Fit multivariable logistic regression
# ==========================================================


def fit_logit_model(df, outcome, predictors, logger):
    X = df[predictors]
    X = sm.add_constant(X)
    y = df[outcome]

    model = sm.Logit(y, X).fit(disp=0)
    logger.info(model.summary())
    return model


# ==========================================================
# Compute predicted probabilities
# ==========================================================


def get_pred_probs(model, df, predictors):
    X = df[predictors]
    X = sm.add_constant(X)
    probs = model.predict(X)
    return probs


# ==========================================================
# ROC analysis
# ==========================================================


def compute_roc(y_true, y_prob, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Bootstrap for 95% CI
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        try:
            fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_prob[idx])
            aucs.append(auc(fpr_b, tpr_b))
        except:
            continue
    aucs = np.array(aucs)
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)

    # Best threshold by Youden index
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]
    best_sensitivity = tpr[best_idx]
    best_specificity = 1 - fpr[best_idx]

    return (
        fpr,
        tpr,
        thresholds,
        roc_auc,
        ci_lower,
        ci_upper,
        best_threshold,
        best_sensitivity,
        best_specificity,
    )


# ==========================================================
# Save ROC table
# ==========================================================


def save_roc_table(fpr, tpr, thresholds):
    df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})
    out_path = config.TABLE_DIR / "Table4_roc.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ==========================================================
# Plot ROC
# ==========================================================


def plot_roc(
    fpr, tpr, roc_auc, ci_lower, ci_upper, best_threshold, best_sens, best_spec
):
    plt.figure(figsize=(7, 7))
    plt.plot(
        fpr,
        tpr,
        color="blue",
        lw=2,
        label=f"AUC = {roc_auc:.3f} (95% CI {ci_lower:.3f}-{ci_upper:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    plt.scatter(
        1 - best_spec,
        best_sens,
        color="red",
        zorder=5,
        label=f"Best threshold={best_threshold:.3f}",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for CR-POPF Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path = config.FIGURE_DIR / "Figure6_roc_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


# ==========================================================
# Main
# ==========================================================


def main():
    logger = setup_logger(config.LOG_DIR / "07_roc_curve.log", "roc_curve")
    logger.info("START SCI-grade ROC analysis")

    df = load_data(logger)

    outcome = "cr_popf"
    predictors = ["PS", "tumor_high_risk_location", "tumor_size_mm", "cbd_distance_mm"]

    model = fit_logit_model(df, outcome, predictors, logger)
    probs = get_pred_probs(model, df, predictors)

    (
        fpr,
        tpr,
        thresholds,
        roc_auc,
        ci_lower,
        ci_upper,
        best_thresh,
        best_sens,
        best_spec,
    ) = compute_roc(df[outcome].values, probs.values)

    table_path = save_roc_table(fpr, tpr, thresholds)
    logger.info(f"ROC table saved: {table_path}")

    fig_path = plot_roc(
        fpr, tpr, roc_auc, ci_lower, ci_upper, best_thresh, best_sens, best_spec
    )
    logger.info(f"ROC figure saved: {fig_path}")

    logger.info("DONE - ROC ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
