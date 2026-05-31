"""
06_multivariable_logistic.py

Purpose:
    SCI-grade multivariable logistic regression
    for CR-POPF with 4 clinical predictors.
    Forest plot includes variable type and OR interpretation.

Outputs:
    tables/Table4_multivariable_logistic.xlsx
    figures/Figure6_multivariable_forest.png
    logs/06_multivariable_logistic.log
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
# Define predictors
# ==========================================================

binary_vars = ["PS", "tumor_high_risk_location"]

continuous_vars = ["tumor_size_mm", "cbd_distance_mm"]

all_vars = binary_vars + continuous_vars

# ==========================================================
# Multivariable logistic regression
# ==========================================================


def run_multivariable_logit(df, outcome, predictors, logger):
    data = df[[outcome] + predictors].dropna()
    y = data[outcome]
    X = data[predictors]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=0)
    logger.info(model.summary2().as_text())

    results = []
    for var in predictors:
        coef = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        or_val = np.exp(coef)
        ci_low = np.exp(coef - 1.96 * se)
        ci_high = np.exp(coef + 1.96 * se)
        var_type = "Binary" if var in binary_vars else "Continuous"
        # OR interpretation string
        if var_type == "Binary":
            or_note = "1=Yes / 0=No"
        else:
            or_note = "per mm"
        results.append(
            {
                "Variable": var,
                "Type": var_type,
                "Adjusted OR": or_val,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "P": p,
                "OR_note": or_note,
            }
        )
    return pd.DataFrame(results), model


# ==========================================================
# Check multicollinearity (VIF)
# ==========================================================


def check_vif(df, predictors, logger):
    X = df[predictors].dropna()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    logger.info("VIF check:\n" + str(vif_data))
    return vif_data


# ==========================================================
# Save table
# ==========================================================


def save_table(table, logger):
    out_path = config.TABLE_DIR / "Table4_multivariable_logistic.xlsx"
    table.to_excel(out_path, index=False)
    logger.info(f"Saved Table4: {out_path}")


# ==========================================================
# Forest plot with OR interpretation
# ==========================================================


def plot_forest(table, logger):
    df = table.sort_values("Adjusted OR")
    y = np.arange(len(df))
    plt.figure(figsize=(9, 6))
    plt.xscale("log")
    plt.hlines(y, df["CI_low"], df["CI_high"], color="black")
    plt.scatter(df["Adjusted OR"], y, color="black", s=40)
    plt.axvline(1, linestyle="--", color="red")

    # Variable labels with type and OR note
    labels = [
        f"{row['Variable']} ({row['Type']}, {row['OR_note']})"
        for _, row in df.iterrows()
    ]
    plt.yticks(y, labels)
    plt.xlabel("Adjusted OR (log scale)")
    plt.title("Multivariable Logistic Regression for CR-POPF")
    plt.tight_layout()
    out_path = config.FIGURE_DIR / "Figure6_multivariable_forest.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved forest plot: {out_path}")


# ==========================================================
# Figure 2: OR + P combined bubble plot (multivariable)
# 气泡越大 → 越显著
# OR>1 左右摆 → 风险增加/减少
# P<0.05 蓝线下方 → 显著
# 同时看 OR 大小和统计学意义，非常直观
# ==========================================================
def plot_or_p_combined_multivariable(table, logger):
    """
    Bubble plot: Adjusted OR vs P value
    Bubble size: -log10(P)
    Labels: Variable name + type
    """
    df = table.copy()
    plt.figure(figsize=(8, 6))

    # bubble size based on significance
    sizes = -np.log10(df["P"] + 1e-10) * 50

    plt.scatter(
        df["Adjusted OR"], df["P"], s=sizes, alpha=0.7, color="teal", edgecolors="k"
    )

    # log scales
    plt.xscale("log")
    plt.yscale("log")

    # reference lines
    plt.axvline(1, linestyle="--", color="red", linewidth=1)
    plt.axhline(0.05, linestyle="--", color="blue", linewidth=1)

    # add variable labels
    for i in range(len(df)):
        plt.text(
            df["Adjusted OR"].iloc[i] * 1.02,
            df["P"].iloc[i] * 1.02,
            f"{df['Variable'].iloc[i]} ({df['Type'].iloc[i]})",
            fontsize=9,
        )

    plt.xlabel("Adjusted OR (log scale)")
    plt.ylabel("P value (log scale)")
    plt.title("Multivariable Logistic Regression: OR vs P")
    plt.tight_layout()

    out_path = config.FIGURE_DIR / "Figure7_multivariable_or_p.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved multivariable OR+P bubble plot: {out_path}")


# ==========================================================
# Main
# ==========================================================


def main():
    logger = setup_logger(
        config.LOG_DIR / "06_multivariable_logistic.log", "multivariable"
    )
    logger.info("START MULTIVARIABLE LOGISTIC PIPELINE")

    df = load_data(logger)

    # VIF
    check_vif(df, all_vars, logger)

    # Run model
    table, model = run_multivariable_logit(df, "cr_popf", all_vars, logger)

    save_table(table, logger)
    plot_forest(table, logger)
    plot_or_p_combined_multivariable(table, logger)

    logger.info("DONE - MULTIVARIABLE LOGISTIC REGRESSION COMPLETED")


if __name__ == "__main__":
    main()
