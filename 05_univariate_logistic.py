"""
05_univariate_logistic.py

Purpose:
    Scientific-grade univariate logistic regression
    + forest plot for CR-POPF.

Outputs:
    tables/Table3_univariate_logistic.xlsx
    figures/Figure_univariate_forest.png
    logs/05_univariate_logistic.log
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
# Candidate variables (clinical-driven)
# ==========================================================


def select_variables(df):

    candidates = [
        # exposure (核心研究变量)
        "PS",
        # patient factors
        "sex",
        # "age",
        # "BMI",
        # tumor factors
        "tumor_large",
        "tumor_high_risk_location",
        "tumor_size_mm",
        "mpd_close",
        "mpd_diameter_mm",
        # "mpd_distance_mm",
        "cbd_distance_mm",
        # surgical factors
        # "operation_time",
        # "blood_loss",
    ]

    variables = [v for v in candidates if v in df.columns]

    return variables


# ==========================================================
# Univariate logistic regression
# ==========================================================


def run_univariate_logit(df, outcome, var):

    data = df[[outcome, var]].dropna()

    if data[var].nunique() < 2:
        return None

    y = data[outcome]
    X = sm.add_constant(data[var])

    try:
        model = sm.Logit(y, X).fit(disp=0)

        coef = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]

        or_val = np.exp(coef)
        ci_low = np.exp(coef - 1.96 * se)
        ci_high = np.exp(coef + 1.96 * se)

        return {
            "Variable": var,
            "OR": or_val,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "P value": p,
        }

    except Exception:
        return None


# ==========================================================
# Build table
# ==========================================================


def build_table(df, logger):

    outcome = "cr_popf"

    variables = select_variables(df)

    logger.info(f"Variables used: {variables}")

    results = []

    for var in variables:

        logger.info(f"Running: {var}")

        res = run_univariate_logit(df, outcome, var)

        if res:
            results.append(res)

    table = pd.DataFrame(results)

    table["P value"] = table["P value"].astype(float)

    table = table.sort_values("P value")

    return table


# ==========================================================
# Save table
# ==========================================================


def save_table(table, logger):

    out_path = config.TABLE_DIR / "Table3_univariate_logistic.xlsx"

    table.to_excel(out_path, index=False)

    logger.info(f"Saved table: {out_path}")


# ==========================================================
# Forest plot (SCI-style)
# ==========================================================


def plot_forest(table, logger):

    df = table.copy()

    df = df.sort_values("OR")

    y_pos = np.arange(len(df))

    plt.figure(figsize=(9, 6))

    # log scale (important in medical journals)
    plt.xscale("log")

    # CI lines
    plt.hlines(y=y_pos, xmin=df["CI_low"], xmax=df["CI_high"], color="black")

    # OR points
    plt.scatter(df["OR"], y_pos, color="black", s=40)

    # reference line
    plt.axvline(1, linestyle="--", color="red", linewidth=1)

    # labels
    plt.yticks(y_pos, df["Variable"])

    plt.xlabel("Odds Ratio (log scale)")
    plt.title("Univariate Logistic Regression for CR-POPF")

    plt.tight_layout()

    out_path = config.FIGURE_DIR / "Figure_univariate_forest.png"

    plt.savefig(out_path, dpi=300)

    plt.close()

    logger.info(f"Saved forest plot: {out_path}")


# ==========================================================
# Main
# ==========================================================


def main():

    logger = setup_logger(
        config.LOG_DIR / "05_univariate_logistic.log", "univariate_logistic"
    )

    logger.info("=" * 60)
    logger.info("START UNIVARIATE LOGISTIC (SCI VERSION)")
    logger.info("=" * 60)

    df = load_data(logger)

    table = build_table(df, logger)

    save_table(table, logger)

    plot_forest(table, logger)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
