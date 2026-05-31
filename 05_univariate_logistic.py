"""
05_univariate_logistic.py

Purpose:
    SCI-grade univariate logistic regression pipeline
    + 4 publication-quality figures

Outputs:
    tables/Table3_univariate_logistic.xlsx
    figures/Figure2_forest.png
    figures/Figure3_pvalue_ranking.png
    figures/Figure4_or_p_combined.png
    figures/Figure5_violin_ps.png
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
# Variables
# ==========================================================


def select_variables(df):

    candidates = [
        "PS",
        "sex",
        "tumor_large",
        "tumor_high_risk_location",
        "tumor_size_mm",
        "mpd_close",
        "mpd_diameter_mm",
        "cbd_distance_mm",
    ]

    return [v for v in candidates if v in df.columns]


# ==========================================================
# Logistic
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
            "P": p,
        }

    except Exception:
        return None


# ==========================================================
# Table
# ==========================================================


def build_table(df, logger):

    outcome = "cr_popf"
    variables = select_variables(df)

    results = []

    for v in variables:

        logger.info(f"Running {v}")

        res = run_univariate_logit(df, outcome, v)

        if res:
            results.append(res)

    table = pd.DataFrame(results)

    table = table.sort_values("P")

    return table


# ==========================================================
# Figure 1: Forest plot
# ==========================================================


def plot_forest(table):

    df = table.sort_values("OR")

    y = np.arange(len(df))

    plt.figure(figsize=(9, 6))
    plt.xscale("log")

    plt.hlines(y, df["CI_low"], df["CI_high"], color="black")
    plt.scatter(df["OR"], y, color="black", s=40)
    plt.axvline(1, linestyle="--", color="red")

    plt.yticks(y, df["Variable"])
    plt.xlabel("Odds Ratio (log scale)")
    plt.title("Univariate Logistic Regression")

    plt.tight_layout()

    plt.savefig(config.FIGURE_DIR / "Figure2_forest.png", dpi=300)
    plt.close()


# ==========================================================
# Figure 2: P-value ranking
# ==========================================================


def plot_pvalue_ranking(table):

    df = table.sort_values("P")

    plt.figure(figsize=(8, 5))

    plt.barh(df["Variable"], -np.log10(df["P"]))

    plt.axvline(-np.log10(0.05), linestyle="--", color="red")

    plt.xlabel("-log10(P value)")
    plt.title("P-value Ranking of Univariate Predictors")

    plt.tight_layout()

    plt.savefig(config.FIGURE_DIR / "Figure3_pvalue_ranking.png", dpi=300)
    plt.close()


# ==========================================================
# Figure 3: OR vs P combined plot
# ==========================================================


def plot_or_p_combined(table):

    df = table.copy()

    plt.figure(figsize=(8, 6))

    sizes = -np.log10(df["P"] + 1e-10) * 50

    plt.scatter(df["OR"], df["P"], s=sizes)

    plt.xscale("log")
    plt.yscale("log")

    plt.axvline(1, linestyle="--", color="red")
    plt.axhline(0.05, linestyle="--", color="blue")

    for i in range(len(df)):
        plt.text(df["OR"].iloc[i], df["P"].iloc[i], df["Variable"].iloc[i])

    plt.xlabel("OR (log scale)")
    plt.ylabel("P value (log scale)")
    plt.title("OR vs P combined plot")

    plt.tight_layout()

    plt.savefig(config.FIGURE_DIR / "Figure4_or_p_combined.png", dpi=300)
    plt.close()


# ==========================================================
# Figure 4: PS distribution (violin)
# ==========================================================


def plot_violin_ps(df):

    import seaborn as sns

    plt.figure(figsize=(6, 5))

    sns.violinplot(x="PS", y="tumor_size_mm", data=df)

    plt.title("Tumor Size Distribution by PS Group")

    plt.tight_layout()

    plt.savefig(config.FIGURE_DIR / "Figure5_violin_ps.png", dpi=300)
    plt.close()


# ==========================================================
# Save table
# ==========================================================


def save_table(table):

    table.to_excel(config.TABLE_DIR / "Table3_univariate_logistic.xlsx", index=False)


# ==========================================================
# Main
# ==========================================================


def main():

    logger = setup_logger(config.LOG_DIR / "05_univariate_logistic.log", "univariate")

    logger.info("START UNIVARIATE PIPELINE (SCI LEVEL)")

    df = load_data(logger)

    table = build_table(df, logger)

    save_table(table)

    plot_forest(table)
    plot_pvalue_ranking(table)
    plot_or_p_combined(table)
    plot_violin_ps(df)

    logger.info("DONE - ALL FIGURES GENERATED")


if __name__ == "__main__":
    main()
