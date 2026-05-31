"""
04_outcome_analysis.py

Purpose:
    Compare postoperative outcomes between
    PS and NPS groups.

Output:
    tables/Table2_outcomes.xlsx
    logs/04_outcome_analysis.log
"""

import pandas as pd
from scipy import stats

from src.config import config
from src.utils import setup_logger

# ==========================================================
# Load Data
# ==========================================================


def load_data(logger):

    df = pd.read_pickle(config.ANALYSIS_FILE)

    logger.info(f"Dataset loaded: {df.shape}")

    return df


# ==========================================================
# Binary Outcome Analysis
# ==========================================================


def analyze_binary_outcome(df, outcome):

    ps = df[df["PS"] == 1]
    nps = df[df["PS"] == 0]

    # contingency table
    ct = pd.crosstab(df[outcome], df["PS"])

    try:

        if ct.shape == (2, 2):

            _, p = stats.fisher_exact(ct)

        else:

            _, p, _, _ = stats.chi2_contingency(ct)

    except Exception:

        p = float("nan")

    ps_n = ps[outcome].sum()
    ps_total = len(ps)

    nps_n = nps[outcome].sum()
    nps_total = len(nps)

    return {
        "Outcome": outcome,
        "PS": f"{ps_n}/{ps_total} ({ps_n / ps_total * 100:.1f}%)",
        "NPS": f"{nps_n}/{nps_total} ({nps_n / nps_total * 100:.1f}%)",
        "P value": f"{p:.4f}" if pd.notna(p) else "",
    }


# ==========================================================
# Build Outcome Table
# ==========================================================


def build_outcome_table(df, logger):

    outcomes = {
        "fistula_any": "Any pancreatic fistula",
        "cr_popf": "CR-POPF (ISGPF B+C)",
        "infection_any": "Any infection",
        "reoperation": "Reoperation",
    }

    results = []

    for var, label in outcomes.items():

        if var not in df.columns:

            logger.warning(f"{var} not found")

            continue

        row = analyze_binary_outcome(df, var)

        row["Outcome"] = label

        results.append(row)

    table = pd.DataFrame(results)

    return table


# ==========================================================
# Save
# ==========================================================


def save_table(table, logger):

    out_path = config.TABLE_DIR / "Table2_outcomes.xlsx"

    table.to_excel(out_path, index=False)

    logger.info(f"Outcome table saved to {out_path}")


# ==========================================================
# Main
# ==========================================================


def main():

    logger = setup_logger(
        config.LOG_DIR / "04_outcome_analysis.log",
        "outcome_analysis",
    )

    logger.info("=" * 60)
    logger.info("START OUTCOME ANALYSIS")
    logger.info("=" * 60)

    df = load_data(logger)

    table = build_outcome_table(df, logger)

    save_table(table, logger)

    logger.info("=" * 60)
    logger.info("OUTCOME ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
