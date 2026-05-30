"""
03_baseline_table.py

Purpose:
    Generate Table 1 (Baseline characteristics)
    comparing PS vs NPS groups.

Output:
    tables/Table1_baseline.xlsx
    logs/03_baseline_table.log
"""

import pandas as pd
import numpy as np
from scipy import stats

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
# Split groups
# ==========================================================


def split_groups(df):

    ps = df[df["PS"] == 1]
    nps = df[df["PS"] == 0]

    return ps, nps


# ==========================================================
# Continuous variables
# ==========================================================


def analyze_continuous(var, ps, nps):

    ps_vals = pd.to_numeric(ps[var], errors="coerce").dropna()
    nps_vals = pd.to_numeric(nps[var], errors="coerce").dropna()

    if len(ps_vals) < 2 or len(nps_vals) < 2:
        return None, None, np.nan

    try:
        _, p = stats.mannwhitneyu(ps_vals, nps_vals, alternative="two-sided")
    except Exception:
        p = np.nan

    # mean
    ps_mean = ps_vals.mean()
    nps_mean = nps_vals.mean()

    # range (min-max)
    ps_min = ps_vals.min()
    ps_max = ps_vals.max()

    nps_min = nps_vals.min()
    nps_max = nps_vals.max()

    ps_str = f"{ps_mean:.2f} ({ps_min:.2f}-{ps_max:.2f})"
    nps_str = f"{nps_mean:.2f} ({nps_min:.2f}-{nps_max:.2f})"

    return ps_str, nps_str, round(p, 4) if pd.notna(p) else np.nan


# ==========================================================
# Categorical variables
# ==========================================================


def analyze_categorical(var, df, ps, nps):

    ct = pd.crosstab(df[var], df["PS"])

    # Fisher for 2x2
    if ct.shape == (2, 2):
        _, p = stats.fisher_exact(ct)
    else:
        _, p, _, _ = stats.chi2_contingency(ct)

    def fmt(group):
        vc = group.value_counts(dropna=False)
        total = len(group)
        return "; ".join([f"{k}: {v} ({v/total*100:.1f}%)" for k, v in vc.items()])

    return fmt(ps[var]), fmt(nps[var]), p


# ==========================================================
# Variable filtering
# ==========================================================


def select_variables(df):

    exclude = {
        "PS",
        "住院号",
        "病人姓名",
        "身份证号",
        "联系人电话",
        "现住址",
        "入院日期",
        "出院日期",
        "手术日期",
    }

    vars_ = [c for c in df.columns if c not in exclude]

    return vars_


# ==========================================================
# Main Table 1 builder
# ==========================================================


def build_table1(df, logger):

    ps, nps = split_groups(df)

    variables = select_variables(df)

    results = []

    for var in variables:

        try:
            # numeric → continuous
            if pd.api.types.is_numeric_dtype(df[var]):

                ps_s, nps_s, p = analyze_continuous(var, ps, nps)

            # categorical → non-numeric
            else:
                continue
                # skipped for now, cuz we do not need these variables for further analysis
                ps_s, nps_s, p = analyze_categorical(var, df, ps, nps)

            if p is None:
                continue

            results.append(
                {"Variable": var, "PS": ps_s, "NPS": nps_s, "P value": f"{p:.4f}" if pd.notna(p) else ""}
            )

        except Exception as e:
            logger.warning(f"Skip {var}: {e}")

    table1 = pd.DataFrame(results)

    # optional: sort by variable name
    # table1 = table1.sort_values("Variable")

    return table1


# ==========================================================
# Save
# ==========================================================


def save_table(table1, logger):

    out_path = config.TABLE_DIR / "Table1_baseline.xlsx"

    table1.to_excel(out_path, index=False)

    logger.info(f"Table1 saved to {out_path}")


# ==========================================================
# Main
# ==========================================================


def main():

    logger = setup_logger(config.LOG_DIR / "03_baseline_table.log", "table1")

    logger.info("=" * 60)
    logger.info("START TABLE 1")
    logger.info("=" * 60)

    df = load_data(logger)

    table1 = build_table1(df, logger)

    save_table(table1, logger)

    logger.info("=" * 60)
    logger.info("TABLE 1 DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
