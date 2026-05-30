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

    ps_vals = ps[var].dropna()
    nps_vals = nps[var].dropna()

    # skip if empty
    if len(ps_vals) < 2 or len(nps_vals) < 2:
        return None, None, np.nan

    # normality check (robust)
    try:
        p1 = stats.shapiro(ps_vals.sample(min(50, len(ps_vals)))).pvalue
        p2 = stats.shapiro(nps_vals.sample(min(50, len(nps_vals)))).pvalue
    except:
        p1, p2 = 0, 0

    normal = (p1 > 0.05) and (p2 > 0.05)

    if normal:
        stat, p = stats.ttest_ind(ps_vals, nps_vals, equal_var=False)

        ps_str = f"{ps_vals.mean():.2f} ± {ps_vals.std():.2f}"
        nps_str = f"{nps_vals.mean():.2f} ± {nps_vals.std():.2f}"

    else:
        stat, p = stats.mannwhitneyu(ps_vals, nps_vals)

        ps_str = f"{ps_vals.median():.2f} ({ps_vals.quantile(0.25):.2f}-{ps_vals.quantile(0.75):.2f})"
        nps_str = f"{nps_vals.median():.2f} ({nps_vals.quantile(0.25):.2f}-{nps_vals.quantile(0.75):.2f})"

    return ps_str, nps_str, p


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
        return "; ".join(
            [f"{k}: {v} ({v/total*100:.1f}%)" for k, v in vc.items()]
        )

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

                ps_s, nps_s, p = analyze_categorical(var, df, ps, nps)

            if p is None:
                continue

            results.append({
                "Variable": var,
                "PS": ps_s,
                "NPS": nps_s,
                "P value": round(p, 4)
            })

        except Exception as e:
            logger.warning(f"Skip {var}: {e}")

    table1 = pd.DataFrame(results)

    # optional: sort by variable name
    table1 = table1.sort_values("Variable")

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

    logger = setup_logger(
        config.LOG_DIR / "03_baseline_table.log",
        "table1"
    )

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