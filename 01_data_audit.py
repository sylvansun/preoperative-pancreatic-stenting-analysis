"""
01_data_audit.py
"""

import pandas as pd

from src.config import config
from src.utils import setup_logger

# ==========================================================
# Load Data
# ==========================================================


def load_data(logger):

    logger.info("Loading raw dataset")

    df = pd.read_excel(config.RAW_FILE)

    logger.info(f"Shape: {df.shape}")

    return df


# ==========================================================
# Variable Dictionary
# ==========================================================


def export_variable_dictionary(df, logger):

    out = pd.DataFrame({"variable": df.columns, "dtype": df.dtypes.astype(str)})

    out.to_excel(config.AUDIT_DIR / "variable_dictionary.xlsx", index=False)

    logger.info("Variable dictionary saved")


# ==========================================================
# Missing Data
# ==========================================================


def export_missing_summary(df, logger):

    out = pd.DataFrame({"variable": df.columns, "missing_n": df.isna().sum()})

    out["missing_pct"] = (out["missing_n"] / len(df) * 100).round(2)

    out = out.sort_values("missing_pct", ascending=False)

    out.to_excel(config.AUDIT_DIR / "missing_summary.xlsx", index=False)

    logger.info("Missing summary saved")


# ==========================================================
# PS Distribution
# ==========================================================


def export_ps_distribution(df, logger):

    if "PS" not in df.columns:
        logger.warning("PS column not found")
        return

    out = df["PS"].value_counts(dropna=False).reset_index()

    out.columns = ["PS", "n"]

    out.to_excel(config.AUDIT_DIR / "ps_distribution.xlsx", index=False)

    logger.info("PS distribution saved")


# ==========================================================
# Duplicate Check
# ==========================================================


def export_duplicate_check(df, logger):

    if "住院号" not in df.columns:
        logger.warning("住院号 column not found")
        return

    dup = df[df["住院号"].duplicated(keep=False)]

    dup.to_excel(config.AUDIT_DIR / "duplicate_hospital_id.xlsx", index=False)

    logger.info(f"Duplicate records: {len(dup)}")


# ==========================================================
# Main
# ==========================================================


def main():

    logger = setup_logger(config.LOG_DIR / "01_data_audit.log", "data_audit")

    logger.info("=" * 50)
    logger.info("START DATA AUDIT")
    logger.info("=" * 50)

    df = load_data(logger)

    export_variable_dictionary(df, logger)
    export_missing_summary(df, logger)
    export_ps_distribution(df, logger)
    export_duplicate_check(df, logger)

    logger.info("=" * 50)
    logger.info("DATA AUDIT COMPLETE")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
