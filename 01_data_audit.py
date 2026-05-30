"""
01_data_audit.py

Purpose:
    Perform initial data quality check on raw clinical dataset.

Outputs:
    data/audit/
        variable_dictionary.xlsx
        missing_summary.xlsx
        ps_distribution.xlsx
        duplicate_hospital_id.xlsx

Logs:
    logs/01_data_audit.log
"""

from pathlib import Path
import logging
import pandas as pd

# ==========================================================
# Path Configuration
# ==========================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "raw.xlsx"

AUDIT_DIR = ROOT / "data" / "audit"
LOG_DIR = ROOT / "logs"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Logger
# ==========================================================


def setup_logger():

    logger = logging.getLogger("data_audit")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    handler = logging.FileHandler(LOG_DIR / "01_data_audit.log", encoding="utf-8")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ==========================================================
# Load Data
# ==========================================================


def load_data(logger):

    logger.info("Loading raw dataset")

    df = pd.read_excel(DATA_FILE)

    logger.info(f"Shape: {df.shape}")

    return df


# ==========================================================
# Audit: Variable Dictionary
# ==========================================================


def export_variable_dictionary(df, logger):

    out = pd.DataFrame({"variable": df.columns, "dtype": df.dtypes.astype(str)})

    out.to_excel(AUDIT_DIR / "variable_dictionary.xlsx", index=False)

    logger.info("Variable dictionary saved")


# ==========================================================
# Audit: Missing Data
# ==========================================================


def export_missing_summary(df, logger):

    out = pd.DataFrame({"variable": df.columns, "missing_n": df.isna().sum()})

    out["missing_pct"] = (out["missing_n"] / len(df) * 100).round(2)

    out = out.sort_values("missing_pct", ascending=False)

    out.to_excel(AUDIT_DIR / "missing_summary.xlsx", index=False)

    logger.info("Missing summary saved")


# ==========================================================
# Audit: PS Distribution
# ==========================================================


def export_ps_distribution(df, logger):

    if "PS" not in df.columns:
        logger.warning("PS column not found")
        return

    out = df["PS"].value_counts(dropna=False).reset_index()

    out.columns = ["PS", "n"]

    out.to_excel(AUDIT_DIR / "ps_distribution.xlsx", index=False)

    logger.info("PS distribution saved")


# ==========================================================
# Audit: Duplicate ID Check
# ==========================================================


def export_duplicate_check(df, logger):

    if "住院号" not in df.columns:
        logger.warning("住院号 column not found")
        return

    dup = df[df["住院号"].duplicated(keep=False)].copy()

    dup.to_excel(AUDIT_DIR / "duplicate_hospital_id.xlsx", index=False)

    logger.info(f"Duplicate records: {len(dup)}")


# ==========================================================
# Main
# ==========================================================


def main():

    logger = setup_logger()

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
