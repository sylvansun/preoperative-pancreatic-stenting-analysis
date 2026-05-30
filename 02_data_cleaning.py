"""
02_data_cleaning.py

Purpose:
    Clean raw clinical dataset and generate analysis-ready dataset.

Outputs:
    data/analysis_dataset.pkl
    logs/02_data_cleaning.log
"""

from src.config import config
import pandas as pd
import logging
import numpy as np

# ==========================================================
# Logger
# ==========================================================


def setup_logger():

    logger = logging.getLogger("data_cleaning")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    handler = logging.FileHandler(config.LOG_DIR / "02_data_cleaning.log")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ==========================================================
# Load raw data
# ==========================================================


def load_data(logger):

    df = pd.read_excel(config.RAW_FILE)

    logger.info(f"Raw shape: {df.shape}")

    return df


# ==========================================================
# Remove identifiers (PII)
# ==========================================================


def drop_pii(df, logger):

    pii_cols = ["病人姓名", "身份证号", "联系人电话", "现住址", "住院号"]

    existing = [c for c in pii_cols if c in df.columns]

    df = df.drop(columns=existing)

    logger.info(f"Removed PII columns: {existing}")

    return df


# ==========================================================
# Basic encoding
# ==========================================================


def encode_basic_vars(df, logger):

    # PS: 1=PS, 2=NPDS → 1/0
    if "PS" in df.columns:
        df["PS"] = df["PS"].map({1: 1, 2: 0})
        logger.info("PS encoded to binary (1=PS,0=NPDS)")

    # sex encoding (M/F)
    if "病人性别" in df.columns:

        df["sex"] = df["病人性别"].map({"M": 1, "F": 0})

        logger.info("Sex encoded (M=1, F=0)")

        df = df.drop(columns=["病人性别"])

    return df


# ==========================================================
# Derived clinical variables
# ==========================================================


def create_derived_vars(df, logger):

    # tumor size grouping
    if "肿瘤大小" in df.columns:

        df["tumor_large"] = np.where(df["肿瘤大小"] > 20, 1, 0)

    # high-risk location (head/uncinate)
    if "肿瘤位置（钩突1，头2，颈体3）" in df.columns:

        df["tumor_high_risk_location"] = (
            df["肿瘤位置（钩突1，头2，颈体3）"].isin([1, 2]).astype(int)
        )

    # close to MPD
    if "胰管距离≤2mm" in df.columns:

        df["mpd_close"] = df["胰管距离≤2mm"].fillna(0).astype(int)

    return df


# ==========================================================
# Consistency checks
# ==========================================================


def run_checks(df, logger):

    assert df.shape[0] > 0, "Empty dataset!"

    if "PS" in df.columns:
        logger.info(f"PS distribution:\n{df['PS'].value_counts(dropna=False)}")

    logger.info("Basic consistency checks passed")

    return df


# ==========================================================
# Save dataset
# ==========================================================


def save_dataset(df, logger):

    df.to_pickle(config.ANALYSIS_FILE)

    logger.info(f"Saved analysis dataset to {config.ANALYSIS_FILE}")


# ==========================================================
# Main
# ==========================================================


def main():

    logger = setup_logger()

    logger.info("=" * 50)
    logger.info("START DATA CLEANING")
    logger.info("=" * 50)

    df = load_data(logger)

    df = drop_pii(df, logger)
    df = encode_basic_vars(df, logger)
    df = create_derived_vars(df, logger)
    df = run_checks(df, logger)

    save_dataset(df, logger)

    logger.info("=" * 50)
    logger.info("DATA CLEANING COMPLETE")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
