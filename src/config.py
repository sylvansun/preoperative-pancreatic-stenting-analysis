"""
config.py
"""

from pathlib import Path


class Config:

    def __init__(self):

        # project root
        self.ROOT = Path(__file__).resolve().parent.parent

        # data
        self.DATA_DIR = self.ROOT / "data"
        self.RAW_FILE = self.DATA_DIR / "raw.xlsx"

        # audit outputs
        self.AUDIT_DIR = self.DATA_DIR / "audit"

        # outputs
        self.TABLE_DIR = self.ROOT / "tables"
        self.FIGURE_DIR = self.ROOT / "figures"
        self.ANALYSIS_FILE = self.DATA_DIR / "analysis_dataset.pkl"
        self.LOG_DIR = self.ROOT / "logs"

        # variables
        self.CATEGORICAL_VARS = {"operation_group"}
        # ensure folders exist
        self._init_dirs()

    def _init_dirs(self):

        for d in [
            self.DATA_DIR,
            self.AUDIT_DIR,
            self.TABLE_DIR,
            self.FIGURE_DIR,
            self.LOG_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# singleton
config = Config()
