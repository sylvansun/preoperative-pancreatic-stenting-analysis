from pathlib import Path


class Config:
    """
    Central configuration for the whole project.
    All paths should be defined here.
    """

    def __init__(self):

        # project root (auto-detected)
        self.ROOT = Path(__file__).resolve().parent.parent

        # data
        self.DATA_DIR = self.ROOT / "data"
        self.RAW_FILE = self.DATA_DIR / "raw.xlsx"
        self.ANALYSIS_FILE = self.DATA_DIR / "analysis_dataset.pkl"
        self.AUDIT_DIR = self.DATA_DIR / "audit"

        # outputs
        self.FIGURE_DIR = self.ROOT / "figures"
        self.TABLE_DIR = self.ROOT / "tables"
        self.LOG_DIR = self.ROOT / "logs"

        # ensure directories exist
        self._create_dirs()

    def _create_dirs(self):

        for d in [
            self.DATA_DIR,
            self.AUDIT_DIR,
            self.FIGURE_DIR,
            self.TABLE_DIR,
            self.LOG_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# global singleton (recommended pattern for research projects)
config = Config()
