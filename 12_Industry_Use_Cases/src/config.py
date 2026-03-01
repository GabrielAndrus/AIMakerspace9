import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    JOB_DB_PATH: str = "data/jobs.db"

    def __post_init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)


settings = Settings()
