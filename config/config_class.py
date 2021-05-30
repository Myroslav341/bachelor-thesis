import os


class _RowsManagerConfig:
    pass


class Config(_RowsManagerConfig):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
