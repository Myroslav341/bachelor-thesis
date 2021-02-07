import os


class _RowsManagerConfig:
    # if angle for current vector > NEW_ROW_ANGLE, based on this vector new row will be created
    NEW_ROW_ANGLE = 70


class Config(_RowsManagerConfig):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
