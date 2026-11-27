from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODEL_DIR / "face_mask_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


def find_data_dir():
    """Find the folder that contains both `with_mask` and `without_mask`."""
    base = ROOT / "data"
    if base.exists():
        for p in base.rglob("with_mask"):
            if p.is_dir() and (p.parent / "without_mask").is_dir():
                return p.parent
    return base / "data"


DATA_DIR = find_data_dir()  # override manually here if needed

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VAL_TEST_SPLIT = 0.30   # 70% train / 15% val / 15% test

EPOCHS_HEAD = 5
EPOCHS_FINE_TUNE = 3
LR_HEAD = 1e-3
LR_FINE_TUNE = 1e-5
FINE_TUNE_LAYERS = 30
