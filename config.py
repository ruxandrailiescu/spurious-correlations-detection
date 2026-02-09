import os


DATA_PATH = os.environ.get("DATA_PATH", "./")
RESULTS_PATH = os.environ.get("RESULTS_PATH", "./results/")
CACHE_PATH = os.environ.get("CACHE_PATH", "./cache/")


# Fields:
#   metadata   – CSV with columns: y (label), a (attribute/environment), split (0/1/2)
#   directory  – subfolder under DATA_PATH
#   image_dir  – subfolder for images (empty string for text datasets)
#   classes    – ordered class names
#   class_bias – per-class spurious attribute value (only for datasets with known SC)
#   text_file  – (csv_filename, text_column) for text datasets; None for image datasets
#   classes_wn – WordNet reference words used for hypernym/hyponym filtering
#   classes_explicit – human description of class concepts (used for keyword filtering)

DATASETS = {
    "ISIC": {
        "metadata": "metadata_isic.csv",
        "directory": "archive",
        "image_dir": "",
        "classes": ["benign", "malignant"],
        "class_bias": None,
        "text_file": None,
        "classes_wn": ["benign", "malignant"],
        "classes_explicit": "skin lesion diagnosis, malignancy, benign and malignant tumors, and disease type names (melanoma, carcinoma, keratosis, dermatofibroma, vascular)",
    },
}


SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
TRAIN_VAL_SPLITS = ["train", "val"]

TEMP_INIT = 4.5
SEED = 1007

STOPWORDS_EXTRA = {"next", "many"}


def is_text_dataset(dataset: str) -> bool:
    return DATASETS[dataset]["text_file"] is not None


def get_classes(dataset: str) -> list[str]:
    return DATASETS[dataset]["classes"]


def get_metadata_path(dataset: str) -> str:
    d = DATASETS[dataset]
    return os.path.join(DATA_PATH, d["directory"], d["metadata"])
