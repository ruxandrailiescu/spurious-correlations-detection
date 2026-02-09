import os


DATA_PATH = os.environ.get("DATA_PATH", "./data/")
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
    "Waterbirds": {
        "metadata": "metadata_waterbirds.csv",
        "directory": "waterbirds",
        "image_dir": "waterbird_complete95_forest2water2",
        "classes": ["landbird", "waterbird"],
        "class_bias": [0, 1],
        "text_file": None,
        "classes_wn": ["bird", "bird"],
        "classes_explicit": "birds and any specific species of birds",
    },
    "CelebA": {
        "metadata": "metadata_celeba.csv",
        "directory": "celeba",
        "image_dir": "img_align_celeba",
        "classes": ["non-blonde hair", "blonde hair"],
        "class_bias": [1, 0],
        "text_file": None,
        "classes_wn": ["hair", "hair"],
        "classes_explicit": "hair and its color (blonde, black, brown, red, gray etc.)",
    },
    "CivilComments": {
        "metadata": "metadata_civilcomments_coarse.csv",
        "directory": "civilcomments",
        "image_dir": "",
        "classes": ["non-offensive", "offensive"],
        "class_bias": [1, 0],
        "text_file": ("civilcomments_coarse.csv", "comment_text"),
        "classes_wn": ["non-offensive", "offensive"],
        "classes_explicit": "remove any offensive words or remarks",
    },
    "MNLI": {
        "metadata": "metadata_mnli.csv",
        "directory": "mnli",
        "image_dir": "",
        "classes": ["fiction", "government", "slate", "telephone", "travel"],
        "class_bias": None,
        "text_file": ("mnli_questions.csv", "question_text"),
        "classes_wn": ["fiction", "government", "slate", "telephone", "travel"],
        "classes_explicit": "genre labels (fiction, government, slate, telephone, travel) and directly related terms",
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
