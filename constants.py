import os


DATA_PATH = os.environ.get('DATA_PATH', './data/')
RESULTS_PATH = os.environ.get('RESULTS_PATH', './results/')
CKPT_PATH = os.environ.get('CKPT_PATH', './ckpt/')
CACHE_PATH = os.environ.get('CACHE_PATH', './cache/')

DATASET_DIR = {
    'MMLUPro': 'mmlu_pro',
}

DATASET_FILE = {
    'MMLUPro': 'mmlu_pro_v2.csv',
}

METADATA_FILE = {
    'MMLUPro': 'metadata_mmlu_pro.csv',
}

SPLITS = ['train', 'val']

DATASET_SPLITS = {
    'train': 0,
    'val': 1,
    'test': 2,
}

DATASET_CLASSES = {
    'MMLUPro': ['correct', 'incorrect'],
}

SEED = 5454