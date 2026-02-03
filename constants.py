import os


RESULTS_PATH = os.environ.get('RESULTS_PATH', './results/')
CLASSIFIER_PATH = os.environ.get('CLASSIFIER_PATH', './classifiers/')
DATA_PATH = os.environ.get('DATA_PATH', './data/')

DATASET_FILE = {
    'mnli': 'input_mnli.csv'
}

METADATA_FILE = {
    'mnli': 'metadata_mnli.csv'
}

DATASET_SPLITS = {
    'train': 0,
    'val': 1,
    'test': 2,
}

DATASET_CLASSES = {
    'mnli': ['fiction', 'government', 'slate', 'telephone', 'travel']
}

SEED = 5454

SPLITS = ['train', 'val']
