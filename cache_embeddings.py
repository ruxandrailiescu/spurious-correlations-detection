import pandas as pd
import os
import constants
import torch
from models import SentenceEncoder
import numpy as np
import fire


def cache_embeddings_text(dataset: str = 'MMLUPro', device: str = 'cuda'):
    path = os.path.join(constants.DATA_PATH,
                        constants.DATASET_DIR[dataset],
                        constants.DATASET_FILE[dataset])
    df = pd.read_csv(path)
    texts = df['text'].to_numpy()
    
    metadata_path = os.path.join(constants.DATA_PATH,
                                 constants.DATASET_DIR[dataset],
                                 constants.METADATA_FILE[dataset])
    mtd = pd.read_csv(metadata_path)
    split_labels = mtd.split.to_numpy()

    device = torch.device(device)
    model = SentenceEncoder()
    model.eval()
    model.to(device)

    output_dir = os.path.join(constants.CACHE_PATH,
                              'model_outputs',
                              dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # cache data embeddings
    for split in constants.SPLITS:
        output_path = os.path.join(output_dir, f'{split}.npy')
        if not os.path.isfile(output_path):
            split_texts = texts[split_labels == constants.DATASET_SPLITS[split]]
            split_embeddings = model.encode_texts_batched(split_texts, device, bs=128).numpy()
            np.save(output_path, split_embeddings)

    # cache class name embeddings
    class_names = constants.DATASET_CLASSES[dataset]
    class_embeddings = model.encode_texts_batched(class_names, device, bs=128).numpy()
    class_embeddings_path = os.path.join(output_dir, 'class_embeddings.npy')
    np.save(class_embeddings_path, class_embeddings)


if __name__ == '__main__':
    fire.Fire(cache_embeddings_text)