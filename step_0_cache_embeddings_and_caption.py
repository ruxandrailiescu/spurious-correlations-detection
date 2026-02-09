import argparse
import logging
import os
import numpy as np
import torch
import config
import utils
import pandas as pd
import open_clip

logging.basicConfig(level=logging.INFO)


"""
Step 0: Cache embeddings and caption images.

For image datasets (Waterbirds, CelebA, ISIC):
  - Encode all train/val images through CLIP's vision encoder → save as .npy
  - Generate captions for training images using Microsoft GIT
  - Encode class names through CLIP's text encoder → save as .npy

For text datasets (CivilComments):
  - Encode all train/val texts through SentenceTransformer → save as .npy
  - Encode class names through SentenceTransformer → save as .npy

Usage:
    python step_0_cache_embeddings_and_caption.py --dataset Waterbirds
    python step_0_cache_embeddings_and_caption.py --dataset CivilComments --device cpu
"""


def cache_embeddings_and_caption(dataset: str, clip_model: str, device_str: str):
    device = torch.device(device_str)
    ds_config = config.DATASETS[dataset]
    class_names = ds_config["classes"]

    output_dir = os.path.join(config.CACHE_PATH, "model_outputs", dataset)
    os.makedirs(output_dir, exist_ok=True)

    if config.is_text_dataset(dataset):
        text_file, text_column = ds_config["text_file"]
        csv_path = os.path.join(config.DATA_PATH, ds_config["directory"], text_file)
        metadata = pd.read_csv(config.get_metadata_path(dataset))
        df = pd.read_csv(csv_path)
        texts = df[text_column].to_numpy()
        split_labels = metadata.split.to_numpy()

        encoder = utils.SentenceEncoder()
        encoder.eval().to(device)

        for split in config.TRAIN_VAL_SPLITS:
            out_path = os.path.join(output_dir, f"{split}.npy")
            if os.path.isfile(out_path):
                logging.info(f"Embeddings already cached: {out_path}")
                continue
            split_texts = texts[split_labels == config.SPLIT_IDS[split]]
            embeddings = encoder.encode_texts(split_texts.tolist(), device).numpy()
            np.save(out_path, embeddings)
            logging.info(f"Saved {split} embeddings in {out_path}")

        # Cache class embeddings
        class_emb = encoder.encode_texts(class_names, device).numpy()

    else:
        clip_version, clip_arch = clip_model.split("/")
        model, _, preprocess = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_version)
        model.eval().to(device)

        for split in config.TRAIN_VAL_SPLITS:
            out_path = os.path.join(output_dir, f"{split}.npy")
            if os.path.isfile(out_path):
                logging.info(f"Embeddings already cached: {out_path}")
                continue
            ds = utils.ImageDataset(dataset, split=split, transform=preprocess)
            embeddings = utils.compute_image_embeddings(model.encode_image, ds, device)
            np.save(out_path, embeddings)
            logging.info(f"Saved {split} embeddings in {out_path}")

        # Caption training images
        utils.caption_images(dataset, device)

        # Save image paths + captions to CSV
        captions_csv_path = os.path.join(config.CACHE_PATH, "captions", dataset, "captions.csv")
        captions_txt_path = os.path.join(config.CACHE_PATH, "captions", dataset, "train.txt")
        with open(captions_txt_path, "r") as f:
            captions = f.read().split("\n")
        metadata = pd.read_csv(config.get_metadata_path(dataset))
        train_filenames = metadata.filename[metadata.split == config.SPLIT_IDS["train"]].to_numpy()
        image_dir = ds_config["image_dir"]
        relative_paths = [os.path.join(image_dir, p) for p in train_filenames]
        pd.DataFrame({"image_path": relative_paths, "caption": captions}).to_csv(
            captions_csv_path, index=False
        )
        logging.info(f"Saved captions CSV to {captions_csv_path}")

        # Cache class name embeddings via CLIP text encoder
        tokenizer = open_clip.get_tokenizer(clip_arch)
        with torch.no_grad():
            tokens = tokenizer(class_names).to(device)
            class_emb = model.encode_text(tokens).cpu().numpy()

    class_emb_path = os.path.join(output_dir, "class_embeddings.npy")
    np.save(class_emb_path, class_emb)
    logging.info(f"Saved class embeddings in {class_emb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 0: Cache embeddings and captions")
    parser.add_argument("--dataset", type=str, default="Waterbirds", choices=list(config.DATASETS.keys()))
    parser.add_argument("--clip", type=str, default="openai/ViT-L-14", help="CLIP model (version/architecture)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cache_embeddings_and_caption(args.dataset, args.clip, args.device)
