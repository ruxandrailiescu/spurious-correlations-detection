import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
import config
import utils

logging.basicConfig(level=logging.INFO)


"""
Step 2a: Extract keywords and filter out class-related concepts.

1. Extracts candidate keywords from training data (captions for image datasets,
   raw text for text datasets) using YAKE.
2. Filters keywords to remove class-related concepts using a rule-based approach
   (string matching + lemmatization + WordNet hypernym/hyponym filtering).
3. Encodes the filtered keywords and saves their embeddings.

Usage:
    python step_2a_filter.py --dataset Waterbirds --only_spurious
    python step_2a_filter.py --dataset CivilComments --device cpu
"""


def extract_and_filter_keywords(dataset: str, device_str: str, only_spurious: bool):
    device = torch.device(device_str)
    ds_config = config.DATASETS[dataset]
    ending = "_only_spurious" if only_spurious else ""

    # Extract keywords per class using YAKE
    kw_cache_dir = os.path.join(config.CACHE_PATH, "keywords", dataset)
    os.makedirs(kw_cache_dir, exist_ok=True)
    kw_cache_file = os.path.join(kw_cache_dir, f"keywords{ending}.npy")

    if os.path.isfile(kw_cache_file):
        logging.info(f"Loading cached keywords from {kw_cache_file}")
        kws_dict = np.load(kw_cache_file, allow_pickle=True).item()
        class_keywords = [kws_dict[i] for i in range(len(ds_config["classes"]))]
    else:
        logging.info("Extracting keywords with YAKE...")
        metadata = pd.read_csv(config.get_metadata_path(dataset))
        train_mask = metadata.split == config.SPLIT_IDS["train"]
        train_labels = metadata.y[train_mask].to_numpy()
        train_envs = metadata.a[train_mask].to_numpy()

        # Load the training texts (captions or raw text)
        texts = np.array(utils.load_training_texts(dataset))

        class_keywords = []
        for cls_idx in range(len(ds_config["classes"])):
            if only_spurious and ds_config["class_bias"] is not None:
                mask = (train_labels == cls_idx) & (train_envs == ds_config["class_bias"][cls_idx])
            else:
                mask = train_labels == cls_idx
            class_texts = texts[mask].tolist()

            ngram_sizes = [3] if config.is_text_dataset(dataset) else [3, 5]
            kws = utils.extract_keywords(class_texts, ngram_sizes=ngram_sizes, top=256)
            class_keywords.append(kws)
            logging.info(f"  Class '{ds_config['classes'][cls_idx]}': {len(kws)} keywords")

        # Cache
        kws_dict = {i: kws for i, kws in enumerate(class_keywords)}
        np.save(kw_cache_file, kws_dict)

    # Save extracted n-grams per class to CSV
    ngrams_csv_path = os.path.join(kw_cache_dir, f"extracted_ngrams{ending}.csv")
    rows = []
    for cls_idx, kws in enumerate(class_keywords):
        cls_name = ds_config["classes"][cls_idx]
        for kw in kws:
            rows.append({"class": cls_name, "keyword": kw})
    pd.DataFrame(rows).to_csv(ngrams_csv_path, index=False)
    logging.info(f"Saved extracted n-grams to {ngrams_csv_path}")

    # Merge all class keywords into a global set
    all_keywords = list(set(sum(class_keywords, [])))
    logging.info(f"Total unique keywords before filtering: {len(all_keywords)}")

    # Filter out class-related keywords
    filtered_kw_path = os.path.join(kw_cache_dir, f"filtered_keywords{ending}.pt")

    if os.path.isfile(filtered_kw_path):
        logging.info(f"Loading cached filtered keywords from {filtered_kw_path}")
        saved = torch.load(filtered_kw_path)
        clean_keywords = saved["clean"]
    else:
        logging.info("Filtering keywords (rule-based: string match + WordNet)...")
        clean_keywords = utils.filter_keywords_rule_based(
            keywords=all_keywords,
            class_names=ds_config["classes"],
            wn_class_names=ds_config["classes_wn"],
            class_description=ds_config["classes_explicit"],
        )
        # Cache both the clean set and what was filtered out for inspection
        torch.save({"clean": clean_keywords, "all": all_keywords}, filtered_kw_path)

    logging.info(f"Keywords after filtering: {len(clean_keywords)}")
    logging.info(f"Filtered out: {len(all_keywords) - len(clean_keywords)} keywords")

    # Save remaining (clean) keywords to CSV
    remaining_csv_path = os.path.join(kw_cache_dir, f"remaining_keywords{ending}.csv")
    pd.DataFrame({"keyword": clean_keywords}).to_csv(remaining_csv_path, index=False)
    logging.info(f"Saved remaining keywords to {remaining_csv_path}")

    # Encode filtered keywords
    logging.info("Encoding filtered keywords...")
    encoder = utils.get_text_encoder(dataset)
    keyword_embeddings = encoder.encode_texts(clean_keywords, device)

    # Save keywords + embeddings
    output_dir = os.path.join(config.CACHE_PATH, "biases", dataset)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {"keywords": clean_keywords, "keywords_embeddings": keyword_embeddings},
        os.path.join(output_dir, f"filtered_keywords_and_embeddings{ending}.pt"),
    )

    # Save filtered-out keywords as CSV for manual inspection
    filtered_out = list(set(all_keywords) - set(clean_keywords))
    pd.DataFrame({"keyword": filtered_out}).to_csv(
        os.path.join(output_dir, f"filtered_out_keywords{ending}.csv"), index=False
    )

    logging.info("Done. Keywords and embeddings cached.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2a: Extract and filter keywords")
    parser.add_argument("--dataset", type=str, default="Waterbirds", choices=list(config.DATASETS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--only_spurious", action="store_true")
    args = parser.parse_args()

    extract_and_filter_keywords(args.dataset, args.device, args.only_spurious)
