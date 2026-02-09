import argparse
import os
import numpy as np
import pandas as pd
import torch
import config
import utils


"""
Step 2b/c: Rank keywords by spuriousness and apply thresholding.

For each filtered keyword from Step 2a:
  - Compute its cosine similarity with every class weight vector from the
    ERM classifier (Step 1).
  - Compute the "gap" = similarity to the most-aligned class minus the
    similarity to the second-most-aligned class.
  - A large gap means this keyword is strongly biased toward one class,
    making it a likely spurious correlation.

Keywords are ranked by gap (descending) and saved as a CSV.

Usage:
    python step_2bc_rank_and_threshold.py --dataset Waterbirds --only_spurious
    python step_2bc_rank_and_threshold.py --dataset CelebA --device cpu
"""


def rank_keywords(dataset: str, device_str: str, only_spurious: bool):
    device = torch.device(device_str)
    ending = "_only_spurious" if only_spurious else ""
    class_names = np.array(config.get_classes(dataset))

    # Load trained ERM classifier weights
    classifier_path = os.path.join(
        config.CACHE_PATH, "classifiers", dataset, f"erm_classifier{ending}.pt"
    )
    state_dict = torch.load(classifier_path, weights_only=True)
    weights = state_dict["weight"].to(device)  # shape: [n_classes, emb_dim]

    # Load keyword embeddings
    kw_path = os.path.join(
        config.CACHE_PATH, "biases", dataset,
        f"filtered_keywords_and_embeddings{ending}.pt",
    )
    kw_dict = torch.load(kw_path)
    keywords = np.array(kw_dict["keywords"])
    kw_embeddings = kw_dict["keywords_embeddings"].to(device)
    kw_embeddings = kw_embeddings / kw_embeddings.norm(p=2, dim=1, keepdim=True)

    # Compute similarity and gap
    with torch.no_grad():
        # similarities[c, k] = cosine similarity between class c weight and keyword k
        similarities = (weights @ kw_embeddings.T).cpu().numpy()

    n_keywords = similarities.shape[1]

    # For each keyword, find the two most similar classes
    sorted_classes = np.argsort(similarities, axis=0)
    max_cls = sorted_classes[-1]       # index of most-similar class per keyword
    second_cls = sorted_classes[-2]    # index of second-most-similar class

    max_sim = similarities[max_cls, np.arange(n_keywords)]
    second_sim = similarities[second_cls, np.arange(n_keywords)]
    gap = max_sim - second_sim

    # Sort keywords by gap (highest = most spuriously biased)
    ranking = np.argsort(gap)[::-1]

    # Count keyword occurrences in training data
    training_texts = utils.load_training_texts(dataset)
    counts = utils.count_keyword_occurrences(keywords.tolist(), training_texts)

    # Build and save ranking dataframe
    df = pd.DataFrame({
        "bias": keywords[ranking],
        "score": gap[ranking],
        "max_class": class_names[max_cls[ranking]],
        "max_class_score": max_sim[ranking],
        "second_max_class": class_names[second_cls[ranking]],
        "second_max_class_score": second_sim[ranking],
        "count": counts[ranking],
    })

    output_dir = os.path.join(config.CACHE_PATH, "biases", dataset)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"global_ranking{ending}.csv")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2b/c: Rank and threshold keywords")
    parser.add_argument("--dataset", type=str, default="Waterbirds", choices=list(config.DATASETS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--only_spurious", action="store_true")
    args = parser.parse_args()

    rank_keywords(args.dataset, args.device, args.only_spurious)
