import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
import utils


"""
Step 1: Train an ERM (Empirical Risk Minimization) linear classifier.

Trains a temperature-scaled linear layer on top of frozen embeddings from Step 0.
The classifier weights are initialized from the class name embeddings (giving the
model a zero-shot starting point), then fine-tuned with cross-entropy loss.

The --only_spurious flag restricts the training set to samples where the class
label matches the known spurious attribute, amplifying spurious features in the
learned weight vectors (which Step 2 will then detect).

Usage:
    python step_1_erm.py --dataset Waterbirds --only_spurious
    python step_1_erm.py --dataset CelebA --device cpu
"""


def train_erm(dataset: str, device_str: str, batch_size: int, only_spurious: bool):
    # Reproducibility
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(device_str)
    ending = "_only_spurious" if only_spurious else ""

    # Load cached embeddings
    datasets = utils.load_embedding_splits(dataset, only_spurious=only_spurious)
    train_labels = datasets["train"].labels
    classes = np.unique(train_labels)
    n_classes = len(classes)
    input_dim = datasets["train"].embeddings.shape[-1]

    # Class-balanced weights for the loss
    class_weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Initialize classifier from class name embeddings
    class_emb_path = os.path.join(config.CACHE_PATH, "model_outputs", dataset, "class_embeddings.npy")
    class_emb = torch.tensor(np.load(class_emb_path))
    class_emb = class_emb / class_emb.norm(p=2, dim=1, keepdim=True)

    classifier = utils.TemperatureScaledLinear(input_dim, n_classes)
    classifier.temperature.requires_grad_(False)
    with torch.no_grad():
        classifier.weight.data.copy_(class_emb)

    # Dataloaders
    loaders = {
        split: DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == "train"))
        for split in config.TRAIN_VAL_SPLITS
    }

    # Output paths
    output_dir = os.path.join(config.CACHE_PATH, "classifiers", dataset)
    os.makedirs(output_dir, exist_ok=True)
    classifier_path = os.path.join(output_dir, f"erm_classifier{ending}.pt")

    # Training loop with early stopping
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_acc = 0.0
    patience, no_improve = 5, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    pbar = tqdm(desc="ERM Training")
    while True:
        train_loss, train_acc = utils.train_one_epoch(classifier, device, loaders["train"], loss_fn, optimizer)
        val_loss, val_acc = utils.evaluate(classifier, device, loaders["val"], loss_fn)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        pbar.set_postfix(train_acc=f"{train_acc:.4f}", val_acc=f"{val_acc:.4f}")
        pbar.update(1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(classifier.state_dict(), classifier_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    pbar.close()

    print(f"\nBest validation balanced accuracy: {best_val_acc:.4f}")
    print(f"Classifier saved to: {classifier_path}")

    # Save confusion matrix on val set
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    classifier.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in loaders["val"]:
            logits = classifier(batch_data.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(batch_labels.numpy())

    cm = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_preds))
    cm_path = os.path.join(output_dir, f"erm_confusion_matrices{ending}.npy")
    np.save(cm_path, {"val": cm})

    # Save training metrics
    torch.save(history, os.path.join(output_dir, f"erm_metrics{ending}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Train ERM classifier")
    parser.add_argument("--dataset", type=str, default="Waterbirds", choices=list(config.DATASETS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--only_spurious", action="store_true")
    args = parser.parse_args()

    train_erm(args.dataset, args.device, args.batch_size, args.only_spurious)
