import logging
import os
import re
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yake
from nltk.corpus import stopwords, wordnet as wn
from PIL import Image, ImageFile
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import config


# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Download NLTK data (only downloads if not already present)
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)


# Datasets

class ImageDataset(Dataset):
    """Loads images from disk with their labels for a given dataset and split."""

    def __init__(self, dataset: str, split: str, transform=None):
        self.name = dataset
        self.split = split
        self.transform = transform

        ds = config.DATASETS[dataset]
        self.root_dir = os.path.join(config.DATA_PATH, ds["directory"], ds["image_dir"])
        metadata = pd.read_csv(config.get_metadata_path(dataset))

        mask = metadata.split == config.SPLIT_IDS[split]
        self.paths = metadata.filename[mask].to_numpy()
        self.labels = metadata.y[mask].to_numpy(dtype=np.int32)
        self.envs = metadata.a[mask].to_numpy(dtype=np.int32) if "a" in metadata.columns else np.zeros_like(self.labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.paths[index])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return index, image, self.envs[index], self.labels[index]


class EmbDataset(Dataset):
    """Simple dataset wrapping pre-computed embeddings and labels."""

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# Model wrappers

class CLIPTextEncoder(nn.Module):
    """Wraps OpenCLIP for batched text encoding."""

    def __init__(self, architecture="ViT-L-14", pretrained="openai"):
        super().__init__()
        import open_clip
        self.clip, _, _ = open_clip.create_model_and_transforms(architecture, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(architecture)
        del self.clip.visual  # we only need the text encoder

    def encode_texts(self, texts: list[str], device, batch_size=128):
        self.to(device)
        self.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[i : i + batch_size]).to(device)
                emb = self.clip.encode_text(tokens).cpu()
                all_embeddings.append(emb)
        return torch.cat(all_embeddings, dim=0)


class SentenceEncoder(nn.Module):
    """Wraps sentence-transformers for batched text encoding."""

    def __init__(self, model_name="Alibaba-NLP/gte-large-en-v1.5"):
        super().__init__()
        import sentence_transformers
        self.encoder = sentence_transformers.SentenceTransformer(model_name, trust_remote_code=True)

    def encode_texts(self, texts: list[str], device, batch_size=128):
        self.to(device)
        self.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                emb = self.encoder.encode(
                    texts[i : i + batch_size], convert_to_tensor=True, device=device
                ).cpu()
                all_embeddings.append(emb)
        return torch.cat(all_embeddings, dim=0)


class TemperatureScaledLinear(nn.Linear):
    """Linear layer whose logits are scaled by exp(temperature)."""

    def __init__(self, in_features, out_features, init_temp=None):
        super().__init__(in_features, out_features, bias=False)
        if init_temp is None:
            init_temp = config.TEMP_INIT
        self.temperature = nn.Parameter(torch.tensor(init_temp, dtype=torch.float32))

    def forward(self, x):
        return super().forward(x) * self.temperature.exp()


# Embedding helpers

def normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row of an embedding matrix."""
    norms = np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))
    return embeddings / norms


def compute_image_embeddings(model_encode_fn, dataset, device, batch_size=128):
    """Run a model's encode function over a dataset and return embeddings as numpy."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    indices_list, emb_list = [], []
    with torch.no_grad():
        for batch_indices, batch_imgs, _, _ in tqdm(dataloader, desc="Encoding"):
            emb = model_encode_fn(batch_imgs.to(device)).cpu().numpy()
            emb_list.append(emb)
            indices_list.append(batch_indices.numpy())
    indices = np.concatenate(indices_list)
    embeddings = np.concatenate(emb_list)
    # Sort by original index in case dataloader shuffled
    embeddings = embeddings[np.argsort(indices)]
    return embeddings


def get_text_encoder(dataset: str):
    """Return the appropriate text encoder for the dataset type."""
    if config.is_text_dataset(dataset):
        return SentenceEncoder()
    return CLIPTextEncoder()


# Captioning (for image datasets)

def caption_images(dataset: str, device):
    """Generate captions for training images using Microsoft GIT."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    output_dir = os.path.join(config.CACHE_PATH, "captions", dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.txt")

    if os.path.exists(output_path):
        logging.info(f"Captions already cached at {output_path}")
        return

    checkpoint = "microsoft/git-large-coco"
    logging.info(f"Loading captioning model: {checkpoint}")
    processor = AutoProcessor.from_pretrained(checkpoint)
    preprocess = lambda x: processor(images=x, return_tensors="pt").pixel_values[0]
    model = AutoModelForCausalLM.from_pretrained(checkpoint).eval().to(device)

    ds = ImageDataset(dataset, split="train", transform=preprocess)
    dataloader = DataLoader(ds, batch_size=32, shuffle=False)

    with open(output_path, "w") as f:
        first_batch = True
        for _, images, _, _ in tqdm(dataloader, desc="Captioning"):
            if not first_batch:
                f.write("\n")
            first_batch = False
            with torch.no_grad():
                output = model.generate(pixel_values=images.to(device), max_length=50)
            captions = processor.batch_decode(output, skip_special_tokens=True)
            f.write("\n".join(captions))

    logging.info(f"Saved captions to {output_path}")


# Dataset loading (for training on cached embeddings)

def load_embedding_splits(dataset: str, only_spurious=False):
    """
    Load cached embeddings for train/val splits and return as EmbDataset dicts.

    If only_spurious=True, the training set is restricted to samples where the
    class label aligns with the known spurious attribute (e.g., waterbirds on water).
    """
    emb_dir = os.path.join(config.CACHE_PATH, "model_outputs", dataset)
    metadata = pd.read_csv(config.get_metadata_path(dataset))

    datasets = {}
    for split in config.TRAIN_VAL_SPLITS:
        emb_path = os.path.join(emb_dir, f"{split}.npy")
        embeddings = normalize(np.load(emb_path))
        mask = metadata.split == config.SPLIT_IDS[split]
        labels = metadata.y[mask].to_numpy()

        if only_spurious and split != "test":
            class_bias = config.DATASETS[dataset]["class_bias"]
            envs = metadata.a[mask].to_numpy()
            keep = np.zeros(len(labels), dtype=bool)
            for cls, bias_att in enumerate(class_bias):
                keep |= (labels == cls) & (envs == bias_att)
            embeddings = embeddings[keep]
            labels = labels[keep]

        datasets[split] = EmbDataset(embeddings, labels)

    return datasets


# Training / Validation loops

def train_one_epoch(model, device, dataloader, criterion, optimizer):
    """Single epoch of training. Returns (avg_loss, balanced_accuracy)."""
    model.to(device).train()
    total_loss, n_batches = 0.0, 0
    all_labels, all_preds = [], []

    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.long().to(device)

        logits = model(batch_data)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Keep weights normalized (as in original)
        with torch.no_grad():
            model.weight.data /= model.weight.data.norm(p=2, dim=1, keepdim=True)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(batch_labels.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return total_loss / n_batches, balanced_accuracy_score(all_labels, all_preds)


def evaluate(model, device, dataloader, criterion):
    """Evaluate model. Returns (avg_loss, balanced_accuracy)."""
    model.to(device).eval()
    total_loss, n_batches = 0.0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.long().to(device)

            logits = model(batch_data)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return total_loss / n_batches, balanced_accuracy_score(all_labels, all_preds)


# Keyword extraction (YAKE)

def extract_keywords(texts: list[str], ngram_sizes: list[int] = None, top=256):
    """
    Extract keywords from a collection of texts using YAKE.

    Args:
        texts: List of text strings to extract keywords from.
        ngram_sizes: List of max n-gram sizes for YAKE. Default: [3, 5].
        top: Max number of keywords per n-gram size.
    """
    if ngram_sizes is None:
        ngram_sizes = [3, 5]

    all_text = " .\n".join(str(t) for t in texts)
    stop_words = set(stopwords.words("english"))
    all_keywords = []

    for n in ngram_sizes:
        extractor = yake.KeywordExtractor(lan="en", n=n, dedupLim=0.9, top=top, features=None)
        keywords = [kw for kw, _ in extractor.extract_keywords(all_text)]
        all_keywords.extend(keywords)

    # Clean and deduplicate
    all_keywords = [k.replace(".", "").strip() for k in all_keywords]
    all_keywords = list(set(all_keywords) - stop_words)
    return all_keywords


# Keyword filtering (rule-based)

def _get_wordnet_related(reference_word: str, pos="n") -> set:
    """Get all hypernyms and hyponyms of a reference word from WordNet."""
    related = set()
    synsets = wn.synsets(reference_word, pos=pos) if pos else wn.synsets(reference_word)
    for synset in synsets:
        # Add all hyponyms (more specific words)
        related |= set(synset.closure(lambda x: x.hyponyms()))
        # Add all hypernyms (more general words)
        related |= set(synset.closure(lambda x: x.hypernyms()))
        related.add(synset)
    return related


def filter_keywords_rule_based(
    keywords: list[str],
    class_names: list[str],
    wn_class_names: list[str],
    class_description: str,
) -> list[str]:
    """
    Filter keywords to remove those directly related to the class concepts.

    1. **Direct string match**: Remove keywords that contain any class name
       word or word from the explicit class description.
    2. **Lemmatization match**: Same check but on lemmatized forms.
    3. **WordNet relation check**: Remove keywords whose words are hypernyms
       or hyponyms of the class reference words.

    Args:
        keywords: Raw keywords from YAKE.
        class_names: Class label strings (e.g. ["landbird", "waterbird"]).
        wn_class_names: WordNet reference words for each class.
        class_description: Human description of class concepts to filter.

    Returns:
        Filtered list of keywords (class-unrelated concepts only).
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    all_stop_words = set(stopwords.words("english")) | config.STOPWORDS_EXTRA

    # Build a set of "forbidden" words from class names and their description
    forbidden_words = set()
    for cls in class_names:
        for token in nltk.word_tokenize(cls.lower()):
            if token not in all_stop_words:
                forbidden_words.add(lemmatizer.lemmatize(token))

    # Also add words from the explicit description
    for token in nltk.word_tokenize(class_description.lower()):
        if token not in all_stop_words and len(token) > 2:
            forbidden_words.add(lemmatizer.lemmatize(token))

    # Build WordNet related synsets for each class reference word
    wn_related = set()
    for wn_cls in wn_class_names:
        wn_related |= _get_wordnet_related(wn_cls, pos="n")
        wn_related |= _get_wordnet_related(wn_cls, pos=None)

    clean_keywords = []
    for kw in keywords:
        tokens = nltk.word_tokenize(kw.lower())
        lemmas = [lemmatizer.lemmatize(t) for t in tokens]
        content_lemmas = [l for l in lemmas if l not in all_stop_words]

        # Layer 1 & 2: Check if any content word matches a forbidden word
        if any(lemma in forbidden_words for lemma in content_lemmas):
            continue

        # Layer 3: Check WordNet relations for each content word
        is_related = False
        for word in content_lemmas:
            word_synsets = wn.synsets(word)
            if any(ws in wn_related for ws in word_synsets):
                is_related = True
                break
        if is_related:
            continue

        clean_keywords.append(kw)

    return clean_keywords


# Keyword occurrence counting (for ranking step)

def count_keyword_occurrences(keywords: list[str], texts: list[str]) -> np.ndarray:
    """Count how many training texts contain each keyword (word-boundary match)."""
    all_text = " ".join(texts).lower()
    counts = []
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        counts.append(len(re.findall(pattern, all_text)))
    return np.array(counts)


def load_training_texts(dataset: str) -> list[str]:
    """Load training texts: captions for image datasets, raw text for text datasets."""
    if config.is_text_dataset(dataset):
        text_file, text_column = config.DATASETS[dataset]["text_file"]
        csv_path = os.path.join(config.DATA_PATH, config.DATASETS[dataset]["directory"], text_file)
        metadata = pd.read_csv(config.get_metadata_path(dataset))
        df = pd.read_csv(csv_path)
        return df[text_column][metadata.split == config.SPLIT_IDS["train"]].tolist()
    else:
        captions_path = os.path.join(config.CACHE_PATH, "captions", dataset, "train.txt")
        with open(captions_path, "r") as f:
            return f.read().split("\n")
