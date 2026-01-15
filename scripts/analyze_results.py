import pandas as pd
import os
import torch

CACHE_PATH = './cache'
DATASET = 'MNLI'
CLASSES = ['correct', 'incorrect']
biases_dir = os.path.join(CACHE_PATH, 'biases', DATASET)


def load_all_rankings():
    rankings = {}
    max_gap_path = os.path.join(biases_dir, 'max_gap_ranking.csv')
    if os.path.exists(max_gap_path):
        rankings['max_gap'] = pd.read_csv(max_gap_path)
    for cls in CLASSES:
        cls_path = os.path.join(biases_dir, f'{cls}_ranking.csv')
        if os.path.exists(cls_path):
            rankings[cls] = pd.read_csv(cls_path)
    return rankings


def load_detected_biases():
    biases = {}
    for cls in CLASSES:
        bias_path = os.path.join(biases_dir, f'{cls}_biases.pt')
        if os.path.exists(bias_path):
            bias_data = torch.load(bias_path, weights_only=False)
            biases[cls] = bias_data['biases']
    return biases


def load_erm_metrics():
    metrics_path = os.path.join(CACHE_PATH, DATASET, 'erm_metrics.pt')
    if os.path.exists(metrics_path):
        return torch.load(metrics_path, weights_only=False)
    return None


def check_filtered_keywords():
    keywords = {}
    processed_keywords_path = os.path.join(CACHE_PATH, 'keywords', DATASET, 'filtered_keywords.pt')
    if os.path.exists(processed_keywords_path):
        keywords = torch.load(processed_keywords_path)
    return keywords


def print_str(str):
    print(f"\n" + "="*80)
    print(f"{str}")
    print("="*80)


def print_biases(rankings, biases, cls):
    if cls in biases.keys():
        cls_ranking = rankings[cls]
        print_str(f"Biases for class {str(cls).upper()}")
        for i, kw in enumerate(biases[cls], 1):
            row = cls_ranking[cls_ranking['bias'] == kw]
            score = row['score'].values[0]
            print(f"{i:4d}. {kw} (score: {score:.4f})")


def print_keywords():
    keywords = check_filtered_keywords()

    print(f"Keywords after LLM-filtering: {len(keywords['llm'])}")
    print(f"Keywords after WordNet-filtering: {len(keywords['clean'])}")

    print_str("LLM KEYWORDS:")
    for i, kw in enumerate(sorted(keywords['llm']), 1):
        print(f"{i:4d}. {kw}")

    print_str("WORDNET KEYWORDS:")
    for i, kw in enumerate(sorted(keywords['clean']), 1):
        print(f"{i:4d}. {kw}")

    print_str("KEYWORDS REMOVED BY WORDNET:")
    removed = set(keywords['llm']) - set(keywords['clean'])
    if removed:
        for i, kw in enumerate(sorted(removed), 1):
            print(f"{i:4d}. {kw}")
    else:
        print("(None - all LLM keywords passed WordNet filtering)")   


def max_gap_biases(rankings, n=50):
    df = rankings['max_gap'].head(n)
    for i, row in df.iterrows():
        class_sims = {cls: row[cls] for cls in CLASSES}
        max_cls = max(class_sims, key=class_sims.get)
        min_cls = min(class_sims, key=class_sims.get)
        
        print(f"{i+1:2d}. '{row['bias']}'")
        print(f"    Score: {row['score']:.4f} | Most associated: {max_cls} ({row[max_cls]:.3f})")
        print(f"    Least associated: {min_cls} ({row[min_cls]:.3f})")
        print()


def main():
    # rankings = load_all_rankings()
    # biases = load_detected_biases()
    erm_metrics = load_erm_metrics()
    if erm_metrics:
        print("\n--- ERM TRAINING SUMMARY ---")
        print(f"  Final Training Accuracy: {erm_metrics['train_accs'][-1]:.4f}")
        print(f"  Final Validation Accuracy: {erm_metrics['val_accs'][-1]:.4f}")
        print(f"  Best Validation Accuracy: {max(erm_metrics['val_accs']):.4f}")
        print(f"  Training Epochs: {len(erm_metrics['train_accs'])}")
        print("\n")
    # max_gap_biases(rankings)
    # print_biases(rankings, biases, "computer science")


if __name__ == '__main__':
    main()