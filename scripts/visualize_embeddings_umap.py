import constants
import fire
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import umap


def visualize_weights_and_biases(
    dataset,
    exp_number,
    only_spurious=False,
    top_n_biases=50,
    label_top_n=10,
    n_neighbors=15,
    min_dist=0.1,
):
    """
    UMAP visualization of classifier weights and bias embeddings.

    Args:
        dataset: Dataset name
        exp_number: Experiment number
        only_spurious: Whether to use only_spurious variant
        top_n_biases: Number of top biases to plot
        label_top_n: Number of top biases to label on the plot
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
    """
    ending = "_only_spurious" if only_spurious else ""
    dataset = str(dataset).lower()
    exp_dir = f"{dataset}_exp{exp_number}"

    # Load classifier weights
    classifier_path = os.path.join(
        constants.CLASSIFIER_PATH,
        dataset,
        f'erm_classifier{ending}.pt'
    )
    state_dict = torch.load(classifier_path, weights_only=True, map_location=torch.device('cpu'))
    weights = state_dict['weight'].cpu().numpy()  # shape: [n_classes, embedding_dim]

    # Load bias embeddings
    bias_path = os.path.join(
        constants.RESULTS_PATH,
        exp_dir,
        f'filtered_keywords_and_embeddings{ending}.pt',
    )
    bias_dict = torch.load(bias_path, map_location=torch.device('cpu'))
    bias_embeddings = bias_dict['keywords_embeddings'].cpu().numpy()  # shape: [n_biases, embedding_dim]
    keywords = np.array(bias_dict['keywords'])

    # L2 normalize
    weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    bias_embeddings = bias_embeddings / np.linalg.norm(bias_embeddings, axis=1, keepdims=True)

    # Load ranking to get max_class for each bias
    ranking_path = os.path.join(
        constants.RESULTS_PATH,
        exp_dir,
        f'bias_global_ranking{ending}.csv'
    )
    ranking_df = pd.read_csv(ranking_path)

    # Get top N biases from ranking
    top_biases = ranking_df.head(top_n_biases)
    top_bias_names = top_biases['bias'].tolist()
    top_bias_classes = top_biases['max_class'].tolist()

    # Get embeddings for top biases (maintain ranking order)
    bias_name_to_idx = {name: i for i, name in enumerate(keywords)}
    top_bias_indices = [bias_name_to_idx[name] for name in top_bias_names if name in bias_name_to_idx]
    top_bias_embeddings = bias_embeddings[top_bias_indices]

    # Filter to only biases we found
    found_biases = [name for name in top_bias_names if name in bias_name_to_idx]
    found_classes = [cls for name, cls in zip(top_bias_names, top_bias_classes) if name in bias_name_to_idx]

    # Combine weights and bias embeddings for UMAP
    all_embeddings = np.vstack([weights, top_bias_embeddings])

    # Apply UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=constants.SEED,
    )
    all_2d = reducer.fit_transform(all_embeddings)

    weights_2d = all_2d[:len(weights)]
    biases_2d = all_2d[len(weights):]

    # Get class names and create color map
    class_names = constants.DATASET_CLASSES[dataset]
    n_classes = len(class_names)
    cmap = plt.cm.get_cmap('tab10', n_classes)
    class_to_color = {cls: cmap(i) for i, cls in enumerate(class_names)}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot bias embeddings (colored by max_class)
    for i, (x, y) in enumerate(biases_2d):
        cls = found_classes[i]
        color = class_to_color[cls]
        ax.scatter(x, y, c=[color], s=30, alpha=0.6)

        # Label top N biases
        if i < label_top_n:
            ax.annotate(found_biases[i], (x, y), fontsize=8, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')

    # Plot classifier weights (larger markers with labels)
    for i, (x, y) in enumerate(weights_2d):
        cls = class_names[i]
        color = class_to_color[cls]
        ax.scatter(x, y, c=[color], s=200, marker='*', edgecolors='black', linewidths=1.5)
        ax.annotate(f'{cls.upper()}', (x, y), fontsize=12, fontweight='bold',
                   xytext=(10, 10), textcoords='offset points')

    # Create legend
    legend_elements = [plt.scatter([], [], c=[class_to_color[cls]], s=100, label=cls)
                       for cls in class_names]
    ax.legend(handles=legend_elements, title='Class', loc='upper right')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'{dataset}: Classifier Weights and Top {top_n_biases} Bias Embeddings (UMAP)\n(Stars = class weights, dots = bias keywords)')

    # Save plot
    output_dir = os.path.join(constants.RESULTS_PATH, exp_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'weights_and_biases_umap{ending}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {output_path}")


if __name__ == '__main__':
    fire.Fire(visualize_weights_and_biases)
