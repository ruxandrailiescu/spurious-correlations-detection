import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


cm_dict = np.load('./cache/cache/classifiers/MMMU/erm_confusion_matrices.npy', allow_pickle=True).item()
val_cm = cm_dict['val']


print("Validation Confusion Matrix:")
print(val_cm)


def parse_confusion_matrix(cm, class_names=None):
    n_classes = cm.shape[0]
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    accuracy = np.diag(cm).sum() / cm.sum()
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    
    return accuracy



def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()


# class_names = ['fiction', 'government', 'slate', 'telephone', 'travel']
class_names = ['easy', 'hard', 'medium']
parse_confusion_matrix(val_cm, class_names)
plot_confusion_matrix(val_cm, class_names, title='Test Confusion Matrix')
# plot_confusion_matrix(val_cm, class_names, title='Normalized Confusion Matrix', normalize=True)