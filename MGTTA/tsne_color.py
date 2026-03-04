import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import random

def tsne_per_layer_colored(cls_tensor_list, labels_list, D, target_label_set, perplexity=30, random_state=42):
    """
    Applies t-SNE to each layer's [CLS] token embeddings across samples, filtered by label set and colored by class.

    Args:
        cls_tensor (torch.Tensor): Tensor of shape (N, D*L)
        labels (torch.Tensor): Tensor of shape (N,) with integer class labels
        D (int): Hidden dimension
        target_label_set (set): Set of target labels to include
        perplexity (float): t-SNE perplexity
        random_state (int): Random seed for reproducibility

    Returns:
        None (plots are shown)
    """
    N, DL = cls_tensor_list[0].shape
    L = DL // D

    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    axes = axes.flatten()
    # methods = ["Ours", "ToMe"]
    methods = ["Baseline"]
    layer_ind = [3, 6, 11]

    # Filter by label
    for j in [0]:
        method_name = methods[j]
        labels = labels_list[j]
        cls_tensor = cls_tensor_list[j]

        labels_np = labels.cpu().numpy()
        mask = np.isin(labels_np, list(target_label_set))
        cls_np = cls_tensor[mask].cpu().numpy()
        filtered_labels = labels_np[mask]
        filtered_labels = np.where(filtered_labels<3.5, 0, np.where(filtered_labels>3.5, 1, filtered_labels))        # only track fish and bird

        # Relabel for color mapping
        unique_labels = sorted(set(filtered_labels))
        label_to_color_idx = {label: idx for idx, label in enumerate(unique_labels)}
        color_indices = np.array([label_to_color_idx[lbl] for lbl in filtered_labels])

        # Color palette
        palette = np.array(sns.color_palette("hls", len(unique_labels)))

        # for ind, i in enumerate([2,5,10]):
        for ind, i in enumerate([2,8,10]):
            print(i)
            layer_features = cls_np[:, i * D:(i + 1) * D]  # shape (M, D)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
            tsne_result = tsne.fit_transform(layer_features)

            ax = axes[ind+j*3]
            for lbl in unique_labels:
                idx = (filtered_labels == lbl)
                ax.scatter(tsne_result[idx, 0], tsne_result[idx, 1],
                        label=str(lbl), color=palette[label_to_color_idx[lbl]], s=5)

            # ax.set_title(f"Layer {i + 1} ({method_name})", fontsize=25)
            ax.set_title(f"Layer {layer_ind[ind]} ({method_name})", fontsize=25)
            ax.axis('off')

        for j in range(L, len(axes)):
            axes[j].axis('off')

        # Optional: create a legend outside the last subplot
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                            label=str(lbl),
                            markerfacecolor=palette[label_to_color_idx[lbl]],
                            markersize=6) for lbl in unique_labels]
        # fig.legend(handles=handles, loc='lower center', ncol=10, fontsize='small')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.savefig("baseline_tsne.png")

N, D, L = 10032, 768, 12
# cls_tensor_ours = torch.load("/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-10_23_bs64_ToMe_8_best_params_tsne_iter75/rendition-cls_feats.pt")
# cls_tensor_tome = torch.load("/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-10_23_bs64_ToMe_8_tsne_iter75/rendition-cls_feats.pt")
# labels_ours = torch.load("/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-10_23_bs64_ToMe_8_best_params_tsne_iter75/rendition-cls_tgts.pt")
# labels_tome = torch.load("/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-10_23_bs64_ToMe_8_tsne_iter75/rendition-cls_tgts.pt")
cls_tensor_foabp = torch.load("/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2026-01-26_19_bs64_tsne_iter75/rendition-cls_feats.pt")
labels_foabp = torch.load("/home/xiongyizhe/CVPR2026/MGTTA/outputs/foa_bp_2026-01-26_19_bs64_tsne_iter75/rendition-cls_tgts.pt")
# import pdb; pdb.set_trace()

# target_label_set = random.sample(range(200), 10) # e.g., select class labels 0–19
# target_label_set = [0,1,2,3,4,5,6,7] # 0-3: fish 4-7: bird
target_label_set = [0,1,3,5,6] # 0-3: fish 4-7: bird
tsne_per_layer_colored((cls_tensor_foabp, None), (labels_foabp, None), D, target_label_set)

