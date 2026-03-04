import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne_per_layer(cls_tensor, D, perplexity=30, random_state=42):
    """
    Applies t-SNE to each layer's [CLS] token embeddings across samples.
    
    Args:
        cls_tensor (torch.Tensor): Tensor of shape (N, D*L)
        D (int): Hidden dimension size
        perplexity (float): t-SNE perplexity
        random_state (int): t-SNE random seed
        
    Returns:
        None (displays plots)
    """
    N, DL = cls_tensor.shape
    L = DL // D
    cls_np = cls_tensor.cpu().numpy()  # convert to numpy if needed

    fig, axes = plt.subplots(nrows=(L + 3) // 4, ncols=4, figsize=(20, 5 * ((L + 3) // 4)))
    axes = axes.flatten()

    for i in range(L):
        print(i)
        layer_features = cls_np[:, i * D:(i + 1) * D]  # shape (N, D)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        tsne_result = tsne.fit_transform(layer_features)

        ax = axes[i]
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5)
        ax.set_title(f"Layer {i + 1}")
        ax.axis('off')

    for j in range(L, len(axes)):
        axes[j].axis('off')  # hide unused subplots

    plt.tight_layout()
    plt.savefig("ours_tsne.png")


# Example
# cls_tensor = torch.load("/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-10_23_bs64_ToMe_8_best_params_tsne_iter75/rendition-cls_feats.pt")
cls_tensor = torch.load("/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-10_23_bs64_ToMe_8_tsne_iter75/rendition-cls_feats.pt")
# cls_tgt = torch.load("/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-10_23_bs64_ToMe_8_best_params_tsne_iter75/rendition-cls_tgts.pt")
cls_tgt = torch.load("/nlp_group/xiongyizhe/research/TTA/MGTTA/outputs/foa_bp_2025-07-10_23_bs64_ToMe_8_tsne_iter75/rendition-cls_tgts.pt")
import pdb; pdb.set_trace()
cls_tensor = cls_tensor[:1000, :]
D = 768
tsne_per_layer(cls_tensor, D)