import os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def cosine_sim_matrix(A, eps=1e-12):
    # A: (N, D)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    return An @ An.T

def class_mean_weights_variable(W, neuron_active):
    """
    Compute mean weights for each class with variable active neurons.
    W: (300, 6272)
    neuron_active: list of length 10, e.g. [9,8,9,...]
    """
    Wc = []
    n_per_class = 30

    for c, n_act in enumerate(neuron_active):
        base = c * n_per_class
        idx = np.arange(base, base + n_act)  # Only active neurons
        Wc.append(W[idx].mean(axis=0))
    return np.stack(Wc, axis=0)  # (10, D)

def plot_class_signature_topk_variable(Wc, out_dir, neuron_active, c, k=30):
    """Plot class signature (top-k deviations from global mean)."""
    mu = Wc.mean(axis=0)
    d = Wc[c] - mu

    idx = np.argsort(np.abs(d))[::-1][:k]
    vals = d[idx]

    plt.figure(figsize=(9, 3))
    plt.bar(range(k), vals)
    plt.xticks(range(k), idx, rotation=90, fontsize=7)
    plt.ylabel("Δ weight (class mean - global mean)")
    plt.title(
        f"Class {c} signature (Top-{k}) | active neurons = {neuron_active[c]}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"class_signature_top{k}_class{c}.png"), dpi=200)
    plt.close()

def plot_intra_class_similarity_variable(W, out_dir, neuron_active, c):
    """Plot cosine similarity matrix within a single class."""
    n_per_class = 30
    base = c * n_per_class
    n_act = neuron_active[c]

    A = W[base : base + n_act]  # (n_act, 6272)

    # Cosine similarity
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    S = An @ An.T

    plt.figure(figsize=(4, 4))
    plt.imshow(S, vmin=-1, vmax=1, cmap="seismic", aspect="equal")
    plt.colorbar()
    plt.title(f"Class {c} intra-class similarity (n={n_act})")
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"intra_class_similarity_class{c}.png"), dpi=200)
    plt.close()

def plot_class_overlap_matrix(Wc, out_dir, k=50):
    """Plot Jaccard overlap matrix of top-k features across classes."""
    ensure_dir(out_dir)
    n_classes = Wc.shape[0]
    top_sets = []
    for c in range(n_classes):
        idx = np.argsort(np.abs(Wc[c]))[::-1][:k]
        top_sets.append(set(idx.tolist()))

    M = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            M[i, j] = inter / union if union > 0 else 0.0

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(M, vmin=0, vmax=1, aspect="equal")
    plt.colorbar()
    plt.xticks(range(n_classes), range(n_classes))
    plt.yticks(range(n_classes), range(n_classes))
    plt.title(f"Class overlap (Jaccard of Top-{k} |weights| in class means)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"class_overlap_top{k}.png"), dpi=200)
    plt.close()

def plot_class_feature_heatmap_sorted(Wc, out_dir):
    """Plot class × feature heatmap sorted by feature variance."""
    ensure_dir(out_dir)
    order = np.argsort(np.var(Wc, axis=0))[::-1]
    Wcs = Wc[:, order]

    v = np.percentile(np.abs(Wcs), 99)
    if v == 0:
        v = 1e-8

    plt.figure(figsize=(12, 4))
    plt.imshow(Wcs, aspect="auto", cmap="seismic", vmin=-v, vmax=v)
    plt.colorbar()
    plt.yticks(range(Wc.shape[0]), [f"class {i}" for i in range(Wc.shape[0])])
    plt.xlabel("Feature index (sorted by class variance)")
    plt.title("Class × Feature heatmap (class-variance sorted)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_feature_heatmap_sorted.png"), dpi=200)
    plt.close()

def main():
    # Paths: modify according to your log folder
    log_dir = "./visual"  # e.g., self.logger.log_path
    w_path = "../nas/logs/STDP-CSNN/CIFAR10/kfold/s2stdp+ncg_update500_300/4/weights.npy"
    out_dir = os.path.join(log_dir, "figs")
    ensure_dir(out_dir)

    W = np.load(w_path)
    print("weights shape:", W.shape)

    n_classes = 10
    n_per_class = 30
    assert W.shape[0] == n_classes * n_per_class, "weights rows must be 10*30=300"
    D = W.shape[1]
    neuron_active = [9, 8, 9, 10, 9, 10, 7, 10, 9, 9]
    
    # Compute class means
    Wc = class_mean_weights_variable(W, neuron_active)
    print("class mean shape:", Wc.shape, "D =", D)

    # Figure 1: Heatmap sorted by variance
    plot_class_feature_heatmap_sorted(Wc, out_dir)

    # Figure 2: Class signature (top-k deviations) - show 3 examples
    plot_class_signature_topk_variable(Wc, out_dir, neuron_active, c=0, k=30)
    plot_class_signature_topk_variable(Wc, out_dir, neuron_active, c=1, k=30)
    plot_class_signature_topk_variable(Wc, out_dir, neuron_active, c=2, k=30)

    # Figure 3: Intra-class similarity (single class example)
    plot_intra_class_similarity_variable(W, out_dir, neuron_active, c=0)

    # Bonus: Class overlap matrix (feature sharing ratio)
    plot_class_overlap_matrix(Wc, out_dir, k=50)

    print("Saved figures to:", out_dir)

if __name__ == "__main__":
    main()