# -----------------------
# GAN VALIDATION: PCA ANALYSIS
# -----------------------
from sklearn.decomposition import PCA

# Combine real + synthetic for PCA
X_pca_input = np.vstack([X_norm, synthetic_features])
labels_pca = np.array(['Real']*len(X_norm) + ['Synthetic']*len(synthetic_features))

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_pca_input)

plt.figure(figsize=(7,6))
for label in ['Real', 'Synthetic']:
    idx = labels_pca == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.6)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Real vs GAN-Generated Samples')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

pca_path = os.path.join(MODEL_DIR, "PCA_real_vs_synthetic.png")
plt.savefig(pca_path, dpi=300)
print(f"PCA plot saved: {pca_path}")

plt.show()


# -----------------------
# GAN VALIDATION: DESCRIPTOR DISTRIBUTION CHECK
# -----------------------
import seaborn as sns

# Select a few representative features (first 3 for simplicity)
num_features_to_plot = min(3, X_norm.shape[1])

for i in range(num_features_to_plot):
    plt.figure(figsize=(6,4))
    sns.kdeplot(X_norm[:, i], label='Real', fill=True)
    sns.kdeplot(synthetic_features[:, i], label='Synthetic', fill=True)
    plt.title(f'Distribution Comparison - Feature {i}')
    plt.legend()
    plt.tight_layout()

    dist_path = os.path.join(MODEL_DIR, f"feature_{i}_distribution.png")
    plt.savefig(dist_path, dpi=300)
    print(f"Saved distribution plot: {dist_path}")

    plt.show()

# -----------------------
# GAN VALIDATION: t-SNE ANALYSIS
# -----------------------
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    random_state=SEED,
    perplexity=30,
    max_iter=1000,
    learning_rate='auto',
    init='pca'
)
X_tsne = tsne.fit_transform(X_pca_input)

plt.figure(figsize=(7,6))
for label in ['Real', 'Synthetic']:
    idx = labels_pca == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label, alpha=0.6)

plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('t-SNE of Real vs GAN-Generated Samples')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

tsne_path = os.path.join(MODEL_DIR, "tSNE_real_vs_synthetic.png")
plt.savefig(tsne_path, dpi=300)
print(f"t-SNE plot saved: {tsne_path}")

plt.show()
