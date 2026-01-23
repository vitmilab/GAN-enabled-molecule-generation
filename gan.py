import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# -----------------------
# SETTINGS
# -----------------------
SEED = 42
INPUT_CSV = r'E:/ML/GAN/QikProp.csv'
EXTERNAL_CSV = r'E:/ML/GAN/Drugbank_descriptors.csv'
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# FIX RANDOMNESS (core reproducibility)
# -----------------------
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

set_global_seed(SEED)
rng = np.random.RandomState(SEED)

# -----------------------
# LOAD & NORMALIZE
# -----------------------
df = pd.read_csv(INPUT_CSV)
mol_ids = df['molecule'].values
y = df['Label'].values.astype(int)
feature_cols = [c for c in df.columns if c not in ('molecule', 'Label')]
X = df[feature_cols].values.astype('float32')

# Min-Max normalization (store for later use)
X_min, X_max = X.min(axis=0), X.max(axis=0)
range_ = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
X_norm = (X - X_min) / range_

print(f"Data loaded: {X_norm.shape}, Labels distribution: {np.bincount(y)}")

# -----------------------
# GAN MODELS
# -----------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + label_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_data, labels):
        c = self.label_emb(labels)
        x = torch.cat([x_data, c], dim=1)
        return self.net(x)

# -----------------------
# HYPERPARAMETERS
# -----------------------
latent_dim = 100
data_dim = X_norm.shape[1]
n_classes = 2
batch_size = 64
gan_epochs = 1000
lr = 1e-4
label_smooth_real = 0.9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)

# -----------------------
#  INIT MODELS
# -----------------------
X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)
gen = Generator(latent_dim, n_classes, data_dim).to(device)
disc = Discriminator(data_dim, n_classes).to(device)
optimizer_G = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# -----------------------
# Prepare to store losses
# -----------------------
D_losses = []
G_losses = []

# -----------------------
# TRAIN cGAN (Deterministic)
# -----------------------
print("\n🔹 Training cGAN...")
for epoch in range(gan_epochs):
    set_global_seed(SEED + epoch)  # ensure reproducible noise per epoch

    idx = rng.choice(X_tensor.shape[0], batch_size, replace=False)
    real_samples = X_tensor[idx]
    real_labels = y_tensor[idx]

    z = torch.randn(batch_size, latent_dim, device=device)
    gen_labels = torch.from_numpy(rng.randint(0, n_classes, size=(batch_size,))).to(device)
    fake_samples = gen(z, gen_labels)

    optimizer_D.zero_grad()
    real_target = torch.full((batch_size, 1), label_smooth_real, device=device)
    fake_target = torch.full((batch_size, 1), 1.0 - label_smooth_real, device=device)
    real_valid = disc(real_samples, real_labels)
    fake_valid = disc(fake_samples.detach(), gen_labels)
    d_loss = 0.5 * (criterion(real_valid, real_target) + criterion(fake_valid, fake_target))
    d_loss.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()
    gen_target = torch.full((batch_size, 1), label_smooth_real, device=device)
    validity = disc(fake_samples, gen_labels)
    g_loss = criterion(validity, gen_target)
    g_loss.backward()
    optimizer_G.step()

# store losses
    D_losses.append(d_loss.item())
    G_losses.append(g_loss.item())
    
    if epoch % 100 == 0 or epoch == gan_epochs - 1:
        print(f"Epoch {epoch}/{gan_epochs} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

print("Finished GAN training.")

torch.save(gen.state_dict(), os.path.join(MODEL_DIR, "generator.pth"))
torch.save(disc.state_dict(), os.path.join(MODEL_DIR, "discriminator.pth"))

# -----------------------
# Save losses for plotting
# -----------------------
loss_df = pd.DataFrame({'epoch': range(gan_epochs), 'D_loss': D_losses, 'G_loss': G_losses})
loss_df.to_csv(os.path.join(MODEL_DIR, "gan_losses.csv"), index=False)

# -----------------------
# Plot GAN convergence curves
# -----------------------
def moving_average(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')

window_size = 10
D_loss_smooth = np.convolve(D_losses, np.ones(window_size)/window_size, mode='valid')
G_loss_smooth = np.convolve(G_losses, np.ones(window_size)/window_size, mode='valid')
epochs_smooth = np.arange(window_size - 1, gan_epochs)  # match smoothed lengths

plt.figure(figsize=(10,6))
plt.plot(epochs_smooth, D_loss_smooth, label='Discriminator Loss', color='red', alpha=0.8)
plt.plot(epochs_smooth, G_loss_smooth, label='Generator Loss', color='blue', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Loss Curves Over 1000 Epochs')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save figure to MODEL_DIR
fig_path = os.path.join(MODEL_DIR, "GAN_loss_curves.png")
plt.savefig(fig_path, dpi=300)  # high-res for manuscript
print(f"GAN convergence figure saved: {fig_path}")

# Display the figure interactively as well
plt.show()

# -----------------------
# SYNTHETIC GENERATION (Balanced)
# -----------------------
num_synthetic = 100
synthetic_labels = np.array([0]*(num_synthetic//2) + [1]*(num_synthetic//2))
rng.shuffle(synthetic_labels)

with torch.no_grad():
    set_global_seed(SEED + 999)
    z_syn = torch.randn(num_synthetic, latent_dim, device=device)
    gen_labels_t = torch.tensor(synthetic_labels, dtype=torch.long, device=device)
    synthetic_features = gen(z_syn, gen_labels_t).cpu().numpy()

print("Synthetic data generated:", synthetic_features.shape, "Label counts:", np.bincount(synthetic_labels))

# -----------------------
# COMBINE + STANDARDIZE
# -----------------------
X_combined = np.vstack([X_norm, synthetic_features])
y_combined = np.concatenate([y, synthetic_labels])
perm = np.random.RandomState(SEED).permutation(X_combined.shape[0])
X_combined, y_combined = X_combined[perm], y_combined[perm]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

# -----------------------
# SPLIT DATA (3:1:1)
# -----------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_combined, test_size=0.2, stratify=y_combined, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEED)

print(f"Splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -----------------------
# XGBoost TRAINING (Deterministic)
# -----------------------
xgb_params = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=SEED,
    n_jobs=1,  # single-thread for deterministic results
    tree_method='hist',
    verbosity=1
)

xgb_params['eval_metric'] = 'logloss'
clf = XGBClassifier(**xgb_params)

# Fit model and track logloss
eval_results = {}
clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)
eval_results = clf.evals_result()

# Plot and save figure

# -----------------------
# Extract logloss curves
# -----------------------
train_logloss = eval_results['validation_0']['logloss']
val_logloss = eval_results['validation_1']['logloss']
epochs = np.arange(1, len(train_logloss)+1)

# -----------------------
# Plot logloss curves
# -----------------------
plt.figure(figsize=(8,5))
plt.plot(epochs, train_logloss, label='Train LogLoss', color='blue', alpha=0.8)
plt.plot(epochs, val_logloss, label='Validation LogLoss', color='orange', alpha=0.8)
plt.xlabel('Boosting Round')
plt.ylabel('LogLoss')
plt.title('XGBoost Training vs. Validation LogLoss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save figure
fig_path = os.path.join(MODEL_DIR, "XGB_TrainVal_LogLoss.png")
plt.savefig(fig_path, dpi=300)
print(f"XGBoost train/validation logloss figure saved: {fig_path}")

plt.show()

joblib.dump(clf, os.path.join(MODEL_DIR, "xgb_model.joblib"))
clf.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))

# -----------------------
# EVALUATION (with probabilities)
# -----------------------
y_pred_test = clf.predict(X_test)
y_prob_test = clf.predict_proba(X_test)[:, 1]  # probability of being 'active'

acc_test = accuracy_score(y_test, y_pred_test)
print("\nTEST PERFORMANCE")
print(f"Accuracy: {acc_test:.4f}")
print(classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# Optional: show probabilities for first 5 test samples
for i in range(5):
    print(f"Sample {i}: Predicted class={y_pred_test[i]}, Probability={y_prob_test[i]:.4f}")

# -----------------------
# EXTERNAL PREDICTION WITH PROBABILITIES
# -----------------------
if os.path.exists(EXTERNAL_CSV):
    df_ext = pd.read_csv(EXTERNAL_CSV)
    X_ext = df_ext[feature_cols].values.astype('float32')
    X_ext_norm = (X_ext - X_min) / range_
    X_ext_scaled = scaler.transform(np.nan_to_num(X_ext_norm))

    y_prob_ext = clf.predict_proba(X_ext_scaled)[:, 1]  # probability of active
    y_pred_ext = (y_prob_ext >= 0.5).astype(int)  # default threshold 0.5

    # Add to dataframe
    df_ext['predicted_activity'] = y_pred_ext
    df_ext['predicted_probability'] = y_prob_ext

    df_ext.to_csv(r'E:/ML/GAN_New/Drugbank_prediction.csv', index=False)
    print(f"Saved external predictions with probabilities for {df_ext.shape[0]} compounds.")
else:
    print("External file not found — skipping.")

# -----------------------
# SUMMARY
# -----------------------
print("\n=== SUMMARY ===")
print(f"Seed: {SEED}")
print(f"GAN epochs: {gan_epochs}")
print(f"Train/Val/Test sizes: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
print(f"Final test accuracy: {acc_test:.4f}")
print("All outputs are reproducible and saved in:", MODEL_DIR)