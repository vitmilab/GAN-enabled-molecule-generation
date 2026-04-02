# GAN-Enabled Molecule Generation for TEAD4 Activity Prediction

This project implements a conditional Generative Adversarial Network (cGAN) to generate synthetic molecular descriptors, which are used to augment training data for improving classification performance in TEAD4 activity prediction.

The pipeline integrates:
- Descriptor-based molecular representation (QikProp)
- cGAN-based data augmentation
- XGBoost classifier for activity prediction

## Workflow
1. Load molecular descriptor dataset (QikProp)
2. Normalize features (Min-Max scaling)
3. Train conditional GAN (cGAN)
4. Generate synthetic samples (balanced classes)
5. Split real dataset into train/validation/test
6. Augment training set with synthetic data
7. Standardize features (fit on training only)
8. Train XGBoost classifier
9. Evaluate on validation and test sets
10. Predict external DrugBank molecules


## Data Availability
- The training dataset (QikProp descriptor matrix of TEAD4 actives/inactives)
- External screening descriptors (DrugBank) were generated using the Schrödinger QikProp module.

## Requirements
- Python >= 3.9
- PyTorch
- NumPy, Pandas, Scikit-learn
- XGBoost
- Matplotlib
- Joblib

conda create -n gan_env python=3.9

conda activate gan_env

## Install dependencies:
pip install -r requirements.txt

## Usage
Run the pipeline in the following order:

python gan.py              # Train cGAN and generate synthetic data

python gan_validation.py   # Inspect GAN training performance

python evaluation.py       # Train XGBoost model and evaluate performance
