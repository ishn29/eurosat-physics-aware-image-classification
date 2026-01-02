# Physics-Aware Multispectral Learning for Land-Cover Classification

This project investigates whether incorporating physically meaningful inductive biases into multispectral convolutional neural networks improves land-cover classification performance. Using the EuroSAT dataset, we compare RGB, multispectral, and physics-aware architectures under controlled experimental conditions and analyze robustness to spectral band dropout.

---

## Motivation

Standard computer vision models treat multispectral satellite imagery as collections of unstructured channels. However, spectral bands correspond to distinct physical properties such as vegetation health, moisture content, and material composition. This project explores whether explicitly encoding these physical relationships into the model architecture improves generalization, robustness, and interpretability.

---

## Dataset

We use the EuroSAT dataset, which provides 64×64 satellite image patches across 10 land-cover classes. Experiments are conducted using both RGB imagery and the full 13-band Sentinel-2 multispectral data. Train, validation, and test splits are frozen and stratified to ensure reproducibility across all experiments.

---

## Methods

Three models are evaluated:

- **RGB Baseline:** Standard ResNet18 trained on RGB imagery.
- **Multispectral Baseline:** ResNet18 adapted to accept 13 spectral bands.
- **Physics-Aware Model:** A custom architecture that processes physically meaningful spectral band groups (visible, red-edge, near-infrared, and short-wave infrared) through separate convolutional stems before feature fusion.

All models are trained using identical optimization protocols and evaluated on a held-out test set.

---

## Results

We compare three models on the EuroSAT land-cover classification task:

| Model            | Input Type            | Test Accuracy |
| ---------------- | --------------------- | ------------- |
| ResNet18 (RGB)   | 3-band RGB            | **75.7%**     |
| ResNet18 (MS)    | 13-band multispectral | **92.6%**     |
| Physics-aware MS | Band-grouped 13-band  | **94.7%**     |

The multispectral baseline significantly outperforms the RGB model, confirming that physically meaningful spectral information is crucial for land-cover classification. The physics-aware architecture further improves performance, indicating that explicitly encoding physical band groupings provides a useful inductive bias beyond simply increasing input dimensionality.

Confusion matrices for each model are provided in the `results/` directory.

### Robustness to Spectral Band Dropout

We evaluate robustness by zeroing out physically meaningful band groups at test time. The physics-aware model consistently degrades more gracefully than the standard multispectral model, particularly when near-infrared and short-wave infrared bands are removed, suggesting that the inductive bias encourages more stable, distributed representations.

Full robustness results are available in `results/band_dropout_results.csv`.

---

## Reproducibility

To reproduce the experiments:

`pip install -r requirements.txt`

Run the notebooks in order

`01_dataset_preparation_and_splits.ipynb`

`02_rgb_baseline_models.ipynb`

`03_multispectral_models.ipynb`

`04_physics_aware_models.ipynb`

`05_analysis_robustness_interpretability.ipynb`


---

## Key Findings

- Multispectral inputs provide a substantial performance improvement over RGB imagery.
- Explicitly encoding physical structure into the architecture further improves accuracy.
- Physics-aware models exhibit greater robustness to the removal of critical spectral bands.

## Future Work

Future directions include domain-shift evaluation across geographic regions, incorporation of attention mechanisms for band weighting, and extension to temporal remote sensing datasets.
