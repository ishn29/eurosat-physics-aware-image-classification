# eurosat-physics-aware-image-classification
Physics-aware satellite image classification using multispectral Sentinel-2 data (EuroSAT).


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
