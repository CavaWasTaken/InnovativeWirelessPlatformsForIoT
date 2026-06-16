# 🌿 Deep Learning for Evapotranspiration Estimation
### LSTM-Based AET Prediction in Piedmont, Italy

A deep learning project for estimating **Actual Evapotranspiration (AET)** using a Long Short-Term Memory (LSTM) neural network, trained on agrometeorological data from the Piedmont region in northern Italy.

---

## 📖 Overview

Evapotranspiration (ET) is a key component of the water cycle, quantifying the volume of water vapor transferred from the Earth's surface to the atmosphere. Accurate ET estimation is essential for:

- Optimizing irrigation scheduling in agriculture
- Early detection of drought conditions
- Improving weather prediction models
- Monitoring climate change impacts

This project implements an LSTM model to predict daily AET values from climatological time series data, and includes a systematic review of 159 papers on machine learning approaches for ET estimation.

---

## 🗂️ Repository Structure

```
├── data/
│   ├── raw/               # Raw MADIA and GLEAM dataset files
│   └── processed/         # Aligned, cleaned, and normalized dataset
├── notebooks/
│   └── lstm_aet.ipynb     # Main training and evaluation notebook
├── src/
│   ├── preprocessing.py   # Data loading, alignment, PCA, normalization
│   ├── model.py           # LSTM architecture definition
│   └── evaluate.py        # Metrics and visualization
├── results/
│   └── figures/           # Scatter plots, correlation matrices, etc.
└── README.md
```

---

## 📦 Datasets

### MADIA — Meteorological Variables for Agriculture
Agrometeorological variables across Italy at 0.25° resolution, derived from ERA5 reanalysis data (1981–2022). Owned and maintained by CREA (Council for Agricultural Research and Economics).

- Download: [Zenodo](https://zenodo.org/records/7252361)
- Features: min/mean/max air temperature, min/max relative humidity, wind speed, solar radiation, precipitation, reference ET₀

### GLEAM — Global Land Evaporation Amsterdam Model
Satellite-based framework providing global daily AET estimates at 0.25° resolution.

- Download: [gleam.eu](https://www.gleam.eu/) (registration required, SFTP)
- Used for: daily AET target values (2018–2022)

**Study area:** Piedmont region, Italy — latitudes 44.0°–46.5°, longitudes 6.5°–9.0°  
**Time period:** 2018-01-01 to 2022-01-01

---

## ⚙️ Methods

### Preprocessing
1. Spatial alignment of MADIA and GLEAM grid points (nearest-neighbor matching)
2. Removal of records with missing AET values (160,688 → 158,862 samples)
3. Standard Scaler normalization (zero mean, unit variance)
4. PCA to reduce collinearity (9 features → 6 principal components)

### Model Architecture
A stacked LSTM network with 122,753 trainable parameters:

| Layer | Type | Details |
|-------|------|---------|
| 1 | LSTM | 128 units, `return_sequences=True` |
| 2 | LSTM | 64 units, `return_sequences=False` |
| 3 | Dropout | 10% dropout rate |
| 4 | Dense | 64 units, ReLU activation |
| 5 | Dense | 1 unit (AET output) |

- **Input:** sequences of 30 consecutive days of climatological data
- **Output:** AET value for the following day
- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Training:** 50 epochs, batch size 64, 80/20 train-test split

---

## 📊 Results

### Main Model (PCA-based, 30-day sequences)

| Metric | Value |
|--------|-------|
| R² | 0.9392 |
| RMSE | 0.2697 mm/day |
| MAE | 0.2000 mm/day |

### Robustness Tests

| Configuration | R² | RMSE | MAE |
|---|---|---|---|
| PCA-based (30 days) | 0.9392 | 0.2697 | 0.2000 |
| PCA-based (14 days) | 0.9141 | 0.3212 | 0.2297 |
| PCA-based (7 days) | 0.8556 | 0.4152 | 0.2733 |
| PCA + 10% missing data (30 days) | 0.8832 | 0.3738 | 0.2578 |
| Feature-select model (30 days) | 0.9414 | 0.2648 | 0.1940 |
| Feature-select, no precipitation (30 days) | 0.9366 | 0.2753 | 0.2023 |
| Transferability to Veneto region | 0.5744 | 0.7125 | 0.4625 |

> **Note:** Transferability to a different region (Veneto) shows significantly lower performance, indicating the model learns region-specific temporal patterns.

---

## 🚀 Getting Started

### Requirements

```bash
pip install tensorflow pandas numpy scikit-learn netCDF4 matplotlib seaborn
```

### Training

```python
# Clone the repo
git clone https://github.com/CavaWasTaken/InnovativeWirelessPlatformsForIoT.git
cd InnovativeWirelessPlatformsForIoT

# Download and place datasets in data/raw/
# Then run the notebook
jupyter notebook notebooks/lstm_aet.ipynb
```

> Training was performed on Google Colab with GPU acceleration (~932 seconds for 50 epochs).

---

## 📝 Citation

If you use this work, please cite:

```
L. Cavallaro, "Deep Learning for Evapotranspiration Estimation: A Systematic Review
and LSTM-Based Case Study in Piedmont, Italy", 2025.
```

---

## 📚 Key References

- Parisse et al. (2023) — MADIA dataset. *Data in Brief*, vol. 46.
- Babaeian et al. (2022) — Short- and mid-term AET forecasts with deep learning. *Journal of Hydrology*, vol. 612.
- GLEAM dataset — [gleam.eu](https://www.gleam.eu/)

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for details.
