<h1 align="center">🌱 Woody Plant Encroachment Monitoring (Remote Sensing + ML)</h1>

<p align="center">
Open-source framework for monitoring woody plant encroachment in floodplain wetlands using 
<strong>multi-season Sentinel-2 imagery, spectral indices, terrain variables, and machine learning (Random Forest &amp; XGBoost)</strong>.
</p>

<p align="center">
This repository contains all the scripts, environment requirements, and instructions needed to reproduce the analysis — 
from preprocessing Sentinel-2 data to generating encroachment statistics and publication-ready figures.
</p>


---

## 🌱 Key Features
- Sentinel-2 preprocessing (SAFE → stacked multi-band GeoTIFFs).
- Automatic extraction of seasonal spectral indices (NDVI, NDWI, EVI, MSAVI2, etc.).
- Integration of DEM-based terrain variables (elevation, slope, curvature).
- Stratified label splitting (70/30 train/test).
- Random Forest (RF) and XGBoost (XGB) classification with hyperparameter tuning.
- Pixel-level classification at ROI and wetland boundaries.
- Automated computation of **area statistics & change detection (2016–2025)**.
- Generation of comparison figures for **RF vs XGBoost models** (Panels A–D in the paper).

---

## 📂 Repository Structure

```
woody-plant-encroachment-remote-sensing/
│
├── scripts/                      # All analysis scripts (numbered pipeline)
│   ├── 01_preprocess_s2_safe.py
│   ├── 02_stratified_split_labels.py
│   ├── 03_extract_seasonal_features_train.py
│   ├── 03b_extract_seasonal_features_test.py
│   ├── 04_train_test_rf.py
│   ├── 04b_train_test_xgb.py
│   ├── 05_build_feature_stack_area.py
│   ├── 05b_classify_rf_area.py
│   ├── 05c_plot_classified_map.py
│   ├── 05d_classify_xgb_area.py
│   ├── 06_area_stats.py
│   ├── 06b_area_plots.py
│   ├── 07_wetland_stats.py
│   ├── 07b_wetland_plots.py
│   └── 08_rf_xgb_comparison_panels.py
│
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── .gitignore                    # Ignore large rasters/outputs
└── README.md                     # Project documentation (this file)
```

---

## ⚙️ Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/abbi776/woody-plant-encroachment-remote-sensing.git
cd woody-plant-encroachment-remote-sensing

# Create a fresh environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install required packages
pip install -r requirements.txt
```

---

## 🚀 Pipeline Steps

The workflow is broken into **numbered scripts** for clarity:

1. **Preprocess Sentinel-2 SAFE archive**
   ```bash
   python scripts/01_preprocess_s2_safe.py
   ```

2. **Stratified train/test split of polygons**
   ```bash
   python scripts/02_stratified_split_labels.py
   ```

3. **Extract seasonal features** (spectral + terrain)  
   - Training: `03_extract_seasonal_features_train.py`  
   - Testing:  `03b_extract_seasonal_features_test.py`

4. **Train + test classifiers**  
   - RF:  `04_train_test_rf.py`  
   - XGB: `04b_train_test_xgb.py`  
   - Performs hyperparameter tuning + evaluation.

5. **Classify full rasters (per ROI / wetland)**  
   - Build feature stack → `05_build_feature_stack_area.py`  
   - Classify with RF → `05b_classify_rf_area.py`  
   - Classify with XGB → `05d_classify_xgb_area.py`  
   - Plot classified map → `05c_plot_classified_map.py`

6. **ROI-level statistics & plots**  
   - Compute areas → `06_area_stats.py`  
   - Generate plots → `06b_area_plots.py`

7. **Wetland-level statistics & plots**  
   - Compute areas → `07_wetland_stats.py`  
   - Generate plots → `07b_wetland_plots.py`

8. **RF vs XGBoost comparison panels** (A–D)  
   ```bash
   python scripts/08_rf_xgb_comparison_panels.py
   ```

---

## 📊 Outputs
- Classified rasters (`.tif`) for each ROI/wetland and year.
- Excel tables of class areas, change detection, and wetland summaries.
- Publication-ready plots:
  - Line plots, bar charts, stacked areas.
  - RF vs XGB comparison panels.
  - Wetland-level encroachment/retreat counts.

---

## 📁 Data & Results Organization

Since raw rasters and outputs are large, they are **not stored in this repo**.  
Organize your local project like this:

```
data/
  ├── sentinel2/          # Sentinel-2 SAFE archives (.zip or .SAFE folders)
  ├── labels/             # Training/testing shapefiles
  ├── dem/                # DEM, slope, curvature rasters
  └── wetlands/           # Wetland boundary shapefiles

results/
  ├── models/             # Saved RF/XGB models (.pkl)
  ├── rasters/            # Classified rasters
  ├── stats/              # ROI & wetland statistics (Excel/CSV)
  └── figures/            # Publication-ready plots
```

⚠️ Note: `.gitignore` excludes these outputs so they don’t get pushed to GitHub.

---

## 🧑‍🤝‍🧑 Contributing
Contributions are welcome!  
Please open an issue or submit a pull request if you’d like to improve code efficiency, add models, or extend to other ecosystems.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 📧 Contact
For questions or collaboration:  
**Abdullah Toqeer**  
PhD Candidate, Charles Sturt University  
Email: *toqeerabdullah776@gmail.com*  
