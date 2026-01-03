Benchmarking Sentinel-1 SAR Features for Woody Plant Encroachment Mapping

Open-source framework for benchmarking Sentinel-1 SAR features and applying them to woody plant encroachment (WPE) mapping in floodplain wetlands using:

multi-season SAR features, polarization indices, textures, and machine learning (RF, SVM, XGBoost).

This repository contains all scripts required to reproduce the workflow — from Sentinel-1 preprocessing in GEE to interpretable wall-to-wall mapping.

🌱 Key Features

Sentinel-1 preprocessing in Google Earth Engine (GEE)

Border-noise masking, speckle filtering, ellipsoidal RTC

Unified SAR feature stack (VV/VH + indices + textures)

Polygon-based feature extraction for machine learning

Leave-One-Polygon-Out (LOPO) validation

Random Forest, SVM, and XGBoost comparison

Global SHAP feature importance

Wall-to-wall classification: 2016, 2018, 2025

Parts of the preprocessing build on
Mullissa et al. (2021) – Sentinel-1 ARD (GEE)
https://github.com/adugnag/gee_s1_ard

📂 Repository Structure
benchmarking-sentinel1-sar-features/
│
├── gee/                         
│   ├── 00_utils_sar.js
│   ├── 01_border_noise.js
│   ├── 02_speckle_filter.js
│   ├── 03_rtc.js
│   ├── 04_feature_extraction.js
│   └── main_workflow.js
│
├── scripts/                     
│   ├── 00_prepare_feature_rasters.py
│   ├── 01_extract_training_features.py
│   ├── 02_model_ablation_experiments.py
│   ├── 03_feature_importance_shap.py
│   ├── 04_train_final_models_for_mapping.py
│   ├── 05_build_yearly_stacks.py
│   └── 06_apply_models_to_yearly_stacks.py
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md

⚙️ Installation
git clone https://github.com/YOURNAME/benchmarking-sentinel1-sar-features.git
cd benchmarking-sentinel1-sar-features

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt

🚀 Pipeline Steps
1️⃣ Sentinel-1 preprocessing (Google Earth Engine)

Run scripts inside /gee in order:

Utilities

Border-noise masking

Refined Lee speckle filter

Ellipsoidal RTC

SAR indices + GLCM textures

Export seasonal mosaics

Output: seasonal multi-band SAR feature rasters.

2️⃣ Prepare feature rasters
python scripts/00_prepare_feature_rasters.py

3️⃣ Extract polygon training features
python scripts/01_extract_training_features.py


Produces LOPO-ready tables.

4️⃣ Feature ablation & model benchmarking
python scripts/02_model_ablation_experiments.py


Benchmarks:

VV/VH only

indices only

textures only

combined sets

full stack

Across RF / SVM / XGB.

5️⃣ SHAP feature importance
python scripts/03_feature_importance_shap.py

6️⃣ Train final mapping models
python scripts/04_train_final_models_for_mapping.py

7️⃣ Build yearly stacks
python scripts/05_build_yearly_stacks.py


Creates stacks for:

2016

2018

2025

8️⃣ Wall-to-wall mapping
python scripts/06_apply_models_to_yearly_stacks.py


Outputs classified woody vegetation maps.

📁 Data & Results Organization
data/
  ├── s1_features/
  ├── wetlands/
  ├── labels/
  └── yearly_stacks/

results/
  ├── models/
  ├── shap/
  ├── predictions/
  └── figures/

🧑‍🤝‍🧑 Contributing

Pull requests and feedback are welcome.

📜 License

MIT License.

📧 Contact

Abdullah Toqeer
PhD Candidate — Charles Sturt University
📩 toqeerabdullah776@gmail.com