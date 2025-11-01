# IntraMap AI Prototype (UJIIndoorLoc Edition)

> This is a **complete, end-to-end plan** to build an Indoor Positioning AI prototype for **IntraMap** using the **UJIIndoorLoc** dataset. It includes goals, data understanding, preprocessing, baselines, modern models, evaluation, error analysis, model packaging, and mobile integration. It’s written so you can paste sections into Copilot/Cursor and iterate quickly.

---

## 0) What you’ll deliver at the end

* A **reproducible training repo** with:

  * A **hierarchical model**: `Building → Floor → (x, y)` position.
  * A strong **kNN baseline** + an **MLP/GBM** regressor.
  * **Scikit-learn pipelines** to ensure identical preprocessing at train/inference.
  * **Evaluation suite** with building/floor accuracy and 2-D error (meters).
  * Exported models in **ONNX**/**TorchScript** or **.joblib**.
* A **FastAPI microservice** exposing `/predict` (JSON in/out).
* A **mobile-ready inference spec** (Android Wi-Fi scan → features → REST/TFLite).
* A short **error analysis report** + **ablation notes**.

---

## 1) UJIIndoorLoc at a glance

* **Task**: Indoor localization via Wi-Fi RSSI fingerprinting.
* **Features**:

  * ~**520** Wi-Fi Access Points (WAP_*** columns), RSSI in dBm (e.g., `-104` to `0`; **-100** often used for missing).
  * **LATITUDE, LONGITUDE**, **FLOOR**, **BUILDINGID**, plus metadata (`SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`).
* **Targets**:

  1. **Building** (classification)
  2. **Floor** (classification)
  3. **Position** (regression → 2D meters)
* **Files** (already in your workspace):

  * `/mnt/data/TrainingData.csv`
  * `/mnt/data/ValidationData.csv`

---

## 2) Modeling goals & metrics

* **Primary KPIs**

  * **Building Accuracy** (↑)
  * **Floor Accuracy** (↑)
  * **2D Position Error (meters)** (↓): compute on **correct building & floor**; also report overall with penalty.
* **Secondary**

  * **Latency** (ms per inference) and **model size** (edge deployability).
  * **Robustness across devices** (PHONEID) and users (USERID).

**Metric definitions**

```text
- Building Acc = correct building / total
- Floor Acc    = correct floor / total
- 2D Error (m) = Euclidean distance in meters between predicted and true (x,y)
  using a local projection (UTM/ENU) of LAT,LON
- “Strict” 2D Error: only on samples with correct building & floor
- “Overall” 2D Error: use strict error, else apply a large penalty (e.g., 50–100 m)
```

---

## 3) Environment & project skeleton

```bash
# (a) Create environment
uv venv --python 3.11
source .venv/bin/activate

# (b) Install core libs
pip install numpy pandas scikit-learn pyproj joblib xgboost torch fastapi uvicorn pydantic

# (c) Repo structure
intra-map/
  data/
    TrainingData.csv
    ValidationData.csv
  src/
    data_io.py
    features.py
    metrics.py
    baselines.py
    models.py
    train.py
    evaluate.py
    serve.py
    utils_geo.py
  models/
  reports/
  notebooks/
```

---

## 4) Load & quick EDA

> Understand distributions, missing patterns, per-AP coverage, and device heterogeneity.

```python
# src/data_io.py
import pandas as pd
from pathlib import Path

def load_uji(train_path: str, val_path: str):
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    return train, val

if __name__ == "__main__":
    train, val = load_uji("/mnt/data/TrainingData.csv", "/mnt/data/ValidationData.csv")
    print(train.shape, val.shape)
    print(train.columns[:10])
    print(train[['BUILDINGID','FLOOR','LATITUDE','LONGITUDE']].describe())
```

**EDA checklist**

* Count **AP coverage**: for each WAP column, % of samples not equal to missing value (e.g., `-100`/`-110`).
* Distribution of **RSSI** (clip outliers to `[-110, -30]`).
* **Class balance** for `BUILDINGID`, `FLOOR` per building.
* **Device shift**: distributions by `PHONEID`, `USERID`.
* **Time drift** via `TIMESTAMP`.

---

## 5) Geodesy utilities (meters, not degrees)

Convert `LATITUDE, LONGITUDE` to a **local Cartesian** system (UTM or ENU) for regression/evaluation.

```python
# src/utils_geo.py
from pyproj import Transformer
import numpy as np
import pandas as pd

def build_transformer(lat0: float, lon0: float):
    # Local tangent plane (ENU) around campus centroid
    return Transformer.from_crs(
        {"proj":"latlong","datum":"WGS84"},
        {"proj":"aeqd","lat_0":lat0,"lon_0":lon0,"x_0":0,"y_0":0,"units":"m","no_defs":True},
        always_xy=True
    )

def add_xy(df: pd.DataFrame, lat_col='LATITUDE', lon_col='LONGITUDE'):
    lat0 = df[lat_col].median()
    lon0 = df[lon_col].median()
    tfm  = build_transformer(lat0, lon0)
    x, y = tfm.transform(df[lon_col].values, df[lat_col].values)
    df['X_M'] = x
    df['Y_M'] = y
    return df, (lat0, lon0)
```

---

## 6) Preprocessing & feature engineering

**Key rules**

* Treat **missing RSSI** as **very weak** (e.g., `-110`), not zero.
* **Clip** RSSI to `[-110, -30]`.
* **Standardize** per AP (RobustScaler) to reduce device variance.
* Remove **dead APs** (coverage below threshold, e.g., <1–2%).
* Keep **top-K APs** by coverage/variance (e.g., 120–200) for smaller models.
* Add **auxiliary features**:

  * **AP count** above a threshold (e.g., RSSI > -85).
  * **RSSI mean/median/max** per sample.
  * Optional: **per-sample mean centering** of RSSIs (device normalization).
* Targets:

  * `y_building = BUILDINGID`
  * `y_floor    = FLOOR`
  * `y_xy       = (X_M, Y_M)`

```python
# src/features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

WAP_PREFIX = "WAP"

class RssiCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, missing_value=-100, fill_value=-110, min_rssi=-110, max_rssi=-30):
        self.missing_value = missing_value
        self.fill_value    = fill_value
        self.min_rssi      = min_rssi
        self.max_rssi      = max_rssi
        self.cols          = None
    def fit(self, X, y=None):
        self.cols = [c for c in X.columns if c.startswith(WAP_PREFIX)]
        return self
    def transform(self, X):
        X = X.copy()
        X[self.cols] = X[self.cols].replace(self.missing_value, self.fill_value)
        X[self.cols] = X[self.cols].clip(self.min_rssi, self.max_rssi)
        return X

class APSelector(BaseEstimator, TransformerMixin):
    def __init__(self, coverage_min=0.02, top_k=200):
        self.coverage_min = coverage_min
        self.top_k        = top_k
        self.keep_cols    = None
    def fit(self, X, y=None):
        wap_cols = [c for c in X.columns if c.startswith(WAP_PREFIX)]
        coverage = (X[wap_cols] > -110).mean().sort_values(ascending=False)
        eligible = coverage[coverage >= self.coverage_min]
        self.keep_cols = list(eligible.index[:self.top_k])
        return self
    def transform(self, X):
        return X[self.keep_cols]
```

---

## 7) Train/validation split strategy (avoid leakage)

* Use **provided ValidationData.csv** for **final validation**.
* For **in-training CV**, use **GroupKFold** by `PHONEID` or `USERID` to simulate device/user generalization.
* Keep **stratification** for building/floor when possible.

```python
# src/train.py (split sketch)
from sklearn.model_selection import GroupKFold

groups = train_df['PHONEID']  # or USERID
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(train_X, train_y_building, groups=groups):
    X_tr, X_te = train_X.iloc[tr], train_X.iloc[te]
    ...
```

---

## 8) Baselines (must-have)

### 8.1 kNN (cosine or Euclidean on standardized RSSI)

* **Pros**: Simple, strong baseline for Wi-Fi FP.
* **Cons**: Memory/latency ↑ with dataset size.

```python
# src/baselines.py
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from features import RssiCleaner, APSelector

def knn_building_pipeline():
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("clf",   KNeighborsClassifier(n_neighbors=7, metric="cosine"))
    ])

def knn_floor_pipeline():
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("clf",   KNeighborsClassifier(n_neighbors=7, metric="cosine"))
    ])

def knn_xy_pipeline():
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(coverage_min=0.02, top_k=200)),
        ("sc",    RobustScaler(with_centering=True)),
        ("reg",   KNeighborsRegressor(n_neighbors=5, metric="euclidean", weights="distance"))
    ])
```

**Usage**: Train separate models for **building**, **floor**, and **(x,y)** (see §10 hierarchical inference).

---

## 9) Stronger models

### 9.1 Gradient Boosting / XGBoost (tabular SOTA classic)

```python
# src/models.py
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from features import RssiCleaner, APSelector

def xgb_building():
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(top_k=250)),
        ("sc",    RobustScaler()),
        ("xgb",   xgb.XGBClassifier(
                    n_estimators=600, max_depth=8, subsample=0.8, colsample_bytree=0.6,
                    learning_rate=0.05, tree_method="hist", eval_metric="mlogloss"))
    ])

def xgb_floor():
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(top_k=250)),
        ("sc",    RobustScaler()),
        ("xgb",   xgb.XGBClassifier(
                    n_estimators=700, max_depth=10, subsample=0.8, colsample_bytree=0.7,
                    learning_rate=0.05, tree_method="hist", eval_metric="mlogloss"))
    ])

def xgb_xy():
    return Pipeline([
        ("clean", RssiCleaner()),
        ("ap",    APSelector(top_k=300)),
        ("sc",    RobustScaler()),
        ("xgb",   xgb.XGBRegressor(
                    n_estimators=1000, max_depth=10, subsample=0.8, colsample_bytree=0.7,
                    learning_rate=0.05, tree_method="hist", objective="squarederror"))
    ])
```

### 9.2 MLP (neural) for (x, y)

* Add **PHONEID one-hot/embedding** as optional covariate (device adaptation).
* Use **Dropout** + **BatchNorm**.
* Export via **TorchScript** for edge deployment.

```python
# src/models.py (MLP sketch)
import torch, torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=(512,256,128), out_dim=2, p=0.2):
        super().__init__()
        layers=[]
        d=in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(p)]
            d=h
        layers += [nn.Linear(d,out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
```

> You can still keep scikit-learn for preprocessing + AP selection, then pass the NumPy array to PyTorch.

---

## 10) Hierarchical inference (recommended)

1. **Predict Building**
2. **Predict Floor**, optionally **conditioned on building** (e.g., a per-building floor model or masking floors not in predicted building).
3. **Predict (x, y)** using:

   * a **global regressor**; or
   * **per-building** regressors (often better).

```python
# src/evaluate.py (pseudo)
def hierarchical_predict(X_row):
    b = building_model.predict(X_row)[0]
    f = floor_model_b[b].predict(X_row)[0]   # or a single floor_model with post-mask
    xy = reg_model_b[b].predict(X_row)[0]    # per-building regressor
    return b, f, xy
```

---

## 11) Training loop + evaluation

```python
# src/metrics.py
import numpy as np

def meter_error(y_true_xy, y_pred_xy):
    return np.linalg.norm(y_true_xy - y_pred_xy, axis=1)

def strict_position_error(y_true_b, y_pred_b, y_true_f, y_pred_f, xy_true, xy_pred, penalty=100.0):
    ok = (y_true_b==y_pred_b) & (y_true_f==y_pred_f)
    err = meter_error(xy_true, xy_pred)
    strict = err[ok]
    overall = np.where(ok, err, penalty)
    return strict.mean() if len(strict)>0 else np.nan, overall.mean()
```

```python
# src/evaluate.py (sketch)
from sklearn.metrics import accuracy_score

def evaluate_split(building_model, floor_model, reg_model, X, yb, yf, xy):
    b_pred = building_model.predict(X)
    f_pred = floor_model.predict(X)  # or conditional
    xy_pred = reg_model.predict(X)

    b_acc = accuracy_score(yb, b_pred)
    f_acc = accuracy_score(yf, f_pred)

    strict_mean, overall_mean = strict_position_error(yb, b_pred, yf, f_pred, xy, xy_pred)
    return {"building_acc": b_acc, "floor_acc": f_acc,
            "strict_2d_m": strict_mean, "overall_2d_m": overall_mean}
```

---

## 12) Put it together (script skeleton)

```python
# src/train.py (end-to-end sketch)
import pandas as pd
from data_io import load_uji
from utils_geo import add_xy
from baselines import knn_building_pipeline, knn_floor_pipeline, knn_xy_pipeline
from models import xgb_building, xgb_floor, xgb_xy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

train, val = load_uji("/mnt/data/TrainingData.csv", "/mnt/data/ValidationData.csv")

# Add metric space
train, center = add_xy(train)
val, _ = add_xy(val)

# Features/targets
wap_cols = [c for c in train.columns if c.startswith("WAP")]
X_tr = train[wap_cols + ['PHONEID']]  # keep PHONEID if you plan to use it
X_va = val[wap_cols + ['PHONEID']]

y_b_tr = train['BUILDINGID']; y_b_va = val['BUILDINGID']
y_f_tr = train['FLOOR'];      y_f_va = val['FLOOR']
xy_tr   = train[['X_M','Y_M']].values
xy_va   = val[['X_M','Y_M']].values

# Models (start with kNN; swap in XGB later)
building_model = knn_building_pipeline()
floor_model    = knn_floor_pipeline()
xy_model       = knn_xy_pipeline()

building_model.fit(X_tr, y_b_tr)
floor_model.fit(X_tr, y_f_tr)
xy_model.fit(X_tr, xy_tr)

# Evaluate
from evaluate import evaluate_split
metrics = evaluate_split(building_model, floor_model, xy_model,
                         X_va, y_b_va, y_f_va, xy_va)
print(metrics)

# Save
dump(building_model, "models/building_knn.joblib")
dump(floor_model,    "models/floor_knn.joblib")
dump(xy_model,       "models/xy_knn.joblib")
```

---

## 13) Error analysis (tighten the loop)

* **Confusion matrices** for building & floor.
* **Floor-wise and building-wise error maps** (scatter true vs predicted).
* **Error vs RSSI density** (fewer strong APs → larger error).
* **Per-device** breakdown (PHONEID) → detect domain shift.
* **Time** trends (TIMESTAMP) → drift.
* **Spatial KDE** of residuals → “blind spots”.

**Quick plots to add in a notebook**

* `sns.heatmap(confusion_matrix(...))` (or matplotlib).
* Scatter: true `(X_M, Y_M)` vs residual vectors.
* Histograms of 2D error per building/floor.

---

## 14) Ablations (what to test)

* **AP selection**: top_k ∈ {100, 150, 200, 300}.
* **Scaler**: Standard vs Robust vs per-sample mean centering.
* **Distance metric** in kNN: cosine vs euclidean, `weights="distance"`.
* **Hierarchical design**: global vs per-building floor/regressors.
* **Model**: kNN vs XGBoost vs MLP.
* **Penalty** for wrong building/floor (50 vs 100 m) for overall metric.

Keep a simple table in `reports/ablations.md`.

---

## 15) Packaging for inference

### 15.1 FastAPI server

```python
# src/serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()
b_model = joblib.load("models/building_knn.joblib")
f_model = joblib.load("models/floor_knn.joblib")
xy_model= joblib.load("models/xy_knn.joblib")

class WifiSample(BaseModel):
    # Dict of { "WAP001": -78, "WAP002": -110, ... } with missing omitted
    rssi: dict
    phone_id: int | None = None

@app.post("/predict")
def predict(sample: WifiSample):
    # Build a single-row dataframe matching training columns
    # (Ensure same AP list & preprocessing. If AP missing → fill -110)
    import pandas as pd
    wap_cols = [c for c in b_model.named_steps['ap'].keep_cols]  # example access
    row = {c: sample.rssi.get(c, -110) for c in wap_cols}
    if sample.phone_id is not None:
        row['PHONEID'] = sample.phone_id
    X = pd.DataFrame([row])

    b = int(b_model.predict(X)[0])
    f = int(f_model.predict(X)[0])
    xy = xy_model.predict(X)[0].tolist()
    return {"building": b, "floor": f, "x_m": xy[0], "y_m": xy[1]}
```

Run:

```bash
uvicorn src.serve:app --reload --port 8080
```

### 15.2 ONNX / TorchScript (optional)

* For scikit-learn/XGBoost: `skl2onnx` or `onnxmltools`, then use **onnxruntime**.
* For PyTorch MLP:

  ```python
  scripted = torch.jit.script(model)
  scripted.save("models/xy_mlp.ts")
  ```

---

## 16) Android integration (IntraMap app)

**Client pipeline**

1. **Scan Wi-Fi** (Android `WifiManager` / `WifiScanner`):

   * Collect `(BSSID → RSSI dBm)`.
   * Map to your **training AP schema (WAP columns)**.
2. **Preprocess on device**

   * Missing → `-110`
   * Clip to `[-110, -30]`
   * (Optional) mean-center per sample
3. **Inference**

   * **Option A**: POST to FastAPI `/predict` (quick to prototype).
   * **Option B**: Embed **TFLite/ONNX** model (edge, offline, low-latency).
4. **Map rendering**

   * Convert `(x_m, y_m)` to **floor plan coordinates**.
   * Show **Building/Floor** UI and drop a marker.
   * Add **confidence** (e.g., kNN neighbor vote entropy / regressor predictive std).

**Consistency rule**: **Use the exact same preprocessing** paths at inference as in training.

---

## 17) Domain gap → IntraMap (Building 10) plan

UJIIndoorLoc ≠ your campus. To transfer:

* **Few-shot calibration**: collect ~**300–1000** labeled points in Building 10 (grid 2–3 m).
* **AP remapping**: build a **mapping** from live BSSID to a **stable AP index space** (WAP_***). Maintain a dictionary and keep it versioned.
* **Standardization**:

  * Subtract per-sample mean RSSI (reduces device offsets).
  * RobustScaler learned on **IntraMap** data (not UJI).
* **Fine-tune models** on your labeled IntraMap set.
* **Semi-supervised**: record unlabeled scans along user paths; use pseudo-labels from strong kNN to expand training.
* **Floor-map constraints**: snap predictions to reachable corridors/rooms (basic map-matching post-process).

---

## 18) Data collection spec (for your on-site phase)

* **Grid**: hallways 2–3 m; intersections & narrow areas 1–2 m.
* **At each point** (≥10 s dwell):

  * 8–10 scans; store **RSSI vector**, `TIMESTAMP`, `DEVICE`, **(lat, lon or (x,y) on map)**
  * Phone **orientations**: front/left/right/back (optional)
  * Record **AP count > -85** & strongest AP ID (for QC)
* **Metadata**: building, floor, room/area label.
* **QC**: re-measure when **AP count < 8** or max RSSI weaker than `-85` dBm (possible dead zone).

---

## 19) Quality gates to “ship the prototype”

* **≥ 98%** Building Acc
* **≥ 90–95%** Floor Acc (varies by dataset)
* **Strict 2D Error ≤ 6–8 m** median (UJI-level)
* **P50 latency ≤ 15 ms**, **P95 ≤ 50 ms** (server), or **≤ 10 ms** (edge)
* **Model size ≤ 10–20 MB** for mobile (prefer ONNX/TFLite/quantization)

---

## 20) Quickstart notebook cells (paste into Copilot)

> Replace model choices (kNN ↔ XGB/MLP) as you iterate.

```python
import pandas as pd, numpy as np
from joblib import dump, load
from data_io import load_uji
from utils_geo import add_xy
from baselines import knn_building_pipeline, knn_floor_pipeline, knn_xy_pipeline
from evaluate import evaluate_split

train, val = load_uji("/mnt/data/TrainingData.csv", "/mnt/data/ValidationData.csv")
train, center = add_xy(train)
val, _ = add_xy(val)

wap_cols = [c for c in train.columns if c.startswith("WAP")]
feat_cols = wap_cols + ['PHONEID']  # keep PHONEID if you want device-aware features

X_tr = train[feat_cols]; X_va = val[feat_cols]
y_b_tr = train['BUILDINGID']; y_b_va = val['BUILDINGID']
y_f_tr = train['FLOOR'];      y_f_va = val['FLOOR']
xy_tr   = train[['X_M','Y_M']].values
xy_va   = val[['X_M','Y_M']].values

b_model = knn_building_pipeline().fit(X_tr, y_b_tr)
f_model = knn_floor_pipeline().fit(X_tr, y_f_tr)
xy_model= knn_xy_pipeline().fit(X_tr, xy_tr)

metrics = evaluate_split(b_model, f_model, xy_model, X_va, y_b_va, y_f_va, xy_va)
metrics
```

---

## 21) Production hardening (when you iterate)

* **Monitoring**:

  * Track **distribution drift** of RSSI and AP coverage vs training stats.
  * Log **latency**, **error proxy** (e.g., neighbor distances), and **floor mismatch rates**.
* **Versioning**:

  * Version **AP dictionary**, **preprocessing** pipelines, and **models** together.
  * Store all with a semantic version: `intra-map-aps@v1.2 + preproc@v0.3 + model@v0.9`.
* **Fallbacks**:

  * If **building entropy** is high → request a **rescan** or widen k in kNN.
  * If **AP count** < threshold → switch to last good location, or sensor fusion (later).
* **Security/Privacy**:

  * No MAC addresses or PII stored long-term.
  * Aggregate AP stats; rotate device identifiers.

---

## 22) Roadmap beyond v1

* **Sensor fusion**: add inertial dead-reckoning (IMU), barometer (floor), BLE beacons.
* **Graph constraints**: map-matching via A* / HMM on a corridor graph.
* **Sequence models**: Kalman Filter / Particle Filter / LSTM over scan sequences.
* **Few-shot domain adaptation**: CORAL/feature whitening per device; learn a small **PHONEID embedding**.

---

## 23) Checklist (copy/paste)

* [ ] Load UJI data & add (x,y) meters
* [ ] Preprocess RSSI (fill/clip/scale)
* [ ] Select APs (coverage/variance)
* [ ] Train **Building** classifier (kNN/XGB)
* [ ] Train **Floor** classifier (global or per-building)
* [ ] Train **(x,y)** regressor (per-building preferred)
* [ ] Evaluate on ValidationData.csv (report 4 metrics)
* [ ] Error analysis (confusions, maps, device breakdown)
* [ ] Export models + lock preprocessing
* [ ] Wrap FastAPI `/predict`
* [ ] Android client integration spec
* [ ] Plan few-shot calibration for Building 10

---

### Notes for Copilot

* Use the **pipelines** above to ensure **identical preprocessing** at train & inference.
* Start with **kNN** (fast to validate), then swap in **XGBoost**/**MLP** where helpful.
* Keep the **AP selection** transformer state (the selected AP list) with the model artifact.