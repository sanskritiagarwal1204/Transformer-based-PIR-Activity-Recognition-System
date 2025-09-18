
# Transformer‑based PIR Activity Recognition (PyTorch)



A compact pipeline that classifies **indoor activities** from a **55‑sensor PIR array** using a **Transformer encoder over the sensor axis**, fused with **tabular time/ambient features**.

---

## Dataset

**File:** `pirvision_office_dataset2.csv`  
**Shape:** `7651 × 59`

**Columns (core)**
- **PIR sensors:** `PIR_1 … PIR_55` (one column per sensor).  
- **Label:** integer activity class.  
- **Time meta:** `Date`, `Time` (combined into a `datetime`).  
- **Environment:** `Temperature_F`.

**Derived in the notebook**
- Cyclical encodings: `hour_sin/cos`, `minute_sin/cos`, `day_sin/cos`, `dow_sin/cos`, `month_sin/cos`.  
- Standardization:  
  - sensor block → flatten → `StandardScaler` → reshape to `(seq_len=55, 1)`,  
  - tabular block (time encodings + temperature) → `StandardScaler`. 

**Label distribution (example run)**  
- `{0: 6247, 1: 833, 3: 571}` (imbalanced; consider class weighting or a sampler).

> **Sequence view:** each **row** is treated as a **sequence of 55 tokens** (sensors). This models **spatial** relations across sensors at a point in time, not temporal windows. 

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio numpy pandas scipy scikit-learn matplotlib seaborn tqdm PyYAML
# For CUDA builds, install PyTorch per your CUDA version from pytorch.org
```

---

## Model

**Class:** `TransformerModel(seq_length, num_classes, tabular_dim)`

**Backbone (sensor‑axis Transformer)**
- Per‑token embedding: `Linear(1 → 32)` applied to each of the 55 sensor tokens.  
- Learnable positional embedding: `(1, seq_len=55, 32)`.  
- Encoder: `TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True) × 2`. 

**Fusion & head**
- Global pool over sensor tokens → vector.  
- Concatenate with standardized **tabular** vector.  
- MLP head: `Linear(32 + tabular_dim → 128) → Dropout(0.3) → ReLU → Dropout(0.2) → Linear(128 → num_classes)`. 

**Optimization**
- Loss: `CrossEntropyLoss` (consider class weights).  
- Optimizer: `Adam(lr=1e‑3)`. 

---

## Hyperparameters (defaults)

| Name | Value |
|---|---|
| Sequence length | 55 (sensors as tokens) |
| Embedding dim | 32 |
| Heads | 4 |
| Transformer layers | 2 |
| FFN dim | 128 |
| Transformer dropout | 0.1 |
| Head dropouts | 0.3, 0.2 |
| Batch size | 32 |
| Epochs | 50 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Num classes | 3 |


---

## Data flow (row → tensors)

1. **Sensors**: `[PIR_1…PIR_55]` → standardized → reshape to `(55, 1)`.  
2. **Tabular**: time encodings + `Temperature_F` → standardized → vector.  
3. **Forward**: sensor tokens → Transformer → pooled vector → concat tabular → MLP → logits.  
4. **Metrics**: Accuracy, Precision, Recall, F1; confusion matrix + per‑class scores in the notebook. 

---
