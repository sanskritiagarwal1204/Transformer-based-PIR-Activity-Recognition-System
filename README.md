
# PIRvision Presence Detection - Team 16

This repository contains the final submission for the PIRvision FoG Presence Detection Challenge. The notebook implements end-to-end training, evaluation, and visualization for the multi-class occupancy classification task using PIR sensor data, temporal features, and a Transformer-based deep learning model.

# Submitted by: Team 16
## Team members:
#### **Sakshi Badole** : ***CS24MTECH11008***
#### **Sanskriti Agarwal** : ***CS24MTECH14002***
#### **Aviraj Antala** : ***CS24MTECH14011***

---

## Project Overview

- **Dataset**: PIRvision Office Dataset (provided)
- **Goal**: Predict presence class (0, 1, 3) using PIR sensor readings and temporal features.
- **Model**: Transformer-based hybrid model combining time series and tabular features.
- **Evaluation**: 5-fold cross-validation and final train-test evaluation with detailed metrics.

---

## Folder Structure and Required Files

Ensure the following files are uploaded in the working directory before starting:

```
pirvision_office_dataset2.csv
team_16.pth
team_16_label_encoder.joblib
team_16_scaler_tab.joblib
team_16_scaler_ts.joblib
team_16.ipynb
```

---

## Dependencies

Install the required Python libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib torch torchvision
```

---

## How to Run

1. Open `team_16.ipynb` in Google Colab.
2. Execute cells **sequentially** from the top **or jump directly to evaluation**.
3. If only running evaluation, upload the following:
   - `pirvision_office_dataset2.csv`
   - `team_16.pth`
   - `team_16_label_encoder.joblib`
   - `team_16_scaler_ts.joblib`
   - `team_16_scaler_tab.joblib`
4. Run the evaluation section titled:
   ```
   # Run the following cells for evaluation
   ```

---

## Evaluation from Saved Model

The evaluation is handled using the `evaluate_model()` function.

Example usage:

```python
from joblib import load
label_encoder = load("team_16_label_encoder.joblib")
scaler_ts = load("team_16_scaler_ts.joblib")
scaler_tab = load("team_16_scaler_tab.joblib")

accuracy = evaluate_model(
    datafile_location="pirvision_office_dataset2.csv",
    checkpoint="team_16.pth",
    label_encoder=label_encoder,
    scaler_ts=scaler_ts,
    scaler_tab=scaler_tab,
    seq_length=55  # Number of PIR sensors
)
print(f"Evaluation Accuracy: {accuracy:.4f}")
```

---

## Output

The notebook prints:

- Accuracy and Macro F1 score
- Class-wise Precision, Recall, F1
- Confusion matrix
- Training/validation curves

All visualizations are included in the notebook itself.

---

## Notes

- Do not change file names to avoid loading errors.
- Model was trained on sequence length 55 (PIR_1 to PIR_55).
- Early stopping is implemented during training.

---

Team 16 - PIRvision Challenge Final Submission
