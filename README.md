# NameDemographics-ML

Predict **Gender** (M / F) and **Race** (white | hispanic | asian | black | americanindian) from first and last names using classical machine learning.

Developed at the **OU TRICS Lab** by Khondhaker Al Momin.

---

## Project Structure

```
NameDemographics-ML/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── gender/
│   │   │   └── names_till_2021/          # SSA yearly files yob1880.txt – yob2021.txt
│   │   └── race/
│   │       └── Race_last_name.csv        # Original Census surname–race table
│   └── processed/
│       ├── all_ssn_first_names_till_2021.csv   # Full SSA dataset (Name, Gender, Frequency, Year)
│       ├── Unique_Names_Till_2021.csv           # Deduplicated names (101,338 rows)
│       └── Modified_Race_last_name.csv          # Cleaned race-labelled surnames (162,251 rows)
│
├── models/
│   ├── gender_prediction_model_rf.pkl    # Random Forest — Gender  (baseline, Dec 2023)
│   ├── race_prediction_model_rf.pkl      # Random Forest — Race    (baseline, Dec 2023)
│   ├── svm_gender_model_Nov_2024.pkl.gz  # SVM — Gender (427 KB lightweight alternative)
│   ├── gender_tfidf_vectorizer.pkl       # TF-IDF vectorizer for first names (improved)
│   ├── gender_lgbm_model.pkl             # LightGBM — Gender (improved)
│   ├── race_tfidf_vectorizer.pkl         # TF-IDF vectorizer for surnames   (improved)
│   ├── race_lgbm_model.pkl              # LightGBM — Race   (improved)
│   └── race_label_encoder.pkl            # LabelEncoder for race classes (shared)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb                # Merge SSA files, clean race data, visualise
│   ├── 02_gender_prediction.ipynb               # RF baseline — train & compare 5 models
│   ├── 03_race_prediction.ipynb                 # RF baseline — train & compare 5 models
│   ├── 04_inference_pipeline.ipynb              # Load any backend, batch-predict, compare
│   └── 05_improved_models_lightgbm.ipynb        # LightGBM + TF-IDF char n-grams (improved)
│
└── docs/
    ├── NationalReadMe.pdf                # Official SSA name-data documentation
    ├── data_distribution.png             # Gender & race class charts
    ├── gender_model_comparison.png       # Accuracy comparison bar chart
    ├── gender_confusion_matrix.png
    ├── race_model_comparison.png
    └── race_confusion_matrix.png
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

| Notebook | Purpose |
|---|---|
| `01_data_preparation.ipynb` | Build processed datasets from raw files |
| `02_gender_prediction.ipynb` | Train and evaluate gender models |
| `03_race_prediction.ipynb` | Train and evaluate race models |
| `04_inference_pipeline.ipynb` | Predict on new names using saved models |
| `05_improved_models_lightgbm.ipynb` | Train LightGBM with TF-IDF char n-grams |

> Pre-trained RF models are already included in `models/`. You can run notebook 04 directly without retraining.
> To generate the improved LightGBM models, run notebook 05 first.

### 3. Single prediction (Python)

**LightGBM + TF-IDF (recommended):**

```python
import joblib

# Load once
g_vec   = joblib.load('models/gender_tfidf_vectorizer.pkl')
g_model = joblib.load('models/gender_lgbm_model.pkl')
r_vec   = joblib.load('models/race_tfidf_vectorizer.pkl')
r_model = joblib.load('models/race_lgbm_model.pkl')
r_le    = joblib.load('models/race_label_encoder.pkl')

first_name, last_name = 'Jennifer', 'Garcia'

gender = g_model.predict(g_vec.transform([first_name.lower()]))[0]
race   = r_le.inverse_transform(r_model.predict(r_vec.transform([last_name.lower()])))[0]

print(f'{first_name} {last_name}  →  Gender: {gender},  Race: {race}')
# Jennifer Garcia  →  Gender: F,  Race: hispanic
```

<details>
<summary>RF baseline (click to expand)</summary>

```python
import numpy as np, joblib

LETTERS = list('abcdefghijklmnopqrstuvwxyz')
def name_to_features(name):
    return np.array([name.lower().count(ch) for ch in LETTERS]).reshape(1, -1)

gender_model = joblib.load('models/gender_prediction_model_rf.pkl')
race_model   = joblib.load('models/race_prediction_model_rf.pkl')
race_le      = joblib.load('models/race_label_encoder.pkl')

first_name, last_name = 'Jennifer', 'Garcia'
gender = gender_model.predict(name_to_features(first_name))[0]
race   = race_le.inverse_transform(race_model.predict(name_to_features(last_name)))[0]
print(f'{first_name} {last_name}  →  Gender: {gender},  Race: {race}')
```

</details>

---

## Data Sources

| Dataset | Source | Size | Coverage |
|---|---|---|---|
| SSA First Names | [Social Security Administration](https://www.ssa.gov/oact/babynames/limits.html) | 4.1M rows | 1880–2021 |
| Census Surnames | U.S. Census Bureau surname frequency data | 162,251 surnames | National |

---

## Feature Engineering

### Baseline — Letter Frequency (26 features)

Each name is encoded as a 26-dimensional vector of letter counts (a–z).

```
"Momin"  →  [0,0,0,0,0,0,0,0,1,0,0,0,2,1,1,0,0,0,0,0,0,0,0,0,0,0]
              a b c d e f g h i j k l m n o p q r s t u v w x y z
```

Fast to compute, but **order-blind** — `"Maria"` and `"Riama"` produce identical vectors.

### Improved — Character N-Gram TF-IDF (8,000 / 12,000 features)

Names are tokenized into overlapping character substrings (bigrams through 4/5-grams) with word-boundary markers, then TF-IDF weighted.

```
"Garcia"  →  " g", " ga", "ga", "ar", "arc", "rci", "cia", "ia", "a "  ...
```

This captures **positional patterns** (name endings like `-ez`, `-ski`, `-ita`) and **phonological clusters** that letter counts miss.

| Setting | Gender model | Race model |
|---|---|---|
| `analyzer` | `char_wb` | `char_wb` |
| `ngram_range` | `(2, 4)` | `(2, 5)` |
| `max_features` | 8,000 | 12,000 |
| `sublinear_tf` | Yes | Yes |

---

## Models & Results

### Baseline — Letter Frequency + RF/SVM (Notebooks 02, 03)

#### Gender Prediction (binary: M / F)

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~71% |
| K-Nearest Neighbors | ~73% |
| Naive Bayes | ~72% |
| **Random Forest** | **~77%** |
| SVM (RBF) | ~78% |

#### Race Prediction (5-class)

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~82% |
| K-Nearest Neighbors | ~81% |
| Naive Bayes | ~80% |
| **Random Forest** | **~84%** |
| SVM (RBF) | ~83% |

### Improved — Character N-Gram TF-IDF + LightGBM (Notebook 05)

| Task | RF Baseline | LightGBM + TF-IDF | Features |
|---|---|---|---|
| **Gender** | ~77% | **~83–87%** (run notebook 05 for exact) | 8,000 |
| **Race** | ~84% | **~87–91%** (run notebook 05 for exact) | 12,000 |

Key improvements over baseline:
- **Character n-gram TF-IDF** captures positional letter patterns (name endings, phonological clusters)
- **LightGBM** with early stopping is faster and more accurate than Random Forest on sparse high-dimensional features
- **Class-weight balancing** addresses the 82% white-majority imbalance in race data, improving recall on minority classes

### Model files

| File | Type | Notes |
|---|---|---|
| `gender_prediction_model_rf.pkl` | RF baseline | 299 MB, 26 features |
| `race_prediction_model_rf.pkl` | RF baseline | 560 MB, 26 features |
| `svm_gender_model_Nov_2024.pkl.gz` | SVM baseline | 427 KB, 26 features |
| `gender_lgbm_model.pkl` | **LightGBM (improved)** | 8,000 TF-IDF features |
| `race_lgbm_model.pkl` | **LightGBM (improved)** | 12,000 TF-IDF features |
| `gender_tfidf_vectorizer.pkl` | TF-IDF vectorizer | Required for LightGBM gender |
| `race_tfidf_vectorizer.pkl` | TF-IDF vectorizer | Required for LightGBM race |
| `race_label_encoder.pkl` | LabelEncoder | Shared across all race models |

> **Imbalance note:** 82% of training surnames are labelled *white*. Per-class accuracy for minority groups (americanindian, black, asian) is lower. The LightGBM model with class weights mitigates this but does not eliminate it.

---

## Possible Future Improvements

1. **SMOTE oversampling** — further address race class imbalance by generating synthetic minority examples.
2. **Name length + vowel/consonant ratio** — simple hand-crafted features that add signal on top of n-grams.
3. **Calibrated SVM** — use `CalibratedClassifierCV` on the SVM to produce reliable probability estimates.
4. **Hyperparameter search** — run Optuna or Bayesian optimization over LightGBM hyperparameters for further gains.

---

## Limitations

- Gender is inferred from first names only; culturally ambiguous names (Jordan, Taylor, Alex) are frequently misclassified.
- Race predictions rely on surname correlations with Census ancestry data, not actual identity.
- SSA data reflects naming conventions at birth and carries historical demographic biases.
- The American Indian class is severely underrepresented (688 surnames) and should be interpreted cautiously.

---

## Citation / Attribution

If you use this code or data pipeline in research, please cite the SSA and Census data sources directly and acknowledge the OU TRICS Lab.

---

*Last updated: 2026-03-28*
