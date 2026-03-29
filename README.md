# Race-Gender Prediction Model

Predict **Gender** (M / F) and **Race** (white | hispanic | asian | black | americanindian) from first and last names using classical machine learning.

Developed at the **OU TRICS Lab** by Khondhaker Al Momin and Arif Sadri.

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
│   │   │   └── names_till_2021/              # SSA yearly files yob1880.txt – yob2021.txt
│   │   └── race/
│   │       └── Race_last_name.csv            # Original Census surname–race table
│   └── processed/
│       ├── all_ssn_first_names_till_2021.csv  # Full merged SSA dataset
│       ├── Unique_Names_Till_2021.csv         # Deduplicated first names (101,338 rows)
│       └── Race_last_name.csv                 # Cleaned race-labelled surnames (162,251 rows)
│
├── models/
│   ├── gender_tfidf_vectorizer.pkl            # TF-IDF vectorizer for first names
│   ├── gender_lgbm_model.pkl                  # LightGBM — Gender
│   ├── race_tfidf_vectorizer.pkl              # TF-IDF vectorizer for surnames
│   ├── race_lgbm_model.pkl                    # LightGBM — Race (unweighted)
│   ├── race_lgbm_model_balanced.pkl           # LightGBM — Race (balanced, better minority recall)
│   └── race_label_encoder.pkl                 # LabelEncoder for race classes
│
├── notebooks/
│   ├── 01_data_preparation.ipynb              # Merge SSA files, clean race data, visualise
│   ├── 02_gender_prediction.ipynb             # Baseline — compare 5 models (letter frequency)
│   ├── 03_race_prediction.ipynb               # Baseline — compare 5 models (letter frequency)
│   ├── 04_inference_pipeline.ipynb            # Load LightGBM models, batch-predict, demo
│   └── 05_improved_models_lightgbm.ipynb      # LightGBM + TF-IDF char n-grams (production)
│
└── docs/
    ├── NationalReadMe.pdf                     # Official SSA name-data documentation
    ├── gender_lgbm_confusion_matrix.png       # Gender confusion matrix (LightGBM)
    ├── gender_ngram_importance.png            # Top discriminative n-grams by gender
    ├── race_all_confusion_matrices.png        # Race confusion matrices (RF vs LGBM variants)
    ├── race_lgbm_confusion_matrix.png         # Race confusion matrix (LightGBM)
    └── lgbm_vs_rf_comparison.png              # Accuracy & macro recall comparison chart
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
| `02_gender_prediction.ipynb` | Baseline exploration: 5 models on letter-frequency features |
| `03_race_prediction.ipynb` | Baseline exploration: 5 models on letter-frequency features |
| `04_inference_pipeline.ipynb` | Predict on new names using saved LightGBM models |
| `05_improved_models_lightgbm.ipynb` | Train LightGBM with TF-IDF char n-grams (production models) |

> Pre-trained LightGBM models are included in `models/`. You can run notebook 04 directly without retraining.
> To retrain, run notebooks 01 → 05 in order.

### 3. Single prediction (Python)

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

### Baseline — Letter Frequency + Classical ML (Notebooks 02, 03)

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

### Production — Character N-Gram TF-IDF + LightGBM (Notebook 05)

| Task | Baseline (RF) | LightGBM + TF-IDF | Features |
|---|---|---|---|
| **Gender** | ~77% | **~83–87%** (run notebook 05 for exact) | 8,000 |
| **Race** | ~84% | **~85–88%** (run notebook 05 for exact) | 12,000 |

Key improvements over baseline:
- **Character n-gram TF-IDF** captures positional letter patterns (name endings, phonological clusters)
- **LightGBM** with early stopping is faster and more accurate than Random Forest on sparse high-dimensional features
- **Balanced variant** with class weights improves recall on minority classes at the cost of some overall accuracy

### Model files

| File | Type | Notes |
|---|---|---|
| `gender_lgbm_model.pkl` | LightGBM | Gender classifier, 8,000 TF-IDF features |
| `race_lgbm_model.pkl` | LightGBM | Race classifier (unweighted), 12,000 TF-IDF features |
| `race_lgbm_model_balanced.pkl` | LightGBM | Race classifier (balanced), better minority recall |
| `gender_tfidf_vectorizer.pkl` | TF-IDF vectorizer | Required for gender prediction |
| `race_tfidf_vectorizer.pkl` | TF-IDF vectorizer | Required for race prediction |
| `race_label_encoder.pkl` | LabelEncoder | Maps encoded labels to race strings |

> **Imbalance note:** 82% of training surnames are labelled *white*. Per-class accuracy for minority groups (americanindian, black, asian) is lower. The balanced LightGBM variant with class weights mitigates this but does not eliminate it.

---

## Possible Future Improvements

1. **SMOTE oversampling** — further address race class imbalance by generating synthetic minority examples.
2. **Additional features** — name length, vowel/consonant ratio, and positional character embeddings.
3. **Hyperparameter search** — use Optuna or Bayesian optimization over LightGBM hyperparameters for further gains.
4. **Deep learning** — character-level CNN or BiLSTM for sequence-aware name encoding.

---

## Limitations

- Gender is inferred from first names only; culturally ambiguous names (Jordan, Taylor, Alex) are frequently misclassified.
- Race predictions rely on surname correlations with Census ancestry data, not actual identity.
- SSA data reflects naming conventions at birth and carries historical demographic biases.
- The American Indian class is severely underrepresented (688 surnames) and should be interpreted cautiously.

---

## Citation

If you use this repository in your research, please cite it as:

> Momin, K. A., & Sadri, A. (2026). *Race and Gender Prediction from Names Using Machine Learning* [Computer software]. OU TRICS Lab. https://github.com/khondhaker/Race-Gender-Prediction

**BibTeX:**

```bibtex
@software{momin2026racegender,
  author       = {Momin, Khondhaker Al and Sadri, Arif},
  title        = {Race and Gender Prediction from Names Using Machine Learning},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/khondhaker/Race-Gender-Prediction},
  note         = {OU TRICS Lab}
}
```

### Data Sources

Please also cite the original data providers:

> U.S. Social Security Administration. (2022). *Beyond the top 1000 names: National data* [Data set]. https://www.ssa.gov/oact/babynames/limits.html

> U.S. Census Bureau. (2010). *Frequently occurring surnames from the 2010 Census* [Data set]. https://www.census.gov/topics/population/genealogy/data/2010_surnames.html

---

*Last updated: 2026-03-28*
