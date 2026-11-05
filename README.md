# PCS26/84 â€” DEEP LEARNING-BASED FAKE NEWS DETECTION

**TruthLens AI** â€” B.Tech (CSE) Final-Year Project
Integral University, Lucknow

A research-oriented application for binary text classification of news articles using machine learning and deep learning.

**Label mapping:** Fake = 0, Real = 1

> **Important limitation:** The models predict labels learned from the research dataset. They do not independently verify factual claims or establish whether an article is true or false. The dataset has source-related confounding, so reported performance must not be interpreted as real-world fact-checking accuracy.

## 1. Project Overview

The project investigates machine learning and deep learning approaches for identifying patterns associated with fake and real news articles.

The research pipeline consists of five Jupyter notebooks:

1. Dataset inspection, cleaning and exploratory data analysis.
2. Classical machine learning baselines and leakage-aware experiments.
3. LSTM and BiLSTM model development.
4. DistilBERT transformer experiment.
5. Model evaluation, error analysis and explainability.

The project also includes **TruthLens AI**, a web application with a FastAPI backend and an HTML, CSS and JavaScript frontend.

## 2. Dataset

The project uses the ISOT Fake News Dataset.

### Dataset files

The `dataset/` directory contains:

- `Fake.csv`
- `True.csv`
- `combined_fake_news.csv`
- `combined_fake_news_clean.csv`

The original raw dataset contains:

| Class | Records |
|---|---:|
| Fake | 23,481 |
| Real | 21,417 |

After cleaning:

- 631 records with empty article bodies were removed (630 Fake).
- 209 exact duplicate records were removed.
- Final cleaned dataset: **44,058 records**.

### Data splitting

The saved group-aware split manifest contains:

| Partition | Records |
|---|---:|
| Training | 30,796 |
| Validation | 6,652 |
| Test | 6,610 |
| **Total** | **44,058** |

The split groups identical title-and-body records to reduce duplicate leakage between partitions. The saved manifest is located at:

`results/tables/data_split_manifest.csv`

Use the existing manifest rather than regenerating the split.

**Important preprocessing decisions:**

- `subject` and `date` are not used as model inputs because they can act as label proxies.
- The cleaned dataset and saved split manifest should remain consistent across experiments.
- Notebook 01 documents dataset quality, label assignment and cleaning.

Dataset reference:

Ahmed, H., Traore, I., and Saad, S. (2017). *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques.* Lecture Notes in Computer Science, pp. 127â€“138. https://doi.org/10.1007/978-3-319-69155-8_9

## 3. Technologies Used

| Area | Technologies |
|---|---|
| Programming | Python |
| Data analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Deep learning | PyTorch |
| Transformer experiment | Hugging Face Transformers |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Development | Jupyter Notebook, VS Code |

## 4. Project Structure

```text
Fake-news detection/
â”‚
â”œâ”€â”€ backend/
â”‚   â””â”€â”€ main.py
â”‚
â”œâ”€â”€ dataset/
â”‚   â”œâ”€â”€ Fake.csv
â”‚   â”œâ”€â”€ True.csv
â”‚   â”œâ”€â”€ combined_fake_news.csv
â”‚   â””â”€â”€ combined_fake_news_clean.csv
â”‚
â”œâ”€â”€ frontend/
â”‚   â”œâ”€â”€ index.html
â”‚   â”œâ”€â”€ style.css
â”‚   â””â”€â”€ script.js
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ phase1/
â”‚   â”‚   â””â”€â”€ ml_selected_pipeline.joblib
â”‚   â””â”€â”€ bilstm_best_corrected.pt
â”‚
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_dataset_audit_eda.ipynb
â”‚   â”œâ”€â”€ 02_ml_baselines.ipynb
â”‚   â”œâ”€â”€ 03_lstm_bilstm.ipynb
â”‚   â”œâ”€â”€ 04_transformer_experiment.ipynb
â”‚   â””â”€â”€ 05_evaluation_explainability.ipynb
â”‚
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ figures/
â”‚   â””â”€â”€ tables/
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ fakenews_utils.py
â”‚   â””â”€â”€ fakenews_dl.py
â”‚
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â”œâ”€â”€ combine_dataset.py
â””â”€â”€ inspect_dataset.py
```

Additional model checkpoints and experiment artefacts are stored under `models/` and `results/`.

## 5. Environment Setup

The project was developed and tested in a Windows virtual environment.

```powershell
cd "C:\Users\mi\Desktop\Fake-news detection"

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For a CPU-only PyTorch installation, if required:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The project uses Jupyter notebooks. Select the appropriate virtual-environment kernel in VS Code.

## 6. Notebook Execution Order

| No. | Notebook | Purpose |
|---|---|---|
| 01 | `01_dataset_audit_eda.ipynb` | Dataset inspection, cleaning, EDA and split verification |
| 02 | `02_ml_baselines.ipynb` | Logistic Regression, Linear SVM and other ML baselines |
| 03 | `03_lstm_bilstm.ipynb` | LSTM and BiLSTM training and evaluation |
| 04 | `04_transformer_experiment.ipynb` | DistilBERT feasibility and fine-tuning experiment |
| 05 | `05_evaluation_explainability.ipynb` | Consolidated evaluation, error analysis and explainability |

Recommended order: 01 â†’ 02 â†’ 03 â†’ 04 â†’ 05.

**Transformer status:** Notebook 04 includes a completed DistilBERT pilot fine-tuning experiment using Google Colab GPU resources. The pilot validation results are documented below. Full-data training and final test evaluation remain pending.

Final-test cells should only be executed according to their notebook instructions, after the relevant models and configurations are frozen.
## 7. Model Evaluation Results

The project reports validation and final-test results separately. Validation data was used for model selection, while the final test set was evaluated after the relevant models and configurations were frozen.

### 7.1 Selected Linear SVM V3 — Validation Results

The V3 Linear SVM is the selected machine-learning pipeline used by the TruthLens AI application API.

| Metric | Validation Score |
|---|---:|
| Accuracy | 99.29% |
| Fake Precision | 99.51% |
| Fake Recall | 99.15% |
| Macro F1-score | 99.29% |
| ROC-AUC | 99.97% |

**Model artifact:**

`models/phase1/ml_selected_pipeline.joblib`

The model was selected using validation performance. The final test set was not used for Phase 1 model selection.

### 7.2 Verified Phase 1 Final-Test Comparison

The following results were obtained on the same held-out final test set of 6,609 eligible examples.

| Model / Experiment | Accuracy | Macro F1-score | ROC-AUC |
|---|---:|---:|---:|
| Linear SVM V3 (full text) | 99.20% | 99.20% | 99.96% |
| BiLSTM (frozen Phase 1 checkpoint) | 98.94% | 98.94% | 99.95% |
| Linear SVM (information-matched, first 200 tokens) | 98.71% | 98.71% | 99.89% |

**Best verified final-test accuracy:** Linear SVM V3, approximately 99.20%.

The full-text Linear SVM V3 pipeline is the model currently served by the TruthLens AI API. The BiLSTM and information-matched SVM are separate research comparisons and are not used by the live prediction endpoint.

The comparison results are saved in:

- `results/tables/p1_final_test_comparison.csv`
- `results/tables/p1_final_test_comparison.json`
- `results/tables/p1_final_test_predictions.csv`

### 7.3 DistilBERT — Pilot Validation Experiment

A pretrained DistilBERT model (`distilbert-base-uncased`) was fine-tuned using Google Colab with a Tesla T4 GPU.

| Experiment Detail | Value |
|---|---:|
| Training samples | 500 |
| Validation samples | 300 |
| Epochs | 2 |
| Maximum sequence length | 256 |
| Validation accuracy | 95.00% |
| Macro F1-score | 94.99% |
| ROC-AUC | 99.14% |
| Average precision | 99.04% |

**Experiment scope and limitation:**

- These metrics are from a pilot subset, not the complete validation dataset.
- The final test set was not evaluated.
- These results must not be directly compared with models evaluated on all 6,643 validation rows.
- The trained checkpoint and tokenizer are stored in `models/transformer_distilbert/`.
- Detailed metrics, predictions, training history and plots are available under `results/tables/` and `results/figures/`.
- Full-data fine-tuning and final test evaluation remain pending.

### 7.4 Model Selection and Reporting

- Linear SVM V3 is the selected model used by the TruthLens AI API.
- The Phase 1 BiLSTM is reported as a separate deep-learning comparison.
- Final-test results are reported separately from validation results.
- DistilBERT pilot results are reported separately because they use a 500-row training subset and 300-row validation subset.
- The corrected BiLSTM checkpoint is retained as a separate experiment and should not be confused with the frozen Phase 1 BiLSTM checkpoint used in the comparison.

**Dataset limitation:** The dataset contains strong associations between subject categories and labels. Although subject and date are excluded from model inputs, publisher and source-related language patterns may still influence predictions. Therefore, high dataset scores should not be interpreted as equivalent real-world fact-checking accuracy.


## 8. TruthLens AI Web Application

TruthLens AI provides an interface for submitting news articles and viewing model predictions.

### Frontend

The frontend is implemented using HTML, CSS and JavaScript.

It contains five main sections:

1. **Dashboard:** Displays prediction counts and recent activity.
2. **News Analysis:** Accepts a headline and article text and displays the API prediction.
3. **Prediction History:** Stores and filters previous predictions using browser local storage.
4. **Analytics:** Displays the selected Linear SVM and separate BiLSTM evaluation metrics.
5. **About Project:** Describes the objective, methodology, models and technology stack.

Additional interface features include copying prediction results and printing or saving the result as a PDF.

### Backend

The backend uses FastAPI and serves predictions through the selected Linear SVM pipeline.

Model path:

`models/phase1/ml_selected_pipeline.joblib`

Available endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API home |
| `/health` | GET | Backend and model health check |
| `/predict` | POST | News classification |

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### Run the application locally

**Terminal 1 â€” Start the backend:**

```powershell
cd "C:\Users\mi\Desktop\Fake-news detection"

.\.venv\Scripts\Activate.ps1

python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 â€” Start the frontend:**

```powershell
cd "C:\Users\mi\Desktop\Fake-news detection"

python -m http.server 5500 --directory frontend
```

Open the application in a browser:

`http://127.0.0.1:5500`

The frontend sends prediction requests to the local FastAPI backend.

## 9. Model Artefacts

Important saved artefacts include:

- `models/phase1/ml_selected_pipeline.joblib` â€” selected Linear SVM pipeline used by the API.
- `models/phase1/ml_logistic_regression_pipeline.joblib` â€” Logistic Regression pipeline.
- `models/phase1/ml_info_matched_svm.joblib` â€” information-matched SVM.
- `models/phase1/lstm_phase1_best.pt` â€” Phase 1 LSTM checkpoint.
- `models/phase1/bilstm_phase1_best.pt` â€” Phase 1 BiLSTM checkpoint.
- `models/bilstm_best_corrected.pt` â€” separately evaluated corrected BiLSTM checkpoint.
- `results/tables/` â€” evaluation metrics, audit reports, comparison tables and verification outputs.
- `results/figures/` â€” plots and visualizations.

Legacy checkpoints are retained as separate historical artefacts. Their results should not be mixed with the Phase 1 evaluation without checking their preprocessing and evaluation methodology.

## 10. Limitations and Future Work

### Limitations

1. **Source confounding:** The Real class is associated with a particular publisher, which may introduce source-specific patterns.
2. **Dataset coverage:** Experiments use one research dataset and do not establish generalization to all news sources or time periods.
3. **Shortcut learning:** Models may learn formatting, vocabulary or publisher-related patterns rather than factual correctness.
4. **Uncalibrated scores:** Prediction confidence or decision margins should not be interpreted as verified probabilities of truth.
5. **Computational constraints:** The DistilBERT pilot was completed using a GPU, but full-data fine-tuning and final test evaluation remain pending.

### Future Work

- Complete DistilBERT fine-tuning using suitable GPU resources.
- Evaluate on additional independent news datasets.
- Improve generalization across publishers and domains.
- Expand explainability and error analysis.
- Explore calibrated prediction confidence.
- Integrate independent source verification and fact-checking services.

## 11. References

- Ahmed, H., Traore, I., and Saad, S. (2017). Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. https://doi.org/10.1007/978-3-319-69155-8_9
- Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory.
- Schuster, M., and Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks.
- Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.
- Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.
- Geirhos et al. (2020). Shortcut Learning in Deep Neural Networks. arXiv:2004.07780.
- Zeiler, M. D., and Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. arXiv:1311.2901.
- Jain, S., and Wallace, B. C. (2019). Attention is not Explanation. arXiv:1902.10186.

---

**Project Code:** PCS26/84
**Project Title:** DEEP LEARNING-BASED FAKE NEWS DETECTION
**Application Name:** TruthLens AI
## Fresh Environment Setup (Windows)

Python 3.12 is recommended. Run these commands from the extracted project root:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
