# Fairness Analysis of an Auto Insurance Prediction Model

An algorithmic fairness audit examining whether a logistic regression claim-prediction model treats blue-collar auto insurance customers equitably compared to other customer groups.

---

## Project Overview

An auto insurance company uses a logistic regression model to predict the probability that a customer will file at least one claim in the coming year. Customers whose predicted probability exceeds **60%** are flagged as high-risk and will have their policies cancelled at renewal.

This project investigates whether that model — and the cancellation policy built on top of it — treats **blue-collar customers** (electricians, plumbers, construction workers, and other trades) fairly relative to all other customers. The analysis follows a structured fairness audit framework covering base rates, model calibration, classification accuracy, and differential error rates.

---

## Key Findings

| Finding | Detail |
|---|---|
| **Cancellation rate disparity** | 16.0% of blue-collar customers are cancelled vs. 7.2% of non-blue-collar customers |
| **Over-representation in cancellations** | Blue-collar workers are 22% of the customer base but 39% of all cancellations |
| **Model miscalibration** | Controlling for predicted probability, blue-collar customers have actual claim rates ~2 pp higher than predicted (p = 0.039) |
| **False Positive Rate gap** | Blue-collar customers who will NOT claim are wrongly cancelled at 6.1% vs. 2.6% for non-blue-collar — a 2.4× disparity |
| **Accuracy gap** | Model accuracy is 73.0% for blue-collar vs. 79.1% for non-blue-collar at the 60% threshold |
| **Similar ranking ability** | ROC-AUC is comparable across groups (Blue Collar: 0.802, Non-Blue Collar: 0.792) |

---

## Repository Structure

```
├── Fairness_Analysis_of_Auto_Insurance_Prediction_Model.ipynb   # Main analysis notebook
├── autodata.csv                                                  # Dataset (loaded via URL in notebook)
└── README.md
```

---

## Notebook Structure

The notebook is organized into the following sections:

1. **Setup and Data Loading** — imports and creates the binary high-risk flag at the 60% threshold
2. **Base Rates and Predicted Probabilities** — compares actual claim rates and the distribution of model scores across groups
3. **Classification and Cancellation Rates** — quantifies who gets cancelled and at what rate, across all job classes
4. **Model Calibration** — calibration curves and a regression test to check whether the model predicts risk equally well for both groups
5. **Accuracy and Discriminative Power** — overall accuracy, accuracy across all thresholds, and ROC-AUC curves
6. **Differential Error Rates** — False Positive, False Negative, False Discovery, and False Omission rates at the 60% threshold and across all thresholds
7. **Executive Summary** — a non-technical summary of findings and recommendations for management

---

## Data

The dataset (`autodata.csv`) contains 9,656 auto insurance customer records with the following key variables:

| Variable | Description |
|---|---|
| `hadclaim` | 1 if the customer filed at least one claim, 0 otherwise |
| `probclaim` | Predicted probability of a claim from the logistic regression model |
| `job_bluecollar` | 1 if the customer works in a blue-collar job, 0 otherwise |
| `jobclass` | Full job category (Blue Collar, Clerical, Professional, Manager, Lawyer, Student, Home Maker, Doctor, Unknown) |

The dataset also includes demographic, vehicle, and driving record features used by the underlying prediction model (e.g., `clm_freq5`, `kidsdriv`, `travtime`, `mvr_pts`, `age`, `urban`, `revolked`). The model does **not** use income, education, or job classification as features.

The data originates from a publicly available auto insurance claim dataset and was pre-processed to generate the binary outcome (`hadclaim`) and the model's predicted probabilities (`probclaim`).

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

The notebook loads `autodata.csv` directly from a GitHub URL, so no manual data download is needed.

---

## How to Run

This notebook was built in **Google Colab**. The easiest way to run it is to open it directly in Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → GitHub**
3. Paste in the repository URL:
   ```
   https://github.com/quentinsjehn/auto-insurance-fairness-analysis
   ```
4. Select `Fairness_Analysis_of_Auto_Insurance_Prediction_Model.ipynb` and click open
5. Run all cells in order (**Runtime → Run all**)

All dependencies (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`) are pre-installed in Colab. The dataset is loaded directly from a URL, so no file uploads are needed.

Alternatively, if you want to clone the repository locally:

```bash
git clone https://github.com/quentinsjehn/auto-insurance-fairness-analysis.git
cd auto-insurance-fairness-analysis
```

---

## Context

This analysis was completed as part of **General Business 894**. It is inspired by the academic literature on algorithmic fairness — particularly the research stemming from ProPublica's investigation of the COMPAS recidivism algorithm — which showed that when two groups have different underlying outcome rates, a well-calibrated model will necessarily produce different false positive and false negative rates. This project applies that same framework to the insurance domain.
