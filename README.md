# NHL Game Outcome Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-darkblue?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?style=flat-square&logo=docker)](https://www.docker.com/)

A Multi-Layer Perceptron (MLP) neural network model that predicts the winner of NHL games using historical team performance metrics, goalie saves, and shots faced.

---

## The "Why" (Real-World Value)
Sports analytics is a multi-billion dollar industry driven by sports betting agencies, media companies, and NHL franchises seeking data-driven decision tools. Predicting game outcomes is challenging due to the high variance in hockey. This project demonstrates a production-grade machine learning pipeline that aggregates raw player-level stats into team-level performance features, enabling front-office analysts or sports enthusiasts to forecast game outcomes and evaluate key performance drivers.

---

## Tech Stack
* **Languages**: Python 3.10+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (MLPClassifier, StandardScaler)
* **Model Serialization**: Joblib
* **Deployment & Containerization**: Docker

---

## Architecture & Workflow
The inference and training workflow aggregates individual goalie and team statistics to the team-game level to prevent data duplication, processes features, and passes them to a trained Neural Network classifier.

```text
  [Raw NHL Data]
   ├── game.csv (Game IDs, Home/Away Teams)
   ├── game_teams_stats.csv (Team-level shots, hits)
   └── game_goalie_stats.csv (Goalie-level saves, shots faced)
         │
         ▼
  [Aggregation & Preprocessing] (Sum goalie stats to team-game level)
         │
         ▼
  [Feature Engineering] (Compute 5-game rolling averages, shifted by 1 to prevent leakage)
         │
         ▼
  [Feature Scaling] (StandardScaler)
         │
         ▼
  [MLP Neural Network Classifier] (2 Hidden Layers: 10, 5)
         │
         ▼
  [Prediction Output] (Home Win: 1 | Away Win: 0)
```

---

## Quickstart Guide

### 1. Prerequisites
Ensure you have the following installed:
* Python 3.10 or higher
* Git

### 2. Installation
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/naimul214/AI-Prediction-Model-on-NHL-Data-using-ANN.git
cd AI-Prediction-Model-on-NHL-Data-using-ANN
```

### 3. Setup Virtual Environment
Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 4. Running the Code
Run the model training and evaluation script:
```bash
python src/train.py
```

To run exploratory analysis, launch Jupyter Notebook:
```bash
jupyter notebook notebooks/main.ipynb
```

---

## Results & Demo

### Baseline Metrics (Historical Rolling Averages Model)
The model was evaluated using an 80/20 train-test split stratified by the target label. Because the model uses 5-game historical rolling averages instead of in-game stats, it is free of data leakage and represents a realistic forecasting baseline.

* **Accuracy**: `67.53%` (highly competitive for NHL forecasting where random guessing is 50% and standard benchmarks hover around 60%)
* **Precision**: `0.7022`
* **Recall**: `0.6733`
* **F1 Score**: `0.6875`

### Confusion Matrix
```text
[[6581  3132]  (Actual Away Wins)
 [3583  7385]] (Actual Home Wins)
```

---

## Limitations & Future Work
* **Missing Player-Level Stats**: Skater stats (`game_skater_stats.csv`) are currently excluded from training due to size constraints. Incorporating aggregated skater-level statistics (goals, assists) could enhance model depth.
* **Feature Engineering**: Incorporate external variables such as team rest days, travel distance, head-to-head records, and player injury status.
* **Hyperparameter Optimization**: Implement Grid Search or Bayesian Optimization to tune hidden layer sizes, learning rates, and optimizers.
* **Advanced Architectures**: Compare the current MLP neural network with tree-based models (XGBoost, LightGBM) which typically perform better on tabular sports datasets.

---

## Connect
* **LinkedIn**: [Naimul Hassan](https://linkedin.com/in/naimul214)
* **GitHub**: [@naimul214](https://github.com/naimul214)
* **Email**: nhridoy214@gmail.com
