# Text Classification Project

## Overview
This project implements Naive Bayes and Logistic Regression for spam detection using email datasets. The code converts emails into numerical feature representations (Bag of Words and Bernoulli approaches) and classifies them using **Multinomial Naive Bayes, Bernoulli Naive Bayes, and Logistic Regression**.


## Requirements
The requirements for this project were Python 3.9 or later. I also used the following dependencies:
```bash
pip install numpy pandas scikit-learn nltk
```
If you are using **macOS**, ensure that VS Code is using the correct Python interpreter:
```bash
which python3   # Should return a path like /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
python3 -m pip install --upgrade pip
```

## Usage
### Step 1: Preprocessing (Feature Extraction)
Run the following command to process email datasets and generate feature matrices:
```bash
python3 first_phase.py
```
This will generate the Bag of Words (BoW) and Bernoulli feature matrices and store them in the `processed_data/` directory as CSV files.

### Step 2: Train and Evaluate Multinomial Naive Bayes (BoW Model)
```bash
python3 second_phase.py
```
Expected output (example):
```
Multinomial Naive Bayes (Bag of Words) Evaluation:
Accuracy: 0.9364
Precision: 0.9000
Recall: 0.9060
F1 Score: 0.9030
```

## Troubleshooting





