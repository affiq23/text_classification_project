# Text Classification Project

## Overview
This project implements **Naive Bayes and Logistic Regression** for spam detection using email datasets. The system converts emails into numerical feature representations (Bag of Words and Bernoulli approaches) and classifies them using **Multinomial Naive Bayes, Bernoulli Naive Bayes, and Logistic Regression**.

## Project Structure
```
text_classification_project/
│── processed_data/         # Directory containing generated CSV files
│── datasets/               # Original email datasets (enron1, enron2, enron4)
│── first_phase.py          # Preprocesses datasets and converts emails into feature vectors
│── second_phase.py         # Implements Multinomial Naive Bayes classifier
│── third_phase.py          # Implements Bernoulli Naive Bayes classifier
│── fourth_phase.py         # Implements Logistic Regression
│── README.md               # Project documentation (this file)
```

## Requirements
This project requires **Python 3.9 or later** and the following dependencies:
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
This will generate the **Bag of Words (BoW) and Bernoulli feature matrices** and store them in the `processed_data/` directory as CSV files.

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

### Step 3: Train and Evaluate Bernoulli Naive Bayes (Bernoulli Model)
```bash
python3 third_phase.py
```

### Step 4: Train and Evaluate Logistic Regression
```bash
python3 fourth_phase.py
```

## Results & Evaluation
Each classifier outputs standard evaluation metrics:
- **Accuracy**: Percentage of correctly classified emails
- **Precision**: Ratio of correctly classified spam emails to total predicted spam emails
- **Recall**: Ratio of correctly classified spam emails to actual spam emails
- **F1 Score**: Harmonic mean of precision and recall

## Troubleshooting
### Issue: `ModuleNotFoundError: No module named 'numpy'`
Ensure dependencies are installed in the correct Python version:
```bash
python3 -m pip install numpy pandas scikit-learn nltk
```




