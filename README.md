# Text Classification Project

## Overview
This project for CS4375 implements Naive Bayes and Logistic Regression for spam detection using email datasets. 

## File Structure 
This is how I organized my file structure in my local environment.
```bash
text_classification_project/
│── processed_data/         # directory containing generated CSV files
│── datasets/               # original email datasets given by professor
│── first_phase.py          # preprocesses datasets to convert to vectors
│── second_phase.py         # implements Multinomial Naive Bayes classifier
│── third_phase.py          # implements Bernoulli Naive Bayes classifier
│── fourth_phase.py         # implements Logistic Regression
│── README.md               # documentation
```

## Requirements
The requirements for this project were Python 3.9 or later. I am running Python 3.12.4 in my terminal, and I also used the following dependencies:
```bash
pip install numpy pandas scikit-learn nltk
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





