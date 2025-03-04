# Text Classification Project
Affiq Mohammed

CS4375.004

## Overview
This project implements Naive Bayes and Logistic Regression for spam detection using email datasets. 

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

NumPy: I used for numerical computations and matrix operations.

Pandas: I used to load, process, and manage CSV datasets.

Scikit-learn: I used for machine learning utilities like accuracy, precision, and F1-score calculations.

NLTK: I used for NLP tasks like stopword removal and tokenization.
```bash
pip install numpy pandas scikit-learn nltk
```

## Running the Project
### Step 1: Preprocessing (Feature Extraction)
Run the following command to process email datasets and generate feature matrices:
```bash
python3 first_phase.py
```
This will generate the Bag of Words (BoW) and Bernoulli feature matrices and store them in the `processed_data/` directory as CSV files.

### Step 2: Train and Evaluate Multinomial Naive Bayes (BoW Model)
Run the following command to train and evaluate the Multinomial Naive Bayes model using the BoW representation:
```bash
python3 second_phase.py
```

### Step 3: Train and Evaluate Bernoulli Naive Bayes (Bernoulli Model)
Run the following command to train and evaluate the Bernoulli Naive Bayes model using the Bernoulli representation:
```bash
python3 third_phase.py
```

### Step 4: Train and Evaluate Logistic Regression
Run the following command to train and evaluate the Logistic Regression model:
```bash
python3 fourth_phase.py
```

## Troubleshooting

1. **Dependencies**: make sure all required dependencies are installed using the provided `pip install` command; I had some issues with the depenencies working with my local environment, so if the project isn't running, this is probably the main error.

2. **File Path**: I have my local file path listed in each part of the project, so that might need to be changed if running locally on another machine.

