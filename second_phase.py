import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_multinomial_nb(train_df):
    """
    Train a Multinomial Naive Bayes classifier.
    
    Args:
        train_df (DataFrame): Training data with features as columns (Bag-of-Words counts)
                              and a 'label' column (1 for spam, 0 for ham).
                              
    Returns:
        model (dict): Contains log class priors and log likelihoods for each word per class.
    """
    # Separate features and labels
    X_train = train_df.drop('label', axis=1).values  # shape: (n_train, V)
    y_train = train_df['label'].values  # 1 for spam, 0 for ham
    n_train, V = X_train.shape

    # Define the classes
    classes = [0, 1]
    class_priors = {}
    word_counts = {}  # total count per word for each class
    total_word_counts = {}  # sum of all word counts for each class

    # Compute counts and class priors
    for c in classes:
        indices = np.where(y_train == c)[0]
        X_c = X_train[indices]
        count_c = np.sum(X_c, axis=0)  # sum counts for each word
        word_counts[c] = count_c
        total_word_counts[c] = np.sum(count_c)
        class_priors[c] = len(indices) / n_train

    # Compute likelihoods with add-one Laplace smoothing in log-space
    likelihoods = {}
    for c in classes:
        # P(word|class) = (count(word, c) + 1) / (total words in class + V)
        likelihoods[c] = (word_counts[c] + 1) / (total_word_counts[c] + V)
        likelihoods[c] = np.log(likelihoods[c])  # convert to log-space

    # Convert class priors to log-space
    log_class_priors = {c: np.log(class_priors[c]) for c in classes}

    model = {
        'log_class_priors': log_class_priors,
        'log_likelihoods': likelihoods,
        'vocab_size': V
    }
    return model

def predict_multinomial_nb(model, X):
    """
    Predict the class labels for a set of examples.
    
    Args:
        model (dict): The trained model containing log class priors and log likelihoods.
        X (np.array): Feature matrix for test examples.
        
    Returns:
        predictions (np.array): Predicted class labels.
    """
    log_class_priors = model['log_class_priors']
    log_likelihoods = model['log_likelihoods']
    classes = [0, 1]
    n_examples = X.shape[0]
    predictions = []

    for i in range(n_examples):
        # For each test example, compute score for each class:
        scores = {}
        for c in classes:
            # Score = log(P(c)) + sum_{j} count(word_j)*log(P(word_j|c))
            scores[c] = log_class_priors[c] + np.sum(X[i] * log_likelihoods[c])
        # Predict the class with the highest score
        pred = max(scores, key=scores.get)
        predictions.append(pred)
    return np.array(predictions)

def evaluate_model(y_true, y_pred):
    """
    Evaluate the predictions using accuracy, precision, recall, and F1-score.
    
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        
    Returns:
        metrics (tuple): (accuracy, precision, recall, f1_score)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

def main():
    # Example using the enron1 Bag-of-Words datasets
    train_file = 'processed_data/enron1_bow_train.csv'
    test_file = 'processed_data/enron1_bow_test.csv'
    
    # Load the training and test data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Train the model
    model = train_multinomial_nb(train_df)
    
    # Prepare test features and labels
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Predict on the test set
    y_pred = predict_multinomial_nb(model, X_test)
    
    # Evaluate the predictions
    acc, prec, rec, f1 = evaluate_model(y_test, y_pred)
    
    print("Multinomial Naive Bayes (Bag of Words) Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
