import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_discrete_nb(train_df):
    """
    Train a Discrete (Bernoulli) Naive Bayes classifier.
    
    Args:
        train_df (DataFrame): Training data with binary features as columns (Bernoulli representation)
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
    word_presence = {}  # count of documents where word is present, for each class
    document_counts = {}  # count of documents in each class

    # Compute counts and class priors
    for c in classes:
        indices = np.where(y_train == c)[0]
        X_c = X_train[indices]
        
        # For binary features, sum gives count of documents where word is present
        presence_c = np.sum(X_c, axis=0)
        word_presence[c] = presence_c
        document_counts[c] = len(indices)
        class_priors[c] = document_counts[c] / n_train

    # Compute likelihoods with add-one Laplace smoothing in log-space
    # P(word|class) = (count of documents in class c where word appears + 1) / (total documents in class c + 2)
    # We add 2 in denominator because in Bernoulli, each feature has 2 possible values (0 or 1)
    likelihoods = {}
    for c in classes:
        # Probability of word presence given class
        likelihoods[c] = (word_presence[c] + 1) / (document_counts[c] + 2)
        
        # Also need probability of word absence given class
        # P(word=0|class) = 1 - P(word=1|class)
        likelihoods[c + 2] = 1 - likelihoods[c]  # Store absence probabilities with offset
        
        # Convert to log-space
        likelihoods[c] = np.log(likelihoods[c])
        likelihoods[c + 2] = np.log(likelihoods[c + 2])

    # Convert class priors to log-space
    log_class_priors = {c: np.log(class_priors[c]) for c in classes}

    model = {
        'log_class_priors': log_class_priors,
        'log_likelihoods': likelihoods,
        'vocab_size': V
    }
    return model

def predict_discrete_nb(model, X):
    """
    Predict the class labels for a set of examples using Discrete Naive Bayes.
    
    Args:
        model (dict): The trained model containing log class priors and log likelihoods.
        X (np.array): Binary feature matrix for test examples.
        
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
            # For Bernoulli model, we need to consider both presence (1) and absence (0) of features
            # Score = log(P(c)) + sum_{j: word_j=1} log(P(word_j=1|c)) + sum_{j: word_j=0} log(P(word_j=0|c))
            
            # Initialize with class prior
            score = log_class_priors[c]
            
            # Add contribution of present words (where feature value is 1)
            present_indices = np.where(X[i] == 1)[0]
            score += np.sum(log_likelihoods[c][present_indices])
            
            # Add contribution of absent words (where feature value is 0)
            absent_indices = np.where(X[i] == 0)[0]
            score += np.sum(log_likelihoods[c + 2][absent_indices])
            
            scores[c] = score
            
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

def run_bernoulli_nb():
    """Test the Discrete Naive Bayes implementation on all three datasets."""
    datasets = ['enron1', 'enron2', 'enron4']
    results = {}
    
    for dataset in datasets:
        print(f"\nEvaluating Discrete Naive Bayes on {dataset} dataset...")
        
        # Load the training and test data
        train_file = f'processed_data/{dataset}_bernoulli_train.csv'
        test_file = f'processed_data/{dataset}_bernoulli_test.csv'
        
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Train the model
        model = train_discrete_nb(train_df)
        
        # Prepare test features and labels
        X_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values
        
        # Predict on the test set
        y_pred = predict_discrete_nb(model, X_test)
        
        # Evaluate the predictions
        acc, prec, rec, f1 = evaluate_model(y_test, y_pred)
        results[dataset] = (acc, prec, rec, f1)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return results

if __name__ == "__main__":
    run_bernoulli_nb()