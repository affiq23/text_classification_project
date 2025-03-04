import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def sigmoid(z):
    """Sigmoid function."""
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X_train, y_train, lambda_val=0.1, learning_rate=0.01, max_iter=1000, tol=1e-5):
    """
    Train logistic regression model with L2 regularization using gradient ascent.
    
    Args:
        X_train (np.array): Feature matrix.
        y_train (np.array): Labels.
        lambda_val (float): L2 regularization parameter.
        learning_rate (float): Step size for gradient ascent.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        
    Returns:
        w (np.array): Learned weights.
    """
    n, d = X_train.shape
    w = np.zeros(d)  # Initialize weights
    
    # Add bias term
    X_train_bias = np.c_[np.ones(n), X_train]
    w = np.zeros(d + 1)  # Initialize weights with bias
    
    prev_loss = float('inf')
    
    for i in range(max_iter):
        # Compute predictions
        z = np.dot(X_train_bias, w)
        y_pred = sigmoid(z)
        
        # Compute gradient (for maximizing log-likelihood, hence the plus sign)
        gradient = np.dot(X_train_bias.T, (y_train - y_pred))
        
        # Add L2 regularization term (don't regularize bias)
        gradient[1:] -= lambda_val * w[1:]
        
        # Update weights
        w += learning_rate * gradient
        
        # Compute log-likelihood with regularization (for convergence check)
        log_likelihood = np.sum(y_train * np.log(y_pred + 1e-10) + (1 - y_train) * np.log(1 - y_pred + 1e-10))
        # Add regularization term (don't regularize bias)
        reg_term = 0.5 * lambda_val * np.sum(w[1:] ** 2)
        current_loss = -(log_likelihood - reg_term) / n  # Negative because we're maximizing
        
        # Check for convergence
        if abs(prev_loss - current_loss) < tol:
            print(f"Converged after {i+1} iterations.")
            break
            
        prev_loss = current_loss
        
    return w

def predict_logistic_regression(X, w):
    """
    Predict using logistic regression model.
    
    Args:
        X (np.array): Feature matrix.
        w (np.array): Learned weights.
        
    Returns:
        predictions (np.array): Predicted class labels.
    """
    # Add bias term
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Compute probabilities
    probs = sigmoid(np.dot(X_bias, w))
    
    # Convert to binary predictions
    predictions = (probs >= 0.5).astype(int)
    
    return predictions

def tune_lambda(X_train, y_train, lambda_values=None):
    """
    Tune regularization parameter lambda using validation set.
    
    Args:
        X_train (np.array): Training feature matrix.
        y_train (np.array): Training labels.
        lambda_values (list): List of lambda values to test.
        
    Returns:
        best_lambda (float): Lambda value with highest accuracy on validation set.
    """
    if lambda_values is None:
        lambda_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    # Split data into training and validation sets (70/30)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42)
    
    best_accuracy = 0
    best_lambda = lambda_values[0]
    
    print("\nTuning lambda parameter:")
    print("-" * 50)
    print(f"{'Lambda':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 50)
    
    for lambda_val in lambda_values:
        # Train model with current lambda
        w = train_logistic_regression(X_train_split, y_train_split, lambda_val=lambda_val, max_iter=500)
        
        # Evaluate on validation set
        y_pred = predict_logistic_regression(X_val, w)
        
        # Calculate metrics
        acc, prec, rec, f1 = evaluate_model(y_val, y_pred)
        
        print(f"{lambda_val:<10.4f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
        
        # Update best lambda if current accuracy is higher
        if acc > best_accuracy:
            best_accuracy = acc
            best_lambda = lambda_val
    
    print(f"\nBest lambda: {best_lambda} (validation accuracy: {best_accuracy:.4f})")
    return best_lambda

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

def run_logistic_regression():
    """Test the Logistic Regression implementation on all datasets with both representations."""
    datasets = ['enron1', 'enron2', 'enron4']
    representations = ['bow', 'bernoulli']
    results = {}
    
    for dataset in datasets:
        for representation in representations:
            print(f"\n\nEvaluating Logistic Regression on {dataset} dataset with {representation} representation...")
            
            # Load the training and test data
            train_file = f'processed_data/{dataset}_{representation}_train.csv'
            test_file = f'processed_data/{dataset}_{representation}_test.csv'
            
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            # Prepare training and test features and labels
            X_train = train_df.drop('label', axis=1).values
            y_train = train_df['label'].values
            X_test = test_df.drop('label', axis=1).values
            y_test = test_df['label'].values
            
            # Tune lambda parameter
            best_lambda = tune_lambda(X_train, y_train)
            
            # Train model with best lambda on full training set
            print(f"\nTraining final model with lambda={best_lambda}...")
            w = train_logistic_regression(X_train, y_train, lambda_val=best_lambda, max_iter=1000)
            
            # Predict on test set
            y_pred = predict_logistic_regression(X_test, w)
            
            # Evaluate the predictions
            acc, prec, rec, f1 = evaluate_model(y_test, y_pred)
            results[f"{dataset}_{representation}"] = (acc, prec, rec, f1)
            
            print(f"\nTest set evaluation:")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"F1 Score: {f1:.4f}")
    
    return results

if __name__ == "__main__":
    run_logistic_regression()