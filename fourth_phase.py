import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# sigmoid function with overflow prevention
def sigmoid(input_values):
    input_values = np.clip(input_values, -500, 500)  # to prevent overflow when computing exponentials
    return 1 / (1 + np.exp(-input_values))  # computes sigmoid function

# train logistic regression model with L2 regularization using gradient ascent
def train_logistic_regression(training_features, training_labels, regularization_strength=0.1, learning_rate=0.01, max_iterations=1000, convergence_tolerance=1e-5):
    # training_features: feature matrix
    # training_labels: binary labels (0 or 1)
    # regularization_strength: L2 regularization parameter
    # learning_rate: step size for gradient ascent
    # max_iterations: max num of iterations
    # convergence_tolerance: convergence tolerance

    num_samples, num_features = training_features.shape  # get dataset dimensions
    # add bias term
    features_with_bias = np.c_[np.ones(num_samples), training_features]
    model_weights = np.zeros(num_features + 1)  # initialize weights with bias
    
    previous_model_loss = float('inf')  # initialize loss with infinity for convergence check
    
    for iteration in range(max_iterations):
        # compute predictions
        linear_combination = np.dot(features_with_bias, model_weights)
        predicted_probabilities = sigmoid(linear_combination)
        
        # compute gradient 
        gradient = np.dot(features_with_bias.T, (training_labels - predicted_probabilities))
        
        # add L2 regularization term to gradient
        gradient[1:] -= regularization_strength * model_weights[1:]
        
        # update weights
        model_weights += learning_rate * gradient
        
        # computes natural log of predicted probabilities
        log_likelihood = np.sum(training_labels * np.log(predicted_probabilities + 1e-10) + 
                                (1 - training_labels) * np.log(1 - predicted_probabilities + 1e-10))  # 1e-10 is to prevent log(0)
        regularization_term = 0.5 * regularization_strength * np.sum(model_weights[1:] ** 2)  # computes L2 regularization term
        current_model_loss = -(log_likelihood - regularization_term) / num_samples  # negative because maximizing log-likelihood 
        
        # check for convergence
        if abs(previous_model_loss - current_model_loss) < convergence_tolerance:  # check if difference is less than tolerance
            break
            
        previous_model_loss = current_model_loss
        
    return model_weights  # returns learned weights

# predict logistic regression model using learned weights
def predict_logistic_regression(test_features, learned_weights):
    
    # add bias term
    features_with_bias = np.c_[np.ones(test_features.shape[0]), test_features]
    
    # compute probabilities
    predicted_probabilities = sigmoid(np.dot(features_with_bias, learned_weights))
    
    # convert to binary predictions
    return (predicted_probabilities >= 0.5).astype(int)

def tune_regularization_parameter(training_features, training_labels, regularization_values=None):
    # training_features: feature matrix
    # training_labels: binary labels (0 or 1)
    # regularization_values: list of lambda values to test
  
    if regularization_values is None:  # default values for lambda
        regularization_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    # split data into training and validation sets (70/30)
    training_features_subset, validation_features, training_labels_subset, validation_labels = train_test_split(
        training_features, training_labels, test_size=0.3, random_state=42)
    
    best_validation_accuracy = 0
    best_regularization_value = regularization_values[0]
    
    # use list to collect results for cleaner printing
    tuning_results = []
    
    for current_regularization_strength in regularization_values:
        # train model with current lambda
        learned_weights = train_logistic_regression(
            training_features_subset, 
            training_labels_subset, 
            regularization_strength=current_regularization_strength, 
            max_iterations=500
        )
        
        # evaluate on validation set
        validation_predictions = predict_logistic_regression(validation_features, learned_weights)
        
        # calculate metrics
        validation_accuracy, validation_precision, validation_recall, validation_f1 = evaluate_model_performance(
            validation_labels, 
            validation_predictions
        )
        
        # store results
        tuning_results.append({
            'regularization_strength': current_regularization_strength,
            'accuracy': validation_accuracy,
            'precision': validation_precision,
            'recall': validation_recall,
            'f1_score': validation_f1
        })
        
        # update best lambda if current accuracy is higher
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_regularization_value = current_regularization_strength
    
    # print metrics in tabular format for easier comparison
    print("\nRegularization Parameter Tuning Results:")
    print("-" * 70)
    print(f"{'Lambda':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print("-" * 70)
    for result in tuning_results:
        print(f"{result['regularization_strength']:>10.4f} {result['accuracy']:>10.4f} {result['precision']:>10.4f} "
              f"{result['recall']:>10.4f} {result['f1_score']:>10.4f}")
    
    print(f"\nBest Lambda: {best_regularization_value} (Validation Accuracy: {best_validation_accuracy:.4f})")
    return best_regularization_value

# evaluate model performance using accuracy, precision, recall, and F1 score
def evaluate_model_performance(true_labels, predicted_labels):
   
    model_accuracy = accuracy_score(true_labels, predicted_labels)
    model_precision = precision_score(true_labels, predicted_labels)
    model_recall = recall_score(true_labels, predicted_labels)
    model_f1_score = f1_score(true_labels, predicted_labels)
    return model_accuracy, model_precision, model_recall, model_f1_score

# test logistic regression model on all datasets
def run_logistic_regression_experiments():
   
    dataset_names = ['enron1', 'enron2', 'enron4']
    feature_representations = ['bow', 'bernoulli']
    experimental_results = {}
    
    for dataset_name in dataset_names:
        for representation_type in feature_representations:
            print(f"\nEvaluating: {dataset_name} dataset | {representation_type} representation") 
            print("-" * 50) 
            
            # load formatted training/test data
            training_data_filepath = f'processed_data/{dataset_name}_{representation_type}_train.csv'
            test_data_filepath = f'processed_data/{dataset_name}_{representation_type}_test.csv'
            
            training_dataframe = pd.read_csv(training_data_filepath)
            test_dataframe = pd.read_csv(test_data_filepath)
            
            # prepare training and test features and labels
            training_features = training_dataframe.drop('label', axis=1).values
            training_labels = training_dataframe['label'].values
            test_features = test_dataframe.drop('label', axis=1).values
            test_labels = test_dataframe['label'].values
            
            # tune lambda parameter
            best_regularization_strength = tune_regularization_parameter(training_features, training_labels)
            
            # train model with best lambda on full training set
            final_model_weights = train_logistic_regression(
                training_features, 
                training_labels, 
                regularization_strength=best_regularization_strength, 
                max_iterations=1000
            )
            
            # predict on test set
            test_predictions = predict_logistic_regression(test_features, final_model_weights)
            
            # evaluate predictions
            model_accuracy, model_precision, model_recall, model_f1_score = evaluate_model_performance(
                test_labels, 
                test_predictions
            )
            
            # store results
            experimental_results[f"{dataset_name}_{representation_type}"] = (
                model_accuracy, model_precision, model_recall, model_f1_score
            )
            
            # print metrics
            print("\nTest Set Evaluation:")
            print(f"{'Accuracy:':<15} {model_accuracy:.4f}")
            print(f"{'Precision:':<15} {model_precision:.4f}")
            print(f"{'Recall:':<15} {model_recall:.4f}")
            print(f"{'F1 Score:':<15} {model_f1_score:.4f}")
    
    return experimental_results

def main():
    run_logistic_regression_experiments()

if __name__ == "__main__":
    main()