import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# train a Multinomial Naive Bayes classifier using BoW representation
def train_multinomial_naive_bayes(training_dataframe):
    # separate word counts from labels (spam/ham)
    feature_matrix = training_dataframe.drop('label', axis=1).values  # shape: (num_samples, vocab_size)
    class_labels = training_dataframe['label'].values  # 1 for spam, 0 for ham
    num_samples, vocabulary_size = feature_matrix.shape  # get dataset dimensions
    unique_classes = [0, 1]  # define possible classes (ham/spam)
    class_prior_probabilities = {}  # prior probabilities for each class
    class_word_counts = {}  # stores word frequencies per class
    total_class_word_counts = {}

    # compute word counts and class priors
    for current_class in unique_classes:
        class_indices = np.where(class_labels == current_class)[0]  # find all emails of class
        class_specific_samples = feature_matrix[class_indices]  # extract class-specific samples
        word_count_per_class = np.sum(class_specific_samples, axis=0)  # sum up word counts for this class
        class_word_counts[current_class] = word_count_per_class
        total_class_word_counts[current_class] = np.sum(word_count_per_class)  # Total words in class
        class_prior_probabilities[current_class] = len(class_indices) / num_samples  # P(class) = num_class_samples / total_samples

    # compute likelihoods with Laplace smoothing (add-one smoothing)
    word_likelihoods = {}
    for current_class in unique_classes:
        word_likelihoods[current_class] = (class_word_counts[current_class] + 1) / (total_class_word_counts[current_class] + vocabulary_size)  # smoothed probabilities
        word_likelihoods[current_class] = np.log(word_likelihoods[current_class])  # convert to log-space for numerical stability

    # convert priors to log-space
    log_class_priors = {c: np.log(class_prior_probabilities[c]) for c in unique_classes}
    
    return {
        'log_class_priors': log_class_priors,
        'log_likelihoods': word_likelihoods,
        'vocab_size': vocabulary_size
    }

# predict class labels for test emails using the trained model
def predict_naive_bayes(trained_model, test_feature_matrix):
    log_class_priors = trained_model['log_class_priors']
    log_likelihoods = trained_model['log_likelihoods']
    unique_classes = [0, 1]  # ham/spam
    num_test_samples = test_feature_matrix.shape[0]
    class_predictions = []

    for sample_index in range(num_test_samples):
        class_scores = {}  # store log probabilities for each class
        for current_class in unique_classes:
            # log-prob of the class + sum of word probabilities for the email
            class_scores[current_class] = log_class_priors[current_class] + np.sum(test_feature_matrix[sample_index] * log_likelihoods[current_class])
        
        # pick class with the highest probability
        class_predictions.append(max(class_scores, key=class_scores.get))
    
    return np.array(class_predictions)  # convert to numpy array

# evaluate model performance using accuracy, precision, recall, and F1 score
def evaluate_model_performance(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return accuracy, precision, recall, f1

# load in datasets, train model, predict, evaluate model
def main():
    train_filepath = '/Users/affiq/Projects/text_classification_project/processed_data/enron1_bow_train.csv'
    test_filepath = '/Users/affiq/Projects/text_classification_project/processed_data/enron1_bow_test.csv'

    # load in datasets
    training_dataframe = pd.read_csv(train_filepath)
    test_dataframe = pd.read_csv(test_filepath)

    # train model
    trained_model = train_multinomial_naive_bayes(training_dataframe)

    # extract test set features and labels
    test_feature_matrix = test_dataframe.drop('label', axis=1).values
    test_labels = test_dataframe['label'].values

    # predict using trained model
    predicted_labels = predict_naive_bayes(trained_model, test_feature_matrix)

    # evaluate model performance
    accuracy, precision, recall, f1 = evaluate_model_performance(test_labels, predicted_labels)

    # print metrics
    print("Multinomial Naive Bayes (BoW) Evaluation:")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()