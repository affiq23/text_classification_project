import os
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('stopwords')

def read_emails(directory):
    """Read all email files from a directory."""
    emails = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    content = f.read()
                    emails.append(content)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return emails

def preprocess_text(text):
    """Convert text to lowercase, remove punctuation, and tokenize."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize (split into words)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 1]
    
    return words

def build_vocabulary(processed_emails, max_features=5000):
    """
    Build a vocabulary from a list of processed emails.
    Limit to max_features most common words to keep matrix size manageable.
    """
    # Count all words
    word_counts = Counter()
    for email in processed_emails:
        word_counts.update(email)
    
    # Get the most common words
    most_common = word_counts.most_common(max_features)
    vocabulary = [word for word, count in most_common]
    
    return vocabulary

def email_to_bow_vector(email, word_to_index):
    """Convert an email to a Bag of Words vector."""
    vector = [0] * len(word_to_index)
    for word in email:
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    return vector

def email_to_bernoulli_vector(email, word_to_index):
    """Convert an email to a Bernoulli vector."""
    vector = [0] * len(word_to_index)
    for word in email:
        if word in word_to_index:
            vector[word_to_index[word]] = 1
    return vector

def process_dataset(dataset_name, base_dir, max_features=5000):
    """
    Process a single dataset (train and test) into BoW and Bernoulli representations.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'enron1')
        base_dir (str): Base directory containing the datasets
        max_features (int): Maximum number of features to include in the vocabulary
    """
    print(f"Processing {dataset_name}...")
    
    # Construct paths to the dataset directories
    dataset_dir = os.path.join(base_dir, dataset_name)
    train_spam_dir = os.path.join(dataset_dir, 'train', 'spam')
    train_ham_dir = os.path.join(dataset_dir, 'train', 'ham')
    test_spam_dir = os.path.join(dataset_dir, 'test', 'spam')
    test_ham_dir = os.path.join(dataset_dir, 'test', 'ham')
    
    # Read and preprocess emails
    print("Reading and preprocessing emails...")
    train_spam = [preprocess_text(email) for email in read_emails(train_spam_dir)]
    train_ham = [preprocess_text(email) for email in read_emails(train_ham_dir)]
    test_spam = [preprocess_text(email) for email in read_emails(test_spam_dir)]
    test_ham = [preprocess_text(email) for email in read_emails(test_ham_dir)]
    
    # Build vocabulary from training data only
    print("Building vocabulary...")
    all_train_processed = train_spam + train_ham
    vocabulary = build_vocabulary(all_train_processed, max_features)
    
    # Create word to index mapping
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    
    # Convert emails to feature vectors
    print("Converting emails to feature vectors...")
    # Training set
    train_spam_bow = [email_to_bow_vector(email, word_to_index) for email in train_spam]
    train_ham_bow = [email_to_bow_vector(email, word_to_index) for email in train_ham]
    train_spam_bernoulli = [email_to_bernoulli_vector(email, word_to_index) for email in train_spam]
    train_ham_bernoulli = [email_to_bernoulli_vector(email, word_to_index) for email in train_ham]
    
    # Test set
    test_spam_bow = [email_to_bow_vector(email, word_to_index) for email in test_spam]
    test_ham_bow = [email_to_bow_vector(email, word_to_index) for email in test_ham]
    test_spam_bernoulli = [email_to_bernoulli_vector(email, word_to_index) for email in test_spam]
    test_ham_bernoulli = [email_to_bernoulli_vector(email, word_to_index) for email in test_ham]
    
    # Create labels (1 for spam, 0 for ham)
    train_spam_labels = [1] * len(train_spam_bow)
    train_ham_labels = [0] * len(train_ham_bow)
    test_spam_labels = [1] * len(test_spam_bow)
    test_ham_labels = [0] * len(test_ham_bow)
    
    # Combine spam and ham examples
    train_bow_features = train_spam_bow + train_ham_bow
    train_bow_labels = train_spam_labels + train_ham_labels
    test_bow_features = test_spam_bow + test_ham_bow
    test_bow_labels = test_spam_labels + test_ham_labels
    
    train_bernoulli_features = train_spam_bernoulli + train_ham_bernoulli
    train_bernoulli_labels = train_spam_labels + train_ham_labels
    test_bernoulli_features = test_spam_bernoulli + test_ham_bernoulli
    test_bernoulli_labels = test_spam_labels + test_ham_labels
    
    # Save datasets to CSV files
    print("Saving datasets to CSV files...")
    
    # Create DataFrames
    train_bow_df = pd.DataFrame(train_bow_features, columns=vocabulary)
    train_bow_df['label'] = train_bow_labels
    
    test_bow_df = pd.DataFrame(test_bow_features, columns=vocabulary)
    test_bow_df['label'] = test_bow_labels
    
    train_bernoulli_df = pd.DataFrame(train_bernoulli_features, columns=vocabulary)
    train_bernoulli_df['label'] = train_bernoulli_labels
    
    test_bernoulli_df = pd.DataFrame(test_bernoulli_features, columns=vocabulary)
    test_bernoulli_df['label'] = test_bernoulli_labels
    
    # Save to CSV
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    train_bow_df.to_csv(os.path.join(output_dir, f'{dataset_name}_bow_train.csv'), index=False)
    test_bow_df.to_csv(os.path.join(output_dir, f'{dataset_name}_bow_test.csv'), index=False)
    train_bernoulli_df.to_csv(os.path.join(output_dir, f'{dataset_name}_bernoulli_train.csv'), index=False)
    test_bernoulli_df.to_csv(os.path.join(output_dir, f'{dataset_name}_bernoulli_test.csv'), index=False)
    
    print(f"Finished processing {dataset_name}.")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Training examples: {len(train_bow_features)}")
    print(f"Test examples: {len(test_bow_features)}")
    print()

def main():
    """Main function to process all datasets."""
    # Base directory where all datasets are located
    base_dir = '/Users/affiq/Projects/text_classification_project/datasets'  
    
    # Process each dataset
    datasets = ['enron1', 'enron2', 'enron4']
    for dataset_name in datasets:
        process_dataset(dataset_name, base_dir)

if __name__ == "__main__":
    main()