import os
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter

# handle potential SSL certification issues; I was having an issue getting NumPy and Pandas to work
# might not be needed anymore but I left in case
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# download stopwords if not already present; should only need once but I kept 
nltk.download('stopwords', quiet=True)

# collects email contents from a specified directory and handles errors if any
def get_emails(email_folder):
    collected_emails = []  # store email contents
    for email_file in os.listdir(email_folder):  # iterate 
        full_path = os.path.join(email_folder, email_file)  # construct full path
        if os.path.isfile(full_path): 
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as file: 
                    collected_emails.append(file.read())  # store email content
            except IOError:  # handle read errors
                print(f"Skipping {full_path}") 
    return collected_emails  

# basically clean text
def clean_and_tokenize(raw_text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', raw_text.lower())  # remove non-word characters, make lowercase
    word_tokens = cleaned_text.split()  
    stop_words = set(stopwords.words('english')) 
    return [word for word in word_tokens if word not in stop_words and len(word) > 1]  # filter stopwords and short words

# create a focused vocabulary from list of processed emails
def get_top_words(processed_emails, max_word_count=1000):  # limit vocabulary size to 1000 words: wanted to limit overfitting
    word_frequency = Counter()  
    for email in processed_emails:
        word_frequency.update(email)  # update count from each email
    return [word for word, _ in word_frequency.most_common(max_word_count)] 

# generate feature vectors using BoW representation
def create_feature_vector(email_tokens, word_mapping):
    return [email_tokens.count(word) for word in word_mapping]  # count occurrences of each word in vocabulary

# process email datasets
def process_dataset(dataset_name, project_root, max_features=1000):  
    print(f"processing dataset: {dataset_name}")
    
    dataset_path = os.path.join(project_root, dataset_name)  # dataset directory
    
    # define folders
    train_spam_dir, train_ham_dir = os.path.join(dataset_path, 'train', 'spam'), os.path.join(dataset_path, 'train', 'ham')
    test_spam_dir, test_ham_dir = os.path.join(dataset_path, 'test', 'spam'), os.path.join(dataset_path, 'test', 'ham')
    
    # read and process emails
    train_spam_emails = [clean_and_tokenize(email) for email in get_emails(train_spam_dir)]
    train_ham_emails = [clean_and_tokenize(email) for email in get_emails(train_ham_dir)]
    test_spam_emails = [clean_and_tokenize(email) for email in get_emails(test_spam_dir)]
    test_ham_emails = [clean_and_tokenize(email) for email in get_emails(test_ham_dir)]
    
    # build vocabulary from training data
    vocabulary = get_top_words(train_spam_emails + train_ham_emails, max_features)
    
    # create feature vectors for all email sets
    train_spam_features = [create_feature_vector(email, vocabulary) for email in train_spam_emails]
    train_ham_features = [create_feature_vector(email, vocabulary) for email in train_ham_emails]
    test_spam_features = [create_feature_vector(email, vocabulary) for email in test_spam_emails]
    test_ham_features = [create_feature_vector(email, vocabulary) for email in test_ham_emails]
    
    # convert feature vectors to DataFrame
    def prepare_dataframe(features, labels, vocab):
        df = pd.DataFrame(features, columns=vocab)  # create DataFrame with vocabulary as columns
        df['label'] = labels  # add label column (1 = spam, 0 = ham)
        return df
    
    # create labeled datasets
    output_mappings = {
        'bow_train': (train_spam_features + train_ham_features, [1] * len(train_spam_features) + [0] * len(train_ham_features)),
        'bow_test': (test_spam_features + test_ham_features, [1] * len(test_spam_features) + [0] * len(test_ham_features)),
        'bernoulli_train': (train_spam_features + train_ham_features, [1] * len(train_spam_features) + [0] * len(train_ham_features)),
        'bernoulli_test': (test_spam_features + test_ham_features, [1] * len(test_spam_features) + [0] * len(test_ham_features))
    }
    
    os.makedirs('processed_data', exist_ok=True)  # make sure output directory exists
    for name, (features, labels) in output_mappings.items():
        prepare_dataframe(features, labels, vocabulary).to_csv(f'processed_data/{dataset_name}_{name}.csv', index=False)  # save CSV
    
    # print dataset summary
    print(f"completed {dataset_name}")
    print(f"vocabulary size: {len(vocabulary)}")
    print(f"total training examples: {len(output_mappings['bow_train'][0])}")
    print(f"total test examples: {len(output_mappings['bow_test'][0])}\n")

def main():
    dataset_directory = '/Users/affiq/Projects/text_classification_project/datasets'  # IMPORTANT: change this line to your directory to run
    target_datasets = ['enron1', 'enron2', 'enron4']  # datasets to process
    for dataset in target_datasets:
        process_dataset(dataset, dataset_directory)  # process each dataset

if __name__ == "__main__":
    main()  