import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import re
from transformers import BertModel, BertTokenizer

# Load the TSV data
data_path = '../dataset/tsv_data_horizontal/complex_test.tsv'
data = pd.read_csv(data_path, sep='\t', header=None).values

# Clean 'row x is:' in table
def clean_out_sub_table(tsv_data):
    table_texts_first = [" ".join(row[2:-2]) for row in tsv_data]
    table_texts_second = [re.sub(r'\. (.*?) :', ',', row) for row in table_texts_first]
    # print(table_texts_second[0])
    table_texts_final = []

    for row in table_texts_second:
        intermediate = row.split(" : ")
        if len(intermediate) > 1:
            table_texts_final.append(intermediate[1])
        else:
            table_texts_final.append(intermediate[0])

    return table_texts_final

# Convert data into required format
def preprocess_tsv_data(tsv_data):
    # Concatenate table-related columns into a single string for each row
    # Starting from column 3
    table_texts = clean_out_sub_table(tsv_data)
    statements = tsv_data[:, -2]
    labels = tsv_data[:, -1].astype(int)
    return table_texts, statements, labels

# Define a function to extract features for each transaction
def extract_features(text):
    # Tokenize the text
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)])
    # Get the hidden states for each token
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs[2]
    # Concatenate the last 4 hidden states
    token_vecs = []
    for layer in range(-4, 0):
        token_vecs.append(hidden_states[layer][0])
    # Calculate the mean of the last 4 hidden states
    features = []
    for token in token_vecs:
        features.append(torch.mean(token, dim=0))
    # Return the features as a tensor
    return torch.stack(features)

# Preprocess the TSV data and convert into data frame
table_texts, statements, labels = preprocess_tsv_data(data)
combined = np.column_stack((table_texts, statements, labels))
df = pd.DataFrame(combined, columns=['Table_texts', 'Statement', 'Labels'])

# Preprocess the TSV data
table_texts, statements, labels = preprocess_tsv_data(data)

# Print a sample test case
# print("Sample table text:", table_texts[0])
# print("Sample statement:", statements[0])
# print("Sample label:", labels[0])

# # TF-IDF representation
# vectorizer = TfidfVectorizer(max_features=500)
# table_features = vectorizer.fit_transform(table_texts)
# statement_features = vectorizer.transform(statements)
# # Combine features by calculating similarity
# similarities = cosine_similarity(table_features, statement_features)

# model_sentence_transformer = SentenceTransformer('bert-base-nli-mean-tokens')
# # Generate embeddings for table and statements
# table_embeddings = model_sentence_transformer.encode(table_texts)
# statement_embeddings = model_sentence_transformer.encode(statements)
# similarities = cosine_similarity(table_embeddings, statement_embeddings)


#Load the pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Extract features for each transaction
table_features = []
statement_features = []

for i in range(len(df)):
    feature_instance = extract_features(df.iloc[i]["Table_texts"])
    feature_statement_instance = extract_features(df.iloc[i]["Statement"])
    table_features.append(feature_instance)
    statement_features.append(feature_statement_instance)
    print(feature_instance)
    print(feature_statement_instance)
# Concatenate the features and convert to a numpy array
table_features = torch.cat(table_features).numpy()
statement_features = torch.cat(statement_features).numpy()

# Assume `table_features` and `statement_features` are tensors (not numpy arrays)
cosine_similarities = []
for table_feature, statement_feature in zip(table_features, statement_features):
    # Calculate cosine similarity using PyTorch
    similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(table_feature).unsqueeze(0),
        torch.tensor(statement_feature).unsqueeze(0)
    )
    cosine_similarities.append(similarity.item())

# `cosine_similarities` contains the similarity values for each instance
print("Cosine Similarities:", cosine_similarities)

# #Print the feature tensor
# print(table_features)
# print(statement_features)

# Prepare training and testing data
X_train, X_test, y_train, y_test = train_test_split(cosine_similarities, labels, test_size=0.2, random_state=42)

# K-fold cross-validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []

# Perfrom k-fold cross-validation
for train_index, test_index in kf.split(cosine_similarities):
    X_train, X_test = cosine_similarities[train_index], cosine_similarities[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Train a simple logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Fold Accuracy: {accuracy:.2f}")


# Calculate the average accuracy
average_accuracy = sum(accuracies) / k
print(f"Average Accuracy across {k} folders: {average_accuracy:.2f}")
