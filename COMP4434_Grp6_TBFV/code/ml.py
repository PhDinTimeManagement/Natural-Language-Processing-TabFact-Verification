import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertTokenizer
import re

# Load the TSV data
data_path = '../processed_datasets/tsv_data_horizontal/complex_test.tsv'
data = pd.read_csv(data_path, sep='\t', header=None).values

# Clean 'row x is:' in table
def clean_out_sub_table(tsv_data):
    table_texts_first = [" ".join(row[2:-2]) for row in tsv_data]
    table_texts_second = [re.sub(r'\. (.*?) :', ',', row) for row in table_texts_first]
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
    table_texts = clean_out_sub_table(tsv_data)
    statements = tsv_data[:, -2]
    labels = tsv_data[:, -1].astype(int)
    return table_texts, statements, labels

# Define a function to extract features for each transaction using [CLS] token
def extract_features_batch(texts, tokenizer, model):
    encoded_inputs = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # Access hidden states
    # Take mean of the last 4 hidden states for [CLS]
    features = torch.mean(torch.stack(hidden_states[-4:]), dim=0)[:, 0, :]  # [CLS] token
    return features

# Preprocess the TSV data
table_texts, statements, labels = preprocess_tsv_data(data)

# Convert data into a dataframe
df = pd.DataFrame({"Table_texts": table_texts, "Statements": statements, "Labels": labels})

# Load the pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Extract features for the table and statement texts
print("Extracting features from table texts...")
table_features = extract_features_batch(df["Table_texts"].tolist(), tokenizer, model)

print("Extracting features from statements...")
statement_features = extract_features_batch(df["Statements"].tolist(), tokenizer, model)

# Calculate cosine similarities
print("Calculating cosine similarities...")
cosine_similarities = torch.nn.functional.cosine_similarity(
    table_features, statement_features, dim=1
).cpu().numpy()  # Convert to numpy array

# Prepare labels as numpy array
labels = np.array(labels)

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    cosine_similarities.reshape(-1, 1), labels, test_size=0.2, random_state=42
)

# K-Fold cross-validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []

print(f"Performing {k}-fold cross-validation...")
for fold, (train_index, test_index) in enumerate(kf.split(cosine_similarities)):
    # Prepare training and testing data
    X_train = cosine_similarities[train_index].reshape(-1, 1)
    X_test = cosine_similarities[test_index].reshape(-1, 1)
    y_train = labels[train_index]
    y_test = labels[test_index]

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}")

# Calculate the average accuracy
average_accuracy = sum(accuracies) / k
print(f"Average Accuracy across {k} folds: {average_accuracy:.2f}")