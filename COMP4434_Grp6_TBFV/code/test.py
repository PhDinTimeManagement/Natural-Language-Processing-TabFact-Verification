import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the TSV data
data_path = '../dataset/tsv_data_horizontal/complex_test.tsv'
data = pd.read_csv(data_path, sep='\t', header=None).values

# Convert data into required format
def preprocess_tsv_data(tsv_data):
    # Concatenate table-related columns into a single string for each row
    # Starting from column 3
    table_texts = [" ".join(row[2:-2]) for row in tsv_data]

    statements = tsv_data[:, -2]
    labels = tsv_data[:, -1].astype(int)
    return table_texts, statements, labels

# Preprocess the TSV data
table_texts, statements, labels = preprocess_tsv_data(data)

# Print a sample test case
print("Sample table text:", table_texts[0])
print("Sample statement:", statements[0])
print("Sample label:", labels[0])

# # TF-IDF representation
# vectorizer = TfidfVectorizer(max_features=500)
# table_features = vectorizer.fit_transform(table_texts)
# statement_features = vectorizer.transform(statements)
#
# # Combine features by calculating similarity
# similarities = cosine_similarity(table_features, statement_features)
#
# # Prepare training and testing data
# X_train, X_test, y_train, y_test = train_test_split(similarities, labels, test_size=0.2, random_state=42)

# # Train a simple logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# # Evaluate the model
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # Train an SVM classifier
# svm_model = SVC(kernel='linear', C=1.0)
# svm_model.fit(X_train, y_train)
#
# # Evaluate the model
# y_pred = svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")