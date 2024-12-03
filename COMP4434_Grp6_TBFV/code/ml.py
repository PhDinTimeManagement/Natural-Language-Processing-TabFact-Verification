import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import re
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the TSV data
data_path = '../dataset/tsv_data_horizontal/complex_test.tsv'
data = pd.read_csv(data_path, sep='\t', header=None).values


#-----------------Data Preprocessing-----------------
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

# Preprocess the TSV data
table_texts, statements, labels = preprocess_tsv_data(data)

# Print a sample test case
print("Sample table text:", table_texts[0])
print("Sample statement:", statements[0])
print("Sample label:", labels[0])


#-----------------TF-IDF Feature Extraction-----------------
vectorizer = TfidfVectorizer(max_features=500)
table_features = vectorizer.fit_transform(table_texts)
statement_features = vectorizer.transform(statements)
# Combine features by calculating similarity
similarities = cosine_similarity(table_features, statement_features)


#-----------------Prepare training and testing data-----------------
# Prepare training and testing data
X_train, X_test, y_train, y_test = train_test_split(similarities, labels, test_size=0.2, random_state=42)


#-----------------Cross-Validation Function-----------------
def cross_validate_model(model, X, y, k=5, is_regression=False):
    """
    Perform K-Fold Cross-Validation for a given model.

    Parameters:
    model: The machine learning model (e.g., LogisticRegression, Lasso, SVC).
    X: Feature matrix.
    y: Labels.
    k: Number of folds (default is 5).
    is_regression: Whether the model is a regression model (default is False).

    Returns:
    average_accuracy: The average accuracy across all folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Predict on test data

        # If regression, convert predictions to discrete labels
        if is_regression:
            y_pred = np.round(y_pred).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Fold Accuracy: {accuracy:.2f}")

    average_accuracy = sum(accuracies) / k
    return average_accuracy


#-----------------Logistic Regression Cross-Validation-----------------
print("\nLogistic Regression Cross-Validation:")
logistic_model = LogisticRegression()
logistic_avg_accuracy = cross_validate_model(logistic_model, similarities, labels, k=5)
print(f"Average Accuracy for Logistic Regression: {logistic_avg_accuracy:.2f}")


#-----------------Lasso Regression-----------------
# Train a Lasso Regression model
print("\nLasso Regression Cross-Validation:")
lasso_model = Lasso(alpha=0.1)
lasso_avg_accuracy = cross_validate_model(lasso_model, similarities, labels, k=5, is_regression=True)
print(f"Average Accuracy for Lasso Regression: {lasso_avg_accuracy:.2f}")


#-----------------Decision Tree Cross-Validation-----------------
print("\nDecision Tree Cross-Validation:")
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_avg_accuracy = cross_validate_model(decision_tree_model, similarities, labels, k=5)
print(f"Average Accuracy for Decision Tree: {decision_tree_avg_accuracy:.2f}")


#-----------------Support Vector Machine-----------------
print("\nSupport Vector Machine Cross-Validation:")
svm_model = SVC(kernel='linear')
svm_avg_accuracy = cross_validate_model(svm_model, similarities, labels, k=5)
print(f"Average Accuracy for Support Vector Machine: {svm_avg_accuracy:.2f}")


#-----------------Random Forest-----------------
print("\nRandom Forest Cross-Validation:")
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_avg_accuracy = cross_validate_model(random_forest_model, similarities, labels, k=5)
print(f"Average Accuracy for Random Forest: {random_forest_avg_accuracy:.2f}")
