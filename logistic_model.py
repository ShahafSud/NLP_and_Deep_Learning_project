import ast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd

dataset_folder_path = 'data_folder'

class LogisticRegressionTextClassifier:
    def __init__(self, max_features=5000, random_state=42):
        self.vectorizer = TfidfVectorizer(max_features=max_features)  # TF-IDF vectorizer
        self.model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'))  # Logistic regression model
        self.label_encoder = MultiLabelBinarizer()  # Multi-label binarizer for encoding multiple emotions

    def preprocess_data(self, X, y, X_test, y_test):
        # Convert text data into TF-IDF features
        X = X.astype(str)
        X_tfidf = self.vectorizer.fit_transform(X)
        # Encode the target labels (multi-labels)
        y_encoded = self.label_encoder.fit_transform(y)

        X_test_tfidf = self.vectorizer.transform(X_test.astype(str))
        y_test_encoded = self.label_encoder.transform(y_test)

        return X_tfidf, y_encoded, X_test_tfidf, y_test_encoded

    def train(self, X, y, X_test, y_test):
        # Preprocess the data (convert text and encode labels)
        X_tfidf, y_encoded, X_test_tfidf, y_test_encoded = self.preprocess_data(X, y, X_test, y_test)



        # Train the logistic regression model
        self.model.fit(X_tfidf, y_encoded)

        # Predict the labels for the test set
        y_pred = self.model.predict(X_test_tfidf)

        # Evaluate the model with accuracy and a detailed classification report
        accuracy = accuracy_score(y_test_encoded, y_pred)
        report = classification_report(y_test_encoded, y_pred, target_names=[str(i) for i in self.label_encoder.classes_], zero_division=0)
        return accuracy, report

    def predict(self, texts):
        # Transform new input texts to TF-IDF features
        X_tfidf = self.vectorizer.transform(texts)

        # Predict the emotion labels for the new texts
        predictions = self.model.predict(X_tfidf)

        # Convert the predicted binary labels back to original numeric labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        return predicted_labels


# Example Usage
if __name__ == "__main__":
    # Load the GoEmotions dataset
    train_data = pd.read_csv(f'{dataset_folder_path}/TFidf/train_prep_with_labels.csv')
    val_data = pd.read_csv(f'{dataset_folder_path}/TFidf/val_prep_with_labels.csv')
    test_data = pd.read_csv(f'{dataset_folder_path}/TFidf/test_prep_with_labels.csv')

    train_data['label'] = train_data['label'].apply(ast.literal_eval)
    val_data['label'] = val_data['label'].apply(ast.literal_eval)
    test_data['label'] = test_data['label'].apply(ast.literal_eval)

    train_texts = train_data['sample']
    val_texts = val_data['sample']
    test_texts = test_data['sample']

    train_labels = train_data['label']
    val_labels = val_data['label']
    test_labels = test_data['label']

    # Create an instance of the LogisticRegressionTextClassifier
    classifier = LogisticRegressionTextClassifier()

    # Train the model and evaluate its performance on the training set
    accuracy, report = classifier.train(train_texts, train_labels, test_texts, test_labels)

    # Print the accuracy and classification report for the training set
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Evaluate on the test set
    print("\nEvaluating on the test set:")
    test_pred = classifier.predict(test_texts)
    # test_accuracy = accuracy_score(test_labels, test_pred)
    test_labels_encoded = classifier.label_encoder.transform(test_labels)
    test_accuracy = accuracy_score(test_labels_encoded, classifier.label_encoder.transform(test_pred))
    print(f'Test Accuracy: {test_accuracy:4f}')
