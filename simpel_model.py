import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

dataset_folder_path = 'data_folder'
class SimpleModel:
    def __init__(self, always_predict: int):
        self.neutral_class = always_predict
        print(f'This models prediction is always: {always_predict}')

    def predict(self, X):
        return np.full(len(X), self.neutral_class)

    def evaluate(self, y_true):
        """Evaluate the model by printing the accuracy and classification report."""
        y_pred = self.predict(y_true)  # We use y_true here because we always predict neutral
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, labels=[self.neutral_class]))

# Sample usage
if __name__ == "__main__":
    train_labels = pd.read_csv(f'{dataset_folder_path}/train_prep_with_labels.csv')['label']
    max_label = train_labels.value_counts().idxmax()
    labels = pd.read_csv(f'{dataset_folder_path}/test_prep_with_labels.csv')['label']
    # Initialize the simple model

    model = SimpleModel(always_predict=max_label)

    # Evaluate the model with mock data
    model.evaluate(labels)
