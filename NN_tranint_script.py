import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class SimpleClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_size: int = 128):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second layer
        x = self.fc3(x)  # Output logits (softmax will be applied externally or during training)
        return x


dataset_folder_path = 'data_folder'
num_classes = 28

print('Loading Dataset...')
with open(f'{dataset_folder_path}/NN/train_prep_ped_with_labels.pkl', 'rb') as f:
    train_X, train_y = pickle.load(f)
with open(f'{dataset_folder_path}/NN/train_prep_ped_with_labels.pkl', 'rb') as f:
    val_X, val_y = pickle.load(f)
with open(f'{dataset_folder_path}/NN/train_prep_ped_with_labels.pkl', 'rb') as f:
    test_X, test_y = pickle.load(f)

print('Converting Dataset To Tensors...')

train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)  # One-hot encoded labels

val_X = torch.tensor(val_X, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32)

test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

print(f"Train data shape: {train_X.shape}, Labels shape: {train_y.shape}")
print(f"Dev data shape: {val_X.shape}, Labels shape: {val_y.shape}")
print(f"Test data shape: {test_X.shape}, Labels shape: {test_y.shape}")

print('Building The Model...')

model = SimpleClassifier()


