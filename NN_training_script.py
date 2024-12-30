import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

random.seed(0)
n_epoch = 10
h_size = 2048

class SimpleClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_size: int = h_size):
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
with open(f'{dataset_folder_path}/NN/val_prep_ped_with_labels.pkl', 'rb') as f:
    val_X, val_y = pickle.load(f)
with open(f'{dataset_folder_path}/NN/test_prep_ped_with_labels.pkl', 'rb') as f:
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_X, train_y = train_X.to(device), train_y.to(device)
val_X, val_y = val_X.to(device), val_y.to(device)
test_X, test_y = test_X.to(device), test_y.to(device)

model = SimpleClassifier(num_features=train_X.shape[1], num_classes=num_classes).to(device)

print('Training The Model...')

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = n_epoch
batch_size = 32

train_data = torch.utils.data.TensorDataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = torch.utils.data.TensorDataset(val_X, val_y)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(batch_X)  # Forward pass
        loss = loss_fn(outputs, batch_y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        train_loss += loss.item() * batch_X.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Evaluate on the test set
print('Evaluating The Model...')
test_data = torch.utils.data.TensorDataset(test_X, test_y)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        test_loss += loss.item() * batch_X.size(0)

        # Accuracy calculation
        true_labels = torch.argmax(batch_y, dim=1)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == true_labels).sum().item()
        total += batch_y.size(0)


test_loss /= len(test_loader.dataset)
accuracy = correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4%}")


# Save the trained model
torch.save(model.state_dict(), f"{dataset_folder_path}/NN/simple_classifier.pt")
print("Model saved!")
