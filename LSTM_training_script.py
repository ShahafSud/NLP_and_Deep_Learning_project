import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

random.seed(0)
learning_rate = 0.001
n_epochs = 10
h_size = 128
n_layers = 6
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the last time step's output
        out = out[:, -1, :]
        out = self.fc(out)
        return out


dataset_folder_path = 'data_folder'
num_classes = 28

print('Loading Dataset...')
with open(f'{dataset_folder_path}/Transformer/train_data.pkl', 'rb') as f:
    train_X, train_y = pickle.load(f)
with open(f'{dataset_folder_path}/Transformer/val_data.pkl', 'rb') as f:
    val_X, val_y = pickle.load(f)
with open(f'{dataset_folder_path}/Transformer/test_data.pkl', 'rb') as f:
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

model = LSTMModel(input_size=7, hidden_size=h_size, num_layers=n_layers, output_size=num_classes).to(device)

print('Training The Model...')

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = n_epochs
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
torch.save(model.state_dict(), f"{dataset_folder_path}/Transformer/LSTM_classifier.pt")
print("Model saved!")
