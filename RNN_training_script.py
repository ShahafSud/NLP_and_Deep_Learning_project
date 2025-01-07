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
vocab_size = 50


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, num_words=7):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_words = num_words

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer to map accumulated outputs to the final output
        self.fc = nn.Linear(num_words * hidden_size * num_layers, output_size)

    def forward(self, x):
        # Initialize hidden state (for RNN)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)  # out shape: (batch_size, seq_length, hidden_size)

        # Initialize the accumulated output tensor
        accumulated_out = torch.zeros(x.size(0), self.num_words * self.hidden_size * self.num_layers).to(x.device)

        # Iterate over time steps (words) to accumulate the RNN outputs
        for i in range(self.num_words):
            # Extract hidden states for all layers at time step i
            hidden_states_at_i = out[:, i, :]  # Shape: (batch_size, hidden_size)

            # Add the hidden states to the correct slice in the accumulated output
            accumulated_out[:, i * self.hidden_size * self.num_layers : (i + 1) * self.hidden_size * self.num_layers] = \
                hidden_states_at_i.unsqueeze(1).repeat(1, self.num_layers)  # Repeat for num_layers

        # Pass the accumulated output through the fully connected layer
        out = self.fc(accumulated_out)  # shape: (batch_size, output_size)

        return out


dataset_folder_path = 'data_folder'
num_classes = 28

print('Loading Dataset...')

with open(f'{dataset_folder_path}/RNN/train_data.pkl', 'rb') as f:
    train_X, train_y = pickle.load(f)
with open(f'{dataset_folder_path}/RNN/val_data.pkl', 'rb') as f:
    val_X, val_y = pickle.load(f)
with open(f'{dataset_folder_path}/RNN/test_data.pkl', 'rb') as f:
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

model = RNNModel(input_size=vocab_size, hidden_size=h_size, num_layers=n_layers, output_size=num_classes).to(device)

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
