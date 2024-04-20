import json
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # Import tqdm for progress bars
from train_utils import bag_of_words, tokenize, stem
from mai import maiNeuralNetWork


def load_config(config_path):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding failed in '{config_path}'. Details: {e}")
        return None


config = load_config('config.json')
if config is not None:
    print("Configuration loaded successfully.")
else:
    print("Failed to load configuration.")

# Extract hyperparameters
training_params = config['training_params']
num_epochs = training_params['num_epochs']
learning_rate = training_params['learning_rate']
hidden_size = training_params['hidden_size']
batch_size = training_params['batch_size']
patience = training_params['patience']
dropout_prob = training_params['dropout']  # Add dropout probability

# Define function to plot training and validation loss
def plot_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)


with open('data/MaiDataset.json', 'r') as f:
    intents = json.load(f)

# Remove the check for the 'intents' key and directly proceed with processing the dataset

all_words = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X = []
y = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y.append(label)

# Manually split the data
split_ratio = 0.8  # 80% train, 20% validation
split_index = int(len(X) * split_ratio)

X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

input_size = len(X_train[0])
output_size = len(tags)

print(input_size, output_size)


class maiDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = torch.tensor(X).float()
        self.y_data = torch.tensor(y)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


train_dataset = maiDataset(X_train, y_train)
val_dataset = maiDataset(X_val, y_val)

# Convert target labels to LongTensor
train_dataset.y_data = train_dataset.y_data.long()
val_dataset.y_data = val_dataset.y_data.long()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = maiNeuralNetWork(input_size, hidden_size,
                         output_size, dropout_prob).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss values for plotting
train_loss_values = []
val_loss_values = []

# Initialize consecutive worse epochs counter
consecutive_worse_epochs = 0

# Train the model
for epoch in range(num_epochs):
    model.train()
    train_epoch_loss = 0
    train_loader_size = len(train_loader)
    for i, (words, labels) in enumerate(train_loader):
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()

        # Display progress bar
        print(
            f'\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {train_epoch_loss / (i+1):.4f}', end='', flush=True)

    # Calculate average training loss for the epoch
    train_epoch_loss /= train_loader_size
    train_loss_values.append(train_epoch_loss)

    # Validation
    model.eval()
    val_epoch_loss = 0
    val_loader_size = len(val_loader)
    for i, (words, labels) in enumerate(val_loader):
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        val_epoch_loss += loss.item()

    # Calculate average validation loss for the epoch
    val_epoch_loss /= val_loader_size
    val_loss_values.append(val_epoch_loss)

    if (epoch+1) % 100 == 0:
        print(
            f', Val Loss: {val_epoch_loss:.4f}')

    # Check if validation loss exceeds training loss
    if val_epoch_loss > train_epoch_loss:
        consecutive_worse_epochs += 1
    else:
        consecutive_worse_epochs = 0

    if consecutive_worse_epochs >= patience:
        print(
            f'\nValidation loss exceeded training loss for {patience} consecutive epochs. Exiting training loop.')
        plot_loss(train_loss_values, val_loss_values, 'graphs/loss_plot.png')
        break

print(
    f'final training loss: {train_loss_values[-1]:.4f}, final validation loss: {val_loss_values[-1]:.4f}')
plot_loss(train_loss_values, val_loss_values, 'graphs/loss_plot.png')
