import numpy as np
import random
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from train_utils import bag_of_words, tokenize, stem
from mai import maiNeuralNetWork

# Clear screen based on the platform


def clear_screen():
    if os.name == 'posix':  # For Linux and macOS
        os.system('clear')
    elif os.name == 'nt':   # For Windows
        os.system('cls')
    else:
        # For other operating systems, print a bunch of newlines to mimic clearing
        print('\n' * 100)


clear_screen()


def print_banner():
    banner = """
                    __       __   ______   ______ 
                    /  \     /  | /      \ /      |
                    $$  \   /$$ |/$$$$$$  |$$$$$$/ 
                    $$$  \ /$$$ |$$ |__$$ |  $$ |  
                    $$$$  /$$$$ |$$    $$ |  $$ |  
                    $$ $$ $$/$$ |$$$$$$$$ |  $$ |  
                    $$ |$$$/ $$ |$$ |  $$ | _$$ |_ 
                    $$ | $/  $$ |$$ |  $$ |/ $$   |
                    $$/      $$/ $$/   $$/ $$$$$$/ 
                                                

                      Ioannis (Yannis) A. Bouhras  
   
    """
    print(banner)


print_banner()


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

# Define function to plot training loss


def plot_loss(train_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)


with open('data/MaiDataset.json', 'r') as f:
    intents = json.load(f)

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
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters

input_size = len(X_train[0])
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = maiNeuralNetWork(input_size, hidden_size,
                         output_size, dropout_prob).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss values for plotting
train_loss_values = []

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_values.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the training loss
plot_loss(train_loss_values, 'graphs/training_loss.png')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "model/mai-model.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
