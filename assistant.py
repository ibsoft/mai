import os
import random
import json

import torch

from mai import maiNeuralNetWork
from train_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/MaiDataset.json', 'r') as json_data:
    intents = json.load(json_data)


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

FILE = "model/mai-model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

training_params = config['training_params']
dropout_prob = training_params['dropout']

model = maiNeuralNetWork(input_size, hidden_size,
                         output_size, dropout_prob).to(device)
model.load_state_dict(model_state)
model.eval()

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


bot_name = "MAI"
print("Let's GO! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
