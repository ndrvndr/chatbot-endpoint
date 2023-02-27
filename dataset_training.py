import json
with open('data/dataset.json', 'r') as f:
    dataset = json.load(f)
    
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nlp_function import tokenization, remove_punctuation, remove_stopWords, stemming_token, vectorization
from nn_model import neural_network

all_token = []
tags = []
xy = []

for dataset in dataset['intents']:
    tag = dataset['tag']
    tags.append(tag)
    for pattern in dataset['patterns']:
        w = tokenization(pattern)
        all_token.extend(w)
        xy.append((w, tag))

all_token = remove_punctuation(all_token)
all_token = remove_stopWords(all_token)
all_token = [stemming_token(w) for w in all_token]
all_token = sorted(set(all_token))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    
    # bag = remove_punctuation(pattern_sentence)
    # bag = remove_stopWords(pattern_sentence)
    
    bag = vectorization(pattern_sentence, all_token)
    
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1100
batch_size = 10
learning_rate = 0.001
input_layer = len(X_train[0])
hidden_layer = 5
output_layer = len(tags)
print(input_layer)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = neural_network(input_layer, hidden_layer, output_layer).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_layer": input_layer,
"hidden_layer": hidden_layer,
"output_layer": output_layer,
"all_token": all_token,
"tags": tags
}

FILE = "data/data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
