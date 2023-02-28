import openai

openai.api_key = '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'

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import json
with open('data/dataset.json', 'r') as json_data:
    dataset = json.load(json_data)

import random

from nlp_function import tokenization, remove_punctuation, remove_stopWords, stemming_token, vectorization
from nn_model import neural_network

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE = "data/data.pth"
data = torch.load(FILE)

input_layer = data["input_layer"]
hidden_layer = data["hidden_layer"]
output_layer = data["output_layer"]
all_token = data['all_token']
tags = data['tags']
model_state = data["model_state"]

model = neural_network(input_layer, hidden_layer, output_layer).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Uvers"

def get_response(msg):
    sentence = tokenization(msg)
    sentence = [word.lower() for word in sentence if word not in punctuation]
    stop_words = set(stopwords.words('indonesian'))
    sentence = [word for word in sentence if not word in stop_words]
    sentence = [stemming_token(w) for w in sentence]
    
    X = vectorization(sentence, all_token)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    completion = openai.Completion.create(
        model = "text-davinci-003",
        prompt = msg,
        max_tokens = 1000,
        temperature = 0,
        n = 1
    )
    
    if prob.item() > 0.75:
        for intent in dataset['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {intent['responses']}")
                # return intent['responses']
    
    else:
        # response = completion.choices[0].text
        # print(f"{bot_name}: {response}")
    
        print(f"{bot_name}: Saya tidak mengerti...Tolong masukan kata kunci yang lain")
        # return "Saya tidak mengerti...Tolong masukan kata kunci yang lain!"
        
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

