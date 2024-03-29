import json
import torch
import nltk

with open("data/dataset.json", "r") as json_data:
    dataset = json.load(json_data)
    
with open('data/additional_words.json') as file:
    json_data = json.load(file)

json_set = set(json_data)

from nltk.corpus import stopwords
from string import punctuation
from nlp_function import (
    tokenization,
    stemming_token,
    vectorization,
)
from nn_model import neural_network

nltk.download("stopwords")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE = "data/data.pth"
map_location = torch.device("cpu")
data = torch.load(FILE, map_location)

input_layer = data["input_layer"]
hidden_layer = data["hidden_layer"]
output_layer = data["output_layer"]
all_token = data["all_token"]
tags = data["tags"]
model_state = data["model_state"]

model = neural_network(input_layer, hidden_layer, output_layer).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Uvers"


def get_response(msg):
    sentence = tokenization(msg)
    sentence = [word.lower() for word in sentence if word not in punctuation]
    stop_words = set(stopwords.words("indonesian"))
    stop_words.update(json_set)
    sentence = [word for word in sentence if not word in stop_words]
    sentence = [stemming_token(w) for w in sentence]
    
    sentence = " ".join(sentence)
    sentence = sentence.replace("kos-kosan", "kos").replace("kost-kostan", "kos").replace("kost", "kos").replace("kost", "kos").replace("asrama", "kos")
    sentence = sentence.split()

        
    print(sentence)

    X = vectorization(sentence, all_token)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(tag)
    print(prob)

    if prob.item() > 0.7:
        for intent in dataset["intents"]:
            if tag == intent["tag"]:
                return intent["responses"]

    else:
        return "Maaf, saya tidak mengerti maksud Anda.<br /><br />Bisakah Anda memeriksa kembali kalimat Anda untuk memastikan tidak ada kesalahan pengetikan?<br /><br />Untuk bantuan lebih lanjut, mohon hubungi nomor layanan pelanggan kami di:<br /><br />0778 473399 / 466869<br />atau<br /><a href=\"https://bit.ly/whatsappOfficialUvers\">WhatsApp Official UVERS</a><br /><br />Atau kirimkan email ke <a href=\"mailto:admisi@uvers.ac.id\">admisi@uvers.ac.id</a>"


if __name__ == "__main__":
    print(
        "Anda telah terhubungan dengan Uvers (ketik 'quit' untuk mengakhiri percakapan)"
    )
    while True:
        print("\n")
        sentence = input("Kamu: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
