# Chatbot Endpoint

This project aims to provide an HTTP request endpoint service that is connected to a frontend chatbot. It also implements an Indonesian language chatbot that can answer user questions regarding university enrollment information.

## About the Project

This project utilizes deep learning technology and Natural Language Processing (NLP) to provide an interactive experience for users who want to enroll in a university. The provided HTTP request endpoint allows users to interact with the chatbot through HTTP requests from the frontend.

## Key Features

**Indonesian Language Chatbot:** The chatbot is implemented using a deep learning model powered by the PyTorch library. The model has been trained using relevant data to answer user questions. Natural language processing is performed using the NLTK library for tokenization and removal of common words (stopwords) in the Indonesian language, as well as the Satrawi library for stemming Indonesian words. Additionally, built-in Python methods such as .lower() are used to convert all tokens to lowercase, and string.punctuation is used to remove punctuation marks.

**Customizable Topic:** The chatbot's topic can be easily modified by adjusting the dataset.json file with the desired theme. By updating the dataset with relevant training data, the chatbot can be trained to answer questions on a different topic. Additionally, the hyperparameters can be adjusted to accommodate the number of sample data and the complexity level of the chatbot.

## Installation

Clone the project

```bash
  git clone https://github.com/ndrvndr/chatbot-endpoint.git
```

Create virtual environment

```bash
  python -m venv .venv
```

Activate .venv

```bash
  .\.venv\Scripts\activate
```

Install Dependencies

```bash
  pip install
```

Run

```bash
  python app_demo.py to interact with bot via terminal

  or

  python app.py and copy webserviceurl/request to VITE_ENDPOINT,
  to interact with the bot via separate frontend
```

Separate frontend can be accessed [here](https://github.com/ndrvndr/chatbot-app)

## Attribution

This project is inspired and use code from the following [repo](https://github.com/patrickloeber/pytorch-chatbot)

## Authors

- [@ndrvndr](https://github.com/ndrvndr)

## Feedback

If you have any feedback, please reach out to me at andreavindra37@gmail.com

<!-- GitAds-Verify: 93A7R5W8INUZM4G8ZCW3JGI5VJRNG51Y -->
