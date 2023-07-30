# Quora Question Pairs Similarity using BERT
![image](https://github.com/mohamed-em2m/sentaces_similarty/assets/126331291/4f21695e-33d9-4618-827d-4488ad90f437)

This project analyzes the similarity of sentence pairs from the Quora Question Pairs dataset using BERT (Bidirectional Encoder Representations from Transformers) model.

## Dependencies

- Python 3.x
- transformers
- torch
- streamlit
- tqdm
- ...

## Installation

To run this code, follow these steps:

1. Clone this repository:

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
Install the required libraries:
bash
Copy code
pip install transformers torch streamlit tqdm ...
Run the Streamlit app to check sentence similarity:
bash
Copy code
streamlit run app.py
Usage
After running the Streamlit app, open your web browser and go to the provided link (e.g., http://localhost:8501).

Enter two sentences in the input boxes.

Click the "Check Similarity" button to see if the sentences are the same or not.

Example
Here's an example of how to use the code in Python:

python
Copy code
# Python code to analyze sentence similarity using BERT

import torch
import streamlit as st
import spacy

# [Add the rest of the code here]
# ...
# ...
# ...

# Example usage
word = ['my name is mohamed ', ...]
for i in range(9):
    r = torch.randint(len(word), size=(1,))
    r2 = torch.randint(len(word), size=(1,))
    h = tok(word[r], word[r2])    
    e = model(h)
    ans = 'the same' if int(torch.sigmoid(e) >= .5) else 'not the same'
    print(f'{word[r]} is {ans} {word[r2]}')
License
[Add your license text or link here, e.g., MIT License]

Full Code Repository
Find the complete code, data, and other related files in our GitHub repository:

Link to your GitHub repository

css
Copy code

In this improved version, we added a section explaining the usage of the code, provided an

```bash
The code starts with importing the required libraries and installing the transformers library using pip.

It imports necessary modules from PyTorch, Transformers, and Streamlit. Streamlit is a Python library used to build interactive web applications for data science and machine learning.

It initializes the BERT tokenizer by loading the pre-trained "bert-base-uncased" model checkpoint from Hugging Face's AutoTokenizer class.

The code defines a custom PyTorch module called bert_compare, which is a neural network for sentence comparison using BERT and a Convolutional Neural Network (CNN) layer.

In the forward() method of the bert_compare module, it processes the input data using BERT, applies two CNN layers with ReLU activation, flattens the output, and applies a linear layer to obtain the final comparison score.

A model object is instantiated from the bert_compare class, and an AdamW optimizer is created to optimize the model's parameters.

The code defines a helper function tok(x, y) to tokenize the input sentences using the pre-trained tokenizer. It returns the tokenized input data as PyTorch tensors.

A sample input is tokenized using the tok() function, and the model is called with this input. The model processes the input through its forward() method and returns a score for sentence similarity.

The code installs the tqdm library, which provides a progress bar for loops.

The model is set to training mode using model.train().

The code defines a list of example sentences word to compare sentence pairs.

A loop iterates over random sentence pairs from the word list, tokenizes each pair, feeds them to the model, calculates the similarity score, and prints the result.

Overall, this code demonstrates how to use BERT for sentence similarity analysis using a custom PyTorch model that includes a CNN layer. It tokenizes the input sentences, processes them through the model, and outputs similarity scores for random sentence pairs from the provided word list. ```
