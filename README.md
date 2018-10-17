# Sentiment-Analysis

The objective of this project is to develop an algorithm which can detect whether the sentiment of a given sentence is positive or negative. The algorithm makes use of a Recurrent Neural Network model which is implemented using LSTM's. It is a classification problem solved using the aforementioned algorithm. The model is implemented using a Bi-Directional LSTM.


# Approach

## Data Cleaning and Pre-Processing

The dataset contains a collection of reviews from various social websites such as Yelp,IMDB and AMAZON with their respective sentiments i.e postive sentiment (label 1) and negative sentiment (label 0). 
The dataset is cleaned and special symbols and characters are removed using various libararies in python. Certain short forms of different words are replaced with their proper forms for e.g couldnt is changed to could not and i'll is changed to I will.Stop words and common words are removed from all the messages using Parts of Speech (Pos) tagging and capital letter are converted to small letters. Each message/text is converted to tokens and is furthur converted into vectors (word embeddings) of dimension 384.Padding is applied to each message so that each message/text is of same length.

## Training 

The training is done using a Bi-directional LSTM network with hidden dimension as 300 nodes. The output layer of the network is a dense layer with softmax activation function applied on it. Early stopping has also been implemented to decrease the training time. The model configeration is saved in model.json and weights are saved in model.h5.

## Steps to Run 

1. The main file is main.py
2. The training can be done using the function train_dataset() in the file train.py
3. Testing to determine whether a sentence is negative or positive is determined using the function test_sentence() in the file test.py
4. The dataset should be in a file named 'dataset.csv' and the file should contain 2 labels namely 'text' and 'Sentiment'
