import pandas as pd

import csv

import re
import en_core_web_sm
import nltk
from train import train_dataset 
from test import test_sentence 

nlp = en_core_web_sm.load()


pos_set={'NN','NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ','RB','RBR','RBS'}



def filter_word(word,postag):
    if postag in pos_set:
        return word
    else:
        return ''


def remove_special_characters(question):
    question = re.sub(r'[^\w]', ' ', question)
    question = re.sub(' +',' ',question)
    question = re.sub(r"n\'t", " not", question)
    question = re.sub(r"\'re", " are", question)
    question = re.sub(r"\'s", " is", question)
    question = re.sub(r"\'d", " would", question)
    question = re.sub(r"\'ll", " will", question)
    question = re.sub(r"\'t", " not", question)
    question = re.sub(r"couldnt ", " could not ", question)
    question = re.sub(r"thats ", " that is ", question)
    question = re.sub(r"cant ", " can not ", question)
    question = re.sub(r"dont ", " do not ", question)
    question = re.sub(r"im ", " i am ", question)
    question = re.sub(r"5lb ", " ", question)
    question = re.sub(r"15g ", " ", question)
    question = re.sub(r"shouldve ", " should have ", question)
    question = re.sub(r"wont ", " will not ", question)
    question = re.sub(r"hes ", " he is ", question)
    question = re.sub(r"ive ", " i have ", question)
    question = re.sub(r"wouldnt ", " would not ", question)
    question = re.sub(r"8pm ", " ", question)
    question = re.sub(r" s ", " is ", question)
    question = re.sub(r"wasn t ", " was not ", question)
    question = re.sub(r"don t ", " do not ", question)
    question = re.sub(r"didn t ", " did not ", question)
    return question




def clean_dataset():
     file=pd.read_csv('dataset.csv',encoding='latin-1')
     f = open('dataset_cleaned.csv', 'w', newline='')
     writer = csv.writer(f)
     
     sentence = file['text']
     labellist = file['sentiment']
     writer.writerow(['text','sentiment'])
     for x_sent,label in zip(sentence,labellist):
            if len(x_sent) > 0:
                x_sent = x_sent.lower()
                x_sent = remove_special_characters(x_sent)
                
                print(x_sent)
                
                word_tokens = nltk.word_tokenize(x_sent)
                filtered_sentence = [w for w in word_tokens]
                
                new_filtered_sentence=[]
                
     
                token_map = {}
                token_map = nltk.pos_tag(word_tokens)
                token_dictionary={}
                
                for i in token_map:
                    token_dictionary[str(i[0])] = str(i[1])


                for i in filtered_sentence:
                    if (str(i[0])=='@'):
                        continue
                    else:
                        temp = filter_word(str(i),str(token_dictionary[str(i)]))
                        new_filtered_sentence.append(temp)
                    
                new_sent=""
                for i in new_filtered_sentence:
                    new_sent = new_sent + str(i).strip() +" "
                    
                new_sent = remove_special_characters(new_sent)
                new_sent = new_sent.strip()
                
                writer.writerow([new_sent,label])

 
#This function is used to clean our dataset    
clean_dataset()
#This function helps to train on our dataset
train_dataset()




sentence="I loved the food that was served here"
test_sentence(sentence)
     