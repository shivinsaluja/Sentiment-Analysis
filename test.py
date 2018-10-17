
import numpy as np

from keras.models import model_from_json


from nltk.tokenize import  word_tokenize

import tensorflow as tf
import re
import en_core_web_sm

nlp = en_core_web_sm.load()

pos_set={'NN','NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ','RB','RBR','RBS'}

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global model
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    global graph
    graph = tf.get_default_graph()



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

def pad_vec_sequences(sequences,maxlen=20):
    new_sequences = []
    for sequence in sequences:
        orig_len, vec_len = np.shape(sequence)
        if orig_len < maxlen:
            new = np.zeros((maxlen,vec_len))
            new[maxlen-orig_len:,:] = sequence
        else:
            new = sequence[orig_len-maxlen:,:]
            
        new_sequences.append(new)
    return np.array(new_sequences)
    



def filter_word(word,postag):
    if postag in pos_set:
        return word
    else:
        return ''
    
    
def test_sentence(x_sent):
    
    load_model()
    x_sent = x_sent.lower()
    x_sent = remove_special_characters(x_sent)
    
    word_tokens = word_tokenize(x_sent)
    tokenized_sentence = [w for w in word_tokens]
    #tokenized_sentence.sort()
    
    new_filtered_sentence=[]
    word_tokens_spacy = nlp(x_sent)
    token_map = {}
                
    for tok in word_tokens_spacy:
        token_map[str(tok.text)] = str(tok.tag_)
        
    for i in tokenized_sentence:
        temp = filter_word(i,token_map[str(i)])
        new_filtered_sentence.append(temp)
                    
    new_sent=""
    for i in new_filtered_sentence:
        new_sent = new_sent + str(i).strip() +" "
                    
    new_sent = remove_special_characters(new_sent)
    new_sent = new_sent.strip()
    

    x_sent = new_sent
    print(x_sent)
    
    All_statements = []
    check = []
    X_all_doc_vec = []
    x_doc = nlp(str(x_sent))
    x_doc_vec = x_doc.vector/x_doc.vector_norm
    x_vec_seq = []
    
    for word in x_doc:
        
        x_vec_seq.append(word.vector/word.vector_norm)
        
        
    x_vec_seq = np.array(x_vec_seq)
    All_statements.append(x_sent)
    X_all_doc_vec.append(x_doc_vec)
    check.append(x_vec_seq)
    check = pad_vec_sequences(check)
    
    
    with graph.as_default():
        
        ynew = np.argmax((model.predict(check)))
        check_prob = model.predict(check)
        print(check_prob)
        

        if ynew==0:
            print("This is a negative sentence")
        else:
            print("This is a positive sentence")
            
            