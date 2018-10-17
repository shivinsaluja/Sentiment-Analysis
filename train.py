

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, LSTM, Input,add
from keras.utils import np_utils
import keras.backend as K
from nltk.tokenize import  word_tokenize

from keras.callbacks import EarlyStopping


import en_core_web_sm
nlp = en_core_web_sm.load()



class ProcessDataset(object):
    
    def __init__(self):


        All_statements = []
        All_statements_vectors = []
        X_all_doc_vec = []
        
        All_Sentiments = []
        
        file = pd.read_csv('dataset_cleaned.csv',encoding='latin-1')
        
        sentence = file['text']
        labellist = file['sentiment'];

        for x_sent,label in zip(sentence,labellist):
                x_sent = str(x_sent)
                if len(x_sent) > 0:
                    x_sent = x_sent.lower()
                    word_tokens = word_tokenize(x_sent)
                    
                    tokenized_sentence = [w for w in word_tokens]
                    x_sent = ""
                    
                    for i in tokenized_sentence:
                        
                        x_sent = x_sent+str(i).strip()+" "

                    x_doc = nlp(str(x_sent))
                    
                    x_doc_vec = x_doc.vector/x_doc.vector_norm
                    x_vec_seq = []
                    
                    for word in x_doc:
                        x_vec_seq.append(word.vector/word.vector_norm)
                    
                    x_vec_seq = np.array(x_vec_seq)
                    All_statements.append(x_sent)
                    X_all_doc_vec.append(x_doc_vec)
                    All_statements_vectors.append(x_vec_seq)
                    All_Sentiments.append(label)



        self.All_statements = All_statements
        self.All_statements_vectors = All_statements_vectors
        self.X_all_doc_vec = X_all_doc_vec
        self.All_Sentiments = All_Sentiments


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
    



def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


nb_classes = 2
epochs = 30



def train_dataset():
    
    batch_size = 30
    processed_dataset = ProcessDataset()
    print("Dataset Processed")
    X_all = pad_vec_sequences(processed_dataset.All_statements_vectors)
    check_dataset = X_all
    print(check_dataset)
    All_Sentiments = processed_dataset.All_Sentiments

    
    Labels = np_utils.to_categorical(All_Sentiments)
    x_train, x_test, y_train, y_test = train_test_split(X_all, Labels, test_size=0.2)
    
    print("Training for the very first time")
    max_len = 20
    hidden_dim = 300
    K.clear_session()
    
    sequence = Input(shape=(max_len,384), dtype='float32')
    
    forwards = LSTM(hidden_dim,dropout=0.1, recurrent_dropout=0.1)(sequence)
    
    backwards = LSTM(hidden_dim,dropout=0.1, recurrent_dropout=0.1,go_backwards=True)(sequence)
    merged = add([forwards, backwards])
    after_dp = Dropout(0.1)(merged)
    output = Dense(nb_classes, activation='softmax')(after_dp)
    model=Model(inputs=sequence, outputs=output)
    model.compile('adam', 'categorical_crossentropy',metrics=['accuracy', mean_pred])

    ES = EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto')
    model.fit(x_train, y_train,batch_size=batch_size,nb_epoch=epochs,validation_data=[x_test, y_test],callbacks=[ES])
    model_json = model.to_json()
    
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights("model.h5")
        print("Saved model to disk")
        