#from msilib.schema import Class
from tensorflow.keras.layers import Input,Embedding,Conv1D,MaxPool1D,concatenate,Flatten,Dropout
from tensorflow.keras.layers import Dense,MaxPooling1D,Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import sys
import datetime

class TextCNN():
    def __init__(self,shell_args):
        self.model = None
        self.tokenizer = None
        self.x = None
        self.y = None 
        
        self.path = shell_args['path']
        self.batch_size = shell_args['batch_size']
        self.epochs = shell_args['epochs']
        self.isTrained = shell_args['isTrained']
        self.model_save_path = shell_args['model_save_path']

    def net(self):
        main_input = Input(shape=(40,))
        # word embedding
        embed = Embedding(len(self.tokenizer.word_index) + 1, 300, input_length=40)(main_input)
        
        cnn1 = Conv1D(200, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=38)(cnn1)
        cnn2 = Conv1D(200, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=37)(cnn2)
        cnn3 = Conv1D(200, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=36)(cnn3)

        # merge the layer output
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = Flatten()(cnn)
        drop = Dropout(0.2)(flat)
        main_output = Dense(6, activation='softmax')(drop)
        
        self.model = Model(inputs=main_input, outputs=main_output)
    
    def readData(self):
        data = pd.read_csv(self.path)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(data.text.values.tolist())

        label_transform_dict = {}

        labels = data['label'].tolist()
        set_labels = list(set(labels))
        for label in set_labels:
            label_transform_dict[str(label)] = int(label)-1

        self.x = pad_sequences(self.tokenizer.texts_to_sequences(data.text.values.tolist()),maxlen=40,padding='post')
        # convert the label array into one-hot array
        self.y = np.eye(len(label_transform_dict))[[label_transform_dict[str(label)] for label in labels]]

    def train(self):
        if self.isTrained == False:
            # data
            train_x,o_x,train_y,o_y = train_test_split(self.x,self.y,test_size=0.2,random_state=11)
            test_x,_,test_y,_ = train_test_split(o_x,o_y,test_size=0.5,random_state=11)
            
            # model setting
            self.net()
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
            
            # train 
            print()
            print("**********START TRAIN*******************")
            print()
            self.model.fit(train_x, train_y, batch_size=self.batch_size, epochs=self.epochs,
            validation_data=(test_x,test_y),callbacks=[early_stop])
            self.model.save(self.model_save_path)
            print()
            print("**********END TRAIN*******************")
        else:
            self.model = keras.models.load_model(self.model_save_path)
        
    def test(self):
        _,x,_,y = train_test_split(self.x,self.y,test_size=0.1,random_state=11)
        _,valid_x,_,valid_y = train_test_split(x,y,test_size=0.5,random_state=11)
        
        preds = self.model.predict(valid_x)

        start = datetime.datetime.now()
        standard_format = str(start.year)+'-'+str(start.month)+'-'+str(start.day)+"  "+ \
                          str(start.hour)+':'+str(start.minute)+':'+str(start.second)
        
        model_name = 'textcnn'

        with open('logs.txt','a') as f:
            sys.stdout = f     
            print(model_name)
            print("************",end="")
            print(standard_format,end="")
            print("************")
            print(sm.classification_report([np.argmax(item) for item in valid_y], [np.argmax(item) for item in preds],digits=4))
