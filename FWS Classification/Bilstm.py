from tensorflow.keras.layers import Bidirectional,LSTM,Layer,Lambda,RepeatVector,Permute,Multiply,BatchNormalization
from tensorflow.keras.layers import Input,Embedding,Conv1D,MaxPool1D,concatenate,Flatten,Dropout
from tensorflow.keras.layers import Dense,MaxPooling1D,Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow import keras
from keras import initializers, regularizers, constraints
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import pandas as pd
import sys
import datetime

class Bilstm():
    def __init__(self,shell_args):
        self.model = None
        self.tokenizer = None
        self.x = None
        self.y = None 
        
        self.model_save_path = shell_args['model_save_path']
        self.isTrained = shell_args['isTrained']
        self.path = shell_args['path']
        self.batch_size = shell_args['batch_size']
        self.epochs = shell_args['epochs']
    
    def attention_3d_block(self,inputs):
        input_dim = int(inputs.shape[2])
        a = inputs
        a = Dense(input_dim,activation='softmax')(a)
    
        if False:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((1, 2), name='attention_vec')(a)

        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul

    def net(self):
        # input layer
        input = Input(shape=(40,),dtype='int32')
        # embedding layer
        net = Embedding(len(self.tokenizer.word_index)+1,300,mask_zero=True)(input)
        # bilstm layer
        net = Bidirectional(LSTM(128,dropout=0.2,return_sequences=True))(net)
        # attention layer
        net = BatchNormalization()(net)
        net = self.attention_3d_block(net)
        net = Flatten()(net)
        # output layer
        outputs = Dense(6,activation='softmax')(net)
        self.model = Model(inputs=input,outputs=outputs)

    def readData(self):
        data = pd.read_excel(self.path)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(data.text.values.tolist())

        y_labels = [str(item) for item in data.CAT.value_counts().index.tolist()]
        y_labels_to_id = dict([(item, i)for i,item in enumerate(y_labels)]) 

        self.x = pad_sequences(self.tokenizer.texts_to_sequences(data.text.values.tolist()),maxlen=40,padding='post')
        #将一个label数组转化成one-hot数组。
        self.y = np.eye(len(y_labels_to_id))[[y_labels_to_id[str(item)] for item in data.CAT.values.tolist()]]

    def train(self):
        # data
        if self.isTrained == False:
            train_x,o_x,train_y,o_y = train_test_split(self.x,self.y,test_size=0.2,random_state=11)
            test_x,_,test_y,_ = train_test_split(o_x,o_y,test_size=0.5,random_state=11)

            self.net() 
            self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
            
            print()
            print("**********START TRAIN*******************")
            print()
            self.model.fit(train_x, train_y, batch_size=self.batch_size, 
                        epochs=self.epochs,validation_data=(test_x,test_y),callbacks=[early_stop]
                        )
            self.model.save(self.model_save_path)
            print()
            print("**********END TRAIN*******************")
        else:
            self.model = keras.models.load_model(self.model_save_path)
    
    def test(self):
        print()
        print("predict result:")
        _,x,_,y = train_test_split(self.x,self.y,test_size=0.1,random_state=11)
        _,valid_x,_,valid_y = train_test_split(x,y,test_size=0.5,random_state=11)

        preds = self.model.predict(valid_x)

        start = datetime.datetime.now()
        standard_format = str(start.year)+'-'+str(start.month)+'-'+str(start.day)+"  "+ \
                          str(start.hour)+':'+str(start.minute)+':'+str(start.second)

        model_name = 'bilstm'
        
        with open('logs.txt','a') as f:
            sys.stdout = f     
            print(model_name)
            print("************",end="")
            print(standard_format,end="")
            print("************") 
            print(sm.classification_report([np.argmax(item) for item in valid_y], 
                 [np.argmax(item) for item in preds],digits=4)
                 )

    
