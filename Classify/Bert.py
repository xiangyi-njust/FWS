from glob import escape
import tensorflow as tf
import pandas as pd
from transformers import TFBertForSequenceClassification,BertTokenizer
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import datetime
import sys
max_len = 40

class Bert:
    def __init__(self,arg_val):
        self.weight = arg_val['weight']
        self.tokenizer = BertTokenizer.from_pretrained(self.weight)
        self.model = TFBertForSequenceClassification.from_pretrained(self.weight,num_labels=6)
        self.epochs = arg_val['epochs']
        self.batch_size = arg_val['batch_size']
        self.path = arg_val['path']
        self.isTrained = arg_val['isTrained']
        self.model_save_path = arg_val['model_save_path']
        
        self.data = None
        self.ds_test_encoded = None
        self.ds_train_encoded = None
        self.y_labels_to_id = None

    def convert_example_to_feature(self,text):
        return self.tokenizer.encode_plus(text, 
                                    add_special_tokens = True, # add [CLS], [SEP]
                                    max_length = max_len, # max length of the text that can go to BERT
                                    pad_to_max_length = True, # add [PAD] tokens
                                    return_attention_mask = True, # add attention mask to not focus on pad tokens
                                    )

    def map_example_to_dict(self,input_ids, attention_masks, token_type_ids, label):
        return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_masks,
    }, label

    def encode_examples(self,sens,labels):
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
        
        for i in range(len(sens)):
            review = sens[i]
            label = self.y_labels_to_id[str(labels[i])]
            
            bert_input = self.convert_example_to_feature(review)
            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])
        
        return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(
            self.map_example_to_dict)

    def readData(self):
        self.data = pd.read_excel(self.path)
        y_labels = [str(item) for item in self.data.CAT.value_counts().index.tolist()]
        self.y_labels_to_id = dict([(item, i)for i,item in enumerate(y_labels)])
        
        sens = self.data['text'].tolist()
        labels = self.data['CAT'].tolist()
        train_sens,test_sens,train_labels,test_labels = train_test_split(sens,labels,test_size=0.1,random_state=11)
        
        self.ds_train_encoded = self.encode_examples(train_sens,train_labels).batch(self.batch_size)
        self.ds_test_encoded = self.encode_examples(test_sens,test_labels).batch(self.batch_size)

    def train(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5,epsilon=1e-08,clipnorm=1)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        if self.isTrained == False:
            self.model.fit(self.ds_train_encoded, epochs=self.epochs, validation_data=self.ds_test_encoded,
            callbacks=[callback])
            # .ckpt format
            self.model.save_weights(self.model_save_path)
        else:
            self.model.load_weights(self.model_save_path)

    def test(self):
        preds,trues = [],[]
        for i,item in enumerate(self.ds_test_encoded.as_numpy_iterator()):
            preds.extend([np.argmax(item) for item in self.model.predict(item[0]).logits])
            trues.extend(item[1])

        start = datetime.datetime.now()
        standard_format = str(start.year)+'-'+str(start.month)+'-'+str(start.day)+"  "+ \
                          str(start.hour)+':'+str(start.minute)+':'+str(start.second)
        
        model_name =''
        if self.weight == 'lordtt13/COVID-SciBERT':
            model_name = 'scibert'
        else:
            model_name = 'bert'
        
        with open('logs.txt','a') as f:
            sys.stdout = f     
            print(model_name)
            print("************",end="")
            print(standard_format,end="")
            print("************")

            print(sm.classification_report(trues,preds,digits=4))