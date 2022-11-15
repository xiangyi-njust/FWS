import argparse
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer,TFBertForSequenceClassification
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str,default='../Dataset/Corpus_For_FWS_TypeClassify_Predict.csv')
parser.add_argument('--weight_path',type=str,default='scibert/')

def MapExampleToDict(input_ids, attention_masks, token_type_ids):
    return {
    "input_ids": input_ids,
    "token_type_ids": token_type_ids,
    "attention_mask": attention_masks,
    }

def GetData(config):

    pred_texts,weight,max_len = config

    tokenizer = BertTokenizer.from_pretrained(weight)

    input_ids,attention_masks,token_type_ids = [],[],[]

    for text in pred_texts:
      input = tokenizer.encode_plus(text,
                     add_special_tokens=True,
                     max_length=max_len,
                     pad_to_max_length=True,
                     return_attention_mask=True
                     )
      input_ids.append(input['input_ids'])
      attention_masks.append(input['attention_mask'])
      token_type_ids.append(input['token_type_ids'])

    return tf.data.Dataset.from_tensor_slices((input_ids,attention_masks,token_type_ids)).map(MapExampleToDict)

def main():
    args = parser.parse_args()
    shell_args = args._get_kwargs()
    arg_val = {}
    for arg,val in shell_args:
        arg_val[arg] = val

    # construct data
    path = arg_val['path']
    weight_path = arg_val['weight_path']
    weight = 'lordtt13/COVID-SciBERT'
    max_len = 40

    df = pd.read_csv(path)
    texts = df['text'].tolist()
    config = (texts,weight,max_len)
    dataset = GetData(config).batch(64)

    # construct model
    model = TFBertForSequenceClassification.from_pretrained(weight,num_labels=6)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5,epsilon=1e-08,clipnorm=1)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    model.load_weights(weight_path)

    preds = []

    # 预测的时候也是按照批量来的嘛
    for i,item in enumerate(dataset):
        preds.extend([np.argmax(item) for item in model.predict(item).logits])

    labels = [0,1,2,3,4,5]
    label_transform_dict = {}

    for label in labels:
        label_transform_dict[str(label)] = label+1

    preds = [label_transform_dict[label] for label in preds]

    df['label'] = preds
    df.to_csv('predict_result.csv')

if __name__=='__main__':
    main()