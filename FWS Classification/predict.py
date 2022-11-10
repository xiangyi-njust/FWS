import argparse
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str,default='../Dataset/Corpus_For_FWS_TypeClassify.xlsx')

def MapExampleToList(self,input_ids,attention_masks,token_type_ids):
    return {
        "input_ids":input_ids,
        "token_type_ids":token_type_ids,
        "attention_mask":attention_masks,
    }

def GetData(config):

    pred_texts,weight,max_len = config

    tokenizer = BertTokenizer.from_pretrained('')

    inputs = tokenizer.encode_plus(pred_texts,
                                add_special_tokens=True,
                                max_len = max_len,
                                pad_to_max_length=True,
                                return_attention_mask=True
                                )
    
    input_ids = inputs['input_id']
    attention_masks = inputs['attention_mask']
    token_ids = inputs['token_type_ids']

    return tf.data.Dataset.from_tensor_slices((input_ids,attention_masks,token_ids)).map(MapExampleToList)

def main():
    args = parser.parse_args()
    shell_args = args._get_kwargs()
    arg_val = {}
    for arg,val in shell_args:
        arg_val[arg] = val

    # construct data
    path = arg_val['path']
    weight_path = arg_val['weight_path']
    weight = ''

    df = pd.read_excel(path)
    texts = df['text'].tolist()
    config = (texts,weight,60)
    dataset = GetData(config)

    # construct model
    model = BertForSequenceClassification.from_pretrained(weight,num_classes=6)
    model.load_weight(weight_path)

    preds = []
    for i,item in enumerate(dataset.as_numpy_iterator()):
        preds.extend([np.argmax(item) for item in model.predict(item[0]).logits])

    df['label'] = preds
    df.to_excel(path)

if __name__=='__main__':
    main()