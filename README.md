# Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain

## Overview
<b> Data and Code for paper "Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain"

The research content of the paper is to extract fws from acadamic paper,we choose the nlp as example,use the ACL、EMNLP、NAACL as our origin dataset, 
* After human annotation the future work sentence,we use some traditional machine learning models to judge one sentence is fws or not.
* Ater that,we classify the fws in paper into six types,we use bert、textcnn、bilstm to implement the experiment.
* In additional,We compared the differences between keywords extracted from future work sentences and abstracts to verify whether
the work in future work sentences would be realistic in subsequent real studies.

## Directory structure
FWS
<pre>
├─ Classify
│    ├─ Bert.py
│    ├─ Bilstm.py
│    ├─ TextCNN.py
│    ├─ data
│    │    └─ TypeClassify.xlsx
│    ├─ deep_learning.ipynb
│    ├─ main.py
│    └─ weights
│           ├─ bert
│           ├─ bilstm
│           ├─ scibert
│           └─ textcnn
├─ Evaluate
│    ├─ ExtractKeyphrases.ipynb
│    ├─ Rejust.ipynb
│    └─ data
│           ├─ keyphrase
│           └─ raw
├─ Extract
│    ├─ data
│    │    └─ recognition.xlsx
│    └─ main.py
└─ README.md
</pre>

## Dataset discription

## Quick start

> recognition 
<pre>python main.py </pre>

> classify
when you input these command,some tips will output to help to use,such as:
<pre>python main.py --model bert --weight bert-base-uncased --batch_size 64 --isTrained True --path data/TypeClassify --model_save_path weight/bert/</pre>

> extract keywords
we provide two notebooks,you can follow the steps to extract keywords and do some preprocess work

## Citation
Please cite the following paper if you use this codes and dataset in your work
