# Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain

## Overview
<b> Data and Code for paper "Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain"

The research object of the paper is future work sentence(fws), we choose the nlp domain as example,use the ACL、EMNLP、NAACL as our origin dataset, our main work include below parts:
* After human annotation the future work sentence,we use some traditional machine learning models to judge one sentence is fws or not.
* Ater that,we classify the fws in paper into six types,we use bert、scibert、textcnn、bilstm to implement the experiment.
* In additional,We compared the differences between keywords extracted from future work sentences and abstracts to verify whether
the work in future work sentences would be realistic in subsequent real studies.

## Directory structure
FWS
<pre>
FWS                                              Root Directory
├─ Classify
│    ├─ Bert.py                                  bert code for fws type classify
│    ├─ Bilstm.py                                bilstm code for fws type classify
│    ├─ TextCNN.py                               textcnn code for fws type classify
│    ├─ data                                     fws type dataset
│    │    └─ TypeClassify.xlsx
│    ├─ logs.txt                                 record the model training result
│    ├─ main.py                                  receive the command parameter and choose the model to train
│    ├─ run.py                                   use this to reproduct our result
│    └─ weights                                  network weights
│           ├─ bert
│           ├─ bilstm
│           ├─ scibert
│           └─ textcnn
├─ ExtractKeyphrase
│    ├─ ExtractKeyphrases.ipynb                  code for extract keyphrases from fws and abstract in paper
│    ├─ Rejust.ipynb                             rejust the extract result
│    └─ data
│           ├─ keyphrase                         keyphrases file, include:before rejust(xlsx)、after rejust(json)
│           └─ raw                               raw data, include: fws、abstract and title、stopwords and so on
├─ README.md
└─ Recognize
       ├─ data                                   fws and no-fws dataset
       │    └─ recognition.xlsx
       ├─ main.py                                receive the command parameter and choose the model to train
       └─ run.py                                 use this to reproduct our result
</pre>

## Dataset discription

## Quick start
In order to reproduct our experiment result,you can follow these steps:

> recognition 

based on your system,open the terminal and type this command
<pre>python Recognition/run.py </pre>

> classify

based on your system,open the terminal and type this command
<pre>python Classify/run.py</pre>

> extract keywords

we provide two notebooks,you can follow the steps to extract keywords and do some preprocess work

## Citation
Please cite the following paper if you use this codes and dataset in your work
