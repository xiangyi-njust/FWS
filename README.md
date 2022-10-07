# Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain

## Overview
<b> Data and Code for the paper "Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain"

The research object of the paper is future work sentence(FWS), we choose the NLP domain as an example, and use the ACL、EMNLP、NAACL as our origin dataset, our main work includes below parts:
* After human annotation of the future work sentence, we use some traditional machine learning models to judge whether one sentence is FWS or not.
* After that, we classify the FWS in paper into six types, we use bert、scibert、textcnn、bilstm to implement the experiment.
* In addition, We compared the differences between keywords extracted from future work sentences and abstracts to verify whether
the work in future work sentences would be realistic in subsequent real studies.

## Directory structure
FWS
<pre>
FWS                                              Root Directory
├─ Classify
│    ├─ Bert.py                                  bert code for FWS type classify
│    ├─ Bilstm.py                                bilstm code for FWS type classify
│    ├─ TextCNN.py                               textcnn code for FWS type classify
│    ├─ data                                     fws type dataset
│    │    └─ TypeClassify.xlsx
│    ├─ logs.txt                                 record the model training result
│    ├─ main.py                                  receive the command parameter and chooses the model to train
│    ├─ run.py                                   use this to reproduce our result
│    └─ weights                                  network weights
│           ├─ bert
│           ├─ bilstm
│           ├─ scibert
│           └─ textcnn
├─ ExtractKeyphrase
│    ├─ ExtractKeyphrases.ipynb                  code for extracting keyphrases from FWS and abstract in paper
│    ├─ Rejust.ipynb                             rejust the extract result
│    └─ data
│           ├─ keyphrase                         keyphrases file, include:before rejust(xlsx)、after rejust(JSON)
│           └─ raw dataset 						 including fws、abstract and title、stopwords, and so on
├─ README.md
└─ Recognize
       ├─ data                                   FWS and no-FWS dataset
       │    └─ recognition.xlsx
       ├─ main.py                                receive the command parameter and chooses the model to train
       └─ run.py                                 use this to reproduce our result
</pre>

## Dataset discription

## Quick start
To reproduce our experiment result, you can follow these steps:

> recognition 

based on your system, open the terminal and type this command
<pre>python Recognition/run.py </pre>

> classify

based on your system, open the terminal and type this command
<pre>python Classify/run.py</pre>

> extract keywords

we provide two notebooks, you can follow the steps to extract keywords and do some preprocess work

## Citation
Please cite the following paper if you use these codes and datasets in your work
