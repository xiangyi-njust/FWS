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
FWS
├─ Dataset
│    ├─ Corpus For KeyphraseExtraction
│    │    ├─ Future work sentence.csv
│    │    ├─ Stopwords.csv
│    │    ├─ Title and Abstract.csv
│    │    └─ replace.txt
│    ├─ Corpus_For_FWS_Recognition.csv
│    ├─ Corpus_For_FWS_Recognition_Predict.csv
│    ├─ Corpus_For_FWS_TypeClassify.csv
│    ├─ Corpus_For_FWS_TypeClassify_Predict.csv
│    └─ Keyphrases
│           ├─ abs_keywords.xlsx
│           ├─ abs_keywords_rejusted.json
│           ├─ fws_keywords.xlsx
│           └─ fws_keywords_rejusted.json
├─ FWS Classification
│    ├─ Bert.py
│    ├─ Bilstm.py
│    ├─ TextCNN.py
│    ├─ logs.txt
│    ├─ main.py
│    ├─ predict.py
│    ├─ run.py
│    └─ weights
│           ├─ bilstm
│           └─ textcnn
├─ FWS Recognition
│    ├─ main.py
│    └─ run.py
└─ README.md
</pre>

## Dataset discription

## Quick start
To reproduce our experiment result, you can follow these steps:

> recognition 

based on your system, open the terminal in the specified directory and type this command
<pre>python run.py </pre>

> classify

based on your system, open the terminal in the specified directory and type this command
<pre>python run.py</pre>

> extract keywords

we provide two notebooks, you can follow the steps to extract keywords and do some preprocess work

## Citation
Please cite the following paper if you use these codes and datasets in your work.

> Chengzhi Zhang, Wenke Hao, Zhicheng Li, Yuchen Qian, Yuzhuo Wang. Automatic Recognition and Classification of Future Work Sentences from Scientific Literactures in the Domain of Natural Language Processing. *Journal of Informetrics*, 2022. (under review)
