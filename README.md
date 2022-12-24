# Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain

## Overview
<b> Data and Code for the paper "Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain".</b>

The aim of this paper is automatic recognition and classification of Future Work Sentences (FWS) from academic articles. We choose the NLP domain as an example, and use papers from three main conferences, naemly ACL、EMNLP、NAACL, as exprimental dataset. Our work includes the followig aspects:
* After human annotation of the future work sentence, we use some traditional machine learning models including logistic regression (LR),  naive Bayes (NB)   support vector machine (SVM)  and random forest (RF), to judge whether one sentence is FWS or not.
* After that, we classify the FWS in paper into six types including Method, Resources, Evaluation, Application, Problem and Other, via Bert, Scibert, Textcnn and Bilstm models.
* In addition, We compare differences between keywords which are extracted from future work sentences and abstracts in other papers published several years later, to evaluate the effectiveness of FWS.

### Directory structure
FWS
<pre>
FWS                                                  Root directory
├─ Dataset                                           Experimental datasets
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
├─ FWS Classification                            Source code of Classification of Future Work Sentence
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
├─ FWS Recognition                              Source code of Recognition of Future Work Sentence
│    ├─ main.py
│    └─ run.py
└─ README.md
</pre>

## Dataset discription

<b>We release our all train dataset in *Dataset* directory: </b>

<li><b>Corpus_For_FWS_Recognition.csv</b>: Traning dataset for classification of Future Work Sentence, it contains 9,009 FWS and 55,887 Non-FWS respectively.
<li><b>Corpus_For_FWS_TypeClassify.csv</b>: Traning dataset for Recognition of Future Work Sentence, it contains 9,009 records.
<li>Additionaly, we release sample our test dataset, if you need the whole data, contact us please.

## Quick start
To reproduce our experiment result, you can follow these steps:

> Recognition 

based on your system, open the terminal in the *FWS Recognition* directory and type this command
<pre>python run.py </pre>

> Classify

based on your system, open the terminal in the *FWS Classification* directory and type this command
<pre>python run.py</pre>

> Extract keywords

We provide two notebooks, you can follow the steps to extract keywords and do some preprocess work

## Citation
Please cite the following paper if you use these codes and datasets in your work.

> Chengzhi Zhang, Wenke Hao, Zhicheng Li, Yuchen Qian, Yuzhuo Wang. Automatic Recognition and Classification of Future Work Sentences from Scientific Literactures in the Domain of Natural Language Processing. *Journal of Informetrics*, 2022. (under review)
