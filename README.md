# Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain

## Overview
<b> Data and source Cdde for the paper "Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain".</b>

The aim of this paper is automatic recognition and classification of Future Work Sentences (FWS) from academic articles. We choose Natural Language Preocessing (NLP) domain as an example, and use papers from three main conferences, namey ACL, EMNLP and NAACL (These conferences can be visited via https://aclanthology.org/), as exprimental dataset. Our work includes the followig aspects:
* **FWS Recognition**: After human annotation of the future work sentence, we use traditional machine learning models including Logistic Regression (LR), Naïve Bayes (NB), Support Vector Machine (SVM)  and Random Forest (RF), to judge whether one sentence is FWS or not.
* **FWS Classification**: After FWS Recognition, we classify the FWS in paper into six types including Method, Resources, Evaluation, Application, Problem and Other, via Bert, Scibert, Textcnn and Bilstm models.
* **FWS Evaluation**: In addition, we compare difference between keywords which are extracted from the FWS and abstracts in other papers published several years later, to evaluate the effectiveness of FWS.

### Directory structure
<pre>
FWS                                                  Root directory
├─ Dataset                                           Experimental datasets
│    ├─ Corpus For KeyphraseExtraction               Corpus for content analysis of FWS                 
│    │    └─ Title and Abstract.csv                  Corpus for content analysis of FWS，incuding title and absrtract
│    │
│    ├─ Corpus_For_FWS_Recognition.csv               Training dataset for FWS recognition 
│    ├─ Corpus_For_FWS_Recognition_Predict.csv       Sample testing dataset for recognition of FWS
│    ├─ Corpus_For_FWS_TypeClassify.csv              Training dataset for FWS classification 
│    └─ Corpus_For_FWS_TypeClassify_Predict.csv      Sample testing dataset for FWS classification 
│   
├─ FWS Classification                                Module of FWS classification  
│    ├─ Bert.py					     Source code of BERT/SciBERT classification model
│    ├─ Bilstm.py				     Source code of Bi-LSTM model
│    ├─ TextCNN.py				     Source code of TextCNN model
│    ├─ logs.txt				     Log file which records classification performance of classification model
│    ├─ main.py					     Source code for selecting a model to train Corpus_For_FWS_Recognition by command line arguments
│    ├─ predict.py				     Source code for using trained model to predict label of FWS in test dataset
│    ├─ run.py					     Source code to start training process of FWS classification
│    └─ weights					     Model's weight
│           ├─ bilstm                                Weight of Bi-LSTM model
│           └─ textcnn                               Weight of TextCNN model
│
├─ FWS Recognition                                   Module of FWS recognition 
│    ├─ main.py					     Source code of data preprocessing, training and testing of FWS recognition model
│    └─ run.py					     Source code to start training of FWS recognition
│
└─ README.md
</pre>

## Dataset discription

We release our all train dataset in *Dataset* directory: 

<li><b>Corpus_For_FWS_Recognition.csv</b>: Traning dataset for classification of Future Work Sentence, it contains 9, 009 FWS and 55, 887 Non-FWS respectively.
<li><b>Corpus_For_FWS_TypeClassify.csv</b>: Traning dataset for Recognition of Future Work Sentence, it contains 9, 009 records.

<b>Each line of Corpus_For_FWS_Recognition includes: </b>
<li>id: Paper ID in ACL Anthology.    
<li>year: Year of publication
<li>text: Content of FWS or Non-FWS.
<li>label: 1: FWS and 0: Non-FWS.
<li>chapter: Type of chapter headings.
	
<b>Each line of Corpus_For_FWS_TypeClassify.csv includes: </b>
<li>id: Paper ID in ACL Anthology.
<li>lable: Six types of FWS including method, resources, evaluation, application, problem and other.
<li>text: Content of FWS. 	
		
Additionaly, we release sample our test dataset, if you need the whole data, contact us please.

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

> Chengzhi Zhang, Wenke Hao, Zhicheng Li, Yuchen Qian, Yuzhuo Wang. Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain. ***Journal of Informetrics***, 2022. [[doi]()]  [[arXiv]()]  [[Dataset & Source Code]](https://github.com/xiangyi-njust/FWS/)
