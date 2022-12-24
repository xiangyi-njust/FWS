# Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain

## Overview
<b> Data and source Cdde for the paper "Automatic Recognition and Classification of Future Work Sentences from Academic Articles in a Specific Domain".</b>

The aim of this paper is automatic recognition and classification of Future Work Sentences (FWS) from academic articles. We choose the NLP domain as an example, and use papers from three main conferences, naemly ACL, EMNLP and NAACL (These conferences can be visited via https://aclanthology.org/), as exprimental dataset. Our work includes the followig aspects:
* After human annotation of the future work sentence, we use some traditional machine learning models including logistic regression (LR),  naive Bayes (NB)   support vector machine (SVM)  and random forest (RF), to judge whether one sentence is FWS or not.
* After that, we classify the FWS in paper into six types including Method, Resources, Evaluation, Application, Problem and Other, via Bert, Scibert, Textcnn and Bilstm models.
* In addition, We compare differences between keywords which are extracted from future work sentences and abstracts in other papers published several years later, to evaluate the effectiveness of FWS.

### Directory structure
<pre>
FWS                                                  Root directory
├─ Dataset                                           Experimental datasets
│    ├─ Corpus For KeyphraseExtraction               Corpus for content analysis of FWS                 
│    │    └─ Title and Abstract.csv                  Corpus for content analysis of FWS，incuding title and absrtract
│    │
     ├─ Corpus_For_FWS_Recognition.csv               Training dataset for FWS recognition 
│    ├─ Corpus_For_FWS_Recognition_Predict.csv       Sample testing dataset for recognition of FWS
│    ├─ Corpus_For_FWS_TypeClassify.csv              Training dataset for FWS classification 
│    └─ Corpus_For_FWS_TypeClassify_Predict.csv      Sample testing dataset for FWS classification 
│   
├─ FWS Classification                                Source code of FWS classification  
│    ├─ Bert.py					     Source code of BERT/SciBERT classification model
│    ├─ Bilstm.py				     Source code of Bi-LSTM model
│    ├─ TextCNN.py				     Source code of TextCNN model
│    ├─ logs.txt				     Log file of model ouput about the classification result
│    ├─ main.py					     Source code for the passing of command line parameters and the selection of models
│    ├─ predict.py				     Source code for trained model's classification result in testing dataset
│    ├─ run.py					     Source code to start training of FWS classification
│    └─ weights					     Model's weight
│           ├─ bilstm
│           └─ textcnn
│
├─ FWS Recognition                                   Source code of FWS recognition 
│    ├─ main.py					     Source code of preprocessing, model constructing, training and testing
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

> Chengzhi Zhang, Wenke Hao, Zhicheng Li, Yuchen Qian, Yuzhuo Wang. Automatic Recognition and Classification of Future Work Sentences from Scientific Literactures in the Domain of Natural Language Processing. *Journal of Informetrics*, 2022. (under review)
