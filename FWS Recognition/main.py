from xmlrpc.client import Boolean
import sklearn.metrics as sm
import pickle
import pandas as pd
import argparse
import inspect
from ast import Interactive
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.feature_selection import VarianceThreshold,SelectKBest,RFE,SelectFromModel,chi2
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
import re
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# user's intention can divide two types:
# 1. use model to extract fws  -- my target is not package a tool for others to use
# 2. review our experiment   -- my first intention

parser = argparse.ArgumentParser()
parser.add_argument('--flag',type=Boolean,default=False)
parser.add_argument('--ori_path',type=str)
parser.add_argument('--tar_path',type=str)
parser.add_argument('--model',type=str)
parser.add_argument('--cv',type=int,default=10)

def remove_punctuation(line):
    line = str(line)
    line = line.strip()
    line = line.replace('-', ' ')
    line = line.replace('\'s', '')
    rule2 = re.compile(u"\\(.*?\\)|{.*?}|\\[.*?]")
    line = rule2.sub('', line)
    rule1 = re.compile(u"[^a-zA-Z\\s+]")
    line = rule1.sub('', line).lower()
    return line

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemm(sentence):
    tokens = word_tokenize(sentence)  # 分词
    tagged_sent = pos_tag(tokens)     # 获取单词词性
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
    return ' '.join(lemmas_sent)

def preprocess(texts):
    re_texts = [remove_punctuation(text) for text in texts]
    le_texts = [lemm(text) for text in re_texts]
    return texts

def getTfidf(ori_path,tar_path):
    # read data
    df = pd.read_excel(ori_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    texts = preprocess(texts)
    # extract the tfidf feature
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=1.0, sublinear_tf=True, analyzer='word',)
    vectorizer.fit(texts)
    texts = vectorizer.transform(texts)

    # dump
    data = (texts, labels)
    fp = open(tar_path, 'wb')
    pickle.dump(data, fp)
    fp.close()

def evaluate(clf,texts,labels,cv):
    scores = cross_validate(clf,texts,labels,cv=cv,n_jobs=-1, \
            scoring=['precision_macro','recall_macro','f1_macro'])

    print("macro result: ")
    print(" precision scores:")
    print(scores['test_precision_macro'])
    print(" mean scores : ",scores['test_precision_macro'].mean())
    print()
    print(" recall_macro scores:")
    print(scores['test_recall_macro'])
    print(" mean scores : ",scores['test_recall_macro'].mean())
    print()
    print(" f1_macro scores:")
    print(scores['test_f1_macro'])
    print(" mean scores : ",scores['test_f1_macro'].mean())
    print()

def svm(texts,labels,cv=10):
    clf = LinearSVC(max_iter=5000, C=15,)
    selector = RFE(clf, n_features_to_select=5200, step=5000).fit(texts, labels)
    texts = selector.transform(texts)
    evaluate(clf,texts,labels,cv)

def nb(texts,labels,cv=10):
    clf = BernoulliNB(alpha=0.0001,)
    texts = SelectFromModel(clf, norm_order=1).fit_transform(texts, labels)
    evaluate(clf,texts,labels,cv)

def rf(texts,labels,cv=10):
    clf = RandomForestClassifier(n_estimators=200)
    selector = RFE(clf, n_features_to_select=5200, step=5000).fit(texts, labels)
    texts = selector.transform(texts)
    evaluate(clf,texts,labels,cv)

def knn(texts,labels,cv=10):
    clf = KNeighborsClassifier(n_neighbors=12)
    selector = VarianceThreshold(0.00001)
    texts = selector.fit_transform(texts, labels)
    texts = SelectKBest(chi2, k=90).fit_transform(texts, labels)
    evaluate(clf,texts,labels,cv)

def lr(texts,labels,cv=10):
    clf = LR(max_iter=1000)
    selector = RFE(clf, n_features_to_select=5200, step=5000).fit(texts, labels)
    texts = selector.transform(texts)
    evaluate(clf,texts,labels,cv)

def help(): 
    print()
    print("*****************************************************************")
    print("*  This python script will help you to understand our experiment")
    print("*  we use five machine learning models to extract fws from paper")
    print("*  follow below's commnd,you can see the model's performance in our dataset")
    print("*  Notice!: when you read the source code you will find that some parameters \
about the classifier has configure,these parameters are getted by our multiple experiments")
    print()
    print("*  command example:")
    print("python extract.py --model svm --cv 10 ")
    print()
    print("*  parameters meaning:")
    print("--model: configure the classifier you want to use")
    print("--cv: configure the n-fold cross-val")
    print()
    print("for any problem,you can contact us with this email:xiangyi@njust.edu.cn")
    print("*****************************************************************")

def main():
    # shell args get 
    args = parser.parse_args()
    shell_args = args._get_kwargs()
    arg_val = {}
    for arg,val in shell_args:
        arg_val[arg] = val
    
    cv = arg_val['cv']
    flag = arg_val['flag']

    # get tfidf
    if arg_val['model'] == None:
        help()
    else:
        if flag == False:
            getTfidf(arg_val['ori_path'],arg_val['tar_path'])
        data_fp = open(arg_val['tar_path'], 'rb')
        texts, labels = pickle.load(data_fp)
        data_fp.close()
    
    # chooose model
    if arg_val['model'] == 'svm':
        svm(texts,labels,cv)
    elif arg_val['model'] == 'random-forest':
        rf(texts,labels,cv)
    elif arg_val['model'] == 'knn':
        knn(texts,labels,cv)
    elif arg_val['model'] == 'logistic-regression':
        lr(texts,labels,cv)
    elif arg_val['model'] == 'naive-bayes':
        nb(texts,labels,cv)
    else:
        exit()

if __name__ == '__main__':
    main()



