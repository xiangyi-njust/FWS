import os

# svm
os.system("python main.py --model svm --ori_path ..//Dataset//Corpus_For_FWS_Recognition.xlsx --tar_path data//tfidf.pkl")
# naive bayes
os.system("python main.py --model nb --tar_path data//tfidf.pkl --flag True")
# logistic learning
os.system("python main.py --model lr --tar_path data//tfidf.pkl --flag True")
# random forest
os.system("python main.py --model rf --tar_path data//tfidf.pkl --flag True")
