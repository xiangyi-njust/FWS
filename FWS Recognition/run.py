import os

# svm
os.system("python main.py --model svm --ori_path data//recognition.xlsx --tar_path data//tfidf.pkl")
# knn
os.system("python main.py --model knn --tar_path data//tfidf.pkl --flag True")
# naive bayes
os.system("python main.py --model nb --tar_path data//tfidf.pkl --flag True")
# logistic learning
os.system("python main.py --model lr --tar_path data//tfidf.pkl --flag True")
# random forest
os.system("python main.py --model rf --tar_path data//tfidf.pkl --flag True")
