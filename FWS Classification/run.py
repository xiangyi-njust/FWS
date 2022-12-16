import os

# the result will output in logs.txt
# bert
os.system('python main.py --model bert --weight bert-base-uncased --epochs 6 --batch_size 64 --path /../Dataset/Corpus_For_FWS_TypeClassify.csv --model_save_path weights/bert/')
scibet
os.system('python main.py --model bert --weight lordtt13/COVID-SciBERT --epochs 6 --batch_size 64 --path /../Dataset/Corpus_For_FWS_TypeClassify.csv --model_save_path weights/scibert/')
textcnn
os.system('python main.py --model textcnn --path ../Dataset/Corpus_For_FWS_TypeClassify.csv --model_save_path weights/textcnn --isTrained True')
# bilstm
os.system('python main.py --model bilstm --path ../Dataset/Corpus_For_FWS_TypeClassify.csv --model_save_path weights/bilstm --isTrained True')
