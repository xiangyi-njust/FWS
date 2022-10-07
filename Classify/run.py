import os

# the result will output in logs.txt
# bert
os.system('python main.py --model bert --weight bert-base-uncased --batch_size 64 --path data/TypeClassify.xlsx --model_save_path weights/bert/ --isTrained True')
# scibet
os.system('python main.py --model bert --weight lordtt13/COVID-SciBERT --batch_size 64 --path data/TypeClassify.xlsx --model_save_path weights/scibert/ --isTrained True')
# textcnn
os.system('python main.py --model textcnn --path data/TypeClassify.xlsx --model_save_path weights/textcnn --isTrained True')
# bilstm
os.system('python main.py --model bilstm --path data/TypeClassify.xlsx --model_save_path weights/bilstm --isTrained True')