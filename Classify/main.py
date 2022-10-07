import argparse
import inspect
parser = argparse.ArgumentParser()
from Bert import*
from Bilstm import*
from TextCNN import*

parser.add_argument('--model',type=str)
parser.add_argument('--path',type=str)
parser.add_argument('--weight',type=str,default='bert-base-uncased')
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--epochs',type=int,default=5)
parser.add_argument('--isTrained',type=bool,default=False)
parser.add_argument('--model_save_path',type=str)

def help():
    print("*******************************")
    print("* Welcome! Thanks you for watching this,the information below will help you to use our tool ")
    print("* we provide four model to classify the fws type in paper,include:bert/scibert,textcnn,bilstm")
    print("* when you need to use these models,you should use below's command")
    print("  1. Bert/SciBert")
    print(" example：")
    print(" python main.py --model bert --epochs 20 --wegiht bert-base-uncased --path data/TypeClassify.xlsx ")
    print("*******************************")

def main():
    args = parser.parse_args()
    shell_args = args._get_kwargs()
    arg_val = {}
    for arg,val in shell_args:
        arg_val[arg] = val
    
    model = None 

    if arg_val['model'] == 'bert':
        model = Bert(arg_val)
    elif arg_val['model'] == 'textcnn':
        model = TextCNN(arg_val)
    elif arg_val['model'] == 'bilstm':
        model = Bilstm(arg_val)
    else:
        help()

    if model != None:
        model.readData()
        model.train()
        model.test()

if __name__=='__main__':
    main()
