# -*- coding: utf-8 -*-
import time
starttime=int(time.time())

import transformers
import datasets
from datasets  import Dataset
# from datasets import load_dataset

print(f"Running on transformers v{transformers.__version__} and datasets v{datasets.__version__}")

import argparse, re, sys, logging
import pandas as pd
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer, EvalPrediction)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# From https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb
import json
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(labels)
    print(preds)
    json.dump(labels,open('labels.json',"w"))
    json.dump(preds,open('preds.json',"w"))
    f1 = f1_score(labels, preds, average="samples")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "eval_f1": f1}


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

parser = argparse.ArgumentParser(description="A Transformer Model for Genre Classification")
parser.add_argument('--mname', type=str, default='xlm-roberta-base', help='Model name according to HuggingFace transformers')
parser.add_argument('--cname', type=str, default=None, help='Model name to save to the hub')
parser.add_argument('--tname', type=str, default='', help='Tokenizer')
parser.add_argument('--testmodel', type=str, help='Saved model')
parser.add_argument('-p', '--prefix', type=str, default='', help='defaults for input files, annotations and dictionaries')
parser.add_argument('-i', '--inputfile', type=str, default='.ol', help='one-doc-per-line training corpus')
parser.add_argument('-a', '--annotations', type=str, default='.csv', help='tab-separated FTDs with the header and text ids')
parser.add_argument('--binary', type=float, default=0, help='to treat annotations as binary above a threshold')
parser.add_argument('-t', '--testfile', type=str, help='one-doc-per-line test only corpus')

parser.add_argument('-x', '--maxlen', type=int, default=400, help='to shorten docs')
# parser.add_argument('-g', '--gensplit', type=int, default=0, help='to generate extra examples if longer than maxlen')

parser.add_argument('-l', '--loss', type=str, default='binary_crossentropy', help='loss for training')
parser.add_argument('-m', '--metrics', type=str, default='f1', help='metrics for validation')
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('-e', '--epochs', type=int, default=6)
parser.add_argument( '--batch_size', type=int, default=32)
parser.add_argument('-b',  '--batch_prediction', type=int, default=2000)

parser.add_argument( '--valsplit', type=float, default=0.05)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('-c', '--cv_folds', type=int, default=0)
parser.add_argument('-k', '--topk', type=int, default=2, help='topK predicted labels to output')
parser.add_argument('-v', '--verbosity', type=int, default=1)

outname=re.sub(' ','=','_'.join(sys.argv[min(len(sys.argv),2):]))  
outname=re.sub('/','@',outname)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('genres')
logger.setLevel(logging.WARNING)

if args.verbosity>0:
    print(f'Parameter list: {outname}', file=sys.stderr)

maxchars=args.maxlen*6 # an approximation for the max line length in chars

"""## Load data"""
if len(args.prefix)>0:
    args.inputfile=args.prefix+args.inputfile
    args.annotations=args.prefix+args.annotations
y_train = pd.read_csv(args.annotations,header=0,index_col=0,sep='\t')

outf=open(outname+'.pred',"w")

with open(args.inputfile,'r') as f:
    docs=f.readlines()
if args.binary>0:
    binfunc=lambda x : 1 if x>args.binary else 0
    y_train = y_train.applymap(binfunc)
else:
    maxval=y_train.max()
    y_train = y_train / maxval  #[0..2] need to be within the softmax output range
train_df=pd.DataFrame([docs, y_train.values], index=["text","labels"]).T
id2label={i:x for i, x in enumerate(list(y_train.columns.values))}
trainX, evalX = train_test_split(train_df, test_size=0.05)

num_labels = y_train.shape[1]

if args.verbosity>0:
    print(f'Train data: {y_train.shape[0]} documents, {num_labels} labels', file=sys.stderr)
    print(y_train.columns, file=sys.stderr)
    print(evalX.head()[['labels']], file=sys.stderr)
    print(evalX.head()[['text']], file=sys.stderr)
    print(id2label)

tokenizer = AutoTokenizer.from_pretrained(args.mname, problem_type="multi_label_classification")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
trainX_encoded = Dataset.from_pandas(trainX).map(tokenize, batched=True, batch_size=None)
trainX_encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

evalX_encoded = Dataset.from_pandas(evalX).map(tokenize, batched=True, batch_size=None)
evalX_encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


model = AutoModelForSequenceClassification.from_pretrained(
    args.mname, 
    problem_type="multi_label_classification",
    num_labels=num_labels,
    id2label=id2label
)

training_arguments = TrainingArguments(
    args.mname,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model=None, # compute_metrics,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    compute_metrics=None,
    train_dataset=trainX_encoded,
    eval_dataset=evalX_encoded,
)

beforetraining =int(time.time())
if args.verbosity>0:
    print(f'Loading: {beforetraining-starttime} secs', file=sys.stderr)

trainer.train()

aftertraining =int(time.time())
if args.verbosity>0:
    print(f'Training: {aftertraining-beforetraining} secs', file=sys.stderr)

modeldir = './trainer-' + args.tname
model.save_pretrained(modeldir)
tokenizer.save_pretrained(modeldir)

model.push_to_hub(args.tname)
tokenizer.push_to_hub(args.tname)

saving =int(time.time())
if args.verbosity>0:
    print(f'Saving to the hub: {saving-aftertraining} secs', file=sys.stderr)

results = trainer.evaluate()
print(results)
