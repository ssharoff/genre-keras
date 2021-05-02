#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) 2021  Serge Sharoff
# A multi-label genre classifier for the Huggingface transformers module

import time
starttime=int(time.time())

import sys, re, os
import logging

import argparse

import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn import metrics

import simpletransformers

from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs, ClassificationModel, ClassificationArgs

)

def printpredict(raw_outputs, outf):
    for fscores in raw_outputs:
        outstr=['__label__%s %.3f' % (y_train.columns[i], fscores[i])  for i in np.argsort(-fscores)[:args.topk]]
        print('\t'.join(outstr), file=outf)
    outf.flush()
    os.fsync(outf.fileno())  # on our HPC the killed jobs don't flush to disk

parser = argparse.ArgumentParser(description="A Transformer Model for Genre Classification")
parser.add_argument('--cname', type=str, default='xlmroberta', help='Model type according to simple transformers')
parser.add_argument('--mname', type=str, default='xlm-roberta-base', help='Model name according to HuggingFace transformers')
parser.add_argument('--tname', type=str, default='', help='Tokenizer')
parser.add_argument('--testmodel', type=str, help='Saved model')
parser.add_argument('-p', '--prefix', type=str, default='', help='defaults for input files, annotations and dictionaries')
parser.add_argument('-i', '--inputfile', type=str, default='.ol', help='one-doc-per-line training corpus')
parser.add_argument('-a', '--annotations', type=str, default='.csv', help='tab-separated FTDs with the header and text ids')
parser.add_argument('--binary', type=float, default=0, help='to treat annotations as binary above a threshold')
parser.add_argument('-t', '--testfile', type=str, help='one-doc-per-line test only corpus')

parser.add_argument('-x', '--maxlen', type=int, default=400, help='to shorten docs')
parser.add_argument('-g', '--gensplit', type=int, default=0, help='to generate extra examples if longer than maxlen')

parser.add_argument('-l', '--loss', type=str, default='binary_crossentropy', help='loss for training')
parser.add_argument('-m', '--metrics', type=str, default='mae', help='metrics for validation')
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('-e', '--epochs', type=int, default=6)
parser.add_argument( '--batch_size', type=int, default=64)
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

# device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
# if args.verbosity>0:
#     print(f'Using device {device}', file=sys.stderr)

maxchars=args.maxlen*6 # an approximation for the max line length in chars

if len(args.prefix)>0:
    args.inputfile=args.prefix+args.inputfile
    args.annotations=args.prefix+args.annotations
y_train = pd.read_csv(args.annotations,header=0,index_col=0,sep='\t')

outf=open(outname+'.pred',"w")
   
if args.testmodel:
    if y_train.shape[1]>1:
        model_args = MultiLabelClassificationArgs(
            max_seq_length=args.maxlen,
            process_count = 11)
        model = MultiLabelClassificationModel(args.cname, args.testmodel, args = model_args, use_cuda = args.gpu)
    else:
        model_args = ClassificationArgs(
            max_seq_length=args.maxlen,
            process_count = 11)
        model = ClassificationModel(args.cname, args.testmodel, args = model_args, use_cuda = args.gpu)
        
    if args.verbosity>0:
        print(model.config, file=sys.stderr)
else:
    with open(args.inputfile,'r') as f:
        docs=f.readlines()
    if args.binary>0:
        binfunc=lambda x : 1 if x>args.binary else 0
        y_train = y_train.applymap(binfunc)
    else:
        maxval=y_train.max()
        y_train = y_train / maxval  #[0..2] need to be within the softmax output range
    train_df=pd.DataFrame([docs, y_train.values], index=['text','labels']).T
    trainX, evalX = train_test_split(train_df, test_size=0.1)
    if args.verbosity>0:
        print(f'Train data: {y_train.shape[0]} documents, {y_train.shape[1]} labels', file=sys.stderr)
        print(y_train.columns, file=sys.stderr)
        print(evalX.head()[['labels']], file=sys.stderr)
        print(evalX.head()[['text']], file=sys.stderr)

    if y_train.shape[1]>1:
        model_args = MultiLabelClassificationArgs(
            num_train_epochs=args.epochs,
            max_seq_length=args.maxlen,
            evaluate_during_training=True,
            save_eval_checkpoints=False,
            save_steps=-1,
            use_early_stopping=True,
            early_stopping_delta=0.01,
            reprocess_input_data=False,
            evaluate_during_training_verbose=args.verbosity,
            output_dir=outname,
            manual_seed = args.seed,
            # cache_dir
            # best_model_dir
            process_count = 10)

        model = MultiLabelClassificationModel(args.cname, args.mname, num_labels=y_train.shape[1], use_cuda = args.gpu, args=model_args)
    else:
        model_args = ClassificationArgs(
            num_train_epochs=args.epochs,
            max_seq_length=args.maxlen,
            evaluate_during_training=True,
            save_eval_checkpoints=False,
            save_steps=-1,
            use_early_stopping=True,
            early_stopping_delta=0.01,
            reprocess_input_data=False,
            evaluate_during_training_verbose=args.verbosity,
            output_dir=outname,
            manual_seed = args.seed,
            # cache_dir
            # best_model_dir
            process_count = 10)

        model = ClassificationModel(args.cname, args.mname, num_labels=y_train.shape[1], use_cuda = args.gpu, args=model_args)
        
    model.train_model(trainX,eval_df=evalX)
    print(evalX.head()['text'], file=sys.stderr)
    predictions, raw_outputs = model.predict(evalX['text'].tolist())
    printpredict(raw_outputs, outf)

beforetesting=int(time.time())
if args.verbosity>0:
    print(f'Training: {beforetesting-starttime} secs', file=sys.stderr)

if args.testfile:
    batchforpred = []
    for line in open(args.testfile):
        line=line[:maxchars]
        batchforpred.append(line)
        if len(batchforpred)>=args.batch_prediction:
            predictions, raw_outputs = model.predict(batchforpred)
            printpredict(raw_outputs, outf)
            batchforpred = []
    if len(batchforpred)>0: # remaining
        predictions, raw_outputs = model.predict(batchforpred)
        printpredict(raw_outputs, outf)
testingtime=int(time.time())

if args.verbosity>0:
    print(f'Testing: {testingtime-beforetesting} secs', file=sys.stderr)
