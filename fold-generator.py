#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2021  Serge Sharoff
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
# A simple generator of folds for my data

import sys, re, os
import pandas as pd

import argparse

from sklearn.model_selection import KFold

def createfold(fname,f,index): # fname is the output  dir, f=train|test
    with open(f'{fname}/{f}.ol', 'w') as fout:
        for i in index:
            fout.write(x_train[i])
    y_train.iloc[index].to_csv(f'{fname}/{f}.csv', sep='\t')

parser = argparse.ArgumentParser(description="A fold generator")
parser.add_argument('-p', '--prefix', type=str, default='', help='defaults for embeddings, input files, annotations and dictionaries')
parser.add_argument('-i', '--inputfile', type=str, default='.ol', help='one-doc-per-line training corpus')
parser.add_argument('-a', '--annotations', type=str, default='.csv', help='tab-separated FTDs with the header and text ids')
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-c', '--cv_folds', type=int, default=10)
parser.add_argument('-v', '--verbosity', type=int, default=1)

outname=re.sub(' ','=','_'.join(sys.argv)) 
outname=re.sub('/','@',outname)

args = parser.parse_args()
if args.verbosity>0:
    print(f'Parameter list: {outname}', file=sys.stderr)


if len(args.prefix)>0:
    args.inputfile=args.prefix+args.inputfile
    args.annotations=args.prefix+args.annotations

with open(args.inputfile) as f:
    x_train=f.readlines()

y_train = pd.read_csv(args.annotations,header=0,index_col=0,sep='\t')

if args.verbosity>0:
    print(f'Train data: {len(x_train)} train, {len(y_train)} labels', file=sys.stderr)

kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
fnum=1
for train_index, test_index in kf.split(x_train):
    fname='f'+str(fnum)
    os.mkdir(fname)
    createfold(fname,'ftrain',train_index)
    createfold(fname,'ftest',test_index)
    if args.verbosity>1:
        print(f'Fold{fnum}: {len(train_index)} in train, {len(test_index)} in test', file=sys.stderr)
    fnum+=1
