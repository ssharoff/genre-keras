#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2019  Serge Sharoff
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
# The tool

import sys
import smallutils as ut
import pandas as pd

annotations=sys.argv[1]
limit=int(sys.argv[2]) if len(sys.argv)>2 else 1

y_train = pd.read_csv(annotations,header=0,index_col=0,sep='\t')

collected=set()
for i in range(len(y_train)):
    labels=ut.getlabels(y_train.values[i],y_train.columns, limit)
    print(labels)
    collected.add(labels)
print('Train data: %d train, %d labels' % (len(y_train), len(collected)), file=sys.stderr)

