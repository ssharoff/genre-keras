#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys
import smallutils as ut

names={'A1' : 'discussion', 'A01' : 'discussion', 'A4' : 'fiction', 'A7' : 'instruction', 'A8' : 'news', 'A9' : 'legal',
       'A11' : 'personal', 'A12' : 'promotion', 'A14' : 'academic', 'A16' : 'info', 'A17' : 'review', 'A13' : 'propaganda', 'A20' : 'apell', 'A22' : 'nontext'}
def getlabel(s):
    i=s.rfind('__')
    return s[i+2:]
u=0
d=0
g=0
f=ut.myopen(sys.argv[1]) if len(sys.argv)>1 else sys.stdin
unkthr=float(sys.argv[2]) if len(sys.argv)>2 else 0.3
doublethr=float(sys.argv[2]) if len(sys.argv)>3 else 0.7
for line in f:
    v=line.rstrip().split()
    if len(v)<5:
        continue
    c1=float(v[1])
    c2=float(v[3])
    if c1<=unkthr:
        u+=1
    else:
        label=names[getlabel(v[0])]
        if c2/c1>doublethr:
            label+='/'+names[getlabel(v[2])]
            d+=1
        else:
            g+=1
        print(label+'\t'+v[-1])
print('Good: %i\tDouble: %i\tUndef: %i' % (g,d,u), file=sys.stderr)

        
