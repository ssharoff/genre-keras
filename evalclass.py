#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2018  Serge Sharoff
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
# A simple evaluation tool for P@k R@k for each category
# Prediction file:
#__label__A1 0.208093 __label__A11 0.190819
#__label__A1 0.315412 __label__A11 0.305447

#Against the gold standard:
#__label__A1 __label__A11  Сын заучивает Высоцкого наизусть . Казалось бы ,  
#__label__A11  Лежу под одеялом в больничной палате . Мужчина 

import sys
import numpy as np
# from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn import metrics
from collections import defaultdict

def hamming_score(y_true, y_pred):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


ratio=0.7
if len(sys.argv)>3:
    ratio=float(sys.argv[3])
def getfield(field):
    if field[:10]=='__label__A':
        x=field[10:].split()
        try:
            i=x[0]#i=int(x[0])
        except:
            print('error in processing %s' % field, file=sys.stderr)
            i=0
        return(i)
    else:
        return(0)

def readpredict(filename):
#__label__A1 0.315412 __label__A11 0.305447
    y=[]
    for line in open(filename):
        fields = line.strip().split()
        topvl=1
        a=[]
        for i in range(0,len(fields),2):
            fdi=getfield(fields[i])
            if fdi:
                try:
                    value=float(fields[i+1])
                except:
                    print('%s[%i]: %s' % (fields[i], i+1, line),file=sys.stderr)
                    value = 0
                if topvl==1 or value/topvl>ratio:
                    a.append(fdi)
                    topvl=value+1e-9
                else:
                    break
        if len(a)==0: #wasn't able to get any value with certainty
            a.append(0)
        y.append(a)
        print('A'+' A'.join(a))
    return y
def readgold(filename):
#__label__A1	__label__A11	Сын заучивает Высоцкого наизусть . Казалось бы ,
    y=[]
    ccount=defaultdict(int)
    for line in open(filename):
        fields = line.strip().split("\t")
        a=[]
        for field in fields:
            fdi=getfield(field)
            if fdi:
                a.append(fdi)
                ccount[fdi]+=1
        y.append(a)
    return y, ccount

y,ccount=readgold(sys.argv[1])
predictions=readpredict(sys.argv[2])
mlb = MultiLabelBinarizer()
yfull = y + predictions
yfull = mlb.fit_transform(yfull)
n_classes = yfull.shape[1]

y_test = yfull[0:len(y)]
y_predictions = yfull[len(y):]

print("Label ranking average precision: %0.3f" % metrics.label_ranking_average_precision_score(y_test,y_predictions), file=sys.stderr)
print("Accuracy score: %0.3f" % metrics.accuracy_score(y_test,y_predictions), file=sys.stderr)
print("F1 score: %0.3f" % metrics.f1_score(y_test,y_predictions,average='samples'), file=sys.stderr)
print("F0.5 score: %0.3f" % metrics.fbeta_score(y_test,y_predictions,beta=0.5,average='samples'), file=sys.stderr)
print("Hamming loss: %0.3f" % metrics.hamming_loss(y_test, y_predictions), file=sys.stderr)
print("Hamming score: %0.3f" % hamming_score(y_test, y_predictions), file=sys.stderr)

print('Per class:', file=sys.stderr)

precision = dict()
recall = dict()
thresholds = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], thresholds[i] = metrics.precision_recall_curve(y_test[:, i],
                                                         y_predictions[:, i])
    #average_precision[i] = metrics.label_ranking_average_precision_score(y_test[:, i], y_predictions[:, i])
    average_precision[i] = metrics.average_precision_score(y_test[:, i], y_predictions[:, i])
    #average_precision[i] = hamming_score(y_test[:, i], y_predictions[:, i])
    #print(precision[i], file=sys.stderr)
    p1 = metrics.recall_score(y_test[:, i], y_predictions[:, i])
    accuracy = metrics.accuracy_score(y_test[:, i], y_predictions[:, i])
    f1 = metrics.f1_score(y_test[:, i], y_predictions[:, i])
    # f1 = metrics.hamming_loss(y_test[:, i], y_predictions[:, i])
    print("A%s (%i)\tAP: %0.3f\tP1: %0.3f\tF1: %0.3f" % (mlb.classes_[i],ccount[mlb.classes_[i]], p1, average_precision[i], f1), file=sys.stderr)

# precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test.ravel(),
#     y_predictions.ravel())

###############################################################################
# Plot the micro-averaged Precision-Recall curve
# ...............................................
#
plotting=0
if plotting:
    import matplotlib.pyplot as plt

    # plt.figure()
    # plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
    #          where='post')
    # plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
    #                  color='b')

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title(
    #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    #     .format(average_precision["micro"]))

    ###############################################################################
    # Plot Precision-Recall curve for each class and iso-f1 curves
    # .............................................................
    #
    from itertools import cycle
    # setup plot details
    # colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    #https://matplotlib.org/users/colors.html #T10 palette
    colors = cycle('blue orange green red purple brown pink gray olive cyan'.split())

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    # labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    # labels.append('micro-average Precision-recall (area = {0:0.2f})'
    #               ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('A{0} (AveP = {1:0.3f})'
                      ''.format(mlb.classes_[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves for individual genres')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=11), ncol=2)


    plt.savefig('prec.recall.pdf')
    plt.show()
    # plt.savefig('prec.recall.pdf')
