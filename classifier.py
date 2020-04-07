#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2018  Serge Sharoff
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
# A genre classifier in Keras, an expansion of the idea from
# https://www.depends-on-the-definition.com/classify-toxic-comments-on-wikipedia

import time
starttime=int(time.time())

import sys, re, pickle, random, os
import pandas as pd
import numpy as np
import smallutils as ut

import argparse
import pickle

from keras.preprocessing import sequence
from keras.models import Model, Input, load_model
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, SpatialDropout1D, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dropout, concatenate, InputSpec, CuDNNLSTM

from keras.optimizers import Adam

from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K

from sklearn import metrics
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description="A Keras Model for Genre Classification")
parser.add_argument('--mname', type=str, default='bilstma', help='Model type')
parser.add_argument('--testmodel', type=str, help='Save model file in H5')
parser.add_argument('-p', '--prefix', type=str, default='', help='defaults for embeddings, input files, annotations and dictionaries')
parser.add_argument('-1', '--embeddings', type=str, default='.vec', help='source embeddings')
parser.add_argument('-i', '--inputfile', type=str, default='.ol', help='one-doc-per-line training corpus')
parser.add_argument('-t', '--testfile', type=str, help='one-doc-per-line test only corpus')
parser.add_argument('-a', '--annotations', type=str, default='.csv', help='tab-separated FTDs with the header and text ids')
parser.add_argument('-d', '--dictionary', type=str, default='.brieftag.num', help='frequencies and POS annotations')
parser.add_argument('-f', '--frqlimit', type=int, default=4500, help='how many words left with their forms')
parser.add_argument('-x', '--maxlen', type=int, default=400, help='to shorten docs')
parser.add_argument('-g', '--gensplit', type=int, default=0, help='to generate extra examples if longer than maxlen')
parser.add_argument('-w', '--wordlist', type=str, help='extra words to add to the lexicon for testing')
parser.add_argument('-l', '--loss', type=str, default='binary_crossentropy', help='loss for training from keras')
parser.add_argument('-m', '--metrics', type=str, default='mae', help='metrics from keras')
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-b', '--binary', type=float, default=0)
parser.add_argument( '--batch_size', type=int, default=64)
parser.add_argument( '--dropout', type=float, default=0.2)
parser.add_argument( '--valsplit', type=float, default=0.05)
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-c', '--cv_folds', type=int, default=0)
parser.add_argument('-k', '--topk', type=int, default=2, help='topK predicted labels to output')
parser.add_argument('-v', '--verbosity', type=int, default=1)

outname=re.sub(' ','=','_'.join(sys.argv[min(len(sys.argv),22):]))  # to refrain from adding mantra parameters to the file name
outname=re.sub('/','@',outname)

args = parser.parse_args()
ut.verbosity=args.verbosity
np.random.seed(args.seed)
if args.verbosity>0:
    print('Parameter list: %s' % outname, file=sys.stderr)


if len(args.prefix)>0:
    args.dictionary=args.prefix+args.dictionary
    args.inputfile=args.prefix+args.inputfile
    args.embeddings=args.prefix+args.embeddings

dictlist,frqlist=ut.readfrqdict(args.dictionary,args.frqlimit)

usejson=True if args.inputfile.endswith('.json') else False
with open(args.inputfile) as f:
    X_train=[ut.mixedstr(l,dictlist,frqlist,usejson) for l in f]
y_train = pd.read_csv(args.annotations,header=0,index_col=0,sep='\t')
if args.binary>0:
    binfunc=lambda x : 1 if x>args.binary else 0
    y_train = y_train.applymap(binfunc)
else:
    maxval=y_train.max()
    y_train = y_train / maxval  #[0..2] annotations need to be within the range of the sigmoid function

if args.verbosity>0:
    print('Train data: %d train, %d labels' % (len(X_train), len(y_train)), file=sys.stderr)
    if args.verbosity>1:
        print('Train samples from %s' % args.inputfile, file=sys.stderr)
        for i in random.sample(range(len(X_train)), k=5): # print 5 random docs
            labels=ut.getlabels(y_train.values[i],y_train.columns) # ','.join(['%.1f' % x for x in y_train.values[i]]) 
            print('%d\t%s\t%s' % (i+1, labels, ' '.join(X_train[i][:50])), file=sys.stderr) # i+1 aligns with line numbers
wlist=set([w for doc in X_train for w in doc])
if args.wordlist:
    with open(args.wordlist) as f:
        wl=set(' '.join(f.readlines()).split())
    wlist=wlist.union(wl)
if args.verbosity>0:
    print('Lex size: %d' % len(wlist), file=sys.stderr)

sp,w2i = ut.read_embeddings(args.embeddings,vocab=wlist)
if args.verbosity>0:
    print('Read %d embeddings in %d dimensions' % (sp.shape[0], sp.shape[1]), file=sys.stderr)

x_train=[]
for doc in X_train:
    if args.maxlen>0 and len(doc)>args.maxlen:
        startpos=np.random.random_integers(len(doc)-args.maxlen)
        endpos=startpos+args.maxlen
    else:
        startpos=0
        endpos=len(doc)
    x_train.append([w2i[w] for w in doc[startpos:endpos]])
    
if not sp.shape[0]==len(w2i):
    print('!!ERROR: Old lex size: %d, new %d. Offending words' % (sp.shape[0], len(w2i)), file=sys.stderr)
    for w in w2i:
        if sp.shape[0]<=w2i[w]:
            print(w, file=sys.stderr)
if args.verbosity>0:
    print('Average train sentence length: %d' % np.mean(list(map(len, x_train)), dtype=int), file=sys.stderr)

if args.gensplit>0:
    for i in range(len(x_train)):
        doc = x_train[i]
        newchunks=int(args.gensplit*len(doc)/args.maxlen)-1
        if newchunks>0:
            window=int(len(doc)/newchunks)
            for j in range(min(newchunks,5)):
                newdoc=doc[window*j:args.maxlen]
                x_train.append(newdoc)
                y_train=y_train.append(y_train.iloc[i])
    if args.verbosity>0:
        print('New doc set is %d' % len(x_train), file=sys.stderr)

x_train = sequence.pad_sequences(x_train, maxlen=args.maxlen)
loadtime=int(time.time())
if args.verbosity>0:
    print('Load time: %d sec' % (loadtime-starttime), file=sys.stderr)

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average attention mechanism from:
        Zhou, Peng, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao and Bo Xu.
        “Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.”
        ACL (2016). http://www.aclweb.org/anthology/P16-2034
    How to use:
    see: [BLOGPOST]
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.w]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, h, mask=None):
        h_shape = K.shape(h)
        d_w, T = h_shape[0], h_shape[1]
        
        logits = K.dot(h, self.w)  # w^T h
        logits = K.reshape(logits, (d_w, T))
        alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))  # exp
        
        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            alpha = alpha * mask
        alpha = alpha / K.sum(alpha, axis=1, keepdims=True) # softmax
        r = K.sum(h * K.expand_dims(alpha), axis=1)  # r = h*alpha^T
        h_star = K.tanh(r)  # h^* = tanh(r)
        if self.return_attention:
            return [h_star, alpha]
        return h_star

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

def createmodel(mname='FT'):
    inp = Input(shape=(args.maxlen, ))
    x = Embedding(sp.shape[0], sp.shape[1], weights=[sp], trainable=True)(inp)
    if args.verbosity>0:
        print('Running '+mname, file=sys.stderr)
    if mname=='CNN':
        x = Conv1D(256, 5, activation='sigmoid')(x)
        x = MaxPooling1D(5)(x)
        x = SpatialDropout1D(args.dropout)(x)
        x = Conv1D(128, 5, activation='sigmoid')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(64, 5, activation='sigmoid')(x)
        x = GlobalMaxPooling1D()(x)
    elif mname=='bilstm':
        x = Bidirectional(LSTM(100))(x)
    elif mname=='bilstma': #bilstm with attention, see https://github.com/tsterbak/keras_attention
        dpre = args.dropout
        d = args.dropout
        rd = args.dropout
        x = SpatialDropout1D(dpre)(x)
        x = Bidirectional(LSTM(units=128, return_sequences=True, dropout=d, recurrent_dropout=rd))(x)
        # x = Bidirectional(CuDNNLSTM(units=128, return_sequences = True, name='lstm'))(x)
        x = SpatialDropout1D(d)(x)
        x, attn = AttentionWeightedAverage(return_attention=True)(x)
    else:
        x = GlobalMaxPooling1D()(x) # a simple imitation of fasttext
        x, attn = AttentionWeightedAverage(return_attention=True)(x)
    x = Dropout(args.dropout)(x)
    output = Dense(y_train.shape[1], activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss=args.loss, 
              optimizer=Adam(0.01),
                  metrics=[args.metrics]) #cosine and mae because our output is a vector
    return(model)

if args.cv_folds>0:
    scores_t = []
    predict_t = np.zeros((x_train.shape[0],y_train.shape[1]))
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    for train_index, test_index in kf.split(x_train):
        kfold_X_train = x_train[train_index]
        kfold_X_test = x_train[test_index]
        kfold_y_test = y_train.values[test_index]
        kfold_y_train = y_train.values[train_index]
        K.clear_session()
        model=createmodel(args.mname)

        hist = model.fit(kfold_X_train, kfold_y_train, batch_size=args.batch_size, epochs=args.epochs, validation_split=args.valsplit, verbose=args.verbosity)
        predict_t[test_index] = model.predict(kfold_X_test, batch_size=args.batch_size, verbose=args.verbosity)
        #np.savetxt(outname+'.pred.mat',predict_t)
        if not args.testfile:
            with open(outname+'.pred',"w") as outf:
                for fscores in predict_t:
                    outstr=['__label__%s %.3f' % (y_train.columns[i], fscores[i])  for i in np.argsort(-fscores)[:args.topk]]
                    print('\t'.join(outstr), file=outf)
        score=metrics.pairwise.cosine_similarity(kfold_y_test, predict_t[test_index]).mean()
        scores_t.append(score)
        if args.verbosity>0:
            print('Cosine similarity %.3f' % score)
    if args.verbosity>0:
        print('Total CV cosine score (%d folds) is %.3f (+/- %0.3f)' % (args.cv_folds,np.mean(scores_t), 2*np.std(scores_t)))
elif args.testmodel:
    model = load_model(args.testmodel,custom_objects={'AttentionWeightedAverage':AttentionWeightedAverage})
    w2i = pickle.load(open(args.testmodel+'.map','rb'))
else:
    if args.verbosity>0:
        print('Building a model for the full set, %i ol %i csv' % (len(x_train), len(y_train.values)), file=sys.stderr)
    model=createmodel(args.mname)
    hist = model.fit(x_train, y_train.values, batch_size=args.batch_size, epochs=args.epochs, validation_split=args.valsplit, verbose=args.verbosity)
    # model_filename = '%s_%s.h5' % (args.annotations, args.mname)
    # pickle.dump(w2i, open('%s.map' % model_filename, 'wb'))
    # print('Mappings of words to representations in the model are saved to %s.map' % model_filename, file=sys.stderr)
    # model.save(model_filename)
    # print('My production model saved to ', model_filename, file=sys.stderr)

traintime=int(time.time())

# So far loading from the h5 file does not work
# File "classifier.py", line 277, in <module>
#     predict_t = model.predict(x_test, batch_size=args.batch_size, verbose=args.verbosity)
# File "python-3.6.0/lib/python3.6/site-packages/Keras-2.0.9-py3.6.egg/keras/engine/training.py", line 1730, in predict
# File "python3.6/site-packages/Keras-2.0.9-py3.6.egg/keras/engine/training.py", line 154, in _standardize_input_data
# ValueError: Error when checking : expected input_1 to have shape (None, 500) but got array with shape (1, 400)    

if args.verbosity>0:
    print('Train time: %d sec' % (traintime-loadtime), file=sys.stderr)

if args.testfile:
    if args.verbosity>0:
        print('Predicting on the test set %s into %s' % (args.testfile, outname), file=sys.stderr)
    f=sys.stdin if args.testfile=='-' else ut.myopen(args.testfile)
    outf=open(outname+'.pred',"w")
    for l in f:
        X_testdoc=ut.mixedstr(l,dictlist,frqlist)
        x_testdoc=[]
        for w in X_testdoc:
            if not w in w2i:
                w='<unk>'
            x_testdoc.append(w2i[w])
        x_test = sequence.pad_sequences([x_testdoc], maxlen=args.maxlen)
        predict_t = model.predict(x_test, batch_size=args.batch_size, verbose=args.verbosity)
        for fscores in predict_t:
            outstr=['__label__%s %.3f' % (y_train.columns[i], fscores[i])  for i in np.argsort(-fscores)[:args.topk]]
            print('\t'.join(outstr), file=outf)
#model.save(outname+'.hd5')
#pickle.dump(w2i,open(outname+'w2i.pkl',"wb"))
# x = load_model(args.method+'.hd5')
# y_hat=x.predict(x_test)

outf.close()
testtime=int(time.time())
if args.verbosity>0:
    print('Test time: %d sec' % (testtime-traintime), file=sys.stderr)


# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
