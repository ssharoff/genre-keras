# Neural network for genre classification

This repository contains two classifiers. One is the older LSTM-based version, which gives the reason for the repository name, see the section below. 

The current approach uses the more modern BERT-based transformers and it has been tested with a number of transformer models, most successfully with XLM-Roberta.  There are two BERT-based classifiers here as well. One is based on fine-tuned HuggingFace transformers, which is the easiest to apply to a new corpus formatted as one document per line:

```
./thub-genres-apply.py sharoff/genres FNAME
```

with minimal dependencies (the extra modules to install are pytorch and transformers).

For your own fine-tuning experiments it is probably easier to use the other version, which is based on simplified transformers.  The usage is:

```
./classifier-bert.py -p en -e 3
```

(use any other parameters `./classifier-bert.py -h`. Empirically, training beyond three epochs is not needed).

For testing this model use

```
./classifier-bert.py -p en --testmodel MNAME -t FNAME
```

where MNAME is the directory with the latest checkpoint and FNAME is a corpus in the one-line-per-document format.

# LSTMs for genre classification

This is an older model based on a end-to-end neural classifier which aims at providing text classification using a mixed representation, which uses the most common words and the Part-Of-Speech tags for the less common words.  The idea is that this captures the genre categories without relying too much on keywords specific to the training corpus.  For example, a review text:

`It won the SCBWI Golden Kite Award for best nonfiction book of 1999 and has sold about 50,000 copies.`

converts into a mixed representation as

`It won the PROPN ADJ NOUN NOUN for best NOUN NOUN of [\#] and has sold about [\#] NOUN. '

The system of categories for training follows:
```
@Article{sharoff18genres,
  author = 	 {Serge Sharoff},
  title = 	 {Functional Text Dimensions for the annotation of {Web} corpora},
  journal = 	 {Corpora},
  volume =       {13},
  number =       {1},
  pages = 	 {65--95},
  year = 	 {2018}
}
```
[http://corpus.leeds.ac.uk/serge/publications/2018-ftd.pdf]


The neural implementation is based on Keras/Tensorflow.  It started from the example code from [https://www.depends-on-the-definition.com/classify-toxic-comments-on-wikipedia]

This implements a FastText like architecture. Later CNN and bi-LSTM versions have been added, but they didn't help with producing a better classifier. However, a bi-LSTM model with attention showed the best results, so this is the default setting now.


The arguments for the script are self-explanatory (run `classifier.py -h`).  A typical invocation would be:

`classifier.py -1 en.vec.gz -i en.ol -a en-reduced.csv -d en.brieftag.num --cv_folds 10`

The format for the training file follows FastText: one line per document.  The annotation file is a tab-separated table giving for each training document its position on each functional dimension.  This is similar to the probabilities assigned to a document by the topic models, except that this classifier uses them in a supervised mode; for historic reasons the values need to range from 0 to 2.

The most common words and the POS tags for the less frequent ones are coming from a dictionary:

```
num word pos
109833303 the  DET
55324951 and  CCONJ
...
254488 equipment  NOUN
```

This can be obtained, for example, from an available CONLLU file by

`cut -f 2,4 -s CONLLU.file | sort | uniq -c | sort -nsr >CONLLU.num`


