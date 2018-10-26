# Neural network for genre classification

This is a simple neural classifier which aims at providing text classification using a mixed representation, which uses the most common words and the Part-Of-Speech tags for the less common words.  The idea is that this captures the genre categories without relying too much on keywords specific to the training corpus.  For example, a review text:

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

CNN and bi-LSTM versions have been added, but they didn't help with producing better classifier.  And they are much slower to train.


The arguments for the script are self-explanatory (run `classifier.py -h`).  A typical invocation would be:

`classifier.py -1 en.vec.gz -i en.ol.xz -a en-reduced.csv -d en.brieftag.num --cv_folds 10`

The format for the training file follows FastText: one line per document.  The annotation file is a tab-separated table giving for each training document its probabilities for each label (similarly to the probabilities assigned to a document by the topic models, except that this classifier uses them in a supervised mode).  The most common words and the POS tags for the less frequent ones are coming from a dictionary:

`num word pos`
`109833303 the  DET`
`55324951 and  CCONJ`
` 254488 equipment  NOUN`

This can be obtained, for example, for an available CONLLU file by

`cut -f 2,4 -s CONLLU.file | sort | uniq -c | sort -nsr >CONLLU.num`


