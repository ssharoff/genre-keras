# Neural network for genre classification

This is a simple neural classifier which aims at providing text classification using a mixed representation, which uses the most common words and the Part-Of-Speech tags for the less common words.  The idea is that this captures the genre categories without relying too much on keywords specific to the training corpus.  For example, a review text:

`It won the SCBWI Golden Kite Award for best nonfiction book of 1999 and has sold about 50,000 copies.`

converts into a mixed representation as
It won the PROPN ADJ NOUN NOUN for best NOUN NOUN of [\#] and has sold about [\#] NOUN. 

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


The neural implementation is based on Keras/Tensorflow and started from the example code from [https://www.depends-on-the-definition.com/classify-toxic-comments-on-wikipedia]


