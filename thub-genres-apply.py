from transformers import pipeline
import sys

mname=sys.argv[1] # mostly sharoff/genres
testname=sys.argv[2]
testoutname=testname+'.pred'
fout=open(testoutname,"w")

classifier = pipeline("text-classification",model=mname,device=0)
# print(classifier("Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it.", top_k=2))
# print(classifier("The gratitude of every home in our Island, in our Empire, and indeed throughout the world, except in the abodes of the guilty, goes out to the British airmen who, undaunted by odds, unwearied in their constant challenge and mortal danger, are turning the tide of the World War by their prowess and by their devotion. Never in the field of human conflict was so much owed by so many to so few. ", top_k=2))

linelimit = 500 * 7 # a crude approximation for the input limit, but usually Transformers truncate more

labelpref = '__label__'

def detectLabelandScore(p):
    outs= labelpref + p['label'] +' '+ str(round(p['score'],4))
    return outs

with open(testname) as f:
    for l in f:
        pred=classifier(l[:min(len(l),linelimit)], top_k=2, truncation=True)
        outs=detectLabelandScore(pred[0])+'\t'+detectLabelandScore(pred[1])
        print(outs, file=fout)
