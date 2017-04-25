import pandas as pd
import computation.tfidf as tf

docA = "the cat sat on my face cat act cat"
docB = "the dog sat on my bed"

bowA = docA.split(" ")
bowB = docB.split(" ")


wordSet = set(bowA).union(set(bowB))

wordDictA = dict.fromkeys(wordSet,0)
wordDictB = dict.fromkeys(wordSet,0)

for word in bowA:
    wordDictA[word] += 1

for word in bowB:
    wordDictB[word] += 1

#printing word and occurances.
print(pd.DataFrame([wordDictA, wordDictB]))

#finding TF's for the sentences.
tfbowA=tf.computeTF(wordDictA, bowA)
tfbowB=tf.computeTF(wordDictB, bowB)

#finding IDF's for the sentences.
idfs = tf.computeIDf([wordDictA, wordDictB])

#computing tfidf for A and B
tfidfbowA= tf.computeTFIDF(tfbowA, idfs)
tfidfbowB= tf.computeTFIDF(tfbowB, idfs)

#the computed tfidf's
print(pd.DataFrame([tfidfbowA,tfidfbowB]))
