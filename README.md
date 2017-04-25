# NLP
 
Natural Language processing is teaching computer to read human language.

Code in this repo are written in Python which perform some basic tasks such as computing tfidf vectors, 
LSA(not implemented yet). After performing tfidf vectorization the the program plots the vectors using pandas
on the terminal.



# tf-idf
tfidf is the first step to teaching computer to read
tfidf better stratergy to count

TermFrequency or tf is calculated by:-

        tf = 1 + log(number of times a word appears), if the word appear once or more than once.
        tf = 0 , if the word doesn't appears.

Inverse Document Frequency or idf is calculated by:-

        idf = log(number of docs/number of docs containing word)

The tf-idf vector is calculalted by:-

        tfidf=termfrequency*inverse frequency

tfidf score to rank the importance of the word in the document.

The tf here is calculated by using:-

        tf = 1 + log(number of times a word appears)

Rather than the traditional
        tf = no of times the word apprears / total number of words in documents

## Counting the number of occurances of the word
### Plotting them using pandas:-

            act  bed  cat  dog  face  my  on  sat  the
        0    1    0    3    0     1   1   1    1    1
        1    0    1    0    1     0   1   1    1    1

First column is the document number for which we are looking to find word on the top pannel.

## Finding the tf-idf vectors for the same:-

                act       bed       cat       dog      face   my   on  sat  the
        0  0.693147  0.000000  1.454647  0.000000  0.693147  0.0  0.0  0.0  0.0
        1  0.000000  0.693147  0.000000  0.693147  0.000000  0.0  0.0  0.0  0.0

Similar to the first table, First column is the document number for which we are looking to find word 
on the top.


This repo consists of research done on NLP using may resources listed below.

# Tutorials:-

    Stanford NLP Course: https://www.youtube.com/watch?v=nfoudtpBV68
    Chris McCormick: 
          Site: http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/
          GitHub: https://github.com/chrisjmccormick/LSA_Classification