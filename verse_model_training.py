from bibleabbr import *

import numpy as np
import shorttext
import sqlite3
from functools import reduce
from collections import defaultdict


# Loading Scripture
dbconn = sqlite3.connect('ESVCrossway.sqlite')



def retrieve_docs_as_verses(dbconn):
    cursor = dbconn.cursor()
    for book in wholebible_book_iterator():
        for chap in range(1, numchaps[book]+1):
            chap_label = book+'_'+str(chap)
            result = cursor.execute('select verse, scripture from bible where book is "'+book+'" and chapter='+str(chap))
            for texttuple in result:
                verse = texttuple[0]
                scripture = texttuple[1]
                doc_label = chap_label+':'+str(verse)
                yield doc_label, scripture
    cursor.close()
    

classdict = defaultdict(lambda : [])
for doc_label, verse in retrieve_docs_as_verses(dbconn):
    classdict[doc_label] = [verse]
    
    
# Loading word-embedding model
wmodel = shorttext.utils.load_word2vec_model('wordembedD/GoogleNews-vectors-negative300.bin')
ftmodel = shorttext.utils.load_word2vec_model('wordembedD/wiki-news-300d-1M.vec', binary=False)


# CNN Model (word2vec)
kmodel = shorttext.classifiers.CNNWordEmbed(len(classdict.keys()), vecsize=wmodel.vector_size)
classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wmodel)
classifier.train(classdict, kmodel)
classifier.save_compact_model('biblechap_CNN_verse.pkl')

# CNN Model (fasttext)
kmodel = shorttext.classifiers.CNNWordEmbed(len(classdict.keys()), vecsize=ftmodel.vector_size)
classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(ftmodel)
classifier.train(classdict, kmodel)
classifier.save_compact_model('biblechap_CNN_verse_ft.pkl')


# C-LSTM (word2vec)
kmodel = shorttext.classifiers.CLSTMWordEmbed(len(classdict.keys()), vecsize=wmodel.vector_size)
classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wmodel)
classifier.train(classdict, kmodel)
classifier.save_compact_model('biblechap_CLSTM_verse.pkl')

# C-LSTM (fasttext)
kmodel = shorttext.classifiers.CLSTMWordEmbed(len(classdict.keys()), vecsize=ftmodel.vector_size)
classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(ftmodel)
classifier.train(classdict, kmodel)
classifier.save_compact_model('biblechap_CLSTM_verse_ft.pkl')

