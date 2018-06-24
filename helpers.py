import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def readFile_(path): 
    result = open(path, "r").read().split("\n")
    b = [i.decode("utf-8") for i in result] 
    print type(result[0])
    print type(b[0])
    return b
            
             
def eliminateStopwordsDoc(document, stopList):
    """ 
        Eliminate stopwords in a single document
    """
    A = [] 
    for word in document.split():
      if word not in stopList:
        A.append(word)
    return ' '.join(A)

def eliminateStopWordsCorpus(corpus, stopList):
    """ 
       Eliminate stopwords in whole document corpus 
    """
    newCorpus = [] 
    for document in corpus:
      newCorpus.append(eliminateStopwordsDoc(document, stopList))
    return newCorpus 



def get_boolean_tf(docs, vocab):
    matrix = np.zeros((len(docs), len(vocab)), int)  
    for i in range(len(docs)):
        for j in range(len(vocab)): 
            if vocab[j] in docs[i].split():
               matrix[i][j] = 1

    return matrix

def tf_idf_vectorizer(docs, maxdf, mindf, ngram_range_): 
    """ Get tfidf vectors for documents 'docs',  with params max_df and min_df 
        Returns a tuple of tfidf vectors and vocabulary (tfidf, vocab)
        ngram_range_ is a tuple in some range e.q
    """ 
    vectorizer = TfidfVectorizer(max_df=maxdf, min_df=mindf, ngram_range=ngram_range_)
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    # Get the vocabulary 
    vocabulary_dict   = vectorizer.vocabulary_
    vocabulary = [] 
    for (a, _) in vocabulary_dict.items(): 
	 vocabulary.append(a) 
    return (tfidf_matrix, vocabulary)


def train_word2vec(docs, num_hidden_layers, window_, mincount):
    """ Train word2vec model to get word vectors, 
        returns a tupple of vocabulary and word vectors list 
    """
    doc_lst = [] 
    for doc in docs:
       doc_lst.append(doc.split()) 
       
    w2v_model = Word2Vec(doc_lst, size=num_hidden_layers, window=window_, min_count=mincount, workers=4)
    vocabulary_tuple = w2v_model.vocab.items() 
    vocabulary = list()
    for (a, _) in vocabulary_tuple:
       vocabulary.append(a)
        
    word_vectors = [] 
    for word in vocabulary: 
        word_vectors.append(w2v_model[word])
        
    return (word_vectors, vocabulary, w2v_model)
    


def extract_keywords(documents, low, high):
    """
        Keyword extractin based on document frequency values
        Keywords are words with frequency values between low and high  
    """
    (a, keywords) = tf_idf_vectorizer(documents, high, low, (0,1)) 
    print "The number of keywords are ", len(keywords)
    return keywords

    
def enrich_text(documents, w2v_train_set, keywords,  n_related_terms, w2v_trained):
    
    """
       Expanding text with related words in order to explicitly separate it 
       from other documents.
       if text contains a sentence with keyword then it will be expanded 
       by adding related words to that keyword to the sentence 
    """
    
    if w2v_trained == None: 
        (vectors, vocab, model) = train_word2vec(w2v_train_set, 300, 6, 1)
    else: 
        model = w2v_trained 
        
    # Getting related terms to the keywords 
    related_terms = []
    related_terms_dict = {}
    expanded_keywords = keywords 
    
    for item in expanded_keywords:
        a = getSimilarTerms(item, n_related_terms,  model)
        for i in a:
           string = ""
           try: 
               string = string + " " + i
           except UnicodeDecodeError:
               print "cannot decode the bytsstring %s", i
        if string != "":
            related_terms.append(string)
            related_terms_dict[item] = string
    
    documents_split = []
    for doc in documents:
    	documents_split.append(doc.split()) 
    
    # text enrichment  
    for doc in documents_split: 
    	for i in range(len(doc)): 
    	   if doc[i] in expanded_keywords:
    	   	  doc[i] = related_terms_dict[doc[i]]

    final = [] 
    for i in documents_split: 
    	final.append(" ".join(i))
        
    return (final, related_terms)

def getSimilarTerms(term, N,  model):
    a = []  
    try: 
        a = model.most_similar(term, topn=N)
    except: 
        print "The word ", term, " does not exists!"
    result = [term] * N  

    # print a
    K = N 
    for (i, _) in a:
        K = K -1
        result = result + [i] * K 
    return result 

## LEMMATIZATION 
def lemmatize(doc):
    """
    Lemmatization
    """
    lemm = {} 
    with open('lem.txt') as f:
        for line in f:
            a =  re.sub(' +',' ',line).split()
            lemm[a[1]] = a[0]  
    error = 0
    success = 0
    docl = []
    for i in doc.split():
        try:
            success += 1
            docl.append(lemm[i])
        except KeyError:
            error += 1
            docl.append(i)
    return " ".join(docl),success, error