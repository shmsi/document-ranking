#!/usr/bin/env python
# -*- coding: utf-8 -*-
from helpers import readFile_
from helpers import enrich_text
from helpers import lemmatize
from helpers import getSimilarTerms
from helpers import train_word2vec
from helpers import extract_keywords

from sklearn.metrics.pairwise import cosine_similarity 
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


# processed data to train word2vec 
file_name = "add file name here" 
processed  = readFile_(file_name)

def tfidf_search(query, documents, num_results):
    """
    tfidf_search retrieves documents based on their cosine similarity to the query. 
    query       - searched word or sentence.  
    documents   - the text to be searched.
    num_results - number of results to be retrieved.  
    """
    vectorizer = TfidfVectorizer(max_df = 1000, min_df=2) 
    X = vectorizer.fit_transform(documents) 

    # Get tfidf vector of test document 
    vect = vectorizer.transform(query)

    # cosine similarity  
    query_vect  = vect[0]
    # dict contains pairwise similarities between query and documents 
    # Key document: value similarity 
    dict = {}
    similarity = cosine_similarity(vect, X)  

    import numpy as np
    S =  similarity[0]

    for i in  range(len(documents)): 
        dict[i] = S[i] 

    # sort dictionary according to values 
    import operator
    sortedD = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

    # return relevant results
    query_result = list() 
    for i in range(num_results):
        res, similariity = sortedD[i]

        # DEBUG : Check the similarity values 
        # print (i, ": ",  similariity,  documents[res])
        query_result.append(documents[res]) 
    
    return query_result 
    
def query_expansion_search(query, documents, num_results, isEnriched, w2vmodel):
    """
    This function applies tfidf search on an enriched dataset. 
    query       - searched word or sentence.  
    documents   - the text to be searched.
    num_results - number of results to be retrieved. 
    isEnriched  - if isEnriched then the documents will not be enriched. 
    w2model     - word embedding model. If None the a function will be called in order to 
                  get word embedding model
    """
    if w2vmodel == None:
       _, vocab, model = train_word2vec(processed, 300, 6, 5)
    else: 
       model = w2vmodel 

    keywords = extract_keywords(documents, 80, 12000) 
    Q = lemmatize(query)[0].split()
    Q_final = []   
    for i in Q:
        Q_final += getSimilarTerms(i, 7, model) 

    refferences = {}

    if isEnriched == False:
       (final, related_terms) = enrich_text(documents, [], keywords, 5, model)

    for i in range(len(final)):
    	refferences[final[i]] = documents[i]

    result = []

    print ("Query is: ", " ".join(Q_final))

    print (type(final))

    for doc in tfidf_search([" ".join(Q_final)], final, num_results): 
    	result.append(refferences[doc])


    return result


""" 
USAGE: 
model =Word2Vec.load('model')
query = "the query"
documents = readFile_(file_name) 
num_results = 200

tfidf_search_result = tfidf_search(query, documents, num_results)
query_expansion_search = query_expansion_search(query, documents, num_results, isEnriched, w2vmodel):
"""