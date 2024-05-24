# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:31:59 2023

@author: Nabeel
"""
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Query_Suggestion:
    def __init__(self, query_file):
        # load the refined queries from the file
        self.queries = []
        with open(query_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                self.queries.append(row[1])

        # create a TfidfVectorizer object and fit it to the refined queries
        self.vectorizer = TfidfVectorizer()
        self.query_vectors = self.vectorizer.fit_transform(self.queries)

    def suggest_similar_queries(self, query, n=5):
        import nltk
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # tokenize the query
        query_tokens = nltk.word_tokenize(query)

        # create a TfidfVectorizer object and fit it to the refined queries
        vectorizer = TfidfVectorizer()
        query_vectors = vectorizer.fit_transform(self.queries)

        # compute the TF-IDF vector for the input query
        query_vector = vectorizer.transform([' '.join(query_tokens)])

        # compute the cosine similarity between the input query and each refined query
        similarities = cosine_similarity(query_vector, query_vectors)

        # find the n most similar queries to the input query
        similar_indices = similarities.argsort()[0][-n-1:-1][::-1]
        similar_queries = [self.queries[idx] for idx in similar_indices]

        # return the list of similar queries
        return similar_queries