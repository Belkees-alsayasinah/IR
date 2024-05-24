# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:49:07 2023

@author: Nabeel
"""
import csv
class Query_Refinement:
    def refine_query(self, query):
        import nltk
        from nltk.corpus import wordnet

        # tokenize the query
        query_tokens = nltk.word_tokenize(query)

        # create a set to store the refined query
        refined_query = set(query_tokens)

        # loop through each token and add related words
        for token in query_tokens:
            # find synonyms and related words for the token
            synsets = wordnet.synsets(token)
            for synset in synsets:
                for lemma in synset.lemmas():
                    # add the lemma to the refined query if it's not the same as the original token
                    if lemma.name() != token:
                        refined_query.add(lemma.name())

        # return the refined query as a string
        return " ".join(refined_query)

    def refine_queries_file(self, input_file, output_file):
        # open the input and output files
        with open(input_file, 'r', newline='') as f_in, open(output_file, 'w', newline='') as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)

            # loop through each row in the input file
            for id_left, text_left in reader:
                # apply query refinement to the text_left column
                refined_query = self.refine_query(text_left)
                # write the id_left and refined query to the output file
                writer.writerow([id_left, refined_query])