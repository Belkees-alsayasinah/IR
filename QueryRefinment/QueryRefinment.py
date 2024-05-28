import nltk
from nltk.corpus import wordnet
import json

class Query_Refinement:
    def refine_query(self, query):
        query_tokens = nltk.word_tokenize(query)
        refined_query = set(query_tokens)
        for token in query_tokens:

            synsets = wordnet.synsets(token)
            for synset in synsets:
                for lemma in synset.lemmas():

                    if lemma.name() != token:
                        refined_query.add(lemma.name())

        return " ".join(refined_query)

    def refine_queries_file(self, input_file, output_file):
        with open(input_file, 'r') as f_in:
            queries = [json.loads(line) for line in f_in]

        for query in queries:
            refined_query = self.refine_query(query['query'])
            query['query'] = refined_query


        with open(output_file, 'w') as f_out:
            for query in queries:
                f_out.write(json.dumps(query) + '\n')

refine = Query_Refinement()
refine.refine_queries_file(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl",
                           r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.result.jsonl")
