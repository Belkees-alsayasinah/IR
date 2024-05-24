from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from First.TextProcessing import TextProcessor, process_text

# نفسا أفضل نتيجة بس هون عم درب الموديل وأحفظ بس
file_path = "C:/Users/sayas/.ir_datasets/lotte/lotte_extracted/lotte/lifestyle/dev/try.tsv"

with open(file_path, "r", encoding="utf-8") as file:
    data = file.readlines()

tagged_data = [TaggedDocument(words=word_tokenize(process_text(_d.strip(), TextProcessor()).lower()), tags=[str(i)]) for
               i, _d in enumerate(data)]

model = Doc2Vec(vector_size=500, window=2, workers=4, min_count=2, epochs=80)

model.build_vocab(tagged_data)

model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

model.save("d2v.model")
