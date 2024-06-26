import csv
import logging

import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from spellchecker import SpellChecker
import inflect
import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import unicodedata
import contractions
from textblob import TextBlob
import spacy

# Load spaCy model for advanced text processing
nlp = spacy.load('en_core_web_sm')
with open(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\common_words.txt", 'r',
          encoding='utf-8') as file:
    words_to_remove = file.read().splitlines()


class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self.inflect_engine = inflect.engine()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def spelling_correction(self, text):
        corrected_words = []
        words = self.tokenizer.tokenize(text)
        for word in words:
            corrected_word = self.spell_checker.correction(word)
            if corrected_word is not None:
                corrected_words.append(corrected_word)
        return ' '.join(corrected_words)

    def clean_text(self, text, words_to_remove):
        words = text.split()
        cleaned_words = [word for word in words if word not in words_to_remove]
        cleaned_text = ' '.join(cleaned_words)
        return cleaned_text

    def number_to_words(self, text):
        words = self.tokenizer.tokenize(text)
        converted_words = []
        for word in words:
            if word.replace('.', '', 1).isdigit():
                converted_words.append(word)
            else:
                if word.isdigit():
                    try:
                        num = int(word)
                        if num <= 999999999999999:
                            converted_word = self.inflect_engine.number_to_words(word)
                            converted_words.append(converted_word)
                        else:
                            converted_words.append("[Number Out of Range]")
                    except inflect.NumOutOfRangeError:
                        converted_words.append("[Number Out of Range]")
                else:
                    converted_words.append(word)
        return ' '.join(converted_words)

    def remove_html_tags(self, text):
        try:

            if '<' in text and '>' in text:
                return BeautifulSoup(text, "html.parser").get_text()
            else:

                return text
        except MarkupResemblesLocatorWarning:

            logging.warning("MarkupResemblesLocatorWarning: The input looks more like a filename than markup.")

            return text

    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKD", text)

    def expand_contractions(self, text):
        return contractions.fix(text)

    def cleaned_text(self, text):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalization_example(self, text):
        return text.lower()

    def stemming_example(self, text):
        words = self.tokenizer.tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def lemmatization_example(self, text):
        words = self.tokenizer.tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def remove_stopwords(self, text):
        words = self.tokenizer.tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_special_characters_and_emojis(self, text):
        return re.sub(r'[^A-Za-z0-9\s]+', '', text)

    def context_aware_spelling_correction(self, text):
        return str(TextBlob(text).correct())

    def replace_synonyms(self, text):
        words = self.tokenizer.tokenize(text)
        synonym_words = [self.get_synonym(word) for word in words]
        return ' '.join(synonym_words)

    def get_synonym(self, word):
        synonyms = nltk.corpus.wordnet.synsets(word)
        if synonyms:
            return synonyms[0].lemmas()[0].name()
        return word

    def handle_negations(self, text):
        words = self.tokenizer.tokenize(text)
        negated_text = []
        negate = False
        for word in words:
            if word.lower() in ['not', "n't"]:
                negate = True
            elif negate:
                negated_text.append(f"NOT_{word}")
                negate = False
            else:
                negated_text.append(word)
        return ' '.join(negated_text)

    def remove_non_english_words(self, text):
        words = self.tokenizer.tokenize(text)
        english_words = [word for word in words if wordnet.synsets(word)]
        return ' '.join(english_words)


def process_text(text, processor):
    if text is None:
        return text
    text = processor.cleaned_text(text)
    text = processor.normalization_example(text)
    text = processor.stemming_example(text)
    text = processor.lemmatization_example(text)
    text = processor.remove_stopwords(text)
    text = processor.number_to_words(text)
    text = processor.remove_punctuation(text)
    text = processor.clean_text(text, words_to_remove)
    text = processor.expand_contractions(text)
    text = processor.normalize_unicode(text)
    text = processor.handle_negations(text)
    text = processor.remove_urls(text)
    return text


def main():
    with open(r'C:\Users\sayas\.ir_datasets\antique\collection.tsv', 'r',
              encoding='utf-8') as input_file, open(
        r'C:\Users\sayas\.ir_datasets\antique\LastResult.tsv', 'w',
        encoding='utf-8') as output_file:
        reader = csv.reader(input_file, delimiter='\t')
        writer = csv.writer(output_file, delimiter='\t')

        text_processor = TextProcessor()

        for row in reader:
            if len(row) >= 2:
                text = row[1]
                processed_text = process_text(text, text_processor)
                writer.writerow([row[0], processed_text])
                print(processed_text + '\n')
            else:
                print("Skipping row with insufficient data:", row)


if __name__ == '__main__':
    main()
# wordnet
