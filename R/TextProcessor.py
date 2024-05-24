# import csv
# import logging
#
# import nltk
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords
# from spellchecker import SpellChecker
# import inflect
# import re
# from bs4 import BeautifulSoup
# from langdetect import detect, LangDetectException
#
# class TextProcessor:
#     def __init__(self):
#         self.stemmer = PorterStemmer()
#         self.lemmatizer = WordNetLemmatizer()
#         self.spell_checker = SpellChecker()
#         self.inflect_engine = inflect.engine()
#         self.stop_words = set(stopwords.words('english'))
#
#     def detect_language(self, text):
#         try:
#             return detect(text)
#         except LangDetectException:
#             return None
#
#     def spelling_correction(self, text):
#         corrected_words = []
#         words = nltk.word_tokenize(text)
#         for word in words:
#             corrected_word = self.spell_checker.correction(word)
#             if corrected_word:
#                 corrected_words.append(corrected_word)
#         return ' '.join(corrected_words)
#
#     def number_to_words(self, text):
#         words = nltk.word_tokenize(text)
#         converted_words = []
#         for word in words:
#             if word.isdigit():
#                 try:
#                     number = int(word)
#                     if number > 10 ** 10:  # Set a threshold for what you consider "too large"
#                         logging.warning(f"Number too large to convert: {word}")
#                         converted_words.append("[Number Too Large]")
#                     else:
#                         logging.debug(f"Converting number: {word}")
#                         converted_word = self.inflect_engine.number_to_words(word)
#                         converted_words.append(converted_word)
#                 except inflect.NumOutOfRangeError:
#                     converted_words.append("[Number Out of Range]")
#                 except IndexError as e:
#                     logging.error(f"Error converting number: {word} - {str(e)}")
#                     converted_words.append("[Conversion Error]")
#             else:
#                 converted_words.append(word)
#         return ' '.join(converted_words)
#
#     def remove_html_tags(self, text):
#         return BeautifulSoup(text, "html.parser").get_text()
#
#     def remove_urls(self, text):
#         return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#
#     def remove_punctuation(self, text):
#         return re.sub(r'[^\w\s]', '', text)
#
#     def remove_special_characters(self, text):
#         return re.sub(r'[^a-zA-Z0-9\s]', '', text)
#
#     def cleaned_text(self, text):
#         cleaned_text = re.sub(r'\W', ' ', text)
#         cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
#         return cleaned_text.strip()
#
#     def normalize_text(self, text):
#         return text.lower()
#
#     def stemming(self, text):
#         words = nltk.word_tokenize(text)
#         stemmed_words = [self.stemmer.stem(word) for word in words]
#         return ' '.join(stemmed_words)
#
#     def lemmatization(self, text):
#         words = nltk.word_tokenize(text)
#         lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
#         return ' '.join(lemmatized_words)
#
#     def remove_stopwords(self, text):
#         words = nltk.word_tokenize(text)
#         filtered_words = [word for word in words if word.lower() not in self.stop_words]
#         return ' '.join(filtered_words)
#
#     def process(self, text):
#         if not text:
#             return ""
#
#         text = self.remove_urls(text)
#         text = self.remove_special_characters(text)
#         text = self.cleaned_text(text)
#         text = self.normalize_text(text)
#         text = self.spelling_correction(text)
#         text = self.stemming(text)
#         text = self.lemmatization(text)
#         text = self.remove_stopwords(text)
#         text = self.number_to_words(text)
#         return text
#
# def process_text(text, processor):
#     return processor.process(text)
#
# def main():
#     input_file_path = r'C:\Users\sayas\.ir_datasets\antique\new_processed_collection.tsv'
#     output_file_path = r'C:\Users\sayas\.ir_datasets\antique\new_preprocessed_collection.tsv'
#
#     with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
#         reader = csv.reader(input_file, delimiter='\t')
#         writer = csv.writer(output_file, delimiter='\t')
#         text_processor = TextProcessor()
#
#         for row in reader:
#             if len(row) >= 2:
#                 text = row[1]
#                 processed_text = process_text(text, text_processor)
#                 writer.writerow([row[0], processed_text])
#                 logging.debug(processed_text + '\n')
#             else:
#                 logging.warning("Skipping row with insufficient data: %s", row)
#
# if __name__ == '__main__':
#     main()
