import nltk
from nltk.corpus import wordnet
import json

class Query_Refinement:
    def refine_query(self, query):
        # تجزئة الاستعلام إلى كلمات
        query_tokens = nltk.word_tokenize(query)

        # إنشاء مجموعة لتخزين الاستعلام المحسن
        refined_query = set(query_tokens)

        # تكرار على كل كلمة وإضافة الكلمات ذات الصلة
        for token in query_tokens:
            # البحث عن المرادفات والكلمات ذات الصلة بالكلمة
            synsets = wordnet.synsets(token)
            for synset in synsets:
                for lemma in synset.lemmas():
                    # إضافة المرادف إلى الاستعلام المحسن إذا لم يكن هو نفس الكلمة الأصلية
                    if lemma.name() != token:
                        refined_query.add(lemma.name())

        # إعادة الاستعلام المحسن كسلسلة نصية
        return " ".join(refined_query)

    def refine_queries_file(self, input_file, output_file):
        # فتح ملف الإدخال وقراءة البيانات
        with open(input_file, 'r') as f_in:
            queries = [json.loads(line) for line in f_in]

        # تكرار على كل استعلام وتحسينه
        for query in queries:
            refined_query = self.refine_query(query['query'])
            query['query'] = refined_query

        # كتابة الاستعلامات المحسنة إلى ملف الإخراج
        with open(output_file, 'w') as f_out:
            for query in queries:
                f_out.write(json.dumps(query) + '\n')


# مثال على الاستخدام
refine = Query_Refinement()
refine.refine_queries_file(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl",
                           r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.result.jsonl")
