import json
from SearchDocunent import search_documents


def precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_at_k = retrieved_docs[:k]
    relevant_count = len(set(retrieved_at_k) & set(relevant_docs))
    return relevant_count / k


def average_precision(relevant_docs, retrieved_docs):
    relevant_set = set(relevant_docs)
    precisions = []
    for k in range(1, len(retrieved_docs) + 1):
        if retrieved_docs[k - 1] in relevant_set:
            precisions.append(precision_at_k(relevant_docs, retrieved_docs, k))
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant_set)


def mean_average_precision(queries_and_answers):
    average_precisions = []
    for qa in queries_and_answers:
        query = qa['query']
        relevant_docs = qa['answer_pids']
        retrieved_docs = search_documents(query)
        ap = average_precision(relevant_docs, retrieved_docs)
        average_precisions.append(ap)
    return sum(average_precisions) / len(average_precisions)


file_paths = [r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.forum.jsonl',r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl']

queries_and_answers = []
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            queries_and_answers.append(data)

map_score = mean_average_precision(queries_and_answers)
print(f"Mean Average Precision (MAP): {map_score}")
