import json

# File paths (change as needed)
file_queries = r'C:\Users\sayas\.ir_datasets\antique\test\queries.txt'
file_answers = r'C:\Users\sayas\.ir_datasets\antique\test\qq.txt'
output_file = r'C:\Users\sayas\.ir_datasets\antique\test\Answers.jsonl'

# Read and parse the queries file
queries = {}
with open(file_queries, 'r', encoding='utf-8') as f:
    for line in f:
        qid, query = line.strip().split('\t')
        queries[qid] = query

# Read and parse the answers file
answers = {}
with open(file_answers, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        qid = parts[0]
        answer_pid = parts[2]
        if qid not in answers:
            answers[qid] = []
        answers[qid].append(answer_pid)

# Write to the JSONL output file
with open(output_file, 'w', encoding='utf-8') as f:
    for qid in queries:
        query = queries[qid]
        answer_pids = answers.get(qid, [])
        record = {
            'qid': int(qid),
            'query': query,
            'answer_pids': answer_pids
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"Data has been written to {output_file}")
