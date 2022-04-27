import json
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_source_docs(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def save_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            fd.write(example + "\n")


import sys
import os
file = sys.argv[1]

cnndm = load_source_docs(file)

summaries = []
docs = {}
negative = []
ids_neg = []
ids_pos = []
claims = []
idx_claims = []
for idx, d in enumerate(tqdm(cnndm)):
    for idx_sent, sent in enumerate(json.loads(d['sentences'])):
        sent = sent[0]
        if sent not in docs:
            docs[sent] = []
        docs[sent].append((d['id'], idx_sent))

    sent = d['summary']
    if sent not in docs:
        docs[sent] = []
    docs[sent].append((d['id'], idx_sent))


sents = []
ids_docs = []
for sent in docs.keys():
    sents.append(sent)
    id_line = set()
    for id_, idx in docs[sent]:
        id_line.add(str(id_)+'-'+str(idx))
    id_line = ' '.join(id_line)
    ids_docs.append(id_line)

assert len(sents) == len(ids_docs)

new_file = os.path.splitext(file)[0] + ".txt"
save_data(sents, new_file)

new_file = os.path.splitext(file)[0] + "-" + "idx_sents.txt"
save_data(ids_docs, new_file)
