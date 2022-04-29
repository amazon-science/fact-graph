import pandas
import math
import numbers
import json
import sys
from datasets import load_dataset

labels = {0: 'INCORRECT', 1: 'CORRECT'}


def save_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


dataset_hf = load_dataset("xsum")
data_hf = {}
for example in dataset_hf["test"]:
    data_hf[example['id']] = example['document']

def process_csv(csv_file, json_file):
    data = pandas.read_csv(csv_file, delimiter='\t')

    processed_data = []

    for index, row in data.iterrows():
        example = {}
        example['id'] = row['id']
        example['summary'] = row['context']
        example['label'] = labels[row['sentlabel']]

        id_xsum = example['id'].split("_")[0]

        example['article'] = data_hf[id_xsum]

        hals = []
        for i in range(20):
            idx_words = row['dep_idx'+str(i)]

            if isinstance(idx_words, numbers.Number) and math.isnan(idx_words):
                continue

            words = row['dep_words' + str(i)]
            label = int(row['dep_label' + str(i)])
            rel = row['dep'+ str(i)]

            hals.append((idx_words, words, label, rel))

        example['hallucinations'] = json.dumps(hals)
        processed_data.append(example)
        save_data(processed_data, json_file)


folder_edge_level_data = sys.argv[1]
folder_preprocessed_edge_level_data = sys.argv[2]

process_csv(folder_edge_level_data + '/train.tsv', folder_preprocessed_edge_level_data + '/train.json')
process_csv(folder_edge_level_data + '/test.tsv', folder_preprocessed_edge_level_data + '/test.json')


