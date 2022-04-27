from datasets import load_dataset
import sys
import unidecode
from utils import *

def get_dict(data):
    dict_data = {}
    duplicates = {}
    count = 0
    for d in data:
        key = d['summary'] + '-sum-art-' + d['article']
        if key in dict_data:
            count += 1
            key_dup = d['summary'] + '-sum-art-' + d['article']
            duplicates[key_dup] = d
        dict_data[key] = d
    return dict_data, duplicates


file = sys.argv[1]
file_output = sys.argv[2]

with open(file) as file:
    frank = json.load(file)

dataset = load_dataset("cnn_dailymail", '3.0.0')
hash_cnndm = set()

for d in dataset['train']:
    hash_cnndm.add(d['id'])
for d in dataset['test']:
    hash_cnndm.add(d['id'])
for d in dataset['validation']:
    hash_cnndm.add(d['id'])

dataset = load_dataset("xsum")
hash_xsum = set()

for d in dataset['train']:
    hash_xsum.add(d['id'])
for d in dataset['test']:
    hash_xsum.add(d['id'])
for d in dataset['validation']:
    hash_xsum.add(d['id'])


labels_cont_cnndm = []
labels_cont_xsum = []
new_data = []
labels_cont = []

for idx, example in enumerate(frank):
    assert len(example['summary_sentences']) == len(example['summary_sentences_annotations'])

    for s, sa in zip(example['summary_sentences'], example['summary_sentences_annotations']):
        new_example = {}
        new_example['summary'] = unidecode.unidecode(s)
        new_example['article'] = unidecode.unidecode(example['article'])
        new_example['id'] = example['hash'] + "_" + example['model_name'] + "_" + str(len(new_data))
        new_example['id_order'] = len(new_data)
        new_example['model_name'] = example['model_name']
        new_example['split'] = example['split']

        new_example['source'] = 'frank'

        if example['hash'] in hash_cnndm:
            new_example['domain'] = 'cnndm'
        elif example['hash'] in hash_xsum:
            new_example['domain'] = 'xsum'
        else:
            print(d['hash'])
            print('error')
            exit()

        labels = []
        for k, v in sa.items():
            if v[0] == 'NoE':
                new_label = 'CORRECT'
            else:
                new_label = 'INCORRECT'
            labels.append(new_label)

        label = most_common(labels)

        new_example['label'] = label

        new_data.append(new_example)
        labels_cont.append(label)
        if new_example['domain'] == 'cnndm':
            labels_cont_cnndm.append(new_example['label'])
        if new_example['domain'] == 'xsum':
            labels_cont_xsum.append(new_example['label'])


save_data(new_data, file_output)
