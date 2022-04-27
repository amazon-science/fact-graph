import json
import random
import collections
import unidecode
from utils import *


def load_source_docs_mapping(file):
    data = load_source_docs(file)
    return {example["source"] + "_" + example["id_dataset"]: example for example in data}


def stats_data(data):
    print('size', len(data))
    domain_cont = []
    source_cont = []
    labels_cont = []
    for d in data:
        domain_cont.append(d['domain'])
        source_cont.append(d['source'])
        labels_cont.append(d['label'])

    print('labels', collections.Counter(labels_cont))
    print('domain', collections.Counter(domain_cont))
    print('source', collections.Counter(source_cont))


def remove_duplicates(data):
    dict_data = {}
    duplicates = {}
    count = 0
    for example in data:
        key = example['summary'] + '-sum-art-' + example['article']
        if key in dict_data:
            count += 1
            if key not in duplicates:
                duplicates[key] = [dict_data[key]]
            duplicates[key].append(example)
        dict_data[key] = example
    print(count)
    return dict_data, duplicates

### FactCC
file = 'factcc/processed.json'
data = load_source_docs(file)

consolidated_data = []

for example in data:
    new_example = {}
    new_example['domain'] = 'cnndm'
    new_example['source'] = 'factcc'
    new_example['summary'] = unidecode.unidecode(example['claim'])
    new_example['article'] = unidecode.unidecode(example['text'])
    new_example['id_dataset'] = example['id']
    new_example['id_order'] = example['id_order']
    new_example['id'] = len(consolidated_data)
    new_example['label'] = example['label']

    consolidated_data.append(new_example)

### Xsum Maynez

file = 'maynez/processed.json'
data_xsum = load_source_docs(file)

for example in data_xsum:
    new_example = {}
    new_example['domain'] = 'xsum'
    new_example['source'] = 'maynez'
    new_example['summary'] = unidecode.unidecode(example['summary'])
    new_example['article'] = unidecode.unidecode(example['article'])
    new_example['id_dataset'] = example['id']
    new_example['id'] = len(consolidated_data)
    new_example['label'] = example['label']

    consolidated_data.append(new_example)


### QAGS

file = 'qags/processed.json'
data_qgs = load_source_docs(file)
for example in data_qgs:
    example['id_dataset'] = example['id']
    example['id'] = len(consolidated_data)

    consolidated_data.append(example)

file = 'frank/processed.json'
frank_data = load_source_docs(file)
for example in frank_data:
    example['id_dataset'] = example['id']
    example['source'] = 'frank'
    example['id'] = len(consolidated_data)

    consolidated_data.append(example)

print('size consolidated', len(consolidated_data))
dict_data, dup_data = remove_duplicates(consolidated_data)

# for k in dup_data.keys():
#     for d in dup_data[k]:
#         print(d["source"], d["id_dataset"])
#     print("---")

consolidated_data = []

for k in dict_data.keys():
    consolidated_data.append(dict_data[k])

print('size consolidated no duplicates', len(consolidated_data))
dict_data, dup_data = remove_duplicates(consolidated_data)

print('consolidated data')
stats_data(consolidated_data)


file_dev_ids = 'mappings/dev_ids.json'
data_ids_dev = load_source_docs_mapping(file_dev_ids)
print('len data_ids_dev', len(data_ids_dev))
file_test_ids = 'mappings/test_ids.json'
data_ids_test = load_source_docs_mapping(file_test_ids)
print('len data_ids_test', len(data_ids_test))


train_data = []
dev_data = []
test_data = []

for example in consolidated_data:
    key_example = example["source"] + "_" + example["id_dataset"]

    if key_example in data_ids_dev:
        dev_data.append(example)
    elif key_example in data_ids_test:
        test_data.append(example)
    else:
        train_data.append(example)


folder_save = 'processed_dataset'

print('train data')
stats_data(train_data)
output_file = folder_save + '/train.json'
save_data(train_data, output_file)

print('dev data')
stats_data(dev_data)
output_file = folder_save + '/dev.json'
save_data(dev_data, output_file)

print('test data')
stats_data(test_data)
output_file = folder_save + '/test.json'
save_data(test_data, output_file)
