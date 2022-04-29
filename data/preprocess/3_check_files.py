import json


def load_source_docs(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data



# file_train = '/home/ubuntu/fact_project/code/test_models/qags/data_v2/created_dataset/train-all-sents-amr-5.json'
# file_dev = '/home/ubuntu/fact_project/code/test_models/qags/data_v2/created_dataset/dev-all-sents-amr-5.json'
# file_test = '/home/ubuntu/fact_project/code/test_models/qags/data_v2/created_dataset/test-all-sents-amr-5.json'

file_train = '/home/ubuntu/fact_project/code/test_models/qags/data/created_dataset/train-sents-amr.json'
file_dev = '/home/ubuntu/fact_project/code/test_models/qags/data/created_dataset/dev-sents-amr.json'
file_test = '/home/ubuntu/fact_project/code/test_models/qags/data/created_dataset/test-sents-amr.json'

# file_train = '/home/ubuntu/fact_project/code/test_models/qags/data_v2/created_dataset/train-5most-sents-amr.json'
# file_dev = '/home/ubuntu/fact_project/code/test_models/qags/data_v2/created_dataset/dev-5most-sents-amr.json'
# file_test = '/home/ubuntu/fact_project/code/test_models/qags/data_v2/created_dataset/test-5most-sents-amr.json'


import collections

def count_size(file):
    sizes = []
    sizes_sent = []
    empty = 0
    for d in file:
        sizes.append(len(d['graphs']))
        sizes_sent.append(len(d['sentences']))

        if not d['graph_claim']['amr_simple']:
            empty += 1

    d = collections.Counter(sizes)
    sizes = collections.OrderedDict(sorted(d.items()))

    d = collections.Counter(sizes_sent)
    sizes_sent = collections.OrderedDict(sorted(d.items()))
    print('sizes graph', sizes)
    print('sizes sent', sizes_sent)
    print('empty', empty)





file_train = load_source_docs(file_train)
count_size(file_train)
file_dev = load_source_docs(file_dev)
count_size(file_dev)
file_test = load_source_docs(file_test)
count_size(file_test)