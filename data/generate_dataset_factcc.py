import json
from tqdm import tqdm
from datasets import load_dataset
import sys

folder_dataset = sys.argv[1]

file_val = folder_dataset + '/val/data-dev.jsonl'
file_test = folder_dataset + '/test/data-dev.jsonl'
file_processed = sys.argv[2]

dataset_hf = load_dataset("cnn_dailymail", "3.0.0")

data_hf = {}
for d in dataset_hf["validation"]:
    data_hf[d['id']] = d['article']
for d in dataset_hf["test"]:
    data_hf[d['id']] = d['article']


with open(file_val) as fd:
    dataset_val = [json.loads(line) for line in fd]

with open(file_test) as fd:
    dataset_test = [json.loads(line) for line in fd]

dataset = dataset_val + dataset_test

for idx, example in tqdm(enumerate(dataset)):
    try:
        example['text'] = data_hf[example['id'].split("/")[-1]]
    except:
        example['text'] = data_hf[example['id'].split("-")[-1]]
    example['id_order'] = idx
    example['id'] = example['id'] + "_" + str(idx)

with open(file_processed, 'w') as fd:
    for example in dataset:
        fd.write(json.dumps(example, ensure_ascii=False) + "\n")

