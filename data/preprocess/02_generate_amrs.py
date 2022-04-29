import amrlib
import json
import sys
import os
import numpy as np
from tqdm import tqdm
def load_source_docs(file_path, to_dict=False):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if to_dict:
        data = {example["id"]: example for example in data}
    return data


def save_data(data, file, name_suffix):
    output_file = os.path.splitext(file)[0] + "-" + name_suffix + ".json"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


stog = amrlib.load_stog_model(model_dir='model_parse_xfm_bart_large-v0_1_0')

input_file = sys.argv[1]
amr_file = sys.argv[2]

input_data = load_source_docs(input_file)

sentences = []
for example in input_data:
    sentences.extend([sents[0] for sents in json.loads(example['sentences'])])
    sentences.append(example['summary'])


sentences = list(set(sentences))
print("Total of sentences:", len(sentences))
sentences = [list(sents) for sents in chunks(sentences, 20)]
amr_file = open(amr_file, 'w')
for sents in tqdm(sentences):
    try:
        graphs = stog.parse_sents(sents, add_metadata=True)

        for g in graphs:
            amr_file.write(g + "\n\n")
    except:
        print("Error during parsing.")

