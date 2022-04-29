from utils import *
import sys
from datasets import load_dataset


def consolidate_annotations(data):

    dataset_hf = load_dataset("xsum")

    data_hf = {}
    for example in dataset_hf["validation"]:
        data_hf[example['id']] = example['document']
    for example in dataset_hf["test"]:
        data_hf[example['id']] = example['document']

    dict_data = {}

    for example in data[1:]:
        id_sum = str(example[0]) + "_" + str(example[1])
        summary = example[2]
        hal = example[3]
        if hal == 'extrinsic' or hal == 'intrinsic':
            hal = 'INCORRECT'
        else:
            hal = 'CORRECT'
        if id_sum not in dict_data:
            dict_data[id_sum] = {}
        dict_data[id_sum]['summary'] = summary
        if 'labels' not in dict_data[id_sum]:
            dict_data[id_sum]['labels'] = []

        dict_data[id_sum]['labels'].append(hal)
        label = most_common(dict_data[id_sum]['labels'])
        dict_data[id_sum]['label'] = label
        dict_data[id_sum]['article'] = data_hf[id_sum.split("_")[0]]

    consolidated_data = []
    for idx, k in enumerate(dict_data.keys()):
        del dict_data[k]['labels']
        dict_data[k]['id_order'] = idx
        dict_data[k]['id'] = k
        consolidated_data.append(dict_data[k])

    return consolidated_data


file = sys.argv[1]
output_file = sys.argv[2]
data = read_csv(file)
data_maynez = consolidate_annotations(data)

save_data(data_maynez, output_file)
