import json
from tqdm import tqdm
from amr_utils.amr_readers import AMR_Reader
reader = AMR_Reader()
import collections
from utils import simplify_amr_hal, check_hallucinations, read_csv
import os

def load_source_docs(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def save_data(data, output_file):
    with open(output_file, 'w', encoding="utf-8") as fd:
        for example in data:
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")

def get_amrs_file(files):
    data = []
    data_align = {}
    for f in tqdm(files):
        print(f)
        amrs, alignments = reader.load(f, remove_wiki=True, output_alignments=True)
        data.extend(amrs)
        data_align.update(alignments)
    assert len(data) == len(data_align)
    return data, data_align

from collections import defaultdict, Counter

import sys
file_spans = sys.argv[1]
amr_file = sys.argv[2]
number_graphs = int(sys.argv[3])
hal_file = sys.argv[4]
hal_data = read_csv(hal_file)

hal_dict = {}
for example in hal_data[1:]:
    id_sum = str(example[0]) + "_" + str(example[1])
    if id_sum not in hal_dict:
        hal_dict[id_sum] = []
    hal_dict[id_sum].append(example[4])
hal_data = hal_dict

data = load_source_docs(file_spans)
amr_data, align_data = get_amrs_file([amr_file])

dict_amr_data = {}
for amr in amr_data:
    dict_amr_data[amr.metadata['snt']] = amr

error_log = 0
error_log_claim = 0
error_log_g = 0

count_error_word = 0
count_error_word_map = 0
error_inc = 0
error_c = 0
number_words_edge = []

for example in tqdm(data):
    try:
        sentence = example['summary']
        hallucinated_nodes, hallucinated_word_ids, map_nodes_tokens, hallucinated_words = check_hallucinations(example, dict_amr_data[sentence], align_data, hal_data)
        sent_tok = dict_amr_data[sentence].tokens
        example['summary_tok'] = " ".join(sent_tok)
        if not map_nodes_tokens:
            count_error_word_map += 1

        data_hal_amr = hallucinated_nodes, hallucinated_word_ids, map_nodes_tokens

        graph_simple, triples, hal, words_amr = simplify_amr_hal(dict_amr_data[sentence], data_hal_amr)

        if not words_amr:
            count_error_word += 1

        inc_words = 0
        c_words = 0
        for w in words_amr:
            if w[4] == 0:
                inc_words += 1
            else:
                c_words += 1

        if example['label'] == 'INCORRECT' and inc_words == 0:
            error_inc += 1
            if len(hallucinated_word_ids) > 0:
                words_amr.append((str(hallucinated_word_ids[0]), str(hallucinated_word_ids[0]), "", "", 0, "", "", "", "0", "0"))
            elif len(hallucinated_words) > 0:
                example['summary_tok'] = example['summary_tok'] + ' ' + hallucinated_words[0]
                example['summary'] = example['summary'] + ' ' + hallucinated_words[0]
                idx_last_word = len(example['summary_tok'].split()) - 1
                words_amr.append(
                    (str(idx_last_word), str(idx_last_word), "", "", 0, "", "", "", "0", "0"))
            # else:
            #     import pdb
            #     pdb.set_trace()
        if example['label'] == 'CORRECT' and inc_words > 0:
            error_c += 1

        example['graph_summary'] = {}

        example['hallucination_amr'] = json.dumps(words_amr)
        graph_simple = ' '.join(graph_simple)
        example['graph_summary']['amr_simple'] = graph_simple
        example['graph_summary']['triples'] = json.dumps(triples)

    except Exception as e:
        error_log_claim += 1
        print("skip graph claim", error_log_claim)
        example['graph_summary'] = {}
        example['graph_summary']['amr_simple'] = ''
        example['graph_summary']['triples'] = ''

    amr_graphs = []

    best_sents = {}
    sents = json.loads(example['sentences'])
    for sentence in sents[:number_graphs]:
        best_sents[int(sentence[1])] = sentence[0]

    best_sents_list = []
    best_sents = collections.OrderedDict(sorted(best_sents.items()))
    for k, v in best_sents.items():
        best_sents_list.append(v)

    for sentence in best_sents_list:
        sentence = sentence.strip()
        if sentence not in dict_amr_data:
            error_log += 1
            continue
        try:

            data_hal_amr = [], [], map_nodes_tokens
            graph_simple, triples, _, _ = simplify_amr_hal(dict_amr_data[sentence], data_hal_amr, doc_graph=True)

            graph_dict = {}
            graph_simple = ' '.join(graph_simple)
            graph_dict['amr_simple'] = graph_simple
            graph_dict['triples'] = json.dumps(triples)

            amr_graphs.append(graph_dict)

        except:
            error_log_g += 1
            pass

    example['graphs'] = amr_graphs


#print('count_error_word_map', count_error_word_map)
#print('count_error_word', count_error_word)
print('Number of AMR graphs that are labeled as incorrect but do not have incorrect edges (probably due alignment errors):', error_inc)
#print('correct error', error_c)
#print("skipped graph sents:", error_log_g)
print("skipped documents sentences:", error_log)
print("skipped graphs:", error_log_claim)

name_suffix = 'amr'
file = file_spans.split(".json")[0]
new_file = os.path.splitext(file)[0] + "-" + name_suffix + ".json"
print(new_file)
save_data(data, new_file)
