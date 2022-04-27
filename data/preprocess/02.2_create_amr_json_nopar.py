import json
from tqdm import tqdm
import glob
from amr_utils.amr_readers import AMR_Reader
reader = AMR_Reader()
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import simplify_amr_nopar


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def apply_transformation_parallel(data, function, size_chunks, workers):

    data_list = list(chunks(data, size_chunks))
    final_datapoints = []
    with tqdm(total=len(data_list)) as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, data in enumerate(data_list):
                job = executor.submit(function, data)
                futures[job] = idx

            for job in as_completed(futures):
                datapoint = job.result()
                r = futures[job]
                pbar.update(1)
                final_datapoints.extend(datapoint)
                del futures[job]
    return final_datapoints


def apply_transformation_parallel_dict(data, function, size_chunks, workers):

    data_list = list(chunks(data, size_chunks))
    final_datapoints = {}
    with tqdm(total=len(data_list)) as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, data in enumerate(data_list):
                job = executor.submit(function, data)
                futures[job] = idx

            for job in as_completed(futures):
                datapoint = job.result()
                r = futures[job]
                pbar.update(1)
                final_datapoints.update(datapoint)
                del futures[job]
    return final_datapoints

def load_source_docs(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def save_data(data, output_file):
    with open(output_file, 'w', encoding="utf-8") as fd:
        for example in data:
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def get_amrs_file(file):
    files = [file]
    data = []
    for f in tqdm(files):
        print(f)
        amrs = reader.load(f, remove_wiki=True)
        data.extend(amrs)
    return data


def load_amrs(files):
    data = {}
    for f in tqdm(files):
        print(f)
        amrs = reader.load(f, remove_wiki=True)
        data[f] = amrs
    return data


def get_amrs(folder):
    files = glob.glob(folder + '/*.output')
    data = apply_transformation_parallel_dict(files, load_amrs, 1, 60)
    return data


import sys
import os
file = sys.argv[1]
amr_file = sys.argv[2]
number_graphs = int(sys.argv[3])

data = load_source_docs(file)

sents = {}
for d in data:
    for sent in d['sentences']:
        sent = " ".join(sent.split())
        sents[sent] = d

amr_data = get_amrs_file(amr_file)
dict_amr_data = {}
total = 0
for amr in amr_data:
    dict_amr_data[amr.metadata['snt']] = amr
    total += 1

# doc_data_ids = amr_file.replace(".sents.amr", ".idx_sents")
# doc_data_sents_ids = amr_file.replace(".amr", "")
#
# map_sent_to_docs = {}
# for idx, (line1, line2) in enumerate(zip(open(doc_data_ids, 'r').readlines(),
#                                          open(doc_data_sents_ids, 'r').readlines())):
#     line = line1.strip().split()
#     sent = line2 = " ".join(line2.split())
#     if sent not in map_sent_to_docs:
#         map_sent_to_docs[sent] = []
#     for r in line:
#         r = r.split('-')
#         doc_id = r[0]
#         idx_sent = int(r[-1])
#         map_sent_to_docs[sent].append((doc_id, idx_sent, sent))
#
# import pdb
# pdb.set_trace()

error_log = 0
error_log_claim = 0
error_log_g = 0
for d in tqdm(data):
    try:
        s = " ".join(d['summary'].split())
        #s = d['summary'].strip()
        graph = dict_amr_data[s].graph_string()

        graph_simple, triples = simplify_amr_nopar(graph)

        d['graph_summary'] = {}
        #d['graph_claim']['amr'] = graph
        graph_simple = ' '.join(graph_simple)
        d['graph_summary']['amr_simple'] = graph_simple
        d['graph_summary']['triples'] = json.dumps(triples)

    except:
        error_log_claim += 1
        print("skip graph claim", error_log_claim)
        d['graph_summary'] = {}
        #d['graph_claim']['amr'] = ''
        d['graph_summary']['amr_simple'] = ''
        d['graph_summary']['triples'] = ''


    amr_graphs = []

    best_sents = {}
    sents = json.loads(d['sentences'])
    for s in sents[:number_graphs]:
        best_sents[int(s[1])] = s[0]

    best_sents_list = []
    best_sents = collections.OrderedDict(sorted(best_sents.items()))
    for k, v in best_sents.items():
        best_sents_list.append(v)

    # import pdb
    # pdb.set_trace()

    #del(d['sentences'])

    for s in best_sents_list:
        s = " ".join(s.split())
        #s = s.strip()
        if s not in dict_amr_data:
            error_log += 1
            print("sent skipped", error_log)
            continue
        try:

            graph = dict_amr_data[s].graph_string()
            graph_simple, triples = simplify_amr_nopar(graph)

            graph_dict = {}
            #graph_dict['amr'] = graph
            graph_simple = ' '.join(graph_simple)
            graph_dict['amr_simple'] = graph_simple
            graph_dict['triples'] = json.dumps(triples)

            amr_graphs.append(graph_dict)

            # if len(amr_graphs) == number_graphs:
            #     break
        except:
            error_log_g += 1
            print("graph sent skipped", error_log_g)
            pass
    d['graphs'] = amr_graphs
    # import pdb
    # pdb.set_trace()



print("skipped graph sents", error_log_g)
print("skipped sents", error_log)
print("skipped graph claim", error_log_claim)

name_suffix = 'amr'
new_file = os.path.splitext(file)[0] + "-" + name_suffix + ".json"
print(new_file)
save_data(data, new_file)




