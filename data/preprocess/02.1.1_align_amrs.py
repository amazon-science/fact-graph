from amrlib.alignments.faa_aligner import FAA_Aligner
import json
import os
from amr_utils.amr_readers import AMR_Reader
reader = AMR_Reader()
import spacy
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def apply_function_parallel(data, transformation, size_chunks, workers):

    data_list = list(chunks(data, size_chunks))
    set_start_method('spawn', force=True)
    final_datapoints = []
    with tqdm(total=len(data_list)) as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, data in enumerate(data_list):
                job = executor.submit(transformation, data)
                futures[job] = idx

            for job in as_completed(futures):
                datapoint = job.result()
                r = futures[job]
                pbar.update(1)
                final_datapoints.extend(datapoint)
                del futures[job]
    return final_datapoints


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

def get_amrs_file(file):
    files = [file]
    data = []
    for f in tqdm(files):
        print(f)
        amrs = reader.load(f, remove_wiki=True)
        data.extend(amrs)
    return data


def align_amr(amr_data):
    aligned_data = []
    for amr in amr_data:
        graph = amr.graph_string()
        sent = amr.metadata['snt']
        # sent_nlp = nlp(sent)
        # tokens = [token.text for token in sent_nlp]
        # tokens = ' '.join(tokens)
        #tokens = tokens.lower()

        tokens = sent
        sents = [tokens]
        graph_strings = [graph]

        if tokens == None:
            print("error token")
            exit()

        if graph == None:
            print("error graph")
            exit()

        try:
            inference = FAA_Aligner(bin_dir='fast_align/build')
            amr_surface_aligns, alignments_strings = inference.align_sents(sents, graph_strings)

            datapoint = {}
            datapoint['tok'] = tokens
            datapoint['snt'] = sent
            datapoint['alignments'] = alignments_strings[0]
            datapoint['surface'] = amr_surface_aligns[0]
            aligned_data.append(datapoint)
        except:
            continue
    return aligned_data


import sys

if __name__ == '__main__':
    amr_file = sys.argv[1]
    print('amr_file: ', amr_file)

    amr_data = get_amrs_file(amr_file)
    dict_amr_data = {}
    total = 0
    count_errors = 0

    amr_aligned_data = apply_function_parallel(amr_data, align_amr, 200, 70)

    file_align = open(amr_file + ".align", 'w')
    print(len(amr_aligned_data))

    for amr in tqdm(amr_aligned_data):
        file_align.write("# ::tok " + amr['tok'] + "\n")
        file_align.write("# ::snt " + amr['snt'] + "\n")
        file_align.write("# ::alignments " + amr['alignments'] + "\n")
        file_align.write(amr['surface'] + "\n\n")

    file_align.close()
    print('number of warnings', count_errors)


# from amr_utils.amr_readers import AMR_Reader
#
# reader = AMR_Reader()
# amrs, alignments = reader


