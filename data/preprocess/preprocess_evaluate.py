import json
import os
from tqdm import tqdm
import augmentation_ops as ops
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method
import sys
import amrlib
from tempfile import NamedTemporaryFile
from amr_utils.amr_readers import AMR_Reader
reader = AMR_Reader()
from amrlib.alignments.faa_aligner import FAA_Aligner
import spacy
nlp = spacy.load("en_core_web_sm")
from utils import simplify_amr_hal, map_nodes
import collections
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def apply_transformation_parallel(data, operation, transformation, num_sents, size_chunks, workers):

    data_list = list(chunks(data, size_chunks))
    set_start_method('spawn', force=True)
    final_datapoints = []
    with tqdm(total=len(data_list)) as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, data in enumerate(data_list):
                job = executor.submit(transformation, data, operation, num_sents, idx)
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


def save_data(data, file):
    output_file = file + ".processed"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def apply_transformation(data, operation, num_sents, idx_process):
    for idx, example in enumerate(data):
        try:
            new_example = operation.transform(example, number_sents=num_sents)
            if new_example:
                data[idx] = new_example
        except Exception as e:
            print("Caught exception:", e)
    return data


def extract_sents(data, num_sents):
    #sent_op = ops.SelectSentencesScore(model_type='multi-qa-MiniLM-L6-cos-v1')
    sent_op = ops.SelectSentencesScore()
    #data = apply_transformation(data, sent_op, num_sents, 0)
    data = apply_transformation_parallel(data, sent_op, apply_transformation, num_sents, 1000, 5)
    return data


def align_amr(amr):
    graph = amr.graph_string()
    sent = amr.metadata['snt']

    sent_nlp = nlp(sent)
    tokens = [token.text for token in sent_nlp]
    tokens = ' '.join(tokens)
    tokens = tokens.lower()

    sents = [tokens]
    graph_strings = [graph]

    try:
        inference = FAA_Aligner(bin_dir=os.path.dirname(__file__) + '/fast_align/build')
        amr_surface_aligns, alignments_strings = inference.align_sents(sents, graph_strings)
        amr.metadata['tok'] = tokens
        amr.metadata['alignments'] = alignments_strings[0]
    except:
        print("error during AMR alignment.")

    return amr


def extract_amrs(data):

    stog = amrlib.load_stog_model(model_dir=os.path.dirname(__file__) + '/model_parse_xfm_bart_large-v0_1_0')

    sentences_documents = []
    sentences_summaries = []
    for example in data:
        sentences_documents.extend([sents[0] for sents in json.loads(example['sentences'])])
        sentences_summaries.append(example['summary'])

    sentences_documents = list(set(sentences_documents))
    sentences_summaries = list(set(sentences_summaries))
    #print("Total of sentences:", len(sentences))
    sentences_documents = [list(sents) for sents in chunks(sentences_documents, 20)]
    sentences_summaries = [list(sents) for sents in chunks(sentences_summaries, 20)]

    data_graphs = []
    data_graphs_summaries = []

    for sents in tqdm(sentences_documents):
        try:
            graphs = stog.parse_sents(sents, add_metadata=True)
            for g in graphs:
                file_graph = NamedTemporaryFile(mode='w+', delete=False)
                file_graph.write(g)
                file_graph.close()
                amrs = reader.load(file_graph.name, remove_wiki=True)
                amr = amrs[0]
                data_graphs.append(amr)
        except:
            print("Error during parsing.")

    for sents in tqdm(sentences_summaries):
        try:
            graphs = stog.parse_sents(sents, add_metadata=True)
            for g in graphs:
                file_graph = NamedTemporaryFile(mode='w+', delete=False)
                file_graph.write(g)
                file_graph.close()
                amrs = reader.load(file_graph.name, remove_wiki=True)
                amr = amrs[0]
                amr = align_amr(amr)
                data_graphs_summaries.append(amr)
        except:
            print("Error during parsing.")



    return data_graphs, data_graphs_summaries


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

def process_amr(data, amr_data, amr_data_summaries, number_graphs):

    dict_amr_data = {}
    amr_data = amr_data
    for amr in amr_data:
        dict_amr_data[amr.metadata['snt']] = amr

    file_graph = NamedTemporaryFile(mode='w+', delete=False)
    for amr in amr_data_summaries:
        file_graph.write("# ::tok " + amr.metadata['tok'] + "\n")
        file_graph.write("# ::snt " + amr.metadata['snt'] + "\n")
        file_graph.write("# ::alignments " + amr.metadata['alignments'] + "\n")
        file_graph.write(amr.graph_string() + "\n\n")
    file_graph.close()

    amrs, alignments = reader.load(file_graph.name, remove_wiki=True, output_alignments=True)
    amr_summaries = amrs
    align_data = alignments

    dict_amr_data_summaries = {}
    for amr in amr_summaries:
        dict_amr_data_summaries[amr.metadata['snt']] = amr

    error_log = 0
    error_log_claim = 0
    error_log_g = 0

    count_error_word = 0
    count_error_word_map = 0

    for example in tqdm(data):
        try:
            # import pdb
            # pdb.set_trace()
            sentence = example['summary']
            map_nodes_tokens = map_nodes(dict_amr_data_summaries[sentence], align_data)
            sent_tok = dict_amr_data_summaries[sentence].tokens
            example['summary_tok'] = " ".join(sent_tok)
            if not map_nodes_tokens:
                count_error_word_map += 1

            data_hal_amr = [], [], map_nodes_tokens

            graph_simple, triples, hal, words_amr = simplify_amr_hal(dict_amr_data_summaries[sentence], data_hal_amr)

            if not words_amr:
                count_error_word += 1

            example['graph_summary'] = {}

            example['hallucination_amr'] = json.dumps(words_amr)
            graph_simple = ' '.join(graph_simple)
            example['graph_summary']['amr_simple'] = graph_simple
            example['graph_summary']['triples'] = json.dumps(triples)

        except Exception as e:
            error_log_claim += 1
            #print("skip claim graph", error_log_claim)
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

    print("skipped documents sentences:", error_log)
    print("skipped graphs:", error_log_claim)

def main(file, num_sents):

    data = load_source_docs(file, to_dict=False)
    print("Loaded %d sample(s)." % len(data))

    data = extract_sents(data, num_sents)
    amr_data, amr_data_summaries = extract_amrs(data)
    process_amr(data, amr_data, amr_data_summaries, num_sents)

    save_data(data, file)


if __name__ == "__main__":
    file = sys.argv[1]
    num_sents = int(sys.argv[2])
    main(file, num_sents)
