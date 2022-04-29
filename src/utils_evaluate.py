import json
from preprocess import align_triples, generate_edge_tensors
from utils import process_triples, update_triples, align_triples_tok
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

label_to_id = {"CORRECT": 1, "INCORRECT": 0}


def preprocess_function(examples, tokenizer, num_document_graphs):
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    max_seq_length = 512
    args = (examples["summary"], examples["article"])
    result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)

    result["graphs"] = []
    result["graphs_attn"] = []
    result["mask_graph"] = []
    result["edge_index"] = []
    result["edge_type"] = []

    max_seq_length = 300

    for graphs, graph_summary in zip(examples["graphs"], examples["graph_summary"]):

        if graph_summary["amr_simple"]:
            claim_graph_string_tok = tokenizer.tokenize("[CLS] " + graph_summary["amr_simple"] + " [SEP]")

            triples_claim_graph = process_triples(json.loads(graph_summary["triples"]))
            triples_claim_graph = align_triples(claim_graph_string_tok,
                                                "[CLS] " + graph_summary["amr_simple"] + " [SEP]",
                                                triples_claim_graph, 1)
        else:
            logging.warning("No claim graph.")
            claim_graph_string_tok = "[CLS] [SEP]".split()
            triples_claim_graph = []

        input_ids_graph_claim = tokenizer.convert_tokens_to_ids(claim_graph_string_tok)

        assert len(claim_graph_string_tok) == len(input_ids_graph_claim), "length mismatched"

        padding_length_a = max_seq_length - len(claim_graph_string_tok)
        input_ids_graph_claim = input_ids_graph_claim + ([pad_token] * padding_length_a)
        graph_claim_attention_mask = [1] * len(claim_graph_string_tok) + ([0] * padding_length_a)

        input_ids_graph_claim = input_ids_graph_claim[:max_seq_length]
        graph_claim_attention_mask = graph_claim_attention_mask[:max_seq_length]

        edge_index, edge_types = generate_edge_tensors(triples_claim_graph, max_seq_length)

        enc_graphs = [input_ids_graph_claim]
        enc_graphs_attn = [graph_claim_attention_mask]
        mask_graphs = [1]
        graph_edge_index = [edge_index]
        graph_edge_type = [edge_types]

        # tokenize document graphs
        for g in graphs[:num_document_graphs]:
            doc_graph_string_tok = tokenizer.tokenize("[CLS] " + g["amr_simple"] + " [SEP]")

            triples = json.loads(g["triples"])
            triples = process_triples(triples)
            all_triples = align_triples(doc_graph_string_tok, "[CLS] " + g["amr_simple"] + " [SEP]", triples, 1)

            edge_index, edge_types = generate_edge_tensors(all_triples, max_seq_length)

            input_ids_graph = tokenizer.convert_tokens_to_ids(doc_graph_string_tok)
            assert len(doc_graph_string_tok) == len(input_ids_graph), "length mismatched"
            padding_length_a = max_seq_length - len(doc_graph_string_tok)
            input_ids_graph = input_ids_graph + ([pad_token] * padding_length_a)
            graph_attention_mask = [1] * len(doc_graph_string_tok) + ([0] * padding_length_a)

            input_ids_graph = input_ids_graph[:max_seq_length]
            graph_attention_mask = graph_attention_mask[:max_seq_length]

            enc_graphs.append(input_ids_graph)
            enc_graphs_attn.append(graph_attention_mask)
            mask_graphs.append(1)
            graph_edge_index.append(edge_index)
            graph_edge_type.append(edge_types)

        # pad document graphs
        while len(enc_graphs) < num_document_graphs + 1:
            doc_sent_tok = tokenizer.tokenize("[CLS] [SEP]")
            padding_length_a = max_seq_length - len(doc_sent_tok)
            input_ids_sent = tokenizer.convert_tokens_to_ids(doc_sent_tok)
            input_ids_sent = input_ids_sent + ([pad_token] * padding_length_a)
            sent_attention_mask = [1] * len(doc_sent_tok) + ([0] * padding_length_a)
            enc_graphs.append(input_ids_sent)
            enc_graphs_attn.append(sent_attention_mask)
            mask_graphs.append(0)
            edge_index, edge_types = generate_edge_tensors([], max_seq_length)
            graph_edge_index.append(edge_index)
            graph_edge_type.append(edge_types)

        result["graphs"].append(enc_graphs)
        result["graphs_attn"].append(enc_graphs_attn)
        result["mask_graph"].append(mask_graphs)
        result["edge_index"].append(graph_edge_index)
        result["edge_type"].append(graph_edge_type)

    assert len(result["graphs"]) == len(result["graphs_attn"]) \
           == len(result["input_ids"]) == len(result["edge_index"]) == len(result["edge_type"])

    return result


def preprocess_edge_function(examples, tokenizer, num_document_graphs):

    max_seq_length = tokenizer.model_max_length
    label_to_id = {'CORRECT': 1, 'INCORRECT': 0}

    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    result = {}
    result['input_ids'] = []
    result['attention_mask'] = []

    result['summary'] = []
    result['article'] = []
    result['attention_mask'] = []

    result['head'] = []
    result['tail'] = []
    result['head_mask'] = []
    result['tail_mask'] = []

    result['head_graph'] = []
    result['tail_graph'] = []
    result['head_mask_graph'] = []
    result['tail_mask_graph'] = []

    result['head_nodes_graph'] = []
    result['tail_nodes_graph'] = []
    result['head_words'] = []
    result['tail_words'] = []

    result['input_ids_graph'] = []
    result['attention_mask_graph'] = []
    result['edge_index'] = []
    result['edge_type'] = []
    result["mask_graph"] = []

    result['label_ann'] = []

    for text, claim, hal, graph_claim, graphs in zip(examples["article"], examples["summary_tok"],
                                                     examples["hallucination_amr"], examples["graph_summary"],
                                                     examples["graphs"]):
        ############ GRAPH INPUTS
        input_ids_graphs = []
        input_attention_mask_graphs = []
        graph_edge_index = []
        graph_edge_type = []
        mask_graphs = []
        for idx_g, g in enumerate(graphs[:num_document_graphs]):
            tokens_graph_input = []
            tokens_graph_input.extend(tokenizer.tokenize('[CLS]'))
            index_now = len(tokens_graph_input)
            all_triples = []
            graph_string = ''
            triples = json.loads(g["triples"])
            all_triples = update_triples(all_triples, triples, len(graph_string.split()))
            graph_string += g["amr_simple"] + ' [SEP] '
            for (word_index, word) in enumerate(g["amr_simple"].split(' ')):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens_graph_input.extend(word_tokens)
                    index_now += len(word_tokens)

            tokens_graph_input.extend(tokenizer.tokenize('[SEP]'))
            index_now += 1

            tokens_graph_input_more = []
            tokens_graph_input_more.extend(tokenizer.tokenize('[SEP]'))
            index_now += 1

            all_triples_graphs, _ = align_triples_tok(tokens_graph_input[1:],
                                                      graph_string, all_triples, 1)

            index_now_start = index_now

            index_map_graph = {}
            for (word_index, word) in enumerate(graph_claim["amr_simple"].split(' ')):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens_graph_input_more.extend(word_tokens)
                    index_now += len(word_tokens)
                    index_map_graph[word_index] = index_now - 1
            tokens_graph_input_more.extend(tokenizer.tokenize('[SEP]'))

            triples_claim = json.loads(graph_claim["triples"])
            triples_claim_graph, _ = \
                align_triples_tok(tokens_graph_input_more[1:-1], graph_claim["amr_simple"], triples_claim, 0)
            triples_claim = update_triples([], triples_claim, index_now_start)

            if all_triples_graphs is None:
                print('no triples graphs')
                all_triples = triples_claim
            else:
                all_triples = all_triples_graphs + triples_claim

            all_triples = process_triples(all_triples)
            max_seq_length_graph = 320  # TODO: parameter for max size of AMR graph
            edge_index, edge_types = generate_edge_tensors(all_triples, max_seq_length_graph)

            graph_edge_index.append(edge_index)
            graph_edge_type.append(edge_types)

            if len(tokens_graph_input) + len(tokens_graph_input_more) > max_seq_length_graph:
                extra_len = len(tokens_graph_input) + len(tokens_graph_input_more) - max_seq_length_graph + 2
                tokens_graph_input = tokens_graph_input[:-extra_len]
                for w in index_map_graph:
                    index_map_graph[w] = index_map_graph[w] - extra_len

            tokens_graph_input = tokens_graph_input + tokens_graph_input_more

            input_ids_graph = tokenizer.convert_tokens_to_ids(tokens_graph_input)

            assert len(tokens_graph_input) == len(input_ids_graph), 'length mismatched graph'
            padding_length_a = max_seq_length_graph - len(tokens_graph_input)
            input_ids_graph = input_ids_graph + ([pad_token] * padding_length_a)
            input_attention_mask = [1] * len(tokens_graph_input) + ([0] * padding_length_a)

            input_ids_graphs.append(input_ids_graph)
            input_attention_mask_graphs.append(input_attention_mask)
            mask_graphs.append(1)

        # pad document graphs
        while len(input_ids_graphs) < num_document_graphs:
            doc_sent_tok = tokenizer.tokenize("[CLS] [SEP]")
            padding_length_a = max_seq_length_graph - len(doc_sent_tok)
            input_ids_sent = tokenizer.convert_tokens_to_ids(doc_sent_tok)
            input_ids_sent = input_ids_sent + ([pad_token] * padding_length_a)
            sent_attention_mask = [1] * len(doc_sent_tok) + ([0] * padding_length_a)
            input_ids_graphs.append(input_ids_sent)
            input_attention_mask_graphs.append(sent_attention_mask)
            mask_graphs.append(0)
            edge_index, edge_types = generate_edge_tensors([], max_seq_length_graph)
            graph_edge_index.append(edge_index)
            graph_edge_type.append(edge_types)

        ############### TEXT

        tokens_input_more = []
        tokens_input_more.extend(tokenizer.tokenize('[CLS]'))
        index_now = len(tokens_input_more)

        index_map = {}
        for (word_index, word) in enumerate(claim.split(' ')):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens_input_more.extend(word_tokens)
                index_now += len(word_tokens)
                index_map[word_index] = index_now - 1

        tokens_input = []
        tokens_input.extend(tokenizer.tokenize('[SEP]'))
        for (word_index, word) in enumerate(text.split(' ')):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens_input.extend(word_tokens)
        tokens_input.extend(tokenizer.tokenize('[SEP]'))

        if len(tokens_input) + len(tokens_input_more) > max_seq_length:
            extra_len = len(tokens_input) + len(tokens_input_more) - max_seq_length + 2
            tokens_input = tokens_input[:-extra_len]

        tokens_input = tokens_input_more + tokens_input

        output_edge_data = create_edge_level_data(hal, index_map, index_map_graph)

        child_indices, head_indices, child_indices_mask, head_indices_mask, head_indices_graph, \
        child_indices_graph, head_indices_mask_graph, child_indices_mask_graph, \
        child_words, head_words, child_nodes_graph, head_nodes_graph, mask_entail, num_dependencies = output_edge_data

        if num_dependencies == 0:
            print('reject')
            continue

        input_ids = tokenizer.convert_tokens_to_ids(tokens_input)

        assert len(tokens_input) == len(input_ids), 'length mismatched'
        padding_length_a = max_seq_length - len(tokens_input)
        input_ids = input_ids + ([pad_token] * padding_length_a)
        input_attention_mask = [1] * len(tokens_input) + ([0] * padding_length_a)

        result['input_ids'].append(input_ids)
        result['attention_mask'].append(input_attention_mask)

        result['head'].append(head_indices)
        result['tail'].append(child_indices)
        result['head_mask'].append(head_indices_mask)
        result['tail_mask'].append(child_indices_mask)

        result['head_graph'].append(head_indices_graph)
        result['tail_graph'].append(child_indices_graph)
        result['head_mask_graph'].append(head_indices_mask_graph)
        result['tail_mask_graph'].append(child_indices_mask_graph)

        result['head_nodes_graph'].append(head_nodes_graph)
        result['tail_nodes_graph'].append(child_nodes_graph)
        result['head_words'].append(head_words)
        result['tail_words'].append(child_words)

        result['label_ann'].append(mask_entail)

        result['edge_index'].append(graph_edge_index)
        result['edge_type'].append(graph_edge_type)

        result['input_ids_graph'].append(input_ids_graphs)
        result['attention_mask_graph'].append(input_attention_mask_graphs)
        result["mask_graph"].append(mask_graphs)

        result['summary'].append(claim)
        result["article"].append(text)

    assert len(result['edge_index']) == len(result['edge_type']) == len(result['input_ids_graph']) \
           == len(result['attention_mask_graph']) == len(result['mask_graph'])

    return result


def create_edge_level_data(hal, index_map, index_map_graph):
    num_deps_per_ex = 30
    words_per_node = 5

    child_indices = [[0] * num_deps_per_ex for i in range(words_per_node)]
    head_indices = [[0] * num_deps_per_ex for i in range(words_per_node)]
    child_indices_mask = [[0] * num_deps_per_ex for i in range(words_per_node)]
    head_indices_mask = [[0] * num_deps_per_ex for i in range(words_per_node)]

    child_indices_graph = [[0] * num_deps_per_ex for i in range(words_per_node)]
    head_indices_graph = [[0] * num_deps_per_ex for i in range(words_per_node)]
    child_indices_mask_graph = [[0] * num_deps_per_ex for i in range(words_per_node)]
    head_indices_mask_graph = [[0] * num_deps_per_ex for i in range(words_per_node)]

    mask_entail = [-100] * num_deps_per_ex
    mask_cont = [0] * num_deps_per_ex
    num_dependencies = 0

    child_words = []
    head_words = []
    child_nodes_graph = []
    head_nodes_graph = []
    i = 0
    for k, row in enumerate(json.loads(hal)):

        if i >= num_deps_per_ex:
            break

        child_idxs = [int(r) for r in row[0].split(' ')]
        head_idxs = [int(r) for r in row[1].split(' ')]

        child_words.append(list(set([r for r in row[2].split(' ')])))
        head_words.append(list(set([r for r in row[3].split(' ')])))

        child_idxs_graph = [int(r) for r in row[8].split(' ')]
        head_idxs_graph = [int(r) for r in row[9].split(' ')]

        child_nodes_graph.append([r for r in row[6].split(' ')])
        head_nodes_graph.append([r for r in row[7].split(' ')])


        num_dependencies += 1
        d_label = int(row[4])

        if d_label == 1:
            mask_entail[i] = 1
        else:
            mask_entail[i] = 0
            mask_cont[i] = 1

        for idx_c, c in enumerate(child_idxs[:words_per_node]):
            child_indices[idx_c][i] = index_map[c]
            child_indices_mask[idx_c][i] = 1
        for idx_c, c in enumerate(head_idxs[:words_per_node]):
            head_indices[idx_c][i] = index_map[c]
            head_indices_mask[idx_c][i] = 1

        for idx_c, c in enumerate(child_idxs_graph[:words_per_node]):
            child_indices_graph[idx_c][i] = index_map_graph[c]
            child_indices_mask_graph[idx_c][i] = 1
        for idx_c, c in enumerate(head_idxs_graph[:words_per_node]):
            head_indices_graph[idx_c][i] = index_map_graph[c]
            head_indices_mask_graph[idx_c][i] = 1

        i += 1

    output_edge_data = child_indices, head_indices, child_indices_mask, head_indices_mask, head_indices_graph, \
                       child_indices_graph, head_indices_mask_graph, child_indices_mask_graph, \
                       child_words, head_words, child_nodes_graph, head_nodes_graph, mask_entail, num_dependencies

    return output_edge_data
