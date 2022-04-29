import torch
import os
import json
from torch_geometric.data import Data, Batch
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
from collections import defaultdict
import unidecode

def transform_graph_geometric(embeddings, edge_index, edge_type):

    list_geometric_data = [Data(x=emb, edge_index=torch.tensor(edge_index[idx], dtype=torch.long),
                                y=torch.tensor(edge_type[idx], dtype=torch.long)) for idx, emb in enumerate(embeddings)]

    bdl = Batch.from_data_list(list_geometric_data)
    bdl = bdl.to("cuda:" + str(torch.cuda.current_device()))

    return bdl


def save_metrics(metrics, output_file):
    if os.path.exists(output_file):
        append_write = "a"  # append if already exists
    else:
        append_write = "w"  # make a new file if not

    with open(output_file, append_write, encoding="utf-8") as fd:
            fd.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def maybe_save_checkpoint(metrics, save_dir, global_step, model, tokenizer, best_metric):

    best_bacc = 0
    folder_checkpoint = ""

    output_file = os.path.join(save_dir, "best_checkpoint.json")
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            best_bacc = data[0][best_metric]
            folder_checkpoint = data[0]["folder_checkpoint"]

    if metrics[best_metric] > best_bacc:

        save_dir_name = "step_{}".format(global_step)
        save_sub_dir = os.path.join(save_dir, save_dir_name)
        os.mkdir(save_sub_dir)
        torch.save(model, save_sub_dir + "/model.pt")
        #model.save_pretrained(save_sub_dir)
        tokenizer.save_pretrained(save_sub_dir)

        if folder_checkpoint:
            os.system("rm -rf " + folder_checkpoint)

        os.system("rm -rf " + output_file)
        metrics["folder_checkpoint"] = save_sub_dir
        save_metrics(metrics, output_file)


def calculate_metrics(global_step, pred_labels, validation_data, best_metric):

    data_source = np.array(validation_data["train"]["domain"])
    labels_val = np.array(validation_data["train"]["label"])
    pred_labels = np.array(pred_labels)
    assert len(labels_val) == len(pred_labels) == len(data_source)

    labels_source = defaultdict(list)
    preds_source = defaultdict(list)
    for ds, lv, pv in zip(data_source, labels_val, pred_labels):
        labels_source[ds].append(lv)
        preds_source[ds].append(pv)

    metrics = {"step": global_step,
               "accuracy": np.round_((pred_labels == labels_val).astype(np.float32).mean().item(), 4),
               "bacc": np.round_(balanced_accuracy_score(y_true=labels_val, y_pred=pred_labels), 4)}

    metrics["f1"] = np.round_(f1_score(y_true=labels_val, y_pred=pred_labels, average="micro"), 4)
    metrics["f1_macro"] = np.round_(f1_score(y_true=labels_val, y_pred=pred_labels, average="macro"), 4)
    metrics["size"] = len(labels_val)

    selected_metrics = []
    for source in labels_source.keys():
        if source not in metrics:
            metrics[source] = {}
            metrics[source]["bacc"] = np.round_(
                balanced_accuracy_score(y_true=labels_source[source], y_pred=preds_source[source]), 4)
            metrics[source]["f1"] = np.round_(
                f1_score(y_true=labels_source[source], y_pred=preds_source[source], average="micro"), 4)
            metrics[source]["size"] = len(labels_source[source])
            selected_metrics.append(metrics[source]["bacc"])

    selected_metrics = np.mean(selected_metrics)
    metrics["avg_bacc"] = np.round_(selected_metrics, 4)

    if best_metric is not None:
        if selected_metrics > best_metric[0]:
            best_metric = [np.round_(selected_metrics, 4),
                           np.round_(metrics["f1"], 4)]

        metrics["best_bacc"] = best_metric

    return best_metric, metrics


def align_triples_tok(graph_string_tok, graph_string, all_triples, size_claim_graph=None):
    graph_string = graph_string.split()

    map = {}
    idx_graph_string = 0
    map[idx_graph_string] = []

    try:
        word = ''
        for idx, tok in enumerate(graph_string_tok):

            word += unidecode.unidecode(tok.replace("##", ""))

            unaccented_string = unidecode.unidecode(graph_string[idx_graph_string].lower())

            if not unaccented_string.startswith(word):
                idx_graph_string += 1
                map[idx_graph_string] = []
                word = ''
                word += tok.replace("##", "")

            map[idx_graph_string].append(idx)

        assert len(map) == len(graph_string)

        for k in map.keys():
            original_word = graph_string[k]
            original_word = unidecode.unidecode(original_word)

            recoveredword = ''
            for tok_idx in map[k]:
                tok = graph_string_tok[tok_idx]
                recoveredword += tok.replace("##", '')

            recoveredword = unidecode.unidecode(recoveredword)
            assert original_word.lower() == recoveredword.lower()

        new_all_triples = []
        triples_classification = []
        for t in all_triples:
            t0s = map[t[0]]
            t1s = map[t[1]]
            t2 = t[2]

            if len(t) > 3:
                t2s = map[t[2]]
                hal = t[3]

                if size_claim_graph:
                    triples_classification.append((t0s[-1] + size_claim_graph, t1s[-1] + size_claim_graph, t2s[-1] + size_claim_graph, hal))
                else:
                    triples_classification.append((t0s[-1], t1s[-1], t2s[-1], hal))

                for t0 in t0s:
                    for t1 in t1s:
                        for t2 in t2s:
                            if size_claim_graph:
                                new_all_triples.append((t0 + size_claim_graph, t1 + size_claim_graph, t2 + size_claim_graph, hal))
                            else:
                                new_all_triples.append((t0, t1, t2, hal))
            else:

                for t0 in t0s:
                    for t1 in t1s:
                        if size_claim_graph:
                            new_all_triples.append((t0 + size_claim_graph, t1 + size_claim_graph, t2))
                        else:
                            new_all_triples.append((t0, t1, t2))

    except Exception as error_msg:
        #return None
        print(error_msg)
        print(len(map), len(graph_string))
        print(graph_string)
        print(graph_string_tok)
        for k, v in map.items():
            print(graph_string[k], "           ", ' '.join(graph_string_tok[vv] for vv in v))

        if original_word:
            print(original_word, recoveredword)

        import pdb
        pdb.set_trace()
        exit()

    return new_all_triples, triples_classification


def update_triples(all_triples, triples, index_now):

    updated_triples = []
    for t in triples:
        updated_triples.append((t[0] + index_now, t[1] + index_now, t[2] + index_now, t[3]))

    return all_triples + updated_triples


def generate_edge_tensors(triples, max_seq_length_graph):

    set_edges = {'d': 0, 'r': 1}

    edge_index_1 = []
    edge_index_2 = []
    edge_types = []

    for t in triples:
        t0 = t[0]
        t1 = t[1]
        t2 = t[2]

        if t0 >= max_seq_length_graph:
            continue
        if t1 >= max_seq_length_graph:
            continue

        edge_index_1.append(t0)
        edge_index_2.append(t1)
        edge_types.append(set_edges[t2])

    edge_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
    edge_types = torch.tensor(edge_types, dtype=torch.long)
    # edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    # edge_types = torch.tensor([0], dtype=torch.long)

    return edge_index, edge_types


def process_triples(triples):
    new_triples = []
    for triple in triples:
        n1, n2, rel, hal = triple
        new_triples.append((n1, rel, 'd'))
        new_triples.append((rel, n1, 'r'))
        new_triples.append((rel, n2, 'd'))
        new_triples.append((n2, rel, 'r'))

    return new_triples


def processing_edge_level_data(examples, tokenizer, num_document_graphs, max_seq_length):

    label_to_id = {'CORRECT': 1, 'INCORRECT': 0}

    num_deps_per_ex = 30
    words_per_node = 5
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    result = {}
    result['input_ids'] = []
    result['attention_mask'] = []
    result["label"] = []

    result['head'] = []
    result['tail'] = []
    result['head_mask'] = []
    result['tail_mask'] = []

    result['head_graph'] = []
    result['tail_graph'] = []
    result['head_mask_graph'] = []
    result['tail_mask_graph'] = []

    result['input_ids_graph'] = []
    result['attention_mask_graph'] = []
    result['edge_index'] = []
    result['edge_type'] = []
    result["mask_graph"] = []

    result['label_ann'] = []
    rejected_ex = 0

    for text, claim, hal, label, graph_claim, graphs in zip(examples["article"], examples["summary"],
                                                            examples["hallucination_amr"],
                                                            examples["label"], examples["graph_summary"],
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
            max_seq_length_graph = 320
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

        input_arcs = [[0] * 20 for _ in range(num_deps_per_ex)]
        sentence_label = label_to_id[label]

        i = 0
        for k, row in enumerate(json.loads(hal)):

            if i >= num_deps_per_ex:
                break

            child_idxs = [int(r) for r in row[0].split(' ')]
            head_idxs = [int(r) for r in row[1].split(' ')]

            child_idxs_graph = [int(r) for r in row[8].split(' ')]
            head_idxs_graph = [int(r) for r in row[9].split(' ')]

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

        if num_dependencies == 0:
            print('reject', rejected_ex)
            rejected_ex += 1
            continue

        input_ids = tokenizer.convert_tokens_to_ids(tokens_input)

        assert len(tokens_input) == len(input_ids), 'length mismatched'
        padding_length_a = max_seq_length - len(tokens_input)
        input_ids = input_ids + ([pad_token] * padding_length_a)
        input_attention_mask = [1] * len(tokens_input) + ([0] * padding_length_a)

        result['input_ids'].append(input_ids)
        result['attention_mask'].append(input_attention_mask)
        result["label"].append(sentence_label)

        result['head'].append(head_indices)
        result['tail'].append(child_indices)
        result['head_mask'].append(head_indices_mask)
        result['tail_mask'].append(child_indices_mask)

        result['head_graph'].append(head_indices_graph)
        result['tail_graph'].append(child_indices_graph)
        result['head_mask_graph'].append(head_indices_mask_graph)
        result['tail_mask_graph'].append(child_indices_mask_graph)

        result['label_ann'].append(mask_entail)

        result['edge_index'].append(graph_edge_index)
        result['edge_type'].append(graph_edge_type)

        result['input_ids_graph'].append(input_ids_graphs)
        result['attention_mask_graph'].append(input_attention_mask_graphs)
        result["mask_graph"].append(mask_graphs)

    assert len(result['edge_index']) == len(result['edge_type']) == len(result['input_ids_graph']) \
           == len(result['attention_mask_graph']) == len(result['mask_graph'])

    return result
